#!/usr/bin/env python3
# tide_ink_mpi_pygame.py
#
# VIS 141B / visramps-style: MPI (mpirun) + pygame renderer (no tkinter)
# - rank 0 fetches NOAA tide predictions and broadcasts to all ranks
# - each rank renders one screen
# - flow field is computed in GLOBAL wall coordinates so the whole 4x5 wall reads as one “big wave”
# - ink accumulates on an alpha surface with gentle fade (sediment feel) instead of deleting tags
# - robust shutdown: ESC / Q / Ctrl+C should restore the display

import os, sys, time, math, random, json, signal
import urllib.request, urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# -------------------- MPI (optional fallback) --------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    RANK = comm.Get_rank()
    SIZE = comm.Get_size()
except Exception:
    comm = None
    RANK = 0
    SIZE = 1

INDEX = RANK

# -------------------- WALL LAYOUT (4 x 5) --------------------
# visramps: 4 columns (left->right) x 5 rows (top->bottom)
COLS, ROWS = 4, 5

# IMPORTANT: mapping assumes INDEX increases down a column first:
# col = INDEX // ROWS, row = INDEX % ROWS
col = INDEX // ROWS          # 0..3
row = INDEX % ROWS           # 0..4

SLOT = col                   # 0..3  (0 past / 1 now / 2 future / 3 avg)
REGION = row                 # 0..4

VIEW_NAMES = ["PAST", "NOW", "FUTURE", "AVG"]
VIEW_NAME = VIEW_NAMES[SLOT]

# -------------------- NOAA CONFIG --------------------
BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
DAYS_LOOKAHEAD = 2
INTERVAL = "15"            # "h" / "15" / "hilo"
FETCH_EVERY_SEC = 120

REGION_STATIONS = [
    ("SoCal South (San Diego)", "9410170"),
    ("SoCal (La Jolla)",        "9410230"),
    ("Central Coast (SB)",      "9411340"),
    ("Bay Area (SF)",           "9414290"),
    ("NorCal (Monterey)",       "9413450"),
]
REGION_NAME, STATION = REGION_STATIONS[REGION]

# -------------------- RENDER CONFIG --------------------
FPS = 60
W, H = 1920, 1200  # per-screen resolution on visramps

SPAWN_RATE_BASE = 50
MAX_PARTICLES = 2400

# fade overlay alpha for sediment feel:
# lower = stays longer (more deposition), higher = clears faster
FADE_ALPHA = 8  # 4~14 recommended

BG_RGB = (5, 10, 18)

# -------------------- UTIL --------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def lerp(a, b, t):
    return a + (b - a) * t

def parse_noaa_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

def fetch_series(station_id: str):
    now_utc = datetime.now(timezone.utc)
    begin = now_utc.strftime("%Y%m%d")
    end = (now_utc + timedelta(days=DAYS_LOOKAHEAD)).strftime("%Y%m%d")

    params = {
        "product": "predictions",
        "application": "vis141b",
        "station": station_id,
        "begin_date": begin,
        "end_date": end,
        "datum": "MLLW",
        "time_zone": "gmt",
        "units": "metric",
        "interval": INTERVAL,
        "format": "json",
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError("No predictions returned")

    series = [(parse_noaa_time(p["t"]), float(p["v"])) for p in preds]
    series.sort(key=lambda x: x[0])
    return now_utc, series

def pick_prev_next(series, now_utc):
    for i in range(len(series) - 1):
        t0, v0 = series[i]
        t1, v1 = series[i + 1]
        if t0 <= now_utc < t1:
            return (t0, v0), (t1, v1)
    return (series[-2][0], series[-2][1]), (series[-1][0], series[-1][1])

def compute_views(now_utc, series):
    (t0, v0), (t1, v1) = pick_prev_next(series, now_utc)
    span = max((t1 - t0).total_seconds(), 1.0)
    prog = clamp((now_utc - t0).total_seconds() / span, 0.0, 1.0)

    past_v = v0
    future_v = v1
    now_v = lerp(v0, v1, prog)
    avg_v = sum(v for _, v in series) / max(len(series), 1)
    return past_v, now_v, future_v, avg_v, t0, t1, prog

def selected_value(past_v, now_v, future_v, avg_v):
    if SLOT == 0:
        return past_v
    if SLOT == 1:
        return now_v
    if SLOT == 2:
        return future_v
    return avg_v

# -------------------- PARTICLES --------------------
@dataclass
class Particle:
    x: float
    y: float
    px: float
    py: float
    vx: float
    vy: float
    life: float
    size: float
    jitter: float

def flow_global(gx, gy, t, GW, GH, strength, freq):
    # same "ink flow" concept as your tkinter version, but in global wall coords
    nx = (gx / GW) * 2 - 1
    ny = (gy / GH) * 2 - 1
    a = math.sin((nx * 2.1 + ny * 1.7 + t) * freq)
    b = math.cos((ny * 2.0 - nx * 1.3 - t) * freq)
    ang = a + b + 0.8 * math.sin((nx * nx + ny * ny) * 3.0 + t * 0.7)
    return math.cos(ang) * strength, math.sin(ang) * strength

# -------------------- MAIN --------------------
def main():
    # If you ever need framebuffer mode on headless linux, you can uncomment these.
    # On macOS, DO NOT set fbcon.
    # os.environ.setdefault("SDL_VIDEODRIVER", "fbcon")
    # os.environ.setdefault("SDL_FBDEV", "/dev/fb0")
    # os.environ.setdefault("SDL_NOMOUSE", "1")

    import pygame
    pygame.init()
    
    PREVIEW = os.environ.get("VISRAMPS_PREVIEW", "0") == "1"
    
    if PREVIEW:
        # windowed
        W, H = 1200, 800
        screen = pygame.display.set_mode((W, H))
        
    else:
        # fullscreen (per-node)
        W, H = 1920, 1200
        screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
        
        # robust quit flag (for signal handler)
        state = {"running": True}

    def shutdown():
        state["running"] = False

    def handle_sigint(sig, frame):
        shutdown()

    signal.signal(signal.SIGINT, handle_sigint)
    # SIGTERM may not exist on some platforms the same way, but usually does:
    try:
        signal.signal(signal.SIGTERM, handle_sigint)
    except Exception:
        pass

    # SLOT/REGION-specific seed (keeps “grain/pattern” stable)
    random.seed(2000 + REGION * 10 + SLOT)

    pygame.init()
    pygame.display.set_caption(f"Tide Ink MPI | idx {INDEX} | {REGION_NAME} | {VIEW_NAME}")

    # Fullscreen per node
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    # Accumulation surface (alpha)
    ink = pygame.Surface((W, H), pygame.SRCALPHA)
    fade = pygame.Surface((W, H), pygame.SRCALPHA)
    fade.fill((0, 0, 0, FADE_ALPHA))

    clock = pygame.time.Clock()

    # HUD font (optional; if font fails, we just skip text)
    try:
        font = pygame.font.SysFont("Helvetica", 18, bold=True)
        font_small = pygame.font.SysFont("Helvetica", 16, bold=False)
    except Exception:
        font = None
        font_small = None

    show_info = False

    # Particle system params (SLOT variation)
    phase_offset = SLOT * 1.3
    spawn_rate = SPAWN_RATE_BASE + SLOT * 6
    energy_bias = 0.90 + SLOT * 0.07

    particles = []
    start = time.time()
    last_fetch = 0.0

    # shared data state
    past_v = now_v = future_v = avg_v = 0.6
    t0 = t1 = None
    prog = 0.0
    last_fetch_ok = False

    GW, GH = COLS * W, ROWS * H  # global wall dimensions

    # helper: rank0 fetch + broadcast
    def update_data():
        nonlocal past_v, now_v, future_v, avg_v, t0, t1, prog, last_fetch_ok

        payload = None
        if RANK == 0:
            try:
                now_utc, series = fetch_series(STATION)
                pv, nv, fv, av, _t0, _t1, _prog = compute_views(now_utc, series)
                payload = {
                    "ok": True,
                    "past": pv, "now": nv, "future": fv, "avg": av,
                    "t0": _t0.strftime("%Y-%m-%d %H:%M"),
                    "t1": _t1.strftime("%Y-%m-%d %H:%M"),
                    "prog": _prog,
                    "utc": True,
                }
            except Exception:
                payload = {
                    "ok": False,
                    "past": past_v, "now": now_v, "future": future_v, "avg": avg_v,
                    "t0": None, "t1": None,
                    "prog": prog,
                    "utc": True,
                }

        if comm is not None:
            payload = comm.bcast(payload, root=0)
        else:
            # single-process fallback
            payload = payload or {"ok": False, "past": past_v, "now": now_v, "future": future_v, "avg": avg_v,
                                  "t0": None, "t1": None, "prog": prog, "utc": True}

        past_v = float(payload["past"])
        now_v = float(payload["now"])
        future_v = float(payload["future"])
        avg_v = float(payload["avg"])
        prog = float(payload["prog"])
        last_fetch_ok = bool(payload["ok"])

        if payload.get("t0"):
            t0 = parse_noaa_time(payload["t0"])
        else:
            t0 = None
        if payload.get("t1"):
            t1 = parse_noaa_time(payload["t1"])
        else:
            t1 = None

    # initial data pull (so you don’t wait 2 minutes)
    update_data()
    last_fetch = time.time()

    # main loop
    try:
        while state["running"]:
            dt = clock.tick(FPS) / 1000.0
            now = time.time()

            # events
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    shutdown()
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        shutdown()
                    elif e.key in (pygame.K_i, pygame.K_I):
                        show_info = not show_info

            # periodic data fetch (rank0) + broadcast
            if now - last_fetch >= FETCH_EVERY_SEC:
                update_data()
                last_fetch = now

            # choose view value for this SLOT
            v = selected_value(past_v, now_v, future_v, avg_v)
            tv = clamp((v + 0.5) / 2.3, 0.0, 1.0)

            # palette + motion (same “feel” as original)
            ink_r = lerp(80, 255, tv)
            ink_g = lerp(110, 225, tv)
            ink_b = lerp(200, 255, tv)

            strength = lerp(0.6, 2.6, tv) * energy_bias
            freq = lerp(1.1, 3.2, tv)

            # fade (sediment)
            ink.blit(fade, (0, 0))

            # spawn
            nspawn = int(spawn_rate * dt) + 2
            for _ in range(nspawn):
                if len(particles) >= MAX_PARTICLES:
                    break
                x = random.uniform(0, W)
                y = random.uniform(0, H)
                vx = random.uniform(-0.5, 0.5)
                vy = random.uniform(-0.5, 0.5)
                life = random.uniform(2.2, 6.0)
                size = random.uniform(1.8, 7.5)
                jitter = random.uniform(-55, 55)
                particles.append(Particle(x, y, x, y, vx, vy, life, size, jitter))

            # time for flow
            t = ((now - start) * 0.35) + phase_offset

            alive = []
            for p in particles:
                # global coords -> makes the whole wall read as one big flow
                gx = p.x + col * W
                gy = p.y + row * H

                fx, fy = flow_global(gx, gy, t, GW, GH, strength, freq)

                p.vx = lerp(p.vx, fx, 0.10)
                p.vy = lerp(p.vy, fy, 0.10)

                p.px, p.py = p.x, p.y
                p.x += p.vx * 10 * dt
                p.y += p.vy * 10 * dt
                p.life -= dt
                if p.life <= 0:
                    continue

                # wrap edges
                if p.x < 0: p.x += W
                if p.x >= W: p.x -= W
                if p.y < 0: p.y += H
                if p.y >= H: p.y -= H

                shimmer = 0.5 + 0.5 * math.sin((p.x + p.y) * 0.002 + t * 2.0)

                r = clamp(ink_r + p.jitter * 0.25 + 35 * shimmer, 0, 255)
                g = clamp(ink_g + p.jitter * 0.18, 0, 255)
                b = clamp(ink_b - 25 * shimmer, 0, 255)

                # stroke thickness approximates Tk's “oval dots” accumulation but smoother
                thick = max(1, int(p.size * (0.55 + 0.45 * shimmer)))
                color = (int(r), int(g), int(b), 255)

                pygame.draw.line(ink, color, (p.px, p.py), (p.x, p.y), width=thick)
                pygame.draw.circle(ink, color, (int(p.x), int(p.y)), max(1, thick // 2))

                alive.append(p)

            particles = alive

            # composite
            screen.fill(BG_RGB)
            screen.blit(ink, (0, 0))

            # HUD
            if font is not None:
                bottom = f"{REGION_NAME}  |  {VIEW_NAME}  |  station {STATION}  |  idx {INDEX}  (rank {RANK}/{SIZE})"
                txt = font.render(bottom, True, (255, 255, 255))
                screen.blit(txt, (20, H - 30))

                if show_info and font_small is not None:
                    ok = "OK" if last_fetch_ok else "FETCH FAIL"
                    t0s = t0.strftime("%H:%M") if t0 else "—"
                    t1s = t1.strftime("%H:%M") if t1 else "—"
                    info1 = f"{ok}  |  interval {INTERVAL}  |  refresh {FETCH_EVERY_SEC}s"
                    info2 = f"past={past_v:.2f}  now={now_v:.2f}  future={future_v:.2f}  avg={avg_v:.2f} (m)"
                    info3 = f"bracket UTC: {t0s} -> {t1s}  progress={prog*100:.0f}%"
                    info4 = "ESC/Q quit  |  I toggle info"
                    y = 20
                    for line in (info1, info2, info3, info4):
                        tline = font_small.render(line, True, (255, 255, 255))
                        screen.blit(tline, (20, y))
                        y += 22
                else:
                    hint = "I toggle info  |  ESC/Q quit"
                    tline = font_small.render(hint, True, (255, 255, 255)) if font_small else None
                    if tline:
                        screen.blit(tline, (20, 20))

            pygame.display.flip()

    finally:
        # Robust restore
        try:
            pygame.display.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

        # If using MPI, try to abort cleanly on exit to avoid hung ranks
        if comm is not None:
            try:
                comm.Barrier()
            except Exception:
                pass

if __name__ == "__main__":
    main()