# Tide Ink

Live tidal data visualization across a 20-screen wall installation.

Built for VIS 141B at UC San Diego, 2025.

## What it does

Fetches real-time tidal data from 5 NOAA stations along the California coast — San Diego, La Jolla, Santa Barbara, San Francisco, and Monterey — and renders it as flowing particles across a 4×5 screen wall (20 displays).

Each screen runs as a separate MPI process. Tide height controls particle speed, color temperature, and flow frequency. The wall reads as one continuous surface.

## How to run

```bash
mpirun -n 20 python tide_final.py
```

## Requirements

pygame
mpi4py
requests

## Demo

https://youtu.be/9pVUBeMXvKc
