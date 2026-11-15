# Stage 3 Reference Datasets

The 3D rectangular duct marching solver reference grids (`duct_Re50.npz` through `duct_Re600.npz`, 12,035,304 nodes each) are ~240 MB each (~1.4 GB total), exceeding GitHub's 100 MB per-file limit.

To generate them locally, run:
```bash
python 3/generate_data.py
```
Or download precomputed grids from the GitHub repository Releases page.
