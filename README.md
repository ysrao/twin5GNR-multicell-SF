# SF Multicell Twin — an open 5G NR ray-tracing lab

A reproducible radio digital twin for teaching and research: a three-site,
nine-sector 5G NR layout at 3.6 GHz, ray-traced with **Sionna RT 2.2.0** over
a 1 km × 1 km San Francisco tile (Telegraph Hill / North Beach, centre
37.801105, −122.404175). The city scene is built entirely from open municipal
data, every trace runs on a free Colab T4 GPU, and the interactive web
artefact reads the trace output directly.

All results are ray-traced simulations, not measurements.

**Start here**
- ▶ Interactive artefact: https://ysrao.github.io/twin5GNR-multicell-SF/twin5GNR-multicell-SionnaRT-OpenScene-Nov14.html
- ▶ Notebook in Colab: https://colab.research.google.com/github/ysrao/twin5GNR-multicell-SF/blob/main/notebook/twin5GNR_open_scene_T4.ipynb

## Quickstart for students (fully local, $0)

1. Download `data/city_open.ply` and `data/SF_sites.csv`.
2. Open the notebook in Colab with a T4 GPU runtime and put both files in the
   Colab Files pane (or set `DATA_SOURCE = 'upload'` in step 3).
3. Run all. A full-tile trace takes about 15 s; results are written to `./out`
   on your own storage. Nothing is read from the authors' accounts.

## What's in the package

| Path | Contents |
|---|---|
| `notebook/twin5GNR_open_scene_T4.ipynb` | Every trace in the paper: full tile, tower-B window, sensitivity runs, tower-shift sweep |
| `data/city_open.ply` | The open city scene: 1,471 LOD1 buildings over terrain, 110,396 triangles, local ENU metres |
| `data/SF_sites.csv` | 76 site rows: 12 macro tower-shift cases (`macro-3.6GHz`) and 64 small-cell rows (`smallcell-28GHz`) |
| `data/results/` | Result files for the full tile and the tower-B window |
| `scripts/fetch_sources.py`, `scripts/build_scene.py` | Download the open inputs and rebuild the scene |
| `twin5GNR-multicell-SionnaRT-OpenScene-Nov14.html` | The interactive artefact (GitHub Pages) |

Result files hold, per receiver, the RSRP of each of the nine sectors (dBm per
resource element) and the best-server value; `null` means no ray-traced path.

## Rebuilding the scene

```
pip install numpy shapely trimesh mapbox-earcut matplotlib pillow
python scripts/fetch_sources.py
python scripts/build_scene.py
```

Sources: City and County of San Francisco **Building Footprints** (DataSF
dataset `ynuv-fyni`, CC0), with LiDAR ground and median roof heights per
building, and **AWS Terrain Tiles** (terrarium, zoom 15). A rebuild from
today's DataSF download matches `data/city_open.ply` to within a few
centimetres almost everywhere, but includes two large waterfront footprints
east and north of tower B that the published scene does not, giving 1,473
buildings. Use `data/city_open.ply` to reproduce the published numbers.

## Lab exercises

`SF_sites.csv` defines each tower shifted 125 m east, west, north or south of
its as-built position, at 20, 30, 35 or 50 m respectively — 12 designed cases.
Pick cases in the notebook's step 9, run, and compare coverage. Because
direction and height are paired, design a follow-up sweep that separates
them. Further exercises: trace the tower-B window and explain what its
agreement with the 40 m grid does and does not establish; test per-site
antenna clearance; add a second radio material.

## Roadmap — improvements welcome

- Vegetation layer (ITU-R P.833)
- Additional radio materials beyond ITU concrete
- A second city tile
- Classroom materials: worked solutions, instructor notes

Please open an issue or a pull request. Accepted contributions are credited in
the release notes. Versioned releases are deposited to Zenodo for a DOI.

## Licensing

Code (notebook and scripts): **MIT** — see `LICENSE`. Documentation,
`SF_sites.csv`, result data, figures and the built city scene: **CC BY 4.0**
(attribution to Rao S. Yenamandra). The scene's source data (DataSF building
footprints) is CC0; open-source components remain under their own terms.

## Citation

See `CITATION.cff`.
