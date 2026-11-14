# SF Multicell Twin — an open 5G NR ray-tracing lab

A complete, reproducible radio digital twin for teaching and research: a
three-site, nine-sector 5G NR layout (3.6 GHz) ray-traced with **Sionna RT**
over a downtown San Francisco city tile, with an interactive web artefact
wired directly to the trace output.

**Start here**
- ▶ Interactive artefact (14 Nov 2026, open city scene): https://ysrao.github.io/twin5GNR-multicell-SF/twin5GNR-multicell-SionnaRT-OpenScene-Nov14.html — the "Run Sionna RT on T4 (Colab)" button on that page opens the notebook that produced its data.
- Published v7 reference page: https://ysrao.github.io/twin5GNR-multicell-SF/twin5GNR-multicell-SionnaRT-Veg.html
- Notebook (Colab, author-hosted copy): https://colab.research.google.com/drive/1N_8UYlC4-CAJZH1zzeMLSJArcV7XsZmz

## Quickstart for students (fully local, $0)

1. Get the city scene: download **`city_open.ply`** (link in *Data and code
   availability* below) — an open scene built from the City of San
   Francisco's published building footprints — or rebuild it yourself from
   DataSF open data with `build_scene.py`. Follow the notebook's open-scene
   cells to load it. (The scene is redistributable; no permission needed.)
2. Download `SF_sites.csv` and the notebook from the links in
   *Data and code availability* below.
3. Open the notebook in Colab (T4 GPU runtime), set `DATA_SOURCE = 'local'`
   in step 3, and **Run all**. A full-tile trace takes ~15 s. Results are
   written to `./out` on your own storage — the notebook never reads the
   author's accounts at run time.

## What's in the package

| Piece | Where |
|---|---|
| Notebook `twin5GNR_v7_SionnaRT_T4.ipynb` | Colab link above / author's Drive |
| Site file `SF_sites.csv` (76 rows: 12 macro shifts + 64 small-cell rows) | Author's Drive (view-only) |
| City scene `city_open.ply` + `build_scene.py` | Author's Drive (view-only) |
| Result files (open-scene full-tile, window, decomposition and sweep runs) + plots | Author's Drive (view-only) |
| Interactive artefacts (wired re-run page + published v7/v8 pages) | This repo (GitHub Pages) |
| Implementation code package | `multicell-code-v7-v8.zip` in this repo |

## Lab exercises (built in)

`SF_sites.csv` defines each tower shifted 125 m E/W/N/S from its as-built
position, at heights of 20/30/35/50 m — 12 designed cases. Set the case in
the notebook's CSV cell (step 4b), re-run, and compare coverage. The v8 rows
extend the same idea to a 28 GHz small-cell layout. The notebook also walks
through a **verification exercise**: a 10 m run over a 260 × 260 m window
centred on tower B checks the tracer independently of the scene (99.7% of
cells with a path; 89.2% coverage at RSRP ≥ −90 dBm).

## A result worth knowing before you build on this

The same layout traced on two different scenes of the same tile gives very
different coverage: **80.5%** (original fused scene, published v7) vs
**24.3%** (open city scene, full tile) at RSRP ≥ −90 dBm — even though the
two scenes agree closely on path *existence* (89.9% vs 80.8% of cells). The
gap lives in path *strength*, not path existence. Scene provenance and
detail are first-order variables in ray-traced coverage studies — report
them like RF parameters. The re-run does not revise v7's published numbers;
the wired page states the comparison openly.

## Roadmap — improvements welcome

- Vegetation layer (ITU-R P.833) on the re-run pipeline
- The 28 GHz v8 small-cell trace from the CSV's v8 rows
- A second city tile; additional radio materials beyond ITU concrete
- Re-run on a higher-fidelity / Sionna-ready scene when available
- Classroom materials: worked solutions, instructor notes

Please open an issue or a pull request. Accepted contributions are credited
in the release notes. Versioned releases are deposited to Zenodo for a DOI.

## Data and code availability

- Notebook, `SF_sites.csv`, result JSONs, plots: author's Google Drive,
  view-only links (download sources; the package runs fully local).
  - SF_sites.csv: https://drive.google.com/file/d/1bD1k7IzARGZEmANB1xVUUKYJtr61ohXR/view
  - Full-tile results (open scene): https://drive.google.com/file/d/1aiQiKDGYkEnFiilPZLIHS_qJu1ORzhjV/view
  - Tower-B window results (open scene): https://drive.google.com/file/d/1oFRKx0_r--I5AdFLEe6jQ0HK9Mv83NMm/view
- City scene: built from City and County of San Francisco **Building
  Footprints** (DataSF, CC0) and AWS Terrain Tiles — both redistributable.
  - Scene file `city_open.ply`: https://drive.google.com/file/d/14gMKPHQSxf_2Li5zqpwiXW9VMl8IjAAb/view
  - Build script `build_scene.py`: https://drive.google.com/file/d/1TepGtR12uGksRZ4CDupPjK-q-dc8yBKB/view
- Software: Sionna RT 2.2.0, Mitsuba 3, trimesh (all open-source).

## Licensing

Code (this repo and the notebook's code): **MIT** — see `LICENSE`.
Documentation, `SF_sites.csv`, result data, figures, and the built city
scene: **CC BY 4.0** (attribution to Rao S. Yenamandra). The scene's source
data (DataSF building footprints) is CC0; open-source components remain
under their own terms.

## Citation

See `CITATION.cff`. Please cite the package (and the accompanying paper,
once posted) when you use it in teaching or research.
