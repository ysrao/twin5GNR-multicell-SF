#!/usr/bin/env python3
"""Build an open-data Sionna-ready city scene for the SF twin tile.

Sources (all open):
  - Building footprints + LiDAR ground/roof statistics: City of San Francisco
    "Building Footprints" dataset (data.sf.gov/resource/ynuv-fyni), CC0.
    Per building: gnd_mediancm (NAVD88 ground, cm), hgt_median_m (height, m).
  - Terrain for streets: AWS Terrain Tiles (terrarium, z15; USGS 3DEP / SRTM
    composite), sampled on a 15 m grid and datum-aligned to the DataSF ground
    values at building centroids.

Output: city_open.ply — a single merged mesh in local ENU metres about
(37.801105, -122.404175), z = NAVD88 height minus the terrain height at the
origin. Plus scene_stats.json and a top-down preview PNG.
"""
import ast, json, math, pickle
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, shape as shp_shape

LAT0, LON0 = 37.801105, -122.404175
R_EARTH = 6378137.0
EXTENT = 510.0  # keep buildings whose centroid is within +-EXTENT m ENU

def to_enu(lon, lat):
    x = math.radians(lon - LON0) * R_EARTH * math.cos(math.radians(LAT0))
    y = math.radians(lat - LAT0) * R_EARTH
    return x, y

# ---------- terrain ----------
meta = np.load('terrarium_meta.npy')
TX0, TY0, TX1, TY1, TZ = [int(v) for v in meta]
tiles = pickle.load(open('terrarium_tiles.pkl', 'rb'))
H = (TY1 - TY0 + 1) * 256
W = (TX1 - TX0 + 1) * 256
dem = np.zeros((H, W))
for (tx, ty), arr in tiles.items():
    dem[(ty - TY0) * 256:(ty - TY0 + 1) * 256, (tx - TX0) * 256:(tx - TX0 + 1) * 256] = arr

def dem_sample(lon, lat):
    n = 2 ** TZ
    gx = (lon + 180.0) / 360.0 * n * 256 - TX0 * 256
    gy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n * 256 - TY0 * 256
    x0, y0 = int(math.floor(gx)), int(math.floor(gy))
    fx, fy = gx - x0, gy - y0
    x0 = min(max(x0, 0), W - 2); y0 = min(max(y0, 0), H - 2)
    return float((dem[y0, x0] * (1 - fx) * (1 - fy) + dem[y0, x0 + 1] * fx * (1 - fy)
                  + dem[y0 + 1, x0] * (1 - fx) * fy + dem[y0 + 1, x0 + 1] * fx * fy))

# ---------- footprints ----------
rows = json.load(open('footprints_raw.json'))
buildings = []  # (shapely poly in ENU, ground_navd88_m, height_m)
for r in rows:
    try:
        geom = shp_shape(ast.literal_eval(r['shape']) if isinstance(r['shape'], str) else r['shape'])
        gnd = float(r['gnd_mediancm']) / 100.0
        hgt = float(r['hgt_median_m'])
    except Exception:
        continue
    if hgt < 1.0:
        continue
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.area == 0:
            continue
        cx, cy = to_enu(p.centroid.x, p.centroid.y)
        if abs(cx) > EXTENT or abs(cy) > EXTENT:
            continue
        ext = [(to_enu(lon, lat)) for lon, lat in p.exterior.coords]
        holes = [[(to_enu(lon, lat)) for lon, lat in ring.coords] for ring in p.interiors]
        try:
            ep = Polygon(ext, holes)
        except Exception:
            continue
        if ep.is_valid and ep.area > 2.0:
            buildings.append((ep, gnd, hgt))
print('buildings kept:', len(buildings))

# datum alignment: DEM vs DataSF ground at building centroids
offs = []
for p, gnd, hgt in buildings[::7]:
    c = p.centroid
    # invert ENU centroid back to lon/lat approximately for DEM sampling
    lon = LON0 + math.degrees(c.x / (R_EARTH * math.cos(math.radians(LAT0))))
    lat = LAT0 + math.degrees(c.y / R_EARTH)
    offs.append(dem_sample(lon, lat) - gnd)
dem_offset = float(np.median(offs))
print('DEM minus DataSF ground, median offset (m):', round(dem_offset, 2))
z0 = dem_sample(LON0, LAT0) - dem_offset  # NAVD88 height of origin ground
print('origin ground NAVD88 (m):', round(z0, 2))

# ---------- extrude ----------
meshes = []
for p, gnd, hgt in buildings:
    try:
        m = trimesh.creation.extrude_polygon(p, hgt + 1.5)
    except Exception:
        continue
    m.apply_translation([0, 0, (gnd - z0) - 1.5])
    meshes.append(m)
print('prisms built:', len(meshes))

# ---------- ground ----------
step = 15.0
xs = np.arange(-525, 526, step)
ys = np.arange(-525, 526, step)
verts = []
for y in ys:
    for x in xs:
        lon = LON0 + math.degrees(x / (R_EARTH * math.cos(math.radians(LAT0))))
        lat = LAT0 + math.degrees(y / R_EARTH)
        verts.append([x, y, dem_sample(lon, lat) - dem_offset - z0])
verts = np.array(verts)
nx, ny = len(xs), len(ys)
faces = []
for j in range(ny - 1):
    for i in range(nx - 1):
        a = j * nx + i; b = a + 1; c = a + nx; dd = c + 1
        faces.append([a, c, b]); faces.append([b, c, dd])
ground = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
meshes.append(ground)

scene = trimesh.util.concatenate(meshes)
scene.export('city_open.ply')
stats = {
    'buildings': len(meshes) - 1,
    'triangles': int(len(scene.faces)),
    'vertices': int(len(scene.vertices)),
    'enu_extent_m': [[float(v) for v in row] for row in scene.bounds.tolist()],
    'z_range_m': [float(scene.bounds[0][2]), float(scene.bounds[1][2])],
    'dem_offset_applied_m': dem_offset,
    'origin_ground_navd88_m': z0,
    'sources': 'DataSF Building Footprints (ynuv-fyni, CC0); AWS Terrain Tiles z15',
}
json.dump(stats, open('scene_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))

# ---------- preview ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 7))
for p, gnd, hgt in buildings:
    xs_, ys_ = p.exterior.xy
    ax.fill(xs_, ys_, color=plt.cm.viridis(min(hgt / 60.0, 1.0)), edgecolor='none')
ax.plot([-250, 250, 0], [250, 250, -250], 'r*', ms=14, mec='k')
ax.set_aspect('equal'); ax.set_xlim(-520, 520); ax.set_ylim(-520, 520)
ax.set_title('Open scene preview — buildings coloured by height (m)\nred stars: towers A, B, C')
ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
fig.tight_layout(); fig.savefig('scene_preview.png', dpi=130)
print('preview saved')
