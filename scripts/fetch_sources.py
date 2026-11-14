#!/usr/bin/env python3
"""Download the open inputs that build_scene.py needs, into the current directory.

  footprints_raw.json   City and County of San Francisco "Building Footprints"
                        (DataSF dataset ynuv-fyni, CC0), every footprint in a box
                        around the tile, with LiDAR ground and height statistics.
  terrarium_tiles.pkl   AWS Terrain Tiles (terrarium encoding, zoom 15) covering
  terrarium_meta.npy    the tile, decoded to metres.

Usage:  python fetch_sources.py   then   python build_scene.py
"""
import io, json, math, pickle, urllib.parse, urllib.request
import numpy as np
from PIL import Image

LAT0, LON0 = 37.801105, -122.404175
HALF_M = 650.0          # box half-width; build_scene.py keeps |x|,|y| <= 510 m (terrain to 525 m)
ZOOM = 15
DATASF = 'https://data.sf.gov/resource/ynuv-fyni.json'
TILES = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'

dlat = math.degrees(HALF_M / 6378137.0)
dlon = math.degrees(HALF_M / (6378137.0 * math.cos(math.radians(LAT0))))
north, south, west, east = LAT0 + dlat, LAT0 - dlat, LON0 - dlon, LON0 + dlon

def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()

# ---------- building footprints ----------
rows, offset = [], 0
where = f'within_box(shape, {north}, {west}, {south}, {east})'
while True:
    q = urllib.parse.urlencode({'$where': where, '$limit': 5000, '$offset': offset, '$order': ':id'})
    page = json.loads(get(f'{DATASF}?{q}'))
    rows += page
    offset += len(page)
    if len(page) < 5000:
        break
json.dump(rows, open('footprints_raw.json', 'w'))
print(f'footprints: {len(rows)} rows -> footprints_raw.json')

# ---------- terrain tiles ----------
def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)

tx0, ty0 = tile_xy(west, north, ZOOM)
tx1, ty1 = tile_xy(east, south, ZOOM)
tiles = {}
for tx in range(tx0, tx1 + 1):
    for ty in range(ty0, ty1 + 1):
        rgb = np.asarray(Image.open(io.BytesIO(get(TILES.format(z=ZOOM, x=tx, y=ty)))).convert('RGB'), dtype=np.float64)
        tiles[(tx, ty)] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
np.save('terrarium_meta.npy', np.array([tx0, ty0, tx1, ty1, ZOOM]))
pickle.dump(tiles, open('terrarium_tiles.pkl', 'wb'))
print(f'terrain: {len(tiles)} tiles at z{ZOOM} -> terrarium_tiles.pkl, terrarium_meta.npy')
