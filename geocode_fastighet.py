#!/usr/bin/env python3
"""Geocode Swedish fastighetsbeteckningar to WGS84 lat/lon (and boundary polygons).

Uses the public search backend of Lantmäteriet's Min karta
(https://minkarta.lantmateriet.se/). The search returns a representative
point, and (for GeoJSON/GeoPackage) a second call returns the parcel boundary,
both in SWEREF 99 TM (EPSG:3006). We reproject to WGS84 (EPSG:4326) with
Lantmäteriet's own Gauss-Krüger formula.

Input is either a single beteckning string or a CSV file with a header row and
the beteckningar in the first column (extra columns are ignored; the column
delimiter -- comma, semicolon or tab -- is detected automatically). The block/unit
separator may be a colon ("emmaboda emmabo 1:116") or a space ("emmaboda emmabo
1 116"); both are accepted with no extra flag.

Output files are created automatically with a timestamped name, next to the
input CSV (or in the current directory for a string input):
    <name>_<YYYYMMDD_HHMMSS>.csv       always
    <name>_<YYYYMMDD_HHMMSS>.geojson   with --geojson
    <name>_<YYYYMMDD_HHMMSS>.gpkg      with --gpkg

Usage:
    ./geocode_fastighet.py "emmaboda emmabo 1:116"            # -> .csv
    ./geocode_fastighet.py fastigheter.csv                    # -> .csv
    ./geocode_fastighet.py --geojson fastigheter.csv          # -> .csv + .geojson
    ./geocode_fastighet.py --gpkg fastigheter.csv             # -> .csv + .gpkg
    ./geocode_fastighet.py --geojson --gpkg fastigheter.csv   # -> all three

Options:
    --geojson         also write a GeoJSON FeatureCollection (boundary polygons)
    --gpkg            also write an OGC GeoPackage (.gpkg) MULTIPOLYGON layer, WGS84
    --delay SECONDS   pause between beteckningar (default 0.15; use 0 for none)
    --debug           log the exact search text sent and the hits returned per
                      row (and the detected column delimiter) -- use this to see
                      why rows come back "not found"

CSV and GeoJSON use only the standard library; --gpkg requires geopandas
(pip install geopandas). Requests retry automatically on HTTP 429 / transient
errors with backoff (honouring Retry-After), so rate limits slow the run
rather than dropping rows.
"""
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from math import asin, atan, cos, cosh, pi, sin, sinh

SEARCH_URL = "https://minkarta.lantmateriet.se/api/searchservice/searchinput"
GEOM_URL = "https://minkarta.lantmateriet.se/api/searchservice/fastighetsgeometri/v1"
HEADERS = {
    "Referer": "https://minkarta.lantmateriet.se/",
    "Accept": "application/json",
    "User-Agent": "fastighet-geocoder/1.0",
}
DEFAULT_DELAY = 0.15        # seconds between beteckningar (be polite)
MAX_RETRIES = 5            # retries on 429 / transient errors
RETRY_STATUS = {429, 500, 502, 503, 504}


def sweref99tm_to_wgs84(north, east):
    """SWEREF 99 TM (EPSG:3006) -> WGS84 (lat, lon). Lantmäteriet Gauss-Krüger."""
    a, f = 6378137.0, 1 / 298.257222101      # GRS80 ellipsoid
    lon0, k0, fe, fn = 15.0, 0.9996, 500000.0, 0.0
    e2 = f * (2 - f)
    n = f / (2 - f)
    aroof = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    d1 = n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96 - n**4 / 360
    d2 = n**2 / 48 + n**3 / 15 - 437 * n**4 / 1440
    d3 = 17 * n**3 / 480 - 37 * n**4 / 840
    d4 = 4397 * n**4 / 161280
    As = e2 + e2**2 + e2**3 + e2**4
    Bs = -(7 * e2**2 + 17 * e2**3 + 30 * e2**4) / 6
    Cs = (224 * e2**3 + 889 * e2**4) / 120
    Ds = -(4279 * e2**4) / 1260
    dr = pi / 180
    l0 = lon0 * dr
    xi = (north - fn) / (k0 * aroof)
    eta = (east - fe) / (k0 * aroof)
    xip = (xi - d1 * sin(2 * xi) * cosh(2 * eta) - d2 * sin(4 * xi) * cosh(4 * eta)
           - d3 * sin(6 * xi) * cosh(6 * eta) - d4 * sin(8 * xi) * cosh(8 * eta))
    etap = (eta - d1 * cos(2 * xi) * sinh(2 * eta) - d2 * cos(4 * xi) * sinh(4 * eta)
            - d3 * cos(6 * xi) * sinh(6 * eta) - d4 * cos(8 * xi) * sinh(8 * eta))
    phis = asin(sin(xip) / cosh(etap))
    dl = atan(sinh(etap) / cos(xip))
    lon = (l0 + dl) / dr
    lat = (phis + sin(phis) * cos(phis)
           * (As + Bs * sin(phis)**2 + Cs * sin(phis)**4 + Ds * sin(phis)**6)) / dr
    return lat, lon


def _retry_after(headers):
    """Seconds from a Retry-After header (integer form), or None."""
    val = headers.get("Retry-After")
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return None  # missing, or HTTP-date form -> fall back to backoff


def _get_json(url, params, max_retries=MAX_RETRIES):
    """GET JSON, retrying on HTTP 429/5xx and transient network errors.

    Honours a Retry-After header when present, otherwise uses capped
    exponential backoff (1, 2, 4, 8, 16 s). Raises after max_retries.
    """
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == max_retries:
                raise
            wait = _retry_after(e.headers) or min(2 ** attempt, 30)
            print(f"  HTTP {e.code}, retrying in {wait}s "
                  f"({attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
        except urllib.error.URLError:  # DNS, connection reset, timeout, ...
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"  network error, retrying in {wait}s "
                  f"({attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)


def _detect_delimiter(lines):
    """Guess the CSV column delimiter: ';', tab, or ','.

    Excel on European/Swedish Windows saves CSVs delimited with ';', which the
    default ',' reader would swallow into a single column -- making the whole
    line the search text and every lookup fail. Beteckningar contain none of
    these characters, so we pick whichever candidate appears the same non-zero
    number of times in every data row, falling back to ',' (e.g. a
    single-column file has no delimiter at all).
    """
    data = [ln for ln in lines[1:] if ln.strip()][:20]  # sample data rows
    if not data:
        data = [ln for ln in lines if ln.strip()][:20]
    for delim in (";", "\t", ","):
        counts = [ln.count(delim) for ln in data]
        if counts and counts[0] > 0 and all(c == counts[0] for c in counts):
            return delim
    return ","


def _read_text(path):
    """Read a text file, tolerating the encodings Excel produces on Windows.

    Tries UTF-8 (with or without a BOM) first, then Windows-1252 (a.k.a. ANSI /
    CP1252) -- the default Excel uses on a Swedish Windows PC -- so a CSV with
    å/ä/ö saved straight from Excel decodes instead of raising. UTF-8 is tried
    first, so a genuine UTF-8 file is never mis-read as CP1252.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")  # 1 byte -> 1 char, never raises


def _norm(s):
    return " ".join(s.lower().split())


# Trailing "block unit" as two space-separated numbers, e.g. the "1 116" in
# "emmaboda emmabo 1 116". Anchored to the end so numbers earlier in the trakt
# name are left alone.
_SPACE_SEP_RE = re.compile(r"(\d+)\s+(\d+)\s*$")


def normalize_beteckning(beteckning):
    """Accept the block/unit separator as either ':' or a space.

    Lantmäteriet's canonical form uses a colon ("emmaboda emmabo 1:116").
    Some input files use a space instead ("emmaboda emmabo 1 116"); this turns
    the trailing "block unit" pair into "block:unit" so the search and the
    exact-name match both work. Strings that already contain a colon (and any
    that don't end in two numbers) are returned unchanged, so colon- and
    space-style files are both handled with no configuration.
    """
    s = " ".join(beteckning.split())  # collapse/trim whitespace
    if ":" in s:
        return s
    return _SPACE_SEP_RE.sub(r"\1:\2", s)


def _reproject_ring(ring):
    """List of SWEREF99 TM [east, north] -> list of WGS84 [lon, lat]."""
    out = []
    for east, north in ring:
        lat, lon = sweref99tm_to_wgs84(north, east)
        out.append([round(lon, 6), round(lat, 6)])
    return out


def fetch_polygons(objektidentitet):
    """Return MultiPolygon coordinates (WGS84) for a property, or [] if none."""
    data = _get_json(GEOM_URL, {"objektidentitet": objektidentitet})
    multipoly = []
    for area in data.get("enhetsutbredning") or []:
        for geom in area.get("yta") or []:
            gtype = geom.get("type")
            if gtype == "Polygon":
                polys = [geom.get("coordinates", [])]
            elif gtype == "MultiPolygon":
                polys = geom.get("coordinates", [])
            else:
                continue
            for rings in polys:
                reprojected = [_reproject_ring(r) for r in rings if r]
                if reprojected:
                    multipoly.append(reprojected)
    return multipoly


def geocode(beteckning, want_polygon=False, debug=False):
    """Return a dict with matched name, lat, lon, status (and polygon if asked)."""
    result = {"input": beteckning, "matched": "", "lat": "", "lon": "",
              "status": "", "polygon": []}
    query = normalize_beteckning(beteckning)  # accept "1 116" as "1:116"
    if debug:
        print(f"    -> searching for {query!r}", file=sys.stderr, flush=True)
    try:
        data = _get_json(SEARCH_URL, {"searchtext": query})
    except Exception as e:  # network / HTTP / parse error
        result["status"] = f"error: {e}"
        if debug:
            print(f"    !! request failed: {e}", file=sys.stderr, flush=True)
        return result

    results = data.get("sokresultat") or []
    if debug:
        print(f"    <- {len(results)} hit(s) returned", file=sys.stderr, flush=True)
        for hit in results[:5]:
            print(f"         {hit.get('headertext', '')!r}", file=sys.stderr, flush=True)
    if not results:
        result["status"] = "not found"
        return result

    # Prefer an exact name match; otherwise take the first hit and flag it.
    exact = next((r for r in results if _norm(r.get("headertext", "")) == _norm(query)), None)
    chosen = exact or results[0]
    pos = chosen["position"]
    lat, lon = sweref99tm_to_wgs84(pos["north"], pos["east"])
    result["matched"] = chosen.get("headertext", "")
    result["lat"] = round(lat, 6)
    result["lon"] = round(lon, 6)
    result["status"] = "ok" if exact else f"approx: {len(results)} hits, no exact match"

    if want_polygon:
        try:
            result["polygon"] = fetch_polygons(chosen["id"])
            if not result["polygon"]:
                result["status"] += "; no boundary geometry"
        except Exception as e:
            result["status"] += f"; polygon error: {e}"
    return result


def to_feature(r):
    """Build a GeoJSON Feature: MultiPolygon boundary if available, else Point."""
    props = {"input": r["input"], "matched": r["matched"], "status": r["status"]}
    if r["lat"] != "":
        props["lat"], props["lon"] = r["lat"], r["lon"]
    if r["polygon"]:
        geometry = {"type": "MultiPolygon", "coordinates": r["polygon"]}
    elif r["lat"] != "":
        geometry = {"type": "Point", "coordinates": [r["lon"], r["lat"]]}
    else:
        geometry = None
    return {"type": "Feature", "properties": props, "geometry": geometry}


def write_geopackage(rows, path, layer="fastigheter"):
    """Write results as an OGC GeoPackage MULTIPOLYGON layer (WGS84) via geopandas."""
    try:
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, Polygon
    except ImportError:
        raise SystemExit("The --gpkg export needs geopandas: pip install geopandas")

    geoms, records = [], []
    for r in rows:
        if r["polygon"]:
            # r["polygon"] = MultiPolygon coords: [ [exterior_ring, *hole_rings], ... ]
            polys = [Polygon(rings[0], rings[1:]) for rings in r["polygon"]]
            geoms.append(MultiPolygon(polys))
        else:
            geoms.append(None)  # keep attributes (incl. point) with NULL geometry
        records.append({
            "input": r["input"],
            "matched": r["matched"] or None,
            "lat": r["lat"] if r["lat"] != "" else None,
            "lon": r["lon"] if r["lon"] != "" else None,
            "status": r["status"],
        })

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
    if os.path.exists(path):
        os.remove(path)  # overwrite cleanly
    gdf.to_file(path, layer=layer, driver="GPKG")


def main(argv):
    want_geojson = want_gpkg = debug = False
    delay = DEFAULT_DELAY
    while argv and argv[0].startswith("-"):
        opt = argv.pop(0)
        if opt in ("--geojson", "-g"):
            want_geojson = True
        elif opt == "--gpkg":
            want_gpkg = True
        elif opt in ("--debug", "-d"):
            debug = True
        elif opt == "--delay":
            try:
                delay = float(argv.pop(0))
            except (IndexError, ValueError):
                print("--delay requires a number of seconds, e.g. --delay 0.5", file=sys.stderr)
                return 2
        elif opt in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            print(f"unknown option: {opt}", file=sys.stderr)
            return 2

    if not argv:
        print(__doc__)
        return 1

    # Fail fast if a GeoPackage is requested but geopandas is unavailable,
    # rather than after doing all the network lookups.
    if want_gpkg:
        try:
            import geopandas  # noqa: F401
        except ImportError:
            print("The --gpkg export needs geopandas: pip install geopandas", file=sys.stderr)
            return 2

    # An existing file -> CSV with a header row, beteckningar in the first column;
    # outputs land next to it. Otherwise treat all args as one query string.
    if len(argv) == 1 and os.path.isfile(argv[0]):
        inpath = argv[0]
        base = os.path.splitext(os.path.basename(inpath))[0]
        outdir = os.path.dirname(os.path.abspath(inpath))
        # splitlines() handles \n, \r\n and \r; _read_text() copes with UTF-8
        # or Windows-1252 (Excel's default), so Swedish characters don't fail.
        lines = _read_text(inpath).splitlines()
        delimiter = _detect_delimiter(lines)  # ',' or ';' (Excel/EU) or tab
        reader = csv.reader(lines, delimiter=delimiter)
        next(reader, None)  # skip header row
        queries = [row[0].strip() for row in reader if row and row[0].strip()]
        if debug:
            print(f"    input: {len(queries)} data row(s), delimiter="
                  f"{delimiter!r}; using the first column", file=sys.stderr, flush=True)
            for q in queries[:5]:
                print(f"         parsed: {q!r}", file=sys.stderr, flush=True)
    else:
        queries = [" ".join(argv)]
        base, outdir = "fastighet", os.getcwd()

    stem = os.path.join(outdir, f"{base}_{time.strftime('%Y%m%d_%H%M%S')}")
    csv_path = stem + ".csv"

    want_polygon = want_geojson or want_gpkg
    rows = []
    total = len(queries)
    if total == 0:
        print("No beteckningar found in the input -- is the first column empty, "
              "or does the file have only a header row? Run with --debug to see "
              "how the file was parsed.", file=sys.stderr)
        return 1
    # Progress and warnings go to stderr so a summary on stdout stays clean.
    print(f"Looking up {total} beteckning(ar)...", file=sys.stderr, flush=True)
    # utf-8-sig so Swedish characters display correctly in Excel.
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["input", "matched", "lat", "lon", "status"],
            extrasaction="ignore")
        writer.writeheader()
        for i, q in enumerate(queries):
            r = geocode(q, want_polygon=want_polygon, debug=debug)
            rows.append(r)
            writer.writerow(r)
            cf.flush()  # keep partial results on disk during long runs

            # Live progress: one line per row, flagging anything not an exact
            # match (not found, error, or approx) so problems are easy to spot.
            status = r["status"]
            matched = f"  ->  {r['matched']}" if r["matched"] else ""
            flag = "" if status.startswith("ok") else "   <-- CHECK"
            print(f"[{i + 1}/{total}] {q}{matched}  [{status}]{flag}",
                  file=sys.stderr, flush=True)

            if delay > 0 and i + 1 < total:
                time.sleep(delay)  # be polite to the service

    outputs = [csv_path]

    if want_geojson:
        geojson_path = stem + ".geojson"
        collection = {"type": "FeatureCollection", "features": [to_feature(r) for r in rows]}
        with open(geojson_path, "w", encoding="utf-8") as gf:
            json.dump(collection, gf, ensure_ascii=False)
            gf.write("\n")
        outputs.append(geojson_path)

    if want_gpkg:
        gpkg_path = stem + ".gpkg"
        write_geopackage(rows, gpkg_path)
        outputs.append(gpkg_path)

    matched = sum(1 for r in rows if r["status"].startswith("ok"))
    approx = sum(1 for r in rows if r["status"].startswith("approx"))
    missing = len(rows) - matched - approx
    print(f"Done: {len(rows)} row(s) -- {matched} exact, {approx} approx, "
          f"{missing} not found/error.")
    print("Output files:")
    for p in outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
