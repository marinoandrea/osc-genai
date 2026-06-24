#!/usr/bin/env python3
"""Fetch and organise the midm-database.co.uk MIDI transcriptions.

Reproducible, declarative-ish pipeline for the *fetchable* part of the dataset
(pre-existing local clips such as the Goa Leads / Acid Lines / drum packs are NOT
handled here -- they are not downloadable and are managed by hand).

Pipeline:
  1. crawl http://midm-database.co.uk (its HTTPS is broken) starting from the three
     artist pages, discovering album pages and every reachable ``.mid``/``.midi`` link;
     artist is attributed by which artist page the link descends from.
  2. download each unique file to ``<out>/_originals/<Artist>/`` (full songs).
  3. split each song into single-instrument clips (one per track/channel), classify
     each by its track name (+ channel 10 -> Drums), and write them to
     ``<out>/<Instrument>/<Artist>/``. Every note is preserved.
  4. write ``<out>/MANIFEST_midm.csv``.

Usage:
    uv run python scripts/fetch_midm_database.py            # -> data/MIDI
    uv run python scripts/fetch_midm_database.py --out /tmp/midi --force
    uv run python scripts/fetch_midm_database.py --no-originals   # drop full songs
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import mido
from mido import MetaMessage, MidiFile, MidiTrack

BASE = "http://midm-database.co.uk/"
# The site's HTTPS handshake is broken, so we deliberately speak HTTP.
ARTIST_PAGES = {
    "AFX.html": "Aphex Twin",
    "Autechre.html": "Autechre",
    "BOC.html": "Boards of Canada",
}
USER_AGENT = "osc-genai-fetch/1.0 (+midm-database mirror)"

# Instrument classification by track name. Ordered; first keyword hit wins.
# (GM program data on this site is unreliable -- every track defaults to Piano --
# so the human-written track names are the real instrument signal.)
RULES = [
    ("Drums", ["drum", "kick", "kck", "kik", "snare", "snr", "hat", "hh", "clsd", "ohh",
               "clap", "perc", "cymbal", "crash", "ride", "tom", "rim", "machine", "mch",
               "909", "808", "606", "707"]),
    ("Bass", ["bass"]),
    ("Organ", ["organ"]),
    ("Bell", ["bell"]),
    ("Pluck", ["pluck"]),
    ("Pipe", ["ocarina", "flute", "pipe"]),
    ("Strings", ["string", "violin", "cello", "viola"]),
    ("Piano", ["klavier", "piano", "grand", "epiano", "e-piano", "rhodes", "boesendorfer",
               "steinway", "yamaha", "fl keys", " keys", "keys ", "upper", "lower",
               "primo", "secundo", " hand"]),
    ("Lead", ["lead", "acid", " ld ", "ld_", "supersaw"]),
    ("Brass", ["brass"]),
    ("Pad", ["pad"]),
    ("Synth", ["synth", "massive", "sylenth", "poizone", "toxic", "tal-", "poly",
               "saw ", "square", "es p"]),
]


# --------------------------------------------------------------------------- fetch
def fetch(url: str, timeout: int = 30) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001 -- dead links / timeouts are expected
        print(f"  ! fetch failed ({e}): {url}", file=sys.stderr)
        return None


def links(html: str) -> list[str]:
    return re.findall(r'(?:href|src)=["\']?([^"\'> ]+)', html, re.I)


def crawl() -> dict[str, str]:
    """Return ``{mid_url: artist}`` for every reachable MIDI link.

    Structure is exactly two levels deep: artist page -> album pages -> ``.mid``.
    We must NOT BFS freely: every album page carries the site nav (links back to all
    three artist pages), so an unconstrained crawl would attribute every file to
    whichever artist we started from. So: collect album pages linked *directly* from
    each artist page (skipping the nav/index links), then pull ``.mid`` links from
    those album pages only.
    """
    nav = set(ARTIST_PAGES) | {"index.html"}
    mids: dict[str, str] = {}
    for page, artist in ARTIST_PAGES.items():
        raw = fetch(urllib.parse.urljoin(BASE, page))
        if raw is None:
            continue
        albums: list[str] = []
        for href in links(raw.decode("utf-8", "replace")):
            full = urllib.parse.urljoin(BASE, href)
            low = full.lower()
            if low.endswith((".mid", ".midi")):
                mids.setdefault(full, artist)
            elif full.startswith(BASE) and low.endswith(".html"):
                tail = full[len(BASE):]
                if tail not in nav and full not in albums:
                    albums.append(full)
        for album in albums:
            raw = fetch(album)
            if raw is None:
                continue
            for href in links(raw.decode("utf-8", "replace")):
                full = urllib.parse.urljoin(album, href)
                if full.lower().endswith((".mid", ".midi")):
                    mids.setdefault(full, artist)
    return mids


def download(mids: dict[str, str], originals_dir: str, force: bool) -> list[tuple[str, str, str]]:
    """Download each file. Returns ``[(path, artist, source_filename), ...]``."""
    got: list[tuple[str, str, str]] = []
    seen_names: set[str] = set()
    for url, artist in sorted(mids.items()):
        name = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path))
        if name in seen_names:  # same song linked from two albums
            continue
        dest_dir = os.path.join(originals_dir, artist)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest) and not force:
            seen_names.add(name)
            got.append((dest, artist, name))
            continue
        raw = fetch(url)
        if not raw:
            continue
        try:  # validate it parses as MIDI before keeping it
            mido.MidiFile(file=__import__("io").BytesIO(raw))
        except Exception as e:  # noqa: BLE001
            print(f"  ! not a valid MIDI, skipping ({e}): {url}", file=sys.stderr)
            continue
        with open(dest, "wb") as fh:
            fh.write(raw)
        seen_names.add(name)
        got.append((dest, artist, name))
        print(f"  + {artist}/{name}")
    return got


# --------------------------------------------------------------------------- split
def classify(name: str, channel: int) -> str:
    if channel == 9:
        return "Drums"
    n = " " + name.lower() + " "
    for cat, kws in RULES:
        if any(kw in n for kw in kws):
            return cat
    return "Unclassified"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", "".join(c for c in s if c.isprintable())).strip()


def _safe(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 ()_-]+", "-", s).strip(" -")
    return re.sub(r"-+", "-", s)[:60] or "track"


def _write(path: str, ticks: int, metas, events, label: str) -> int:
    mf = MidiFile(type=0, ticks_per_beat=ticks)
    tr = MidiTrack()
    tr.append(MetaMessage("track_name", name=label[:120], time=0))
    merged = [(a, 0, m) for a, m in metas] + [(a, 1, m) for a, m in events]
    merged.sort(key=lambda x: (x[0], x[1]))
    last = 0
    notes = 0
    for a, _, m in merged:
        tr.append(m.copy(time=a - last))
        last = a
        if m.type == "note_on" and m.velocity > 0:
            notes += 1
    tr.append(MetaMessage("end_of_track", time=0))
    mf.tracks.append(tr)
    mf.save(path)
    return notes


def split_song(src_path: str, artist: str, out_root: str, manifest: list, collisions: dict):
    """Split one full song into single-instrument clips under <out>/<Inst>/<Artist>/."""
    stem = os.path.splitext(os.path.basename(src_path))[0]
    mf = MidiFile(src_path)
    ticks = mf.ticks_per_beat

    metas, seen_meta = [], set()
    for tr in mf.tracks:
        a = 0
        for msg in tr:
            a += msg.time
            if msg.is_meta and msg.type in ("set_tempo", "time_signature", "key_signature"):
                key = (a, msg.type, str(msg.bytes()))
                if key not in seen_meta:
                    seen_meta.add(key)
                    metas.append((a, msg))

    for ti, tr in enumerate(mf.tracks):
        names, a = [], 0
        ev_by_ch = defaultdict(list)
        notes_by_ch = Counter()
        for msg in tr:
            a += msg.time
            if msg.is_meta:
                if msg.type == "track_name":
                    nm = _clean(msg.name)
                    if nm and nm.lower() != "control track":
                        names.append(nm)
                continue
            if hasattr(msg, "channel"):
                ev_by_ch[msg.channel].append((a, msg))
                if msg.type == "note_on" and msg.velocity > 0:
                    notes_by_ch[msg.channel] += 1
        active = [c for c in ev_by_ch if notes_by_ch[c] > 0]
        for ch in active:
            label = None
            for nm in names:
                if re.match(rf"0?{ch}\s*:", nm):
                    label = nm
                    break
            if label is None:
                if len(active) == 1 and names:
                    label = names[0]
                elif len(names) == len(active):
                    label = names[active.index(ch)]
                elif names:
                    label = f"{names[0]} ch{ch}"
                else:
                    label = f"ch{ch}"
            inst = classify(label, ch)
            out_dir = os.path.join(out_root, inst, artist)
            os.makedirs(out_dir, exist_ok=True)
            base = f"{stem}__{_safe(label)}"
            key = os.path.join(inst, artist, base)
            fn = base + ".mid" if not collisions[key] else f"{base}-{collisions[key] + 1}.mid"
            collisions[key] += 1
            out_path = os.path.join(out_dir, fn)
            n = _write(out_path, ticks, metas, ev_by_ch[ch], label)
            manifest.append({
                "instrument": inst, "artist": artist,
                "file": os.path.relpath(out_path, out_root),
                "source_song": os.path.basename(src_path),
                "track_label": label, "notes": n,
            })


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/MIDI", help="output root (default: data/MIDI)")
    ap.add_argument("--force", action="store_true", help="re-download even if a file exists")
    ap.add_argument("--no-originals", action="store_true",
                    help="delete the full-song originals after splitting")
    args = ap.parse_args(argv)

    originals = os.path.join(args.out, "_originals")
    os.makedirs(originals, exist_ok=True)

    print("Crawling midm-database.co.uk ...")
    mids = crawl()
    print(f"  found {len(mids)} MIDI links across {len(ARTIST_PAGES)} artists")

    print("Downloading ...")
    songs = download(mids, originals, args.force)
    print(f"  {len(songs)} unique songs available")

    print("Splitting into single-instrument clips ...")
    manifest: list[dict] = []
    collisions: dict[str, int] = defaultdict(int)
    for path, artist, _name in songs:
        split_song(path, artist, args.out, manifest, collisions)

    man_path = os.path.join(args.out, "MANIFEST_midm.csv")
    with open(man_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["instrument", "artist", "file",
                                           "source_song", "track_label", "notes"])
        w.writeheader()
        w.writerows(sorted(manifest, key=lambda r: (r["instrument"], r["artist"], r["file"])))

    if args.no_originals:
        import shutil
        shutil.rmtree(originals, ignore_errors=True)

    by_inst = Counter(r["instrument"] for r in manifest)
    print(f"\nWrote {len(manifest)} clips to {args.out}")
    for inst, n in by_inst.most_common():
        print(f"  {n:3d}  {inst}")
    print(f"manifest: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
