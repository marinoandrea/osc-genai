"""Audio ingestion: turn a real (non-MIDI) **monophonic** instrument line into ``Note``s on the fly.

The duet partner normally plays MIDI; this package adds an alternative front-end that captures the
raw audio of a single-line instrument (a bass), runs our own **YIN** pitch tracker over a sliding
buffer (:mod:`osc_genai.audio.yin`), segments the per-frame pitch into discrete notes
(:mod:`osc_genai.audio.segment`), and feeds those into the same ``HumanStream`` the duet already
consumes — so generation *and* snapshot-to-dataset work unchanged.

YIN estimates a *single* fundamental per frame, so this path is **monophonic only**: if the bass
plays a chord, only its strongest/lowest note is tracked.
"""
