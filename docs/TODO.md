## TODO 

Comfortable fit (high confidence):

- audio/device.py — tiny, ~20 lines
- analysis/normalise.py — ~60 lines
- analysis/groove/ — 4 files, manageable
- Updated audio/decode.py (absorbing make_track_id, parse_filename, utc_now_iso)
- Updated analysis/rhythm.py (absorbing onsets_to_pattern, classify_groove_feel)

Tight but probably fits:

- transcription/hooks.py + updated transcription/whisper.py
- tests/test_utils.py splits

Unlikely to finish cleanly:

- analysis/harmony/ — large, chord templates are verbose
- analysis/structure/ — the biggest module, alignment logic alone is ~200 lines

