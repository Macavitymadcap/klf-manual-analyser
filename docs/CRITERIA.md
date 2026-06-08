# Criteria Reference

KLF Manual Analyser — criterion definitions, research basis, and TOML
specifications for all three modes.

This document is the specification from which the TOML config files are
generated. Each criterion lists its source, its measurable proxy, and its
expected scoring behaviour.

---

## Table of Contents

1. [How criteria work](#how-criteria-work)
2. [Mode: 1988](#mode-1988)
3. [Mode: contemporary](#mode-contemporary)
4. [Mode: 1920s_1930s](#mode-1920s_1930s)
5. [Cross-mode notes](#cross-mode-notes)

---

## How criteria work

Each criterion maps to one of these scoring rules:

| Rule | Description |
|---|---|
| `lte` | Pass if `db_field <= threshold` |
| `gte` | Pass if `db_field >= threshold` |
| `eq` | Pass if `db_field == threshold` |
| `range` | Pass if `threshold_min <= db_field <= threshold_max` |
| `exists` | Pass if any `sections` row for this track has `label == value` |
| `llm` | Routed to LLM with feature values and `prompt_hint`; returns 0–10 score |

Deterministic rules (`lte`, `gte`, `eq`, `range`, `exists`) require no LLM call.
`llm` rules produce a score and a reasoning string; score is normalised 0.0–1.0.

`db_field` (singular string) is used for single-field criteria.
`db_fields` (array of strings) is used for multi-field `llm` criteria.
The two keys are mutually exclusive and validated at load time.

All scores are normalised to 0.0–1.0 before storage. Overall compliance:

```
sum(score_i * weight_i) / sum(weight_i)
```

---

## Mode: 1988

Context: UK singles market, late 1988. Gallup chart. 7" format. Radio One
daytime DJs. Top of the Pops. Stock Aitken Waterman dominance.

The four numbered Golden Rules from The Manual:

> 1. A dance groove that runs all the way through the record
> 2. No longer than three minutes and thirty seconds
> 3. intro → verse → chorus → verse → chorus → breakdown → double chorus → outro
> 4. Lyrics: you will need some, but not many

Sub-sections add criteria for groove, chorus, title, verse, intro, breakdown,
and keys.

---

### `bpm` — Tempo ceiling
**Source**: "No song with a BPM over 135 will ever have a chance of getting to
Number One."

```toml
[[criterion]]
id = "bpm"
name = "Tempo Ceiling"
description = "No song with a BPM over 135 will ever reach Number One."
weight = 1.5
db_field = "tracks.bpm"
rule = "lte"
threshold = 135
unit = "BPM"
fail_message = "BPM exceeds the 135 ceiling. The Manual is clear on this."
```

---

### `duration` — Track length
**Source**: "No longer than 3'30". Just under 3'20 is preferable. Radio One
DJs will fade or talk over anything longer."

```toml
[[criterion]]
id = "duration"
name = "Track Length"
description = """
  No longer than three minutes thirty seconds. Just under 3'20 is preferable.
  Longer tracks get faded or talked over by Radio One DJs.
"""
weight = 1.0
db_field = "tracks.duration"
rule = "lte"
threshold = 210
unit = "seconds"
fail_message = "Track exceeds 3'30\". Radio One will talk over your outro."
```

---

### `groove` — Continuous dance groove
**Source**: "It has to have a dance groove that will run all the way through
the record and that the current 7\" buying generation will find irresistible."
Weight 2.0 — The Manual's most-emphasised criterion.

```toml
[[criterion]]
id = "groove"
name = "Continuous Dance Groove"
description = """
  A single dance groove must run all the way through the record, irresistible
  to the current 7\" buying generation. Without it, nothing else matters.
"""
weight = 2.0
db_fields = [
  "tracks.self_similarity_score",
  "tracks.beat_regularity",
  "tracks.danceability",
  "tracks.groove_consistency"
]
rule = "llm"
prompt_hint = """
  self_similarity_score: how consistently the sonic texture recurs (0–1).
  beat_regularity: how metronomic the groove is (0–1).
  danceability: essentia descriptor (0–1).
  groove_consistency: composite of the above (0–1).
  A track with high self_similarity and beat_regularity has a groove running
  relentlessly start to finish. A track that shifts feel, breaks groove, or
  loses momentum violates the first Golden Rule.
  Score 0-10. High = irresistible, consistent groove throughout.
"""
```

---

### `structure` — Canonical song structure
**Source**: "The Golden Rule for a classic Number One: intro, verse one, chorus
one, verse two, chorus two, breakdown section, double chorus, outro. Each section
in multiples of four bars."

```toml
[[criterion]]
id = "structure"
name = "Canonical Song Structure"
description = """
  intro → verse → chorus → verse → chorus → breakdown → double chorus → outro.
  Each section in multiples of four bars.
"""
weight = 1.5
db_field = "sections.label"
rule = "llm"
prompt_hint = """
  You are given an ordered list of section labels with confidence scores,
  derived by cross-referencing acoustic segmentation with lyric timestamps.
  Low-confidence labels are flagged.
  The canonical Manual structure is:
    intro → verse → chorus → verse → chorus → breakdown → double_chorus → outro
  The breakdown (low-energy relief) and double_chorus (extended climactic chorus)
  are the most significant elements — their absence is a meaningful failing.
  Approximate matches (e.g. a pre_chorus present) should not heavily penalise.
  Score 0-10.
"""
```

---

### `breakdown` — Breakdown section present
**Source**: The Manual is explicit — breakdown is non-negotiable in the template.

```toml
[[criterion]]
id = "breakdown"
name = "Breakdown Section Present"
description = """
  The breakdown section must exist. It strips the track back, providing
  relief before the double chorus explodes back in.
"""
weight = 1.0
db_field = "sections.label"
rule = "exists"
value = "breakdown"
fail_message = "No breakdown section detected. The Manual requires one."
```

---

### `double_chorus` — Double chorus present
**Source**: "back into a double length chorus and outro."

```toml
[[criterion]]
id = "double_chorus"
name = "Double Chorus Present"
description = "After the breakdown, the double chorus must arrive and land hard."
weight = 1.0
db_field = "sections.label"
rule = "exists"
value = "double_chorus"
fail_message = "No double chorus detected. The breakdown leads nowhere."
```

---

### `chorus_hook` — Irresistible chorus with title
**Source**: "The most important element — the part people carry in their heads.
The song title must appear in the chorus. Chorus lyrics must deal with only the
most basic human emotions."

```toml
[[criterion]]
id = "chorus_hook"
name = "Irresistible Chorus and Title Hook"
description = """
  The chorus must be what people carry in their heads. The song title must
  appear in the chorus. Lyrics must deal with basic human emotions only.
"""
weight = 2.0
db_fields = [
  "tracks.hook_phrase",
  "tracks.hook_repetition_count",
  "tracks.hook_first_appearance",
  "tracks.song_name"
]
rule = "llm"
prompt_hint = """
  hook_phrase: most repeated phrase in the track.
  hook_repetition_count: number of times it appears.
  hook_first_appearance: seconds into the track.
  song_name: parsed from filename.
  Evaluate:
  1. Is the hook phrase simple and emotionally direct (not clever or complex)?
  2. Does the song title appear in the lyrics, ideally in the chorus?
  3. Is the hook repeated enough to be genuinely memorable?
  Score 0-10.
"""
```

---

### `lyrics_economy` — Few lyrics, high repetition
**Source**: "Fourthly, lyrics. You will need some, but not many. The chorus
lyric must never deal with anything but the most basic of human emotions."

```toml
[[criterion]]
id = "lyrics_economy"
name = "Lyric Economy and Repetition"
description = """
  You will need lyrics, but not many. Low unique_word_ratio signals deliberate
  repetition — a virtue, not a flaw.
"""
weight = 1.0
db_field = "tracks.unique_word_ratio"
rule = "llm"
prompt_hint = """
  unique_word_ratio: proportion of unique words to total words (0–1).
  Low ratio (e.g. 0.2–0.3) = heavy repetition = positive signal.
  Also consider whether chorus lyric content deals with basic emotions
  (love, loss, joy, longing) rather than complex or cerebral themes.
  Score 0-10. High = economical, emotionally direct, deliberately repetitive.
"""
```

---

### `verse_bass_riff` — Verse driven by a bass riff
**Source**: Sub-section "The Verse (The Bass Riff Factor)": the verse must be
built around a bass riff that propels the track between choruses.

```toml
[[criterion]]
id = "verse_bass_riff"
name = "Verse Bass Riff Factor"
description = """
  The verse must be propelled by a bass riff — as important as the groove
  for carrying the listener between choruses.
"""
weight = 1.0
db_fields = [
  "sections.label",
  "chord_progressions.progression",
  "beat_patterns.kick_pattern"
]
rule = "llm"
prompt_hint = """
  You are given chord progressions for verse sections and the kick/bass
  pattern. Evaluate whether the verse sections appear driven by a repeating
  bass riff providing momentum between choruses.
  Note: bass stem analysis is approximate; chord detection ~70-75% accurate.
  Score 0-10.
"""
```

---

### `intro_length` — Appropriate intro length
**Source**: Sub-section "The Intro". Research confirms 1980s pop intros averaged
~20 seconds.

```toml
[[criterion]]
id = "intro_length"
name = "Intro Length"
description = "Late-1980s pop intros averaged ~20 seconds. 8–30 is the expected range."
weight = 0.75
db_field = "sections.duration WHERE label='intro'"
rule = "range"
threshold_min = 8
threshold_max = 30
unit = "seconds"
fail_message = "Intro is outside the 8–30 second range expected for the era."
```

---

### `chorus_energy` — Chorus louder than verse
**Source**: Implied throughout — the chorus must hit harder than the verse.
The `verse_chorus_delta` field stores this as a normalised value where
0.15 ≈ 3dB lift.

```toml
[[criterion]]
id = "chorus_energy"
name = "Chorus Energy Lift"
description = "The chorus must hit harder than the verse. The lift must be felt."
weight = 0.75
db_field = "tracks.verse_chorus_delta"
rule = "gte"
threshold = 0.15
fail_message = "Insufficient energy difference between verse and chorus (< ~3dB)."
```

---

### `keys_harmony` — Harmonic simplicity
**Source**: Sub-section "Keys, Notes and Chords": stay in accessible keys,
avoid complexity. Simple I-IV-V progressions are a feature.

```toml
[[criterion]]
id = "keys_harmony"
name = "Harmonic Simplicity"
description = """
  Accessible keys and simple chord progressions. Harmonic complexity is
  not a virtue in this context.
"""
weight = 0.75
db_fields = [
  "tracks.key",
  "tracks.mode",
  "tracks.key_confidence",
  "chord_progressions.progression"
]
rule = "llm"
prompt_hint = """
  You are given the key, mode, and chord progressions per section.
  Note: automatic chord detection is approximately 70-75% accurate on
  modern recordings.
  Evaluate harmonic simplicity: major key with I-IV-V or I-V-vi-IV progressions
  is ideal. Unusual key changes or complex jazz harmonics are red flags.
  Score 0-10. High = simple, accessible, pop-appropriate harmony.
"""
```

---

## Mode: contemporary

The same structural philosophy, updated for the streaming era. Criteria that
are identical to 1988 mode: `groove`, `chorus_hook`, `chorus_energy`,
`lyrics_economy`, `keys_harmony`, `verse_bass_riff`.

Research basis:
- Intros: ~5 seconds (down from ~20); ceiling now 15 seconds
- Hook timing: 25% of listeners skip within 5 seconds; hook must arrive by 30s
- BPM: mainstream pop 118–128 (Spotify data, ~9M tracks); no hard ceiling
- Structure: compressed; breakdown not required; hook-hook-hook forms common
- Song length: ≤ 210 seconds but 3' increasingly typical; TikTok clips shorter

---

### `bpm` — Tempo (contemporary, genre-contextual)

```toml
[[criterion]]
id = "bpm"
name = "Tempo (Genre-Contextual)"
description = "Contemporary pop sits at 118–128 BPM. No hard ceiling — genre-appropriate."
weight = 1.0
db_fields = ["tracks.bpm", "tracks.danceability"]
rule = "llm"
prompt_hint = """
  Contemporary mainstream pop: 118–128 BPM. Dance-pop/EDM can reach ~150.
  Ballad-adjacent (80–100 BPM) can chart but needs very strong hook compensation.
  Evaluate whether BPM is appropriate for the apparent genre and energy level.
  Score 0-10.
"""
```

---

### `duration` — Track length (contemporary)

Identical threshold to 1988 (≤ 210s) but shorter is better in streaming context.

```toml
[[criterion]]
id = "duration"
name = "Track Length"
description = """
  Average chart hit has fallen to ~3 minutes. Songs over 3'30\" risk playlist
  removal and skip penalties on streaming platforms.
"""
weight = 1.0
db_field = "tracks.duration"
rule = "lte"
threshold = 210
unit = "seconds"
fail_message = "Track exceeds 3'30\". Streaming playlists prefer concision."
```

---

### `structure` — Structure (contemporary)

```toml
[[criterion]]
id = "structure"
name = "Song Structure (Streaming-Era)"
description = """
  Hook must arrive before 30 seconds. Intro under 10 seconds. Breakdown
  optional. Hook-hook-hook forms increasingly standard.
"""
weight = 1.5
db_field = "sections.label"
rule = "llm"
prompt_hint = """
  Contemporary streaming-era structure:
  1. Does the hook or chorus arrive within the first 30 seconds?
  2. Is the intro under ~10 seconds?
  3. Is the track concise — no extended hookless sections?
  4. Does it avoid long fade-outs, extended outros, lengthy pre-hook buildups?
  Absence of a breakdown is NOT a failing in contemporary mode.
  Presence of a pre_chorus is a positive signal.
  Score 0-10.
"""
```

---

### `intro_length` — Intro length (contemporary)

```toml
[[criterion]]
id = "intro_length"
name = "Intro Length (Streaming-Era)"
description = "Contemporary pop intros average ~5 seconds. Over 15 seconds is a significant risk."
weight = 1.0
db_field = "sections.duration WHERE label='intro'"
rule = "lte"
threshold = 15
unit = "seconds"
fail_message = "Intro exceeds 15 seconds. Streaming listeners will skip."
```

---

### `hook_timing` — Hook arrival time (contemporary only)

New criterion with no 1988 equivalent.

```toml
[[criterion]]
id = "hook_timing"
name = "Hook Arrival Time"
description = """
  The hook or chorus must arrive within 30 seconds. 25% of streaming listeners
  skip before 30 seconds. Song title should appear in the first minute.
"""
weight = 1.5
db_field = "tracks.hook_first_appearance"
rule = "lte"
threshold = 30
unit = "seconds"
fail_message = "Hook arrives after 30 seconds. Streaming listeners have already skipped."
```

---

## Mode: 1920s_1930s

Period-appropriate criteria for early jazz, dance band, and Tin Pan Alley
recordings. Does not apply Manual criteria; asks equivalent questions for the era.

Context: 78rpm shellac singles. Tin Pan Alley AABA structure. Dance hall music.
No click tracks; swing feel intrinsic. Vocals over live band on a single mic.

Research basis:
- Dominant form: 32-bar AABA, often preceded by a ~16-bar sectional verse
- BPM: foxtrot 112–136, Charleston 180–220, slow blues 60–90, waltz 84–90 (3/4)
- Groove feel: swung expected; straight is atypical for most genres
- Chord vocabulary: I-IV-V dominant; ii-V-I appearing by late 1920s
- Duration: 2'30"–3'30" constrained by 78rpm format
- Chord and Whisper accuracy both significantly lower on 1920s recordings

---

### `bpm_danceform` — Tempo appropriate to dance form

```toml
[[criterion]]
id = "bpm_danceform"
name = "Tempo Appropriate to Dance Form"
description = """
  Foxtrot: 112–136 BPM. Charleston: 180–220 BPM. Slow blues/ballad: 60–90 BPM.
  Waltz: 84–90 BPM in 3/4. Out-of-range tempos suggest an unadaptable track.
"""
weight = 1.5
db_fields = ["tracks.bpm", "tracks.time_signature", "tracks.groove_feel"]
rule = "llm"
prompt_hint = """
  Given BPM, time signature, and groove feel, identify the most likely intended
  dance form:
  - Foxtrot (4/4, 112–136 BPM, smooth feel)
  - Charleston (4/4, 180–220 BPM, energetic)
  - Slow blues / ballad (4/4, 60–90 BPM)
  - Waltz (3/4, 84–90 BPM)
  - Other (specify)
  Evaluate whether tempo is appropriate for that form.
  Note: BPM detection on pre-click-track recordings may be imprecise due to
  natural tempo variation; flag low confidence if stability is poor.
  Score 0-10.
"""
```

---

### `swing_feel` — Swung rhythmic feel

```toml
[[criterion]]
id = "swing_feel"
name = "Swung Rhythmic Feel"
description = """
  Jazz and dance band music of the era has a swung feel — off-beats delayed.
  Straight feel is atypical except for Charleston and novelty songs.
"""
weight = 1.5
db_fields = ["tracks.groove_feel", "tracks.bpm"]
rule = "llm"
prompt_hint = """
  groove_feel: 'straight', 'swung', or 'unclear'.
  For most 1920s–30s genres (foxtrot, blues, big band jazz): swung = high score.
  Charleston and novelty songs may have straight feel — acceptable.
  unclear = neutral score.
  Note: groove_feel detection on pre-click-track recordings may be imprecise.
  Score 0-10. Swung = high score for most genres.
"""
```

---

### `aaba_form` — AABA or equivalent period structure

```toml
[[criterion]]
id = "aaba_form"
name = "AABA or Period Song Form"
description = """
  32-bar AABA was the Tin Pan Alley standard. Four 8-bar sections: A, A, B, A.
  Often preceded by a ~16-bar sectional verse in free rhythm.
  12-bar blues is also valid.
"""
weight = 2.0
db_field = "sections.label"
rule = "llm"
prompt_hint = """
  You are given the detected section sequence with confidence scores.
  Terminology note: in 1920s–30s usage, 'the verse' was the long free-rhythm
  intro; 'the chorus' was the entire 32-bar AABA structure.
  Evaluate whether the structure approximates:
  1. AABA form (sectional verse + A A B A sequence), OR
  2. 12-bar blues (AAB lyric over 12-bar harmonic cycle), OR
  3. Another identifiable period form (ABAC, verse-refrain, etc.)
  Structural detection on 1920s recordings is less reliable than on modern ones.
  Score 0-10.
"""
```

---

### `hook_refrain` — Memorable refrain with title

```toml
[[criterion]]
id = "hook_refrain"
name = "Memorable Refrain with Title"
description = """
  The AABA refrain must contain the song title and be the repeatable commercial
  core. Tin Pan Alley songs were sold as sheet music; the refrain was the hook.
"""
weight = 1.5
db_fields = [
  "tracks.hook_phrase",
  "tracks.hook_repetition_count",
  "tracks.song_name"
]
rule = "llm"
prompt_hint = """
  hook_phrase: most repeated phrase.
  hook_repetition_count: how many times it appears.
  song_name: parsed from filename.
  Note: Whisper accuracy on 1920s recordings with surface noise is significantly
  lower than on modern recordings. Treat transcription with appropriate scepticism.
  Evaluate: does the hook phrase appear to contain or relate to the song title?
  Is it simple and memorable enough to function as a commercial refrain?
  Score 0-10. Flag low-confidence transcription explicitly.
"""
```

---

### `harmony_period` — Period-appropriate harmony

```toml
[[criterion]]
id = "harmony_period"
name = "Period-Appropriate Harmony"
description = """
  Dance music favours major keys and I-IV-V. Jazz introduces ii-V-I and 7ths.
  Minor keys common in blues.
"""
weight = 1.0
db_fields = [
  "tracks.key",
  "tracks.mode",
  "chord_progressions.progression"
]
rule = "llm"
prompt_hint = """
  You are given the key, mode, and chord progressions per section.
  Note: chord detection on mono 1920s recordings is significantly less accurate
  than on modern recordings — treat with appropriate scepticism.
  Evaluate harmonic appropriateness:
  - Major key + I-IV-V: expected for foxtrot, dance band
  - Minor key + I-iv-V: expected for blues, some jazz standards
  - ii-V-I: period-appropriate jazz harmony
  - Tritone substitutions, complex alterations: later jazz influence, flag if present
  Score 0-10.
"""
```

---

### `duration_78rpm` — Track length within 78rpm constraints

```toml
[[criterion]]
id = "duration_78rpm"
name = "Track Length (78rpm Format)"
description = """
  78rpm shellac singles held approximately 2'30\"–3'30\" per side.
  Tracks outside this range were not typical commercial singles.
"""
weight = 0.75
db_field = "tracks.duration"
rule = "range"
threshold_min = 150
threshold_max = 230
unit = "seconds"
fail_message = "Track length is outside the 2'30\"–3'50\" expected range for 78rpm singles."
```

---

### `vocal_intelligibility` — Vocal audible above the band

```toml
[[criterion]]
id = "vocal_intelligibility"
name = "Vocal Intelligibility"
description = """
  Early recording technology made vocal intelligibility a genuine challenge.
  A commercially viable track needed a vocal audible above the band.
"""
weight = 0.75
db_fields = [
  "tracks.unique_word_ratio",
  "tracks.hook_repetition_count"
]
rule = "llm"
prompt_hint = """
  Whisper transcription accuracy on 1920s recordings is lower than modern due
  to surface noise, mono recording, and acoustic horn technology.
  A higher-confidence transcript (more words detected, clearer structure) suggests
  a more intelligible vocal. Flag purely instrumental tracks explicitly.
  Note: this criterion is inherently imprecise on historical recordings.
  Score 0-10. Flag uncertainty explicitly.
"""
```

---

## Cross-mode notes

### Whisper accuracy by mode
- **1988**: good accuracy on professionally recorded pop
- **Contemporary**: best accuracy; modern production, clear vocals
- **1920s_1930s**: significantly lower; surface noise, mono, period pronunciation

### Groove feel expectations by mode
- **1988**: `straight` is the norm (drum machines, click tracks)
- **Contemporary**: `straight` for most; `swung` for neo-soul, jazz-pop
- **1920s_1930s**: `swung` expected; `straight` is the exception

### Chord detection accuracy
All modes: ~70–75% on modern recordings, significantly lower on 1920s material.
All harmony-related prompt hints acknowledge this. Do not treat chord data as
ground truth; treat as a rough guide for LLM evaluation.

### Field naming note
`verse_chorus_delta` (not `verse_chorus_delta_db`) is stored normalised 0.0–1.0.
The `chorus_energy` threshold of 0.15 corresponds to approximately 3dB lift.

### Weights summary

| Criterion | 1988 | Contemporary | 1920s_1930s |
|---|---|---|---|
| Groove | 2.0 | 2.0 | — |
| Chorus / Hook | 2.0 | 2.0 | 1.5 (`hook_refrain`) |
| Structure | 1.5 | 1.5 | 2.0 (`aaba_form`) |
| BPM | 1.5 | 1.0 | 1.5 (`bpm_danceform`) |
| Duration | 1.0 | 1.0 | 0.75 (`duration_78rpm`) |
| Breakdown | 1.0 | n/a | n/a |
| Double chorus | 1.0 | n/a | n/a |
| Lyrics economy | 1.0 | 1.0 | n/a |
| Verse bass riff | 1.0 | 1.0 | n/a |
| Hook timing | n/a | 1.5 | n/a |
| Swing feel | n/a | n/a | 1.5 |
| AABA form | n/a | n/a | 2.0 |
| Intro length | 0.75 | 1.0 | n/a |
| Chorus energy | 0.75 | 0.75 | n/a |
| Keys / harmony | 0.75 | 0.75 | 1.0 (`harmony_period`) |
| Vocal intelligibility | n/a | n/a | 0.75 |
