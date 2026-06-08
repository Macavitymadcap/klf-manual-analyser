# Criteria Reference (compact)

## Rule types

| Rule | Key(s) | Behaviour |
|---|---|---|
| lte | db_field, threshold | pass if field ≤ threshold |
| gte | db_field, threshold | pass if field ≥ threshold |
| range | db_field, threshold_min, threshold_max | pass if min ≤ field ≤ max |
| exists | db_field, value | pass if any sections row has label == value for this track |
| llm | db_field OR db_fields, prompt_hint | send to Ollama; expect {"score":0-10,"reasoning":"..."} |

`db_field` (string) vs `db_fields` (array) are mutually exclusive.

## 1988 mode — criteria summary

| id | rule | key threshold | weight |
|---|---|---|---|
| bpm | lte | 135 BPM | 1.5 |
| duration | lte | 210 seconds | 1.0 |
| groove | llm | — | 2.0 |
| structure | llm | — | 1.5 |
| breakdown | exists | "breakdown" | 1.0 |
| double_chorus | exists | "double_chorus" | 1.0 |
| chorus_hook | llm | — | 2.0 |
| lyrics_economy | llm | — | 1.0 |
| verse_bass_riff | llm | — | 1.0 |
| intro_length | range | 8–30 seconds | 0.75 |
| chorus_energy | gte | 0.15 (≈3dB) | 0.75 |
| keys_harmony | llm | — | 0.75 |

## Contemporary mode — differences from 1988

| id | change |
|---|---|
| bpm | lte→llm (genre-contextual); weight 1.5→1.0 |
| intro_length | range 8-30→lte 15; weight 0.75→1.0 |
| structure | prompt updated; no breakdown requirement |
| breakdown | REMOVED |
| double_chorus | REMOVED |
| hook_timing | NEW; lte 30 seconds; weight 1.5 |

## 1920s_1930s mode — criteria

| id | rule | weight |
|---|---|---|
| bpm_danceform | llm | 1.5 |
| swing_feel | llm | 1.5 |
| aaba_form | llm | 2.0 |
| hook_refrain | llm | 1.5 |
| harmony_period | llm | 1.0 |
| duration_78rpm | range 150–230s | 0.75 |
| vocal_intelligibility | llm | 0.75 |

## LLM prompt construction (scoring/prompt.py)

System prompt must include:
- Mode era context (1988 UK pop / streaming era / 1920s dance halls)
- Instruction to respond in JSON only: {"score": int 0-10, "reasoning": str}

User prompt must include:
- Criterion name and description
- Actual field values fetched from SQLite
- The criterion's prompt_hint verbatim
- For 1920s_1930s: explicit caveat about chord/Whisper accuracy limitations

## Scoring formula

```python
overall = sum(score_i * weight_i) / sum(weight_i)
passed = score >= 0.5  # i.e. LLM score >= 5/10
```
