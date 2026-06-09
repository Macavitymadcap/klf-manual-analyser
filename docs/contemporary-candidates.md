# Contemporary Mode Candidates

The contemporary mode criteria differ from 1988 in specific measurable ways — hook within 30 seconds, intro ceiling of 15s instead of 8–30s range, BPM genre-contextual rather than hard-capped at 135, no breakdown/double chorus requirement — so the shortlist should be designed to stress those specific differences as much as match the general spirit of the mode.

Here's how I'd structure it:

---

## The contemporary shortlist

Same logic as the 1988 set: a tier of obvious passes, a tier of interesting edge cases, and a tier of deliberate fails.

---

**Tier 1 — should score high, and if they don't something is wrong**

**Sabrina Carpenter — Espresso** (2024, 2:55, 104 BPM, C major) — the canonical contemporary pop single. The best-selling song of 2024 in the UK, and Carpenter had a record-breaking 21 weeks in the chart with her three singles. Under 3 minutes, immediate hook, no intro to speak of, highly repetitive lyric. If anything scores high on contemporary mode this should be it.

**Chappell Roan — Good Luck, Babe!** (2024, 3:38, ~117 BPM, D major) — 80s new wave structure with a very clear chorus hook arriving early, dense repetition, danceable. Written with Justin Tranter and Dan Nigro. Interesting because it's a deliberately retro sound applied to a streaming-era structure — tests whether the mode rewards the structural pattern regardless of sonic palette.

**Sabrina Carpenter — Please Please Please** (2024) — her second big hit from the same period. Worth running both Carpenter singles to see if they score similarly, since they're from the same album and written to similar templates.

---

**Tier 2 — interesting edge cases**

**Billie Eilish — Birds of a Feather** (2024, 3:30, ~105 BPM) — baroque pop / indie pop / synth-pop, produced by Finneas. Minimalist production, slow build, late hook arrival, very low BPM for a chart-topper. The Manual's contemporary adaptation is genre-contextual on BPM rather than hard-capped, so this is a genuine test: does the LLM scoring correctly treat 105 BPM as appropriate for this style, or does it penalise it?

**Lola Young — Messy** (2024, 3:20 radio edit, indie pop / pop soul) — eleven-week climb to #1 in January 2025, the UK's biggest song of 2025 so far. Driven by emotional directness and lyric repetition rather than a dance groove — more in the Adele tradition than the Sabrina Carpenter one. Structurally conventional but sonically very different from the SAW-adjacent tracks. Tests whether the tool correctly scores based on structural criteria rather than sonic texture.

**Hozier — Too Sweet** (2024, 3:19 radio edit, retro soul / funk rock) — a UK #1 but very far from the streaming-pop template. Slow groove, organic instrumentation, 4:11 full version. Good contrast candidate: does the tool appropriately score this lower on `chorus_energy` and `hook_timing` because the song is genuinely structured differently, rather than just because it sounds different?

---

**Tier 3 — principled fails**

**Gracie Abrams — That's So True** (2024, 2:46, folk-pop) — eight non-consecutive weeks at UK #1. Produced by Aaron Dessner. Acoustic, minimal percussion, no real groove, barely 160 BPM equivalent strumming pattern. Should score low on most groove/danceability criteria. The interesting question is whether it scores *something* on structure and hook criteria despite failing the groove tests — which would be the correct answer, since it clearly does have a hook and a structure.

**Taylor Swift — Fortnight ft. Post Malone** (2024, #1 in the UK) — long by contemporary standards, deliberately anti-hook in its verse structure, Tortured Poets-era production. The hook-within-30-seconds criterion should be a notable miss here.

---

## The comparison you're really running

The parallel to the 1988 set is deliberate. In 1988 you had:
- SAW productions (designed to a formula) → should score high  
- Fairground Attraction / New Order → principled fails  

In the contemporary set you have:
- Carpenter / Roan → designed to streaming-era conventions → should score high  
- Eilish / Young → chart-toppers but operating on different structural logic → interesting middle scores  
- Abrams / Swift → very different structural philosophy → should score lower on groove/danceability criteria specifically  

If the scores roughly track that gradient, the contemporary mode criteria are doing something real. If Espresso and Birds of a Feather score identically, or if Messy scores higher than Good Luck Babe, you've found something worth calibrating.

One extra thought: running **Doctorin' the Tardis itself through contemporary mode** would be a pleasing cross-check. The Manual's claim is the rules are timeless — does the tool agree, or does Tardis score significantly lower on contemporary mode because its intro is too long and its hook timing is off by modern standards?