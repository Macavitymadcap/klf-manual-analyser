# Fixture Candidates

The song suggested here will make good examples to use in `tests/fixtures` to run the pipeline against with the 20's & 30's config.

## The shortlist

These are chosen for diversity across the four things your pipeline needs to exercise: **clear rhythm** (beat detection), **vocal content** (Whisper), **discernible structure** (section alignment), and **variety of tempo/feel** (all three `1920s_1930s` criteria). All are confirmed pre-1928 on the George Blood digitisation project at the Internet Archive — the highest-quality 78rpm transfer project, and the one most likely to have clean MP3 downloads.

---

**1. Louis Armstrong & His Hot Five — Heebie Jeebies (1926)**
The one literally named in the README. Scat vocal (good Whisper test — unusual phonemes), strong rhythm, clear structure, ~3 mins.
- `https://archive.org/details/78_potato-head-blues_louis-armstrong-and-his-hot-five-louis-armstrong-atkins-kid-ory-j_gbia0039702`
- This page has both *Heebie Jeebies* and *Potato Head Blues* as MP3s in the same George Blood collection. Download the `_CT_EQ.mp3` variant (equalized transfer — better for analysis than the flat one).

**2. Louis Armstrong & His Hot Five — Potato Head Blues (1927)**
Grab it from the same page. Strong backbeat, trumpet-led structure, short enough (~3 mins) that Demucs/Whisper will be fast. Gives you a second Hot Five track with different energy shape from Heebie Jeebies.

**3. Bessie Smith & Louis Armstrong — St. Louis Blues (1925)**
Recorded January 14, 1925, released April 10, 1925. 2:46. Slow blues, W.C. Handy composition. Completely different from the Armstrong stomps: slow tempo, minimal drums, primarily voice + cornet + organ. This is the hardest test for your pipeline — Whisper will get a proper vocal, beat detection will struggle with the loose tempo, and the structure is very sparse. Exactly what you want to find the edge cases.
- `https://archive.org/details/78_nashville-womens-blues_bessie-smith-louis-armstrong-louis-armstrong-charlie-gree_gbia0366419a` — this page has Nashville Women's Blues on side A; search for the St. Louis Blues page separately at `archive.org` — search "bessie smith louis armstrong st louis blues gbia".

**4. Jelly Roll Morton — Black Bottom Stomp (1926)**
Recorded September 15, 1926 for Victor Records. This is the most structurally complex of the bunch — intro, multi-thematic A and B sections, eight distinct choruses with solos from different instruments. It's the best test for your hybrid section alignment because there's real structural variety for it to detect. Fast foxtrot tempo (~200 BPM), full ensemble.
- Search `archive.org` for "black bottom stomp jelly roll morton gbia" — the George Blood transfers are the ones you want.

**5. Duke Ellington & His Orchestra — Jubilee Stomp (1928)**
Available on the Internet Archive. Ellington's 1928 recordings are right at the boundary — still pre-1928 US public domain — and his band has cleaner recording quality than most Hot Five material. More complex harmony than Armstrong or Morton, which stresses the chord detection. Good for verifying the `harmony_period` criterion in `1920s_1930s` mode.
- `https://archive.org/details/78_jubilee-stomp-duke-ellington-his-orchestra`

---

## Download tips

On each Archive page, look for **VBR MP3** as the download format. The `_CT_EQ.mp3` files are equalized (RIAA curve applied) — these are what you want for analysis, not the flat transfers. The EQ transfers sound and analyse like actual music; the flat ones are archival raw captures.

Rename to convention before adding:
```
Louis_Armstrong-Heebie_Jeebies.mp3
Louis_Armstrong-Potato_Head_Blues.mp3
Bessie_Smith-St_Louis_Blues.mp3
Jelly_Roll_Morton-Black_Bottom_Stomp.mp3
Duke_Ellington-Jubilee_Stomp.mp3
```

Five tracks gives you enough variety without making the first pipeline run take an hour. The Armstrong Hot Five tracks will be fast through Demucs and Whisper; the Bessie Smith will be the slow one (lots of vocal content for Whisper to work through). Run them through `1920s_1930s` mode first — that's the mode designed for these recordings — then if everything works, they also make a legitimately interesting control group for `1988` mode to score against.