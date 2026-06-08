# Test Fixture Sources

Public domain audio recordings used for unit testing the analysis modules.

All recordings confirmed to be in the public domain in both the United States
(Music Modernization Act — recordings published before 1928) and the United
Kingdom (70-year recording copyright from publication date).

---

## Source criteria

- Published before 1 January 1928 (US public domain threshold)
- Vocal content present (for Whisper testing)
- Clear rhythmic structure (for beat detection testing)
- MP3 format, reasonable audio quality for the era

---

## Recordings

<!-- Add entries as fixtures are sourced. Format:
     ### Artist — Song Title (Year)
     **Source**: URL
     **Archive ID**: identifier
     **Duration**: mm:ss
     **Notes**: anything relevant (e.g. instrumental, poor audio, unusual tempo)
-->

*No fixtures added yet. See README for sourcing instructions.*

---

## Sourcing instructions

Suitable recordings can be found at:

- **Internet Archive 78rpm collections**:
  https://archive.org/details/78rpm
  Search for pre-1928 dance band, jazz, or popular vocal recordings.
  Download MP3 format. Confirm publication date before adding.

- **Open Music Archive**:
  https://openmusicarchive.org
  Explicitly tagged public domain. Good selection of 1920s blues and jazz.

### Naming convention

Downloaded files must be renamed to match the project convention before adding:

```
Artist_Name-Song_Title.mp3
```

Example: `Louis_Armstrong-Heebie_Jeebies.mp3`

### Adding a fixture

1. Download the MP3 and rename to convention
2. Place in `tests/fixtures/`
3. Add an entry to this file with full attribution and source URL
4. Confirm the file is excluded from git LFS if audio files are tracked separately