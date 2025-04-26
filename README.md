
# Forced Alignment

**Audio-Text Alignment using NVIDIA NeMo**

This project provides scripts to align a given text and audio file, producing **timestamped word alignments**.  
It uses NVIDIA's [NeMo Forced Aligner](https://github.com/NVIDIA/NeMo) with the **FastConformer model** for high-quality, fast forced alignment.

---


**Forced Alignment** is the process of taking an **audio recording** and a **matching transcript** and finding **where each word is spoken** in the audio.

It is used for

- **Auto-generate subtitles** (SRT, VTT files)
- **Karaoke-style lyric timing**
- **Speech analytics** 
- **Dubbing and translation**
- **Training data** for speech recognition models
- **Language learning apps**

---

### How It Works 

1. You provide:
   - A text transcript (what is spoken)
   - An audio file (the recording)

2. The Forced Aligner:
   - Breaks the audio into tiny frames
   - Predicts when each word starts and ends
   - Produces a file with:
     - Word
     - Start time
     - Duration

---

## Files Included

| File | Purpose |
|:------------|:--------|
| `nemo_main_opt.py` |  Optimized main forced aligner script (fast, multithreaded CTM parsing) |
| `nemo_main.py` | Clean version of the forced aligner |
| `print_words_timestamp.py` | Script to read `.ctm` output and print words with timestamps |
| `manifest_creation.py` | Helper script to manually create a manifest JSON |
| `force_align.py` | Simple hardcoded example to align a fixed story and audio |
| `main.py` | Early version using `os.system` instead of `subprocess` |

---

## Set Up and Run

### 1. Create a Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

---

### 2. Install Required Libraries

```bash
pip install nemo-toolkit[asr]
pip install nemo_text_processing
pip install git+https://github.com/NVIDIA/NeMo.git@main --no-build-isolation
pip install numpy==1.26.0 --only-binary=:all:
pip install ulid-py
```


---

### 3. Align Your Text and Audio

```bash
python nemo_main_opt.py --text "Your full text transcript here" --audiopath "path/to/your/audio.wav"
```

---

### 4. Print Word Timestamps (Optional)

```bash
python print_words_timestamp.py
```

---


