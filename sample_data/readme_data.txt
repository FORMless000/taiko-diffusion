sample_data/readme_data.txt

How to obtain the data
1. Download the shared dataset from the Dropbox link below:
   https://www.dropbox.com/scl/fo/ipbgqy80vnithcbhjp33f/ALzhBrzJm013t9gtPpoa9VU?rlkey=kqo1ia6iq6zo1ep4j9usybg1u&st=unov6n5l&dl=0

2. The downloaded beatmap archives should be kept as .osz files.

Where to place the data
1. Put the downloaded .osz files inside:
   sample_data/raw/

2. Do not manually place files into sample_data/unpacked/ unless you already know you want pre-unpacked sets there.
   That folder is normally created or refreshed by the repo's unpacking/preprocessing scripts.

3. Other folders under sample_data/ are generated artifacts:
   - sample_data/unpacked/           unpacked beatmap folders
   - sample_data/inference_work/     temporary inference workspace
   - sample_data/inference_outputs/  generated outputs

Typical usage
1. Download .osz files from the Dropbox link.
2. Copy them into sample_data/raw/
3. Run the repo preprocessing or inference workflow.

Important note
- This project mainly expects osu!taiko beatmap sets.
- Many workflows in this repo assume constant-BPM taiko charts.
