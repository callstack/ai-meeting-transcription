# AI Meeting Transcription
Repo showcasing AI meeting transcription tool.

## Summary
This repo showcase a basic tool for meeting transcription. It's targetted at meetings conducted in English, but with little tweaking could be used for other languages as well.

### Workflow
The tool works in a three step process:
1. It extract audio path from given video file or YouTube link
2. It generates speaker diarization (separating different speaker tracks) by using [`pyannote/speaker-diarization-3.0`](https://huggingface.co/pyannote/speaker-diarization-3.0) model
3. Finally it generates transcription using [Open AI Whisper model](https://huggingface.co/openai/whisper-base.en). By default it uses Whisper `base.en` version but you can select other model sizes. The output is saved to `output.sub` file in [SubViewer format](https://wiki.videolan.org/SubViewer/).
   

### Local processing
All processing is done locally on the users machine. The model weights are downloaded to local `~/.cache` folder (on macOS).
- Speaker Diarization 3.0 model weights around 6 MB
- Whisper Base.en model weights around 300 MB

## Setup

### Install Dependencies

Install following dependencies (on macOS):

- `ffmpeg` CLI - [`brew install ffmpeg`](https://formulae.brew.sh/formula/ffmpeg)
- Python 3 installation - e.g. [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Homebrew package](https://formulae.brew.sh/formula/python@3.10).
- Python packages - `pip3 install -r requirements.txt`

### Hugging Face token
In order to download models used by these tool you need to:

1. Generate a private Hugging Face auth token - [instructions here](https://huggingface.co/docs/hub/security-tokens)
2. Create `.env` file inside root repo folder with following content:
```env
HUGGINGFACE_AUTH_TOKEN="your token here..."
```
3. Accept `Speaker diarization 3.0` model terms of service - [link here](https://huggingface.co/pyannote/speaker-diarization-3.0)

## Running

### Web UI

In order to run Web UI just run `python3 ./web-ui.sh` in the repo folder. This should open following Web UI interface in the browser.

### Jupyter Notebook

The tool can be used as Jupyter Labs/Notebook as well, you open the  `Transcription.ipynb` in [Jupyter Labs](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#conda).

### Notes

Speaker diarization steps is the longest part of moder execution. It roughly takes 30s for each 1 minute of the meeting to execute on M1 MacBook Pro. 

## Troubleshooting

1. If you get following error `"Could not download 'pyannote/segmentation-3.0' model. It might be because the model is private or gated so make sure to authenticate."` then make sure you provided [Hugging Face auth](#hugging-face-token) token AND accepted `Speaker diarization 3.0` model [terms of service](https://huggingface.co/pyannote/speaker-diarization-3.0).
