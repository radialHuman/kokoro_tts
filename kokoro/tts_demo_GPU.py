'''
pdf to audio book
TODO
GPU
add torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
 in requirements
'''

from models import build_model
import torch
import soundfile as sf
from kokoro import generate

def main(TEXT : str):
    SAMPLE_RATE = 24000

    device = "cuda" #if torch.cuda.is_available() else "cpu" 

    print(f"Runnin on device: {device}")

    # MODEL = build_model("kokoro-v1_0.pth", device)
    MODEL = build_model("kokoro\kokoro-v0_19.pth", device)
    VOICE_NAME = "am_onyx.pt"
    OUTPUT_FILE = "output/1_{}.wav".format(VOICE_NAME.split(".")[0])

    VOICEPACK = torch.load(f"kokoro/voices/{VOICE_NAME}", weights_only=True).to(device)

    print(f"Loaded voice: {VOICE_NAME}")

    audio = []
    for chunk in TEXT.split("."):
        print(chunk)
        if len(chunk) < 2:
            # a try except block for non verbalizable text is probably better than this hack
            continue
        try:
            snippet, _ = generate(MODEL, chunk, VOICEPACK, lang=VOICE_NAME[0])
        except :
            pass
        audio.extend(snippet)

    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)

if __name__ == "__main__":
    # importing required modules
    from pypdf import PdfReader
    # creating a pdf reader object
    reader = PdfReader('input/example.pdf')
    print(len(reader.pages))
    # extracting text from page

    text = (" .... ").join([i.extract_text().replace("/n"," ") for i in reader.pages[1:5]])
    main(text)