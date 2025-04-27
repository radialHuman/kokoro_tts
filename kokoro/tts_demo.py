'''
pdf to audio book
'''

from models import build_model
import torch
import soundfile as sf
from kokoro import generate
from tqdm import tqdm

def main(filename : str, TEXT : str, part : int):
    SAMPLE_RATE = 24000

    device = "cpu"

    print(f"Runnin on device: {device}")

    # MODEL = build_model("kokoro-v1_0.pth", device)
    MODEL = build_model("kokoro\kokoro-v0_19.pth", device)
    VOICE_NAME = "am_onyx.pt"
    OUTPUT_FILE = "output/{}_{}.wav".format(part, filename)

    VOICEPACK = torch.load(f"kokoro/voices/{VOICE_NAME}", weights_only=True).to(device)

    print(f"Loaded voice: {VOICE_NAME}")

    audio = []
    print(f"===> {part}")
    for chunk in TEXT.split("."):
        # print(chunk)
        if len(chunk) < 2:
            # a try except block for non verbalizable text is probably better than this hack
            continue
        try:
            snippet, _ = generate(MODEL, chunk, VOICEPACK, lang=VOICE_NAME[0])
        except :
            pass
        audio.extend(snippet)

    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)

def combine_wav_files(folder_path : str, output_file :str):
    import os
    from pydub import AudioSegment
    import glob

    # Get a list of all WAV files in the folder
    wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
    
    # Sort the list of files to ensure they're in the correct order
    wav_files.sort()
    print(wav_files)
    
    # Initialize the combined audio
    combined_audio = AudioSegment.empty()
    
    # Iterate over each WAV file and add it to the combined audio with a 1-second gap
    for wav_file in wav_files:
        sound = AudioSegment.from_wav(wav_file)
        combined_audio += sound + AudioSegment.silent(duration=1000)
    
    # Export the combined audio to a new WAV file
    combined_audio.export(output_file, format='wav')



if __name__ == "__main__":
    # importing required modules
    from pypdf import PdfReader
    # creating a pdf reader object
    reader = PdfReader('input/example.pdf')
    print(len(reader.pages))
    # extracting text from page
    name = "Principles_RayDalio"
    for m, page in enumerate(tqdm(reader.pages[:11])):
        main(name, page.extract_text(), m+1)
    # text = (" .... ").join([i.extract_text().replace("/n"," ") for i in reader.pages[1:]])
    # main(text)
    # combinign wav into one
    folder_path = 'output/'
    combine_wav_files(folder_path, "{}.wav".format(folder_path, name))
