'''
pdf to audio book
TODO
fix the combining in order part by sorting properly
'''

from models import build_model
import torch
import soundfile as sf
from kokoro import generate
from tqdm import tqdm
import os
from distutils.version import LooseVersion

device = "cpu"
print(f"Runnin on device: {device}")
VOICE_NAME = "am_onyx.pt"
VOICEPACK = torch.load(f"kokoro/voices/{VOICE_NAME}", weights_only=True).to(device)
print(f"Loaded voice: {VOICE_NAME}")
MODEL = build_model("kokoro\kokoro-v0_19.pth", device)
# MODEL = build_model("kokoro-v1_0.pth", device)
SAMPLE_RATE = 24000

def main(filename : str, content : str, part : int):
    # create folder for the book
    if not os.path.exists("output/{}".format(filename)):
        os.mkdir("output/{}".format(filename))
    output_file = "output/{}/{}.wav".format(filename,part)

    audio = []
    # print(f"===> {part}")
    text = f"Page number {part}." + content
    for chunk in text.split("."):
        # print(chunk)
        if len(chunk) < 2:
            # a try except block for non verbalizable text is probably better than this hack
            continue
        try:
            snippet, _ = generate(MODEL, chunk, VOICEPACK, lang=VOICE_NAME[0])
        except :
            pass
        audio.extend(snippet)

    sf.write(output_file, audio, SAMPLE_RATE)



def combine_wav_files(folder_path : str, output_file :str, delete_chunks : bool= False):
    from pydub import AudioSegment
    import glob

    # print(output_file)
    # Get a list of all WAV files in the folder
    os.chdir(folder_path)
    wav_files = glob.glob('*')
    
    # Sort the list of files to ensure they're in the correct order
    # wav_files.sort(key=int)
    sorted(wav_files,  key=LooseVersion)
    print(wav_files)
    
    # Initialize the combined audio
    combined_audio = AudioSegment.empty()
    
    # Iterate over each WAV file and add it to the combined audio with a 1-second gap
    for wav_file in wav_files:
        sound = AudioSegment.from_wav(wav_file)
        combined_audio += sound + AudioSegment.silent(duration=1000)
    
    if delete_chunks:
        [os.remove(i) for i in wav_files]
    
    # Export the combined audio to a new WAV file
    combined_audio.export(output_file, format='wav')



if __name__ == "__main__":
    # importing required modules
    # from pypdf import PdfReader
    # reader = PdfReader('input/Anthem_AynRand.pdf') # where is the file
    name = "Principles_RayDalio" # whats the book name
    folder_path = 'output' # where should the output be

    # # creating a pdf reader object
    # print(len(reader.pages))
    # # extracting text from page
    # start_from = 0 # for ray it was 26
    # for m, page in enumerate(tqdm(reader.pages[start_from:])):
    #     main(name, page.extract_text(), m+1+start_from)
    # # text = (" .... ").join([i.extract_text().replace("/n"," ") for i in reader.pages[1:]])
    # # main(text)

    # # combinign wav into one
    combine_wav_files( "{}\{}".format(folder_path, name), f"{name}.wav", False)
