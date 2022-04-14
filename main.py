from vosk import Model, KaldiRecognizer, SetLogLevel
from tox_block.prediction import make_single_prediction
from timeit import default_timer as timer
from os import listdir, path
from pathlib import Path
import wave
import json


def transcript(name):
    wf = wave.open(name, "rb")
    text = ""
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        return "Audio file must be WAV format mono PCM."
    rec = KaldiRecognizer(model, wf.getframerate())
    start_time = timer()
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += res['text'] + " "
    res = json.loads(rec.FinalResult())
    text += res['text']
    end_time = timer()
    wf.close()
    return text, end_time - start_time


def analyze(text):
    start_time = timer()
    result = make_single_prediction(text, rescale=False)
    end_time = timer()
    return result, end_time - start_time


def getlen(name):
    f = wave.open(name, "rb")
    frames = f.getnframes()
    rate = f.getframerate()
    dur = frames / float(rate)
    f.close()
    return dur


if __name__ == "__main__":
    if not path.exists("model"):
        print("Can't locate the model.")
        exit(1)
    SetLogLevel(-1)
    model = Model("model")
    dir = str(Path().absolute())
    folder_name = "prepared-data"
    dir += "\\" + folder_name
    all_files = listdir(dir)
    for file in all_files:
        if file[-4:] == ".wav":
            filepath = dir + "\\" + file
            print("File name:", file)
            duration = getlen(filepath)
            print("File duration:", duration, "s")
            text_buffer, time = transcript(filepath)
            print("Time to transcript:", time, "s")
            text_buffer, time = analyze(text_buffer)
            print("Time to analyze:", time, "s")
            print("Result:", text_buffer)
