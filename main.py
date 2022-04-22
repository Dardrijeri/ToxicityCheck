from vosk import Model, KaldiRecognizer, SetLogLevel
from tox_block.prediction import make_single_prediction
from timeit import default_timer as timer
from os import listdir, path
from pathlib import Path
import json
import subprocess


def transcript(name):
    text = ""
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                name,
                                '-ar', '16000', '-ac', '1', '-f', 's16le', '-'],
                               stdout=subprocess.PIPE)
    rec = KaldiRecognizer(model, 16000)
    start_time = timer()
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += res['text'] + " "
    res = json.loads(rec.FinalResult())
    text += res['text']
    end_time = timer()
    return text, end_time - start_time


def analyze(text):
    start_time = timer()
    result = make_single_prediction(text, rescale=False)
    end_time = timer()
    return result, end_time - start_time


def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout.splitlines()[0])


if __name__ == "__main__":
    if not path.exists("model"):
        print("Can't locate the model.")
        exit(1)
    SetLogLevel(-1)
    model = Model("model")
    directory = str(Path().absolute())
    folder_name = "prepared-data"
    directory += "\\" + folder_name
    all_files = listdir(directory)
    for file in all_files:
        filepath = directory + "\\" + file
        print("File name:", file)
        duration = get_length(filepath)
        print("File duration:", duration, "s")
        text_buffer, time = transcript(filepath)
        print("Time to transcript:", time, "s")
        text_buffer, time = analyze(text_buffer)
        print("Time to analyze:", time, "s")
        print("Result:", text_buffer)
