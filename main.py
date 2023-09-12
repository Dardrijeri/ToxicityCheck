import sys
import ctypes
import subprocess
import json
from vosk import Model, KaldiRecognizer, SetLogLevel
from tox_block.prediction import make_single_prediction
from recasepunc import CasePuncPredictor, WordpieceTokenizer, Config
from timeit import default_timer as timer
from os import path
from PySide6.QtWidgets import (QTextEdit, QPushButton, QApplication, QWidget,
                               QFileDialog, QGroupBox, QGridLayout, QLabel)
from PySide6.QtGui import QIcon, QPainter
from PySide6.QtCore import Qt
from PySide6.QtCharts import (QBarSet, QChart, QBarSeries, QValueAxis, QBarCategoryAxis, QChartView)


class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Load vosk model
        self.model = Model("model")

        # Customizing window
        self.setWindowTitle("ToxAnalyzer")
        self.setWindowIcon(QIcon("toxic-symbol.png"))

        # Create widgets
        self.select_button = QPushButton("Select file")
        self.start_button = QPushButton("Process file")
        self.text_display = QTextEdit()
        self.horizontal_group = QGroupBox()
        self.grid_group = QGroupBox()
        self.status_text = QLabel()
        self.chart = QChart()

        # Init variables
        self.file_name = None
        self.text = None
        self.time_transcript = None
        self.time_analyze = None

        # Set starting status
        self.change_status("Select file for processing")

        # Customize text display
        self.text_display.setReadOnly(True)

        # Creating chart
        self.create_chart()
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        # Create layout and add widgets
        main_layout = QGridLayout()
        main_layout.addWidget(self.select_button, 0, 0)
        main_layout.addWidget(self.start_button, 0, 1)
        main_layout.addWidget(self.chart_view, 2, 0)
        main_layout.addWidget(self.text_display, 2, 1)
        main_layout.addWidget(self.status_text, 3, 0, 1, 2)

        # Set dialog layout
        self.setLayout(main_layout)

        # Add buttons signal to slots
        self.select_button.clicked.connect(self.select_file)
        self.start_button.clicked.connect(self.process_file)

    def select_file(self):
        filedialog = QFileDialog()
        filedialog.setFileMode(QFileDialog.ExistingFile)
        filedialog.setViewMode(QFileDialog.Detail)
        if filedialog.exec():
            self.file_name = filedialog.selectedFiles()
            self.change_status("File successfully selected")

    def process_file(self):
        self.change_status("Processing...")
        self.transcript()
        self.restore_punc()
        self.analyze()

    def change_status(self, text_buffer):
        self.status_text.setText(text_buffer)

    def create_chart(self):
        categories = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']
        self.chart.setTitle("Results")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.axis_x = QBarCategoryAxis()
        self.axis_x.append(categories)
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 1)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.chart.legend().setVisible(False)

    def transcript(self):
        text = ""
        process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                    self.file_name[0],
                                    '-ar', '16000', '-ac', '1', '-f', 's16le', '-'],
                                   stdout=subprocess.PIPE)
        rec = KaldiRecognizer(self.model, 16000)
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
        self.text = text
        self.time_transcript = end_time - start_time
        self.text_display.setText(self.text)

    def analyze(self):
        start_time = timer()
        result = make_single_prediction(self.text, rescale=False)
        end_time = timer()
        self.time_analyze = end_time - start_time
        len_result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                     'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', self.file_name[0]],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        duration = float(len_result.stdout.splitlines()[0])
        self.chart.removeAllSeries()
        set_0 = QBarSet("value")
        result_float = []
        #print(result)
        for i in result:
            if i != "text":
                result_float.append(result[i])
        set_0.append(result_float)
        series = QBarSeries()
        series.append(set_0)
        self.chart.addSeries(series)
        series.attachAxis(self.axis_y)
        self.change_status("Done, file duration: {}s time to transcript: {}s analyze: {}s".format(duration,
                                                                                                  self.time_transcript,
                                                                                                  self.time_analyze))

    def restore_punc(self):
        predictor = CasePuncPredictor('checkpoint', lang="en")

        text = self.text
        tokens = list(enumerate(predictor.tokenize(text)))

        results = ""
        for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
            prediction = predictor.map_punc_label(predictor.map_case_label(token[1], case_label), punc_label)

            if token[1][0] == '\'' or (len(results) > 0 and results[-1] == '\''):
                results = results + prediction
            elif token[1][0] != '#':
                results = results + ' ' + prediction
            else:
                results = results + prediction
        self.text = results.strip()
        self.text_display.setText(self.text)


if __name__ == '__main__':
    if not path.exists("model"):
        print("Can't locate the model.")
        exit(1)
    SetLogLevel(-1)
    app_id = u'ToxAnalyzer0.1'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the form
    MainWindow = Window()
    available_geometry = MainWindow.screen().availableGeometry()
    size = available_geometry.height() * 3/4
    MainWindow.resize(size*2, size)
    MainWindow.show()

    # Run the main Qt loop
    sys.exit(app.exec())
