from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
import plotly.graph_objects as go
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OBJ Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.file_label = QLabel("No file selected")
        self.layout.addWidget(self.file_label)

        self.load_button = QPushButton("Load OBJ File")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button)

        self.visualization_label = QLabel("Visualization will appear here")
        self.layout.addWidget(self.visualization_label)

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select OBJ File", "", "OBJ Files (*.obj);;All Files (*)", options=options)
        if file_path:
            self.file_label.setText(file_path)
            self.visualize_obj(file_path)

    def visualize_obj(self, filepath):
        vertices = self.parse_obj_file(filepath)
        pts = np.array(vertices)

        if len(pts) > 0:
            x, y, z = pts.T
            fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
            fig.show(renderer="browser")
        else:
            self.visualization_label.setText("No vertices found in the OBJ file")

    def parse_obj_file(self, filepath):
        vertices = []
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        vertices.append([x, y, z])
        return vertices