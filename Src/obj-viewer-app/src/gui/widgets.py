from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
import plotly.graph_objects as go
import numpy as np

class FileSelectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        
        self.label = QLabel("Select an OBJ file:")
        self.layout.addWidget(self.label)
        
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.button)
        
        self.setLayout(self.layout)
        
    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select OBJ File", "", "OBJ Files (*.obj);;All Files (*)", options=options)
        if file_name:
            self.label.setText(file_name)
            self.parent().load_obj_file(file_name)

class VisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def visualize(self, vertices):
        if len(vertices) > 0:
            pts = np.array(vertices)
            x, y, z = pts.T
            fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
            fig.show(renderer="browser")
        else:
            self.layout.addWidget(QLabel("No vertices found in the OBJ file."))