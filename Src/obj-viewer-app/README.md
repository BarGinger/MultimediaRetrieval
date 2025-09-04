# OBJ Viewer Application

## Overview
The OBJ Viewer Application is a simple GUI-based tool that allows users to select and visualize 3D models stored in OBJ file format. The application provides an intuitive interface for loading OBJ files and rendering their vertices in a 3D space.

## Features
- Select an OBJ file from your filesystem.
- Visualize the 3D model in a dedicated area of the application.
- User-friendly interface built with PyQt5.

## Project Structure
```
obj-viewer-app
├── src
│   ├── main.py               # Entry point of the application
│   ├── gui                   # GUI components
│   │   ├── __init__.py       # GUI package initializer
│   │   ├── main_window.py     # Main window layout and functionality
│   │   └── widgets.py         # Custom widgets for file selection and visualization
│   ├── utils                 # Utility functions
│   │   ├── __init__.py       # Utils package initializer
│   │   └── obj_parser.py      # Function to parse OBJ files
│   └── assets                # Application assets
│       └── styles.qss        # Stylesheet for UI components
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd obj-viewer-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
python src/main.py
```

Once the application is running, you can select an OBJ file using the file selection area on the left. The 3D visualization will update automatically to display the selected model.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.