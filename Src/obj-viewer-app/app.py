from viewer.init import create_dash_app

def main():
    app = create_dash_app()
    print("Starting 3D Shape Viewer...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    main()
