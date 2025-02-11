import os
import io
import importlib.util
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure random key in production

# Define the folder to store uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_student_function(filepath):
    """
    Load a student's Python module from the given file and return its compute_fft function.
    The file must define a function named compute_fft(signal, fs).
    """
    spec = importlib.util.spec_from_file_location("student_module", filepath)
    if spec is None:
        raise ImportError(f"Could not load module spec from file: {filepath}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing the module: {e}") from e

    if not hasattr(module, "compute_fft"):
        raise AttributeError("The provided script does not define a function named 'compute_fft'.")
    
    return getattr(module, "compute_fft")

def test_student_fft(student_fft_func):
    """
    Test the student's compute_fft function using a composite signal.
    Returns a dictionary with:
      - plot_html: An HTML snippet containing the interactive Plotly figure.
      - peak_frequencies: The list of detected peak frequencies.
      - test_passed: Boolean indicating if expected peaks (50 Hz and 120 Hz) are present.
      - message: A text message summarizing the test result.
    """
    # Generate a composite signal: 50 Hz sine + 0.5 * 120 Hz sine
    fs = 1000  # Sampling frequency in Hz
    t = np.arange(0, 1, 1/fs)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    try:
        freq, magnitude, phase = student_fft_func(signal, fs)
    except Exception as e:
        return {
            "plot_html": None,
            "peak_frequencies": [],
            "test_passed": False,
            "message": f"Error executing compute_fft: {e}"
        }
    
    # Create interactive plots using Plotly
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=("Magnitude Spectrum", "Phase Spectrum")
    )
    
    # Magnitude spectrum plot
    fig.add_trace(
        go.Scatter(x=freq, y=magnitude, mode='lines+markers', name='Magnitude'),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    
    # Phase spectrum plot
    fig.add_trace(
        go.Scatter(x=freq, y=phase, mode='lines+markers', name='Phase'),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (radians)", row=2, col=1)
    
    # Generate the interactive plot HTML snippet
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Automated check: find the 5 highest peaks in magnitude and verify expected peaks are present
    peak_indices = np.argsort(magnitude)[-5:]
    peak_freqs = np.sort(np.abs(np.array(freq)[peak_indices]))
    # Check if peaks near 50 Hz and 120 Hz exist (with a tolerance of ±1 Hz)
    test_passed = any(np.isclose(peak_freqs, 50, atol=1)) and any(np.isclose(peak_freqs, 120, atol=1))
    message = (
        "Test Passed: Expected peaks (50 Hz and 120 Hz) were detected."
        if test_passed else "Test Failed: Expected frequency peaks not detected."
    )
    
    return {
        "plot_html": plot_html,
        "peak_frequencies": peak_freqs.tolist(),
        "test_passed": test_passed,
        "message": message
    }

@app.route('/', methods=['GET'])
def index():
    # Render the upload page
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'studentFile' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('index'))
    
    file = request.files['studentFile']
    if file.filename == '':
        flash("No file selected for uploading.")
        return redirect(url_for('index'))
    
    # Save the uploaded file to the uploads folder
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Load the student's compute_fft function
    try:
        student_fft_func = load_student_function(filepath)
    except Exception as e:
        flash(f"Error loading student's file: {e}")
        return redirect(url_for('index'))
    
    # Run the test on the student's function and get results
    test_results = test_student_fft(student_fft_func)
    return render_template('result.html', results=test_results, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
