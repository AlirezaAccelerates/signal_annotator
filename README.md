# WAV signal quality annotator

A local browser interface using Python, NumPy, and SciPy. No Tkinter, Flask, Streamlit, or plotting library required. Python's built-in HTTP server serves the interface on 127.0.0.1; WAV files and labels stay on your machine. A small embedded JavaScript canvas draws the waveform.

## Run

```sh
python signal_annotator.py
```

Your default browser opens automatically. Enter full directory or WAV file paths, one per line, enter an output directory, set the window and step sizes in seconds, and click **Load / apply settings**. Paths refer to the computer running Python; no upload is needed. Subdirectories are searched recursively; WAV files are sorted by full path.

Or provide paths directly:

```sh
python signal_annotator.py /data/session1 /data/session2 --output /data/labels --window 5 --step 2.5
```

Keep the terminal open while annotating. Press Ctrl+C to stop the server. If the browser does not open, copy the printed local URL into your browser. Use `--no-browser` to skip automatic opening or `--port 8765` to choose a port. Internet access is not required after installing NumPy and SciPy. No Tcl/Tk installation is needed.

## Annotate

- Select **Good (0)** or **Poor (1)** for each window. Keyboard shortcuts: `0`, `1`, left arrow, right arrow.
- Select a label before moving either forward or backward. Revisiting a labeled window restores its selection; click another label to correct it.
- At file boundaries, navigation continues to the next or previous file. Each WAV is loaded with `sr, signal = wavfile.read(full_path)` and displayed in time, with a separate lane for each channel.
- Every label is saved immediately. Closing the browser is safe after a label finishes saving; reopening the same paths, output folder, and settings resumes at the first unlabeled window.
- Window and step values are rounded to sample boundaries. Step must be no larger than window size, so no audio is skipped. A window starts at each step while its start is inside the file; final windows can be shorter. Overlapping windows are labeled independently.
- Changing settings and applying them starts or resumes a separate annotation set. Previous annotation sets are retained.

## Output

Each completed WAV produces a UTF-8 text file containing only comma-separated labels, in window order, followed by a newline:

```text
0,1,1,1
```

Filenames have the form `recording.<id>.txt`. The identifier incorporates the full source path, size, modification time, and settings, preventing collisions between files with the same name or different annotation settings.

A companion `recording.<id>.progress.json` saves metadata, current position, and partial labels (`null` for unlabeled windows). The final `.txt` is written only when every window is labeled and updated whenever a completed annotation is corrected. The JSON metadata records sample rate, window samples, and step samples: label `i` covers samples `[i * step_samples, min(i * step_samples + window_samples, frames))`.

The plot uses a fixed per-channel amplitude scale across the entire file, centered at zero (unsigned 8-bit PCM is centered at 128). Dense waveforms use minimum/maximum envelopes per display column to retain spikes. One WAV is held in memory at a time; very large files may take time to load. Use one app instance and one browser tab per output annotation set. The server is intended for local use, not public hosting.
