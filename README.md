# FaceX

FaceX is a local webcam face recognition project. It detects faces and five facial landmarks with YuNet, aligns each face to 112 x 112 pixels, extracts a 512-dimensional ArcFace embedding, and compares it with enrolled people. It runs on the CPU and displays names or `Unknown` in an OpenCV window.

## Quick start (Windows PowerShell)

Install 64-bit Python 3.11 or 3.12. Run these commands from the FaceX folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe init_project.py --download-models
.\.venv\Scripts\python.exe -m src.camera
.\.venv\Scripts\python.exe -m src.enroll --name "Your Name"
.\.venv\Scripts\python.exe -m src.recognize
```

Close each camera window with **Q** before starting the next command. Explicit interpreter paths avoid PowerShell activation-policy issues. The original `venv` in this checkout points to a missing Python installation; create a fresh `.venv` instead of reusing it.

A project-local Python runtime may also be present from development. If `.tools/python/python.exe` exists, use it directly in place of `.venv/Scripts/python.exe`; it does not require activation. `.tools` is ignored by Git and is not part of the portable project.

On Linux/macOS, create the environment with `python3 -m venv .venv`, activate with `source .venv/bin/activate`, then use `python` for the commands below. A graphical desktop and camera permissions are required. Install the `opencv-contrib-python` version from requirements.txt, not a headless variant.

## Model setup

```sh
python init_project.py --download-models
```

This downloads YuNet (232,589 bytes) and ArcFace ResNet100 (261,036,388 bytes), verifies their SHA-256 checksums, and installs them under `models/`. Internet is needed only for installing dependencies and downloading models. Existing verified models are reused; different nonempty models are preserved with an error. Empty placeholders are replaced. Without `--download-models`, setup only creates directories.

| File | Source |
| --- | --- |
| `models/detector_yunet.onnx` | [OpenCV Zoo YuNet 2023mar](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `models/embedder_arcface.onnx` | [ONNX Model Zoo ArcFace ResNet100](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface) |

Default preprocessing is **RGB float32, NCHW, 0-255** (`--preprocessing raw`) for this Model Zoo export. For a compatible external ArcFace model that expects `(RGB - 127.5) / 127.5`, explicitly pass `--preprocessing normalized` to enrollment, evaluation, and recognition. This includes the book's `w600k_r50.onnx` setup. Do not infer preprocessing from the `.onnx` filename. The model must accept `[1,3,112,112]` float32 input and emit 512 features. See the upstream model documentation and licenses before redistributing weights.

## Enroll a person

```sh
python -m src.enroll --name "Belise" --samples 8
python -m src.enroll --name "Isaac" --camera 1
```

1. Keep exactly one person in view, with their whole face visible and good front lighting.
2. Wait for **Ready**, then press **Space** to capture each sample. Change expression and head angle slightly between captures.
3. After all samples are captured, enrollment saves automatically and closes the camera.

The default is eight captures, at least half a second apart. Faces must be at least 90 pixels across and pass a blur check. At least three samples are required. **Q**, **Escape**, or closing the main window cancels without saving partial enrollment. If needed, adjust `--min-face-size` or `--min-sharpness`; lowering them can admit poorer samples.

To deliberately redo an existing identity:

```sh
python -m src.enroll --name "Belise" --replace
```

Names are case-sensitive; `Unknown` and `Too small` are reserved. Embeddings are stored in `data/db/face_db.json`; raw photos and camera recordings are not saved. Each identity retains its normalized samples; matching uses their normalized mean. Writes replace the JSON file atomically. Run one enrollment process at a time. Restart recognition after adding or replacing identities.

The database records the model hash, preprocessing, and alignment version. A mismatch requires re-enrollment into a new database, preventing comparisons between incompatible embeddings. Use `--db data/db/another.json` consistently to keep separate galleries. Back up the database before manually editing or removing identities.

## Live recognition

```sh
python -m src.recognize --camera 0 --threshold 0.45 --margin 0.05
```

The window displays face boxes, five landmark dots, names, cosine similarity, and processing FPS. Press **Q** or **Escape** to quit. Multiple faces can be processed; CPU speed decreases with each additional face. Faces below 70 pixels across are marked `Too small`.

The best match must meet `--threshold` and exceed the runner-up by `--margin`; otherwise the result is `Unknown`. Higher thresholds accept fewer matches. Scores are similarities, not confidence percentages. The defaults are starting values, not calibrated accuracy guarantees. This implementation processes current frames without temporal identity tracking, so labels may flicker under poor lighting or motion.

## Lock and track one enrolled person

```sh
python -m src.face_tracking --target "Isaac"
```

Use the exact enrolled name. This reuses the same YuNet detection, five-point
alignment, ArcFace model, JSON database, and similarity threshold/margin as
`src.recognize`. Unknown faces and other enrolled names cannot acquire the lock.
Only the target gets a box; the window shows `SEARCHING`, `LOCKED`, `UNCERTAIN`, or `LOST`,
plus the smoothed face position and normalized horizontal/vertical errors.
There is no current box or position output while the target is missing.
After more than 24 missed frames, it clears the old geometry and searches for
the same identity again. Q, Escape, or closing the window exits.

Identity is checked every frame by default. `--verify-every 10` uses the PDF's
lighter periodic verification, but can briefly follow an unverified face between
checks. Reappearance after a gap and scenes with multiple eligible faces always
trigger verification. `--lost-timeout`, `--ema-alpha`, and `--dead-zone` control
the grace period, position smoothing, and centered region. Existing `--camera`,
`--db`, `--model`, `--detector`, `--preprocessing`, `--threshold`, and `--margin`
options are supported. A separate status panel displays position, SMILE/NEUTRAL,
eye state, session blink count, EAR, smile score, and all tracking/expression
calibration settings. Missing observations show N/A rather than stale readings.
Expression analysis runs only on the locked face, before any overlays are drawn.

Brief identity failures on a continuously detected face now retain an orange
`UNCERTAIN` box for up to five processed frames (`--uncertain-grace 5`). Each
new box must overlap the previous box by at least 50% IoU. Identity is checked
every frame during uncertainty; a successful target match restores `LOCKED`.
A different recognized identity, multiple eligible faces, invalid face alignment,
or an exhausted grace period drops the lock immediately. Missing detections
still hide the box and position; reacquisition after a gap requires a target match.
Use `--uncertain-grace 0` for the previous strict behavior. Grace can briefly
follow an unknown person occupying the same location; it is continuity evidence,
not identity confirmation. It cannot recover details absent in a dark camera image.

The panel shows the tracking reason, current detection confidence, and best
database match similarity (not necessarily the target's score). `Unknown` means
the match failed the threshold or margin. `N/A` means no valid score was obtained
on that frame, including frames between scheduled identity checks. Detection
confidence is shown only for eligible detections; no eligible face can also mean
the face is smaller than the 70-pixel minimum. These readings help distinguish
detection failures from identity matching failures when lighting changes.

## Whole-person tracking and face association

The new `src.whole_person_tracking` module keeps identity and location as
separate signals. OpenCV's built-in HOG people detector runs on the CPU, while
YuNet supplies faces and ArcFace verifies the enrolled target. A face is
associated with the body whose box contains it; body motion alone never proves
identity.

The tracker exposes three honest states:

* `Confirmed`: the target face matched the enrolled identity this frame.
* `Temporary`: a previously confirmed face/body lock remains on the tracked
  body while the face is absent, turned, occluded, blurred, or not confidently
  recognized. Confidence is multiplied by `exp(-decay_rate)` each frame, so
  this state cannot persist indefinitely.
* `Unknown`: no target face/body lock has been established, or the continuity
  window/confidence floor expired. A body box may still be shown as `Unknown`;
  that is a location observation without an identity claim.

When the face returns and ArcFace matches the enrolled target on the associated
body, the state returns to `Confirmed` and confidence is refreshed from that
match. A face on the locked body that cannot be matched keeps the lock
`Temporary` until it is verified or the continuity window expires. To reduce
state flicker from a poor embedding, three consecutive visible-face
identity misses are tolerated by default (`--face-mismatch-grace 3`). An absent
face still changes to `Temporary` immediately. The model files are
`models/detector_yunet.onnx` (YuNet face detection
and landmarks) and `models/embedder_arcface.onnx` (ArcFace ResNet100 identity
embeddings); both are run on CPU through ONNX Runtime.

Body detection starts independently of face confirmation. A person can enter
side-on or with their face outside the frame and still appear as `Unknown`;
when a matching face is later observed on that body, the state becomes
`Confirmed`.

Run the model-free evaluation (no webcam, model weights, or ONNX Runtime are
needed):

```powershell
py -3.12 -m src.whole_person_tracking --evaluate
```

It exercises head turns, sustained occlusion, path crossing, and a body-detector
dropout and prints the state sequence and pass/fail result for each scenario.
For a live CPU-only
webcam run, enroll a person first and then use:

```powershell
py -3.12 -m src.whole_person_tracking --target "Isaac" --camera 0
```

The window shows the body box, state, decayed confidence, and `LEFT`, `RIGHT`,
`UP`, `DOWN`, or `CENTER` centring signal. Tune `--decay-rate`,
`--temporary-floor`, and `--max-temporary-frames` to match the camera's frame
rate and the amount of occlusion you want to tolerate. After a confirmed body
box is seen, an OpenCV CSRT tracker continues predicting that body when HOG
temporarily returns no box; the prediction is only used for `Temporary`
continuity and is reset as soon as the state becomes `Unknown`. Live YuNet uses a
moderate `--face-confidence 0.60` cutoff by default; raise it if false face
detections appear. The HOG detector is a portable baseline. If HOG misses a
close-up or seated subject, the live path
adds a conservative face-guided upper-body prior and shows `faces=...`,
`bodies=...`, `hog=...`, `pose=...`, and `motion=...` diagnostics at the bottom
of the window. When the pinned MediaPipe 0.10 runtime is available, its CPU
pose detector supplies an independent body box even when no face is visible.
The status text also shows the current reason and ArcFace match score, which
distinguishes `faces=0` from a detected face that failed the identity threshold.
The tracker accepts any detector exposing `detect(frame)` that returns
`PersonDetection` records when a stronger CPU detector is needed.

### Tracking logs

Every tracking run automatically creates a unique CSV file under `data/logs/`
and prints its path. Use `--log-dir path/to/logs` to choose another directory.
Files have exactly three columns: `Timestamp`, `Action Type`, and `Description`.
Timestamps use UTC with an explicit timezone and millisecond precision.

Actions are `SESSION_START`, `TRACKING_SEARCHING`, `TRACKING_LOCKED`,
`TRACKING_UNCERTAIN`, `TRACKING_LOST`, `ERROR`, and `SESSION_END`. One tracking
row is saved per processed frame. Its JSON description contains the frame
number, target, tracking reason, detection confidence, match name and score,
position/direction, smile status and score, eye state, EAR, blink event, and
session blink count. Unavailable readings are `null`. The start record contains
the command settings; the end record distinguishes normal exit, interruption,
and error. Startup/runtime exceptions are recorded before being propagated.

Rows are flushed as they are written, and previous sessions are never overwritten.
Logs accumulate until manually archived or removed; recording every frame can
produce large files on long runs. Forced process termination may omit the final
session record. A log write failure stops the run with an error. The default log
directory is ignored by Git. Logs contain readings and names, not images or
face embeddings.

Install the updated dependencies before running this version. When upgrading an
older environment, replace its OpenCV package first to avoid two packages owning
the same `cv2` files:

```sh
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --upgrade -r requirements.txt
python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86
```

The existing FaceMesh extractor uses MediaPipe 0.10.21 and compatible NumPy/OpenCV
versions pinned in requirements.txt. Tune `--ear-threshold` (0.21),
`--blink-min-frames` (2), `--blink-max-frames` (7), `--closed-frames` (8),
`--smile-on` (0.38), and `--smile-off` (0.35) using the displayed readings.
Blink duration is measured in processed frames, so tune for your actual frame rate.
Short low-EAR periods display EYES CLOSING; sustained closure displays EYES CLOSED.
Blink counts increment once upon reopening after 2-7 low-EAR frames. Missing
landmarks or a lost target reset pending expression state, retaining the session count.

## Inspect individual stages

```sh
python -m src.camera       # Original camera and FPS demo
python -m src.detect       # Original Haar box detector demo
python -m src.landmarks    # YuNet boxes and five points, plus aligned preview
python -m src.alignment    # Same preview for inspecting 112x112 alignment
python -m src.embed        # Live embedding dimension and unit norm
```

The original camera/Haar demos retain their `CAMERA_INDEX` constants. The new webcam commands accept `--camera`, `--detector`, `--model`, `--db`, and `--preprocessing` where applicable; use `--help` for details. `python -m src.harr_5pt` is also a landmark preview, preserving the original filename. Haar cascades alone do not provide the five measured landmarks used by the recognition pipeline.

## Evaluate with held-out photos

Collect fresh images in the expected camera conditions, including enrolled people and people absent from the gallery. Use different captures from enrollment. Create a CSV such as `data/evaluation.csv`:

```csv
path,label
validation/belise_01.jpg,belise
validation/isaac_01.jpg,isaac
validation/visitor_01.jpg,Unknown
```

Paths are relative to the CSV. Each image must contain exactly one face. `Unknown` is the label for any unenrolled person.

```sh
python -m src.evaluate --manifest data/evaluation.csv --threshold 0.45
python -m src.evaluate --manifest data/evaluation.csv --threshold 0.55
```

Compare correct predictions, false accepts of unknown people, false rejects of known people, and wrong identities. Read/detection failures are reported separately and count against total accuracy. Tune on a validation set, then report performance on a separate test set. No evaluation photos or actual enrollments are bundled.

## Structure

```text
FaceX/
  init_project.py        Directory setup and verified model downloads
  requirements.txt      Runtime dependencies
  models/               YuNet and ArcFace ONNX weights
  data/db/              Local enrollment database
  src/
    camera.py           Existing webcam demo
    detect.py           Existing Haar detector demo
    landmarks.py        YuNet detector, Face records, drawing
    alignment.py        Five-point similarity transform
    embed.py            ONNX inference and normalization
    database.py         Storage, model compatibility, cosine matching
    enroll.py           Quality-gated webcam enrollment
    recognize.py        Webcam recognition
    evaluate.py         Held-out image evaluation
    whole_person_tracking.py  CPU whole-person tracking and face association
    harr_5pt.py          Landmark/alignment preview
    config.py           Project-relative paths
    workflow.py         CLI helpers and camera cleanup
  tests/                Automated mathematical/storage tests
  book/                 Reference material directory
```

Run modules with `python -m src.<module>` from the project root, rather than `python src/<module>.py`. Model and database defaults resolve relative to the project, not the shell's working directory.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests cover known-transform recovery, invalid landmarks/embeddings, database round-trips, replacement, model mismatches, unknown rejection, ambiguity, and camera cleanup. No camera is required. Two additional smoke tests run real model inference when weights are present and skip otherwise.

Development validation: all 10 tests passed with Python 3.12.10, NumPy 2.5.3, OpenCV 4.14.0.94, and ONNX Runtime 1.29.0; `pip check` passed. Webcam 0 successfully returned a 640x480 frame. No face was visible during that check, so a real person's enrollment and recognition accuracy remain to be verified interactively.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No Python at ...` | Recreate `.venv` using an installed Python; the original environment is not portable. |
| Missing/empty model | Run `python init_project.py --download-models`. A zero-byte file or Git LFS pointer is not a model. |
| Camera cannot open | Close Teams/Zoom/other camera programs, enable desktop camera permissions, or try `--camera 1`. |
| No window / GUI error | Run in a desktop session and install `opencv-contrib-python` from requirements.txt; remove conflicting OpenCV packages. |
| Everything is Unknown | Inspect landmarks/alignment, verify preprocessing, improve lighting, and re-enroll before lowering the threshold. |
| Wrong people accepted | Increase the threshold/margin and evaluate with held-out unknown people. |
| Model mismatch | Use the original model/settings or enroll again into a separate `--db` file. |
| Slow recognition | ResNet100 is a large CPU model; reduce people in view or supply a compatible smaller ArcFace model and re-enroll. |
| Corrupt JSON | Restore a backup or choose a new database path; invalid data is not silently discarded. |

## Reference and limitations

This project follows the modular enrollment/alignment/embedding/recognition architecture in Gabriel Baziramwabo's *Face Recognition with ArcFace ONNX and 5-Point Alignment*, supplied with this project. The implementation uses YuNet for measured landmarks, one JSON database, and held-out image evaluation; its commands and data format differ from the book's Haar/MediaPipe and NPZ examples. The existing `camera.py` and `detect.py` demos are preserved.

This is an educational recognition application with no liveness or anti-spoofing check. Enroll people with their consent and protect the local biometric database; it is not encrypted. Do not use a webcam match alone as an access-control decision. `.gitignore` covers new databases, images, and model downloads, but files already tracked by Git remain tracked: the original database and ArcFace placeholder are tracked in this repository, so inspect staged changes before committing real biometric data or large weights.
