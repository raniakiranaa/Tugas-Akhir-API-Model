import os, re, glob, tempfile, shutil, subprocess
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from shutil import which as shutil_which

load_dotenv()

def _getenv(key, default=None):
    v = os.getenv(key)
    return v if v is not None and v != "" else default

BASELINE_MODEL_PATH = _getenv("BASELINE_MODEL_PATH", "models/DERN_new")
FINE_TUNE_MODEL_PATH = "models/DERN_fine_tune"
SCALER_PATH         = _getenv("SCALER_PATH", "resource/daisee_scaler.pkl")
CSV_PATH            = _getenv("CSV_PATH", "sample data/sample_data.csv")
RAW_FRAMES_DIR      = _getenv("RAW_FRAMES_DIR", "sample data/sample_data_frame")
OPENFACE_BIN        = _getenv("OPENFACE_BIN", "OpenFace/build/bin/FeatureExtraction")
VIDEO_PATH          = _getenv("VIDEO_PATH", "sample data/sample_data_video.mp4")

NON_FEATURE_COLS = ['subject_id', 'clip_id', 'frame', 'confidence', 'engagement']
INDEX2LABEL = {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"}

VIDEO_FRAME_CAP = 300
MAX_FRAMES = 150
CROP_SIZE = 160 

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        T = int(input_shape[1])         # timesteps, e.g., 150
        F = int(input_shape[-1])        # feature dim, e.g., 64
        self.W = self.add_weight("att_weight", shape=(F, 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("att_bias", shape=(T, 1),
                                 initializer="zeros", trainable=True)
    def call(self, x):  # x: (B, T, F)
        e = tf.tanh(tf.tensordot(x, self.W, axes=[[2],[0]]) + self.b)  # (B, T, 1)
        a = tf.nn.softmax(e, axis=1)                                   # (B, T, 1)
        return tf.reduce_sum(a * x, axis=1)                            # (B, F)

# ---------- DOWNSAMPLING HELPERS (keep 1st, 3rd, 5th, ... = positional stride starting at 0) ----------
def _downsample_df_positional(df: pd.DataFrame, stride: int = 2, max_frames: int = MAX_FRAMES):
    """
    Downsample per-clip *before* scaling and tensor build:
    keep rows at positions 0,2,4,... (the 1st, 3rd, 5th... items), then cap to max_frames.
    """
    if "frame" in df.columns:
        df = df.assign(frame=pd.to_numeric(df["frame"], errors="coerce")).dropna(subset=["frame"])
    df = df.sort_values(["clip_id", "frame"])
    return df.groupby("clip_id", group_keys=False).apply(lambda g: g.iloc[::stride].head(max_frames))

def _list_downsampled_frames_sorted(in_dir: str, stride: int = 2, limit: int = MAX_FRAMES):
    """
    Select image files by position (0,2,4,...) before alignment/OpenFace.
    """
    files = sorted(glob.glob(os.path.join(in_dir, "*.jpg")))
    files = files[::stride]
    return files[:limit]

# ------------------------------------------------------------------------------------------------------

def _clip_id_from_example(filename: str) -> str:
    # e.g., 001_2_001.jpg -> 001_2  |  sample_001.jpg -> sample
    base = os.path.basename(filename)
    return "_".join(base.split("_")[:-1])

def _ensure_openface_executable(path: str):
    if not os.path.isfile(path):
        raise HTTPException(status_code=500, detail=f"OPENFACE_BIN not found at {path}")
    if not os.access(path, os.X_OK):
        try:
            os.chmod(path, os.stat(path).st_mode | 0o111)
        except Exception:
            pass
    if not os.access(path, os.X_OK):
        raise HTTPException(status_code=500, detail=(
            "OpenFace binary is not executable."
        ))

def _align_to_temp(in_dir: str, max_frames: int = MAX_FRAMES, image_size: int = CROP_SIZE):
    # ↓↓↓ DOWNSAMPLE EARLY (positional 0,2,4,...) ↓↓↓
    files = _list_downsampled_frames_sorted(in_dir, stride=2, limit=max_frames)
    if not files:
        raise HTTPException(status_code=400, detail="No frames found in RAW_FRAMES_DIR.")
    tdir = tempfile.mkdtemp(prefix="aligned_")
    saved, missed = 0, 0
    for path in files:
        img = cv2.imread(path)
        if img is None:
            missed += 1
            continue
        faces = face_app.get(img)
        if not faces:
            missed += 1
            continue
        face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        try:
            aligned = face_align.norm_crop(img, landmark=face.kps, image_size=image_size)
        except Exception:
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                missed += 1
                continue
            aligned = cv2.resize(crop, (image_size, image_size))
        out_path = os.path.join(tdir, os.path.basename(path))
        cv2.imwrite(out_path, aligned)
        saved += 1
    if saved == 0:
        shutil.rmtree(tdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Face alignment failed on all frames.")
    return tdir 

def _run_openface_on_dir(in_dir: str):
    _ensure_openface_executable(OPENFACE_BIN)
    out_dir = tempfile.mkdtemp(prefix="openface_")
    cmd = [
        OPENFACE_BIN,
        "-fdir", in_dir,
        "-out_dir", out_dir,
        "-2Dfp", "-3Dfp", "-pose", "-aus", "-gaze", "-pdmparams"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err = res.stderr.decode(errors="ignore")[:2000]
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"OpenFace failed: {err}")
    return out_dir  # caller collects CSV(s)

def _collect_openface_df(out_dir: str, aligned_dir: str) -> pd.DataFrame:
    """
    Combine and clean OpenFace CSVs:
      - combine per-frame CSVs
      - add clip_id, frame (from filename) when needed
      - drop rows where success == 0
      - drop columns: success, face_id, timestamp
    """
    csvs = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    if not csvs:
        raise HTTPException(status_code=500, detail="OpenFace produced no CSV.")

    big_csv = max(csvs, key=os.path.getsize)
    df_big = pd.read_csv(big_csv, low_memory=False)
    if len(df_big) > 1:
        df_list = [df_big]
    else:
        df_list = []
        for csv_file in csvs:
            df = pd.read_csv(csv_file, low_memory=False)
            frame = os.path.splitext(os.path.basename(csv_file))[0].split('_')[-1]
            df['frame'] = frame if 'frame' not in df.columns else df['frame']
            df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    files = sorted(glob.glob(os.path.join(aligned_dir, "*.jpg")))
    clip_id = _clip_id_from_example(files[0]) if files else "clip"
    if 'clip_id' not in combined_df.columns:
        combined_df['clip_id'] = clip_id

    if 'success' in combined_df.columns:
        combined_df = combined_df[combined_df['success'] != 0]
    combined_df.drop(columns=['success', 'face_id', 'timestamp'], inplace=True, errors='ignore')

    if 'frame' in combined_df.columns:
        combined_df['frame'] = pd.to_numeric(combined_df['frame'], errors='coerce')

    if 'engagement' not in combined_df.columns:
        combined_df['engagement'] = 0

    return combined_df

def build_tensors(df: pd.DataFrame, feature_cols, max_frames=MAX_FRAMES):
    """
    NO DOWNSAMPLING HERE anymore. Data should already be downsampled earlier.
    """
    if "frame" in df.columns:
        df = df.assign(frame=pd.to_numeric(df["frame"], errors="coerce")).dropna(subset=["frame"])
    df = df.sort_values(['clip_id', 'frame'])

    seq_list, agg_list, clip_ids = [], [], []
    for clip_id, g in df.groupby('clip_id', sort=True):
        g = g.head(max_frames)  # cap only

        if g.empty:
            continue
        X = g[feature_cols].to_numpy(dtype=np.float32)
        if X.size == 0:
            continue

        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        mn = np.nanmin(X, axis=0);  mx = np.nanmax(X, axis=0)
        agg = np.concatenate([mu, sd, mn, mx], axis=0).astype(np.float32)

        if X.shape[0] < max_frames:
            pad = np.zeros((max_frames - X.shape[0], X.shape[1]), dtype=np.float32)
            X_seq = np.vstack([X, pad])
        else:
            X_seq = X 

        seq_list.append(X_seq)
        agg_list.append(agg)
        clip_ids.append(str(clip_id))

    if not seq_list:
        raise HTTPException(status_code=400, detail="No valid frames found after preprocessing.")
    return clip_ids, np.stack(seq_list), np.stack(agg_list)

def topk(probs, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [{"index": int(i), "label": INDEX2LABEL.get(int(i), str(i)), "prob": float(probs[i])} for i in idx]

def _ensure_ffmpeg():
    exe = shutil_which("ffmpeg")
    if not exe:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found"
        )
    return exe

def _extract_frames_to_temp(video_path: str, cap: int = VIDEO_FRAME_CAP) -> str:
    """
    Extract frames from the given video into a temp folder, keeping 1st, 3rd, 5th, ...
    (i.e., ffmpeg select even-indexed frames n=0,2,4,...).
    """
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    ffmpeg = _ensure_ffmpeg()

    stem = os.path.splitext(os.path.basename(video_path))[0]  # clipid from video name
    out_dir = tempfile.mkdtemp(prefix="frames_")
    out_pattern = os.path.join(out_dir, f"{stem}" + "_%03d.jpg")

    # Keep every 2nd frame, starting from the first: n = 0,2,4,...
    cmd = [
        ffmpeg, "-i", video_path,
        "-vf", "select='not(mod(n\\,2))'",
        "-vsync", "vfr",
        out_pattern,
        "-hide_banner", "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")

    imgs = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
    if not imgs:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No frames extracted from video.")
    if cap and len(imgs) > cap:
        for p in imgs[cap:]:
            try: os.remove(p)
            except: pass

    return out_dir

scaler = joblib.load(SCALER_PATH)
model = load_model(BASELINE_MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)
model_fine_tune = load_model(FINE_TUNE_MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)
app = FastAPI()

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Predict from OpenFace features (CSV path)
@app.get("/predict-from-feature")
def predict_sample():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail=f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH, dtype={'clip_id': str}, low_memory=False)
    if 'engagement' not in df.columns:
        df['engagement'] = 0

    # ↓↓↓ Downsample BEFORE scaling/tensors (1st, 3rd, 5th...)
    df = _downsample_df_positional(df, stride=2, max_frames=MAX_FRAMES)

    if hasattr(scaler, "feature_names_in_"):
        feature_cols = [c for c in scaler.feature_names_in_.tolist() if c in df.columns]
        missing = [c for c in scaler.feature_names_in_.tolist() if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV missing features expected by scaler: {missing[:20]}...")
    else:
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Scale after downsampling
    df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

    clip_ids, X_seq, X_agg = build_tensors(df, feature_cols)
    try:
        y_prob = model.predict([X_seq, X_agg], verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    y_idx = np.argmax(y_prob, axis=1)
    results = []
    for cid, idx, probs in zip(clip_ids, y_idx, y_prob):
        results.append({
            "clip_id": cid,
            "predicted_index": int(idx),
            "predicted_label": INDEX2LABEL.get(int(idx), str(idx)),
            "probabilities": [float(p) for p in probs],
            "top3": topk(probs, k=min(3, probs.shape[0]))
        })
    return {"n_clips": len(results), "predictions": results}

# Predict from raw frames folder
@app.get("/predict-from-frames")
def predict_from_frames():
    if not os.path.isdir(RAW_FRAMES_DIR):
        raise HTTPException(status_code=404, detail=f"{RAW_FRAMES_DIR} not found")

    aligned_dir = None
    openface_dir = None
    try:
        # Align only the downsampled positional frames (0,2,4,...)
        aligned_dir = _align_to_temp(RAW_FRAMES_DIR, max_frames=MAX_FRAMES, image_size=CROP_SIZE)
        openface_dir = _run_openface_on_dir(aligned_dir)
        df = _collect_openface_df(openface_dir, aligned_dir)

        if hasattr(scaler, "feature_names_in_"):
            feature_cols = [c for c in scaler.feature_names_in_.tolist() if c in df.columns]
            missing = [c for c in scaler.feature_names_in_.tolist() if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"OpenFace CSV missing features expected by scaler: {missing[:20]}...")
        else:
            feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # Scale (no additional downsample now)
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

        clip_ids, X_seq, X_agg = build_tensors(df, feature_cols)
        y_prob = model.predict([X_seq, X_agg], verbose=0)
        y_idx = np.argmax(y_prob, axis=1)

        results = []
        for cid, idx, probs in zip(clip_ids, y_idx, y_prob):
            results.append({
                "clip_id": cid,
                "predicted_index": int(idx),
                "predicted_label": INDEX2LABEL.get(int(idx), str(idx)),
                "probabilities": [float(p) for p in probs],
                "top3": topk(probs, k=min(3, probs.shape[0]))
            })
        return {"n_clips": len(results), "predictions": results}
    finally:
        if aligned_dir and os.path.isdir(aligned_dir):
            shutil.rmtree(aligned_dir, ignore_errors=True)
        if openface_dir and os.path.isdir(openface_dir):
            shutil.rmtree(openface_dir, ignore_errors=True)

# Predict from a single configured video file
@app.get("/predict-from-video")
def predict_from_video():
    if not os.path.isfile(VIDEO_PATH):
        raise HTTPException(status_code=404, detail=f"Video not found: {VIDEO_PATH}")

    frames_dir = None
    aligned_dir = None
    openface_dir = None
    try:
        frames_dir = _extract_frames_to_temp(VIDEO_PATH, cap=VIDEO_FRAME_CAP)   # ffmpeg keeps 1st,3rd,5th...
        aligned_dir = _align_to_temp(frames_dir, max_frames=MAX_FRAMES, image_size=CROP_SIZE)
        openface_dir = _run_openface_on_dir(aligned_dir)
        df = _collect_openface_df(openface_dir, aligned_dir)

        if hasattr(scaler, "feature_names_in_"):
            feature_cols = [c for c in scaler.feature_names_in_.tolist() if c in df.columns]
            missing = [c for c in scaler.feature_names_in_.tolist() if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"OpenFace CSV missing features expected by scaler: {missing[:20]}...")
        else:
            feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[feature_cols] = df[feature_cols].fillna(0.0)

        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

        clip_ids, X_seq, X_agg = build_tensors(df, feature_cols)
        y_prob = model.predict([X_seq, X_agg], verbose=0)
        y_idx = np.argmax(y_prob, axis=1)

        results = []
        for cid, idx, probs in zip(clip_ids, y_idx, y_prob):
            results.append({
                "clip_id": cid,
                "predicted_index": int(idx),
                "predicted_label": INDEX2LABEL.get(int(idx), str(idx)),
                "probabilities": [float(p) for p in probs],
                "top3": topk(probs, k=min(3, probs.shape[0]))
            })
        return {"n_clips": len(results), "predictions": results}
    finally:
        for d in (frames_dir, aligned_dir, openface_dir):
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

# Predict from uploaded video (fine-tuned or baseline paths apply the same pipeline)
@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    cap: int = VIDEO_FRAME_CAP
):
    allowed_exts = {".mp4", ".mov", ".mkv", ".avi"}
    orig_name = file.filename or "upload.mp4"
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in allowed_exts:
        ext = ".mp4"  # default

    tmp_video = None
    frames_dir = None
    aligned_dir = None
    openface_dir = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_video = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        await file.close()

        frames_dir = _extract_frames_to_temp(tmp_video, cap=cap)  # ffmpeg keeps 1st,3rd,5th...
        aligned_dir = _align_to_temp(frames_dir, max_frames=MAX_FRAMES, image_size=CROP_SIZE)
        openface_dir = _run_openface_on_dir(aligned_dir)
        df = _collect_openface_df(openface_dir, aligned_dir)

        if hasattr(scaler, "feature_names_in_"):
            feature_cols = [c for c in scaler.feature_names_in_.tolist() if c in df.columns]
            missing = [c for c in scaler.feature_names_in_.tolist() if c not in df.columns]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"OpenFace CSV missing features expected by scaler: {missing[:20]}..."
                )
        else:
            feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[feature_cols] = df[feature_cols].fillna(0.0)

        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

        clip_ids, X_seq, X_agg = build_tensors(df, feature_cols)
        y_prob = model.predict([X_seq, X_agg], verbose=0)
        y_idx = np.argmax(y_prob, axis=1)

        results = []
        for cid, idx, probs in zip(clip_ids, y_idx, y_prob):
            results.append({
                "clip_id": cid,
                "predicted_index": int(idx),
                "predicted_label": INDEX2LABEL.get(int(idx), str(idx)),
                "probabilities": [float(p) for p in probs],
                "top3": topk(probs, k=min(3, probs.shape[0]))
            })

        return {"n_clips": len(results), "predictions": results}
    finally:
        if tmp_video and os.path.isfile(tmp_video):
            try: os.remove(tmp_video)
            except: pass
        for d in (frames_dir, aligned_dir, openface_dir):
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

@app.post("/predict-video-fine-tune")
async def predict_video(
    file: UploadFile = File(...),
    cap: int = VIDEO_FRAME_CAP
):
    allowed_exts = {".mp4", ".mov", ".mkv", ".avi"}
    orig_name = file.filename or "upload.mp4"
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in allowed_exts:
        ext = ".mp4"  # default

    tmp_video = None
    frames_dir = None
    aligned_dir = None
    openface_dir = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_video = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        await file.close()

        frames_dir = _extract_frames_to_temp(tmp_video, cap=cap)  # ffmpeg keeps 1st,3rd,5th...
        aligned_dir = _align_to_temp(frames_dir, max_frames=MAX_FRAMES, image_size=CROP_SIZE)
        openface_dir = _run_openface_on_dir(aligned_dir)
        df = _collect_openface_df(openface_dir, aligned_dir)

        if hasattr(scaler, "feature_names_in_"):
            feature_cols = [c for c in scaler.feature_names_in_.tolist() if c in df.columns]
            missing = [c for c in scaler.feature_names_in_.tolist() if c not in df.columns]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"OpenFace CSV missing features expected by scaler: {missing[:20]}..."
                )
        else:
            feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[feature_cols] = df[feature_cols].fillna(0.0)

        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

        clip_ids, X_seq, X_agg = build_tensors(df, feature_cols)
        y_prob = model_fine_tune.predict([X_seq, X_agg], verbose=0)
        y_idx = np.argmax(y_prob, axis=1)

        results = []
        for cid, idx, probs in zip(clip_ids, y_idx, y_prob):
            results.append({
                "clip_id": cid,
                "predicted_index": int(idx),
                "predicted_label": INDEX2LABEL.get(int(idx), str(idx)),
                "probabilities": [float(p) for p in probs],
                "top3": topk(probs, k=min(3, probs.shape[0]))
            })

        return {"n_clips": len(results), "predictions": results}
    finally:
        if tmp_video and os.path.isfile(tmp_video):
            try: os.remove(tmp_video)
            except: pass
        for d in (frames_dir, aligned_dir, openface_dir):
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)