# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import numpy as np
import time
import threading
import socket
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from werkzeug.serving import WSGIRequestHandler
from collections import deque
import argparse
import os
import sys


class QuietRequestHandler(WSGIRequestHandler):
    """
    FIX: Chi tat log IN RA CHO MOI REQUEST (vd GET /api/stats moi giay,
    GET /video_feed lien tuc...). KHONG dong toi logger 'werkzeug' noi
    chung, nho vay dong banner khoi dong cua Flask
    (" * Running on http://<lan-ip>:port") van hien thi binh thuong.
    """
    def log_request(self, code="-", size="-"):
        pass


def get_lan_ip():
    """Do IP LAN thuc te cua Jetson (khong phai 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Water Bottle Defect Detection")
parser.add_argument(
    "--port", type=int, default=5000,
    help="Flask server port (default: 5000)"
)
args, _ = parser.parse_known_args()

print("[INFO] Mode: GPU (TensorRT)")

# Luon dung GPU/TensorRT, khong con lua chon CPU nua.
import tensorrt as trt
import pycuda.driver as cuda
# FIX 1: Chi goi cuda.init() mot lan duy nhat o day
# KHONG tao context o day - se tao trong inference thread
cuda.init()
print("[INFO] PyCUDA initialized OK")

app = Flask(__name__)

# --- CONFIG ---
ENGINE_PATH  = "best.engine"
LABELS_PATH  = "labels.txt"
INPUT_SIZE   = (448, 448)

NUM_CLASSES     = 4
OUTPUT_CHANNELS = 4 + NUM_CLASSES   # 9
OUTPUT_ANCHORS  = 4116
OUTPUT_SHAPE    = (1, OUTPUT_CHANNELS, OUTPUT_ANCHORS)

CONF_THRESH  = 0.5
IOU_THRESH   = 0.45
CAMERA_INDEX = 0
MAX_HISTORY  = 50

# Nhan duoc coi la "Binh thuong" (cot Tong/Binh thuong/Phat hien loi
# tren web dua vao day). Cac nhan KHAC nhan nay deu tinh la loi.
GOOD_LABEL = "QUALIFIELD"

# Doc labels
if not os.path.exists(LABELS_PATH):
    print("[WARNING] labels.txt not found, using default labels")
    LABELS = ["good", "no_cap", "wrong_cap", "no_label", "wrong_label"]
else:
    with open(LABELS_PATH) as f:
        LABELS = [line.strip() for line in f.readlines()]

# Mau BGR co dinh cho tung loai nhan (thay cho mau random truoc day)
# Luu y: cv2 dung thu tu BGR, khong phai RGB.
COLOR_MAP = {
    "QUALIFIELD":    [0, 255, 0],     # Xanh la
    "MISSING LABEL": [0, 0, 255],     # Do
    "WRONG LABEL":     [0, 165, 255],   # Cam
    "NO CAP":        [0, 255, 255],   # Vang
}
COLORS = np.array(
    [COLOR_MAP.get(l, [255, 255, 255]) for l in LABELS],  # trang neu thieu mapping
    dtype=np.uint8,
)
for _l in LABELS:
    if _l not in COLOR_MAP:
        print("[WARNING] Nhan '%s' khong co trong COLOR_MAP, dung mau trang mac dinh" % _l)

# --- GLOBAL VARIABLES ---
lock         = threading.Lock()
output_frame = None
stats = {
    "total":  0,
    "good":   0,
    "defect": 0,
    "fps":    0.0,
    "device": "GPU",
    "counts": {label: 0 for label in LABELS},
}
history     = deque(maxlen=MAX_HISTORY)
fps_history = deque(maxlen=30)


# ================================================================
#  BYTETRACK (no external library, uses cv2.KalmanFilter)
# ================================================================

class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0
        self.kf.transitionMatrix     = F
        self.kf.processNoiseCov      = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov  = np.eye(4, dtype=np.float32) * 1e-1

        cx, cy, w, h = self._xyxy2cxywh(bbox)
        state = np.array([[cx], [cy], [w], [h],
                          [0.], [0.], [0.], [0.]], dtype=np.float32)
        self.kf.statePre  = state.copy()
        self.kf.statePost = state.copy()

        KalmanBoxTracker.count += 1
        self.id                = KalmanBoxTracker.count
        self.hits              = 1
        self.hit_streak        = 1
        self.age               = 0
        self.time_since_update = 0
        self.class_id          = 0
        self.score             = 0.0
        self.class_history     = []  # [(class_id, score), ...] trong suot doi track

    def record(self, class_id, score):
        """Ghi nhan lop du doan moi nhat cho track nay (dung de bo phieu sau)."""
        self.class_id = class_id
        self.score    = score
        self.class_history.append((class_id, score))

    @staticmethod
    def _xyxy2cxywh(bbox):
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2.0, (y1+y2)/2.0, float(x2-x1), float(y2-y1)

    @staticmethod
    def _cxywh2xyxy(cx, cy, w, h):
        return cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        cx, cy, w, h = self._xyxy2cxywh(bbox)
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def get_state(self):
        s = self.kf.statePost
        return self._cxywh2xyxy(
            float(s[0]), float(s[1]), float(s[2]), float(s[3])
        )


def iou_batch(bb_test, bb_gt):
    bb_gt   = np.expand_dims(bb_gt,   0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1   = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1   = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2   = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2   = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w     = np.maximum(0., xx2 - xx1)
    h     = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_t = ((bb_test[..., 2] - bb_test[..., 0]) *
               (bb_test[..., 3] - bb_test[..., 1]))
    area_g = ((bb_gt[..., 2]   - bb_gt[..., 0]) *
               (bb_gt[..., 3]   - bb_gt[..., 1]))
    return inter / (area_t + area_g - inter + 1e-9)


class ByteTracker(object):
    def __init__(self, max_age=30, min_hits=3,
                 iou_thresh=0.3, high_thresh=0.6, low_thresh=0.1):
        self.max_age     = max_age
        self.min_hits    = min_hits
        self.iou_thresh  = iou_thresh
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh
        self.trackers    = []
        self.frame_count = 0

    def _greedy_match(self, cost_matrix):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int)
        matched   = []
        used_rows = set()
        used_cols = set()
        flat_idx  = np.argsort(cost_matrix.ravel())
        rows, cols = np.unravel_index(flat_idx, cost_matrix.shape)
        for r, c in zip(rows, cols):
            if r not in used_rows and c not in used_cols:
                matched.append([r, c])
                used_rows.add(r)
                used_cols.add(c)
        return np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int)

    def _match(self, dets, trk_boxes):
        if len(trk_boxes) == 0 or len(dets) == 0:
            return (np.empty((0, 2), dtype=int),
                    list(range(len(dets))),
                    list(range(len(trk_boxes))))

        iou_mat  = iou_batch(
            np.array(dets,      dtype=np.float32),
            np.array(trk_boxes, dtype=np.float32)
        )
        cost_mat = 1.0 - iou_mat
        matched  = self._greedy_match(cost_mat)

        m_rows = set(matched[:, 0].tolist()) if len(matched) else set()
        m_cols = set(matched[:, 1].tolist()) if len(matched) else set()
        unmatched_d = [d for d in range(len(dets))      if d not in m_rows]
        unmatched_t = [t for t in range(len(trk_boxes)) if t not in m_cols]

        good = []
        for m in matched:
            if iou_mat[m[0], m[1]] < self.iou_thresh:
                unmatched_d.append(int(m[0]))
                unmatched_t.append(int(m[1]))
            else:
                good.append(m)
        return (np.array(good, dtype=int) if good else np.empty((0, 2), dtype=int),
                unmatched_d, unmatched_t)

    def update(self, detections):
        self.frame_count += 1
        for t in self.trackers:
            t.predict()

        trk_boxes = [t.get_state() for t in self.trackers]

        high_dets = [(b, s, c) for b, s, c in detections if s >= self.high_thresh]
        low_dets  = [(b, s, c) for b, s, c in detections
                     if self.low_thresh <= s < self.high_thresh]
        high_boxes = [b for b, s, c in high_dets]

        matched1, unmatched_high, unmatched_trk = self._match(high_boxes, trk_boxes)
        for m in matched1:
            trk = self.trackers[m[1]]
            trk.update(high_dets[m[0]][0])
            trk.record(high_dets[m[0]][2], high_dets[m[0]][1])

        if low_dets and unmatched_trk:
            low_boxes      = [b for b, s, c in low_dets]
            remain_trk_box = [trk_boxes[i] for i in unmatched_trk]
            matched2, _, still_unmatched = self._match(low_boxes, remain_trk_box)
            for m in matched2:
                t_idx = unmatched_trk[m[1]]
                trk   = self.trackers[t_idx]
                trk.update(low_dets[m[0]][0])
                trk.record(low_dets[m[0]][2], low_dets[m[0]][1])
            unmatched_trk = [unmatched_trk[i] for i in still_unmatched]

        for d_idx in unmatched_high:
            b, s, c = high_dets[d_idx]
            trk = KalmanBoxTracker(b)
            trk.record(c, s)
            self.trackers.append(trk)

        alive, finished = [], []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                alive.append(t)
            else:
                finished.append(t)
        self.trackers = alive

        results = []
        for t in self.trackers:
            if (t.time_since_update == 0 and
                    (t.hits >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                box = t.get_state()
                results.append((box, t.id, t.class_id, t.score))

        # Track vua ket thuc (bien mat khoi khung hinh) -> chot 1 lan cho moi object
        finalized = []
        for t in finished:
            if t.hits >= self.min_hits and t.class_history:
                votes = {}
                for cid, sc in t.class_history:
                    votes[cid] = votes.get(cid, 0) + 1
                final_class = max(votes.items(), key=lambda kv: kv[1])[0]
                final_score = max(sc for cid, sc in t.class_history if cid == final_class)
                finalized.append((t.id, final_class, final_score))

        return results, finalized


# ================================================================
#  PREPROCESSING / POSTPROCESSING
# ================================================================

def preprocess(frame):
    img = cv2.resize(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img, 0))


def postprocess_raw(raw, orig_shape):
    predictions = raw.reshape(OUTPUT_SHAPE)
    predictions = predictions[0].T
    h, w = orig_shape[:2]

    boxes, scores, class_ids = [], [], []
    for pred in predictions:
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        cls_scores = pred[4:]
        class_id   = int(np.argmax(cls_scores))
        confidence = float(cls_scores[class_id])
        if confidence < CONF_THRESH:
            continue
        x1 = int((cx - bw / 2.0) / INPUT_SIZE[0] * w)
        y1 = int((cy - bh / 2.0) / INPUT_SIZE[1] * h)
        x2 = int((cx + bw / 2.0) / INPUT_SIZE[0] * w)
        y2 = int((cy + bh / 2.0) / INPUT_SIZE[1] * h)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    results = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw2, bh2 = boxes[i]
                results.append(((x, y, x + bw2, y + bh2), scores[i], class_ids[i]))
    return results


# ================================================================
#  GPU PATH (TensorRT)
# ================================================================

# ================================================================
#  GPU PATH (TensorRT)
# ================================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size     = trt.volume(engine.get_binding_shape(binding))
        dtype    = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem  = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": dev_mem})
        else:
            outputs.append({"host": host_mem, "device": dev_mem})
    return inputs, outputs, bindings, stream


# ================================================================
#  DRAW + UPDATE STATS
# ================================================================


def draw_and_update(frame, tracked_objects, finalized_objects):
    """
    tracked_objects: cac track DANG hien thi trong frame nay -> chi de VE box.
    finalized_objects: cac track VUA KET THUC (object da di khoi khung hinh)
                       -> dung de CONG SO LIEU, moi object (track_id) chi tinh 1 LAN DUY NHAT.
    """
    global stats, history

    for item in tracked_objects:
        box, track_id, class_id, score = item
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = COLORS[class_id].tolist()
        label = "[%d] %s: %.2f" % (track_id, LABELS[class_id], score)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for track_id, class_id, score in finalized_objects:
        label_name = LABELS[class_id]
        stats["total"] += 1
        stats["counts"][label_name] += 1
        # FIX: nhan thuc te la "QUALIFIELD", khong phai "good", nen phai
        # so sanh voi GOOD_LABEL. Truoc day so sanh voi "good" -> luon
        # sai (moi nhan deu bi tinh la loi, ke ca QUALIFIELD).
        if label_name != GOOD_LABEL:
            stats["defect"] += 1
        else:
            stats["good"] += 1

        # FIX: ghi lich su cho TAT CA nhan (ke ca QUALIFIELD), khong
        # chi rieng loi nhu truoc day.
        history.appendleft({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "labels": [label_name],
            "score":  round(score, 2),
        })

    return frame


# ================================================================
#  INFERENCE HELPERS
# ================================================================

def _open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index=%d" % CAMERA_INDEX)
    else:
        print("[INFO] Camera opened (index=%d)" % CAMERA_INDEX)
    return cap


def _update_fps(t0):
    elapsed = time.time() - t0
    if elapsed > 0:
        fps_history.append(1.0 / elapsed)
        stats["fps"] = float(np.mean(fps_history))


# ================================================================
#  INFERENCE THREAD
# ================================================================

def inference_loop():
    tracker = ByteTracker(
        # FIX: max_age giam tu 30 -> 1 de chot ket qua NGAY khi chai
        # roi khoi khung hinh (khong phai cho ~1s nhu truoc).
        # Danh doi: neu chai bi che khuat tam thoi (occlusion) giua
        # khung hinh, co the bi tinh la "da ra khung" roi tao track moi
        # khi xuat hien lai -> nguy co dem trung 1 chai thanh 2 lan.
        max_age=1, min_hits=3,
        iou_thresh=0.3,
        high_thresh=CONF_THRESH,
        low_thresh=0.1,
    )
    _gpu_loop(tracker)


def _gpu_loop(tracker):
    """
    FIX 1: PyCUDA context management
    - Tao context bang Device(0).make_context() TRONG inference thread
    - Dung try/finally dam bao cuda_ctx.pop() luon duoc goi
    - KHONG dung atexit vi thread co the bi kill truoc khi atexit chay
    """
    global output_frame
    cuda_ctx = None

    try:
        # Tao CUDA context trong chinh thread nay
        cuda_ctx = cuda.Device(0).make_context()
        print("[INFO] CUDA context created in inference thread")

        print("[INFO] Loading TensorRT engine...")
        engine = load_engine(ENGINE_PATH)

        print("\n=== ENGINE BINDINGS ===")
        for binding in engine:
            print("  %s: shape=%s  dtype=%s  input=%s" % (
                binding,
                engine.get_binding_shape(binding),
                engine.get_binding_dtype(binding),
                engine.binding_is_input(binding)
            ))
        print("======================\n")

        context  = engine.create_execution_context()
        inputs, outputs, bindings_list, stream = allocate_buffers(engine)
        cap      = _open_camera()

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            t0  = time.time()
            img = preprocess(frame)
            np.copyto(inputs[0]["host"], img.ravel())

            # KHONG can push/pop moi frame vi context da active trong thread nay
            cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
            t_infer = time.time()
            context.execute_async_v2(bindings=bindings_list, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
            stream.synchronize()

            infer_ms = (time.time() - t_infer) * 1000.0
            print("[GPU] Infer: %.1f ms  |  FPS: %.1f" % (
                infer_ms, 1000.0 / max(infer_ms, 1.0)))

            raw_dets = postprocess_raw(outputs[0]["host"], frame.shape)
            tracked, finalized = tracker.update(raw_dets)
            frame    = draw_and_update(frame, tracked, finalized)
            _update_fps(t0)

            with lock:
                output_frame = frame.copy()

    except Exception as e:
        print("[GPU ERROR] %s" % str(e))
        import traceback
        traceback.print_exc()
    finally:
        # FIX 1: Dam bao pop context truoc khi thoat thread
        if cuda_ctx is not None:
            try:
                cuda_ctx.pop()
                print("[INFO] CUDA context popped cleanly")
            except Exception as ex:
                print("[WARNING] Could not pop CUDA context: %s" % str(ex))


# ================================================================
#  FLASK
# ================================================================

def generate_stream():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            _, buffer = cv2.imencode(
                ".jpg", output_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + frame_bytes + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    # FIX: lay IP client va thoi gian server NGAY luc render trang.
    # render_template chay lai moi khi trang duoc mo/tai lai (F5)
    # -> thong bao chi hien 1 lan/lan tai trang, khong lap lai theo
    # interval fetchStats (vi day khong qua AJAX, ma nhung san vao HTML).
    client_ip   = request.remote_addr
    server_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

    # In DUNG 1 dong ra terminal moi khi co nguoi truy cap/F5 trang chu
    # (log Werkzeug da bi tat o tren nen khong bi spam theo tung request).
    print("[ACCESS] %s - %s" % (client_ip, server_time))

    return render_template(
        "index.html",
        labels=LABELS,
        client_ip=client_ip,
        server_time=server_time,
    )

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total":   stats["total"],
        "good":    stats["good"],
        "defect":  stats["defect"],
        "device":  stats["device"],
        "counts":  stats["counts"],
        "history": list(history)[:10],
    })


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print("[INFO] Starting inference thread...")
    t = threading.Thread(target=inference_loop)
    t.daemon = True
    t.start()

    # FIX 3: Flask chi chay HTTP thuan tuy (port 5000)
    # Truy cap bang: http://<jetson-ip>:5000  (KHONG dung https://)
    # Neu browser tu dong chuyen sang https, dung: http://192.168.1.x:5000
    lan_ip = get_lan_ip()
    print("[INFO] Flask server starting on http://0.0.0.0:%d" % args.port)
    print("[INFO] Access from browser: http://%s:%d" % (lan_ip, args.port))
    print("[INFO] NOTE: Use HTTP (not HTTPS) in your browser!")
    app.run(
        host="0.0.0.0",
        port=args.port,
        threaded=True,
        use_reloader=False,   # Quan trong: tat reloader tranh tao 2 inference thread
        debug=False,          # tat debug mode tranh fork process
        request_handler=QuietRequestHandler,  # FIX: tat log spam tung request
    )
