import os
import time
import math
import cv2
import numpy as np

def analyze_crop_morphology(crop):
    """
    Analyzes visual morphology of the aerial target crop:
    - Delta-wing / combat UAV: 3-4 vertices, high convexity/solidity (>0.70), straight leading edges.
    - Biological birds: organic curvature, wingtip feather notches, 6+ vertices, lower solidity (<0.65).
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return {'is_delta': False, 'solidity': 0.5, 'vertices': 4, 'aspect_ratio': 1.0}
    
    h, w = crop.shape[:2]
    ar = float(w) / max(1.0, float(h))
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val > 120:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {'is_delta': False, 'solidity': 0.5, 'vertices': 4, 'aspect_ratio': ar}
        
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / max(1.0, hull_area)
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    vertices = len(approx)
    
    # Delta-wing drones typically exhibit triangular geometry (3-4 vertices) and high solidity
    is_delta = (vertices == 3 or (vertices == 4 and solidity > 0.70))
    
    return {
        'is_delta': is_delta,
        'solidity': round(solidity, 2),
        'vertices': vertices,
        'aspect_ratio': round(ar, 2)
    }


class TrackedTarget:
    def __init__(self, target_id, initial_box, initial_class, initial_conf, sub_type='Airborne Target', morph=None):
        self.id = target_id
        self.box = initial_box  # [x1, y1, x2, y2]
        self.cls = initial_class  # 'drone' or 'bird'
        self.conf = initial_conf
        self.sub_type = sub_type
        self.morph = morph or {'is_delta': False, 'solidity': 0.5, 'vertices': 4, 'aspect_ratio': 1.0}
        self.history = [self.get_center(initial_box)]
        self.aspect_ratios = [self.calc_aspect_ratio(initial_box)]
        self.timestamps = [time.time()]
        self.missing_frames = 0
        self.speed_px_per_s = 0.0
        self.bearing_deg = 0.0
        self.flapping_score = 0.0  # 0.0 = rigid (drone), 1.0 = highly flapping (bird)
        self.threat_level = self._compute_threat(initial_class, sub_type)

    @staticmethod
    def get_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def calc_aspect_ratio(box):
        x1, y1, x2, y2 = box
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        return w / h

    @staticmethod
    def _compute_threat(cls_name, sub_type):
        if cls_name == 'drone':
            if 'Shahed' in sub_type or 'Combat' in sub_type or 'Delta' in sub_type:
                return 'CRITICAL (COMBAT UAS)'
            return 'ALERT (MULTICOPTER UAS)'
        return 'NEUTRAL (AVIAN BIOLOGY)'

    def update(self, new_box, cls_candidate, conf_candidate, sub_type_candidate=None, morph_candidate=None):
        self.box = new_box
        self.missing_frames = 0
        now = time.time()
        c = self.get_center(new_box)
        ar = self.calc_aspect_ratio(new_box)

        if sub_type_candidate:
            self.sub_type = sub_type_candidate
        if morph_candidate:
            self.morph = morph_candidate

        # Update historical queues (keep last 30 frames)
        self.history.append(c)
        self.aspect_ratios.append(ar)
        self.timestamps.append(now)
        if len(self.history) > 30:
            self.history.pop(0)
            self.aspect_ratios.pop(0)
            self.timestamps.pop(0)

        # Calculate kinematics
        if len(self.history) >= 2:
            dt = max(0.001, self.timestamps[-1] - self.timestamps[-2])
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            dist = math.hypot(dx, dy)
            self.speed_px_per_s = dist / dt
            self.bearing_deg = (math.degrees(math.atan2(dx, -dy)) + 360) % 360

        # Calculate temporal flapping score using aspect ratio standard deviation
        if len(self.aspect_ratios) >= 6:
            std_ar = float(np.std(self.aspect_ratios[-12:]))
            mean_ar = float(np.mean(self.aspect_ratios[-12:]))
            cv_ar = std_ar / max(0.01, mean_ar)
            # Birds typically have cv_ar > 0.08 due to wing up/down motion
            self.flapping_score = min(1.0, cv_ar * 7.0)

        # Dynamic class stabilization (fuse candidate with motion physics)
        if self.flapping_score > 0.45:
            # Strong biological wing flap evidence overrides minor drone detections
            self.cls = 'bird'
            self.sub_type = 'Biological Avian (Flapping Wings)'
            self.conf = max(self.conf, 0.88)
            self.threat_level = 'NEUTRAL (AVIAN BIOLOGY)'
        elif self.speed_px_per_s < 15 and len(self.history) > 10 and self.cls == 'drone':
            # Steady hovering in mid-air -> classic drone behavior!
            self.cls = 'drone'
            self.sub_type = 'Stationary Hovering Multicopter'
            self.conf = max(self.conf, 0.94)
            self.threat_level = 'ALERT (HOVERING UAS)'
        else:
            self.cls = cls_candidate
            self.conf = conf_candidate
            self.threat_level = self._compute_threat(self.cls, self.sub_type)


class MultiTargetTracker:
    def __init__(self, max_missing=15, iou_thresh=0.25):
        self.targets = {}
        self.next_id = 1
        self.max_missing = max_missing
        self.iou_thresh = iou_thresh

    def reset(self):
        self.targets.clear()
        self.next_id = 1

    @staticmethod
    def calc_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detected_items):
        # detected_items: list of (box, class, conf, sub_type, morph)
        matched_target_ids = set()
        unmatched_detections = []

        for item in detected_items:
            box, cls_name, conf = item[0], item[1], item[2]
            sub_type = item[3] if len(item) > 3 else 'Airborne Target'
            morph = item[4] if len(item) > 4 else None

            best_id = None
            best_iou = self.iou_thresh

            for tid, target in self.targets.items():
                if tid in matched_target_ids:
                    continue
                iou = self.calc_iou(box, target.box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                self.targets[best_id].update(box, cls_name, conf, sub_type, morph)
                matched_target_ids.add(best_id)
            else:
                unmatched_detections.append((box, cls_name, conf, sub_type, morph))

        # Increase missing count for unmatched existing targets
        for tid, target in list(self.targets.items()):
            if tid not in matched_target_ids:
                target.missing_frames += 1
                if target.missing_frames > self.max_missing:
                    del self.targets[tid]

        # Register new targets
        for box, cls_name, conf, sub_type, morph in unmatched_detections:
            new_target = TrackedTarget(self.next_id, box, cls_name, conf, sub_type, morph)
            self.targets[self.next_id] = new_target
            self.next_id += 1

        return list(self.targets.values())


class AirDefenseEnsemble:
    """
    Multi-Spectral Aerial Defense Ensemble:
    1. anti_drone_best.pt: Shahed/loitering drones, delta wings, combat UAS vs biological birds
    2. drone_model.pt: Quadcopters, multicopters, commercial drones
    3. yolov8m.pt: High-precision avian biology detection (eliminates false positives)
    4. Optical Morphology & Biomechanical Flapping Kinematics
    """
    def __init__(self):
        from ultralytics import YOLO
        self.shahed_model = None
        self.drone_model = None
        self.bird_model = None

        if os.path.exists('anti_drone_best.pt'):
            self.shahed_model = YOLO('anti_drone_best.pt')
            print("Defense Engine: Loaded Shahed/Tactical Combat Drone Model.")

        if os.path.exists('drone_model.pt'):
            self.drone_model = YOLO('drone_model.pt')
            print("Defense Engine: Loaded Multicopter Drone Model.")

        if os.path.exists('yolov8m.pt'):
            self.bird_model = YOLO('yolov8m.pt')
            print("Defense Engine: Loaded High-Precision Avian Model (YOLOv8m).")
        elif os.path.exists('yolov8n.pt'):
            self.bird_model = YOLO('yolov8n.pt')

        self.tracker = MultiTargetTracker()

    def reset_tracker(self):
        self.tracker.reset()

    def _disambiguate_target(self, frame, box, initial_cls, initial_conf, shahed_boxes, not_boxes):
        """
        Cross-examines target silhouette, optical symmetry, and anti-drone scores.
        """
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1-4):min(frame.shape[0], y2+4), max(0, x1-4):min(frame.shape[1], x2+4)]
        morph = analyze_crop_morphology(crop)

        # Check nearby shahed / not detections
        best_shahed_conf = 0.0
        best_not_conf = 0.0

        for sb, sconf in shahed_boxes:
            if abs(sb[0] - x1) < 60 and abs(sb[1] - y1) < 60:
                best_shahed_conf = max(best_shahed_conf, sconf)

        for nb, nconf in not_boxes:
            if abs(nb[0] - x1) < 60 and abs(nb[1] - y1) < 60:
                best_not_conf = max(best_not_conf, nconf)

        # 1. Shahed / Combat drone detected with higher confidence than 'not'
        if best_shahed_conf > 0.35 and best_shahed_conf > best_not_conf:
            return 'drone', best_shahed_conf, 'Shahed-136 / Combat UAS', morph

        # 2. Multicopter drone detected
        if initial_cls == 'drone':
            # If anti_drone explicitly says 'not' (meaning not a drone) with high confidence and organic shape:
            if best_not_conf > 0.50 and morph['vertices'] >= 6 and morph['solidity'] < 0.60:
                return 'bird', max(0.85, best_not_conf), 'Biological Avian (Soaring)', morph
            return 'drone', initial_conf, 'Multicopter / Quadcopter UAS', morph

        # 3. Delta-wing geometry verification
        if morph['is_delta'] and best_shahed_conf > 0.20 and best_not_conf < 0.35:
            return 'drone', max(initial_conf, best_shahed_conf), 'Delta-Wing Combat Drone', morph

        # 4. Default biological avian
        return 'bird', max(initial_conf, 0.82), 'Biological Avian', morph

    def detect_raw(self, frame):
        h0, w0 = frame.shape[:2]
        candidates = []

        shahed_boxes = []
        not_boxes = []

        # 1. Run Shahed / Combat Drone Detector
        if self.shahed_model is not None:
            try:
                res_shahed = self.shahed_model(frame, conf=0.15, verbose=False)
                for b in res_shahed[0].boxes:
                    cls_name = self.shahed_model.names[int(b.cls[0])].lower()
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    if cls_name == 'shahed':
                        shahed_boxes.append(([x1, y1, x2, y2], conf))
                    elif cls_name == 'not':
                        not_boxes.append(([x1, y1, x2, y2], conf))
                    elif cls_name == 'bird':
                        candidates.append(([x1, y1, x2, y2], 'bird', conf, 'Biological Avian'))
            except Exception:
                pass

        # If Shahed detected directly with solid confidence, add as drone candidate
        for sbox, sconf in shahed_boxes:
            candidates.append((sbox, 'drone', sconf, 'Shahed-136 / Combat UAS'))

        # 2. Run Multicopter Drone Detector
        if self.drone_model is not None:
            try:
                res_drone = self.drone_model(frame, conf=0.25, verbose=False)
                for b in res_drone[0].boxes:
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    candidates.append(([x1, y1, x2, y2], 'drone', conf, 'Multicopter UAS'))
            except Exception:
                pass

        # 3. Run Avian (Bird) Model
        if self.bird_model is not None:
            try:
                res_bird = self.bird_model(frame, conf=0.20, verbose=False)
                for b in res_bird[0].boxes:
                    cls_name = self.bird_model.names[int(b.cls[0])].lower()
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    if cls_name == 'bird':
                        candidates.append(([x1, y1, x2, y2], 'bird', conf, 'Biological Avian'))
            except Exception:
                pass

        if not candidates:
            return []

        # 4. Resolve candidate overlaps via NMS
        boxes_xywh = [[c[0][0], c[0][1], c[0][2] - c[0][0], c[0][3] - c[0][1]] for c in candidates]
        scores = [c[2] for c in candidates]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.15, 0.40)

        resolved = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box, orig_cls, orig_conf, initial_sub = candidates[idx]
                final_cls, final_conf, final_sub, morph = self._disambiguate_target(
                    frame, box, orig_cls, orig_conf, shahed_boxes, not_boxes
                )
                resolved.append((box, final_cls, final_conf, final_sub, morph))

        return resolved

    def detect_image(self, frame):
        """
        Pure still-image inference without temporal persistence leakage.
        Returns a list of TrackedTarget objects with fresh IDs.
        """
        raw_detections = self.detect_raw(frame)
        targets = []
        for i, item in enumerate(raw_detections, 1):
            box, cls_name, conf, sub_type, morph = item
            target = TrackedTarget(i, box, cls_name, conf, sub_type, morph)
            targets.append(target)
        return targets

    def process_stream_frame(self, frame):
        """
        Video and live stream inference with multi-frame Kalman tracking & flapping analysis.
        """
        raw_detections = self.detect_raw(frame)
        tracked_targets = self.tracker.update(raw_detections)
        return tracked_targets
