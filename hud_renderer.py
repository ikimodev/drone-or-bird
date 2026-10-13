import cv2
import numpy as np

def draw_corner_rect(img, x1, y1, x2, y2, color, thickness=2, corner_len=16):
    """Draw tactical HUD corner brackets instead of standard rectangles."""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)

    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)

    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)

    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)

    # Faint enclosing reticle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)

def render_tactical_hud(frame, targets, show_trails=True):
    """
    Renders military-grade Aerospace C2 / Anti-Drone tactical HUD onto frame:
    - Reticle corner brackets and target ID
    - Sub-type classification (Shahed combat drone, multicopter, biological avian)
    - Kinematics (Bearing degree, velocity, aspect ratio, wing flapping frequency)
    - Neon trajectory flight trails
    - System telemetry status bar
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    drone_count = 0
    bird_count = 0

    for t in targets:
        x1, y1, x2, y2 = map(int, t.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        is_drone = (t.cls == 'drone')
        sub_type_label = getattr(t, 'sub_type', 'Target')
        
        if is_drone:
            drone_count += 1
            main_color = (40, 40, 255)     # Neon Red / Crimson
            badge_color = (20, 20, 220)
            tag_title = f"UAS THREAT #{t.id:02d} | {t.conf*100:.0f}%"
            sub_info = f"{sub_type_label.upper()} | SPD: {t.speed_px_per_s:.0f}px/s | HDG: {t.bearing_deg:.0f} DEG"
        else:
            bird_count += 1
            main_color = (0, 230, 118)     # Neon Green / Emerald
            badge_color = (0, 180, 80)
            tag_title = f"AVIAN BIO #{t.id:02d} | {t.conf*100:.0f}%"
            flap_hz = max(2.5, t.flapping_score * 8.0) if hasattr(t, 'flapping_score') else 4.0
            sub_info = f"{sub_type_label.upper()} | FLAP: {flap_hz:.1f}Hz"

        # Draw trajectory trail
        if show_trails and hasattr(t, 'history') and len(t.history) > 1:
            points = np.array(t.history, dtype=np.int32)
            for i in range(1, len(points)):
                thickness = int(np.sqrt(16 * float(i) / len(points))) + 1
                cv2.line(annotated, tuple(points[i - 1]), tuple(points[i]), main_color, thickness, cv2.LINE_AA)

        # Draw Corner Bracket Reticle
        draw_corner_rect(annotated, x1, y1, x2, y2, main_color, thickness=2, corner_len=14)

        # Center target crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(annotated, (cx, cy), main_color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

        # Tactical Header Tag (above box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(tag_title, font, 0.44, 1)
        tag_y1 = max(18, y1 - th - 10)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 12, y1), (12, 16, 24), -1)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 12, y1), badge_color, 1)
        cv2.putText(annotated, tag_title, (x1 + 6, y1 - 5), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry Sub-Tag (below box)
        (stw, sth), _ = cv2.getTextSize(sub_info, font, 0.36, 1)
        sub_y2 = min(h - 4, y2 + sth + 9)
        cv2.rectangle(annotated, (x1, y2), (x1 + stw + 10, sub_y2), (8, 12, 18), -1)
        cv2.rectangle(annotated, (x1, y2), (x1 + stw + 10, sub_y2), (40, 50, 70), 1)
        cv2.putText(annotated, sub_info, (x1 + 5, sub_y2 - 3), font, 0.36, main_color, 1, cv2.LINE_AA)

    # Top Status Bar Overlay
    bar_h = max(28, int(h * 0.055))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (8, 14, 24), -1)
    cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)
    cv2.line(annotated, (0, bar_h), (w, bar_h), (0, 210, 255), 1)

    # Responsive font sizing
    hud_font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = max(0.38, min(0.65, w / 1200.0))
    y_text = int(bar_h * 0.68)

    threat_status = "ALERT: HOSTILE UAS DETECTED" if drone_count > 0 else "C2: AIRSPACE CLEAR (AVIAN ONLY)"
    status_col = (40, 40, 255) if drone_count > 0 else (0, 210, 255)
    cv2.putText(annotated, threat_status, (12, y_text), hud_font, f_scale, status_col, 1, cv2.LINE_AA)

    counters_str = f"DRONES: {drone_count:02d} | BIRDS: {bird_count:02d}"
    (cw, ch), _ = cv2.getTextSize(counters_str, hud_font, f_scale, 1)
    cv2.putText(annotated, counters_str, (max(int(w * 0.55), w - cw - 14), y_text), hud_font, f_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated

