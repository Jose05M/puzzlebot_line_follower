#!/usr/bin/env python3
"""
traffic_light_detector.py
-------------------------
Pipeline completo basado en MCR2 OpenCV:
  1. Gaussian blur          — reducción de ruido
  2. BGR -> HSV             — espacio de color para segmentación
  3. inRange por color      — máscaras binarias (rojo doble rango, amarillo, verde)
  4. bitwise_or / AND       — combinar máscaras de rojo
  5. Candidate mask         — brillo + saturación (bitwise_and)
  6. MORPH_OPEN + CLOSE     — limpieza morfológica
  7. findContours           — detección de blobs
  8. Filtros: área, circularidad, dominancia en ROI
  9. SimpleBlobDetector     — segunda pasada para confirmar
 10. ROI espacial           — solo tercio izquierdo de la imagen
 11. State machine          — red_lock: activa con RED, libera con GREEN
"""

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import math

# ---------------------------------------------------------------------------
# HSV colour ranges
# ---------------------------------------------------------------------------
RED_LOWER_1  = np.array([  0, 120,  80], dtype=np.uint8)
RED_UPPER_1  = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LOWER_2  = np.array([165, 120,  80], dtype=np.uint8)
RED_UPPER_2  = np.array([179, 255, 255], dtype=np.uint8)

YELLOW_LOWER = np.array([ 8, 120,  100], dtype=np.uint8)
YELLOW_UPPER = np.array([ 18, 255, 255], dtype=np.uint8)

GREEN_LOWER  = np.array([ 40, 100,  80], dtype=np.uint8)
GREEN_UPPER  = np.array([ 90, 255, 255], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
KERNEL_SIZE_DEFAULT   = 6
MIN_BLOB_AREA_DEFAULT = 2000
MIN_CIRCULARITY       = 0.95
MIN_DOMINANCE         = 0.20
MIN_CONVEXITY   = 0.80
MAX_ASPECT_RATIO = 1.6

class TrafficLightDetector(Node):

    def __init__(self):
        super().__init__('traffic_light_detector')

        self.declare_parameter('kernel_size',     KERNEL_SIZE_DEFAULT)
        self.declare_parameter('min_blob_area',   MIN_BLOB_AREA_DEFAULT)
        self.declare_parameter('min_circularity', MIN_CIRCULARITY)
        self.declare_parameter('camera_topic',    '/video_source/raw')
        self.declare_parameter('state_topic',     '/traffic_light/state')
        self.declare_parameter('debug_image',     True)

        self.min_area  = self.get_parameter('min_blob_area').value
        self.min_circ  = self.get_parameter('min_circularity').value
        cam_topic      = self.get_parameter('camera_topic').value
        state_topic    = self.get_parameter('state_topic').value
        self.debug_img = self.get_parameter('debug_image').value

        # State machine
        self.red_lock      = False
        self.current_state = 'UNKNOWN'

        # ROS I/O
        self.pub_state = self.create_publisher(String, state_topic, 10)
        self.sub_cam   = self.create_subscription(
            Image, cam_topic, self._image_callback, 10)

        # SimpleBlobDetector — segunda pasada de confirmación
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold        = 30
        params.maxThreshold        = 255
        params.filterByColor       = True
        params.blobColor           = 255
        params.filterByArea        = True
        params.minArea             = self.min_area
        params.maxArea             = 10_000_000
        params.filterByCircularity = True
        params.minCircularity      = 0.30
        params.maxCircularity      = 1.0
        params.filterByConvexity   = True
        params.minConvexity        = 0.5
        params.maxConvexity        = 1.0
        params.filterByInertia     = True
        params.minInertiaRatio     = 0.3
        params.maxInertiaRatio     = 1.0
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        self.get_logger().info(
            f'TrafficLightDetector listo — escuchando en {cam_topic}')

    # -----------------------------------------------------------------------
    # Paso 3-4: máscaras de color
    # -----------------------------------------------------------------------
    def _build_masks(self, hsv):
        mask_r1 = cv2.inRange(hsv, RED_LOWER_1,  RED_UPPER_1)
        mask_r2 = cv2.inRange(hsv, RED_LOWER_2,  RED_UPPER_2)
        mask_y  = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        mask_g  = cv2.inRange(hsv, GREEN_LOWER,  GREEN_UPPER)
        # bitwise_or para unir los dos rangos de rojo
        mask_r  = cv2.bitwise_or(mask_r1, mask_r2)
        return mask_r, mask_y, mask_g

    # -----------------------------------------------------------------------
    # Paso 5-6: candidate mask + morfología
    # -----------------------------------------------------------------------
    def _build_candidate(self, hsv):
        # bitwise_and entre brillo y saturación — candidate mask
        bright    = cv2.inRange(hsv[:, :, 2],  80, 255)
        sat       = cv2.inRange(hsv[:, :, 1],  50, 255)
        candidate = cv2.bitwise_and(bright, sat)
        # MORPH_OPEN elimina ruido, MORPH_CLOSE cierra huecos
        kernel    = np.ones((3, 3), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,  kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        return candidate


    def _best_blob_contours(self, candidate, mask_r, mask_y, mask_g):
        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area  = 0
        best_color = None
        best_cnt   = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue

            # Circularidad
            circularity = (4.0 * math.pi * area) / (perim ** 2)
            if circularity < MIN_CIRCULARITY:
                continue

            # Aspect ratio — rechaza objetos alargados
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > MAX_ASPECT_RATIO:
                continue

            # Convexity — rechaza formas irregulares
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            if (area / hull_area) < MIN_CONVEXITY:
                continue

            # Dominancia de color dentro del blob
            roi_cand = candidate[y:y+h, x:x+w]
            roi_r    = mask_r[y:y+h, x:x+w]
            roi_y    = mask_y[y:y+h, x:x+w]
            roi_g    = mask_g[y:y+h, x:x+w]

            cand_px = cv2.countNonZero(roi_cand)
            if cand_px < 5:
                continue

            red_px    = cv2.countNonZero(cv2.bitwise_and(roi_r, roi_cand))
            yellow_px = cv2.countNonZero(cv2.bitwise_and(roi_y, roi_cand))
            green_px  = cv2.countNonZero(cv2.bitwise_and(roi_g, roi_cand))
            total     = red_px + yellow_px + green_px

            if total == 0:
                continue

            ratios = {
                'RED':    red_px    / total,
                'YELLOW': yellow_px / total,
                'GREEN':  green_px  / total,
            }
            local_color, dominance = max(ratios.items(), key=lambda x: x[1])

            if dominance < MIN_DOMINANCE:
                continue

            if area > best_area:
                best_area  = area
                best_color = local_color
                best_cnt   = cnt

        return best_color, best_cnt

    # -----------------------------------------------------------------------
    # Paso 9: confirmación con SimpleBlobDetector (fallback)
    # -----------------------------------------------------------------------
    def _confirm_with_blob_detector(self, mask_r, mask_y, mask_g):
        best_color = None
        best_size  = 0.0

        for color, mask in [('RED', mask_r), ('YELLOW', mask_y), ('GREEN', mask_g)]:
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            kps = self.blob_detector.detect(binary)
            for kp in kps:
                if kp.size > best_size:
                    best_size  = kp.size
                    best_color = color

        return best_color

    # -----------------------------------------------------------------------
    # State machine
    # -----------------------------------------------------------------------
    def _update_state(self, best_color: str) -> str:
        if best_color == 'GREEN':
            self.red_lock = False
            return 'GREEN'

        if best_color == 'RED':
            self.red_lock = True
            return 'RED'

        if self.red_lock:
            return 'RED'

        if best_color == 'YELLOW':
            return 'YELLOW'

        return 'UNKNOWN'

    # -----------------------------------------------------------------------
    # ROS callback
    # -----------------------------------------------------------------------
    def _image_callback(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3))

        w_t       = msg.width // 2
        roi_frame = frame[:, :w_t, :]

        # Paso 1: Gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Paso 2: BGR -> HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Pasos 3-4: máscaras de color
        mask_r, mask_y, mask_g = self._build_masks(hsv)

        # Pasos 5-6: candidate mask + morfología
        candidate = self._build_candidate(hsv)

        # Pasos 7-8: contornos + dominancia
        best_color, best_cnt = self._best_blob_contours(
            candidate, mask_r, mask_y, mask_g)

        # Paso 9: fallback con SimpleBlobDetector si contornos no encontró nada
        if best_color is None:
            best_color = self._confirm_with_blob_detector(mask_r, mask_y, mask_g)

        # Paso 11: state machine
        state = self._update_state(best_color)

        if state != self.current_state:
            self.get_logger().info(
                f'Estado semáforo: {self.current_state} → {state}')
            self.current_state = state

        out_msg = String()
        out_msg.data = state
        self.pub_state.publish(out_msg)

        annotated = self._annotate_frame(frame, best_color, best_cnt, state)
        cv2.imshow('Traffic Light Detector', annotated)
        cv2.waitKey(1)

    # -----------------------------------------------------------------------
    # Visualización
    # -----------------------------------------------------------------------
    def _annotate_frame(self, frame, best_color, best_cnt, state):
        out = frame.copy()
        colour_map = {
            'RED':     (0,   0, 220),
            'YELLOW':  (0, 200, 220),
            'GREEN':   (0, 200,   0),
            'UNKNOWN': (160, 160, 160),
        }
        cv_colour = colour_map.get(state, (160, 160, 160))

        if best_cnt is not None and best_color is not None:
            (x, y), radius = cv2.minEnclosingCircle(best_cnt)
            cx, cy, r = int(x), int(y), int(radius)
            blob_colour = colour_map.get(best_color, (160, 160, 160))
            cv2.circle(out, (cx, cy), r,  blob_colour, 2)
            cv2.circle(out, (cx, cy), 3,  blob_colour, -1)
            cv2.putText(out, best_color, (cx - 30, cy - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        blob_colour, 2, cv2.LINE_AA)

        banner = f'State: {state}{"  [LOCKED]" if self.red_lock else ""}'
        cv2.rectangle(out, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(out, banner, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cv_colour, 2, cv2.LINE_AA)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
