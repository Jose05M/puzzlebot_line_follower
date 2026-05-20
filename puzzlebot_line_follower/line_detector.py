#!/usr/bin/env python3
#PRUEBA
import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

class LineDetector(Node):

    def __init__(self):
        super().__init__('line_detector')
        # ── Máscara trapezoidal (ROI inferior) ───────────────────────────────
        self.mask = np.zeros((120, 320), dtype=np.uint8)
        top_width = int(320 * 0.9)
        trapezoid = np.array([[((320 - top_width) // 2, 0),
                               ((320 + top_width) // 2, 0),
                               (320, 120),
                               (0, 120)]], dtype=np.int32)
        cv2.fillPoly(self.mask, trapezoid, 255)

        self.sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.kernel = np.ones((3, 3), np.uint8)


        self.kernel = np.ones((3, 3), np.uint8)

        # PARAMETERS
        self.declare_parameter('camera_topic', '/video_source/raw')
        self.declare_parameter('line_error_topic', '/line_error')
        self.declare_parameter('debug_view', True)

        camera_topic = self.get_parameter('camera_topic').value
        error_topic  = self.get_parameter('line_error_topic').value
        self.debug_view = self.get_parameter('debug_view').value
        

        # ROS
        self.sub = self.create_subscription(Image,camera_topic,self.image_callback,10)
        self.pub_error = self.create_publisher(Float32,error_topic,10)

        # INTERNAL STATE
        self.last_error = 0.0
        self.get_logger().info('Connected Components Line Detector Started')


    # ── Funciones ──────────────────────────────────────────────────────────────
    def promedio_linea(self, lines, height):
        xs, ys = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xs += [x1, x2]
            ys += [y1, y2]
        if len(xs) == 0:
            return None
        m, b = np.polyfit(ys, xs, 1)
        y_bottom = height
        y_top = int(height * 0.6)
        return (int(m * y_bottom + b), int(m * y_top + b))
    
    # IMAGE CALLBACK
    def image_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3))
        img = cv2.resize(img, (320, 240))

        height, width = img.shape[:2]

        # ── ROI inferior: 50% inferior para seguimiento de línea ─────────────
        offset_y   = int(height * 0.5)
        roi        = img[offset_y:, :]
        roi_height, roi_width = roi.shape[:2]

        # ════════════════════════════════════════════════════════════════════
        # SEGUIDOR DE LÍNEA
        # ════════════════════════════════════════════════════════════════════
        roi_masked    = cv2.bitwise_and(roi, roi, mask=self.mask)
        gray          = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
        sharpened     = cv2.filter2D(gray, -1, self.sharpen_kernel)
        blurred       = cv2.GaussianBlur(sharpened, (9, 9), 0)
        _, binary_inv = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        morph      = cv2.erode(binary_inv, self.kernel, iterations=2)
        morph      = cv2.dilate(morph, self.kernel, iterations=2)
        edges      = cv2.Canny(morph, 50, 150, apertureSize=3)
        inner_mask = cv2.erode(self.mask, self.kernel, iterations=5)
        edges_roi  = cv2.bitwise_and(edges, edges, mask=inner_mask)

        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180,
                                 threshold=30, minLineLength=15, maxLineGap=100)

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 10:
                    continue
                (left_lines if angle < 0 else right_lines).append(line)

        centro_imagen = roi_width // 2
        resultado_izq = self.promedio_linea(left_lines,  roi_height)
        resultado_der = self.promedio_linea(right_lines, roi_height)

        x_izq = resultado_izq[0] if resultado_izq is not None else 0
        x_der = resultado_der[0] if resultado_der is not None else roi_width

        centro_carril = (x_izq + x_der) // 2
        error         = float(centro_imagen - centro_carril)

        # ── line_detected: True si al menos un lado fue detectado sin fallback
        linea_detectada = (resultado_izq is not None) or (resultado_der is not None)

        # ── Publicar error ───────────────────────────────────────────────────
        error_msg      = Float32()
        error_msg.data = error
        self.pub_error.publish(error_msg)

        self.get_logger().info(
            f'Izq: {"OK" if resultado_izq else "fallback"} | '
            f'Der: {"OK" if resultado_der else "fallback"} | '
            f'Linea: {linea_detectada} | '
            f'Error: {error:.1f}px'
        )

        # ── Debug seguidor ───────────────────────────────────────────────────
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi, (x1, y1), (x2, y2), (255, 100, 0), 1)

        if resultado_izq is not None:
            x_b, x_t = resultado_izq
            cv2.line(roi, (x_b, roi_height), (x_t, int(roi_height * 0.6)), (0, 255, 0), 3)

        if resultado_der is not None:
            x_b, x_t = resultado_der
            cv2.line(roi, (x_b, roi_height), (x_t, int(roi_height * 0.6)), (0, 255, 0), 3)

        cv2.line(roi, (centro_carril, 0), (centro_carril, roi_height), (0, 255, 255), 2)
        cv2.line(roi, (centro_imagen, 0), (centro_imagen, roi_height), (0,   0, 255), 2)
        cv2.putText(roi, f'{"BUSCANDO" if not linea_detectada else f"Error: {error:.1f}px"}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Lane Detector", roi)

        cv2.imshow("Edges", edges_roi)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
