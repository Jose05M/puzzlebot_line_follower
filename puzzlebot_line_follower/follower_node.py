import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32, String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Follower(Node):
    def __init__(self):
        super().__init__('follower_node')

        # ── Máscara trapezoidal (ROI inferior) ───────────────────────────────
        self.mask = np.zeros((120, 320), dtype=np.uint8)
        top_width = int(320 * 0.9)
        trapezoid = np.array([[((320 - top_width) // 2, 0),
                               ((320 + top_width) // 2, 0),
                               (320, 120),
                               (0, 120)]], dtype=np.int32)
        cv2.fillPoly(self.mask, trapezoid, 255)

        self.bridge = CvBridge()
        self.sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.kernel = np.ones((3, 3), np.uint8)

        # ── Configuración semáforo ───────────────────────────────────────────
        self.config_hsv = {
            'ROJO':     [((0,   100, 100), (10,  255, 255)),
                         ((160, 100, 100), (180, 255, 255))],
            'AMARILLO': [((10,  100, 100), (45,  255, 255))],
            'VERDE':    [((46,  100, 100), (80,  255, 255))],
        }
        self.MIN_AREA = 170
        self.MAX_AREA = 1000

        # ── Subscribers & Publishers ─────────────────────────────────────────
        self.sub_video     = self.create_subscription(Image, '/video_source/raw', self.callback_video, 10)
        self.pub_error     = self.create_publisher(Float32, 'error',             10)
        self.pub_semaforo  = self.create_publisher(String,  'estado_semaforo',   10)
        self.pub_linea     = self.create_publisher(Bool,    'line_detected',      10)
        self.pub_debug     = self.create_publisher(Image,   'follower/debug',    10)
        self.pub_edges     = self.create_publisher(Image,   'follower/edges',    10)
        self.pub_debug_sem = self.create_publisher(Image,   'follower/semaforo', 10)

        self.get_logger().info('Nodo de Visión iniciado')

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

    def detectar_semaforo(self, roi_superior):
        hsv = cv2.cvtColor(roi_superior, cv2.COLOR_BGR2HSV)
        for color, rangos in self.config_hsv.items():
            mask_color = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in rangos:
                mask_color = cv2.add(mask_color, cv2.inRange(hsv, np.array(lower), np.array(upper)))
            contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if self.MIN_AREA < area < self.MAX_AREA:
                (x, y), radio = cv2.minEnclosingCircle(c)
                return color, (int(x), int(y)), int(radio)
        return 'NINGUNO', None, None

    # ── Callback principal ───────────────────────────────────────────────────
    def callback_video(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img = cv2.resize(img, (320, 240))
        img = cv2.rotate(img, cv2.ROTATE_180)

        height, width = img.shape[:2]

        # ── ROI superior: ancho completo, 60% superior ────────────────────────
        roi_sup_h    = int(height * 0.6)
        roi_superior = img[:roi_sup_h, :]

        # ── ROI inferior: 50% inferior para seguimiento de línea ─────────────
        offset_y   = int(height * 0.5)
        roi        = img[offset_y:, :]
        roi_height, roi_width = roi.shape[:2]

        # ════════════════════════════════════════════════════════════════════
        # SEMÁFORO
        # ════════════════════════════════════════════════════════════════════
        estado, centro_circ, radio_circ = self.detectar_semaforo(roi_superior)

        msg_sem      = String()
        msg_sem.data = estado
        self.pub_semaforo.publish(msg_sem)

        debug_sem  = roi_superior.copy()
        color_vis  = {'ROJO': (0, 0, 255), 'AMARILLO': (0, 255, 255), 'VERDE': (0, 255, 0)}
        if centro_circ is not None:
            cv2.circle(debug_sem, centro_circ, radio_circ, color_vis[estado], 2)
            cv2.circle(debug_sem, centro_circ, 3, color_vis[estado], -1)
        cv2.putText(debug_sem, f'Sem: {estado}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self.pub_debug_sem.publish(self.bridge.cv2_to_imgmsg(debug_sem, "bgr8"))

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
        msg_linea      = Bool()
        msg_linea.data = linea_detectada
        self.pub_linea.publish(msg_linea)

        # ── Publicar error ───────────────────────────────────────────────────
        error_msg      = Float32()
        error_msg.data = error
        self.pub_error.publish(error_msg)

        self.get_logger().info(
            f'[Sem: {estado}] | '
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

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(roi,   "bgr8"))
        self.pub_edges.publish(self.bridge.cv2_to_imgmsg(edges, "mono8"))

def main(args=None):
    rclpy.init(args=args)
    nodo = Follower()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()