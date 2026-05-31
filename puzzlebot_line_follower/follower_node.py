#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class Follower_Node(Node):

    def __init__(self):
        super().__init__('follower_node')

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

        # Kernels
        self.kernel = np.ones((3, 3), np.uint8)

        self.sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        self.get_logger().info('Hough Lane Detector Started')

    # ---------------------------------------------------------
    # PROMEDIO DE LÍNEAS
    # ---------------------------------------------------------

    def promedio_linea(self, lines, height):
        xs = []
        ys = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            xs += [x1, x2]
            ys += [y1, y2]

        if len(xs) == 0:
            return None

        m, b = np.polyfit(ys, xs, 1)

        y_bottom = height
        y_top    = int(height * 0.4)

        x_bottom = int(m * y_bottom + b)
        x_top    = int(m * y_top + b)

        return (x_bottom, x_top)

    # ---------------------------------------------------------
    # IMAGE CALLBACK
    # ---------------------------------------------------------

    def image_callback(self, msg):

        # ROS -> OpenCV
        frame = np.frombuffer(msg.data,dtype=np.uint8).reshape((msg.height, msg.width, 3))
        frame = cv2.resize(frame, (640, 480))

        height, width = frame.shape[:2]

        # ---------------------------------------------------------
        # ROI RECTANGULAR
        # ---------------------------------------------------------
        roi_y = frame[int(height * 0.4):, :]
        crop_percent = 0.20
        crop_x = int(width * crop_percent)
        roi = roi_y[:, crop_x:width - crop_x]
        roi_height, roi_width = roi.shape[:2]

        # ---------------------------------------------------------
        # PREPROCESAMIENTO
        # ---------------------------------------------------------

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpened = cv2.filter2D(gray,-1,self.sharpen_kernel)
        blurred = cv2.GaussianBlur(sharpened,(7, 7),0)
        _, binary_inv = cv2.threshold(blurred,100,255,cv2.THRESH_BINARY_INV)

        # MORPH
        morph = cv2.erode(binary_inv,self.kernel,iterations=2)
        morph = cv2.dilate(morph,self.kernel,iterations=2)

        # CANNY
        edges = cv2.Canny(morph,70,200)

        # ---------------------------------------------------------
        # HOUGH
        # ---------------------------------------------------------

        lines = cv2.HoughLinesP(edges,1,np.pi / 180,threshold=40,minLineLength=50,maxLineGap=40)

        left_lines  = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # LONGITUD
                length = np.hypot(x2 - x1,y2 - y1)
                if length < 40:
                    continue
                # ÁNGULO
                angle = np.degrees(np.arctan2(y2 - y1,x2 - x1))

                # IGNORAR HORIZONTALES
                if abs(angle) < 20:
                    continue

                # CLASIFICACIÓN
                if angle < 0:
                    left_lines.append(line)
                else:
                    right_lines.append(line)

        # ---------------------------------------------------------
        # PROMEDIO DE CARRIL
        # ---------------------------------------------------------
        center_image = roi_width // 2
        result_left = self.promedio_linea(left_lines,roi_height)
        result_right = self.promedio_linea(right_lines,roi_height)

        x_left = result_left[0] if result_left else 0
        x_right = result_right[0] if result_right else roi_width

        lane_center = (x_left + x_right) // 2

        error = float(center_image - lane_center)

        # ---------------------------------------------------------
        # SMOOTHING
        # ---------------------------------------------------------

        alpha = 0.7

        error = (
            alpha * self.last_error +
            (1 - alpha) * error
        )

        self.last_error = error

        # ---------------------------------------------------------
        # PUBLICAR ERROR
        # ---------------------------------------------------------

        error_msg = Float32()
        error_msg.data = error

        self.pub_error.publish(error_msg)

        # ---------------------------------------------------------
        # DEBUG
        # ---------------------------------------------------------

        if self.debug_view:

            # Dibujar líneas detectadas
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(roi,(x1, y1),(x2, y2),(255, 100, 0),1)

            # Línea izquierda
            if result_left is not None:
                xb, xt = result_left
                cv2.line(roi,(xb, roi_height),(xt, int(roi_height * 0.4)),(0, 255, 0),3)

            # Línea derecha
            if result_right is not None:
                xb, xt = result_right
                cv2.line(roi,(xb, roi_height),(xt, int(roi_height * 0.4)),(0, 255, 0),3)

            # Centro carril
            cv2.line(roi,(lane_center, 0),(lane_center, roi_height),(0, 255, 255),2)

            # Centro imagen
            cv2.line(roi,(center_image, 0),(center_image, roi_height),(0, 0, 255),2)

            cv2.putText(roi,f'Error: {error:.1f}px',(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)

            cv2.imshow("Lane Detector", roi)

            cv2.imshow("Binary", morph)

            cv2.imshow("Edges", edges)

            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Follower_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()