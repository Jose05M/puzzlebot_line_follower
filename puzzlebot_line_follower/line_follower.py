#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class LineFollower(Node):

    def __init__(self):

        super().__init__('line_follower')

        # PARAMETERS
        self.declare_parameter('camera_topic','/video_source/raw')
        self.declare_parameter('line_error_topic','/line_error')
        self.declare_parameter('debug_view',True)

        camera_topic = self.get_parameter('camera_topic').value
        error_topic = self.get_parameter('line_error_topic').value
        self.debug_view = self.get_parameter('debug_view').value

        # ROS
        self.sub = self.create_subscription(Image,camera_topic,self.image_callback,10)
        self.pub_error = self.create_publisher(Float32,error_topic,10)

        # INTERNAL STATE
        self.last_error = 0.0
        self.get_logger().info('Scanline Line Follower Started')

    # IMAGE CALLBACK
    def image_callback(self, msg):
        frame = np.frombuffer(msg.data,dtype=np.uint8).reshape((msg.height, msg.width, 3))

        frame = cv2.resize(frame,(640, 480))

        height, width = frame.shape[:2]

        # ROI
        crop_percent = 0.25
        crop_x = int(width * crop_percent)

        roi = frame[
            :,
            crop_x:width - crop_x
        ]
        roi_height, roi_width = roi.shape[:2]

        # PREPROCESSING
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(7, 7),0)
        _, binary = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # MORPHOLOGY
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,5))
        binary = cv2.erode(binary,kernel,iterations=3)
        binary = cv2.dilate(binary,kernel,iterations=3)

        # SCANLINES
        scan_rows = [int(roi_height * 0.80), int(roi_height * 0.84), int(roi_height * 0.88), int(roi_height * 0.92), int(roi_height * 0.96)]
        weights = [1,2,3,4,5]

        center_image = roi_width // 2
        errors = []
        valid_weights = []
        output = roi.copy()

        # PROCESS EACH ROW
        for i, y in enumerate(scan_rows):

            # Evitar filas fuera de rango
            if y >= roi_height:
                continue

            row_pixels = binary[y]
            white_pixels = np.where(row_pixels == 255)[0]

            if len(white_pixels) > 0:

                # dividir grupos de pixeles consecutivos
                segments = np.split(white_pixels,np.where(np.diff(white_pixels) > 10)[0] + 1)

                best_segment = None
                largest_size = 0

                for segment in segments:
                    segment_size = len(segment)

                    # ignorar ruido pequeño
                    if segment_size < 20:
                        continue

                    # elegir el más grande
                    if segment_size > largest_size:
                        largest_size = segment_size
                        best_segment = segment

                # si encontró segmento válido
                if best_segment is not None:
                    left = best_segment[0]
                    right = best_segment[-1]

                center_line = (left + right) // 2

                error = (center_image - center_line)
                errors.append(error)

                valid_weights.append(weights[i])

                # DEBUG
                if self.debug_view:

                    # línea horizontal
                    cv2.line(output,(0, y),(roi_width, y),(0, 255, 255),1)

                    # borde izquierdo
                    cv2.circle(output,(left, y),4,(0, 255, 0),-1)

                    # borde derecho
                    cv2.circle(output,(right, y),4,(255, 0, 0),-1)

                    # centro línea
                    cv2.circle(output,(center_line, y),4,(0, 0, 255),-1)

        # CALCULAR ERROR FINAL
        final_error = self.last_error

        if len(errors) > 0:

            final_error = np.average(errors,weights=valid_weights)

            # SMOOTHING
            alpha = 0.7
            final_error = (alpha * self.last_error +(1 - alpha) * final_error)
            self.last_error = final_error

        # PUBLICAR ERROR
        error_msg = Float32()
        error_msg.data = float(final_error)
        self.pub_error.publish(error_msg)

        # DEBUG
        if self.debug_view:

            # centro imagen
            cv2.line(output,(center_image, 0),(center_image, roi_height),(255, 255, 255),2)
            cv2.putText(output,f'Error: {final_error:.1f}',(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
            cv2.imshow("ROI",output)
            cv2.imshow("Binary",binary)
            cv2.imshow("real",frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()