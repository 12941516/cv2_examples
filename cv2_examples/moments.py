import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class Moments(Node):
    def __init__(self):
        super().__init__('moments_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
    
    # Gaussian Filter Added
    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src
    
    # Hsv Filter Added
    def redHsvInrange(self, src):
        red_lower_bound = np.array([0,150,150], dtype=np.uint8)
        red_upper_bound = np.array([20,255,255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, red_lower_bound, red_upper_bound)
        return hsv_dst
    
    # Moment-Center point Detector Added
    def findMoments(self, src, binary):
        centers = []
        contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 2000: continue
            m = cv2.moments(contour)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            cv2.circle(src, (cx,cy), 5, (255,100,100), -1)
            cv2.putText(src, f'({cx},{cy})', (cx,cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,100), 3)
        return src
    
    def videoSubscriber_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return
        
        # Use Gaussian & Hsv Filter
        gaussian_src = self.gaussianBlur(src)
        hsv_red_src = self.redHsvInrange(gaussian_src)
        moment_src = self.findMoments(src, hsv_red_src)
        cv2.imshow('gaussian', gaussian_src)
        cv2.imshow('hsv', hsv_red_src)
        cv2.imshow('moment', moment_src)
        
        cv2.imshow('video', src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = Moments()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
