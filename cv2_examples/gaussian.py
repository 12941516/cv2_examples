import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class Gaussian(Node):
    def __init__(self):
        super().__init__('gaussian_node')
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
    
    def videoSubscriber_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return
        
        # Use Gaussian Filter
        gaussian_src = self.gaussianBlur(src)
        cv2.imshow('gaussian', gaussian_src)
        
        cv2.imshow('video', src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = Gaussian()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
