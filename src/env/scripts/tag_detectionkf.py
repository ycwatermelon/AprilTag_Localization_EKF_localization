#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3, TwistStamped
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter


# quaternion --> rotation matrix
def correct_quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])

def apriltag(x, y, z):
    p = PoseStamped()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z
    return p

apriltags = [apriltag(0, 0, 0),
             apriltag(4.5, 0, 3),
]
             
dt = 1.0 / 50
nx = 9
initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_P = [10**2] * 3 + [5**2] * 3 + [2**2] * 3
Q_scale = [0.5**2] * 3 + [0.3**2] * 3 + [0.2**2] * 3
R_scale = [8**2, 8**2, 8**2] + [4**2] * 3 + [2**2] * 3

kf_down = None
last_estimate_down = np.zeros((9, 1))
current_velocity = np.zeros(3)
current_acceleration = np.zeros(3)
initial_position_set = False


def initial_position_callback(msg):
    global kf_down, initial_position_set
    if not initial_position_set:
        initial_state = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            0, 0, 0,  # 初始速度設為 0
            0, 0, 0   # 初始加速度設為 0
        ]
        kf_down = kalman_filter(dt, nx, initial_state, initial_P, Q_scale, R_scale)
        initial_position_set = True

def process_tag_detection_down(detection):
    tag_id = detection.id[0]
    position = detection.pose.pose.pose.position
    orientation = detection.pose.pose.pose.orientation
    
    rel_pos_b = np.array([position.x, position.y, position.z])
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    R = correct_quaternion_to_rotation_matrix(q)
    rel_pos = np.dot(R, rel_pos_b)
    
    if tag_id < len(apriltags):
        tag_pos = np.array([apriltags[tag_id].pose.position.x,
                            apriltags[tag_id].pose.position.y,
                            apriltags[tag_id].pose.position.z])
        measurement = tag_pos - rel_pos
        return tag_id, measurement
    else:
        rospy.logwarn("Detected AprilTag with ID %d is out of range.", tag_id)
        return None, None


def tag_detections_callback_measurement_down(msg, pub):
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection_down(detection)
        if measurement is not None:
            measurement_msg = Vector3()
            measurement_msg.x = measurement[0]
            measurement_msg.y = measurement[1]
            measurement_msg.z = measurement[2]
            pub.publish(measurement_msg)


def tag_detections_callback_kalman_down(msg, pub):
    global kf_down, current_velocity, current_acceleration, last_estimate_down, initial_position_set
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection_down(detection)
        if measurement is not None:
            full_measurement = np.concatenate([measurement, current_velocity, current_acceleration])
            kf_down.predict()
            estimated_state, _ = kf_down.update(full_measurement.reshape(9, 1))
            last_estimate_down = estimated_state

            kalman_msg = Vector3()
            kalman_msg.x = estimated_state[0, 0]
            kalman_msg.y = estimated_state[1, 0]
            kalman_msg.z = estimated_state[2, 0]
            pub.publish(kalman_msg)


    def local_position_callback(data, pub):
        local_pos_msg = Vector3()
        local_pos_msg.x = data.pose.position.x
        local_pos_msg.y = data.pose.position.y
        local_pos_msg.z = data.pose.position.z
        pub.publish(local_pos_msg)

def velocity_callback(msg):
    global current_velocity
    current_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

def imu_callback(msg):
    global current_acceleration
    gravity = np.array([0, 0, 9.81])
    measured_acc = np.array([msg.linear_acceleration.x, 
                             msg.linear_acceleration.y, 
                             msg.linear_acceleration.z])
    current_acceleration = measured_acc - gravity

def main():
    global kf_down
    rospy.init_node('apriltag_measure_uav')
    kf_down = kalman_filter(dt, nx, initial_state, initial_P, Q_scale, R_scale)

    measurement_pub_down = rospy.Publisher('/uav/measurement/down', Vector3, queue_size=10)
    kalman_pub_down = rospy.Publisher('/uav/kalman/down', Vector3, queue_size=10)

    rospy.Subscriber('/ground_truth/state', Odometry, initial_position_callback)
    rospy.Subscriber('/camera_down/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback_measurement_down(msg, measurement_pub_down))
    rospy.Subscriber('/camera_down/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback_kalman_down(msg, kalman_pub_down))
    rospy.Subscriber('/mavros/imu/data', Imu, imu_callback)
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():    
        rate.sleep()
        
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
