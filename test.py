from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

robot_ip = "192.168.5.5"  # Replace with your UR5e IP address

rtde_c = RTDEControl(robot_ip)
rtde_r = RTDEReceive(robot_ip)

# --- Square parameters ---
side = 0.1          # side length [m]
height = 0.3        # Z height above base, parallel to XY plane [m]
center_x = 0.4      # square center X [m]
center_y = 0.0      # square center Y [m]

# Keep current tool orientation (Rx, Ry, Rz) so the TCP stays pointing the same way
current_pose = rtde_r.getActualTCPPose()
rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]

h = side / 2.0

# Four corners of the square (closed loop: return to start)
corners = [
    [center_x - h, center_y - h, height, rx, ry, rz],
    [center_x + h, center_y - h, height, rx, ry, rz],
    [center_x + h, center_y + h, height, rx, ry, rz],
    [center_x - h, center_y + h, height, rx, ry, rz],
    [center_x - h, center_y - h, height, rx, ry, rz],  # close the square
]

velocity = 0.1      # [m/s]
acceleration = 0.3  # [m/s^2]

# Move to first corner, then trace the square
for pose in corners:
    rtde_c.moveL(pose, velocity, acceleration)

rtde_c.disconnect()