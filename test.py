from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import math

robot_ip = "192.168.5.5"  # Replace with your robot IP

rtde_c = RTDEControl(robot_ip)
rtde_r = RTDEReceive(robot_ip)

# Home joint configuration (degrees)
home_deg = [
    120,   # Base
    -90,   # Shoulder
    90,    # Elbow
    -90,   # Wrist 1
    -90,   # Wrist 2
    90     # Wrist 3
]

# Convert degrees to radians
home_rad = [math.radians(angle) for angle in home_deg]

# Motion parameters
speed = 1.0        # rad/s
acceleration = 1.2 # rad/s^2

print("Moving robot to home position...")
rtde_c.moveJ(home_rad, speed, acceleration)

# Print final joint positions
actual_joints = rtde_r.getActualQ()
print("Current joint positions (deg):")
print([round(math.degrees(j), 2) for j in actual_joints])

rtde_c.disconnect()