from typing import Optional

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class BluetoothEStop:
    def __init__(self, stop_button_idx=0, start_button_idx=3) -> None:
        super().__init__()

        self.start_button_idx = start_button_idx
        self.stop_button_idx = stop_button_idx

        self.sub = rospy.Subscriber("/bluetooth_teleop/joy", Joy, self.callback)
        self.pub = rospy.Publisher("/e_stop", Bool, queue_size=10)
        self._pause = False
        self._pause_object = Bool(self._pause)

    @property
    def pause(self):
        return self._pause

    @pause.setter
    def pause(self, new_pause):
        self._pause = new_pause
        self._pause_object.data = self._pause
        print("Setting ESTOP:", self._pause)
        self.pub.publish(self._pause_object)

    def callback(self, msg: Joy):
        if msg.buttons[self.stop_button_idx]:
            self.pause = True
        elif msg.buttons[self.start_button_idx]:
            self.pause = False

    def wait_for_publisher(self):
        while not rospy.is_shutdown() and not self.pub.get_num_connections() == 1:
            # Wait until publisher gets connected
            pass

        print("Connected to Publisher")
