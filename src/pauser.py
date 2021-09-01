from typing import Optional

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class BluetoothEStop:
    def __init__(self, button_idxs: Optional[list] = None) -> None:
        super().__init__()

        if button_idxs is None:
            button_idxs = [0, 1, 2, 3]

        self.button_idxs = button_idxs
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
        flip = False

        for i in self.button_idxs:
            if msg.buttons[i] == 1:
                flip = True
                break

        if flip:
            self.pause = not self.pause

    def wait_for_publisher(self):
        while not rospy.is_shutdown() and not self.pub.get_num_connections() == 1:
            # Wait until publisher gets connected
            pass

        print("Connected to Publisher")
