import os
import shutil

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util import event_pb2
from tensorflow.python.framework.errors_impl import DataLossError
from tensorflow.python.lib.io import tf_record

summary_dir = '../runs/'


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


directories = os.listdir(summary_dir)
directories.sort()

for folder_name in directories:
    path = os.path.join(summary_dir, folder_name)

    event_file = os.listdir(path)

    for e in event_file:
        if e.startswith('events'):
            event_file = e
            break

    event_file = os.path.join(path, event_file)

    print(path)

    counter = 0
    error = False

    num_data_threshold = 25

    exit_ = False

    try:
        for event in my_summary_iterator(event_file):
            summary: Summary = event.summary

            for value in summary.value:
                # pass
                if value.tag == "Error/Dist Error MAE":
                    # print(event)
                    # print(value.simple_value)
                    counter += 1
                if counter > num_data_threshold:
                    exit_ = True
                    break
            if exit_:
                break
    except DataLossError as e:
        print("Data loss", e)
        error = True

    print("Count", counter)

    if (not error and counter < num_data_threshold) or (error and counter <= num_data_threshold * .1):
        shutil.rmtree(path)
        print("REMOVED")
