import csv
from supervision.detection.core import Detections
from icecream import ic

def output_append(output: list, frame_number: int, detections: Detections) -> list:
    """ Append object detection results to list """
    for xyxy, _, confidence, _, tracker_id, data in detections:
        x = xyxy[0]
        y = xyxy[1]
        w = xyxy[2]-xyxy[0]
        h = xyxy[3]-xyxy[1]
        output.append([frame_number, tracker_id, data['class_name'], x, y, w, h, confidence])

    return output


def write_csv(save_path: str, data: list) -> None:
    """
    Write object detection results in csv file
    """
    with open(save_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
        