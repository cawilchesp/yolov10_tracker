import csv
from supervision.detection.core import Detections
from icecream import ic

def output_data_list(output: list, frame_number: int, data: Detections, class_names: dict) -> list:
    """ Append object detection results to list """
    for xyxy, _, confidence, class_id, tracker_id, _ in data:
        x = int(xyxy[0])
        y = int(xyxy[1])
        w = int(xyxy[2]-xyxy[0])
        h = int(xyxy[3]-xyxy[1])
        if tracker_id is not None:
            output.append([frame_number, tracker_id, class_names[class_id], x, y, w, h, confidence])

    return output


def write_csv(save_path: str, data: list) -> None:
    """
    Write object detection results in csv file
    """
    with open(save_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
        