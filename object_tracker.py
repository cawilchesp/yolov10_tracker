from ultralytics import YOLO
import supervision as sv

import sys
import torch
import cv2
import time
from pathlib import Path
import itertools

from imutils.video import FileVideoStream, WebcamVideoStream, FPS

import config
from tools.video_info import VideoInfo
from tools.messages import source_message, progress_message, step_message
from tools.write_csv import output_data_list, write_csv

# For debugging
from icecream import ic


def main(
    source: str = '0',
    output: str = 'output',
    weights: str = 'yolov10b.pt',
    class_filter: list[int] = [],
    image_size: int = 640,
    confidence: int = 0.5,
) -> None:
    # Initialize video source
    source_info, source_flag = VideoInfo.get_source_info(source)
    step_message(next(step_count), 'Origen del Video Inicializado ✅')
    source_message(source, source_info)

    # Check GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize YOLOv10 model
    model = YOLO(weights)
    step_message(next(step_count), f"{Path(weights).stem.upper()} Model Initialized ✅")

    # Initialize ByteTrack
    tracker = sv.ByteTrack()
    step_message(next(step_count), f"ByteTrack Initialized ✅")

    # show_image size
    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    scaled_height = int(scaled_width * source_info.height / source_info.width)
    scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # Outputs
    output_writer = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    # Annotators
    line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness)
    
    # Variables
    progress_times = {}
    results_data = []
    csv_path = None

    # Iniciar procesamiento de video
    step_message(next(step_count), 'Procesamiento de Video Iniciado ✅')
    
    if source_flag == 'stream':
        video_stream = WebcamVideoStream(src=eval(source) if source.isnumeric() else source)
    elif source_flag == 'video':
        video_stream = FileVideoStream(source)

    frame_number = 0
    video_stream.start()
    fps = FPS().start()
    try:
        while video_stream.more() if source_flag == 'video' else True:
            image = video_stream.read()
            if image is None:
                print()
                break

            annotated_image = image.copy()

            # YOLO inference
            results = model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                classes=class_filter,
                verbose=False
            )[0]

            # Processing inference results
            detections = sv.Detections.from_ultralytics(results).with_nms()
            
            # Updating ID with tracker
            detections = tracker.update_with_detections(detections)
                
            # Save object data in list
            results_data = output_data_list(results_data, frame_number, detections, results.names)

            # Draw labels
            object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=object_labels )

            # Draw boxes
            annotated_image = bounding_box_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
            # Draw tracks
            if detections.tracker_id is not None:
                annotated_image = trace_annotator.annotate(
                    scene=annotated_image,
                    detections=detections )

            # Save results
            output_writer.write(annotated_image)

            # Print progress
            progress_message(frame_number, source_info.total_frames)
            frame_number += 1

            # View live results
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n")
                break

            fps.update()

    except KeyboardInterrupt:
        step_message(next(step_count), 'Fin del video ✅')
    step_message(next(step_count), 'Guardando Resultados en el último CSV ✅')
    write_csv(f"{output}.csv", results_data)
    
    fps.stop()
    step_message(next(step_count), f"Elapsed Time: {fps.elapsed():.2f} s")
    step_message(next(step_count), f"FPS: {fps.fps():.2f}")
    
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.OUTPUT_FOLDER}/{config.OUTPUT_NAME}",
        weights=f"{config.MODEL_FOLDER}/{config.MODEL_WEIGHTS}",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
    )
