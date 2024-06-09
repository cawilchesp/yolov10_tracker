from ultralytics import YOLO
import supervision as sv

import sys
import torch
import cv2
import time
from pathlib import Path
import itertools

import config
from tools.video_info import VideoInfo
from tools.messages import source_message, progress_message, step_message

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

    quit()



    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        scaled_height = int(scaled_width * source_info.height / source_info.width)
        scaled_height = scaled_height if source_info.height > scaled_height else source_info.height



    # Annotators
    line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)
    
    # Variables
    progress_times = {}
    results_data = []
    csv_path = None
    video_path = None
    output_writer = None
    source_writer = None
    time_data = []

    # Iniciar procesamiento de video
    step_message(next(step_count), 'Procesamiento de Video Iniciado ✅')
    
    if source_flag == 'stream':
        video_stream = WebcamVideoStream(video_source)
    elif source_flag == 'video':
        video_stream = FileVideoStream(source)

    frame_number = 0
    video_stream.start()
    fps = FPS().start()
    try:
        while video_stream.more() if source_flag == 'video' else True:
            t_frame_start = time_synchronized()
            image = video_stream.read()
            if image is None:
                print()
                break
            t_capture_end = time_synchronized()

            annotated_image = image.copy()

            t_inference_start = time_synchronized()
            # YOLO inference
            results = model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                classes=class_filter,
                verbose=False
            )[0]
            t_inference_end = time_synchronized()

            # Processing inference results
            detections = sv.Detections.from_ultralytics(results)
            detections = detections.with_nms()
            
            # Updating ID with tracker
            detections = tracker.update_with_detections(detections)
                
            # Save object data in list
            results_data = output_data_list(results_data, frame_number, detections, results.names)

            if show_image or save_video:
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
            if save_csv or save_video or save_source:
                # Output file name from source type
                if clip_length == 0:
                    save_path = f'{Path(output)}'
                elif frame_number % clip_length == 0:
                    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
                    save_path = f'{Path(output)}_{timestamp}'
                
                # New video
                if video_path != save_path:
                    video_path = save_path
                    if save_csv:
                        if csv_path is not None and clip_length > 0:
                            step_message(next(step_count), 'Guardando Resultados en CSV')
                            write_csv(f"{csv_path}.csv", results_data)
                            results_data = []
                        csv_path = save_path
                    if save_video:
                        if isinstance(output_writer, cv2.VideoWriter):
                            output_writer.release()
                        output_writer = cv2.VideoWriter(f"{video_path}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))
                    if save_source:
                        if isinstance(source_writer, cv2.VideoWriter):
                            source_writer.release()
                        source_writer = cv2.VideoWriter(f"{video_path}_source.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))
                if save_video: output_writer.write(annotated_image)
                if save_source: source_writer.write(image)

            t_frame_end = time_synchronized()

            # Print progress
            if time_measure:
                progress_times['capture_time'] = t_capture_end - t_frame_start
                progress_times['inference_time'] = t_inference_end - t_inference_start
                progress_times['frame_time'] = t_frame_end - t_frame_start
                time_data.append([frame_number, progress_times['capture_time'], progress_times['inference_time'], progress_times['frame_time']])
                print_times(frame_number, source_info.total_frames, progress_times)
            else:
                print_progress(frame_number, source_info.total_frames)
            frame_number += 1

            # View live results
            if show_image:
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
    write_csv(f"{csv_path}.csv", results_data)
    
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
