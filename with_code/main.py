# import libraries
from ultralytics import YOLO
import cv2
import collections
# It downloads the model automatically. If it is not already downloaded
# I choose the medium model you can choose others. Such as, yolov8n, yolov8s etc.
model = YOLO("yolov8m.pt")


# Automatic tracking image
def automatic_tracking_img(img_pth):
    # Open the img
    img = cv2.imread(img_pth)

    # Give img to the model (tracker=bytetrack.yaml or botsort.yaml)
    result = model.track(img, persist=True, agnostic_nms=True, tracker="bytetrack.yaml")

    # Get result frame
    annotated_img = result[0].plot()

    # Save image
    cv2.imwrite("../datas/test_img_result_automatic.jpg", annotated_img)


# Customized tracking image
def customized_tracking_img(img_pth):
    # Open the img
    img = cv2.imread(img_pth)

    # Give img to the model (tracker=bytetrack.yaml or botsort.yaml)
    result = model.track(img, persist=True, agnostic_nms=True, tracker="bytetrack.yaml")

    # Get boxes
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)

    # Get indexes of object names
    clss = result[0].boxes.cls

    # Get ids of objects
    ids = result[0].boxes.id.cpu().numpy().astype(int)

    for box, cls, id in zip(boxes, clss, ids):
        # Get name of object
        name = result[0].names[int(cls)]

        # Draw Rectangle
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        # Find center of object
        center_of_object = (int((box[2] - box[0]) / 2) + box[0], int((box[3] - box[1]) / 2) + box[1])

        # Put a dot to center of object
        cv2.circle(img, (center_of_object), 5, (0, 255, 0), cv2.FILLED)

        # Write names, center points and ids
        cv2.putText(img, f"id:{id}-{name}-{center_of_object}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

    # Save Result
    cv2.imwrite("../datas/test_img_result_customized.jpg", img)


# Automatic tracking video
def automatic_tracking_video(video_pth):
    # Set Video
    cap = cv2.VideoCapture(video_pth)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set video writer
    writer = cv2.VideoWriter('../datas/test_video_result_automatic.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    # Start
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error starting camera")
            break

        try:
            # Give img to the model (tracker=bytetrack.yaml or botsort.yaml)
            result = model.track(frame, persist=True, agnostic_nms=True, tracker="bytetrack.yaml")

            # Get result frame
            frame = result[0].plot()

        except Exception as e:
            print(e)

        # Show video
        cv2.imshow("frame", frame)

        # Write frames for a video
        writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close video writer and video
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# Customized tracking video
def customized_tracking_video(video_pth):
    # Set Video
    cap = cv2.VideoCapture(video_pth)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set video writer
    writer = cv2.VideoWriter('../datas/test_video_result_customized.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    # Start
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error starting camera")
            break

        try:
            # Give img to the model (tracker=bytetrack.yaml or botsort.yaml)
            result = model.track(frame, persist=True, agnostic_nms=True, tracker="bytetrack.yaml")

            # Get boxes
            boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)

            # Get indexes of object names
            clss = result[0].boxes.cls

            # Get ids of objects
            ids = result[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, id in zip(boxes, clss, ids):
                # Get name of object
                name = result[0].names[int(cls)]

                # Draw Rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                # Find center of object
                center_of_object = (int((box[2] - box[0]) / 2) + box[0], int((box[3] - box[1]) / 2) + box[1])

                # Put a dot to center of object
                cv2.circle(frame, (center_of_object), 5, (0, 255, 0), cv2.FILLED)

                # Write names, center points and ids
                cv2.putText(frame, f"id:{id}-{name}-{center_of_object}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)

        except Exception as e:
            print(e)

        # Show video
        cv2.imshow("frame", frame)

        # Write frames for a video
        writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close video writer and video
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# Customized tracking line video
def customized_tracking_line_video(video_pth):
    # Create list that have fixed size
    line_list = collections.deque(maxlen=50)

    # Set Video
    cap = cv2.VideoCapture(video_pth)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set video writer
    writer = cv2.VideoWriter('../datas/test_video_result_line_customized.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    # Start
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error starting camera")
            break

        try:
            # Give img to the model (tracker=bytetrack.yaml or botsort.yaml)
            result = model.track(frame, persist=True, agnostic_nms=True, tracker="bytetrack.yaml")

            # Get boxes
            boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)

            # Get indexes of object names
            clss = result[0].boxes.cls

            # Get ids of objects
            ids = result[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, id in zip(boxes, clss, ids):
                # Get name of object
                name = result[0].names[int(cls)]

                # Draw Rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                # Find center of object
                center_of_object = (int((box[2] - box[0]) / 2) + box[0], int((box[3] - box[1]) / 2) + box[1])

                # Add center points to list
                line_list.append(center_of_object)

                # Put a dots of lines
                for points in line_list:
                    cv2.circle(frame, (points), 3, (0, 255, 0), cv2.FILLED)

                # Write names, center points and ids
                cv2.putText(frame, f"id:{id}-{name}-{center_of_object}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)

        except Exception as e:
            print(e)

        # Show video
        cv2.imshow("frame", frame)

        # Write frames for a video
        writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close video writer and video
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
img_pth = "../datas/test_img.jpg"

automatic_tracking_img(img_pth)
customized_tracking_img(img_pth)

video_pth = "../datas/test_video.mp4"

automatic_tracking_video(video_pth)
customized_tracking_video(video_pth)
customized_tracking_line_video(video_pth)