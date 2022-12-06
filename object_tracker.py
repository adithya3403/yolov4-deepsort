import math
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416','path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'MP4V', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


def main(_argv):
    midpoints = []
    midpoint_dict = {}
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = './model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('\n\n\nFrame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(
                count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # x,y coordinates of the center of the frame
        centerVideo = (int(frame.shape[1]/2), int(frame.shape[0]/2))

        # rectangle in the center with 50% of the frame size
        centerRectSize = (int(centerVideo[0] - frame.shape[1]/4), int(centerVideo[1] - frame.shape[0]/4)) , (int(centerVideo[0] + frame.shape[1]/4), int(centerVideo[1] + frame.shape[0]/4))

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                cv2.rectangle(frame, centerRectSize[0], centerRectSize[1], (255, 0, 0), 2)
                cv2.putText(frame, text="noObject", org=centerRectSize[0], fontFace=0, fontScale=0.75, color=(255,0,0), thickness=2)
                print("No object detected")
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

        # draw bbox on screen and plot its midpoint
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(
                bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),
                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)
            midpoint = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
            cv2.circle(frame, midpoint, 2, (0, 255, 0), 2)
            # write 4 directions north south east west on the screen
            cv2.putText(frame, "N", (centerVideo[0], 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "S", (centerVideo[0], frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "E", (frame.shape[1]-50, centerVideo[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "W", (50, centerVideo[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # check if the whole object is inside the rectangle
            if ((int(bbox[0])) < centerRectSize[0][0] or (int(bbox[1])) < centerRectSize[0][1] or (int(bbox[2])) > centerRectSize[1][0] or (int(bbox[3])) > centerRectSize[1][1]):
                cv2.rectangle(frame, centerRectSize[0], centerRectSize[1], (255, 0, 0), 2)
                cv2.putText(frame, text="notLocked", org=centerRectSize[0], fontFace=0, fontScale=0.75, color=(255,0,0), thickness=2)
                print()
            else : 
                cv2.rectangle(frame, centerRectSize[0], centerRectSize[1], (0, 255, 0), 2)
                cv2.putText(frame, text="Locked", org=centerRectSize[0], fontFace=0, fontScale=0.75, color=(0,255,0), thickness=2)
            # draw x and y axes
            cv2.line(frame, (0, centerVideo[1]), (frame.shape[1], centerVideo[1]), (255, 0, 0), 2)
            cv2.line(frame, (centerVideo[0], 0), (centerVideo[0], frame.shape[0]), (255, 0, 0), 2)
            # draw line joining the center of the frame and the midpoint of the object
            cv2.line(frame, centerVideo, midpoint, (0, 255, 0), 2)
            # check if the object is in the north, south, east or west of the frame
            if (midpoint[0] < centerVideo[0] and midpoint[1] < centerVideo[1]):
                dir="North-West"
            elif (midpoint[0] < centerVideo[0] and midpoint[1] > centerVideo[1]):
                dir="South-West"
            elif (midpoint[0] > centerVideo[0] and midpoint[1] < centerVideo[1]):
                dir="North-East"
            elif (midpoint[0] > centerVideo[0] and midpoint[1] > centerVideo[1]):
                dir="South-East"
            elif (midpoint[0] == centerVideo[0] and midpoint[1] < centerVideo[1]):
                dir="North"
            elif (midpoint[0] == centerVideo[0] and midpoint[1] > centerVideo[1]):
                dir="South"
            elif (midpoint[0] < centerVideo[0] and midpoint[1] == centerVideo[1]):
                dir="West"
            elif (midpoint[0] > centerVideo[0] and midpoint[1] == centerVideo[1]):
                dir="East"
            cv2.putText(frame, dir, (centerVideo[0], centerVideo[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # start joining the midpoints of the bounding boxes
            
            # print lat long of the object
            lat_drone = 0.000000000
            long_drone = 0.000000000
            latit = midpoint[0] * (90 / frame.shape[1]) - 90 + lat_drone
            longit = midpoint[1] * (180 / frame.shape[0]) - 180 + long_drone
            cv2.putText(frame, "Lat: " + str(latit), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Long: " + str(longit), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # if track.track_id not in midpoint_dict:
            #     midpoint_dict[track.track_id] = [midpoint]
            # else:
            #     midpoint_dict[track.track_id].append(midpoint)
            # if len(midpoint_dict[track.track_id]) > 1:
            #     for i in range(len(midpoint_dict[track.track_id])-1):
            #         # let color be green
            #         cv2.line(frame, midpoint_dict[track.track_id][i], midpoint_dict[track.track_id]
            #                  [i+1], (0, 255, 0), 2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            def findMidPoint(x1, y1, x2, y2):
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                return x, y
            x, y = findMidPoint(int(bbox[0]), int(
                bbox[1]), int(bbox[2]), int(bbox[3]))
            points = [x, y]
            midpoints.append(points)
            length = len(midpoints)
            if length > 2:
                # euclidean distance between two points
                print("Previous coordinate: ", midpoints[length-2])
                print("Current coordinate: ", midpoints[length-1])

                def findDistance(p1, p2):
                    # find 
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distance = findDistance(
                    midpoints[length-1], midpoints[length-2])
                print("Distance: ", "{:.2f}".format(distance))
                # if midpoints[length-1][0] > midpoints[length-2][0]:
                #     print("Direction: Right")
                # else:
                #     print("Direction: Left")

                # the coordinates of the object is given in the form of (x,y)
                # the coordinates of the drone are at the centre of the frame
                def findAngle(p1, p2):
                    dx=p2[0]-p1[0]
                    dy=p2[1]-p1[1]
                    theta=math.atan2(dy,dx)
                    angle=math.degrees(theta)
                    if angle<0:
                        angle=360+angle
                    return angle
                angle=findAngle(centerVideo, midpoints[length-1])
                # print("Angle: ", "{:.2f}".format(angle))
                # give all 8 directions
                # print the direction on the screen at the top left
                # keep in red color
                if angle > 337.5 or angle <= 22.5:
                    dir="North"
                elif angle > 22.5 and angle <= 67.5:
                    dir="North East"
                elif angle > 67.5 and angle <= 112.5:
                    dir="East"
                elif angle > 112.5 and angle <= 157.5:
                    dir="South-East"
                elif angle > 157.5 and angle <= 202.5:
                    dir="South"
                elif angle > 202.5 and angle <= 247.5:
                    dir="South-West"
                elif angle > 247.5 and angle <= 292.5:
                    dir="West"
                elif angle > 292.5 and angle <= 337.5:
                    dir="North-West"
                cv2.putText(frame, "Angle: "+str("{:.2f}".format(angle)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 2)
                # cv2.putText(frame, "moving towards: " + dir +" angle: "+{".2f"}.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # plt.plot(midpoints[length-1][0], midpoints[length-1][1], 'ro')
                # plt.plot(midpoints[length-2][0], midpoints[length-2][1], 'ro')
                # plt.plot([midpoints[length-1][0], midpoints[length-2][0]], [
                #     midpoints[length-1][1], midpoints[length-2][1]], 'r-')
                # plt.axis([0, 1020, 0, 1080])
                # plt.pause(0.05)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # plt.show()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
