import json
from matplotlib.pyplot import figure
import cv2
from matplotlib import pyplot as plt
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def extract_acc_params(gt_path,pred_path,threshold = 0.7):
    results= {"circle":{},"triangle":{}}
    with open(gt_path, 'r') as myjson:
        file_read=myjson.read()
    gt_object = json.loads(file_read)

    with open(pred_path, 'r') as myjson:
        file_read=myjson.read()
    pred_object = json.loads(file_read)

    for key in results.keys():
        if key in gt_object.keys():
            gt_shape = gt_object[key]
            results[key]["poitives"] =len (gt_shape)
            if key in pred_object.keys():
                results[key]["tp"] = 0  
                pred_shape = pred_object[key]
                for one_gt in gt_shape:
                    gt_bbox = {"x1":one_gt[0][0],"x2":one_gt[1][0],"y1":one_gt[0][1],"y2":one_gt[1][1]}
                    for position, one_pred in enumerate(pred_shape):
                        pred_bbox = {"x1":one_pred[0][0],"x2":one_pred[1][0],"y1":one_pred[0][1],"y2":one_pred[1][1]}
                        iou = get_iou(gt_bbox,pred_bbox)
                        if iou>threshold:
                            results[key]["tp"]+=1
                            pred_object[key].pop(position)
                            break


            else:
                results[key]["tp"] = 0          
        else:
            results[key]["poitives"] = None

    for key in results.keys():
        if key in pred_object.keys():
            results[key]["fp"] = len(pred_object[key])
        else:
            results[key]["fp"] = 0

    return results



def metric(bb1, bb2):

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    bb1_x_mid = (bb1['x2'] + bb1['x1'])/2
    bb1_y_mid = (bb1['y2'] + bb1['y1'])/2
    bb2_x_mid = (bb2['x2'] + bb2['x1'])/2
    bb2_y_mid = (bb2['y2'] + bb2['y1'])/2
    bb1_width = (bb1['x2'] - bb1['x1'])
    bb1_height = (bb1['y2'] - bb1['y1'])
    bb2_width = (bb2['x2'] - bb2['x1'])
    bb2_height = (bb2['y2'] - bb2['y1'])

    
    mid_x_dist = (bb1_width - abs (bb1_x_mid - bb2_x_mid))/bb1_width 
    mid_x_dist = max(mid_x_dist,0)
    mid_y_dist = (bb1_height - abs (bb1_y_mid - bb2_y_mid))/bb1_height 
    mid_y_dist = max(mid_y_dist,0)
    width_ratio = min (bb1_width,bb2_width)/max (bb1_width,bb2_width) 
    height_ratio = min (bb1_height,bb2_height)/max (bb1_height,bb2_height)
    return width_ratio*height_ratio*mid_y_dist*mid_x_dist



def compare_iou_and_matric(gt_path,pred_path,threshold = 0.7):
    results= {"circle":{},"triangle":{}}
    with open(gt_path, 'r') as myjson:
        file_read=myjson.read()
    gt_object = json.loads(file_read)

    with open(pred_path, 'r') as myjson:
        file_read=myjson.read()
    pred_object = json.loads(file_read)

    for key in results.keys():
        if key in gt_object.keys():
            gt_shape = gt_object[key]
            results[key]["poitives"] =len (gt_shape)
            if key in pred_object.keys():
                results[key]["tp"] = 0  
                pred_shape = pred_object[key]
                for one_gt in gt_shape:
                    gt_bbox = {"x1":one_gt[0][0],"x2":one_gt[1][0],"y1":one_gt[0][1],"y2":one_gt[1][1]}
                    for position, one_pred in enumerate(pred_shape):
                        pred_bbox = {"x1":one_pred[0][0],"x2":one_pred[1][0],"y1":one_pred[0][1],"y2":one_pred[1][1]}
                        iou = get_iou(gt_bbox,pred_bbox)
                        met = metric(gt_bbox,pred_bbox)

                        if iou>threshold:
                            results[key]["tp"]+=1
                            
                            pred_object[key].pop(position)
                            break



    return 



def compare_iou_and_matric(gt_path,pred_path,img_path,threshold = 0.4):
    fig = plt.figure(figsize=(100, 80))

    num=1
    results= {"circle":{},"triangle":{}}
    with open(gt_path, 'r') as myjson:
        file_read=myjson.read()
    gt_object = json.loads(file_read)

    image = cv2.imread(img_path)
    with open(pred_path, 'r') as myjson:
        file_read=myjson.read()
    pred_object = json.loads(file_read)

    for key in results.keys():
        if key in gt_object.keys():
            gt_shape = gt_object[key]
            results[key]["poitives"] =len (gt_shape)
            if key in pred_object.keys():
                results[key]["tp"] = 0  
                pred_shape = pred_object[key]
                for one_gt in gt_shape:
                    gt_bbox = {"x1":one_gt[0][0],"x2":one_gt[1][0],"y1":one_gt[0][1],"y2":one_gt[1][1]}
                    for position, one_pred in enumerate(pred_shape):
                        pred_bbox = {"x1":one_pred[0][0],"x2":one_pred[1][0],"y1":one_pred[0][1],"y2":one_pred[1][1]}
                        iou = get_iou(gt_bbox,pred_bbox)
                        met = metric(gt_bbox,pred_bbox)

                        if iou>threshold:
                            img = image.copy()
                            start_point = tuple(one_pred[0])
                            end_point = tuple(one_pred[1])
                            color = (0, 255, 255)
                            thickness = 1
                            img = cv2.rectangle(img, start_point, end_point, color, thickness)
                            start_point = tuple(one_gt[0])
                            end_point = tuple(one_gt[1])
                            color = (255, 0, 0)
                            thickness = 1
                            img = cv2.rectangle(img, start_point, end_point, color, thickness)
                            fig.add_subplot(10, 1, num)
                            plt.title("iou: {} my metric: {}".format(iou,met))
                            plt.imshow(img)
                            num+=1
                            break

    plt.show()
