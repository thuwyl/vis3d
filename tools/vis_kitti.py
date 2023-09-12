from curses import keyname
import cv2
import os
import open3d
import argparse
import numpy as np
from dataset.kitti.kitti_dataset import KITTIDataset
from tools.utils import draw_3dframeworks, draw_2dframeworks


parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=None, help='index for the label data', required=True)
parser.add_argument('--output', type=str, default="/home/wyl/ws/vis3d/result/imgs", help='')
args = parser.parse_args()
print(args)
dataset = KITTIDataset(
    dataset_path="/home/wyl/ws/vis3d/data/kitti",
    split="training"
    )

vis = open3d.visualization.Visualizer()
vis.create_window(window_name="3d detection", width=771, height=867)
vis.get_render_option().point_size = 3
opt = vis.get_render_option()
opt.background_color = np.array([1,1,1])


index = args.index
max_num=7481
while True:
    objs, preds_baseline, preds_sgfnet, points, calib, image= dataset.get_objs(int(dataset.valsets[index]))


    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([0, 0, 255/255])
    # vis.add_geometry(pcd)

    # vis gt
    img_gt = image.copy()
    img_pred_baseline = image.copy()
    img_pred_sgfnet = image.copy()
    thickness = 2
    for obj in objs:
        if obj.name == "Car":
            draw_2dframeworks(img_gt, obj.corner_2d, color=[255, 0, 0], thickness=thickness)
        if obj.name == "Pedestrian":
            draw_2dframeworks(img_gt, obj.corner_2d, color=[0, 0, 255], thickness=thickness)
        if obj.name == "Cyclist":
            draw_2dframeworks(img_gt, obj.corner_2d, color=[0, 255, 0], thickness=thickness)

    for obj in preds_baseline:
        if obj.name == "Car" and obj.ioc > 0.3:
            draw_2dframeworks(img_pred_baseline, obj.corner_2d, color=[255, 0, 0], thickness=thickness)
        if obj.name == "Pedestrian" and obj.ioc > 0.3:
            draw_2dframeworks(img_pred_baseline, obj.corner_2d, color=[0, 0, 255], thickness=thickness)
        if obj.name == "Cyclist" and obj.ioc > 0.3:
            draw_2dframeworks(img_pred_baseline, obj.corner_2d, color=[0, 255, 0], thickness=thickness)

    for obj in preds_sgfnet:
        if obj.name == "Car" and obj.ioc > 0.3:
            draw_2dframeworks(img_pred_sgfnet, obj.corner_2d, color=[255, 0, 0], thickness=thickness)
        if obj.name == "Pedestrian" and obj.ioc > 0.3:
            draw_2dframeworks(img_pred_sgfnet, obj.corner_2d, color=[0, 0, 255], thickness=thickness)
        if obj.name == "Cyclist" and obj.ioc > 0.3:
            draw_2dframeworks(img_pred_sgfnet, obj.corner_2d, color=[0, 255, 0], thickness=thickness)

    # for obj in objs:            # blue
    #     if obj.name == "Car" or obj.name == "Pedestrian" or obj.name == "Cyclist":
    #         # draw_3dframeworks(vis, obj.corner_3d)
    #         draw_2dframeworks(image, obj.corner_2d, color=[255, 0, 0], thickness=1)
    
    # for obj in preds_baseline:  # red
    #     if obj.name == "Car" or obj.name == "Pedestrian" or obj.name == "Cyclist":
    #         if obj.ioc > 0.3:
    #             # draw_3dframeworks(vis, obj.corner_3d)
    #             draw_2dframeworks(image, obj.corner_2d, color=[0, 0, 255], thickness=1)
        
    # for obj in preds_sgfnet:    # green
    #     if obj.name == "Car" or obj.name == "Pedestrian" or obj.name == "Cyclist":
    #         if obj.ioc > 0.3:
    #             # draw_3dframeworks(vis, obj.corner_3d)
    #             draw_2dframeworks(image, obj.corner_2d, color=[0, 255, 0], thickness=1)

    cv2.imshow("3dbox_img_gt", img_gt)
    cv2.imshow("3dbox_img_pred_baseline", img_pred_baseline)
    cv2.imshow("3dbox_img_pred_SGFNet", img_pred_sgfnet)
    # vis.run()
    if int(dataset.valsets[index]) in (21, 134, 204, 302):
        cv2.imwrite(os.path.join(args.output, "gt", dataset.valsets[index]+'.png'), img_gt)
        cv2.imwrite(os.path.join(args.output, "baseline", dataset.valsets[index]+'.png'), img_pred_baseline)
        cv2.imwrite(os.path.join(args.output, "sgfnet", dataset.valsets[index]+'.png'), img_pred_sgfnet)
    # vis.clear_geometries()
    key=cv2.waitKey(100) & 0xFF
    if key == ord('s'):
        cv2.imwrite(os.path.join(args.output, "gt", dataset.valsets[index]+'.png'), img_gt)
        cv2.imwrite(os.path.join(args.output, "baseline", dataset.valsets[index]+'.png'), img_pred_baseline)
        cv2.imwrite(os.path.join(args.output, "sgfnet", dataset.valsets[index]+'.png'), img_pred_sgfnet)
    if key == ord('d'):
        index+=1
    if key == ord('a'):
        index-=1
    if key == ord('q'):
        break
    if index < 0:
        index = 0
    if index > max_num:
        index = max_num