import numpy as np
import open3d
import cv2

def draw_3dframeworks(vis, corner_3d):

    corner_3d = np.transpose(corner_3d)

    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])
    colors = np.array([[1., 0., 1.] for j in range(len(lines_box))])
    line_set = open3d.geometry.LineSet()

    line_set.points = open3d.utility.Vector3dVector(corner_3d)
    line_set.lines = open3d.utility.Vector2iVector(lines_box)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)
    # render_option = vis.get_render_option()
    # render_option.point_size = 3
    # render_option.background_color = np.asarray([1, 1, 1])
    # vis.get_render_option().load_from_json('renderoption.json')
    # param = open3d.io.read_pinhole_camera_parameters('BV_1440.json')

    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.update_renderer()

def draw_2dframeworks(img, corner_2d, color=[255, 0, 255], thickness=1):
    # color = [255, 0, 255]
    # thickness = 2
    if corner_2d.min() >= 0:
        for corner_i in range(0, 4):
            i, j = corner_i, (corner_i + 1) % 4
            cv2.line(img, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
            i, j = corner_i + 4, (corner_i + 1) % 4 + 4
            cv2.line(img, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
            i, j = corner_i, corner_i + 4
            cv2.line(img, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)