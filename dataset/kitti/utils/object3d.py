import numpy as np
class Object3d:
    def __init__(self, content, calib):
        super(Object3d, self).__init__()
        lines = content.split()
        lines = list(filter(lambda x: len(x), lines))
        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])
        self.bbox_2d = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox_2d = np.array([float(x) for x in self.bbox_2d])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])
        if len(lines) == 16:
            self.ioc = float(lines[15])

        self.calib = calib
        
        self.corner_3d, self.corner_2d = self.box2corner()

    def rot_y(self, rotation_y):
        cos = np.cos(rotation_y)
        sin = np.sin(rotation_y)
        R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        return R
    def box2corner(self):
        R = self.rot_y(self.rotation_y)
        h, w, l = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        # y = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corner_3d = np.vstack([x, y, z])
        corner_3d = np.dot(R, corner_3d)
        corner_3d[0, :] += self.location[0]
        corner_3d[1, :] += self.location[1]
        corner_3d[2, :] += self.location[2]
        corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
        corner_2d = np.dot(self.calib.P2, corner_3d)
        corner_2d[0, :] /= corner_2d[2, :]
        corner_2d[1, :] /= corner_2d[2, :]
        corner_2d = np.array(corner_2d, dtype=np.int32)
        corner_3d[-1][-1] = 1
        inv_Tr = np.zeros_like(self.calib.Tr_velo_to_cam)
        inv_Tr[0:3, 0:3] = np.transpose(self.calib.Tr_velo_to_cam[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(self.calib.Tr_velo_to_cam[0:3, 0:3]), self.calib.Tr_velo_to_cam[0:3, 3])

        corner_3d = np.dot(inv_Tr, corner_3d)

        return corner_3d, corner_2d
