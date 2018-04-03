import numpy as np
def rgb2lab(rgb):
    return xyztolab(rgbtoxyz(rgb))
    
def rgbtoxyz(rgb):
    rgb /= 255.0
    processed_rgb = np.piecewise(rgb, [rgb > 0.4045, rgb <= 0.045], [lambda rgb: ((rgb + 0.055) / 1.055) ** 2.4, lambda rgb: rgb / 12.92]).T * 100.0
    xyz_weights = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.round(np.matmul(xyz_weights, processed_rgb), decimals=4)
    
    return xyz

def xyztolab(xyz):
    diffuser = np.array([95.047, 100.0, 108.883])
    xyz /= diffuser
    processed_xyz = np.piecewise(xyz, [xyz > 0.008856, xyz <= 0.00856], [lambda xyz: xyz ** (1.0 / 3.0), lambda xyz: (7.787 * xyz) + (16.0 / 116.0)])
    L = round(116 * processed_xyz[0] - 16, 4)
    a = round(500 * (processed_xyz[0] - processed_xyz[1]))
    b = round(200 * (processed_xyz[1] - processed_xyz[2]))
    
    Lab = np.array([L, a, b])
    return Lab