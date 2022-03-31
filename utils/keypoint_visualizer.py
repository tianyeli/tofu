'''
keypoint visualizer

tianye li
Please see LICENSE for the licensing information
'''
import numpy as np

# -----------------------------------------------------------------------------

def draw_landmarks(img, landmarks, color=(0,0,255), selection=None, thickness=3, radius=3, shift=0):
    # assume landmarks is a list of points
    import math, cv2
    canvas = img.copy()
    point_num = len(landmarks)
    ch_num = canvas.shape[2]

    if isinstance(color, tuple):
        colors = np.tile(np.array(list(color)), (point_num, 1))
    elif isinstance(color, np.ndarray):
        if color.ndim == 1:
            colors = np.tile(color, (point_num, 1))
        else:
            colors = color
    else:
        raise RuntimeError(f"invalid color type")

    if ch_num == 4:
        colors = np.concatenate((colors, 255*np.ones((point_num, 1))), axis=1)

    for pp in range( point_num ):
        if selection is not None:
            if pp not in selection:
                continue
        this_lmk = (
            int(math.floor(landmarks[pp][0])),
            int(math.floor(landmarks[pp][1]))
        )
        this_color = colors[pp].tolist()

        cv2.circle( canvas, this_lmk, radius=radius, thickness=thickness, color=this_color, shift=shift )
    return canvas

# -----------------------------------------------------------------------------

def draw_landmarks_w_gt(img, lmk_pred, lmk_gt=None,
                        color_pred=(0,0,255), color_gt=(255,0,0), color_line=(0,255,0),
                        thickness=3, radius=3, shift=0):
    import cv2
    canvas = img.copy()

    # draw gt
    if lmk_gt is not None:
        canvas = draw_landmarks(canvas, lmk_gt, color=color_gt, thickness=thickness, radius=radius, shift=shift) # blue

    # draw predicted
    canvas = draw_landmarks(canvas, lmk_pred, color=color_pred, thickness=thickness, radius=radius, shift=shift) # red

    # draw lines
    if lmk_gt is not None:
        idx_pred = np.unique(np.nonzero(lmk_pred>=0 )[0])
        idx_gt   = np.unique(np.nonzero(lmk_gt>=0 )[0])
        idx_common = np.intersect1d(idx_pred, idx_gt)
        for lid in idx_common:
            cv2.line(
                canvas,
                tuple(lmk_pred[lid].astype(np.int)), 
                tuple(lmk_gt[lid].astype(np.int)),
                color=color_line
            )
    return canvas