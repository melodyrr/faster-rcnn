import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes."""
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


if __name__ == "__main__":
	bbox_a = np.array([[0,0,4,4],[1,1,2,2]])
	bbox_b = np.array([[2,2,4,4],[1,1,4,4],[0,0,2,2]])
	print(bbox_a[:, None, :2])
	print("----")
	print(bbox_b[:, :2])

	bbox_iou(bbox_a, bbox_b)

	tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
	br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
	area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
	area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
	area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

	print(br - tl)
	print("----")
	print(np.prod(br - tl, axis=2))  # np.prod() 连乘

	print((tl < br).all(axis=2))
	print(area_i)
