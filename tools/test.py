from mot_tools.io import *
import pycocotools.mask as rletools
import cv2
import matplotlib.pyplot as plt
import json


def format_array(array):
    return [array[0], array[1], array[2], array[3]]

def KITTI2COCO(instance):
    filePath = r"E:\Challenge\MOTSChallenge\train\instances_txt"

    filename = os.path.join(filePath, "{:04}.txt".format(instance))

    txt = load_txt(filename)


    for obj in txt[1]:
        mask = rletools.decode(obj.mask)
        # plt.imshow(mask)
        # plt.show()
        print(rletools.encode(mask))

        # contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,
        #                                        cv2.CHAIN_APPROX_SIMPLE)
        # segmentation = []
        #
        # for contour in contours:
        #     contour = contour.flatten().tolist()
        #     if len(contour) > 4:
        #         segmentation.append(contour)
        # if len(segmentation) == 0:
        #     continue

        # jsonFile["annotations"].append(
        #     {
        #         "segmentation": segmentation,  # poly
        #         "iscrowd": 0,  # poly格式
        #         "area": float(rletools.area(obj.mask)),  # rletools.area()计算得到
        #         "image_id": frame,  # 对应的image
        #         "bbox": format_array(rletools.toBbox(obj.mask)),  # rletools.toBbox()计算得到
        #         "category_id": obj.class_id,  # 对应论文
        #         "id": num  # 序号
        #     }
        # )

if __name__ == '__main__':
    KITTI2COCO(2)