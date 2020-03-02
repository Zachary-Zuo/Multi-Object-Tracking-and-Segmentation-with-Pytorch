from mot_tools.io import *
import pycocotools.mask as rletools
import cv2
import json


def format_array(array):
    return [array[0], array[1], array[2], array[3]]

def KITTI2COCO(instance):
    filePath = r"E:\Challenge\KITTI_MOTS\instances_txt"

    filename = os.path.join(filePath, "{:04}.txt".format(instance))

    txt = load_txt(filename)

    imgPath = os.path.join(r"E:\Challenge\KITTI_MOTS\training\image_02","{:04}".format(instance))

    imgDir = os.listdir(imgPath)

    jsonFile = {"images": [], "categories": [], "annotations": []}

    maxClassNum = 0
    num = -1
    for frame in range(len(imgDir)):
        img = os.path.join(imgPath, "{:06}.png".format(frame))
        img = cv2.imread(img)
        jsonFile["images"].append(
            {
                # 图片尺寸
                "height": img.shape[0],
                "width": img.shape[1],
                "id": frame,  # 序号
                # "file_name": "{:06}.png".format(frame)
                "file_name": "{:06}.png".format(frame)
            }
        )

        if frame not in txt.keys():
            continue
        for obj in txt[frame]:
            if obj.class_id==10:
                continue
            num += 1
            mask = rletools.decode(obj.mask)
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) > 4:
                    segmentation.append(contour)
            if len(segmentation) == 0:
                continue
            jsonFile["annotations"].append(
                {
                    "segmentation": segmentation,  # poly
                    "iscrowd": 0,  # poly格式
                    "area": float(rletools.area(obj.mask)),  # rletools.area()计算得到
                    "image_id": frame,  # 对应的image
                    "bbox": format_array(rletools.toBbox(obj.mask)),  # rletools.toBbox()计算得到
                    "category_id": obj.class_id,  # 对应论文
                    "id": num  # 序号
                }
            )
            if obj.class_id > maxClassNum:
                maxClassNum = obj.class_id


    jsonFile["categories"].append(
        {
            "supercategory": "car",
            "id": 1,  # 序号
            "name": "car"  # 保持一致
        }
    )
    jsonFile["categories"].append(
        {
            "supercategory": "ped",
            "id": 2,  # 序号
            "name": "ped"  # 保持一致
        }
    )
    jsonFile["categories"].append(
        {
            "supercategory": "mix",
            "id": 10,  # 序号
            "name": "mix"  # 保持一致
        }
    )
    result = json.dumps(jsonFile)
    file = open("{:04}".format(instance)+".json", mode='w')
    data = file.write(result)

if __name__ == '__main__':
    for i in range(21):
        KITTI2COCO(i)