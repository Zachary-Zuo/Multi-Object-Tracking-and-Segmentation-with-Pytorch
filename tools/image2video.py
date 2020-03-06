import os
import cv2

def output_video(imgPath):
    fps = 16
    size = (1920,1080)

    sequence = 11

    videowriter = cv2.VideoWriter("{:04}.avi".format(sequence), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    imgDir = os.listdir(imgPath)

    for frame in range(1,len(imgDir)+1):
        img = os.path.join(imgPath, "{:06}.jpg".format(frame))
        image = cv2.imread(img)

        videowriter.write(image)
        if frame%50==0:
            print("Frame:{}".format(frame))


if __name__=="__main__":
    output_video(r"E:\Challenge\MOTSChallenge\train\images\0011")