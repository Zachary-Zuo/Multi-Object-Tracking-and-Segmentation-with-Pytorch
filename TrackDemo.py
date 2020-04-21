import torch
from tracker.multitracker import MOTSTracker
import argparse
from network.Darknet import parse_model_cfg
from dataloaders.track_dataloader import LoadVideo
import os
import os.path as osp
from utils.utils import *
from network.maskhead import MaskHead
import pycocotools.mask as rletools
from torch.utils.tensorboard import SummaryWriter

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def format_mask_box(box):
    # x1 = int(box[0]/2176*1920)
    # y1 = int(box[1]/1216*1080)
    # x2 = int((box[2]+box[0])/2176*1920)
    # y2 = int((box[3]+box[1])/1216*1080)
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2] + box[0])
    y2 = int(box[3] + box[1])
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2>2176:
        x2=2176
    if y2>1216:
        y2=1216
    # return torch.Tensor([x1, y1, x2, y2]).cuda()
    return [x1, y1, x2, y2]

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    torch.cuda.set_device(0)
    tracker = MOTSTracker(opt, frame_rate=frame_rate)
    seghead = MaskHead()
    seghead.load_state_dict(
        torch.load(r'E:\Challenge\Multi-Object-Tracking-and-Segmentation-with-Pytorch\models\MaskHead_epoch-1.pth',map_location=lambda storage, loc: storage))
    seghead.cuda().eval()
    results = []
    frame_id = 0
    file = open('{:04}.txt'.format(11), "w")
    for path, img, img0 in dataloader:
        # run tracking
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets,featuremap = tracker.update(blob, img0)
        online_box = []
        online_ids = []
        online_obj = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            # online_box.append(format_mask_box(tlwh))
            # online_ids.append(tid)
            online_obj.append((tid,format_mask_box(tlwh)))
        online_obj.sort(key=lambda item: item[1][3] * item[1][2])
        online_obj.reverse()
        # plt.imshow(img[0])
        # plt.show()
        output=seghead(featuremap)
        background = output[0][0].detach().cpu().numpy()
        background = cv2.resize(background,(2176,1216))
        plt.imshow(output[0][0].detach().cpu().numpy())
        plt.show()
        background = normalization(background)
        background[background<0.7]=0
        plt.imshow(background)
        plt.show()
        if len(online_obj)!=0:
            img = img[0]
            for track_id,box in online_obj:
                mask = np.zeros_like(background)
                temp = mask[box[1]:box[3], box[0]:box[2]]
                # plt.imshow(img[box[1]:box[3], box[0]:box[2]])
                # plt.show()
                # print(len(online_obj))

                temp[temp > -1] = 1

                real = background.copy()
                real[mask < 1] = 0
                background[mask > 0] = 0
                real[real > 0] = 1
                # plt.imshow(real)
                # plt.show()
                real = cv2.resize(real, (1920, 1080))
                mask = np.asfortranarray(real)
                mask = mask.astype(np.uint8)
                # plt.imshow(mask)
                # plt.show()
                rle = rletools.encode(mask)
                # print(rletools.area(rle))
                if rletools.area(rle) < 10:
                    continue
                line = ' '.join([str(frame_id + 1), str(int(2000 + track_id)), "2", str(rle['size'][0]), str(rle['size'][1]),
                                 rle['counts'].decode(encoding='UTF-8')])
                file.write(line + '\n')
        frame_id += 1
        if frame_id%50 ==0:
            print(frame_id)
    file.close()
    return frame_id


def track(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    accs = []
    n_frame = 0

    dataloader = LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
                 save_dir=frame_dir, show_image=False, frame_rate=frame_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default=r'E:\Challenge\Towards-Realtime-MOT-master\weight\latest.pt',
                        help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str,
                        default=r'E:\Challenge\Multi-Object-Tracking-and-Segmentation-with-Pytorch\results\0011.avi',
                        help='path to the input video')
    parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'],
                        help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    track(opt)