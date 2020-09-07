from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import numpy as np
import math
import pickle
import cv2
import os
from collections import defaultdict

import pycocotools.coco as coco
import torch
import torch.utils.data as data
from dataclasses import dataclass

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import copy
from detector_utils import Instance, Size2D


@dataclass
class Anno:
    instances: Instance
    video_name: str
    frame_idx: int


class DsslDataset(data.Dataset):
    default_resolution = [608, 896]
    max_objs = 256
    num_categories = 3
    cat_ids = {
        'person': 1,
        'head': 2,
        'face': 3,
    }
    num_joints = 17
    rest_focal_length = 1200
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    def __init__(self, opt=None, split=None):
        dataset_home = opt.custom_dataset_ann_path
        self._data_rng = np.random.RandomState(123)
        self.annotations = []
        self.opt = opt
        self.split = split
        self.categories = set()
        self.images = defaultdict(dict)
        ann_path = Path(dataset_home) / split / 'annotation'
        self.img_dir = Path(dataset_home) / split / 'images'
        all_ann_filepath = list(ann_path.glob('*.pickle'))
        for ann_filepath in all_ann_filepath:
            target_name = ann_filepath.name.split('.')[0]
            with open(str(ann_filepath), 'rb') as f:
                data = pickle.load(f)

            for frame_idx, instances in data.items():
                self.annotations.append(Anno(instances, target_name, frame_idx))
                for inst in instances:
                    for bbox in inst.bboxes:
                        self.categories.add(bbox.category_name)
        self.categories = sorted(list(self.categories))
        # self.cat_ids = {name: idx for idx, name in enumerate(self.categories)}
        if opt.tracking:
            self.annotations = [ann for ann in self.annotations if any([inst.id is not None for inst in ann.instances])]
            self.reid_to_indexes = defaultdict(list)
            self.first_reid_to_anno = {}
            for idx, ann in enumerate(self.annotations):
                for inst in ann.instances:
                    if inst.id is not None:
                        self.reid_to_indexes[inst.id].append(idx)
            # remove_ids = set()
            for k in self.reid_to_indexes.keys():
                self.reid_to_indexes[k] = sorted(self.reid_to_indexes[k])
            #     first_idx = self.reid_to_indexes[k].pop(0)
            #     self.first_reid_to_anno[k] = self.annotations[first_idx]
            #     remove_ids.add(first_idx)
            # [self.annotations.pop(idx) for idx in sorted(list(remove_ids), reverse=True)]
        # print(self.cat_ids)

    def __len__(self):
        return len(self.annotations)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.opt.not_rand_crop) and not disturb:
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            if type(s) == float:
                s = [s, s]
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]['bbox']
            anns[k]['bbox'] = [
                width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            # if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
            #     keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
            #         self.num_joints, 3)
            #     keypoints[:, 0] = width - keypoints[:, 0] - 1
            #     for e in self.flip_idx:
            #         keypoints[e[0]], keypoints[e[1]] = \
            #             keypoints[e[1]].copy(), keypoints[e[0]].copy()
            #     anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

            if 'rot' in self.opt.heads and 'alpha' in anns[k]:
                anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                    else - np.pi - anns[k]['alpha']

            if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
                anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

            if self.opt.velocity and 'velocity' in anns[k]:
                anns[k]['velocity'] = [-10000, -10000, -10000]

        return anns

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def _load_pre_data(self, index):
        ann = self.annotations[index]

        candidates_ids = [inst.id for inst in ann.instances if inst.id is not None]
        target_id = np.random.choice(candidates_ids, size=1)[0]
        pre_indexes = [idx for idx in self.reid_to_indexes[target_id] if
                       self.annotations[idx].frame_idx < ann.frame_idx and ann.frame_idx - self.annotations[
                           idx].frame_idx <= self.opt.max_frame_dist and ann.frame_idx - self.annotations[
                           idx].frame_idx > 5]
        if len(pre_indexes) != 0:
            pre_index = np.random.choice(pre_indexes, size=1)[0]
            pre_ann = self.annotations[pre_index]
            img, anns, _, _ = self._load_data(pre_index, 'pre')
        else:
            pre_ann = ann
            img, anns, _, _ = self._load_data(pre_ann, 'pre')
        return img, anns, ann.frame_idx - pre_ann.frame_idx

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_pre_dets(self, anns, trans_input, trans_output):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        trans = trans_input
        reutrn_hm = self.opt.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = int(ann['category_id'])
            if cls_id > self.opt.num_classes or cls_id <= -99 or \
                    ('iscrowd' in ann and ann['iscrowd'] > 0):
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                conf = 1 if np.random.random() > self.opt.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                # print(ann['track_id'])
                track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.opt.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

        return pre_hm, pre_cts, track_ids

    def __getitem__(self, index):
        opt = self.opt
        img, anns, img_info, img_path = self._load_data(index, 'img')

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
            else np.array([img.shape[1], img.shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0
        if self.split == 'train':
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s
            if np.random.random() < opt.flip:
                flipped = 1
                img = img[:, ::-1, :]
                anns = self._flip_anns(anns, width)

        trans_input = get_affine_transform(
            c, s, rot, [opt.input_w, opt.input_h])
        trans_output = get_affine_transform(
            c, s, rot, [opt.output_w, opt.output_h])
        inp = self._get_input(img, trans_input)
        ret = {'image': inp}
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

        pre_cts, track_ids = None, None
        if opt.tracking:
            pre_image, pre_anns, frame_dist = self._load_pre_data(index)
            if flipped:
                pre_image = pre_image[:, ::-1, :].copy()
                pre_anns = self._flip_anns(pre_anns, width)
            if opt.same_aug_pre and frame_dist != 0:
                trans_input_pre = trans_input
                trans_output_pre = trans_output
            else:
                c_pre, aug_s_pre, _ = self._get_aug_param(
                    c, s, width, height, disturb=True)
                s_pre = s * aug_s_pre
                trans_input_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.input_w, opt.input_h])
                trans_output_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.output_w, opt.output_h])
            pre_img = self._get_input(pre_image, trans_input_pre)
            pre_hm, pre_cts, track_ids = self._get_pre_dets(
                pre_anns, trans_input_pre, trans_output_pre)
            ret['pre_img'] = pre_img
            if opt.pre_hm:
                ret['pre_hm'] = pre_hm

        ### init samples
        self._init_ret(ret, gt_det)
        calib = self._get_calib(img_info, width, height)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(ann['category_id'])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue
            bbox, bbox_amodal = self._get_bbox_output(
                ann['bbox'], trans_output, height, width)
            # if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
            #     self._mask_ignore_or_crowd(ret, cls_id, bbox)
            #     continue
            self._add_instance(
                ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
                calib, pre_cts, track_ids)

        if self.opt.debug > 0:
            gt_det = self._format_gt_det(gt_det)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
                    'img_path': img_path, 'calib': calib,
                    'flipped': flipped}
            ret['meta'] = meta
        cv2.waitKey(0)
        return ret

    def _format_gt_det(self, gt_det):
        if (len(gt_det['scores']) == 0):
            gt_det = {'bboxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                      'scores': np.array([1], dtype=np.float32),
                      'clses': np.array([0], dtype=np.float32),
                      'cts': np.array([[0, 0]], dtype=np.float32),
                      'pre_cts': np.array([[0, 0]], dtype=np.float32),
                      'tracking': np.array([[0, 0]], dtype=np.float32),
                      'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                      'hps': np.zeros((1, 17, 2), dtype=np.float32), }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def _add_instance(
            self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
            aug_s, calib, pre_cts=None, track_ids=None):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
            return
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        ret['cat'][k] = cls_id - 1
        ret['mask'][k] = 1
        if 'wh' in ret:
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1
        ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1
        draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        gt_det['bboxes'].append(
            np.array([ct[0] - w / 2, ct[1] - h / 2,
                      ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
        gt_det['scores'].append(1)
        gt_det['clses'].append(cls_id - 1)
        gt_det['cts'].append(ct)

        if 'tracking' in self.opt.heads:
            if ann['track_id'] in track_ids:
                pre_ct = pre_cts[track_ids.index(ann['track_id'])]
                ret['tracking_mask'][k] = 1
                ret['tracking'][k] = pre_ct - ct_int
                gt_det['tracking'].append(ret['tracking'][k])
            else:
                gt_det['tracking'].append(np.zeros(2, np.float32))

        if 'ltrb' in self.opt.heads:
            ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
                             bbox[2] - ct_int[0], bbox[3] - ct_int[1]
            ret['ltrb_mask'][k] = 1

        if 'ltrb_amodal' in self.opt.heads:
            ret['ltrb_amodal'][k] = \
                bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
                bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
            ret['ltrb_amodal_mask'][k] = 1
            gt_det['ltrb_amodal'].append(bbox_amodal)

        # if 'nuscenes_att' in self.opt.heads:
        #     if ('attributes' in ann) and ann['attributes'] > 0:
        #         att = int(ann['attributes'] - 1)
        #         ret['nuscenes_att'][k][att] = 1
        #         ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
        #     gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

        if 'velocity' in self.opt.heads:
            if ('velocity' in ann) and min(ann['velocity']) > -1000:
                ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
                ret['velocity_mask'][k] = 1
            gt_det['velocity'].append(ret['velocity'][k])

        # if 'hps' in self.opt.heads:
        #     self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)
        #
        # if 'rot' in self.opt.heads:
        #     self._add_rot(ret, ann, k, gt_det)

        if 'dep' in self.opt.heads:
            if 'depth' in ann:
                ret['dep_mask'][k] = 1
                ret['dep'][k] = ann['depth'] * aug_s
                gt_det['dep'].append(ret['dep'][k])
            else:
                gt_det['dep'].append(2)

        if 'dim' in self.opt.heads:
            if 'dim' in ann:
                ret['dim_mask'][k] = 1
                ret['dim'][k] = ann['dim']
                gt_det['dim'].append(ret['dim'][k])
            else:
                gt_det['dim'].append([1, 1, 1])

        if 'amodel_offset' in self.opt.heads:
            if 'amodel_center' in ann:
                amodel_center = affine_transform(ann['amodel_center'], trans_output)
                ret['amodel_offset_mask'][k] = 1
                ret['amodel_offset'][k] = amodel_center - ct_int
                gt_det['amodel_offset'].append(ret['amodel_offset'][k])
            else:
                gt_det['amodel_offset'].append([0, 0])

    def _bbox_to_coco(self, bbox, h, w, index):
        xywh = list(bbox.xywh(Size2D(width=w, height=h)))
        xywh = [int(x) for x in xywh]
        res = {
            "id": 0,
            'track_id': -1,
            "image_id": index,
            "category_id": self.cat_ids[bbox.category_name],
            "segmentation": [],
            "area": xywh[2] * xywh[3],
            "bbox": xywh,
            "iscrowd": 0,
        }
        if 'reid' in bbox.meta:
            res['track_id'] = bbox.meta['reid']

        return res

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                         [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['ind'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask'] = np.zeros((max_objs), dtype=np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
            'dep': 1, 'dim': 3, 'amodel_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                gt_det[head] = []

        if 'hm_hp' in self.opt.heads:
            num_joints = self.num_joints
            ret['hm_hp'] = np.zeros(
                (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
            ret['hm_hp_mask'] = np.zeros(
                (max_objs * num_joints), dtype=np.float32)
            ret['hp_offset'] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32)
            ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret['hp_offset_mask'] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32)
            ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if 'rot' in self.opt.heads:
            ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
            ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
            ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({'rot': []})

    def _get_calib(self, img_info, width, height):
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                              [0, self.rest_focal_length, height / 2, 0],
                              [0, 0, 1, 0]])
        return calib

    def _load_data(self, index_or_anno, name):
        if isinstance(index_or_anno, Anno):
            annotation = index_or_anno
        else:
            annotation = self.annotations[index_or_anno]
        img_filpath = self.img_dir / annotation.video_name / f'{annotation.frame_idx}.jpg'
        img = cv2.imread(str(img_filpath))
        # cv2.imshow(name, cv2.resize(img, (1280, 720)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        anns = []
        for ins in annotation.instances:
            for bbox in ins.bboxes:
                anns.append(self._bbox_to_coco(bbox, h, w, 1))

        image_info = {
            # "id": 1,
            "width": w,
            "height": h,
            "file_name": str(img_filpath),
        }
        return img, anns, image_info, str(img_filpath)


from sklearn.model_selection import train_test_split


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def split(home_path=None):
    home_path = Path(home_path)
    ann_path = home_path / 'full_data'
    img_dir = home_path / 'images'
    all_ann_filepath = list(ann_path.glob('*.pickle'))
    train, val = train_test_split(all_ann_filepath, test_size=0.2)
    train_image_path = home_path / 'dump' / 'train' / 'images'
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_ann_path = home_path / 'dump' / 'train' / 'annotation'
    train_ann_path.mkdir(parents=True, exist_ok=True)
    val_image_path = home_path / 'dump' / 'val' / 'images'
    val_image_path.mkdir(parents=True, exist_ok=True)
    val_ann_path = home_path / 'dump' / 'val' / 'annotation'
    val_ann_path.mkdir(parents=True, exist_ok=True)

    for annotation_file in train:
        video_name = annotation_file.name.split('.')[0]
        with cd(str(train_image_path)):
            os.symlink(f'../../../images/{video_name}', video_name)
        with cd(str(train_ann_path)):
            os.symlink(f'../../../full_data/{annotation_file.name}', annotation_file.name)

    for annotation_file in train:
        video_name = annotation_file.name.split('.')[0]
        with cd(str(val_image_path)):
            os.symlink(f'../../../images/{video_name}', video_name)
        with cd(str(val_ann_path)):
            os.symlink(f'../../../full_data/{annotation_file.name}', annotation_file.name)


class op:
    pass


from opts import opts

if __name__ == '__main__':
    # o = op()
    # o.tracking = True
    #
    #
    # ds = DsslDataset(o, split='train', dataset_home='/home/kstarkov/work2/dataset/P-DESTRE/dump')
    # i = 0

    split('/home/kstarkov/work2/dataset/P-DESTRE')
