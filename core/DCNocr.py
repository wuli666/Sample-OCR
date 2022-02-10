import math
import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

import cv2
import numpy as np
from pathlib import Path
from core.pridict_engine import predictor
from core.lib.utils.utility import initial_logger
from core.lib.utils.NLP_engine import match
from core.lib.NLP_package import BERTConfig
logger = initial_logger()
from core.lib.utils.utility import check_and_read_gif, get_image_file_list

__all__ = ['DCNocr']

character_dict = {
    'rec': {
        'ch': {
            'dict_path': './lib/utils/ppocr_keys_v1.txt'
        },
        'en': {
            'dict_path': './lib/utils/ic15_dict.txt'
        },
        'french': {
            'dict_path': './lib/utils/french_dict.txt'
        },
        'german': {
            'dict_path': './lib/utils/german_dict.txt'
        },
        'korean': {
            'dict_path': './lib/utils/korean_dict.txt'
        },
        'japan': {
            'dict_path': './lib/utils/japan_dict.txt'
        }
    },
}

SUPPORT_DET_MODEL = ['DB']
SUPPORT_REC_MODEL = ['CRNN']
BASE_DIR = os.path.expanduser("./data/models")


def parse_args():
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default=None)
    parser.add_argument("--det_max_side_len", type=float, default=960)

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=2.0)
    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str, default=None)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=30)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_char_dict_path", type=str, default=None)
    parser.add_argument("--use_space_char", type=bool, default=True)

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str, default=None)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=30)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=bool, default=False)
    parser.add_argument("--use_zero_copy_run", type=bool, default=False)

    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--OCR_Model", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--cls", type=str2bool, default=False)
    return parser.parse_args()


class DCNocr(predictor.TextSystem):
    def __init__(self, **kwargs):

        postprocess_params = parse_args()
        postprocess_params.__dict__.update(**kwargs)
        self.use_angle_cls = postprocess_params.use_angle_cls
        lang = postprocess_params.lang
        assert lang in character_dict[
            'rec'], 'param lang must in {}, but got {}'.format(
            character_dict['rec'].keys(), lang)
        if postprocess_params.rec_char_dict_path is None:
            postprocess_params.rec_char_dict_path = character_dict['rec'][lang][
                'dict_path']

        if postprocess_params.det_model_dir is None:
            postprocess_params.det_model_dir = os.path.join(BASE_DIR, 'OCR_Model')
        if postprocess_params.rec_model_dir is None:
            postprocess_params.rec_model_dir = os.path.join(
                BASE_DIR, 'rec/{}'.format(lang))
        if postprocess_params.cls_model_dir is None:
            postprocess_params.cls_model_dir = os.path.join(BASE_DIR, 'cls')

        if postprocess_params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if postprocess_params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        postprocess_params.rec_char_dict_path = Path(
            __file__).parent / postprocess_params.rec_char_dict_path

        super().__init__(postprocess_params)

    def ocr(self, img, det=True, rec=True, cls=False):
        assert isinstance(img, (np.ndarray, list, str))
        if cls and not self.use_angle_cls:
            print('cls should be false when use_angle_cls is false')
            exit(-1)
        self.use_angle_cls = cls
        if isinstance(img, str):
            image_file = img
            # print(image_file, end='\t')
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if det and rec:
            dt_boxes, rec_res = self.__call__(img)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res


def is_letter(val):
    return val.isalpha()


def main():
    args = parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    ocr_engine = DCNocr()
    print("NLP_ENGINE start processing")
    for img_path in image_file_list:
        result = ocr_engine.ocr(img_path,
                                det=args.OCR_Model,
                                rec=args.rec,
                                cls=args.cls)

        b = [x[0] for x in result]
        a = [x[1] for x in result]

        final = []
        result_final = []
        result_final_a = []

        def softmax(x):
            x_row_max = x.max(axis=-1)
            x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
            x = x - x_row_max
            x_exp = np.exp(x)
            x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
            softmax = x_exp / x_exp_row_sum
            return softmax

        def distance(a, b):
            point = a - b
            return math.hypot(point[0], point[1])

        w1 = []
        result_list = []
        for j in range(len(b)):
            point1 = np.array(b[j][0])
            point2 = np.array(b[j][1])
            point3 = np.array(b[j][2])
            point4 = np.array(b[j][3])
            point_x_max = max(point1[0], point2[0], point3[0], point4[0])
            point_x_min = min(point1[0], point2[0], point3[0], point4[0])
            point_y_max = max(point1[1], point2[1], point3[1], point4[1])
            point_y_min = min(point1[1], point2[1], point3[1], point4[1])
            side1 = distance(point2, point1)
            side2 = distance(point2, point3)
            side3 = distance(point4, point1)
            side4 = distance(point4, point3)
            side_base = min((side2 / side1, (side1 / side2)))
            w1.append(side_base)
            z = (side1 + side2 + side3 + side4) / 2
            value = (point_x_max - point_x_min) * (point_y_max - point_y_min)
            result_list.append(value * side_base)
            final.append(value)

        w1 = np.array(w1)
        w1 = softmax(w1)
        max_result = max(result_list)
        index = result_list.index(max_result)

        print(img_path[15:])
        bert = BERTConfig
        # print('index: ', index, 'result: ', max_result)
        all_matchings, _, index = match(a[index][0])
        print(all_matchings[index])

        # output = open('data.xls', 'a', encoding='utf8')
        # output.write('')
        # for i in range(1):
        #     output.write(str(img_path[15:]))
        #     output.write('\t')
        #     output.write(str(all_matchings[index]))
        #     output.write('\t')
        # output.write('\n')
        # output.close()
