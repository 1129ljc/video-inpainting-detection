import os
import json
import sys
import cv2
import numpy as np

from test_test import test


def video2frame(video_dir, frame_dir):
    files = os.listdir(video_dir)
    for i in range(len(files)):
        name = files[i]
        video = os.path.join(video_dir, name)
        video_cap = cv2.VideoCapture(video)
        frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        save_dir = os.path.join(frame_dir, name.split('.')[0])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j in range(10):
            ref, image = video_cap.read()
            cv2.imwrite(os.path.join(save_dir, str(j + 1).zfill(5) + '.png'), image)
            # print(os.path.join(save_dir, str(j + 1).zfill(5) + '.png'))


def frame2result(imagedir, maskdir, resultdir):
    files = os.listdir(imagedir)
    for i in range(len(files)):
        name = files[i]
        video = os.path.join(imagedir, name)
        mask = os.path.join(maskdir, name)
        index_pics = os.listdir(video)
        h, w = 256, 256
        image_1 = cv2.imread(os.path.join(video, index_pics[0]))
        image_2 = cv2.imread(os.path.join(video, index_pics[1]))
        image_3 = cv2.imread(os.path.join(video, index_pics[2]))
        image_4 = cv2.imread(os.path.join(video, index_pics[3]))
        image_5 = cv2.imread(os.path.join(video, index_pics[4]))
        image_6 = cv2.imread(os.path.join(video, index_pics[5]))
        image_7 = cv2.imread(os.path.join(video, index_pics[6]))
        image_8 = cv2.imread(os.path.join(video, index_pics[7]))
        image_1 = cv2.resize(image_1, (h, w))
        image_2 = cv2.resize(image_2, (h, w))
        image_3 = cv2.resize(image_3, (h, w))
        image_4 = cv2.resize(image_4, (h, w))
        image_5 = cv2.resize(image_5, (h, w))
        image_6 = cv2.resize(image_6, (h, w))
        image_7 = cv2.resize(image_7, (h, w))
        image_8 = cv2.resize(image_8, (h, w))
        mask_1 = cv2.imread(os.path.join(mask, index_pics[0]))
        mask_2 = cv2.imread(os.path.join(mask, index_pics[1]))
        mask_3 = cv2.imread(os.path.join(mask, index_pics[2]))
        mask_4 = cv2.imread(os.path.join(mask, index_pics[3]))
        mask_5 = cv2.imread(os.path.join(mask, index_pics[4]))
        mask_6 = cv2.imread(os.path.join(mask, index_pics[5]))
        mask_7 = cv2.imread(os.path.join(mask, index_pics[6]))
        mask_8 = cv2.imread(os.path.join(mask, index_pics[7]))
        mask_1 = cv2.resize(mask_1, (h, w))
        mask_2 = cv2.resize(mask_2, (h, w))
        mask_3 = cv2.resize(mask_3, (h, w))
        mask_4 = cv2.resize(mask_4, (h, w))
        mask_5 = cv2.resize(mask_5, (h, w))
        mask_6 = cv2.resize(mask_6, (h, w))
        mask_7 = cv2.resize(mask_7, (h, w))
        mask_8 = cv2.resize(mask_8, (h, w))
        image_cat1 = np.concatenate([
            np.concatenate([image_1, mask_1], axis=1),
            np.concatenate([image_2, mask_2], axis=1),
            np.concatenate([image_3, mask_3], axis=1),
            np.concatenate([image_4, mask_4], axis=1)
        ], axis=0)
        image_cat2 = np.concatenate([
            np.concatenate([image_5, mask_5], axis=1),
            np.concatenate([image_6, mask_6], axis=1),
            np.concatenate([image_7, mask_7], axis=1),
            np.concatenate([image_8, mask_8], axis=1)
        ], axis=0)
        image_cat = np.concatenate([image_cat1, image_cat2], axis=1)
        image_cat = cv2.resize(image_cat, (512, 512))
        save = os.path.join(resultdir, name.split('.')[0] + '.png')
        cv2.imwrite(save, image_cat)


def test_image(input_a, input_b, input_c):
    input_arg_task_id = input_a
    input_arg_file_path = input_b
    input_arg_ext = input_c

    input_arg_ext_json = json.loads(input_arg_ext)
    input_arg_ext_out_json_path = input_arg_ext_json['JSON_FILE_PATH']

    input_arg_ext_tmp_dir = input_arg_ext_json['TMP_DIR']
    input_arg_ext_tmp_dir = os.path.join(input_arg_ext_tmp_dir, 'ljc_docs')

    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_tmp_dir, 'inpainting_forensics')
    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_out_tmp_path, str(input_arg_task_id))

    input_arg_ext_gpu_id = input_arg_ext_json['GPU_ID']
    images_dir = os.path.join(input_arg_ext_out_tmp_path, 'images')
    masks_dir = os.path.join(input_arg_ext_out_tmp_path, 'masks')
    result_dir = os.path.join(input_arg_ext_out_tmp_path, 'result')

    if not os.path.exists(input_arg_ext_out_tmp_path):
        os.makedirs(input_arg_ext_out_tmp_path)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    algorithm_message = '该算法使用多种预滤波模块，包括限制卷积、高通滤波等，卷积编码器采用网络结构搜索方法，并结合注意力机制，输出为输入视频8个采样帧的二值掩膜图。'
    print(algorithm_message)

    algorithm_args = {
        'dataset': images_dir,
        'ckpt': '/home/dell/soft/ljc_methods/video_inpainting_detection/iid_net/weights/IID_weights.pth',
        'save': masks_dir,
        'gpu_id': input_arg_ext_gpu_id,
    }
    print(algorithm_args)
    video2frame(input_arg_file_path, images_dir)

    test(algorithm_args)

    frame2result(images_dir, masks_dir, result_dir)

    result_json_content = {}
    images_detection = sorted(os.listdir(input_arg_file_path))
    images_location = sorted(os.listdir(result_dir))

    for i in range(len(images_detection)):
        image_detec_name = images_detection[i]
        image_locac_name = images_location[i]
        conclusion = '白色区域为可疑的修补区域'
        image_feature = []
        image_feature_one = {'filepath': os.path.join(result_dir, image_locac_name),
                             'title': '视频逐帧修补定位示意图',
                             'comment': '该图展示待检测视频采样帧可疑的修补区域。'}
        image_feature.append(image_feature_one)

        video_json = {'taskid': str(input_arg_task_id),
                      'conclusion': conclusion,
                      'message': algorithm_message,
                      'confidence': 1.0,
                      'threshold': 0.5,
                      'features': image_feature}
        result_json_content[image_detec_name] = video_json

    json_path = input_arg_ext_out_json_path
    with open(json_path, 'w') as f:
        json.dump(result_json_content, f)
    f.close()


if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    input_3 = sys.argv[3]
    test_image(input_a=input_1, input_b=input_2, input_c=input_3)
