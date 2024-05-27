import cv2
import numpy as np
import torch

import networks
import utils
from dataloader.data_generator_instance_wise import DataGenerator
from dataloader.image_file import ImageFileTrain

path_alpha = "C:/Users/a/dataset/matting/aim_500/mask"
path_original = "C:/Users/a/dataset/matting/aim_500/original"
path_bg = "C:/Users/a/dataset/matting/Distinctions-646/Test/bg/VOCdevkit/VOC2012/JPEGImages/"


def _composite_inner(fg, alpha, bg):
    fg_data = cv2.imread(fg)
    bg_data = cv2.imread(bg)
    alpha_data = cv2.imread(alpha, 0).astype(np.float32) / 255.

    h, w = alpha_data.shape
    bg_data = cv2.resize(bg_data, (w, h), interpolation=cv2.INTER_CUBIC)
    new_bg = fg_data.astype(np.float32) * alpha_data[:, :, None] + bg_data.astype(np.float32) * (
            1 - alpha_data[:, :, None])
    return new_bg


def test_composite_wild(fg1_path, alpha1_path, fg2_path, alpha2_path, bg_path):
    new_bg = _composite_inner(fg2_path, alpha2_path, bg_path)
    fg1_data = cv2.imread(fg1_path)
    alpha1_data = cv2.imread(alpha1_path, 0).astype(np.float32) / 255.
    h, w = alpha1_data.shape
    new_bg = cv2.resize(new_bg, (w, h), interpolation=cv2.INTER_CUBIC)
    new_img = fg1_data.astype(np.float32) * alpha1_data[:, :, None] + new_bg.astype(np.float32) * (
            1 - alpha1_data[:, :, None])
    alpha2_data = cv2.imread(alpha2_path, 0).astype(np.float32) / 255.
    alpha2_data = cv2.resize(alpha2_data, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha2_data = alpha2_data * (1 - alpha1_data)

    alpha1_8bit = (alpha1_data * 255).astype(np.uint8)
    alpha2_8bit = (alpha2_data * 255).astype(np.uint8)
    cv2.imwrite('test/alpha1_output_2.png', alpha1_8bit)
    cv2.imwrite('test/alpha2_output_2.png', alpha2_8bit)
    cv2.imwrite("test/img_output_2.png", new_img)


def test_composite(fg1_path, alpha1_path, fg2_path, alpha2_path, bg_path):
    fg1 = cv2.imread(fg1_path)
    fg2 = cv2.imread(fg2_path)
    alpha1 = cv2.imread(alpha1_path, 0).astype(np.float32) / 255.
    alpha2 = cv2.imread(alpha2_path, 0).astype(np.float32) / 255.

    h, w = alpha1.shape
    fg2 = cv2.resize(fg2, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha2 = cv2.resize(alpha2, (w, h), interpolation=cv2.INTER_NEAREST)

    bg = cv2.imread(bg_path, 1)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

    alpha_tmp = 1 - (1 - alpha1) * (1 - alpha2)
    if np.any(alpha_tmp < 1):
        fg = fg1.astype(np.float32) * alpha1[:, :, None] + fg2.astype(np.float32) * (1 - alpha1[:, :, None])
        # The overlap of two 50% transparency should be 25%
        alpha = alpha_tmp
        fg = fg.astype(np.uint8)

        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255
        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])

        # 转换alpha通道到0-255范围并转为uint8
        alpha_8bit = (alpha * 255).astype(np.uint8)

        # 保存fg图像
        cv2.imwrite('fg_output.png', fg)

        # 保存带有alpha通道的图像
        cv2.imwrite('alpha_output.png', alpha_8bit)

        cv2.imwrite("img_output_1.png", image)


def test_imagefiletrain(fg, alpha, bg):
    imagefiletrain = ImageFileTrain(alpha_dir=alpha, fg_dir=fg, bg_dir=bg, alpha_ext=".png")
    train_dataset = DataGenerator(imagefiletrain, phase="train")
    train_dataset.__getitem__(0)
    # train_dataset.__getitem__(1)
    # train_dataset.__getitem__(2)
    # train_dataset.__getitem__(3)
    # train_dataset.__getitem__(4)
    # train_dataset.__getitem__(5)
    # train_dataset.__getitem__(6)
    # train_dataset.__getitem__(7)


def test_infer(image, mask):
    # laod model
    model = networks.get_generator(encoder="res_shortcut_encoder_29", decoder="res_shortcut_decoder_22")
    model.cuda()

    # load checkpoint
    checkpoint = torch.load("checkpoints/MGMatting_DIM_100k/latest_model.pth")
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params / 1_000_000)
    # inference
    # model = model.eval()
    # image_dict = generator_tensor_dict2(image, mask)
    # single_inference(model, image_dict)


def test_data_generator2():
    train_image_file = ImageFileTrain(alpha_dir="data/Training_set/Adobe_licensed_images/alpha_copy",
                                      fg_dir="data/Training_set/Adobe_licensed_images/fg_copy",
                                      bg_dir="C:/Users/a/dataset/matting/train2017",
                                      alpha_ext=".jpg",
                                      fg_ext=".jpg",
                                      bg_ext=".jpg")

    train_dataset = DataGenerator(train_image_file, phase='train')
    train_dataset.__getitem__(1)



if __name__ == "__main__":
    test_data_generator2()
