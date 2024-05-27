import argparse
import os

import cv2
import numpy as np

from utils import comput_sad_loss, compute_mse_loss, compute_mad_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='./predDIM/', help="pred alpha dir")
    parser.add_argument('--label-dir', type=str,
                        default='/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset/Test_set/alpha_copy/',
                        help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str, default='/input/masks', help="trimap dir")
    parser.add_argument('--benchmark', type=str, default='none', help="benchmark", choices=['none', 'him2k'])

    args = parser.parse_args()

    mse_loss = []
    sad_loss = []
    mad_loss = []

    ### loss_unknown only consider the unknown regions, i.e. trimap==128, as trimap-based methods do
    mse_loss_unknown = []
    sad_loss_unknown = []

    if args.benchmark == 'him2k':
        # 遍历GroundTruth文件夹
        for folder_item in os.listdir(args.label_dir):
            gt_folder_path = os.path.join(args.label_dir, folder_item)
            # 遍历单个ground truth文件夹
            for img in os.listdir(gt_folder_path):
                pred_folder = os.path.join(args.pred_dir, folder_item)
                print(img)
                pred = cv2.imread(os.path.join(pred_folder, img), 0).astype(np.float32)
                label = cv2.imread(os.path.join(gt_folder_path, img), 0).astype(np.float32)
                trimap = cv2.imread(os.path.join(gt_folder_path, img), 0).astype(np.float32)

                mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
                sad_loss_unknown_ = comput_sad_loss(pred, label, trimap)[0]

                trimap[...] = 128

                mse_loss_ = compute_mse_loss(pred, label, trimap)
                sad_loss_ = comput_sad_loss(pred, label, trimap)[0]
                mad_loss_ = compute_mad_loss(pred, label, trimap)

                print('Whole Image: MSE:', mse_loss_, ' SAD:', sad_loss_)
                print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

                mse_loss_unknown.append(mse_loss_unknown_)
                sad_loss_unknown.append(sad_loss_unknown_)

                mse_loss.append(mse_loss_)
                sad_loss.append(sad_loss_)
                mad_loss.append(mad_loss_)

            print('Average:')
            print('Whole Image: MSE:', np.array(mse_loss).mean(), ' SAD:', np.array(sad_loss).mean(), 'MAD:',
                  np.array(mad_loss).mean())
            # print('Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
    else:
        for img in os.listdir(args.label_dir):
            print(img)

            pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
            label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
            trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)

            mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
            sad_loss_unknown_ = comput_sad_loss(pred, label, trimap)[0]

            trimap[...] = 128

            mse_loss_ = compute_mse_loss(pred, label, trimap)
            sad_loss_ = comput_sad_loss(pred, label, trimap)[0]
            mad_loss_ = compute_mad_loss(pred, label, trimap)

            print('Whole Image: MSE:', mse_loss_, ' SAD:', sad_loss_)
            print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

            mse_loss_unknown.append(mse_loss_unknown_)
            sad_loss_unknown.append(sad_loss_unknown_)

            mse_loss.append(mse_loss_)
            sad_loss.append(sad_loss_)
            mad_loss.append(mad_loss_)

        print('Average:')
        print('Whole Image: MSE:', np.array(mse_loss).mean(), ' SAD:', np.array(sad_loss).mean(), 'MAD:',
              np.array(mad_loss).mean())
        # print('Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
