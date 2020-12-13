# Mask Guided Matting via Progressive Refinement Network

<p align="center">
  <img src="teaser.png" width="1050" title="Teaser Image"/>
</p>

This repository includes the official project of Mask Guided (MG) Matting, presented in our paper:

**[Mask Guided Matting via Progressive Refinement Network](https://arxiv.org/abs/1908.00672)**

## Highlights
- **Trimap-free Alpha Estimation:** MG Matting does not require a carefully annotated trimap as guidance inputs. Instead, it takes a general rough mask, which could be generated by segmentation or saliency models automatically, and predicts an alpha matte with great details;
- **Foreground Color Prediction:** MG Matting predicts the foreground color besides alpha matte, we notice and address the inaccuracy of foreground annotations in Composition-1k by Random Alpha Blending;
- **No Additional Training Data:** MG Matting is trained only with the widely-used publicly avaliable synthetic dataset Composition-1k, and shows great performance on both synthetic and real-world benchmarks.

## News
- 12 Dec 2020: Release [Arxiv version of paper](https://arxiv.org/pdf/2011.11961.pdf) and visualizations of sample images and videos.

## TODO
Inference demo, real-world portrait benchmark shall be released soon. Before that, if you want to test your model on the real-world portrait benchmark or compare results with MG Matting, feel free to contact Qihang Yu (yucornetto@gmail.com).

## Dataset
In our experiments, **only Composition-1k training set is used to train the model**. And the obtained model is evaluaed on three dataset: Composition-1k, Distinction-646, and our real-world portrait dataset.

**For Compsition-1k**, please contact Brian Price (bprice@adobe.com) requesting for the dataset. And please refer to [GCA Matting](https://github.com/Yaoyi-Li/GCA-Matting) for dataset preparation.

**For Distinction-646**, please refer to [HAttMatting](https://github.com/wukaoliu/CVPR2020-HAttMatting) for the dataset.

**Our real-world portrait dataset** shall be released to public soon.

## Citation
If you find this work or code useful for your research, please use the following BibTex entry:
```
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
