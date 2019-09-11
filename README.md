# <p align=center>Sequential Sentence Classification</p>
This repo has code and data for our paper ["Pretrained Language Models for Sequential Sentence Classification"](https://arxiv.org/pdf/1909.04054.pdf).

### How to run

```
pip install -r requirements.txt
scripts/train.sh tmp_output_dir
```

Update the `scripts/train.sh` script with the appropriate hyperparameters and datapaths.

### CSAbstrcut dataset

The train, dev, test splits of the dataset are in `data/CSAbstrcut`
### Citing

If you use the data or the model, please cite,
```
@inproceedings{Cohan2019EMNLP,
  title={Pretrained Language Models for Sequential Sentence Classification},
  author={Arman Cohan, Iz Beltagy, Daniel King, Bhavana Dalvi, Dan Weld},
  year={2019},
  booktitle={EMNLP},
}
```
