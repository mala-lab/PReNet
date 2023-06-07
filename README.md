# PReNet
Official implementation of PReNet: "Deep Weakly-supervised Anomaly Detection" in the proceedings of KDD 2023.

## Datasets
All the datasets used in the PReNet paper have been made availble at the [data](https://github.com/mala-lab/PReNet/tree/main/data) folder, including the datasets used in both seen and unseen anomaly detection settings.

## Experiments

The below function is used to perform seen anomaly detection,
```python
run_prenet(args)
```
while *run_prenet_unseenanomaly* is used to perform unseen anomaly detection.
```python
run_prenet_unseenanomaly(args)
```

Using an ensemble size of one runs significantly faster while being able to obtain similarly good performance as using an ensemble size of 30. Thus, an ensemble size of one is recommended for large-scale experiments.

PReNet is implemented using Tensorflow/Keras. The main packages and their versions used in this work are provided as follows:
- keras==2.3.1
- numpy==1.16.2
- pandas==0.23.4
- scikit-learn==0.20.0
- scipy==1.1.0
- tensorboard==1.14.0
- tensorflow==1.14.0

Note that the results of PReNet on the datasets may slightly differ from the results reported in the paper, due to the randomness in the selection of seen anomalies and in the optimization.


## Full Paper
The full paper can be found at [ACM Portal](https://doi.org/10.1145/3580305.3599302) or [arXiv](https://arxiv.org/abs/1910.13601).

## Citation
```bibtex
@inproceedings{pang2023deep,
  title={Deep Weakly-supervised Anomaly Detection},
  author={Pang, Guansong and Shen, Chunhua and Jin, Huidong and van den Hengel, Anton},
  booktitle={Proceedings of the 29th ACM SIGKDD international conference on knowledge discovery \& data mining},
  year={2023}
}
```
