# AnomalyHop: An SSL-based Image Anomaly Localization Method
Extended implementation of [AnomalyHop](https://github.com/BinWang28/AnomalyHop).

Original paper: [**AnomalyHop: An SSL-based Image Anomaly Localization Method**](https://arxiv.org/pdf/2105.03797.pdf)

## Datasets
* **MVTec AD**: Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
* **AITEX**: Download from [AITEX website](https://www.aitex.es/afid/)
* **BTAD**: Download from [Kaggle](https://www.kaggle.com/thtuan/btad-beantech-anomaly-detection)

## How to use
In main folder, with AITEX dataset:
```
python ./src/main.py --kernel 7 6 3 2 4 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.2 0.4 0.5 0.1 -d "aitex"
```
The option ```-d``` permits to choose the dataset.
MVTec AD and AITEX datasets will be automatically downloaded. For BTAD, it is necessary to download the archive and copy into ```datasets``` directory.
For more details, consult ```run.sh``` script.
## Reference
[1] Kaitai Zhang, Bin Wang, Wei Wang, Fahad Sohrab, Moncef Gabbouj, C.-C. Jay Kuo. *AnomalyHop: An SSL-based Image Anomaly Localization Method*. https://arxiv.org/abs/2105.03797

[2] https://github.com/BinWang28/AnomalyHop
