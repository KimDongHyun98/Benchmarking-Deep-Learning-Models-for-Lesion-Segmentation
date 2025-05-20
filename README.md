# Benchmarking Deep Learning Models for Lesion Segmentation (2022)

Selecting an appropriate deep learning model for lesion segmentation can be challenging, especially given the diverse types of medical images and varying user requirements (e.g., efficiency, performance, memory capacity, or training time). 

In this study, we benchmarked nine deep learning models on six different medical datasets to address these challenges.

## Publication  
This research was presented at the *Korea Information Science Society Conference (KSC) 2022* and ranked within the top 10% of submissions in the Thesis Contest.  

**Authors:** Inho Jeong, Donghyun Kim, Seongmin Jo, Hongryul Ahn  
**Paper Title:** *Comparison of lesion segmentation deep learning models according to medical image types*  
[Read the Paper](https://drive.google.com/file/d/1KN5KA6rhCA3yZqFSPaHfTkYoPgjrT2uC/view?usp=sharing)

# Datasets   
We selected datasets based on the following criteria:   
1. **Scar/Wound Images** 
- [ISIC-2017](https://challenge.isic-archive.com/data/#2017)   
- [Foot Ulcer](https://paperswithcode.com/dataset/dfuc2021)

2. **Endoscopic Images (Polyp)**
- [CVC-ClinicDB](https://paperswithcode.com/dataset/cvc-clinicdb)   
- [Kvasir-SEG](https://paperswithcode.com/dataset/kvasir-seg)   

3. **Ultrasound Images** 
- [benign/malignant Breast Ultrasound](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)

# Models   
We categorized the models as follows:  
1. Medical Segmentation Models
   - [UNet](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical) : A widely-used architecture for biomedical image segmentation.
   - [UNet++](https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical) : An enhanced version of UNet with nested skip pathways.
   - [ColonSegNet](https://paperswithcode.com/paper/real-time-polyp-detection-localisation-and) : Designed for real-time polyp detection and segmentation

2. General Segmentation Models
   - [DeeplabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) : A state-of-the-art model for semantic segmentation.
   - [FCN](https://paperswithcode.com/method/fcn) : A fully convolutional network for image segmentation.
   - [SegNet](https://paperswithcode.com/method/segnet) : An encoder-decoder architecture for semantic segmentation.

3. SOTA Models (State-of-the-Art)
   - [ColonFormer](https://paperswithcode.com/paper/colonformer-an-efficient-transformer-based) : A transformer-based model for colon polyp segmentation.
   - [ESFPNet](https://paperswithcode.com/paper/esfpnet-efficient-deep-learning-architecture) : A lightweight and efficient segmentation model.
   - [FCBFormer](https://paperswithcode.com/paper/fcn-transformer-feature-fusion-for-polyp) : Combines transformer and convolutional features for segmentation.

# Experiments
## Data Preprocessing
All images were resized to **224×224** for uniformity. 
All datasets, except the ISIC-2017, were split into train, validation, and test sets with a **6:2:2** ratio.

| Dataset                              | Train set | Validation set | Test set |
|--------------------------------------|-----------|----------------|----------|
| ISIC Challenge                       | 2,000     | 150            | 150      |
| Wound                                | 831       | 278            | 278      |
| CVC-ClinicDB                         | 366       | 123            | 123      |
| Kvasir-SEG                           | 600       | 200            | 200      |
| Breast Ultrasound (Benign Tumors)    | 261       | 88             | 88       |
| Breast Ultrasound (Malignant Tumors) | 126       | 42             | 42       |

## Evaluation Metrics
1. **Dice Score**: Measures overlap between predicted and ground truth masks.

$$ Dice Score = \frac{2 \times |Y \cap Y_{pred}|}{|Y \cup Y_{pred}|} $$

2. **DiceBCELoss**: Combines Dice loss and Binary Cross Entropy.

$$ Dice BCELoss = (1 - Dice Score) + Binary Cross Entropy $$

## Training Configuration
- **Framework**: PyTorch  
- **Batch Size**: 8  
- **Learning Rate**: 1e-4  
- **Weight Decay**: 1e-8  
- **Optimizer**: Adam
- **Epochs**: 100 (with early stopping after 20 epochs without validation improvement)

## Results
| Model          | ISIC Challenge | Wound   | Kvasir-SEG | CVC-ClinicDB | Breast Ultrasound (Benign Tumors) | Breast Ultrasound (Malignant Tumors) |
|----------------|----------------|---------|------------|--------------|------------------------------------|--------------------------------------|
| FCN (resnet101)       | 0.8477         | 0.8310  | 0.8779     | 0.8767       | 0.8128                             | 0.7531                               |
| SegNet (resnet101)    | 0.8328         | 0.8470  | 0.8672     | 0.8422       | 0.8013                             | 0.7415                               |
| DeepLab V3+ (resnet101) | 0.8519       | 0.8242  | 0.8616     | 0.8712       | 0.8350                             | 0.7720                               |
| U-Net          | 0.8295         | 0.8304  | 0.7891     | 0.7622       | 0.7779                             | 0.6687                               |
| U-Net++        | 0.8350         | 0.8100  | 0.7515     | 0.7289       | 0.7669                             | 0.7089                               |
| ColonSegNet    | 0.8199         | 0.8377  | 0.7471     | 0.6896       | 0.7048                             | 0.6455                               |
| FCBFormer      | 0.8642         | 0.8406  | 0.9830     | 0.8958       | 0.8583                             | 0.7974                               |
| ESFPNet-L      | 0.8541         | 0.8077  | 0.9804     | 0.8709       | 0.8210                             | 0.8050                               |
| ColonFormer    | 0.8456         | 0.8393  | 0.9773     | 0.8547       | 0.7945                             | 0.7984                               |

### Insights
1. **FCBFormer** achieved the highest average Dice score across datasets, demonstrating its effectiveness for lesion segmentation.  
2. Lightweight models like **ESFPNet-L** performed exceptionally well on polyp segmentation datasets (Kvasir-SEG, CVC-ClinicDB).  
3. Traditional models (e.g. UNet, UNet++) showed lower performance compared to modern transformer-based architectures.
