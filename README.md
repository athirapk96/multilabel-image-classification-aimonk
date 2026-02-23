# Multilabel Image Classification using Transfer Learning

## Problem Statement

This project focuses on solving a multilabel image classification problem where each image may contain multiple attributes simultaneously. The objective is to build a deep learning model capable of predicting the presence or absence of four different attributes for a given input image.

The dataset consists of:
- A folder containing input images.
- A `labels.txt` file containing annotations for four attributes associated with each image.

Each attribute is represented as:
- **1** → Attribute is present  
- **0** → Attribute is absent  
- **NA** → Attribute information is not available  

The primary goal is to train a multilabel classification model that can accurately predict the attributes present in an image.

---

## Approach

A transfer learning based deep learning approach was adopted to address this problem efficiently.

A pretrained **ResNet50** model, trained on the ImageNet dataset, was used as the backbone for feature extraction. Instead of training a model from scratch, the pretrained network was fine-tuned to adapt it to the multilabel classification task.

To support multilabel prediction, the final classification layer was replaced with a fully connected dense layer using **sigmoid activation**, allowing the model to independently predict the presence of each attribute.

---

## Handling Missing Labels (NA)

As specified in the assignment instructions, images containing missing attribute information (NA values) were not removed from the dataset.

A masking strategy was implemented to ensure that:
- Images with incomplete annotations are still used for training.
- Loss is computed only for attributes with known labels.
- Unknown attributes do not negatively impact model learning.

During training, loss was calculated using a masked formulation:
Loss = Binary Cross Entropy × Class Weight × Mask


This allows the model to learn from available attribute information without discarding valuable training samples.

---

## Handling Class Imbalance

The dataset exhibits class imbalance across different attributes, meaning some attributes appear significantly less frequently than others.

To address this issue, attribute-wise class weights were computed based on the ratio of negative to positive samples for each attribute. These weights were incorporated into a custom **Weighted Binary Cross Entropy Loss** function.

This approach ensures that:
- Misclassification of rare attributes is penalized more heavily.
- The model does not become biased towards majority classes.
- Minority attributes are learned more effectively.

---

## Training Details

- Framework Used: TensorFlow / Keras  
- Backbone Model: ResNet50 (ImageNet pretrained)  
- Input Image Size: 224 × 224  
- Batch Size: 32  
- Optimizer: Adam  
- Learning Rate: 1e-4  
- Output Activation: Sigmoid  
- Loss Function: Masked Weighted Binary Cross Entropy  
- Early Stopping was used to reduce overfitting due to the limited dataset size  

---

## Deliverables

The following outputs were generated as part of the assignment:

- Trained Model File: `aimonk_multilabel_model.h5`  
- Training Loss Curve
- Inference function to predict attributes for unseen images  

---

## Possible Improvements

Due to time constraints, the following enhancements were not implemented but may further improve model performance:

- Focal Loss Can be used instead of weighted BCE to better handle extreme class imbalance by focusing learning on hard-to-classify samples.
- Threshold tuning for individual attributes  
- Learning Rate Scheduling: Using adaptive learning rate strategies such as ReduceLROnPlateau may improve convergence.
- Multilabel stratified train-validation split  
- Data augmentation techniques (rotation, flipping, brightness adjustment)  
- Mean Average Precision (mAP) for performance evaluation  
- Per-Attribute ROC-AUC Score: Helps evaluate model performance independently for each attribute.
- Confusion Matrix per Attribute: Can provide insight into misclassification patterns.

---

## Conclusion

A multilabel image classification system was successfully implemented using transfer learning. Missing labels were handled using a masking strategy, while class imbalance was addressed through weighted loss computation. This ensured effective learning from the available data without discarding images with incomplete attribute information.
