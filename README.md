# ROI & Interval-Adaptive Adversarial Attack (RIA^3)
Official code repository of "Adaptive Adversarial Data Augmentation with Trajectory Constraint for Alzheimer’s Disease Conversion Prediction" published at MICCAI 2025 (early accepted).
The open review and the paper can be found [here](https://papers.miccai.org/miccai-2025/0045-Paper2850.html).

## Contribution 
1) We propose a novel end-to-end framework for synthesizing small-sized pMCI data and predicting 
their AD conversion.
2) By using adversarial attacks with adaptive steps and ROI-wise trainable perturbation intensities, realistic samples are augmented while preserving individual and brain regional heterogeneity in neurodegeneration.
3) We introduce trajectory consistency regularization that ensures the augmented data follow plausible disease progression. 
As a result, the synthesized data preserve biological plausibility and enhance downstream predictive performance, outperforming six augmentation and generative methods across two AD biomarkers and three classifiers.

## Datasets
The ADNI data are available through [this link](https://adni.loni.usc.edu/).
To obtain FDG and Amyloid SUVR, PET scans were parcellated into 148 brain regions based on the Destrieux atlas, and skull stripping, tissue segmentation, and image registration were performed using Freesurfer.
The cerebellum was used as the reference region to calculate the SUVR.

## Running Experiments
The model is implemented for single-GPU running. 

To train the model from scratch (i.e., pMCI augmentation + sMCI/pMCI classification), use the following command:
```
python main.py 
```


## Citation
If you would like to cite our paper, please use the BibTeX below.

```
@InProceedings{ChoHyu_Adaptive_MICCAI2025,
        author = { Cho, Hyuna and Ahn, Hayoung and Wu, Guorong and Kim, Won Hwa},
        title = { { Adaptive Adversarial Data Augmentation with Trajectory Constraint for Alzheimer’s Disease Conversion Prediction } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15966},
        month = {September},
        page = {13 -- 23}
}
}
```

