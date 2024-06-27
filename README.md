## Automated Seismic Fault Detection Using Computer Vision and Deep Learning Techniques 

### Dataset: 
Thebe (Open-source) - can be accessed at 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK

Wisting - DISKO database (2D SEGY files in the folder - **Data**)

### Code:
All are available in the folder - **Final**  
The subfolder - **custom_classes_defs** includes U-Net, F-Net model variants, model setup, and preprocessing functions. 
The subfolder - **weights** contains the loaded weights for all models. 

- seismic_segy_conversion.ipynb for converting seismic SEGY format to NumPy format  
- prepare_patches.ipynb for splitting 3D seismic volume into 2D patches   
- stacked_patches.ipynb for stacking 2D patches and preparing datasets before model training
- training_unetlike.ipynb is an example of model training  
- pred_wisting_restore.ipynb is an example of model prediction on Wisting data









