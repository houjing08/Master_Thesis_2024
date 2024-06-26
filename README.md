## Automated Seismic Fault Detection Using Computer Vision and Deep Learning Techniques 

### Dataset: 
Thebe (Open-source) - can be accessed at 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK

Wisting - DISKO database (2D SEGY files in the folder - Data)

### Code:
can be found in the folder - Final  
folder - custom_classes_defs includes U-Net, F-Net model variants, model setup, and preprocessing functions.  

seismic_segy_conversion.ipynb for converting seismic SEGY format to NumPy format  
prepare_patches.ipynb for splitting 3D seismic volume int0 2D patches   
stacked_patches.ipynb for stacking 2D patches and preparing datasets stacking before model training  
pred_wisting_restore.ipynb is an example for model prediction on Wisting data  









