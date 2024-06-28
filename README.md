## Automated Seismic Fault Detection Using Computer Vision and Deep Learning Techniques 

### Dataset: 
Thebe (Open-source) - can be accessed at 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK

Wisting - DISKO database (2D SEGY files in the folder - **Data**)

### Code:
All are available in the folder - **Code**  
The subfolder - **custom_classes_defs** contains custom classes definitions for 
- the configuration (setup.py),
- data preprocessing functions (preprocessing.py), and
- graphs defining various models.
   
The subfolder - **weights** contains model weights, training history and info. for some models.  

- *seismic_segy_conversion.ipynb* for converting seismic SEGY format to NumPy format  
- *prepare_patches.ipynb* for splitting 3D seismic volume into 2D patches   
- *stacked_patches.ipynb* for stacking 2D patches and preparing datasets before model training
- *thebe.ipynb* for training any variant of U/F-Net models
- *thebe_hed.ipynb* for transfer learning (HED-style) using VGG16
- *thebe_evaluate.ipynb* for evaluating a model, given the path to the saved model weights
- *visualize_layers.ipynb* example for visualizing model activation from selected layers
- *wisting_restore.ipynb* is an example of model prediction on Wisting data
- *thebe_restore.ipynb* is an example of model prediction on Thebe data









