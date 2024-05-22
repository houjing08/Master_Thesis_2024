#!/usr/bin/env python
# coding: utf-8

# In[1]:

from IPython import get_ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last" # all | last | last_expr | none 


# In[3]:


# for name in dir():
#     if not name.startswith('_'):
#         del globals()[name]


# In[4]:


# ============= Import required packaages ==============
import time

# Import all custom variables and modules
from custom_classes_defs.preprocessing import *
# from custom_classes_defs.unet0 import * 
# from custom_classes_defs.Unet_like import *  
# from custom_classes_defs.unet import *  
from custom_classes_defs.fnet0 import *
# from custom_classes_defs.fnet_like import *
# from custom_classes_defs.fnet import *

RND_STATE = 247
BATCH_SIZE = 512
keras.utils.set_random_seed(RND_STATE)
from keras.utils import plot_model

INTERACTIVE_SESSION = False

# -------------------------------------------------------


# In[ ]:


# Verify tensorflow/keras versions
print(f"tensorflow version: {tf.__version__}")
print(f"keras version: {keras.__version__}")

# Verify CPU/GPU availability
print(tf.config.list_physical_devices())
NUM_GPU = len(tf.config.list_physical_devices('GPU'))
print(f"Number of GPUs assigned for computation: {NUM_GPU}")

# if NUM_GPU:
#     # print GPU info
#     get_ipython().system('nvidia-smi')


# ### Data preparation and model configurations

# In[ ]:


print("{}\n\t{}\n{}".format('='*55,'Data preparation and model configurations', '-'*55))
# Images and annations for Thebe seismic data
start = time.time()
img_url = '../thebe_new/seismic'
target_url = '../thebe_new/fault'
seis = Thebe(img_url, target_url)
    
# Create datasets for respective data samples and labels
train_dataset = seis.data_generator('train', batch_size=BATCH_SIZE, cache=NUM_GPU) 
val_dataset = seis.data_generator('val', batch_size=BATCH_SIZE, cache=NUM_GPU) 
x_test, y_test = seis.data_generator('test', as_numpy=True)
        
print("Train Dataset:", train_dataset)

print("Size of training data: {}".format(seis.train_size))
print("Size of validation data: {}".format(seis.validation_size))
print("Size of test data: {}".format(seis.test_size)) 

print('...elapsed time: ___{:5.2f} minutes___'.format((time.time()-start) / 60))


# In[ ]:


if INTERACTIVE_SESSION:
    # X, y = next(train_dataset.as_numpy_iterator())
    # # X, y = next(val_dataset.as_numpy_iterator())
    # seis.display_sample_images(X, y, num_images=4)
    # seis.display_sample_images(X, y, num_images=4)

    seis.display_sample_images(x_test, y_test, num_images=4)
    seis.display_sample_images(x_test, y_test, num_images=4)


# In[ ]:


# Model configurations
conf = model_config(
    epochs=100,
    batch_size=BATCH_SIZE,
    shuffle=True,
    scaling=1,
    save_path='./Thebe/fnet0',
    img_shape=seis.img_size,
    target_size=seis.img_size,
    threshold=seis.threshold,
    pos_label=seis.pos_label,
    train_size=seis.train_size,
    test_size=seis.test_size,
    new_training_session=True,
    multiple_gpu_device=(NUM_GPU>1),
    validation_size=seis.validation_size
)

callbacks = conf.callbacks(
    chkpt_monitor='val_f1_score', 
    es_monitor='val_loss',
    es_patience=100, 
    lr_monitor='val_loss',
)

conf.set( validation_data=val_dataset,  callbacks=callbacks )
m1 = f1_score(positive_label=seis.pos_label, threshold=seis.threshold)
conf.set(
    'compile',
    metrics= ['accuracy', m1]
)

# conf.double_check(INTERACTIVE_SESSION)
conf.info()


# ### Build  model 

# In[ ]:


print("\n\n{}\n\t{}\n{}".format('='*55,f'Build model', '-'*55))

if conf.multiple_gpu_device:

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    with strategy.scope():

        conf = model_config(
            epochs=100,
            batch_size=BATCH_SIZE,
            shuffle=True,
            scaling=1,
            save_path='./Thebe/fnet0',
            img_shape=seis.img_size,
            target_size=seis.img_size,
            threshold=seis.threshold,
            pos_label=seis.pos_label,
            train_size=seis.train_size,
            test_size=seis.test_size,
            new_training_session=True,
            multiple_gpu_device=(NUM_GPU>1),
            validation_size=seis.validation_size
        )

        callbacks = conf.callbacks(
            chkpt_monitor='val_f1_score', 
            es_monitor='val_loss',
            es_patience=100, 
            lr_monitor='val_loss',
        )

        conf.set( validation_data=val_dataset,  callbacks=callbacks )
        m1 = f1_score(positive_label=seis.pos_label, threshold=seis.threshold)
        conf.set(
            'compile',
            metrics= ['accuracy', m1]
        )

        # m_obj = UNET2D(panel_sizes=[32,64,128,256], model_arch=conf.model_arch)
        m_obj = FNET2D(panel_sizes=[32,64,128,256], model_arch=conf.model_arch)
        model = m_obj.build_model()
        model.compile(**conf.compile_args)

else:

    # m_obj = UNET2D(panel_sizes=[32,64,128,256], model_arch=conf.model_arch)
    m_obj = FNET2D(panel_sizes=[32,64,128,256], model_arch=conf.model_arch)
    model = m_obj.build_model()
    model.compile(**conf.compile_args)


# model.summary()
# keras.utils.plot_model(model, 'm_obj.png',show_shapes=True)
# plot_model(model, 'm_obj.png',show_shapes=True)
num_trainable_weights = sum([np.prod(w.shape) for w in model.trainable_weights])
print(f"Total number of parameters: {model.count_params():,}")
print(f"Total trainable wieghts: {num_trainable_weights:,}")
print(f"Total non-trainable wieghts: {model.count_params()-num_trainable_weights:,}")



# ### Train  model

# In[ ]:


print("\n\n{}\n\t{}\n{}".format('='*55,f'Train {m_obj.Name} model', '-'*55))

model, train_history = \
    conf.execute_training(
        model, 
        data=train_dataset, 
        plot_history=INTERACTIVE_SESSION
)


# In[ ]:


if INTERACTIVE_SESSION:
    show_convergence(train_history.history, ['accuracy','val_accuracy'])


# In[ ]:


if INTERACTIVE_SESSION:
    show_convergence(train_history.history, ['f1_score','val_f1_score'])


# In[ ]:


if INTERACTIVE_SESSION:
    show_convergence(train_history.history, 'lr')


# ### Evaluate and Vizualize

# In[ ]:


# print("\n\n{}\n\t{}\n{}".format('='*55,f'Evaluate {m_obj.Name} model', '-'*55))
# y_pred = model.predict(x_test)


# In[ ]:


# if INTERACTIVE_SESSION:
#     seis.display_sample_images(x_test, y_test, y_pred)
    


# In[ ]:


# model.evaluate(x=x_test)


# In[ ]:


# scores = conf.evaluate_sklearn(y_test, y_pred,report=True)
# print(scores)

