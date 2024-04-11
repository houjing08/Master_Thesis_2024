# ============= Import required packaages ==============
import time

# Import all custom variables and modules
from custom_classes_defs.preprocessing import *
# from custom_classes_defs.Unet_like import *   
from custom_classes_defs.fnet import *

RND_STATE = 247
keras.utils.set_random_seed(RND_STATE)

INTERACTIVE_SESSION = True

from tensorflow.data import Dataset as tf_data
# -------------------------------------------------------

if __name__ == '__main__':

    # === Data preparation and model configurations ===
    print("{}\n\t{}\n{}".format('='*55,'Data preparation and model configurations', '-'*55))
    # Hand-written alphabet images
    start = time.time()
    if 'letters' not in dir():
        letters = MNIST_letters(
            './data/az_handwritten_alphabets_in_csv_format/A_Z_Handwritten_Data.csv'
        )
        
    # Load data (split ratio = train:val:test = 5:1:4)
    (x_train, _), (x_test, _), (x_val, _) = \
        letters.split_data(train_ratio=0.1, val_ratio=1./6)

    print("Shape of dataset: {}".format(str(letters.df.shape)))
    print("Shape of training data: {}".format(str(x_train.shape)))
    print("Shape of validation data: {}".format(str(x_val.shape)))
    print("Shape of test data: {}".format(str(x_test.shape)))

    # Model configurations
    conf = model_config(
        epochs=10,
        batch_size=256,
        shuffle=True,
        scaling=1,
        verbose=1,
        save_path='./data',
        augmentation=False,
        pos_label=letters.pos_label
    )

    ### Interactive step:
    if INTERACTIVE_SESSION:
        train = input("New train session? (y/n): ")
        if train.lower()=='y':
            conf.new_training_session = True
        else:
            conf.new_training_session = False
        interact = input("Are you sure, you want to run this session interactively? (y/n): ")
        if interact[0].lower()!='y':
            INTERACTIVE_SESSION = False


    # Create data generator for respective data samples and labels
        
    train_dataset = tf_data.from_tensor_slices(letters.generate_data(x_train)) \
                    .batch(conf.training_args['batch_size'])
    val_dataset = tf_data.from_tensor_slices(letters.generate_data(x_val)) \
                    .batch(conf.training_args['batch_size'])
    x_test, y_test = letters.generate_data(x_test)
    print("Train Dataset:", train_dataset)
    print('...elapsed time: ___{:5.2f} minutes___'.format((time.time()-start) / 60))


    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    conf.set( validation_data=val_dataset,  callbacks=[es_callback] )
    conf.set('compile')
    conf.threshold = letters.threshold
    conf.info()


    # ================ Build  model ======================
    print("\n\n{}\n\t{}\n{}".format('='*55,f'Build model', '-'*55))
    m_obj = FNET2D(panel_sizes=[32,64,128,256], model_arch=conf.model_arch)
    model = m_obj.build_model()
    model.summary()
    # keras.utils.plot_model(model, 'm_obj.png',show_shapes=True)
    print(f"Total trainable wieghts: {model.count_params():,}")


    # ================ Train and evaluate  model ======================
    print("\n\n{}\n\t{}\n{}".format('='*55,f'Train {m_obj.Name} model', '-'*55))

    conf.execute_training(
        model, 
        data=train_dataset, 
        saveas=m_obj.Name, 
        plot_history=INTERACTIVE_SESSION
    )

    print("\n\n{}\n\t{}\n{}".format('='*55,f'Evaluate {m_obj.Name} model', '-'*55))
    decoded_imgs = model.predict(x_test)
    m_obj.evaluate_sklearn(y_test, decoded_imgs,report=True)

    if INTERACTIVE_SESSION:
        display_sample_images(x_test, decoded_imgs, conf.img_shape)
