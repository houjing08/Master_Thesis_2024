# ============= Import required packaages ==============
import time

# Import all custom variables and modules
from custom_classes_defs.preprocessing import *
from custom_classes_defs.deeplabv3plus import *

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
        letters.split_data(train_ratio=0.1, val_ratio=1./6, random_state=RND_STATE)

    print("Shape of dataset: {}".format(str(letters.df.shape)))
    print("Shape of training data: {}".format(str(x_train.shape)))
    print("Shape of validation data: {}".format(str(x_val.shape)))
    print("Shape of test data: {}".format(str(x_test.shape)))

    # Model configurations
    conf = model_config(
        epochs=1,
        batch_size=256,
        shuffle=True,
        scaling=1,
        verbose=1,
        save_path='./data',
        augmentation=True,
        pos_label=letters.pos_label
    )

    ### Interactive step:
    if INTERACTIVE_SESSION:
        train = input("New train session? (y/n): ")
        if train[0].lower()=='y':
            conf.new_training_session = True
        else:
            conf.new_training_session = False

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
    conf.info()

    # ================ Build  model ======================
    print("\n\n{}\n\t{}\n{}".format('='*55,f'Build model', '-'*55))
    deeplab = DeeplabV3Plus(model_arch=conf.model_arch)
    model = deeplab.build_model()
    # model.summary()
    # keras.utils.plot_model(model, 'deeplab.png',show_shapes=True)
    print(f"Total trainable wieghts: {model.count_params():,}")


    # ================ Train and evaluate DeepLab3V+ model ======================
    print("\n\n{}\n\t{}\n{}".format('='*55,f'Train {deeplab.Name} model', '-'*55))

    conf.execute_training(model, train_dataset, deeplab.Name, plot_history=INTERACTIVE_SESSION)

    print("\n\n{}\n\t{}\n{}".format('='*55,f'Evaluate {deeplab.Name} model', '-'*55))
    decoded_imgs = model.predict(x_test)
    deeplab.evaluate_sklearn(y_test, decoded_imgs,report=True)
    display_sample_images(x_test, decoded_imgs, conf.img_shape)