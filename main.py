import numpy as np
from tensorflow import keras
from load_bird_images import load_birds
from plot_birds import inspect_birds, inspect_bird_results
from models import get_bird_model

def train(birds, class_numbers, class_names):
    #load model
    model = get_bird_model(num_classes= np.max(class_numbers) + 1, image_size= np.shape(birds)[1:4])

    #compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #fit model
    data_gen_args = dict(
                        rotation_range=5,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=1
                        )
    datagen = keras.preprocessing.image.ImageDataGenerator(data_gen_args)
    datagen.fit(birds)
    model.fit_generator(
        generator=datagen.flow(birds, class_numbers, batch_size=10),
        steps_per_epoch=100,
        epochs=5,
        #verbose=1,
        #callbacks=None,
        #validation_data=None,
        #validation_steps=None,
        #validation_freq=1,
        #class_weight=None,
        #max_queue_size=10,
        #workers=1,
        #use_multiprocessing=False,
        shuffle=True,
        #initial_epoch=0
        )

    return model

if __name__ == '__main__':
    birds, class_numbers, class_names = load_birds(size = (200,200))
    inspect_birds(birds, class_numbers, class_names)
    model = train(birds, class_numbers, class_names)
    inspect_bird_results(model, birds, class_numbers, class_names)
