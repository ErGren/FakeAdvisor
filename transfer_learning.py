import tensorflow as tf
from tensorflow.keras import applications

from costants import image_size
# from plot import plot_history
from train import train_model


def get_mixed_model(input_shape):
    model = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in model.layers:
        layer.trainable = False

    inputs = model.inputs

    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    model = get_mixed_model(input_shape=image_size + (3,))
    model_train_history = train_model(model, "transfer")
    # plot_history(model_train_history)


if __name__ == "__main__":
    main()
