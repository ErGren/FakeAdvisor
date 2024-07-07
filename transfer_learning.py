import tensorflow as tf
from tensorflow.keras import applications

from costants import image_size, model_name, models_base_path
from metrics import generate_metrics
# from plot import plot_history
from train import train_model


def make_model(input_shape):
    model = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in model.layers:
        layer.trainable = False

    inputs = model.inputs

    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    model_kind = "scratch"
    model = make_model(input_shape=image_size + (3,))
    model_train_history = train_model(model, model_kind)
    # plot_history(model_train_history)
    print(generate_metrics(f"{models_base_path}/{model_kind}/{model_name}.h5"))


if __name__ == "__main__":
    main()
