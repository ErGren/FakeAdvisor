import numpy
import tensorflow as tf
from sklearn.metrics import classification_report
import tensorflow_hub as hub

from costants import image_size, batch_size, test_dir_ai_path, test_dir_real_path


def perform_test(model):
    ai_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_ai_path,
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    predictions_ai = model.predict(ai_test_ds)

    for i in range(predictions_ai.size):
        if predictions_ai[i] >= 0.50:
            predictions_ai[i] = 1
        else:
            predictions_ai[i] = 0

    real_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_real_path,
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    predictions_real = model.predict(real_test_ds)

    for i in range(predictions_real.size):
        if predictions_real[i] >= 0.50:
            predictions_real[i] = 0
        else:
            predictions_real[i] = 1

    y_true = []
    for i in range(predictions_ai.size):
        y_true.append(1)

    for i in range(predictions_real.size):
        y_true.append(0)

    y_pred = numpy.concatenate((predictions_ai, predictions_real), axis=0)

    return classification_report(y_true, y_pred)


def generate_metrics(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return perform_test(model)
