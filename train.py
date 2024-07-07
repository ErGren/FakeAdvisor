import tensorflow as tf

from costants import train_dir_path, image_size, batch_size, epochs, model_name, models_base_path


def train_model(model, kind):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir_path,
        validation_split=0.2,
        subset="training",
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir_path,
        validation_split=0.2,
        subset="validation",
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{models_base_path}/{kind}/{model_name}.h5"),
        # tf.keras.callbacks.ModelCheckpoint(f"{models_base_path}/{kind}/{model_name}_{{epoch}}.h5"),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model_train_history = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, shuffle=True,
    )

    return model_train_history
