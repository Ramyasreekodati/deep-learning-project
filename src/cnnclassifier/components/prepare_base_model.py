import os
import urllib.request as request
from zipfile import ZipFile
from cnnclassifier import logger
import tensorflow as tf
from pathlib import Path
from cnnclassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config : PrepareBaseModelConfig):
        self.config = config
        self.model = None  # Ensure model is initialized

    def get_base_model(self):
        """Loads the base VGG16 model and saves it."""
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Modifies the base model by adding a fully connected layer."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False  # Fix: Ensure each layer is frozen
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:freeze_till]:  # Fix: Correct freeze_till logic
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """Updates the base model with a new fully connected layer."""
        if self.model is None:
            raise ValueError("Base model is not initialized. Call get_base_model() first.")

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the specified path."""
        model.save(path)




