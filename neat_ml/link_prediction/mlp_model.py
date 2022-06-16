import os
import pickle
import tensorflow as tf  # type: ignore
from .model import Model


class MLPModel(Model):
    def __init__(self, config, outdir: str = None) -> None:
        """make an MLP model

        Args:
            config: the classifier config

        Returns:
            The model

        """
        super().__init__(outdir=outdir)
        self.config = config
        model_type = config["classifier_type"]
        model_class = self.dynamically_import_class(model_type)
        model_layers = []
        for layer in config["parameters"]["tf_keras_params"]["layers_config"]["layers"]:
            layer_type = layer["type"]
            layer_class = self.dynamically_import_class(layer_type)
            parameters = layer["parameters"]
            layer_instance = layer_class(**parameters)  # type: ignore
            model_layers.append(layer_instance)
        model_instance = model_class()  # type: ignore
        for l in model_layers:
            model_instance.add(l)
        self.model = model_instance

    def compile(self):
        model_compile_parameters = self.config["parameters"]["tf_keras_params"]
        metrics = (
            model_compile_parameters["metrics_config"]["metrics"]
            if "metrics_config" in model_compile_parameters
            else None
        )
        metrics_class_list = []
        for m in metrics:
            if m["type"].startswith("tensorflow.keras"):
                m_class = self.dynamically_import_class(m["type"])
                m_parameters = m["parameters"]
                m_instance = m_class(**m_parameters)
                metrics_class_list.append(m_instance)
            else:
                metrics_class_list.append([m["type"]])
        self.model.compile(
            loss=model_compile_parameters["loss"],
            optimizer=model_compile_parameters["optimizer"],
            metrics=metrics_class_list,
        )

    def fit(self, train_data, train_labels):
        """Takes a model, generated from make_model(), and calls .fit()

        Args:
            train_data: training data for fitting
            validation_data: validation data for fitting

        Returns:
            The model object

        """
        try:
            classifier_params = self.config["parameters"]["tf_keras_params"][
                "fit_config"
            ]
        except KeyError:
            classifier_params = {}

        callback_list = []
        if "callbacks_list" in classifier_params:
            for callback in classifier_params["callbacks_list"]["callbacks"]:
                c_class = self.dynamically_import_class(callback["type"])
                c_params = (
                    callback["parameters"] if "parameters" in callback else {}
                )
                c_instance = c_class(**c_params)
                callback_list.append(c_instance)
            del classifier_params["callbacks"]

        history = self.model.fit(
            train_data,
            train_labels,
            **classifier_params,
            callbacks=callback_list
        )
        return history

    def save(self) -> None:
        self.model.save(
            os.path.join(self.outdir, self.config["outfile"])
        )

        fn, ext = os.path.splitext(self.config["outfile"])
        model_outfile = fn + "_custom" + ext

        with open(os.path.join(self.outdir, model_outfile), "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> tuple(): # type: ignore
        fn, ext = os.path.splitext(path)
        custom_model_filename = fn + "_custom" + ext
        generic_model_object = tf.keras.models.load_model(path)

        with open(custom_model_filename, "rb") as mf2:
            custom_model_object = pickle.load(mf2)

        return generic_model_object, custom_model_object
