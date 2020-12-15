import tensorflow   # type: ignore
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
        model_type = config['model']['type']
        model_class = self.dynamically_import_class(model_type)
        model_layers = []
        for layer in config['model']['layers']:
            layer_type = layer['type']
            layer_class = self.dynamically_import_class(layer_type)
            parameters = layer['parameters']
            layer_instance = layer_class(**parameters)  # type: ignore
            model_layers.append(layer_instance)
        model_instance = model_class()  # type: ignore
        for l in model_layers:
            model_instance.add(l)
        self.model = model_instance

    def compile(self):
        model_compile_parameters = self.config['model_compile']
        metrics = model_compile_parameters['metrics'] \
            if 'metrics' in model_compile_parameters else None
        metrics_class_list = []
        for m in metrics:
            if m['type'].startswith('tensorflow.keras'):
                m_class = self.dynamically_import_class(m['type'])
                m_parameters = m['parameters']
                m_instance = m_class(**m_parameters)
                metrics_class_list.append(m_instance)
            else:
                metrics_class_list.append([m['type']])
        self.model.compile(
            loss=model_compile_parameters['loss'],
            optimizer=model_compile_parameters['optimizer'],
            metrics=metrics_class_list
        )

    def fit(self, train_data, validation_data):
        """Takes a model, generated from make_model(), and calls .fit()

        Args:
            train_data: training data for fitting
            validation_data: validation data for fitting

        Returns:
            The model object

        """
        try:
            classifier_params = self.config['classifier']['model_fit']['parameters']
        except KeyError:
            classifier_params = {}

        callback_list = []
        if 'callbacks' in classifier_params:
            for callback in classifier_params['callbacks']:
                c_class = self.dynamically_import_class(callback['type'])
                c_params = callback[
                    'parameters'] if 'parameters' in callback else {}
                c_instance = c_class(**c_params)
                callback_list.append(c_instance)
            del classifier_params['callbacks']

        self.model.fit(*train_data, validation_data=validation_data,
                       **classifier_params, callbacks=callback_list)


