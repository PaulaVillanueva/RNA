import json

from layer import SigmoidLayer, InputLayer, ReluLayer, TanhLayer, LinearLayer
from layer_model import LayerModel


class ModelIO:
    def load_model(self, model_path):
        json_data = open(model_path)
        j = json.load(json_data)
        layers = [self.create_layer(l) for l in j['layers']]
        return LayerModel(layers)

    def create_layer(self, jlayer):
         ltype = jlayer["type"]

         return {
          'tanh': lambda j: self._create_tanh_layer(j),
          'relu': lambda j: self._create_relu_layer(j),
          'linear': lambda j: self._create_linear_layer(j),
          'sigmoid': lambda j: self._create_sigmoid_layer(j),
          'input': lambda j: self._create_input_layer(j)
            }[ltype](jlayer)

    def _create_sigmoid_layer(self, j):
        return SigmoidLayer(int(j["size"]), float(j["beta"]))

    def _create_input_layer(self, j):
        return InputLayer(int(j["size"]))


    def _create_relu_layer(self, j):
        return ReluLayer(int(j["size"]), float(j["beta"]))

    def _create_linear_layer(self, j):
        return LinearLayer(int(j["size"]), float(j["beta"]))

    def _create_tanh_layer(self, j):
        return TanhLayer(int(j["size"]),float(j["beta"]))

