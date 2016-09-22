import json

from layer import SigmoidLayer, InputLayer
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
          'sigmoid': lambda j: self._create_sigmoid_layer(j),
          'input': lambda j: self._create_input_layer(j)
            }[ltype](jlayer)

    def _create_sigmoid_layer(self, j):
        return SigmoidLayer(int(j["size"]), float(j["beta"]))

    def _create_input_layer(self, j):
        return InputLayer(int(j["size"]))

