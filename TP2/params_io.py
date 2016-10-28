import numpy as np
import json
from cStringIO import StringIO


class ParamsIO:
    def load_params(self, param_path):
        json_data = open(param_path)
        j = json.load(json_data)
        epochs = int(j["epochs"])
        json_weights = j["weights"]
        #json_categories = j["categories"]
        layout = self.str_to_tuple(str(j["layout"]))
        weights = { self.str_to_tuple(str(k)):np.loadtxt(StringIO(v), delimiter=",") for k,v in json_weights.iteritems()}
        #categories = { self.str_to_tuple(k):np.loadtxt(StringIO(v), delimiter=",") for k,v in json_categories.iteritems()}
        categories={}
        return {
                    "weights": weights,
                    "categories": categories,
                    "layout": layout,
                    "epochs": epochs
                }

    def save_params(self, fname, layout, weights, category_map, epochs):
        wstrings = {}
        category_map_json = {}
        for k in weights.keys():
            s = StringIO()
            np.savetxt(s,weights[k],delimiter=",")
            wstrings[str(k)] = s.getvalue()
            category_map_json[str(k)] = category_map[k]

        d = {}
        d["layout"] = str(layout)
        d["epochs"] = epochs
        d["weights"] = wstrings
        d["categories"] = category_map_json

        with open(fname, 'w') as outfile:
            json.dump(d, outfile)

    def str_to_tuple(self, str_tuple):
        vaux = (str.replace(str.replace(str_tuple, '(', ''), ')','')).split(',')
        return (int(vaux[0]),int(vaux[1]))