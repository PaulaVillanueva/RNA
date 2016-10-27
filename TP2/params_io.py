import numpy as np
import json
from cStringIO import StringIO
from kohonen_category_mapper import KohonenCategoryMapper


class ParamsIO:
    def load_params(self, param_path):
        json_data = open(param_path)
        j = json.load(json_data)
        json_weights = j["weights"]
        weigths = { np.loadtxt(StringIO(v), delimiter=",")}
        categories = j["categories"]
        weigths = [np.loadtxt(StringIO(s), delimiter=",") for s in wstrings]
        biases = [np.loadtxt(StringIO(s), delimiter=",") for s in bstrings]
        return (weigths, biases)

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

