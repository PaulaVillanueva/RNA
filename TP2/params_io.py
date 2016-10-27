import numpy as np
import json
from cStringIO import StringIO
class ParamsIO:
    def load_params(self, param_path):
        json_data = open(param_path)
        j = json.load(json_data)
        wstrings = j["weights"]
        weigths = [np.loadtxt(StringIO(s), delimiter=",") for s in wstrings]
        biases = [np.loadtxt(StringIO(s), delimiter=",") for s in bstrings]
        return (weigths, biases)

    def save_params(self, fname, layout, weights, epochs):
        wstrings = {}
        for k in weights.keys():
            s = StringIO()
            np.savetxt(s,weights[k],delimiter=",")
            wstrings[str(k)] = s.getvalue()

        d = {}
        d["layout"] = str(layout)
        d["epochs"] = epochs
        d["weights"] = wstrings

        with open(fname, 'w') as outfile:
            json.dump(d, outfile)

