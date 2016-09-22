import numpy as np
import json
from cStringIO import StringIO
class ParamsIO:
    def load_params(self, param_path):
        json_data = open(param_path)
        j = json.load(json_data)
        wstrings = j["weights"]
        bstrings = j["biases"]
        weigths = [np.loadtxt(StringIO(s), delimiter=",") for s in wstrings]
        biases = [np.loadtxt(StringIO(s), delimiter=",") for s in bstrings]
        return (weigths, biases)

    def save_params(self, fname, weights, biases):
        wstrings = []
        bstrings = []
        for w in weights:
            s = StringIO()
            np.savetxt(s,w,delimiter=",")
            wstrings.append(s.getvalue())


        for b in biases:
            s = StringIO()
            np.savetxt(s,b,delimiter=",")
            bstrings.append(s.getvalue())

        d = {}
        d["weights"] = wstrings
        d["biases"] = bstrings

        with open(fname, 'w') as outfile:
            json.dump(d, outfile)

