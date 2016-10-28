from data_loader import DataLoader
from kohonen_classifier import KohonenClassifier
from params_io import ParamsIO

params_file = "/home/berna/PycharmProjects/RNA/TP2/kohonen.params"
input_file = "/home/berna/PycharmProjects/RNA/TP2/ds/tp2_testing_dataset.csv"
kparams = ParamsIO().load_params(params_file)
classifier = KohonenClassifier(kparams["layout"], kparams["weights"], kparams["categories"])
loader = DataLoader()
fs, ls = loader.LoadData(input_file)

hits = 0
for x,y in zip(fs,ls):
    z = classifier.classify(x)
    hit = (y == z)
    print "Actual: ", y, " Predicted: ", z, " HIT" if hit else " MISS"
    if hit:
        hits = hits + 1

print hits, " from ", len(y), " samples correctly predicted."