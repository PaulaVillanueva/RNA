#!/usr/bin/python
# -*- coding: latin-1 -*-
import argparse

from heat_map import HeatMap
from data_loader import DataLoader
from kohonen_classifier import KohonenClassifier
from params_io import ParamsIO

print "Kohonen Classifier by P. Villanueva, V. Uhrich, B. Panarello"
parser = argparse.ArgumentParser(description='Parametros de la red')


parser.add_argument('-p', type=str,
                    help='Ruta del archivo de entrada con los pesos', required=True)

parser.add_argument('-x', type=str, default=1,
                    help='Archivo de features', required=True)

args = parser.parse_args()

params_file = args.p
input_file = args.x
kparams = ParamsIO().load_params(params_file)

print "Reading weights...."

classifier = KohonenClassifier(kparams["layout"], kparams["weights"], kparams["categories"])
print "done!"
print "Showing category HEATMAP:"
print HeatMap().show_from_category_dictionary(kparams["categories"], kparams["layout"])
print "Loading samples..."
loader = DataLoader()
fs, ls = loader.LoadData(input_file)
print "Starting classification...."
hits = 0
for x,y in zip(fs,ls):
    z = classifier.classify(x)
    hit = (y == z)
    print "Actual: ", y, " Predicted: ", z, " HIT" if hit else " MISS"
    if hit:
        hits = hits + 1

print hits, " from ", len(ls), " samples correctly predicted."
print "Classification accuracy: " , (float(hits) / len(ls)) * 100, "%."