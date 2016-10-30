#!/usr/bin/python
# -*- coding: latin-1 -*-
import argparse

from heat_map import HeatMap
from data_loader import DataLoader
from kohonen_classifier import KohonenClassifier
from params_io import ParamsIO


parser = argparse.ArgumentParser(description='Parametros')


parser.add_argument('-p', type=str,
                    help='Ruta del archivo de entrada con los pesos para mostrar el mapa', required=True)


args = parser.parse_args()

params_file = args.p

print "Reading weights...."

kparams = ParamsIO().load_params(params_file)

HT = HeatMap()

HT.show_from_category_dictionary(kparams["categories"], kparams["layout"], False )

raw_input("Press any key...")