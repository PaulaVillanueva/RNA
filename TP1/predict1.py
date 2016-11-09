#!/usr/bin/python
# -*- coding: latin-1 -*-
import argparse
import ej1_data_loader
from model_io import ModelIO
from params_io import ParamsIO
from layer_model import LayerModel
from feed_forward_solver import NetworkSolver
import functools


parser = argparse.ArgumentParser(description='Parametros de la red')


parser.add_argument('-m',  type=str,
                    help='Ruta al archivo de modelo (lmodel)', required=True)

parser.add_argument('-p', type=str,
                    help='Ruta del archivo de entrada con los pesos', required=True)

parser.add_argument('-x', type=str, default=1,
                    help='Archivo de features', required=True)

parser.add_argument('-t', type=float, default=0.5,
                    help='Umbral', required=False)

args = parser.parse_args()


loader = ej1_data_loader.Ej1DataLoader()
data = loader.LoadData(args.x)
features = data[0] #shape=(333,10)
labels = data[1]

test_data = zip(features, labels)

mini_batches_testing = [test_data]
threshold = args.t

mloader = ModelIO()
model = mloader.load_model(args.m)

ploader = ParamsIO()
weights, biases = ploader.load_params(args.p)

solver = NetworkSolver(model,weights=weights,biases=biases)

E = solver.predict(mini_batches_testing[0],threshold)

print "Error cuadratico promedio: ", E