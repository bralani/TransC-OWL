'''
Genera tutti i file necessari per l'addestramento e successiva validazione, ovvero:

- triple2id_positive e triple2id_negative
- instanceOf2id_positive e instanceOf2id_negative
- subClassOf2id_positive e subClassOf2id_negative
'''

import os
import random

# variabili da modificare
path_dataset = 'D:/TransC-OWL/data/'
dataset = 'DBpediaYAGO/'
first_line = True
is_grid_search = False
file_in = open(path_dataset + dataset + 'Train/train2id.txt', 'r', encoding='utf-8')
file_out = open(path_dataset + dataset + 'Train/train2id.tsv', 'w', encoding='utf-8')

idx = 0
for line in file_in:
    x=line.split()
    if first_line:
        first_line = False
    else:
        file_out.write(x[0] + '\t' + x[2] + '\t' + x[1] + '\n')

    if is_grid_search and idx > 1000:
        break
    idx += 1

classi_trovate = set()
first_line = True
file_in = open(path_dataset + dataset + 'Train/instanceOf2id.txt', 'r', encoding='utf-8')
file_out = open(path_dataset + dataset + 'Train/train2id.tsv', 'a', encoding='utf-8')
file_out_grid = open(path_dataset + dataset + 'Train/grid_typeof.txt', 'w', encoding='utf-8')
idx = 0
for line in file_in:
    x=line.split()
    if first_line:
        first_line = False
    else:
        classi_trovate.add(x[1])
        file_out.write(x[0] + '\t' + "typeof" + '\t' + "class"+x[1] + '\n')
        if is_grid_search:
            file_out_grid.write(x[0] + ' '+x[1] + '\n')

    if is_grid_search and idx > 1000:
        break
    idx += 1

first_line = True
file_in = open(path_dataset + dataset + 'Train/subclassOf2id.txt', 'r', encoding='utf-8')
file_out = open(path_dataset + dataset + 'Train/train2id.tsv', 'a', encoding='utf-8')
idx = 0
for line in file_in:
    x=line.split()
    if first_line:
        first_line = False
    else:
        file_out.write("class"+x[0] + '\t' + "subclassof" + '\t' +  "class"+x[1] + '\n')

    if is_grid_search and idx > 1000:
        break
    idx += 1



first_line = True
file_in = open(path_dataset + dataset + 'Test/test2id.txt', 'r', encoding='utf-8')
file_out = open(path_dataset + dataset + 'Test/test2id.tsv', 'w', encoding='utf-8')
for line in file_in:
    x=line.split()
    if first_line:
        first_line = False
    else:
        file_out.write(x[0] + '\t' + x[2] + '\t' + x[1] + '\n')

first_line = True
file_in = open(path_dataset + dataset + 'Test/instanceOf2id.txt', 'r', encoding='utf-8')
file_out = open(path_dataset + dataset + 'Test/test2id.tsv', 'a', encoding='utf-8')
for line in file_in:
    x=line.split()
    if first_line:
        first_line = False
    else:
        file_out.write(x[0] + '\t' + "typeof" + '\t' + "class"+x[1] + '\n')