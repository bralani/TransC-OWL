'''
 Per la relazione transitiva:
    - se ci sono almeno n esempi nel TR sia di una classe che di un'altra, allora rimuovi dal TR gli esempi della relazione transitiva
'''

import os
import random
from statistics import mean


# variabili da modificare
path_dataset = 'D:/TransC-OWL/data/'
dataset = 'DBpedia15K/'


# hashmap con i vari dataset
instance2id = {}
instanceof2id = []
class2id = {}
relation2id = {}
subclassof2id = []
num_occorrenze_class = {}
num_occorrenze_rel = {}
transitive = []
asymmetric = []
disjoint = []


def carica_file(file):
    str_command = "[str] = int(last_val)"

    file_in = open(path_dataset + dataset + "/Train/" + file + '.txt', 'r', encoding="utf8")
    # salta la prima riga
    first_line = True
    for line in file_in:
        if(first_line):
            first_line = False
        else:
            riga=line.split()
            str = ''
            last_val = riga[-1]
            # da indice 0 a n-1 ricompone la stringa
            for i in range(0, len(riga)-1):
                str = str + riga[i] + ''

            if file == "class2id":
                num_occorrenze_class[last_val] = 0
            elif file == "relation2id":
                num_occorrenze_rel[last_val] = 0

            exec(file + str_command)

# carica i file
print('Caricamento file...')
carica_file('class2id')
carica_file('relation2id')
carica_file('instance2id')
print('file caricati con successo')

'''
# salta la prima riga
first_line = True
file_in = open(path_dataset + dataset + '/Train/train2id.txt', 'r')
for line in file_in:
    if(first_line):
        first_line = False
    else:
        x=line.split()
        triple=[]

        rel = x[2]

        num_occorrenze_rel[rel] += 1 '''

# salta la prima riga
first_line = True
file_in = open(path_dataset + dataset + '/Train/subclassOf2id.txt', 'r')
for line in file_in:
    if(first_line):
        first_line = False
    else:
        x=line.split()
        triple=[]

        concept_sub = x[0]
        concept_up = x[1]

        num_occorrenze_class[concept_sub] += 1
        num_occorrenze_class[concept_up] += 1
        
        subclassof2id.append([str(concept_sub), str(concept_up)])


# salta la prima riga
first_line = True
file_in = open(path_dataset + dataset + '/Train/instanceOf2id.txt', 'r')
for line in file_in:
    if(first_line):
        first_line = False
    else:
        x=line.split()
        triple=[]

        concept = x[1]

        num_occorrenze_class[concept] += 1

def is_transitive(class_sub, class_up):
    class_to_check = []

    # trova in subclassof2id tutte le triple con class_sub come soggetto
    for triple in subclassof2id:
        if triple[0] == str(class_sub):
            class_to_check.append(triple[1])

    class_to_check = list( dict.fromkeys(class_to_check) )

    # controlla che almeno una di queste triple abbia class_up come oggetto
    for classe in class_to_check:
        for triple in subclassof2id:
            if triple[0] == str(classe) and triple[1] == str(class_up):
                return True
            
    return False

'''
limite_trans = 5

file_in = open(path_dataset + dataset + 'transitive.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    concept_sub = x[0]
    concept_up = x[1]

    if concept_sub in class2id and concept_up in class2id:
        class_sub = class2id[concept_sub]
        class_up = class2id[concept_up]

        if num_occorrenze_class[str(class_sub)] >= limite_trans and num_occorrenze_class[str(class_up)] >= limite_trans:
            if(is_transitive(class_sub, class_up)):
                transitive.append([str(class_sub), str(class_up)])
                print('transitive: ' + str(concept_sub) + ' ' + str(concept_up))
                '''
'''
limite_asymm = 0

file_in = open(path_dataset + dataset + 'asymmetric.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    relation_asymm = x[0]

    if relation_asymm in relation2id and num_occorrenze_rel[str(relation2id[relation_asymm])] >= limite_asymm:
        asymmetric.append([str(relation2id[relation_asymm])])
        print('asymmetric: ' + str(relation_asymm))

'''
file_in = open(path_dataset + dataset + 'instanceof.txt', 'r', encoding="utf8")
for line in file_in:
    x=line.split()
    triple=[]

    instance = x[0]
    concept = x[1]
    
    if concept in class2id:
        if instance not in instance2id:
            instance2id[instance] = len(instance2id) + 1

            class2id[concept] = len(class2id) + 1


        instanceof2id.append([str(instance2id[instance]), str(class2id[concept])])

limite_disjoint = 10
classi_disjoint = []
for classe in class2id.values():
    if str(classe) in num_occorrenze_class and num_occorrenze_class[str(classe)] >= limite_disjoint:
        classi_disjoint.append([str(classe)])

file_in = open(path_dataset + dataset + 'disjoint.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    concept1 = x[0]
    concept2 = x[1]

    if concept1 in class2id and concept2 in class2id:
        class_sub = class2id[concept1]
        class_up = class2id[concept2]

        # se non già presenti, le aggiunge
        if str(class_sub) in num_occorrenze_class and str(class_up) in num_occorrenze_class and num_occorrenze_class[str(class_sub)] >= limite_disjoint and num_occorrenze_class[str(class_up)] >= limite_disjoint and [str(class_sub), str(class_up)] not in disjoint:
            disjoint.append([str(class_sub), str(class_up)])
    

def salva_file(file, folder, set):
    if not os.path.exists(path_dataset + dataset + folder):
        os.makedirs(path_dataset + dataset + folder)
    file_out = open(path_dataset + dataset + folder + '/' + file + '.txt', 'w')
    file_out.write(str(len(set))  + '\n')
    for triple in set:
        file_out.write(' '.join(triple) + '\n')

subclassof2id_all = []
file_in = open(path_dataset + dataset + 'subclassof.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    concept_sub = x[0]
    concept_up = x[1]

    if concept_sub not in class2id:
        class2id[concept_sub] = len(class2id) + 1

    if concept_up not in class2id:
        class2id[concept_up] = len(class2id) + 1
    
    subclassof2id_all.append([str(class2id[concept_sub]), str(class2id[concept_up])])

# salva i file
print('Salvataggio file...')

salva_file('transitive', 'Test', transitive)

#rimuove da subclassof2id le triple che sono in transitive
print(len(subclassof2id))
for triple in transitive:
    for triple2 in subclassof2id:
        if triple[0] == triple2[0] and triple[1] == triple2[1]:
            subclassof2id.remove(triple2)
print(len(subclassof2id))

salva_file('subclassOf2id', 'Train', subclassof2id)

salva_file('subclassOf2id_all', 'Test', subclassof2id_all)

#salva_file('asymmetric', 'Test', asymmetric)

salva_file('disjoint', 'Test', disjoint)

salva_file('disjoint_classes', 'Test', classi_disjoint)

salva_file('instanceof2id', 'Test', instanceof2id)