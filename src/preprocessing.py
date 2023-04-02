'''
Genera tutti i file necessari per l'addestramento e successiva validazione, ovvero:

- triple2id_positive e triple2id_negative
- instanceOf2id_positive e instanceOf2id_negative
- subClassOf2id_positive e subClassOf2id_negative
'''

import os
import random

# variabili da modificare
path_dataset = '/home/matteo/Desktop/TransC-OWL/data/'
dataset = 'DBpedia15K/'
typeof_id = 275

train_perc = 0.7
valid_perc = 0.15
test_perc = 0.15

# hashmap con i vari dataset
entity2id = {}
relation2id = {}
instance2id = {}
class2id = {}
triples = []
false2id = []
instanceof2id = []
falseinstanceof2id = []
subclassof2id = []
inverseOf2 = []


def carica_file(file, reverse = False):
    if reverse:
        str_command = "[int(riga[1])] = riga[0]"
    else:
        str_command = "[riga[0]] = int(riga[1])"

    file_in = open(path_dataset + dataset + file + '.txt', 'r')
    # salta la prima riga
    first_line = True
    for line in file_in:
        if(first_line):
            first_line = False
        else:
            riga=line.split()
            exec(file + str_command)

# carica i file
print('Caricamento file...')
carica_file('entity2id', True)
carica_file('relation2id')
carica_file('instance2id')
carica_file('class2id')
print('file caricati con successo')


file_in = open(path_dataset + dataset + 'instanceof.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    instance = x[0]
    concept = x[1]
    
    if instance not in instance2id:
        instance2id[instance] = len(instance2id)

    if concept not in class2id:
        class2id[concept] = len(class2id)


    instanceof2id.append([str(instance2id[instance]), str(class2id[concept])])


file_in = open(path_dataset + dataset + 'falseTypeOf2id.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    instance_str = entity2id[int(x[0])]
    concept_str = entity2id[int(x[1])]

    if instance_str in instance2id and concept_str in class2id:
        instance = instance2id[instance_str]
        concept = class2id[concept_str]
        
        falseinstanceof2id.append([str(instance), str(concept)])
    

file_in = open(path_dataset + dataset + 'subclassof.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    concept_sub = x[0]
    concept_up = x[1]

    if concept_sub not in class2id:
        class2id[concept_sub] = len(class2id)

    if concept_up not in class2id:
        class2id[concept_up] = len(class2id)
    
    subclassof2id.append([str(class2id[concept_sub]), str(class2id[concept_up])])


scartate = 0
file_in = open(path_dataset + dataset + 'triples.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    head = x[0]
    relation = x[1]
    tail = x[2]
    
    # ignora le typeof
    if relation2id[relation] != typeof_id:
        # scarta tutte le triple che non hanno sia head che tail nell'instance2id
        if head in instance2id and tail in instance2id:
            triple.append(str(instance2id[head]))
            triple.append(str(instance2id[tail]))
            triple.append(str(relation2id[relation]))
            triples.append(triple)
        else:
            scartate += 1
            
file_in = open(path_dataset + dataset + 'inverseOf.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    head = x[0]
    relation = x[1]
    tail = x[2]
    
    # ignora le typeof
    if relation2id[relation] != typeof_id:
        # scarta tutte le triple che non hanno sia head che tail nell'instance2id
        if head in instance2id and tail in instance2id:
            triple.append(str(instance2id[head]))
            triple.append(str(instance2id[tail]))
            triple.append(str(relation2id[relation]))
            triples.append(triple)

file_in = open(path_dataset + dataset + 'false2id.txt', 'r')
for line in file_in:
    x=line.split()
    triple=[]

    head = entity2id[int(x[0])]
    tail = entity2id[int(x[1])]
    relation = int(x[2])

    if head in instance2id and tail in instance2id:
        head = instance2id[head]
        tail = instance2id[tail]
        
        false2id.append([str(head), str(tail), str(relation)])

print('Scartate ' + str(scartate) + ' triple')


# genera il training set, validation set e test set
def suddividi_set(set, no_training = False):

    if no_training:
        act_train_perc = 0
        act_valid_perc = valid_perc / (valid_perc + test_perc)
        act_test_perc = test_perc / (valid_perc + test_perc)
    else:
        act_train_perc = train_perc
        act_valid_perc = valid_perc
        act_test_perc = test_perc

    training_set = []
    validation_set = []
    test_set = []

    # mischia il set
    random.shuffle(set)

    for triple in set:
        rand = random.random()
        if rand < act_train_perc:
            training_set.append(triple)
        elif rand < act_train_perc + act_valid_perc:
            validation_set.append(triple)
        else:
            test_set.append(triple)
    
    return training_set, validation_set, test_set

def salva_file(file, folder, set):
    if not os.path.exists(path_dataset + dataset + folder):
        os.makedirs(path_dataset + dataset + folder)
    file_out = open(path_dataset + dataset + folder + '/' + file + '.txt', 'w')
    file_out.write(str(len(set))  + '\n')
    for triple in set:
        file_out.write(' '.join(triple) + '\n')

# salva i file
print('Salvataggio file...')
training_set, validation_set, test_set = suddividi_set(triples)
salva_file('train2id', 'Train', training_set)
salva_file('valid2id', 'Valid', validation_set)
salva_file('test2id', 'Test', test_set)

training_set, validation_set, test_set = suddividi_set(false2id, True)
salva_file('valid2id_false', 'Valid', validation_set)
salva_file('test2id_false', 'Test', test_set)

training_set, validation_set, test_set = suddividi_set(instanceof2id)
salva_file('instanceOf2id', 'Train', training_set)
salva_file('instanceOf2id', 'Valid', validation_set)
salva_file('instanceOf2id', 'Test', test_set)

training_set, validation_set, test_set = suddividi_set(falseinstanceof2id)
salva_file('falseinstanceOf2id', 'Train', training_set)
salva_file('falseinstanceOf2id', 'Valid', validation_set)
salva_file('falseinstanceOf2id', 'Test', test_set)

training_set, validation_set, test_set = suddividi_set(subclassof2id)
salva_file('subclassOf2id', 'Train', training_set)
salva_file('subclassOf2id', 'Valid', validation_set)
salva_file('subclassOf2id', 'Test', test_set)


# salva i file instance2id.txt, class2id.txt, relation2id.txt
file_out = open(path_dataset + dataset + 'Train/instance2id.txt', 'w')
file_out.write(str(len(instance2id))  + '\n')
for key, value in instance2id.items():
    file_out.write(key + ' ' + str(value) + '\n')


file_out = open(path_dataset + dataset + 'Train/class2id.txt', 'w')
file_out.write(str(len(class2id))  + '\n')
for key, value in class2id.items():
    file_out.write(key + ' ' + str(value) + '\n')


file_out = open(path_dataset + dataset + 'Train/relation2id.txt', 'w')
file_out.write(str(len(relation2id))  + '\n')
for key, value in relation2id.items():
    file_out.write(key + ' ' + str(value) + '\n')
