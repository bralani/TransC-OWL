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
typeof_id = 0

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
subclassof2id = []
TRAINREAL = []
TESTREAL = []
VALIDREAL = []


def carica_file(file, reverse = False, subpath = ''):
    if reverse:
        str_command = "[int(riga[1])] = riga[0]"
    else:
        str_command = "[riga[0]] = int(riga[1])"

    file_in = open(path_dataset + dataset + subpath + file + '.txt', 'r', encoding='utf-8')
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
carica_file('relation2id', False, 'Train/')
carica_file('instance2id', False, 'Train/')
carica_file('class2id', False, 'Train/')
print('file caricati con successo')
    

file_in = open(path_dataset + dataset + 'subclassof.txt', 'r', encoding='utf-8')
for line in file_in:
    x=line.split()
    triple=[]

    concept_sub = x[0]
    concept_up = x[1]

    if concept_sub not in class2id:
        class2id[concept_sub] = len(class2id) + 1

    if concept_up not in class2id:
        class2id[concept_up] = len(class2id) + 1
    
    subclassof2id.append([str(class2id[concept_sub]), str(class2id[concept_up])])


scartate = 0

SCARTATE_TEST = 0
file_in = open(path_dataset + dataset + 'test2id.txt', 'r', encoding='utf-8')
for line in file_in:
    x=line.split()
    triple=[]

    head = entity2id[int(x[0])]
    tail = entity2id[int(x[1])]
    relation = int(x[2])

    if relation != typeof_id:
        if head in instance2id and tail in instance2id:
            head = instance2id[head]
            tail = instance2id[tail]
            TESTREAL.append([str(head), str(tail), str(relation)])
        else:
            SCARTATE_TEST += 1
            

print('SCARTATE_TEST ' + str(SCARTATE_TEST) + ' triple')

SCARTATE_TEST = 0
file_in = open(path_dataset + dataset + 'valid2id.txt', 'r', encoding='utf-8')
for line in file_in:
    x=line.split()
    triple=[]

    head = entity2id[int(x[0])]
    tail = entity2id[int(x[1])]
    relation = int(x[2])

    if head in instance2id and tail in instance2id:
        head = instance2id[head]
        tail = instance2id[tail]

        
        VALIDREAL.append([str(head), str(tail), str(relation)])

print('SCARTATE_VALID ' + str(SCARTATE_TEST) + ' triple')

SCARTATE_TRAIN = 0
instanceoftrain = []
file_in = open(path_dataset + dataset + 'train2id.txt', 'r', encoding='utf-8')
for line in file_in:
    x=line.split()
    triple=[]

    head = entity2id[int(x[0])]
    tail = entity2id[int(x[1])]
    relation = int(x[2])


    if relation != typeof_id:
        if head in instance2id and tail in instance2id:
            head = instance2id[head]
            tail = instance2id[tail]

            TRAINREAL.append([str(head), str(tail), str(relation)])
        else:
            SCARTATE_TRAIN += 1
    else:
        if head not in instance2id:
            instance2id[head] = len(instance2id) + 1

        if tail not in class2id:
            class2id[tail] = len(class2id) + 1


        instanceoftrain.append([str(instance2id[head]), str(class2id[tail])])
            
array_num_instanceof = [0] * len(instance2id)

for triple in instanceoftrain:
    array_num_instanceof[int(triple[0]) - 1] += 1

media = sum(array_num_instanceof) / len(array_num_instanceof)

print('SCARTATE_TRAIN ' + str(SCARTATE_TRAIN) + ' triple')

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
    file_out = open(path_dataset + dataset + folder + '/' + file + '.txt', 'w', encoding='utf-8')
    file_out.write(str(len(set))  + '\n')
    for triple in set:
        file_out.write(' '.join(triple) + '\n')

# salva i file
print('Salvataggio file...')

salva_file('train2id', 'Train', TRAINREAL)
salva_file('test2id', 'Test', TESTREAL)
salva_file('valid2id', 'Valid', VALIDREAL)
#salva_file('instanceOf2id', 'Train', instanceof2id)