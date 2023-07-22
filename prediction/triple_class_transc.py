import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import random

with open("result.txt", "w") as myfile:
    myfile.write("Epoca,Accuracy,Precision,Recall,FPR\n")

epoch = 100
while epoch < 10000:
    owl_relation=[]
    owl_entity=[]
    fast_relation=[]
    fast_entity=[]

    files_src = "D:/TransC-OWL/data/DBpedia15K/Output/"

    def score(h,t,r):
        head = np.array(owl_entity[h])
        tail = np.array(owl_entity[t])
        relation = np.array(owl_relation[r])

        owl_res = (head + relation - tail)
        return (0),(-np.linalg.norm(owl_res,1))


    file_in = open(files_src + 'entity2vec_'+str(epoch)+'.vec', 'r')
    for line in file_in:

        x=line.split()
        sub_mat = []
        for val in x: 
            sub_mat.append(float(val))
        owl_entity.append(sub_mat)
    owl_entity = np.array(owl_entity)

    
    file_in = open(files_src + 'relation2vec_'+str(epoch)+'.vec', 'r')
    for line in file_in:

        x=line.split()
        sub_mat = []
        for val in x: 
            sub_mat.append(float(val))
        owl_relation.append(sub_mat)
    owl_relation = np.array(owl_relation)


    #ritrovare il valore delta per ogni relazione r
    train=[]
    file_in = open(files_src + '../Train/train2id.txt', 'r')
    first_line = True
    for line in file_in:
        if(first_line):
            first_line = False
        else:
            x=line.split()
            triple=[]
            for val in x:
                triple.append(int(val))
            train.append(triple)
    train = np.array(train)

    acc_f = []
    prec_f = []
    rec_f = []
    fpr_f = []

    acc_o = []
    prec_o = []
    rec_o = []
    fpr_o = []

    for i in range(1):
        print(epoch)
        random.shuffle(train)
        num_rel = len(owl_relation)
        owl_delta_r = []
        fast_delta_r = []
        fail = 0
        for i in range(0,num_rel):

            delta_min = -0.75
            sum_fast = []
            sum_owl = []
            num = 0
            error = 0.2; #margine di errore del 20%
            limit = 40

            for val in train:

                if(num > limit):
                    break;
                if(i == val[2]):
                    num += 1
                    fast_temp, owl_temp = score(val[0], val[1], val[2])
                    sum_owl.append(owl_temp)
                    sum_fast.append(fast_temp)
            if(num > 0):
                owl_delta_r.append(float(np.min(sum_owl)))
                fast_delta_r.append(float(np.min(sum_fast)))
            else:
                fail += 1
                owl_delta_r.append(float(delta_min))
                fast_delta_r.append(float(delta_min))
            
        #test
        test_file = files_src + '../Test/test2id.txt'
        fast_result=[]
        owl_result=[]
        file_in = open(test_file, 'r')
        first_line = True   #salto prima riga, contiene numero di triple
        num_test = 0;
        for line in file_in:
            if(first_line):
                x = line.split()
                first_line = False
                for val in x:
                    num_test = int(val)
            else:
                x=line.split()
                triple=[]
                for val in x:
                    triple.append(int(val))
                if(triple[2] != 320):

                    fast_res, owl_res = score(triple[0], triple[1], triple[2])
                    if(fast_res >= fast_delta_r[triple[2]]):
                        fast_result.append(1)
                    else:
                        fast_result.append(0)
                    if(owl_res >= owl_delta_r[triple[2]]):
                        owl_result.append(1)
                    else:
                        owl_result.append(0)
                    
        test_file = files_src + '../Test/test2id_false.txt'
        f_fast_result=[]
        f_owl_result=[]
        file_in = open(test_file, 'r')
        first_line = True   #salto prima riga, contiene numero di triple
        num_test = 0;
        for line in file_in:
            if(first_line):
                first_line = False
            else:
                x=line.split()
                triple=[]
                for val in x:
                    triple.append(int(val))
                if(triple[2] != 320):
                    fast_res, owl_res = score(triple[0], triple[1], triple[2])
                    if(fast_res >= fast_delta_r[triple[2]]):
                        f_fast_result.append(0)
                    else:
                        f_fast_result.append(1)
                    if(owl_res >= owl_delta_r[triple[2]]):
                        f_owl_result.append(0)
                    else:
                        f_owl_result.append(1)
        
        TP = owl_result.count(1)
        TN = f_owl_result.count(1)
        FN = owl_result.count(0)
        FP = f_owl_result.count(0)
        
        acc_o.append((TP+TN)/ (TP+TN+FP+FN))
        prec_o.append(TP/(TP+FP))
        rec_o.append(TP/(TP+FN))
        fpr_o.append(FP/(FP+TN))

    with open("result.txt", "a") as myfile:
        myfile.write(str(epoch)+"," +str(np.mean(acc_o)) + ","+str(np.mean(prec_o)) + ","+str(np.mean(rec_o))+","+str(np.mean(fpr_o))+"\n")
    epoch += 100
print("Finish")
