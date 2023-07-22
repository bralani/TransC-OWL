import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import random

import torch

#with open("result_typeof.txt", "w") as myfile:
#    myfile.write("Epoca,Accuracy,Precision,Recall,FPR\n")

epoch = 1100
while epoch < 1101:

    owl_class_mu=[]
    owl_class_cov=[]
    owl_entity=[]
    fast_class=[]
    fast_entity=[]

    files_src = "D:/TransC-OWL/data/DBpediaYAGO/Output/"

    def score(h,t):
        _, score_t = scores([h],[t])

        return (0),(score_t[0])

    def scores(h,t):
        mu = [owl_class_mu[indice] for indice in t]
        mu = torch.tensor(mu)
        cov = [owl_class_cov[indice] for indice in t]
        cov = torch.tensor(cov)
        entity = [owl_entity[indice] for indice in h]
        entity = torch.tensor(entity)

        diff = entity - mu
        inv_cov = 1 / (torch.sqrt(cov) + 0.001)
        exponential_term = -0.5 * torch.einsum("ni,ni,ni->n", diff, inv_cov, diff)
        log_prob = -0.5 * torch.sum(torch.log(cov), dim=1) + exponential_term

        return (0),(log_prob)

    file_entity_map = open(files_src + 'transprob/entity_map.txt', 'r')

    file_in = open(files_src + 'transprob/entity2vec_'+str(epoch)+'.txt', 'r')
    #file_in = open(files_src + 'transprob/'+str(search[idx])+'/'+str(search[jdx])+'/'+str(search[kdx])+'/entity2vec.txt', 'r')

    for line_map in file_entity_map:
        map=line_map.split()
        if "class" in map[0]:
            continue

        i = int(map[0])
        while len(owl_entity) <= i:
            owl_entity.append([])

        # legge una riga del file
        line = file_in.readline()

        x=line.split()
        sub_mat = []
        for val in x: 
            sub_mat.append(float(val))
        owl_entity[i] = sub_mat
    owl_entity = np.array(owl_entity)


    file_class_map = open(files_src + 'transprob/class_map.txt', 'r')
    file_mu = open(files_src + 'transprob/classmu2vec_'+str(epoch)+'.txt', 'r')
    file_cov = open(files_src + 'transprob/classcov2vec_'+str(epoch)+'.txt', 'r')
    for line_map in file_class_map:
        map=line_map.split()

        # legge una riga del file
        line_mu = file_mu.readline()

        # rimuove la substr "class" dalla stringa
        map[0] = map[0][5:]

        i = int(map[0])
        while len(owl_class_mu) <= i:
            owl_class_mu.append([])

        x=line_mu.split()
        sub_mu = []
        for val in x: 
            sub_mu.append(float(val))
        owl_class_mu[i] = sub_mu

        
        # legge una riga del file
        line_cov = file_cov.readline()
        while len(owl_class_cov) <= i:
            owl_class_cov.append([])

        x=line_cov.split()
        sub_cov = []
        for val in x: 
            sub_cov.append(float(val))
        owl_class_cov[i] = sub_cov

    owl_class_mu = np.array(owl_class_mu)
    owl_class_cov = np.array(owl_class_cov)


    #ritrovare il valore delta per ogni relazione r
    train={}
    file_in = open(files_src + '../Train/instanceOf2id.txt', 'r')
    first_line = True
    for line in file_in:
        if(first_line):
            first_line = False
        else:
            x=line.split()
            triple=[]
            for val in x:
                triple.append(int(val))
            
            if triple[1] not in train:
                train[triple[1]] = []

            train[triple[1]].append(triple)

    acc_f = []
    prec_f = []
    rec_f = []
    fpr_f = []

    acc_o = []
    prec_o = []
    rec_o = []
    fpr_o = []

    sum = 0
    num_tot = 0
    for i in range(1):
        print(epoch)
        num_class = len(owl_class_mu)
        owl_delta_r = []
        fast_delta_r = []
        fail = 0
        for i in range(0,num_class):

            delta_min = -20.0
            sum_fast = []
            sum_owl = []
            num = 0
            error = 0.2; #margine di errore del 20%
            limit = 40

            if len(owl_class_mu[i]) == 0:
                owl_delta_r.append(float(delta_min))
                fast_delta_r.append(float(delta_min))
                continue

            if(i in train):
                get_triples = train[i]
                #preleva tutti gli elementi in posizione 0
                heads = [x[0] for x in get_triples]
                #preleva tutti gli elementi in posizione 1
                tails = [x[1] for x in get_triples]

                _, all_scores = scores(heads, tails)
                owl_delta_r.append(float(torch.min(all_scores)))
                fast_delta_r.append(float(torch.min(all_scores)))
                sum += float(torch.min(all_scores))
                num_tot += 1
            else:
                fail += 1
                owl_delta_r.append(float(delta_min))
                fast_delta_r.append(float(delta_min))

        delta = sum / num_tot

        
            
        #test
        test_file = files_src + '../Test/instanceOf2id.txt'
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

                if (len(owl_entity) <= triple[0] or len(owl_class_mu) < triple[1]) or (len(owl_entity[triple[0]]) == 0 or len(owl_class_mu[triple[1]]) == 0):
                    continue

                fast_res, owl_res = score(triple[0], triple[1])
                if(fast_res >= fast_delta_r[triple[1]]):
                    fast_result.append(1)
                else:
                    fast_result.append(0)
                if(owl_res >= owl_delta_r[triple[1]]):
                    owl_result.append(1)
                else:
                    owl_result.append(0)
                    
        test_file = files_src + '../Test/falseinstanceOf2id.txt'
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
                if (len(owl_entity) <= triple[0] or len(owl_class_mu) < triple[1]) or (len(owl_entity[triple[0]]) == 0 or len(owl_class_mu[triple[1]]) == 0):
                    continue
                fast_res, owl_res = score(triple[0], triple[1])
                if(fast_res >= fast_delta_r[triple[1]]):
                    f_fast_result.append(0)
                else:
                    f_fast_result.append(1)
                if(owl_res >= owl_delta_r[triple[1]]):
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

    #with open("result_typeof.txt", "a") as myfile:
    #    myfile.write(str(epoch)+"," +str(np.mean(acc_o)) + ","+str(np.mean(prec_o)) + ","+str(np.mean(rec_o))+","+str(np.mean(fpr_o))+"\n")
    with open("result_typeof.txt", "a") as myfile:
        myfile.write(str(epoch)+"," +str(np.mean(acc_o)) + ","+str(np.mean(prec_o)) + ","+str(np.mean(rec_o))+","+str(np.mean(fpr_o))+"\n")
    epoch += 50
print("Finish")
