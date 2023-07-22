#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;

int dim = 100;
int test_num = 0, relation_num = 0, valid_num = 0, entity_num = 0;
int epoca_attuale = 0;
int epoche = 2000;
bool OWL = false;
string dataSet = "DBpedia100K";
bool valid;
bool getMinMax = false;
FILE* results;

vector<double> delta_relation;
vector<pair<double, double> > max_min_relation;
vector<vector<double> > entity_vec, relation_vec;
vector<vector<int> >right_triple, wrong_triple;

inline double sqr(double x){
    return x * x;
}

bool check(int h, int t, int r){
    vector<double> tmp;
    tmp.resize(dim);
    for(int i = 0; i < dim; ++i){
        tmp[i] = entity_vec[h][i] + relation_vec[r][i];
    }
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += fabs(tmp[i] - entity_vec[t][i]);
    }

    if(getMinMax){
        if(dis > max_min_relation[r].first)
            max_min_relation[r].first = dis;
        if(dis < max_min_relation[r].second)
            max_min_relation[r].second = dis;
    }
    return dis < delta_relation[r];
}

void init(){
    delta_relation.clear(); entity_vec.clear(); right_triple.clear(); wrong_triple.clear(); max_min_relation.clear();
}

string note = "";

void prepare(bool final_test = false){
    init();
    
    ifstream fin, fin_right;
    if(valid){
        fin.open(("../data/" + dataSet + "/Valid/valid2id_false.txt").c_str());
        fin_right.open(("../data/" + dataSet + "/Valid/valid2id.txt").c_str());
        fin_right >> valid_num;
        fin >> valid_num;
    }else{
        fin.open(("../data/" + dataSet + "/Test/test2id_false.txt").c_str());
        fin_right.open(("../data/" + dataSet + "/Test/test2id.txt").c_str());
        fin_right >> test_num;
        fin >> test_num;
    }
    ifstream fin_relation;
    fin_relation.open(("../data/" + dataSet + "/Train/relation2id.txt").c_str());
    fin_relation >> relation_num;
    fin_relation.close();
    ifstream fin_entity;
    fin_entity.open(("../data/" + dataSet + "/Train/instance2id.txt").c_str());
    fin_entity >> entity_num;
    fin_entity.close();

    if(!final_test)
        delta_relation.resize(relation_num);
    max_min_relation.resize(relation_num);
    for(int i = 0; i < relation_num; ++i){
        max_min_relation[i].first = -1;
        max_min_relation[i].second = 1000000;
    }

    int tmp1, tmp2, tmp3;
    int inputSize = valid ? valid_num : test_num;
    right_triple.resize(inputSize); wrong_triple.resize(inputSize);
    for(int i = 0; i < inputSize; ++i){
        fin >> tmp1 >> tmp2 >> tmp3;
        wrong_triple[i].resize(3);
        wrong_triple[i][0] = tmp1;
        wrong_triple[i][1] = tmp2;
        wrong_triple[i][2] = tmp3;

        fin_right >> tmp1 >> tmp2 >> tmp3;
        right_triple[i].resize(3);
        right_triple[i][0] = tmp1;
        right_triple[i][1] = tmp2;
        right_triple[i][2] = tmp3;
    }
    fin.close(); fin_right.close();

    FILE* f1;
    FILE* f2;
    if(epoche == -1) {
        f1 = fopen(("../data/" + dataSet + "/Output/entity2vec" + note + "_2400.vec").c_str(), "r");
        f2 = fopen(("../data/" + dataSet + "/Output/relation2vec" + note + "_2400.vec").c_str(), "r");
    } else {
        f1 = fopen(("../data/" + dataSet + "/Output/entity2vec" + note + "_" + to_string(epoca_attuale) + ".vec").c_str(), "r");
        f2 = fopen(("../data/" + dataSet + "/Output/relation2vec" + note + "_" + to_string(epoca_attuale) + ".vec").c_str(), "r");
    }
    entity_vec.resize(entity_num);
    for(int i = 0; i < entity_num; ++i){
        entity_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f1, "%lf", &entity_vec[i][j]);
        }
    }
    relation_vec.resize(relation_num);
    for(int i = 0; i < relation_num; ++i){
        relation_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f2, "%lf", &relation_vec[i][j]);
        }
    }
}

vector<double> test(){
    double TP = 0, TN = 0, FP = 0, FN = 0;
    vector<vector<double> > ans;
    ans.resize(relation_num);

    for(int i = 0; i < relation_num; ++i) {
        ans[i].resize(4);
        ans[i][0] = 0; ans[i][1] = 0; ans[i][2] = 0; ans[i][3] = 0;
    }
    int inputSize = valid ? valid_num : test_num;
    for(int i = 0; i < inputSize; ++i){
        if(check(right_triple[i][0], right_triple[i][1], right_triple[i][2])) {
            TP++;
            ans[right_triple[i][2]][0]++;
        }
        else{
            FN++;
            ans[right_triple[i][2]][1]++;
        }
        if(!check(wrong_triple[i][0], wrong_triple[i][1], wrong_triple[i][2])) {
            TN++;
            ans[wrong_triple[i][2]][2]++;
        }
        else {
            FP++;
            ans[wrong_triple[i][2]][3]++;
        }
    }
    if(valid){
        vector<double> returnAns;
        returnAns.resize(relation_num);
        for(int i = 0; i < relation_num; ++i){
            returnAns[i] = (ans[i][0]) * 100 / (ans[i][0] + ans[i][3]);
        }
        return returnAns;
    }else{
        double accuracy = (TP + TN) * 100 / (TP + TN + FP + FN);
        double precision = TP * 100 /(TP + FP);
        double recall = TP * 100 / (TP + FN);
        double FPR = FP * 100 /(FP + TN);

        fprintf(results,"%d,", epoca_attuale);
        fprintf(results,"%f,", accuracy);
        fprintf(results,"%f,", precision);
        fprintf(results,"%f,", recall);
        fprintf(results,"%f\n", FPR);

        vector<double> tmp;
        return tmp;
    }
}

void runValid(){
    getMinMax = true;
    test();
    getMinMax = false;

    vector<double> best_delta_relation, best_ans_relation;
    best_delta_relation.resize(relation_num);
    best_ans_relation.resize(relation_num);
    for(int i = 0; i < relation_num; ++i)
        best_ans_relation[i] = 0;

    for(int i = 0; i < 100; ++i){
        for(int j = 0; j < relation_num; ++j){
            delta_relation[j] = max_min_relation[j].second + (max_min_relation[j].first - max_min_relation[j].second) * i / 100;
        }
        vector<double> ans = test();
        for(int k = 0; k < relation_num; ++k){
            if(ans[k] > best_ans_relation[k]){
                best_ans_relation[k] = ans[k];
                best_delta_relation[k] = delta_relation[k];
            }
        }
    }
    for(int i = 0; i < relation_num; ++i){
        delta_relation[i] = best_delta_relation[i];
    }
    valid = false;
    prepare(true);
    test();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char** argv){
    int i = 0;
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) dataSet = argv[i + 1];
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-OWL", argc, argv)) > 0) OWL = static_cast<bool>(atoi(argv[i + 1]));
    if ((i = ArgPos((char *)"-epoche", argc, argv)) > 0) epoche = atoi(argv[i + 1]);
    cout << "data = " << dataSet << endl;
    cout << "dimension = " << dim << endl;
    if (OWL)
        cout << "OWL = " << "True" << endl;
    else
        cout << "OWL = " << "False" << endl;

    if(OWL) {
        note = "_OWL";
    }

    results = fopen(("../data/" + dataSet + "/Output/results" + note + ".csv").c_str(), "w");
    fprintf(results,"Epoca,Accuracy,Precision,Recall,FPR\n");

    if(epoche == -1) {   
        valid = true;     
        prepare();
        runValid();
    } else {
        for(epoca_attuale = 100; epoca_attuale <= epoche; epoca_attuale += 100)
        {
            valid = true;
            prepare();
            runValid();
        }
    }

    fclose(results);
}