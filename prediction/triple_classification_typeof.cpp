#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

using namespace std;


int dim = 100, sub_test_num = 0, ins_test_num = 0, concept_num = 0, entity_num = 0;
double delta_ins = 0, delta_sub = 0;
bool valid;
bool mix = false;
bool OWL = false;
int epoca_attuale = 0;
int epoche = -1;
FILE* results;
string dataSet = "DBpedia100K";

vector<vector<double> > entity_vec, concept_vec;
vector<double> concept_r;
vector<pair<int, int> > ins_right, ins_wrong, sub_right, sub_wrong;

inline double sqr(double x){
    return x * x;
}

bool checkSubClass(int concept1, int concept2){
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += sqr(concept_vec[concept1][i] - concept_vec[concept2][i]);
    }
    if(sqrt(dis) < fabs(concept_r[concept1] - concept_r[concept2]) && concept_r[concept1] < concept_r[concept2]){
        return true;
    }
    if(sqrt(dis) < concept_r[concept1] + concept_r[concept2]){
        double tmp = (concept_r[concept1] + concept_r[concept2] - sqrt(dis)) / concept_r[concept1];
        if(tmp > delta_sub)
            return true;
    }
    return false;
}

bool checkInstance(int instance, int concept){
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += sqr(entity_vec[instance][i] - concept_vec[concept][i]);
    }
    if(sqrt(dis) < concept_r[concept]){
        return true;
    }
    double tmp = concept_r[concept] / sqrt(dis);
    return tmp > delta_ins;
}

void init(){
    entity_vec.clear(); concept_vec.clear();
    concept_r.clear();
    ins_right.clear(); ins_wrong.clear(); sub_right.clear(); sub_wrong.clear();
}

string note = "";

void prepare(){
    init();

    ifstream fin, fin_right;
    if(valid){
        fin.open(("../data/" + dataSet + "/Valid/falseinstanceOf2id.txt").c_str());
        fin_right.open(("../data/" + dataSet + "/Valid/instanceOf2id.txt").c_str());
    }else{
        fin.open(("../data/" + dataSet + "/Test/falseinstanceOf2id.txt").c_str());
        fin_right.open(("../data/" + dataSet + "/Test/instanceOf2id.txt").c_str());
    }
    fin >> ins_test_num;
    fin_right >> ins_test_num;
    int tmp1, tmp2;
    for(int i = 0; i < ins_test_num; ++i){
        fin >> tmp1 >> tmp2;
        ins_wrong.emplace_back(tmp1, tmp2);
        fin_right >> tmp1 >> tmp2;
        ins_right.emplace_back(tmp1, tmp2);
    }
    fin.close();
    fin_right.close();
    sub_test_num = 0;

    int tmp = 0;
    FILE *fin_num = fopen(("../data/" + dataSet + "/Train/instance2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &entity_num);
    fclose(fin_num);
    fin_num = fopen(("../data/" + dataSet + "/Train/class2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &concept_num);
    fclose(fin_num);

    FILE* f1;
    FILE* f2;
    if(epoche == -1) {
        f1 = fopen(("../data/" + dataSet + "/Output/entity2vec" + note + ".vec").c_str(), "r");
        f2 = fopen(("../data/" + dataSet + "/Output/concept2vec" + note + ".vec").c_str(), "r");
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
    concept_vec.resize(concept_num);
    concept_r.resize(concept_num);
    for(int i = 0; i < concept_num; ++i){
        concept_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f2, "%lf", &concept_vec[i][j]);
        }
        fscanf(f2, "%lf", &concept_r[i]);
    }
}

pair<double, double> test(){
    double TP_ins = 0, TN_ins = 0, FP_ins = 0, FN_ins = 0;
    double TP_sub = 0, TN_sub = 0, FP_sub = 0, FN_sub = 0;
    map<int, double> TP_ins_map, TN_ins_map, FP_ins_map, FN_ins_map;
    set<double> concept_set;

    for(int i = 0; i < ins_test_num; ++i){
        if(checkInstance(ins_right[i].first, ins_right[i].second)) {
            TP_ins++;
            if(TP_ins_map.count(ins_right[i].second) > 0){
                TP_ins_map[ins_right[i].second]++;
            }else{
                TP_ins_map[ins_right[i].second] = 1;
            }
        }else {
            FN_ins++;
            if(FN_ins_map.count(ins_right[i].second) > 0){
                FN_ins_map[ins_right[i].second]++;
            }else{
                FN_ins_map[ins_right[i].second] = 1;
            }
        }
        if(!checkInstance(ins_wrong[i].first, ins_wrong[i].second)) {
            TN_ins++;
            if(TN_ins_map.count(ins_wrong[i].second) > 0){
                TN_ins_map[ins_wrong[i].second]++;
            }else{
                TN_ins_map[ins_wrong[i].second] = 1;
            }
        }else {
            FP_ins++;
            if(FP_ins_map.count(ins_wrong[i].second) > 0){
                FP_ins_map[ins_wrong[i].second]++;
            }else{
                FP_ins_map[ins_wrong[i].second] = 1;
            }
        }
        double concept_s = ins_right[i].second;
        double concept_m = ins_wrong[i].second;
        concept_set.insert(concept_s);
        concept_set.insert(concept_m);
    }
    
    if(valid){
        double ins_ans = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins);
        double sub_ins = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub);
        return make_pair(ins_ans, sub_ins);
    }else{
        double accuracy = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FP_ins + FN_ins);
        double precision = TP_ins * 100 /(TP_ins + FP_ins);
        double recall = TP_ins * 100 / (TP_ins + FN_ins);
        double FPR = FP_ins * 100 /(FP_ins + TN_ins);

        fprintf(results,"%d,", epoca_attuale);
        fprintf(results,"%f,", accuracy);
        fprintf(results,"%f,", precision);
        fprintf(results,"%f,", recall);
        fprintf(results,"%f\n", FPR);

        return make_pair(0.0, 0.0);
    }
}

void runValid(){
    double ins_best_answer = 0, ins_best_delta = 0;
    double sub_best_answer = 0, sub_best_delta = 0;
    for(int i = 0; i < 101; ++i){
        double f = i; f /= 100;
        delta_ins = f;
        pair<double, double> ans = test();
        if(ans.first > ins_best_answer){
            ins_best_answer = ans.first;
            ins_best_delta = f;
        }
    }
    delta_ins = ins_best_delta;
    valid = false;
    prepare();
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

int main(int argc, char**argv){
    int i = 0;
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) dataSet = argv[i + 1];
    if ((i = ArgPos((char *)"-mix", argc, argv)) > 0) mix = static_cast<bool>(atoi(argv[i + 1]));
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-OWL", argc, argv)) > 0) OWL = static_cast<bool>(atoi(argv[i + 1]));
    if ((i = ArgPos((char *)"-epoche", argc, argv)) > 0) epoche = atoi(argv[i + 1]);
    cout << "data = " << dataSet << endl;
    if (mix)
        cout << "mix = " << "True" << endl;
    else
        cout << "mix = " << "False" << endl;
        if (OWL)
        cout << "OWL = " << "True" << endl;
    else
        cout << "OWL = " << "False" << endl;

    if(OWL) {
        note = "_OWL";
    }
    cout << "dimension = " << dim << endl;

    results = fopen(("../data/" + dataSet + "/Output/results" + note + "_typeof.csv").c_str(), "w");
    fprintf(results,"Epoca,Accuracy,Precision,Recall,FPR\n");

    if(epoche == -1) {   
        valid = true;     
        prepare();
        runValid();
    } else {
        for(epoca_attuale = 0; epoca_attuale <= epoche; epoca_attuale += 100)
        {
            valid = true;
            prepare();
            runValid();
        }
    }
}
