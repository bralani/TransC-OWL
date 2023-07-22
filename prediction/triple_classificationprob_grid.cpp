#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <map>

using namespace std;

int dim = 50;
int test_num = 0, relation_num = 0, valid_num = 0, entity_num = 0;
int epoca_attuale = 95;
int epoche = -1;
bool OWL = false;
string dataSet = "DBpedia100K";
map<string, string> entity_map;
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

void prepare(bool final_test, string i, string j, string k){
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
    fin_relation.open(("../data/" + dataSet + "/relation2id.txt").c_str());
    fin_relation >> relation_num;
    fin_relation.close();
    ifstream fin_entity;
    fin_entity.open(("../data/" + dataSet + "/Train/instance2id.txt").c_str());
    fin_entity >> entity_num;
    fin_entity.close();


    entity_vec.resize(entity_num);
    for(int i = 0; i < entity_num; ++i){
        entity_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            entity_vec[i][j] = 0;
        }
    }

    relation_vec.resize(relation_num);
    for(int i = 0; i < relation_num; ++i){
        relation_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            relation_vec[i][j] = 0;
        }
    }

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
    string file1 = "../data/" + dataSet + "/Output/transprob/"+i+"/"+j+"/"+k+"/entity2vec.txt";
    string file2 = "../data/" + dataSet + "/Output/transprob/"+i+"/"+j+"/"+k+"/relation2vec.txt";

    cout << file1 << endl;
    f1 = fopen(file1.c_str(), "r");
    f2 = fopen(file2.c_str(), "r");
    
    if(f1 == NULL || f2 == NULL){
        cout << "File entity2vec or relation2vec not found!" << endl;
        exit(1);
    }

    int entity_num2 = 0;
    string tmp_str;
	ifstream entity2id_num("../data/" + dataSet + "/Output/transprob/entity_map.txt");
	while (getline(entity2id_num, tmp_str)) {
        
		string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);
		string c2 = tmp_str.substr(pos + 1).c_str();

        size_t found = c1.find("class");
        if (found != std::string::npos) {
            continue;
        } 

        int c = atoi(c1.c_str());
        if(c > entity_num2){
            entity_num2 = c;
        }
	}
    entity_num2++;
	entity2id_num.close();

    ifstream entity2id_file("../data/" + dataSet + "/Output/transprob/entity_map.txt");
	while (getline(entity2id_file, tmp_str))
	{
		string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);
		string c2 = tmp_str.substr(pos + 1).c_str();
		entity_map.insert(pair<string, string>(c1, c2));

        size_t found = c1.find("class");
        if (found != std::string::npos) {
            continue;
        } 

        int i = atoi(c1.c_str());
        for(int j = 0; j < dim; ++j){
            fscanf(f1, "%lf", &entity_vec[i][j]);
        }
	}


    
    int relation_num2 = 0;
	ifstream relation2id_num("../data/" + dataSet + "/Output/transprob/relation_map.txt");
	while (getline(relation2id_num, tmp_str)) {
        
		string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);
		string c2 = tmp_str.substr(pos + 1).c_str();

        int c = atoi(c1.c_str());
        if(c > relation_num2){
            relation_num2 = c;
        }
	}
    relation_num2++;
	relation2id_num.close();

    ifstream relation2id_file("../data/" + dataSet + "/Output/transprob/relation_map.txt");
	while (getline(relation2id_file, tmp_str))
	{
		string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);
		string c2 = tmp_str.substr(pos + 1).c_str();

        int i = atoi(c1.c_str());

        for(int j = 0; j < dim; ++j){
            fscanf(f2, "%lf", &relation_vec[i][j]);
        }
	}
    
}

vector<double> test(string i1, string j1, string k1){
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
            double rel = wrong_triple[i][2];
            ans[rel][2]++;
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
            returnAns[i] = (ans[i][0] + ans[i][2]) * 100 / (ans[i][0] + ans[i][1] + ans[i][2] + ans[i][3]);
        }
        return returnAns;
    }else{
        double accuracy = (TP + TN) * 100 / (TP + TN + FP + FN);
        double precision = TP * 100 /(TP + FP);
        double recall = TP * 100 / (TP + FN);
        double FPR = FP * 100 /(FP + TN);

        fprintf(results,"%f,%f,%f,", stof(i1), stof(j1), stof(k1));
        fprintf(results,"%f,", accuracy);
        fprintf(results,"%f,", precision);
        fprintf(results,"%f,", recall);
        fprintf(results,"%f\n", FPR);

        vector<double> tmp;
        return tmp;
    }
}

void runValid(string i1, string j1, string k1){
    getMinMax = true;
    test(i1, j1, k1);
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
        vector<double> ans = test(i1, j1, k1);
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
    prepare(true, i1, j1, k1);
    test(i1, j1, k1);
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
    if ((i = ArgPos((char *)"-epoche", argc, argv)) > 0) epoche = atoi(argv[i + 1]);
    cout << "data = " << dataSet << endl;
    cout << "dimension = " << dim << endl;
    results = fopen(("../data/" + dataSet + "/Output/transprob/results_grid" + note + ".csv").c_str(), "w");
    fprintf(results,"Epoca,Accuracy,Precision,Recall,FPR\n");

    vector<string> grid = {"10", "1", "0.1", "0.01", "0.001", "0.0001"};
    for(int i = 0; i < 2; ++i){
        for(int j=0; j <6; ++j) {
            for(int k=0; k <6; ++k) {
                valid = true;
                prepare(false, grid[i], grid[j], grid[k]);
                runValid(grid[i], grid[j], grid[k]);
            }
        }
        
    }

    fclose(results);
}