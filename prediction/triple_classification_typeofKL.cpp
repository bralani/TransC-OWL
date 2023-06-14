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


int dim = 50, ins_test_num = 0, ins_test_num_false = 0, concept_num = 0, entity_num = 0;
double threshold_typeof = 0, delta_sub = 0;
bool valid;
bool mix = false;
int epoca_attuale = 45;
int epoche = 11;
FILE* results;
string dataSet = "DBpedia100K";
map<string, string> class_map;
map<string, string> entity_map;
map<string, string> relation_map;


vector<vector<double> > entity_vec, class_mu, class_covariance;
vector<pair<int, int> > ins_right, ins_wrong, sub_right, sub_wrong;

inline double sqr(double x){
    return x * x;
}


#include <iostream>
#include <vector>
#define M_PI 3.14159265358979323846

double log_prob_diagonale(vector<double>& x, vector<double>& mean, vector<double>& var) {
    double log_prob = 0.0;

    for (int i = 0; i < dim; i++) {
        double xi = x[i];
        double mean_i = mean[i];
        double diff = xi - mean_i;
        double var_i = var[i];

        // Calcola la log-probabilità per la variabile i-esima
        double log_prob_i = -0.5 * log(2 * M_PI * var_i) - 0.5 * (diff * diff) / var_i;

        // Aggiungi la log-probabilità alla somma totale
        log_prob += log_prob_i;
    }

    return log_prob;
}

bool checkInstance(int instance, int concept){
    
    // Calcola la probabilità di una distribuzione normale gaussiana multivariata
    double probability = log_prob_diagonale(entity_vec[instance], class_mu[concept], class_covariance[concept]);
    if(probability > threshold_typeof) {
        return true;
    } else {
        return false;
    }
}

void init(){
    entity_vec.clear(); class_covariance.clear(), class_mu.clear();
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
    fin >> ins_test_num_false;
    fin_right >> ins_test_num;


    FILE* f1;
    FILE* f2;
    FILE* f3;
    f1 = fopen(("../data/" + dataSet + "/Output/transprob/entity2vec" + note + "_" + to_string(epoca_attuale) + ".txt").c_str(), "r");
    f2 = fopen(("../data/" + dataSet + "/Output/transprob/classmu2vec" + note + "_" + to_string(epoca_attuale) + ".txt").c_str(), "r");
    f3 = fopen(("../data/" + dataSet + "/Output/transprob/classcov2vec" + note + "_" + to_string(epoca_attuale) + ".txt").c_str(), "r");
    if (f3 == NULL || f2 == NULL || f1 == NULL) {
        printf("Errore nell'apertura del file.\n");
        return; // Uscita con errore
    }


    concept_num = 0;
    string tmp_str;
	ifstream class2id_num("../data/" + dataSet + "/Output/transprob/class_map.txt");
	while (getline(class2id_num, tmp_str)) {
        string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);

        size_t found = c1.find("class");
        c1.erase(found, 5);

        int c = atoi(c1.c_str());
        if(c > concept_num){
            concept_num = c;
        }

	}
    concept_num++;
	class2id_num.close();

    class_covariance.resize(concept_num);
    class_mu.resize(concept_num);
    for(int i = 0; i < concept_num; ++i){
        class_covariance[i].resize(dim);
        class_mu[i].resize(dim);
    }

    ifstream class2id_file("../data/" + dataSet + "/Output/transprob/class_map.txt");
	while (getline(class2id_file, tmp_str)) {
		string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);
		string c2 = tmp_str.substr(pos + 1).c_str();
		class_map.insert(pair<string, string>(c2, c1));

        size_t found = c1.find("class");
        c1.erase(found, 5);

        double tmp;
        int i = atoi(c1.c_str());
        
        for(int j = 0; j < dim; ++j){
            fscanf(f2, "%lf", &tmp);
            class_mu[i][j] = tmp;

            fscanf(f3, "%lf", &tmp);
            class_covariance[i][j] = tmp;
        }
	}

    entity_num = 0;
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
        if(c > entity_num){
            entity_num = c;
        }
	}
    entity_num++;
	entity2id_num.close();

    entity_vec.resize(entity_num);
    ifstream entity2id_file("../data/" + dataSet + "/Output/transprob/entity_map.txt");
	while (getline(entity2id_file, tmp_str))
	{
		string::size_type pos = tmp_str.find(' ', 0);
		string c1 = tmp_str.substr(0, pos);
		string c2 = tmp_str.substr(pos + 1).c_str();
		entity_map.insert(pair<string, string>(c1, c2));

        // Using find()
        size_t found = c1.find("class");
        if (found != std::string::npos) {
            continue;
        } 

        int i = atoi(c1.c_str());

        entity_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f1, "%lf", &entity_vec[i][j]);
        }
	}


    if(ins_test_num_false < ins_test_num) {
        ins_test_num = ins_test_num_false;
    }

    int tmp1, tmp2;

    // cicla su tutte le istanze di test per controllare che siano presenti
    // nel dizionario delle entità
    for(int i = 0; i < ins_test_num; ++i){

        fin >> tmp1 >> tmp2;

        if(entity_map.count(to_string(tmp1)) && entity_map.count(to_string(tmp2))) {
            ins_wrong.emplace_back(tmp1, tmp2);
        }
        
        fin_right >> tmp1 >> tmp2;

        if(entity_map.count(to_string(tmp1)) && entity_map.count(to_string(tmp2))) {
            ins_right.emplace_back(tmp1, tmp2);
        }
    }
    fin.close();
    fin_right.close();

    if(ins_wrong.size() < ins_test_num) {
        ins_test_num = ins_wrong.size();
    }

    if(ins_right.size() < ins_test_num) {
        ins_test_num = ins_right.size();
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
        double ins_ans = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FP_ins + FN_ins);
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
    
    for(int i = 0; i <= 1000; ++i){
        threshold_typeof = -i;
        pair<double, double> ans = test();
        if(ans.first > ins_best_answer){
            ins_best_answer = ans.first;
            ins_best_delta = threshold_typeof;
        }
    }
    threshold_typeof = ins_best_delta;
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
    if ((i = ArgPos((char *)"-epoche", argc, argv)) > 0) epoche = atoi(argv[i + 1]);
    cout << "data = " << dataSet << endl;
    if (mix)
        cout << "mix = " << "True" << endl;
    else
        cout << "mix = " << "False" << endl;
    cout << "dimension = " << dim << endl;

    results = fopen(("../data/" + dataSet + "/Output/transprob/results" + note + "_typeof.csv").c_str(), "w");
    fprintf(results,"Epoca,Accuracy,Precision,Recall,FPR\n");

    if(epoche == -1) {   
        valid = true;     
        prepare();
        runValid();
    } else {
        for(epoca_attuale = 5; epoca_attuale <= epoche; epoca_attuale += 5)
        {
            valid = true;
            prepare();
            runValid();
        }
    }
}
