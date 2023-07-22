#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <unordered_map>
#include <cstring>
#include <map>
#include <set>
#include <queue>
#include <cassert>
#include <string>

using namespace std;

int dim = 100, concept_num = 0, entity_num = 0, relation_num = 0, valid_num = 0;
map<int, string> class2id;
vector<int> disjoint_classes;
double delta_ins = 0, delta_sub = 0;
int typeOf_id = 0;
bool getMinMax = false;
FILE* results;
string dataSet = "DBpedia100K";

int instanceOf_id = -1;
int subClassOf_id = -2;

vector<vector<double> > entity_vec, relation_vec, concept_vec;
vector<double> concept_r;
vector<pair<int, int> > ins_right, ins_wrong, sub_right, sub_wrong;
vector<double> delta_relation;
vector<vector<int> > concept_instance;
vector<vector<int> > schema_classi_sub;
vector<vector<int> > schema_classi_up;
vector<pair<double, double> > max_min_relation;
vector<vector<int> >right_triple, wrong_triple;
vector<int> relazioni_escluse;
vector<int> relazioni_antisimmetriche;
map<int, vector<int>> relazioni_disjoint;
map<int, vector<vector<int>>> relazioni_composte;
int total_triple = 0;


inline double sqr(double x){
    return x * x;
}

double euclidian_similarity(vector<double> a, vector<double> b) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return 1.0 / (1.0 + sqrt(sum));
}

double manhattan_similarity(vector<double> a, vector<double> b) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += fabs(a[i] - b[i]);
    }
    return 1.0 / (1.0 + sum);
}


bool checkRelation(int h, int t, int r) {
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

bool checkSubClass(int concept1, int concept2){
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += sqr(concept_vec[concept1][i] - concept_vec[concept2][i]);
    }
    double diff = concept_r[concept1] - concept_r[concept2];
    if(sqrt(dis) < fabs(diff) && concept_r[concept1] < concept_r[concept2]){
        return true;
    }

    if(sqrt(dis) < concept_r[concept1] + concept_r[concept2]){
        double tmp = (concept_r[concept1] + concept_r[concept2] - sqrt(dis)) / concept_r[concept1];
        if(tmp > delta_sub)
            return true;
    }
    return false;

}

bool intersection_classes(int concept1, int concept2){
    double dist_center = 0;
    for(int i = 0; i < dim; ++i){
        dist_center += sqr(concept_vec[concept1][i] - concept_vec[concept2][i]);
    }

    double diff = dist_center - fabs(concept_r[concept1] + concept_r[concept2]);
    if(diff >= 0) {
        return false;
    } else {
        return true;
    }
}

void check_mothers(int classe1, int classe2) {

    if(classe1 == classe2) {
        return;
    }

    // controlla se due classi sono sorelle
    for (int i = 0; i < schema_classi_up[classe1].size(); i++) {
        for (int j = 0; j < schema_classi_up[classe2].size(); j++) {
            if (schema_classi_up[classe1][i] == schema_classi_up[classe2][j]) {
                int madre = schema_classi_up[classe1][i];
                cout << "Madre: " << class2id[madre] << endl;
            }
        }
    }
}

bool check_sisters(int classe1, int classe2) {

    if(classe1 == classe2) {
        return false;
    }

    if(find(schema_classi_up[classe1].begin(), schema_classi_up[classe1].end(), classe2) != schema_classi_up[classe1].end()) {
        return false;
    }

    if(find(schema_classi_up[classe2].begin(), schema_classi_up[classe2].end(), classe1) != schema_classi_up[classe2].end()) {
        return false;
    }

    // controlla se due classi sono sorelle
    for (int i = 0; i < schema_classi_up[classe1].size(); i++) {
        for (int j = 0; j < schema_classi_up[classe2].size(); j++) {
            if (schema_classi_up[classe1][i] == schema_classi_up[classe2][j]) {
                return true;
            }
        }
    }

    return false;
}

bool checkInstance(int instance, int concept){
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += sqr(entity_vec[instance][i] - concept_vec[concept][i]);
    }
    double t = concept_r[concept];

    if(sqrt(dis) < t){
        return true;
    }

   /* double tmp = concept_r[concept] / sqrt(dis);
    return tmp > delta_ins;*/
}


void init(){
    entity_vec.clear(); concept_vec.clear();
    concept_r.clear();
    ins_right.clear(); ins_wrong.clear(); sub_right.clear(); sub_wrong.clear();
}

void loadEmbedding(string epoch) {
    FILE* f1;
    FILE* f2;
    FILE* f3;
    f1 = fopen(("../data/" + dataSet + "/Output/entity2vec_OWL_"+epoch+".vec").c_str(), "r");
    f2 = fopen(("../data/" + dataSet + "/Output/relation2vec_OWL_"+epoch+".vec").c_str(), "r");
    f3 = fopen(("../data/" + dataSet + "/Output/concept2vec_OWL_"+epoch+".vec").c_str(), "r");
    
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
    concept_vec.resize(concept_num);
    concept_r.resize(concept_num);
    for(int i = 0; i < concept_num; ++i){
        concept_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f3, "%lf", &concept_vec[i][j]);
        }
        double tmp;
        fscanf(f3, "%lf", &tmp);
        concept_r[i] = tmp;
    }
}

void prepare(){
    init();

    int tmp = 0;
    FILE *fin_num = fopen(("../data/" + dataSet + "/Train/instance2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &entity_num);
    fclose(fin_num);
    fin_num = fopen(("../data/" + dataSet + "/Train/relation2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &relation_num);
    fclose(fin_num);
    fin_num = fopen(("../data/" + dataSet + "/Train/class2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &concept_num);
    fclose(fin_num);


    string tmp_str;
	ifstream class2id_file("../data/" + dataSet + "/Train/class2id.txt");
	getline(class2id_file, tmp_str);
	while (getline(class2id_file, tmp_str))
	{
		string::size_type pos = tmp_str.find(' ', 0);
		string classe = tmp_str.substr(0, pos);
		int id = atoi(tmp_str.substr(pos + 1).c_str());
		class2id.insert(pair<int, string>(id, classe));
	}
	class2id_file.close();


    ifstream fin, fin_right;
    fin.open(("../data/" + dataSet + "/Valid/valid2id_false.txt").c_str());
    fin_right.open(("../data/" + dataSet + "/Valid/valid2id.txt").c_str());
    fin_right >> valid_num;
    fin >> valid_num;

    delta_relation.resize(relation_num);
    max_min_relation.resize(relation_num);
    for(int i = 0; i < relation_num; ++i){
        max_min_relation[i].first = -1;
        max_min_relation[i].second = 1000000;
    }

    int tmp1, tmp2, tmp3;
    int inputSize = valid_num;
    total_triple += valid_num;
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

        vector<int> tripla;
        tripla.push_back(tmp1);
        tripla.push_back(tmp2);

        // Se non esiste la chiave tmp3, la creo
        if (relazioni_composte.find(tmp3) == relazioni_composte.end()) {
            relazioni_composte[tmp3] = vector<vector<int>>();
        }
        relazioni_composte[tmp3].push_back(tripla);
    }
    fin.close(); fin_right.close();


    int train_num;
    ifstream train_right;
    train_right.open(("../data/" + dataSet + "/Train/train2id.txt").c_str());
    train_right >> train_num;
    total_triple += train_num;
    for(int i = 0; i < train_num; ++i){
        train_right >> tmp1 >> tmp2 >> tmp3;
        vector<int> tripla;
        tripla.push_back(tmp1);
        tripla.push_back(tmp2);

        // Se non esiste la chiave tmp3, la creo
        if (relazioni_composte.find(tmp3) == relazioni_composte.end()) {
            relazioni_composte[tmp3] = vector<vector<int>>();
        }
        relazioni_composte[tmp3].push_back(tripla);
    }
    train_right.close();

    
    int test_num;
    ifstream test_right;
    test_right.open(("../data/" + dataSet + "/Test/test2id.txt").c_str());
    test_right >> test_num;
    total_triple += test_num;
    for(int i = 0; i < test_num; ++i){
        test_right >> tmp1 >> tmp2 >> tmp3;
        vector<int> tripla;
        tripla.push_back(tmp1);
        tripla.push_back(tmp2);

        // Se non esiste la chiave tmp3, la creo
        if (relazioni_composte.find(tmp3) == relazioni_composte.end()) {
            relazioni_composte[tmp3] = vector<vector<int>>();
        }
        relazioni_composte[tmp3].push_back(tripla);
    }
    test_right.close();

    /*
    fin.open(("../data/" + dataSet + "/Test/asymmetric.txt").c_str());
    for(int i = 0; i < 240; ++i){
        fin >> tmp1;
        relazioni_antisimmetriche.push_back(tmp1);
    }
    fin.close();fclose(fin_num);*/


    concept_instance.resize(concept_num);
	ifstream instanceOf_file("../data/" + dataSet + "/Test/instanceOf2id.txt");
	string tmpStr;
    // salta la prima riga
    getline(instanceOf_file, tmpStr);
	while (getline(instanceOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		int a = atoi(tmpStr.substr(0, pos).c_str());
		int b = atoi(tmpStr.substr(pos + 1).c_str());
        if(b >= concept_num) continue;
		concept_instance[b].push_back(a);
	}
	instanceOf_file.close();
    
    schema_classi_sub.resize(concept_num);
    schema_classi_up.resize(concept_num);
	ifstream subclassOf2id("../data/" + dataSet + "/Test/subclassOf2id_all.txt");
    // salta la prima riga
    getline(subclassOf2id, tmpStr);
	while (getline(subclassOf2id, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		int a = atoi(tmpStr.substr(0, pos).c_str());
		int b = atoi(tmpStr.substr(pos + 1).c_str());
        if(a >= concept_num || b >= concept_num) continue;
		schema_classi_sub[b].push_back(a);
		schema_classi_up[a].push_back(b);
	}
	subclassOf2id.close();

    
	ifstream disjoint_classes_file("../data/" + dataSet + "/Test/disjoint_classes.txt");
	getline(disjoint_classes_file, tmp_str);
	while (getline(disjoint_classes_file, tmp_str))
	{
		string::size_type pos = tmp_str.find(' ', 0);
		int classe = atoi(tmp_str.substr(pos + 1).c_str());
		disjoint_classes.push_back(classe);
	}
	disjoint_classes_file.close();

    fin.open(("../data/" + dataSet + "/Test/disjoint.txt").c_str());
    int num_disjoint = 0;
    fin >> num_disjoint;
     for(int i = 0; i < num_disjoint; ++i){
        fin >> tmp1 >> tmp2;

        // Se non esiste la chiave tmp1, la creo
        if (relazioni_disjoint.find(tmp1) == relazioni_disjoint.end()) {
            relazioni_disjoint[tmp1] = vector<int>();
        }
        relazioni_disjoint[tmp1].push_back(tmp2);
    }
    
    // Ordino le relazioni disjoint
    for(int concept0 = 0; concept0 < concept_num; ++concept0)
    {
        if (relazioni_disjoint.find(concept0) == relazioni_disjoint.end()) {
            relazioni_disjoint[concept0] = vector<int>();
        }

        sort(relazioni_disjoint[concept0].begin(), relazioni_disjoint[concept0].end());
    }

    fin.close();fclose(fin_num);

    loadEmbedding("500");
    
}



/*
 * Verifica se è possibile dedurre una regola di transitività tra instance e concept, cioè:
 * - se esiste restituisce la classe madre dell'istanza passata in input e allo stesso tempo sottoclasse
 * della classe passata in input;
 * - se non esiste restituisce -1;
 * 
 * La regola di transitività è la seguente:
 * (instance, instanceOf, concept) <- (concept_output, subClassOf, concept) && (instance, instanceOf, concept_output)
 *                                    con concept_output != concept
 */
vector<int> checkTransitivityInstanceOf(int instance, int concept) {
    vector<int> spiegazioni;

    for(int concept_output = 0; concept_output < concept_num; ++concept_output){
        if(concept_output != concept && checkSubClass(concept_output, concept) && checkInstance(instance, concept_output)) {
            spiegazioni.push_back(concept_output);
        }
    }

    return spiegazioni;
}

/*
 * Verifica se è possibile dedurre una regola di transitività tra concept1 e concept2
 * 
 * La regola di transitività è la seguente:
 * (concept1, subclassOf, concept2) <- (concept1, subClassOf, concept_output) && (concept_output, subClassOf, concept2)
 *                                    con concept_output != concept1 && concept_output != concept2
 */
vector<int> checkTransitivitySubclassOf(int concept1, int concept2) {
    vector<int> spiegazioni;

    for(int concept_output = 0; concept_output < concept_num; ++concept_output){
        if(concept_output != concept1 && concept_output != concept2 && 
        checkSubClass(concept1, concept_output) && checkSubClass(concept_output, concept2)) {
            spiegazioni.push_back(concept_output);
        }
    }

    return spiegazioni;
}


struct my_comparator
{
    // queue elements are vectors so we need to compare those
    bool operator()(std::vector<double> const& a, std::vector<double> const& b) const
    {
        // reverse sort puts the lowest value at the top    
        return a[0] > b[0];
    }
};

// for usability wrap this up in a type alias:
using my_priority_queue = std::priority_queue<std::vector<double>, std::vector<std::vector<double>>, my_comparator>;

/*
 * Verifica se è la relazione è antisimmetrica
 * 
 * La regola di antisimmetria è la seguente:
 * !(head, rel, tail) <- (tail, rel, head) per ogni head e tail
 */
bool checkAntisymmetry(int relation) {
    
    for(int i = 0; i < entity_num; ++i) {
        for(int j = 0; j < entity_num; ++j) {
            if (i != j && find(relazioni_escluse.begin(), relazioni_escluse.end(), i) == relazioni_escluse.end() && find(relazioni_escluse.begin(), relazioni_escluse.end(), j) == relazioni_escluse.end()) {
                if(checkRelation(i, j, relation)) {
                    if(checkRelation(j, i, relation)) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}



/*
 * Verifica se la relazione è antisimmetrica
 * 
 * La regola di antisimmetria è la seguente:
 * !(head, rel, tail) <- (tail, rel, head) per ogni head e tail
 */
map<int, vector<int>> checkDisjoint() {

    map<int, vector<int>> rel_disjoint;
    int count = 0;
    for(int concept1 = 0; concept1 < concept_num; ++concept1) {
        vector<int> tmp;
        
        if(find(disjoint_classes.begin(), disjoint_classes.end(), concept1) != disjoint_classes.end()) {
            for (int concept2 = 0; concept2 < concept_num; ++concept2) {

                if(find(disjoint_classes.begin(), disjoint_classes.end(), concept2) != disjoint_classes.end()) {
                    if(concept1 != concept2 && intersection_classes(concept1, concept2) == false) {
                        tmp.push_back(concept2);
                        count++;
                        //cout << "Le classi " << class2id[concept1] << " e " << class2id[concept2] << " sono disgiunte" << endl;
                    }
                }
            }
        }

        rel_disjoint.insert(pair<int, vector<int>>(concept1, tmp));
    }

    cout << "Numero di classi disgiunte: " << count << endl;

    return rel_disjoint;
}

/*
 * Verifica se la relazione è antisimmetrica
 * 
 * La regola di antisimmetria è la seguente:
 * !(head, rel, tail) <- (tail, rel, head) per ogni head e tail
 */
my_priority_queue checkAntisymmetry() {

    int min = 0;
    int max = -10000;

    my_priority_queue rel_anti_euc;
    //my_priority_queue rel_anti_spear;
    //my_priority_queue rel_anti_man;

    for (int relation = 0; relation < relation_num; ++relation) {
        if(find(relazioni_escluse.begin(), relazioni_escluse.end(), relation) == relazioni_escluse.end()) {
            
            double dis = 0;

            for(int i = 0; i < dim; ++i) {
                dis += fabs(relation_vec[relation][i] + relation_vec[relation][i]);
            }

            vector<double> item_euc;
            item_euc.push_back(dis);
            item_euc.push_back(relation);
            rel_anti_euc.push(item_euc);

            if(dis > max) {
                max = dis;
            }
                    /*   

            double sim_man = manhattan_similarity(vec_sum, vector2);
            vector<double> item_man;
            item_man.push_back(sim_man);
            item_man.push_back(relation);
            rel_anti_man.push(item_man);*/ 
        }
    }

/*
    // Crea una nuova coda con le relazioni ordinate per similarità in base alla posizione dei tre vettori
    my_priority_queue output;
    int pos1 = 1;
    while(!rel_anti_euc.empty()) {
        vector<double> vec1 = rel_anti_euc.top();
        rel_anti_euc.pop();


        // copia la coda rel_anti_man
        my_priority_queue rel_anti_man_temp = rel_anti_man;

        // cerca la relazione nella coda rel_anti_man
        vector<double> vec3;
        int pos3 = 1;
        while(!rel_anti_man_temp.empty()) {
            vec3 = rel_anti_man_temp.top();
            if(vec3[1] == vec1[1]) {
                break;
            }
            rel_anti_man_temp.pop();
            pos3++;
        }
        
        vector<double> item;
        item.push_back((pos1+pos3)/2);
        item.push_back(vec1[1]);
        item.push_back(vec1[0]);
        item.push_back(vec3[0]);
        output.push(item);

        pos1++;
    }*/

    my_priority_queue rel_output;

    // Applica il min max scaler alla coda con priorita
    while(!rel_anti_euc.empty()){
        vector<double> antisimmetrica=rel_anti_euc.top();
        rel_anti_euc.pop();
        antisimmetrica[0] = (antisimmetrica[0] - min) / (max - min);

        rel_output.push(antisimmetrica);
    }

    return rel_output;
}

/*
 * Verifica se la relazione ha delle relazioni inverse
 * 
 * La regola di inversa è la seguente:
 * (head, rel2, tail) <- (tail, rel1, head) per ogni head e tail
 * 
 * cioè se r1 = -r2
 */
my_priority_queue checkInverse() {

    my_priority_queue rel_inverse;

    for (int relation = 0; relation < relation_num; ++relation) {
        if(find(relazioni_escluse.begin(), relazioni_escluse.end(), relation) == relazioni_escluse.end()) {
            for(int rel = 0; rel < relation_num; rel++) {
                if(rel != relation && find(relazioni_escluse.begin(), relazioni_escluse.end(), rel) == relazioni_escluse.end()) {
                    double dis = 0;
                    for(int i = 0; i < dim; ++i) {
                        dis += fabs(relation_vec[relation][i] + relation_vec[rel][i]);
                    }

                    if(dis < 3) {
                        
                        vector<double> inversa;
                                    
                        inversa.push_back(dis);
                        inversa.push_back(relation);
                        inversa.push_back(rel);

                        rel_inverse.push(inversa);
                    }
                }
            }
        }
    }

    return rel_inverse;
}

/*
 * Verifica se la relazione r1 è composta con r2 con un'altra relazione r3
 * 
 * La regola di composizione è la seguente:
 * (x,r3,z) <- (x,r1,y) && (y,r2,z) 
 * 
 * cioè se r1 + r2 = r3
 */
my_priority_queue checkComposition() {
    
    my_priority_queue rel_composte;

    for (int rel1 = 0; rel1 < relation_num; ++rel1) {
        if(find(relazioni_escluse.begin(), relazioni_escluse.end(), rel1) == relazioni_escluse.end()) {
            for (int rel2 = 0; rel2 < relation_num; ++rel2) {
                if(rel1 != rel2 && find(relazioni_escluse.begin(), relazioni_escluse.end(), rel2) == relazioni_escluse.end()) {
                    for(int r3 = 0; r3 < relation_num; r3++) {
                        if(r3 != rel1 && r3 != rel2 && find(relazioni_escluse.begin(), relazioni_escluse.end(), r3) == relazioni_escluse.end()) {
                            double dis = 0;
                            for(int i = 0; i < dim; ++i) {
                                dis += fabs(relation_vec[rel1][i] + relation_vec[rel2][i] - relation_vec[r3][i]);
                            }

                            if(dis < 4) {
                                vector<double> composizione;
                                
                                composizione.push_back(dis);
                                composizione.push_back(rel1);
                                composizione.push_back(rel2);
                                composizione.push_back(r3);

                                rel_composte.push(composizione);
                            }
                        }
                    }
                }
            }
        }
    }

    return rel_composte;
}

/*
 * Ha in input una tripla (head, relation, tail) e da una spiegazione della predizione, nello specifico:
 * - Transitività instanceOf
 * - Transitività subClassOf
 * - Antisimmetria di una relazione
 * - Inversa di due relazioni
 *
 * Se la tripla è un assioma, allora la spiegazione è "La tripla (head, relation, tail) è un assioma."
 */
void explain(int head, int relation, int tail) {
    string vera_falsa = "";
    if(relation != instanceOf_id && relation != subClassOf_id) {
        if(checkRelation(head, tail, relation)) {
            vera_falsa = "vera";
        } else {
            vera_falsa = "falsa";
        }
    }

    if(relation == instanceOf_id){
        // transitività instanceOf
        vector<int> spiegazioni = checkTransitivityInstanceOf(head, tail);
        if(spiegazioni.size() > 0) {
            for (auto& concept_output : spiegazioni) {
                cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è vera perchè esiste (" << head << ", " << instanceOf_id << ", " << concept_output << ") e (" << concept_output << ", " << subClassOf_id << ", " << tail << ")"<< endl;
            }
        } else {
            //cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è un assioma." << endl;
        }

    } else if(relation == subClassOf_id){

        // transitività subClassOf
        vector<int> spiegazioni = checkTransitivitySubclassOf(head, tail);
        if(spiegazioni.size() > 0) {
            for (auto& concept_output : spiegazioni) {
                cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è vera perchè esiste (" << head << ", " << subClassOf_id << ", " << concept_output << ") e (" << concept_output << ", " << subClassOf_id << ", " << tail << ")"<< endl;
            }

        } else {
            //cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è un assioma." << endl;
        }
    } else {

        // proprietà antisimmetrica
        if(vera_falsa == "falsa") {
            if(delta_relation[relation] > 0 && checkAntisymmetry(relation)) {
                cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è falsa perchè è vera (" << tail << ", " << relation << ", " << head << ") e " << relation << " è antisimmetrica." << endl;
            }
        }
        

        /*

        if(vera_falsa == "vera" && delta_relation[relation] > 0) {
            vector<int> inverse = checkInverse(relation);
            // proprietà inversa
            if(inverse.size() > 0) {
                for (auto& rel_inv : inverse) {
                    cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è vera perchè è vera (" << tail << ", " << rel_inv << ", " << head << ") e le due relazioni sono tra loro inverse." << endl;
                }

                //cout << "La tripla (" << head << ", " << relation << ", " << tail << ") è un assioma." << endl;
            }
        }*/
    }
    
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

vector<double> test(){
    double TP = 0, TN = 0, FP = 0, FN = 0;
    vector<vector<double> > ans;
    ans.resize(relation_num);

    for(int i = 0; i < relation_num; ++i) {
        ans[i].resize(4);
        ans[i][0] = 0; ans[i][1] = 0; ans[i][2] = 0; ans[i][3] = 0;
    }
    int inputSize = valid_num;
    for(int i = 0; i < inputSize; ++i){
        if(checkRelation(right_triple[i][0], right_triple[i][1], right_triple[i][2])) {
            TP++;
            ans[right_triple[i][2]][0]++;
        }
        else{
            FN++;
            ans[right_triple[i][2]][1]++;
        }
        if(!checkRelation(wrong_triple[i][0], wrong_triple[i][1], wrong_triple[i][2])) {
            TN++;
            ans[wrong_triple[i][2]][2]++;
        }
        else {
            FP++;
            ans[wrong_triple[i][2]][3]++;
        }
    }

    vector<double> returnAns;
    returnAns.resize(relation_num);
    for(int i = 0; i < relation_num; ++i){
        returnAns[i] = (ans[i][0] + ans[i][2]) * 100 / (ans[i][0] + ans[i][1] + ans[i][2] + ans[i][3]);
    }
    return returnAns;
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
}

void escludi_relazioni() {
    for(int r1 = 0; r1 < relation_num; r1++) {
        double sum = 0;
        for(int i = 0; i < dim; ++i) {
            sum += fabs(relation_vec[r1][i]);
        }

        if(sum < 1) {
            relazioni_escluse.push_back(r1);
        }
    }

    if(dataSet == "DBpediaYAGO" || dataSet == "DBpedia15K") {
        // Escludi la right tributary e depiction e province
        relazioni_escluse.push_back(220);
        relazioni_escluse.push_back(278);
        relazioni_escluse.push_back(205);
        relazioni_escluse.push_back(107);
        relazioni_escluse.push_back(108);
    }

    cout << "Relazioni escluse: " << endl;
    for(auto& r : relazioni_escluse) {
        cout << r << " ";
    }
    cout << endl;
}


// Ritrova le triple in base all'entity e alla relazione
// entity false = head
// entity true = tail
vector<int> getTriples(int relation, bool entity, int x) {

    vector<int> triples;

    for(auto&r  : relazioni_composte[relation]) {
        if(entity) {
            if(x == r[1]) {
                triples.push_back(r[0]);
            }
        } else {
            if(x == r[0]) {
                triples.push_back(r[1]);
            }
        }
    }

    return triples;
}

unordered_map<int, std::vector<int>> cache_classi;

vector<int> getInstancesOfClass(int classe1) {
    if (cache_classi.count(classe1) > 0) {
        // Se le istanze di classe sono già state calcolate, restituisci il risultato memorizzato nella cache_classi
        return cache_classi[classe1];
    }

    vector<int> instances;

    for(auto& i : concept_instance[classe1]) {
        instances.push_back(i);
    }

    
    for(auto& c : schema_classi_sub[classe1]) {
        auto instances2 = getInstancesOfClass(c);
        for(auto& i : instances2) {
            instances.push_back(i);
        }
    }

    // ordina instances
    sort(instances.begin(), instances.end());

    
    cache_classi[classe1] = instances;

    return instances;
}

int num_instances(int classe) {
    vector<int> instances1 = getInstancesOfClass(classe);

    return instances1.size();
}
int find_comuni(int classe1, int classe2) {
    std::vector<int> instances1 = getInstancesOfClass(classe1);
    std::vector<int> instances2 = getInstancesOfClass(classe2);

    int comuni = 0;
    int i = 0; // Indice per instances1
    int j = 0; // Indice per instances2

    while (i < instances1.size() && j < instances2.size()) {
        if (instances1[i] < instances2[j]) {
            i++;
        } else if (instances1[i] > instances2[j]) {
            j++;
        } else {
            comuni++;
            i++;
            j++;
        }
    }

    return comuni;
}
vector<int> check_disjoint(map<int, vector<int>> disjoint, int classe) {
    // preleva le disgiunzioni delle classi superiori
    vector<int> disgiunzioni = disjoint[classe];

     // cicla su schema classi
    for(auto& c : schema_classi_up[classe]) {
        // preleva le disgiunzioni delle classi superiori
        vector<int> disgiunzioni2 = check_disjoint(disjoint, c);

        // aggiungi le disgiunzioni
        for(auto& d : disgiunzioni2) {
            disgiunzioni.push_back(d);
        }
    }

    return disgiunzioni;
    
}
vector<int> remove_duplicates(map<int, vector<int>> disjoint, int classe) {
    vector<int> disgiunzioni = check_disjoint(disjoint, classe);

    // inserisce in una lista i valori che appaiono più di una volta
    vector<int> duplicati;
    for(int d : disgiunzioni) {

        // conta le occorrenze di d in disgiunzioni
        int count = 0;
        for(int d2 : disgiunzioni) {
            if(d2 == d) {
                count++;

                if(count > 1) {
                    break;
                }
            }
        }

        if(count > 1) {
            duplicati.push_back(d);
        }
    }

    vector<int> disj = disjoint[classe];

    // rimuove i duplicati
    for(auto& d : duplicati) {
        disj.erase(remove(disj.begin(), disj.end(), d), disj.end());
    }

    return disj;
}



int main(int argc, char**argv){
    int i = 0;
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) dataSet = argv[i + 1];
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    prepare();
    //runValid();
    
    escludi_relazioni();

/*
    auto rel_composte = checkComposition();
    my_priority_queue rel_composte_finali;
    while(!rel_composte.empty()){
        vector<double> composta=rel_composte.top();
        rel_composte.pop();

        //cout << composta[0] << ": La relazione " << composta[1] << " è composta con " << composta[2] << " = " << composta[3] << endl;

        // Trova le y della relazione 1
        double num = 0;
        double common_num_triple = 0;
        for(auto&r  : relazioni_composte[composta[1]]) {
            int y = r[1];
            vector<int> triple_comuni = getTriples(composta[2], false, y);
            if(triple_comuni.size() > 0) {
                for (auto& t : triple_comuni) {
                    if (checkRelation(r[0], t, composta[3])) {
                        num++;
                    }
                }
            }
            common_num_triple += triple_comuni.size();
        }

        if(common_num_triple < 10 || num == 0) {
            continue;
        }
        double confidenza = num / common_num_triple;
        double lift_num =(num / total_triple);
        double rel3_num = relazioni_composte[composta[3]].size();
        double lift_den =((rel3_num / total_triple) * (common_num_triple / total_triple));

        double lift = lift_num / lift_den;
        rel_composte_finali.push({lift, confidenza, composta[1], composta[2], composta[3]});

    } 

    while(!rel_composte_finali.empty()){

        vector<double> composta=rel_composte_finali.top();
        rel_composte_finali.pop();
        cout << composta[0] << " - " << composta[1] << ": La relazione " << composta[2] << " è composta con " << composta[3] << " = " << composta[4] << endl;
    }

    */

    /*

    auto rel_antisimm = checkAntisymmetry();
    int rel_antisimm_count = 0;

    for(int k = 0; k < 240; k++) {
        vector<double> antisimmetrica=rel_antisimm.top();
        rel_antisimm.pop();

        if(find(relazioni_antisimmetriche.begin(), relazioni_antisimmetriche.end(), antisimmetrica[1]) != relazioni_antisimmetriche.end()) {
            rel_antisimm_count++;
        }
        
        cout << antisimmetrica[0] << ":Relazione antisimmetrica " << antisimmetrica[1] << endl;
    }

    cout << "Relazioni antisimmetriche: " << rel_antisimm_count << endl;

    double prob = 0;

    rel_antisimm = checkAntisymmetry();
    // itera su relazioni antisimmetriche
    for(auto& r : relazioni_antisimmetriche) {

        // fai una copia di rel_antisimm
        auto rel_antisimm_copy = rel_antisimm;

        // cerca la relazione in rel_antisimm
        while(!rel_antisimm_copy.empty()){
            vector<double> antisimmetrica=rel_antisimm_copy.top();
            rel_antisimm_copy.pop();

            if(antisimmetrica[1] == r) {
                prob += antisimmetrica[0];
                break;
            }
        }
    }

    prob = prob / relazioni_antisimmetriche.size();

    cout << "PROB Relazioni antisimmetriche: " << prob << endl;
    */

    FILE* results = fopen(("../data/" + dataSet + "/Output/results_expl.csv").c_str(), "w");

    vector<double> lift_arr;
    for(int epoch = 500; epoch <= 1000; epoch += 100) {
        
        loadEmbedding(to_string(epoch));

        map<int, vector<int>> rel_disjoint = checkDisjoint();
        int vere = 0;
        int vere_reali = 0;
        int num_rel_disjoint = 0;
        double c1_istanze_tot = 0;
        double c2_istanze_tot = 0;
        double istanze_in_comune_tot = 0;
        double istanze_tot = 0;

        for(int concept0 = 0; concept0 < concept_num; concept0++) {
            int vere_tmp = 0;

            if(rel_disjoint[concept0].size() == 0) {
                continue;
            }

            //cout << "Classe " << concept0 << endl;

            //auto array_disjoint = remove_duplicates(rel_disjoint, concept0);
            auto array_disjoint = rel_disjoint[concept0];
            
            int c1_istanze = num_instances(concept0);

            num_rel_disjoint += array_disjoint.size();

            int vere_totali = relazioni_disjoint[concept0].size();
            vere_reali += vere_totali;
            if(vere_totali > 0) {

                // merge sort
                int i = 0; // Indice per la prima lista
                int j = 0; // Indice per la seconda lista

                // Itera finché entrambi gli indici sono all'interno dei limiti delle rispettive liste
                while (i < array_disjoint.size()) {
                    if (j < relazioni_disjoint[concept0].size() && array_disjoint[i] == relazioni_disjoint[concept0][j]) {
                        // Trovata una corrispondenza
                        vere_tmp++;
                        vere++;

                        cout << "Le classi " << class2id[concept0] << " e " << class2id[array_disjoint[i]] << " sono disgiunte" << endl;

                        
                        // cerca le istanze in comune tra c1 e c2
                        int istanze_in_comune = find_comuni(concept0, array_disjoint[i]);

                        double c2_istanze = num_instances(array_disjoint[i]);
                        c1_istanze_tot += c1_istanze;
                        c2_istanze_tot += c2_istanze;
                        istanze_in_comune_tot += istanze_in_comune;
                        istanze_tot += c1_istanze + c2_istanze - istanze_in_comune;

                        if(c1_istanze > 10 && c2_istanze > 10) {
                            double instanze_tot_temp = c1_istanze + c2_istanze - istanze_in_comune;
                            double num = ((c1_istanze-istanze_in_comune)/instanze_tot_temp);
                            double den = (((c1_istanze/instanze_tot_temp)*((instanze_tot_temp-c2_istanze) / instanze_tot_temp)));
                            double lift = num / den;

                            if(lift > 1 && lift < 3)
                                lift_arr.push_back(lift);
                        }
                        

                        i++; // Incrementa l'indice per la prima lista
                        j++; // Incrementa l'indice per la seconda lista
                    } else if (j >= relazioni_disjoint[concept0].size() || array_disjoint[i] < relazioni_disjoint[concept0][j]) {
                        
                        // cerca le istanze in comune tra c1 e c2
                        int istanze_in_comune = find_comuni(concept0, array_disjoint[i]);

                        cout << "Le classi " << class2id[concept0] << " e " << class2id[array_disjoint[i]] << " NON sono disgiunte" << endl;

                        double c2_istanze = num_instances(array_disjoint[i]);
                        c1_istanze_tot += c1_istanze;
                        c2_istanze_tot += c2_istanze;
                        istanze_in_comune_tot += istanze_in_comune;
                        istanze_tot += c1_istanze + c2_istanze - istanze_in_comune;

                        if(c1_istanze > 10 && c2_istanze > 10) {
                            double instanze_tot_temp = c1_istanze + c2_istanze - istanze_in_comune;
                            double num = ((c1_istanze-istanze_in_comune)/instanze_tot_temp);
                            double den = (((c1_istanze/instanze_tot_temp)*((instanze_tot_temp-c2_istanze) / instanze_tot_temp)));
                            double lift = num / den;

                            if(lift > 1 && lift < 3)
                                lift_arr.push_back(lift);
                        }

                        // L'elemento nella prima lista è minore, quindi si incrementa l'indice per la prima lista
                        i++;
                    } else {

                        cout << "Le classi " << class2id[concept0] << " e " << class2id[array_disjoint[i]] << " NON sono disgiunte" << endl;

                        // L'elemento nella seconda lista è minore, quindi si incrementa l'indice per la seconda lista
                        j++;
                    }
                }


                /*
                //itera su concept
                for(auto& concept_check: array_disjoint) {

                    /*
                    if(!check_sisters(concept0, concept_check)) {
                        continue;
                    }

                    // cerca concept_check in relazioni_disjoint[concept0]
                    if (binary_search(relazioni_disjoint[concept0].begin(), relazioni_disjoint[concept0].end(), concept_check)) {
                        vere_tmp++;
                        vere++;

                        cout << "Le classi " << class2id[concept0] << " e " << class2id[concept_check] << " sono disgiunte" << endl;

                    } else {
                        cout << "NO" << endl;
                    }

                }*/

            }
        }


        /*

        // cicla su array_disjoint
        for(auto& c2: array_disjoint) {
            /*
            vector<double> disjoint;
            disjoint.push_back(lift);
            disjoint.push_back(c1);
            disjoint.push_back(c2);
            disjoint_queue.push(disjoint);
        }*/

        double precision = (double)vere / (double)num_rel_disjoint;
        double recall = (double)vere / (double)vere_reali;
        double prob_cond = (c1_istanze_tot-istanze_in_comune_tot)/c1_istanze_tot;
        double prob_implic = 1 - (istanze_in_comune_tot/istanze_tot);
        double num = ((c1_istanze_tot-istanze_in_comune_tot)/istanze_tot);
        double den = (((c1_istanze_tot/istanze_tot)*((istanze_tot-c2_istanze_tot) / istanze_tot)));
        double lift = num / den;

        double dev_std_lift = 0;
        double media_lift = 0;
        double max_lift = 0;
        double min_lift = 10;
        for(auto& l: lift_arr) {
            media_lift += l;
            if(l > max_lift)
                max_lift = l;
            if(l < min_lift)
                min_lift = l;
        }
        media_lift = media_lift / lift_arr.size();

        // calcola la dev_std
        for(auto& l: lift_arr) {
            dev_std_lift += pow(l - media_lift, 2);
        }
        dev_std_lift = sqrt(dev_std_lift / lift_arr.size());

        fprintf(results,"EPOCH %d \n", epoch);
        fprintf(results,"PREC %f\n", precision);
        fprintf(results,"REC %f\n", recall);
        fprintf(results,"LIFT %f\n", lift);
        fprintf(results,"PROB COND %f\n", prob_cond);
        fprintf(results,"PROB IMPLIC %f\n", prob_implic);
    }

    fclose(results);

    double max = 0;
    string max_rel = "";

    
    //double lift = (c1_istanze_tot-istanze_in_comune_tot)/((c1_istanze_tot*(istanze_tot-c2_istanze_tot))/(istanze_tot));
    //cout << "lift: " << lift << endl;

    /*
    double prob_implic = 1 - (istanze_in_comune_tot/istanze_tot);
    double prob_cond = (c1_istanze_tot-istanze_in_comune_tot)/c1_istanze_tot;
    cout << "prob_implic: " << prob_implic << endl;

    while(!disjoint_queue.empty()){
        vector<double> disj=disjoint_queue.top();
        disjoint_queue.pop();

        cout << disj[0] << ": " << class2id[disj[1]] << " " << class2id[disj[2]] << endl;
    }*/


/*
    while(!rel_antisimm.empty()){
        vector<double> antisimmetrica=rel_antisimm.top();
        rel_antisimm.pop();


        cout << antisimmetrica[0] << ":Relazione antisimmetrica " << antisimmetrica[1] << endl;
    }
    auto rel_inverse = checkInverse();
    while(!rel_inverse.empty()){
        vector<double> inversa=rel_inverse.top();
        rel_inverse.pop();

        cout << inversa[0] << ":L'inversa di " << inversa[1] << " è " << inversa[2] << endl;
    }

    int ins_test;
    ifstream fin;
    fin.open(("../data/" + dataSet + "/Train/subclassOf2id.txt").c_str());
    fin >> ins_test;

    int tmp1, tmp2;
    for(int i = 0; i < ins_test; ++i){
        fin >> tmp1 >> tmp2;
        if(tmp1 < concept_num && tmp2 < concept_num)
        explain(tmp1, subClassOf_id, tmp2);
    } 
    fin.close();
    */

}

