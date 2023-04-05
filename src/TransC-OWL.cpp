#include <iostream>
#include <cstring>
#include <map>
#include <vector>
#include <ctime>
#include <fstream>
#include <cmath>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <list>
#include <set>
#include <vector>


using namespace std;

#define pi 3.1415926535897932384626433832795

bool OWL = true;                 // indica se far partire l'algoritmo transC-OWL o transC
bool loadPath = false;            // indica se caricare i vettori già addestrati

string note = "";
const string typeOf = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";
int typeOf_id;

map<int, int> inverse;
map<int, int> equivalentRel;
map<int, int> equivalentClass;
multimap<int, int> rel2range;	// mappa una relazione nel range, che corrisponde ad una classe
multimap<int, int> rel2domain;	// mappa una relazione nel domain, che corrisponde ad una classe
multimap<int,int> cls2false_t;	//data una entità, restituisce una classe falsa (per tail corruption)
list<int> functionalRel;
bool L1Flag = true;
bool bern = false;
double ins_cut = 8.0;
double sub_cut = 8.0;
string dataSet = "DBpedia100K/";

double rand(double min, double max){
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

double normal(double x, double miu, double sigma){
    return 1.0 / sqrt(2 * pi) / sigma * exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma));
}

double randN(double miu, double sigma, double min, double max){
    double x, y, dScope;
    do{
        x = rand(min, max);
        y = normal(x, miu, sigma);
        dScope = rand(0.0, normal(miu, miu, sigma));
    }while(dScope > y);
    return x;
}

double sqr(double x){
    return x * x;
}

double vecLen(vector<double> &a){
    double res = 0;
    for(double i : a)
        res += i * i;
    res = sqrt(res);
    return res;
}

double norm(vector<double> &a) {
    double x = vecLen(a);
    if (x>1)
        for (double &i : a)
            i /=x;
    return 0;
}

void normR(double& r){
    if(r > 1)
        r = 1;
}

int randMax(int x){
    int res = (rand() * rand()) % x;
    while (res<0)
        res+=x;
    return res;
}

//vero se la relazione è di tipo functional
bool isFunctional(int rel_id) {
	if(functionalRel.size() == 0)
		return false;
	for(list<int>::iterator it = functionalRel.begin(); it != functionalRel.end(); ++it)
		if( (*it) == rel_id)
			return true;
	return false;
}

bool hasRange(int rel) {
    return !(rel2range.find(rel) == rel2range.end());
}

bool hasDomain(int rel) {
    return !(rel2domain.find(rel) == rel2range.end());
}



unsigned int relation_num, entity_num, concept_num, triple_num;
vector<vector<int> > concept_instance;
vector<vector<int> > instance_concept;
vector<vector<int> > instance_brother;
vector<vector<int> > sub_up_concept;
vector<vector<int> > up_sub_concept;
vector<vector<int> > concept_brother;
map<int,map<int,int> > left_entity, right_entity;
map<int,double> left_num, right_num;
int *lefHead, *rigHead;
int *lefTail, *rigTail;
double *left_mean, *right_mean;

struct Triple
{
    int h, r, t;
};

Triple *trainHead, *trainTail, *trainList;

struct cmp_head
{
    bool operator()(const Triple &a, const Triple &b)
    {
        return (a.h < b.h) || (a.h == b.h && a.r < b.r) || (a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

struct cmp_tail
{
    bool operator()(const Triple &a, const Triple &b)
    {
        return (a.t < b.t) || (a.t == b.t && a.r < b.r) || (a.t == b.t && a.r == b.r && a.h < b.h);
    }
};

class Train{
public:
    map<pair<int,int>, map<int,int> > ok;
    map<pair<int, int>, int> subClassOf_ok;
    map<pair<int, int>, int> instanceOf_ok;
    vector<pair<int, int>> subClassOf;
    vector<pair<int, int>> instanceOf;
    void addHrt(int x, int y, int z)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        ok[make_pair(x, z)][y]=1;
    }

    void addSubClassOf(int sub, int parent){
        subClassOf.emplace_back(sub, parent);
        subClassOf_ok[make_pair(sub, parent)] = 1;
    }

    void addInstanceOf(int instance, int concept){
        instanceOf.emplace_back(instance, concept);
        instanceOf_ok[make_pair(instance, concept)] = 1;
    }

    void load()
    {
        FILE *fin;
        int tmp;
        fin = fopen(("../data/" + dataSet + "Output/entity2vec" + note + ".vec").c_str(), "r");
        for (int i = 0; i < entity_num; i++)
        {
            for (int j = 0; j < n; j++)
                tmp = fscanf(fin, "%lf", &entity_vec[i][j]);
        }
        fclose(fin);
        fin = fopen(("../data/" + dataSet + "Output/relation2vec" + note + ".vec").c_str(), "r");
        for (int i = 0; i < relation_num; i++)
        {
            for (int j = 0; j < n; j++)
                tmp = fscanf(fin, "%lf", &relation_vec[i][j]);
        }
        fclose(fin);
        fin = fopen(("../data/" + dataSet + "Output/concept2vec" + note + ".vec").c_str(), "r");
        for (int i = 0; i < concept_num; i++) {
            for (int j = 0; j < n; j++)
                tmp = fscanf(fin, "%lf", &concept_vec[i][j]);
            
            tmp = fscanf(fin, "%lf", &concept_r[i]);
        }
        fclose(fin);
    }

    void setup(unsigned int n_in, double rate_in, double margin_in, double margin_ins, double margin_sub){
        n = n_in;
        rate = rate_in;
        margin = margin_in;
        margin_instance = margin_ins;
        margin_subclass = margin_sub;

        for(int i = 0; i < instance_concept.size(); ++i){
            for(int j = 0; j < instance_concept[i].size(); ++j){
                for(int k = 0; k < concept_instance[instance_concept[i][j]].size(); ++k){
                    if(concept_instance[instance_concept[i][j]][k] != i)
                        instance_brother[i].push_back(concept_instance[instance_concept[i][j]][k]);
                }
            }
        }

        for(int i = 0; i < sub_up_concept.size(); ++i){
            for(int j = 0; j < sub_up_concept[i].size(); ++j){
                for(int k = 0; k < up_sub_concept[sub_up_concept[i][j]].size(); ++k){
                    if(up_sub_concept[sub_up_concept[i][j]][k] != i){
                        concept_brother[i].push_back(up_sub_concept[sub_up_concept[i][j]][k]);
                    }
                }
            }
        }

        relation_vec.resize(relation_num);
        for(auto &i : relation_vec)
            i.resize(n);
        entity_vec.resize(entity_num);
        for(auto &i : entity_vec)
            i.resize(n);
        relation_tmp.resize(relation_num);
        for(auto &i : relation_tmp)
            i.resize(n);
        entity_tmp.resize(entity_num);
        for(auto &i : entity_tmp)
            i.resize(n);
        concept_vec.resize(concept_num);
        for(auto &i : concept_vec)
            i.resize(n);
        concept_tmp.resize(concept_num);
        for(auto &i : concept_tmp)
            i.resize(n);
        concept_r.resize(concept_num);
        concept_r_tmp.resize(concept_num);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randN(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randN(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }
        for(int i = 0; i < concept_num; ++i){
            for(int j = 0; j < n; ++j){
                concept_vec[i][j] = randN(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            }
            norm(concept_vec[i]);
        }
        for(int i = 0; i < concept_num; ++i){
            concept_r[i] = rand(0, 1);
        }
        trainSize = fb_h.size() + instanceOf.size() + subClassOf.size();
    }

    void doTrain(){
        int nbatches=100;
        int nepoch = 10000;
        int batchSize = trainSize/nbatches;
        for(int epoch = 0; epoch < nepoch; ++epoch){
            res = 0;
            for(int batch = 0; batch < nbatches; ++batch){
                relation_tmp = relation_vec;
                entity_tmp = entity_vec;
                concept_tmp = concept_vec;
                concept_r_tmp = concept_r;
                for(int k = 0; k < batchSize; ++k){
                    int i = randMax(trainSize);
                    if(i < fb_r.size()){
                        int cut = 10 - (int)(epoch * 8.0 / nepoch);
                        trainHLR(i, cut);
                    }else if(i < fb_r.size() + instanceOf.size()){
                        int cut = 10 - (int)(epoch * ins_cut / nepoch);
                        trainInstanceOf(i, cut);
                    }else{
                        int cut = 10 - (int)(epoch * sub_cut / nepoch);
                        trainSubClassOf(i, cut);
                    }
                }
                relation_vec = relation_tmp;
                entity_vec = entity_tmp;
                concept_vec = concept_tmp;
                concept_r = concept_r_tmp;
            }
            if(epoch % 1 == 0){
                cout<<"epoch:"<<epoch<<' '<<res<<endl;
            }
            if(epoch % 100 == 0 || epoch == nepoch - 1){
                FILE* f2 = fopen(("../data/" + dataSet + "Output/relation2vec" + note + "_" + to_string(epoch) + ".vec").c_str(), "w");
                FILE* f3 = fopen(("../data/" + dataSet + "Output/entity2vec" + note + "_" + to_string(epoch) + ".vec").c_str(), "w");
                FILE* f4 = fopen(("../data/" + dataSet + "Output/concept2vec" + note + "_" + to_string(epoch) + ".vec").c_str(), "w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                for (int i=0; i<concept_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f4,"%.6lf\t",concept_vec[i][ii]);
                    fprintf(f4,"\n");
                    fprintf(f4,"%.6lf\t", concept_r[i]);
                    fprintf(f4,"\n");
                }
                fclose(f2);
                fclose(f3);
                fclose(f4);
            }
        }
    }

private:
    unsigned int n;
    double res;
    double rate, margin, margin_instance, margin_subclass;
    int trainSize;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<double> > relation_vec, entity_vec, concept_vec;
    vector<vector<double> > relation_tmp, entity_tmp, concept_tmp;
    vector<double> concept_r, concept_r_tmp;

    void trainHLR(int i, int cut) {
        // esclude le typeof
        if(fb_r[i] == typeOf_id) return;

        int testa = fb_h[i];
        int coda = fb_l[i];
        int relazione = fb_r[i];
        int j; double pr = 500;
        int testaB, codaB;

        if(OWL) {
            auto pair = corrupt(testa, coda, relazione);
            testaB = pair.first;
            codaB = pair.second;
        } else {
            if(bern) pr = 1000 * right_num[relazione] / (right_num[relazione] + left_num[relazione]);
            if(rand() % 1000 < pr){
                do{
                    if(!instance_brother[coda].empty()){
                        if(rand() % 10 < cut){
                            j = randMax(entity_num);
                        }else{
                            j = rand() % (int)instance_brother[coda].size();
                            j = instance_brother[coda][j];
                        }
                    }else{
                        j = randMax(entity_num);
                    }
                }while(ok[make_pair(testa,relazione)].count(j)>0);

                testaB = testa;
                codaB = j;
            }else{
                do{
                    if(!instance_brother[testa].empty()){
                        if(rand() % 10 < cut){
                            j = randMax(entity_num);
                        }else{
                            j = rand() % (int)instance_brother[testa].size();
                            j = instance_brother[testa][j];
                        }
                    }else{
                        j = randMax(entity_num);
                    }
                }while (ok[make_pair(j,relazione)].count(coda)>0);

                testaB = j;
                codaB = coda;
            }
        }

        doTrainHLR(testa,coda,relazione,testaB,codaB,relazione);

        
        norm(relation_tmp[relazione]);
        norm(entity_tmp[testa]);
        norm(entity_tmp[coda]);
        norm(entity_tmp[testaB]);
        norm(entity_tmp[codaB]);

    }


    //vero se l'entità index è di classe class_id
    bool inRange(int id_rel, int id_obj) {
        //prendi le classi di id_obj!!
        auto cls = instance_concept[id_obj];

        auto ret = rel2range.equal_range(id_rel);
        for (multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it)
            if(find(cls.begin(), cls.end(),it->second)!=cls.end())
                return true;

        return false;
    }

    bool inDomain(int id_rel, int id_sub) {
        //prendi le classi di id_obj!!
        auto cls = instance_concept[id_sub];

        auto ret = rel2domain.equal_range(id_rel);
        for (multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it)
            if(find(cls.begin(), cls.end(),it->second)!=cls.end())
                return true;

        return false;
    }

    int corrupt_head(int h, int r) {
        int lef, rig, mid, ll, rr;
        lef = lefHead[h] - 1;
        rig = rigHead[h];
        while (lef + 1 < rig) {
            mid = (lef + rig) >> 1;
            if (trainHead[mid].r >= r) rig = mid; else
            lef = mid;
        }
        ll = rig;
        lef = lefHead[h];
        rig = rigHead[h] + 1;
        while (lef + 1 < rig) {
            mid = (lef + rig) >> 1;
            if (trainHead[mid].r <= r) lef = mid; else
            rig = mid;
        }
        rr = lef;
        int tmp = randMax(entity_num - (rr - ll + 1));
        if (tmp < trainHead[ll].t) return tmp;
        if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
        lef = ll, rig = rr + 1;
        while (lef + 1 < rig) {
            mid = (lef + rig) >> 1;
            if (trainHead[mid].t - mid + ll - 1 < tmp)
                lef = mid;
            else
                rig = mid;
        }

        return tmp + lef - ll + 1;
    }

    int corrupt_tail(int t, int r) {
        int lef, rig, mid, ll, rr;
        lef = lefTail[t] - 1;
        rig = rigTail[t];
        while (lef + 1 < rig) {
            mid = (lef + rig) >> 1;
            if (trainTail[mid].r >= r) rig = mid; else
            lef = mid;
        }
        ll = rig;
        lef = lefTail[t];
        rig = rigTail[t] + 1;
        while (lef + 1 < rig) {
            mid = (lef + rig) >> 1;
            if (trainTail[mid].r <= r) lef = mid; else
            rig = mid;
        }
        rr = lef;
        int tmp = randMax(entity_num - (rr - ll + 1));
        if (tmp < trainTail[ll].h) return tmp;
        if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
        lef = ll, rig = rr + 1;
        while (lef + 1 < rig) {
            mid = (lef + rig) >> 1;
            if (trainTail[mid].h - mid + ll - 1 < tmp)
                lef = mid;
            else
                rig = mid;
        }

        return tmp + lef - ll + 1;
    }

    int getHeadCorrupted(int coda, int relazione) {
        int j = -1;
        //Formula di Slovin per la dimensione del campione (generati da corrupt_tail)
        float error = 0.2f; //margine di errore del 20%
        int tries = entity_num / (1+ entity_num*error*error);
        for(int i = 0; i < tries; i++) {
            int corrpt_head = corrupt_tail(coda, relazione);
            if(!inDomain(relazione, corrpt_head)){
                j = corrpt_head;
                break;
            }
        }
        return j;
    }

    /*	Ottieni una coda corrotta
    *	Se la classe della coda non rispetta il range
    *		sceglila per l'addestramento
    *	altrimenti scegli un'altra coda
    *	ripeti per n tentativi, prima di rinunciare e andare col metodo standard
    */
    int getTailCorrupted(int testa, int relazione) {
        int j = -1;
        //Formula di Slovin per la dimensione del campione (generati da corrupt_tail)
        float error = 0.2f; //margine di errore del 20%
        int tries = entity_num / (1+ entity_num*error*error);
        for(int i = 0; i < tries; i++) {
            int corrpt_tail = corrupt_head(testa, relazione);
            if(!inRange(relazione, corrpt_tail)){
                j = corrpt_tail;
                break;
            }
        }
        return j;
    }


    pair<int, int> corrupt(int testa, int coda, int relazione) {
        int j = -1; double pr = 500;
        int testaB, codaB;

        if (bern)
            pr = 1000 * right_mean[relazione] / (right_mean[relazione] + left_mean[relazione]);
        else
            pr = 500;

        if (rand() % 1000 < pr || isFunctional(relazione)) {
            if(hasRange(relazione)) {
                j = getTailCorrupted(testa, relazione);
            }
            if(j == -1)
                j = corrupt_head(testa, relazione);

            testaB = testa;
            codaB = j;
        } else {
            if(hasDomain(relazione))
                j = getHeadCorrupted(coda, relazione);
            if(j == -1)
                j = corrupt_tail(coda, relazione);

            testaB = j;
            codaB = coda;
        }

        return make_pair(testaB, codaB);
    }

    void trainInstanceOf(int i, int cut){
        i = i - fb_h.size();
        int j = 0;
        if(rand() % 2 == 0){
            do{
                if(!instance_brother[instanceOf[i].first].empty()){
                    if(rand() % 10 < cut){
                        j = randMax(entity_num);
                    }else{
                        j = rand() % (int)instance_brother[instanceOf[i].first].size();
                        j = instance_brother[instanceOf[i].first][j];
                    }
                }else{
                    j = randMax(entity_num);
                }
            }while(instanceOf_ok.count(make_pair(j, instanceOf[i].second)) > 0);
            doTrainInstanceOf(instanceOf[i].first, instanceOf[i].second, j, instanceOf[i].second);
            norm(entity_tmp[j]);
        }else{
            do{
                if(!concept_brother[instanceOf[i].second].empty()){
                    if(rand() % 10 < cut){
                        j = randMax(concept_num);
                    }else{
                        j = rand() % (int)concept_brother[instanceOf[i].second].size();
                        j = concept_brother[instanceOf[i].second][j];
                    }
                }else{
                    j = randMax(concept_num);
                }
            }while(instanceOf_ok.count(make_pair(instanceOf[i].first, j)) > 0);
            doTrainInstanceOf(instanceOf[i].first, instanceOf[i].second, instanceOf[i].first, j);
            norm(concept_tmp[j]);
            normR(concept_r_tmp[j]);
        }
        norm(entity_tmp[instanceOf[i].first]);
        norm(concept_tmp[instanceOf[i].second]);
        normR(concept_r_tmp[instanceOf[i].second]);
    }

    void trainSubClassOf(int i, int cut){
        i = i - fb_h.size() - instanceOf.size();
        int j = 0;
        if(rand() % 2 == 0){
            do{
                if(!concept_brother[subClassOf[i].first].empty()){
                    if(rand() % 10 < cut){
                        j = randMax(concept_num);
                    }else{
                        j = rand() % (int)concept_brother[subClassOf[i].first].size();
                        j = concept_brother[subClassOf[i].first][j];
                    }
                }else{
                    j = randMax(concept_num);
                }
            }while(subClassOf_ok.count(make_pair(j, subClassOf[i].second)) > 0);
            doTrainSubClassOf(subClassOf[i].first, subClassOf[i].second, j, subClassOf[i].second);
        }else{
            do{
                if(!concept_brother[subClassOf[i].second].empty()){
                    if(rand() % 10 < cut){
                        j = randMax(concept_num);
                    }else{
                        j = rand() % (int)concept_brother[subClassOf[i].second].size();
                        j = concept_brother[subClassOf[i].second][j];
                    }
                }else{
                    j = randMax(concept_num);
                }
            }while(subClassOf_ok.count(make_pair(subClassOf[i].first, j)) > 0);
            doTrainSubClassOf(subClassOf[i].first, subClassOf[i].second, subClassOf[i].first, j);
        }
        norm(concept_tmp[subClassOf[i].first]);
        norm(concept_tmp[subClassOf[i].second]);
        norm(concept_tmp[j]);
        normR(concept_r_tmp[subClassOf[i].first]);
        normR(concept_r_tmp[subClassOf[i].second]);
        normR(concept_r_tmp[j]);
    }


    set<int> getEquivalentClass(int id_class) {
        set<int> cls;
        pair <std::multimap<int,int>::iterator, multimap<int,int>::iterator> ret;
        ret = equivalentClass.equal_range(id_class);
        for (multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it)
            cls.insert(it->second);
        return cls;
    }

    int getFalseClass(int classe) {
        int j = -1;
        pair <std::multimap<int,int>::iterator, multimap<int,int>::iterator> ret;
        ret = cls2false_t.equal_range(classe);
        vector<int> cls_vec;
        for (multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it) {
            cls_vec.push_back(it->second);
        }
        if(cls_vec.size() > 0) {
            int RandIndex = rand() % cls_vec.size();
            j = cls_vec[RandIndex];
        }
        return j;
}

    int getInverse(int rel)
    {
        if (inverse.size() == 0)
            return -1;
        if (inverse.find(rel) != inverse.end())
            return inverse.find(rel)->second;
        else
            return -1;
    }

    int getEquivalentProperty(int rel)
    {
        if (equivalentRel.size() == 0)
            return -1;
        if (equivalentRel.find(rel) != equivalentRel.end())
            return equivalentRel.find(rel)->second;
        else
            return -1;
    }


    void doTrainHLR(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b){
        double sum1 = calcSumHLT(e1_a,e2_a,rel_a);
        double sum2 = calcSumHLT(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
            res+=margin+sum1-sum2;

            if(OWL) {
                if (getInverse(rel_a) != -1)
                {
                    gradientInverseOf(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, getInverse(rel_a));
                }
                else if (getEquivalentProperty(rel_a) != -1)
                {
                    gradientEquivalentProperty(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, getEquivalentProperty(rel_a));
                }
                else
                {
                    gradientHLR(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
                }
            }
            else
            {
                gradientHLR(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
            }
        }
    }

    void doTrainInstanceOf(int e_a, int c_a, int e_b, int c_b){
        double sum1 = calcSumInstanceOf(e_a, c_a);
        double sum2 = calcSumInstanceOf(e_b, c_b);
        if(sum1 + margin_instance > sum2){
            res += (margin_instance + sum1 - sum2);
            gradientInstanceOf(e_a, c_a, e_b, c_b);
        }
    }

    void doTrainSubClassOf(int c1_a, int c2_a, int c1_b, int c2_b){
        double sum1 = calcSumSubClassOf(c1_a, c2_a);
        double sum2 = calcSumSubClassOf(c1_b, c2_b);
        if(sum1 + margin_subclass > sum2){
            res += (margin_subclass + sum1 - sum2);
            gradientSubClassOf(c1_a, c2_a, c1_b, c2_b);
        }
    }

    double calcSumHLT(int e1, int e2, int rel)
    {
        double sum=0;
        if (L1Flag)
            for (int ii=0; ii<n; ii++)
                sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
            for (int ii=0; ii<n; ii++)
                sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }

    double calcSumInstanceOf(int e, int c){
        double dis = 0;
        for(int i = 0; i < n; ++i){
            dis += sqr(entity_vec[e][i] - concept_vec[c][i]);
        }
        if(dis < sqr(concept_r[c])){
            return 0;
        }
        return dis - sqr(concept_r[c]);

    }

    double calcSumSubClassOf(int c1, int c2){
        double dis = 0;
        for(int i = 0; i < n; ++i){
            dis += sqr(concept_vec[c1][i] - concept_vec[c2][i]);
        }
        if(sqrt(dis) < fabs(concept_r[c1] - concept_r[c2])){
            return 0;
        }
        return dis - sqr(concept_r[c2]) + sqr(concept_r[c1]);

    }

    void gradientEquivalentProperty(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, int equivalent)
    {
        // se entità equivalenti, salva array
        for (int ii = 0; ii < n; ii++)
        {
            double x;
            x = (entity_tmp[e2_a][ii] - entity_tmp[e1_a][ii] - relation_tmp[rel_a][ii]);
            if (x > 0)
                x = -1;
            else
                x = 1;
            relation_tmp[rel_a][ii] -= x;
            relation_tmp[equivalent][ii] -= x;
            entity_tmp[e1_a][ii] -= x;
            entity_tmp[e2_a][ii] += x;
            x = (entity_tmp[e2_b][ii] - entity_tmp[e1_b][ii] - relation_tmp[rel_b][ii]);
            if (x > 0)
                x = 1;
            else
                x = -1;
            relation_tmp[rel_b][ii] -= x;
            relation_tmp[equivalent][ii] -= x;
            entity_tmp[e1_b][ii] -= x;
            entity_tmp[e2_b][ii] += x;
        }
    }

    void gradientInverseOf(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, int inverseRel)
    {
        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
            if (L1Flag){
                if (x>0){
                    x=1;
                }
                else{
                    x=-1;
                }
            }

            relation_tmp[rel_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]+=-1*rate*x;

            relation_tmp[inverseRel][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]+=-1*rate*x;

            // Tripla corrotta
            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
            if (L1Flag){
                if (x>0){
                    x=1;
                }
                else{
                    x=-1;
                }
            }
            relation_tmp[rel_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]-=rate*x;
            entity_tmp[e2_b][ii]+=rate*x;

            relation_tmp[inverseRel][ii]-=rate*x;
            entity_tmp[e2_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]+=rate*x;

        }

        norm(relation_tmp[inverseRel]);
    }

    void gradientHLR(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b){
        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
            if (L1Flag){
                if (x>0){
                    x=1;
                }
                else{
                    x=-1;
                }
            }
            relation_tmp[rel_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]+=-1*rate*x;
            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
            if (L1Flag){
                if (x>0){
                    x=1;
                }
                else{
                    x=-1;
                }
            }
            relation_tmp[rel_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]-=rate*x;
            entity_tmp[e2_b][ii]+=rate*x;
        }
    }

    void gradientInstanceOf(int e_a, int c_a, int e_b, int c_b) {

        bool corrupt = false;
        vector<int> corr;
        set<int> eq_cls = getEquivalentClass(c_a);

        if(OWL) {
            corrupt = true;
            for(set<int>::iterator it = eq_cls.begin(); it != eq_cls.end(); ++it) {
                int false_cl = getFalseClass(e_a);
                if(false_cl != -1)
                    corr.push_back(false_cl);
                else {
                    corrupt = false;
                    break;
                }
            }
        }

        double pos_x = gradientPositiveInstanceOf(e_a, c_a, false, 0);
        double neg_x = gradientNegativeInstanceOf(e_b, c_b, false, 0);

/*
        if(OWL && corrupt && eq_cls.size() > 0) {
			int i = 0;
			for(set<int>::iterator it = eq_cls.begin(); it != eq_cls.end(); ++it) {
				int e_Cls = (*it);
                gradientPositiveInstanceOf(e_a, e_Cls, true, pos_x);

                int tail = corr[i];
                gradientNegativeInstanceOf(e_b, tail, true, neg_x);
				i++;
			}
		}*/
    }

    double gradientPositiveInstanceOf(int e_a, int c_a, bool gradient, double dis) {
        if(gradient == false) {
            dis = 0;

            for(int i = 0; i < n; ++i){
                dis += sqr(entity_vec[e_a][i] - concept_vec[c_a][i]);
            }
        }

        if(dis > sqr(concept_r[c_a])){
            for(int j = 0; j < n; ++j){
                double x = 2 * (entity_vec[e_a][j] - concept_vec[c_a][j]);
                entity_tmp[e_a][j] -= x * rate;
                concept_tmp[c_a][j] -= -1 * x * rate;
            }
            concept_r_tmp[c_a] -= -2 * concept_r[c_a] * rate;
        }

        return dis;
    }
    
    double gradientNegativeInstanceOf(int e_a, int c_a, bool gradient, double dis) {
        if(gradient == false) {
            dis = 0;

            for(int i = 0; i < n; ++i){
                dis += sqr(entity_vec[e_a][i] - concept_vec[c_a][i]);
            }
        }

        if(dis > sqr(concept_r[c_a])){
            for(int j = 0; j < n; ++j){
                double x = 2 * (entity_vec[e_a][j] - concept_vec[c_a][j]);
                entity_tmp[e_a][j] += x * rate;
                concept_tmp[c_a][j] += -1 * x * rate;
            }
            concept_r_tmp[c_a] += -2 * concept_r[c_a] * rate;
        }

        return dis;
    }

    void gradientSubClassOf(int c1_a, int c2_a, int c1_b, int c2_b){
        double dis = 0;
        for(int i = 0; i < n; ++i){
            dis += sqr(concept_vec[c1_a][i] - concept_vec[c2_a][i]);
        }
        if(sqrt(dis) > fabs(concept_r[c1_a] - concept_r[c2_a])){
            for(int i = 0; i < n; ++i){
                double x = 2 * (concept_vec[c1_a][i] - concept_vec[c2_a][i]);
                concept_tmp[c1_a][i] -= x * rate;
                concept_tmp[c2_a][i] -= -x * rate;
            }
            concept_r_tmp[c1_a] -= 2 * concept_r[c1_a] * rate;
            concept_r_tmp[c2_a] -= -2 * concept_r[c2_a] * rate;
        }

        dis = 0;
        for(int i = 0; i < n; ++i){
            dis += sqr(concept_vec[c1_b][i] - concept_vec[c2_b][i]);
        }
        if(sqrt(dis) > fabs(concept_r[c1_b] - concept_r[c2_b])){
            for(int i = 0; i < n; ++i){
                double x = 2 * (concept_vec[c1_b][i] - concept_vec[c2_b][i]);
                concept_tmp[c1_b][i] += x * rate;
                concept_tmp[c2_b][i] += -x * rate;
            }
            concept_r_tmp[c1_b] += 2 * concept_r[c1_b] * rate;
            concept_r_tmp[c2_b] += -2 * concept_r[c2_b] * rate;
        }
    }
};
Train train;


void OWLinit(map<string,int> rel2id, map<string,int> class2id) {

	string tmpStr, tmp;

	// trovo le relazioni inverseOf
	ifstream inverse_file("../data/" + dataSet + "Train/inverseOf.txt");
	while (getline(inverse_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		string first = tmp.substr(0, pos);
		if (rel2id.find(first) != rel2id.end())
		{
			string second = tmp.substr(pos + 1);
			second = second.substr(0, second.length());
			if (rel2id.find(second) != rel2id.end())
			{
				int id_first = rel2id.find(first)->second;
				int id_second = rel2id.find(second)->second;
				inverse.insert(pair<int, int>(id_first, id_second));
				inverse.insert(pair<int, int>(id_second, id_first));
			}
		}
	}
	inverse_file.close();

	ifstream eqProp_file("../data/" + dataSet + "Train/equivalentProperty.txt");
	while (getline(eqProp_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		string first = tmp.substr(0, pos);
		if (rel2id.find(first) != rel2id.end())
		{
			string second = tmp.substr(pos + 1);
			second = second.substr(0, second.length());
			if (rel2id.find(second) != rel2id.end())
			{
				int id_first = rel2id.find(first)->second;
				int id_second = rel2id.find(second)->second;
				equivalentRel.insert(pair<int, int>(id_first, id_second));
				equivalentRel.insert(pair<int, int>(id_second, id_first));
			}
		}
	}
	eqProp_file.close();

    ifstream eqClass_file("../data/" + dataSet + "Train/equivalentClass.txt");
    while (getline(eqClass_file, tmp)) {
    	string::size_type pos=tmp.find('\t',0);
		string first = tmp.substr(0,pos);
		if(class2id.find(first) != class2id.end()) {
			string second= tmp.substr(pos+1);
			second = second.substr(0, second.length());
			if(class2id.find(second) != class2id.end()) {
				int id_first = class2id.find(first)->second;
				int id_second = class2id.find(second)->second;
				equivalentClass.insert(pair<int,int>(id_first, id_second));
				equivalentClass.insert(pair<int,int>(id_second,id_first));
			}
		}
    }
    eqClass_file.close();


    ifstream false_file("../data/" + dataSet + "Train/falseinstanceOf2id.txt");
    getline(false_file, tmp);
	while (getline(false_file, tmp)) {
		string::size_type pos = tmp.find(' ',0);
		string last_part= tmp.substr(pos+1);
		int head = atoi(tmp.substr(0,pos).c_str());
		pos=last_part.find(' ',0);
		int cls = atoi(last_part.substr(0,pos).c_str());
		cls2false_t.insert(pair<int,int>(head,cls));
		//cls2false_h.insert(pair<int,int>(cls,head));
	}

    ifstream domain_file("../data/" + dataSet + "Train/rs_domain2id.txt");
    getline(domain_file, tmp);
    while (getline(domain_file, tmp)) {
        string::size_type pos=tmp.find(' ',0);
    	int relation= atoi(tmp.substr(0,pos).c_str());
    	int domain = atoi(tmp.substr(pos+1).c_str());
    	rel2domain.insert(pair<int,int>(relation,domain));
    }
    domain_file.close();

    ifstream range_file("../data/" + dataSet + "Train/rs_range2id.txt");
    getline(range_file, tmp);
    while (getline(range_file, tmp)) {
        string::size_type pos=tmp.find(' ',0);
    	int relation= atoi(tmp.substr(0,pos).c_str());
    	int range = atoi(tmp.substr(pos+1).c_str());
    	rel2range.insert(pair<int,int>(relation,range));
    }
    range_file.close();

    //trovo le relazioni di tipo functional
    ifstream function_file("../data/" + dataSet + "Train/functionalProperty.txt");
    while (getline(function_file, tmp)) {
        if (rel2id.find(tmp) != rel2id.end()){
            functionalRel.push_front(rel2id.find(tmp)->second);
        }
    }
    function_file.close();
}

void prepare(){
    if(OWL)
        note = "_OWL";

    map<string,int> rel2id;
    map<string,int> ent2id;
	map<string, int> class2id;

    FILE* f1 = fopen(("../data/" + dataSet + "Train/instance2id.txt").c_str(),"r");
    FILE* f2 = fopen(("../data/" + dataSet + "Train/relation2id.txt").c_str(),"r");
    FILE* f3 = fopen(("../data/" + dataSet + "Train/class2id.txt").c_str(),"r");
    FILE* f_kb = fopen(("../data/" + dataSet + "Train/train2id.txt").c_str(),"r");
    fscanf(f1, "%d", &entity_num);
    fscanf(f2, "%d", &relation_num);
    fscanf(f3, "%d", &concept_num);
    fscanf(f_kb, "%d", &triple_num);
    int h, t, l;

	trainHead = (Triple *)calloc(triple_num, sizeof(Triple));
	trainTail = (Triple *)calloc(triple_num, sizeof(Triple));
    int i = 0;
    while (fscanf(f_kb, "%d%d%d", &h, &t, &l) == 3) {
        train.addHrt(h, t, l);
        if(bern){
            left_entity[l][h]++;
            right_entity[l][t]++;
        }

		trainHead[i].h = h;
		trainHead[i].r = l;
		trainHead[i].t = t;
		trainTail[i].h = h;
		trainTail[i].r = l;
		trainTail[i].t = t;
        i++;
    }

	sort(trainHead, trainHead + triple_num, cmp_head());
	sort(trainTail, trainTail + triple_num, cmp_tail());

	lefHead = (int *)calloc(entity_num, sizeof(int));
	rigHead = (int *)calloc(entity_num, sizeof(int));
	lefTail = (int *)calloc(entity_num, sizeof(int));
	rigTail = (int *)calloc(entity_num, sizeof(int));
	memset(rigHead, -1, sizeof(int) * entity_num);
	memset(rigTail, -1, sizeof(int) * entity_num);
	for (int i = 1; i < triple_num; i++)
	{
		if (trainTail[i].t != trainTail[i - 1].t)
		{
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h)
		{
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[triple_num - 1].h] = triple_num - 1;
	rigTail[trainTail[triple_num - 1].t] = triple_num - 1;

	left_mean = (double *)calloc(relation_num * 2, sizeof(double));
	right_mean = left_mean + relation_num;
	for (int i = 0; i < entity_num; i++)
	{
		for (int j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}

    fclose(f_kb);
    fclose(f1);
    fclose(f2);
    fclose(f3);
    concept_instance.resize(concept_num);
    instance_concept.resize(entity_num);
    sub_up_concept.resize(concept_num);
    up_sub_concept.resize(concept_num);
    instance_brother.resize(entity_num);
    concept_brother.resize(concept_num);

    if(bern){
        for (int i=0; i<relation_num; i++) {
            double sum1=0,sum2=0;
            for (auto it = left_entity[i].begin(); it != left_entity[i].end(); it++)
            {
                sum1++;
                sum2+=it->second;
            }
            left_num[i]=sum2/sum1;
        }
        for (int i=0; i<relation_num; i++) {
            double sum1=0,sum2=0;
            for (auto it = right_entity[i].begin(); it != right_entity[i].end(); it++)
            {
                sum1++;
                sum2+=it->second;
            }
            right_num[i]=sum2/sum1;
        }
    }

    string tmp;

	ifstream class2id_file("../data/" + dataSet + "Train/class2id.txt");
	getline(class2id_file, tmp);
	while (getline(class2id_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		string classe = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		class2id.insert(pair<string, int>(classe, id));
	}
	class2id_file.close();

	// carico le relazioni per il confronto
	ifstream rel_file("../data/" + dataSet + "Train/relation2id.txt");
	getline(rel_file, tmp);
	while (getline(rel_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		string rel = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		rel2id.insert(pair<string, int>(rel, id));
        if(rel == typeOf)
        	typeOf_id = id;
    }
    printf("ID TypeOf: %d\n",typeOf_id);
	rel_file.close();


	// carico le entità per il confronto
	ifstream ent_file("../data/" + dataSet + "Train/instance2id.txt");
	getline(ent_file, tmp);
	while (getline(ent_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		string ent = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		ent2id.insert(pair<string, int>(ent, id));
	}
	ent_file.close();

	ifstream instanceOf_file("../data/" + dataSet + "Train/instanceOf2id.txt");
	string tmpStr;

    // salta la prima riga
    getline(instanceOf_file, tmpStr);
	while (getline(instanceOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		int a = atoi(tmpStr.substr(0, pos).c_str());
		int b = atoi(tmpStr.substr(pos + 1).c_str());
		train.addInstanceOf(a, b);
		instance_concept[a].push_back(b);
		concept_instance[b].push_back(a);
	}
	instanceOf_file.close();

	ifstream subclassOf_file("../data/" + dataSet + "Train/subclassOf2id.txt");
    // salta la prima riga
    getline(subclassOf_file, tmpStr);
	while (getline(subclassOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		int a = atoi(tmpStr.substr(0, pos).c_str());
		int b = atoi(tmpStr.substr(pos + 1).c_str());
		train.addSubClassOf(a, b);
		sub_up_concept[a].push_back(b);
		up_sub_concept[b].push_back(a);
	}
	subclassOf_file.close();

    if(OWL) {
        OWLinit(rel2id, class2id);
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

int main(int argc, char** argv){
    srand((unsigned) time(NULL));
    int n = 100, i = 0;
    double rate = 0.001, margin = 1, margin_ins = 0.4, margin_sub = 0.3;
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin_ins", argc, argv)) > 0) margin_ins = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin_sub", argc, argv)) > 0) margin_sub = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) dataSet = argv[i + 1];
    if ((i = ArgPos((char *)"-l1flag", argc, argv)) > 0) L1Flag = static_cast<bool>(atoi(argv[i + 1]));
    if ((i = ArgPos((char *)"-bern", argc, argv)) > 0) bern = static_cast<bool>(atoi(argv[i + 1]));
    bern = true;
    L1Flag = false;
    cout << "vector dimension = " << n << endl;
    cout << "learing rate = " << rate << endl;
    cout << "margin = " << margin << endl;
    cout << "margin_ins = " << margin_ins << endl;
    cout << "margin_sub = " << margin_sub << endl;
    if (L1Flag)
        cout << "L1 Flag = " << "True" << endl;
    else
        cout << "L1 Flag = " << "False" << endl;
    if (bern)
        cout << "bern = " << "True" << endl;
    else
        cout << "bern = " << "False" << endl;
    if (OWL)
        cout << "OWL = " << "True" << endl;
    else
        cout << "OWL = " << "False" << endl;
    prepare();
    train.setup(n, rate, margin, margin_ins, margin_sub);
	if (loadPath) train.load();
    train.doTrain();
}