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
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <list>
#include <set>
#include <vector>


using namespace std;

#define pi 3.1415926535897932384626433832795

map<int, int> inverse;
map<int, int> equivalentRel;
bool L1Flag = true;
bool bern = false;
double ins_cut = 8.0;
double sub_cut = 8.0;
string dataSet = "YAGO39K";

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

unsigned int relation_num, entity_num, concept_num, triple_num;
vector<vector<int> > concept_instance;
vector<vector<int> > instance_concept;
vector<vector<int> > instance_brother;
vector<vector<int> > sub_up_concept;
vector<vector<int> > up_sub_concept;
vector<vector<int> > concept_brother;
map<int,map<int,int> > left_entity, right_entity;
map<int,double> left_num, right_num;

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
                FILE* f2 = fopen(("Output/relation2vec_OWL.vec"), "w");
                FILE* f3 = fopen(("Output/entity2vec_OWL.vec"), "w");
                FILE* f4 = fopen(("Output/concept2vec_OWL.vec"), "w");
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

    void trainHLR(int i, int cut){
        int j; double pr = 500;
        if(bern) pr = 1000 * right_num[fb_r[i]] / (right_num[fb_r[i]] + left_num[fb_r[i]]);
        if(rand() % 1000 < pr){
            do{
                if(!instance_brother[fb_l[i]].empty()){
                    if(rand() % 10 < cut){
                        j = randMax(entity_num);
                    }else{
                        j = rand() % (int)instance_brother[fb_l[i]].size();
                        j = instance_brother[fb_l[i]][j];
                    }
                }else{
                    j = randMax(entity_num);
                }
            }while(ok[make_pair(fb_h[i],fb_r[i])].count(j)>0);
            doTrainHLR(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i]);
        }else{
            do{
                if(!instance_brother[fb_h[i]].empty()){
                    if(rand() % 10 < cut){
                        j = randMax(entity_num);
                    }else{
                        j = rand() % (int)instance_brother[fb_h[i]].size();
                        j = instance_brother[fb_h[i]][j];
                    }
                }else{
                    j = randMax(entity_num);
                }
            }while (ok[make_pair(j,fb_r[i])].count(fb_l[i])>0);
            doTrainHLR(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i]);
        }
        norm(relation_tmp[fb_r[i]]);
        norm(entity_tmp[fb_h[i]]);
        norm(entity_tmp[fb_l[i]]);
        norm(entity_tmp[j]);
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
    void gradientInverseOf(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, int inverseRel)
    {
        int lasta1 = e1_a * n;
        int lasta2 = e2_a * n;
        int lastar = rel_a * n;
        int lastb1 = e1_b * n;
        int lastb2 = e2_b * n;
        int lastbr = rel_b * n;
        int lastInverse = inverseRel * n;
        // se entità equivalenti, salva array
        for (int ii = 0; ii < n; ii++)
        {
            double x;
            x = (entity_tmp[e2_a][ii] - entity_tmp[e1_a][ii] - relation_vec[rel_a][ii]);
            if (x > 0)
                x = -1;
            else
                x = 1;

            relation_vec[rel_a][ii] -= x;
            entity_tmp[e1_a][ii] -= x;
            entity_tmp[e2_a][ii] += x;

            relation_vec[inverseRel][ii] -= x;
            entity_tmp[e2_a][ii] -= x;
            entity_tmp[e1_a][ii] += x;

            x = (entity_tmp[e2_b][ii] - entity_tmp[e1_b][ii] - relation_vec[rel_b][ii]);
            if (x > 0)
                x = 1;
            else
                x = -1;
            relation_vec[rel_b][ii] -= x;
            entity_tmp[e1_b][ii] -= x;
            entity_tmp[e2_b][ii] += x;

            relation_vec[inverseRel][ii] -= x;
            entity_tmp[e2_b][ii] -= x;
            entity_tmp[e1_b][ii] += x;
        }
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

    void gradientInstanceOf(int e_a, int c_a, int e_b, int c_b){
        double dis = 0;
        for(int i = 0; i < n; ++i){
            dis += sqr(entity_vec[e_a][i] - concept_vec[c_a][i]);
        }
        if(dis > sqr(concept_r[c_a])){
            for(int j = 0; j < n; ++j){
                double x = 2 * (entity_vec[e_a][j] - concept_vec[c_a][j]);
                entity_tmp[e_a][j] -= x * rate;
                concept_tmp[c_a][j] -= -1 * x * rate;
            }
            concept_r_tmp[c_a] -= -2 * concept_r[c_a] * rate;
        }

        dis = 0;
        for(int i = 0; i < n; ++i){
            dis += sqr(entity_vec[e_b][i] - concept_vec[c_b][i]);
        }
        if(dis > sqr(concept_r[c_b])){
            for(int j = 0; j < n; ++j){
                double x = 2 * (entity_vec[e_b][j] - concept_vec[c_b][j]);
                entity_tmp[e_b][j] += x * rate;
                concept_tmp[c_b][j] += -1 * x * rate;
            }
            concept_r_tmp[c_b] += -2 * concept_r[c_b] * rate;
        }
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
void prepare(){
    map<string,int> rel2id;
    map<string,int> ent2id;
	map<string, int> class2id;

    FILE* f1 = fopen(("data/" + dataSet + "/Train/instance2id.txt").c_str(),"r");
    FILE* f2 = fopen(("data/" + dataSet + "/Train/relation2id.txt").c_str(),"r");
    FILE* f3 = fopen(("data/" + dataSet + "/Train/concept2id.txt").c_str(),"r");
    FILE* f_kb = fopen(("data/" + dataSet + "/Train/triple2id.txt").c_str(),"r");
    fscanf(f1, "%d", &entity_num);
    fscanf(f2, "%d", &relation_num);
    fscanf(f3, "%d", &concept_num);
    fscanf(f_kb, "%d", &triple_num);
    int h, t, l;
    while (fscanf(f_kb, "%d%d%d", &h, &t, &l) == 3) {
        train.addHrt(h, t, l);
        if(bern){
            left_entity[l][h]++;
            right_entity[l][t]++;
        }
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

	ifstream class2id_file("data/" + dataSet + "/Train/concept2id.txt");
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
	ifstream rel_file("data/" + dataSet + "/Train/relation2id.txt");
	getline(rel_file, tmp);
	while (getline(rel_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		string rel = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		rel2id.insert(pair<string, int>(rel, id));
	}
	rel_file.close();

	// trovo le relazioni inverseOf
	ifstream inverse_file("data/" + dataSet + "/Train/inverseOf.txt");
	while (getline(inverse_file, tmp))
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
				inverse.insert(pair<int, int>(id_first, id_second));
				inverse.insert(pair<int, int>(id_second, id_first));
			}
		}
	}
	inverse_file.close();

	// carico le entità per il confronto
	ifstream ent_file("data/" + dataSet + "/Train/instance2id.txt");
	getline(ent_file, tmp);
	while (getline(ent_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		string ent = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		ent2id.insert(pair<string, int>(ent, id));
	}
	ent_file.close();

	ifstream instanceOf_file("data/" + dataSet + "/Train/instanceOf2id.txt");
	string tmpStr;
	while (getline(instanceOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		string a1 = tmpStr.substr(0, pos);
		string b1 = tmpStr.substr(pos + 1);
		int a = ent2id.find(a1)->second;
		int b = ent2id.find(b1)->second;
		train.addInstanceOf(a, b);
		instance_concept[a].push_back(b);
		concept_instance[b].push_back(a);
	}
	instanceOf_file.close();

	ifstream subclassOf_file("data/" + dataSet + "/Train/subclassOf2id.txt");
	while (getline(subclassOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		string a1 = tmpStr.substr(0, pos);
		string b1 = tmpStr.substr(pos + 1);
		int a = class2id.find(a1)->second;
		int b = class2id.find(b1)->second;
		train.addSubClassOf(a, b);
		sub_up_concept[a].push_back(b);
		up_sub_concept[b].push_back(a);
	}
	subclassOf_file.close();


	ifstream eqProp_file("data/" + dataSet + "/Train/equivalentProperty.txt");
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
    prepare();
    train.setup(n, rate, margin, margin_ins, margin_sub);
    train.doTrain();
}