#include <iostream>
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <pthread.h>

using namespace std;

long relationTotal;
long entityTotal;
long conceptTotal;
long threads = 8;
long dimension = 100;
long testTotal, tripleTotal, trainTotal, validTotal;
float *entityVec, *relationVec, *matrix;
vector<vector<double> > concept_vec;
vector<double> concept_r;
bool OWL = true;
string dataSet = "Nell";

struct Triple {
    long h, r, t;
};

struct Mix{
    int number;
    int index;
};

struct cmpHead {
    bool operator()(const Triple &a, const Triple &b) {
        return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

bool cmp(Mix a, Mix b){
    return a.number > b.number;
}

Triple *testList, *tripleList;

string note = "";

void init() {
    FILE *fin;
    long tmp, h, r, t;

    if(OWL) {
        note = "_OWL";
    }

    fin = fopen(("../data/" + dataSet + "/Train/relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);
    relationVec = (float *)calloc(relationTotal * dimension, sizeof(float));
    
    fin = fopen(("../data/" + dataSet + "/Train/instance2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    fin = fopen(("../data/" + dataSet + "/Train/class2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &conceptTotal);
    fclose(fin);

    entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    matrix = (float *)calloc(relationTotal * dimension * dimension, sizeof(float));


    FILE* f_kb1 = fopen(("../data/" + dataSet + "/Test/instanceOf2id.txt").c_str(),"r");
    FILE* f_kb2 = fopen(("../data/" + dataSet + "/Train/instanceOf2id.txt").c_str(),"r");
    FILE* f_kb3 = fopen(("../data/" + dataSet + "/Valid/instanceOf2id.txt").c_str(),"r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));

    for (long i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &h);
        tmp = fscanf(f_kb1, "%ld", &t);
        testList[i].h = h;
        testList[i].t = t;
        tripleList[i].h = h;
        tripleList[i].t = t;
    }

    for (long i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &h);
        tmp = fscanf(f_kb2, "%ld", &t);
        tripleList[i + testTotal].h = h;
        tripleList[i + testTotal].t = t;
    }

    for (long i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &h);
        tmp = fscanf(f_kb3, "%ld", &t);
        tripleList[i + testTotal + trainTotal].h = h;
        tripleList[i + testTotal + trainTotal].t = t;
    }
    
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    sort(tripleList, tripleList + tripleTotal, cmpHead());
}

void prepare() {
    FILE *fin;
    long tmp;
    fin = fopen(("../data/" + dataSet + "/Output/entity2vec" + note + "_1900.vec").c_str(), "r");
    for (long i = 0; i < entityTotal; i++) {
        long last = i * dimension;
        for (long j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &entityVec[last + j]);
    }
    fclose(fin);
    fin = fopen(("../data/" + dataSet + "/Output/relation2vec" + note + "_1900.vec").c_str(), "r");
    for (long i = 0; i < relationTotal; i++) {
        long last = i * dimension;
        for (long j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &relationVec[last + j]);
    }
    fclose(fin);

    concept_vec.resize(conceptTotal);
    concept_r.resize(conceptTotal);
    fin = fopen(("../data/" + dataSet + "/Output/concept2vec" + note + "_1900.vec").c_str(), "r");
    for (long i = 0; i < conceptTotal; i++) {
        concept_vec[i].resize(dimension);
        for(int j = 0; j < dimension; ++j){
            fscanf(fin, "%lf", &concept_vec[i][j]);
        }
        fscanf(fin, "%lf", &concept_r[i]);
    }
    fclose(fin);
}

double sqr(double x){
    return x * x;
}
double calcSum(int e, int c){
    double dis = 0;
    for (long i = 0; i < dimension; i++) {
        double first = entityVec[e * dimension + i];
        double second = concept_vec[c][i];
        dis += sqr(first - second);
    }
    if(dis < sqr(concept_r[c])){
        return 0;
    }
    return dis - sqr(concept_r[c]);
}

bool find(long h, long t) {
    long lef = 0;
    long rig = tripleTotal - 1;
    long mid;
    while (lef + 1 < rig) {
        long mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h ) || (tripleList[mid]. h == h && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].t == t) return true;
    return false;
}

float *l_filter_tot, *r_filter_tot, *l_filter_tot1, *r_filter_tot1, *l_filter_tot3, *r_filter_tot3, *l_filter_tot5, *r_filter_tot5;
float *l_tot, *r_tot, *l_tot1, *r_tot1, *l_tot3, *r_tot3, *l_tot5, *r_tot5;
float *l_filter_rank, *r_filter_rank, *l_rank, *r_rank;
float *l_filter_rank_dao, *r_filter_rank_dao, *l_rank_dao, *r_rank_dao;
float lft[10][1500], rft[10][1500];

void* testMode(void *con) {
    long id;
    id = (unsigned long long)(con);
    long lef = testTotal / (threads) * id;
    long rig = testTotal / (threads) * (id + 1) - 1;
    if (id == threads - 1) rig = testTotal - 1;
    for (long i = lef; i <= rig; i++) {
        long h = testList[i].h;
        long t = testList[i].t;
        float minimal = calcSum(h, t);
        long l_filter_s = 0;
        long l_s = 0;
        long r_filter_s = 0;
        long r_s = 0;
        for (long j = 0; j < entityTotal; j++) {
            if (j != h) {
                float value = calcSum(j, t);
                if (value < minimal) {
                    l_s += 1;
                    if (!find(j, t))
                        l_filter_s += 1;
                }
            }
        }
        for (long j = 0; j < conceptTotal; j++) {
            if (j != t) {
                float value = calcSum(h, j);
                if (value < minimal) {
                    r_s += 1;
                    if (!find(h, j))
                        r_filter_s += 1;
                }
            }
        }

        if (l_filter_s < 10){
            l_filter_tot[id] += 1;
        }
        if (l_filter_s < 1) l_filter_tot1[id] += 1;
        if (l_filter_s < 3) l_filter_tot3[id] += 1;
        if (l_filter_s < 5) l_filter_tot5[id] += 1;

        if (l_s < 10) l_tot[id] += 1;
        if (l_s < 1) l_tot1[id] += 1;
        if (l_s < 3) l_tot3[id] += 1;
        if (l_s < 5) l_tot5[id] += 1;

        if (r_filter_s < 10){
            r_filter_tot[id] += 1;
        }
        if (r_filter_s < 1) r_filter_tot1[id] += 1;
        if (r_filter_s < 3) r_filter_tot3[id] += 1;
        if (r_filter_s < 5) r_filter_tot5[id] += 1;

        if (r_s < 10) r_tot[id] += 1;
        if (r_s < 1) r_tot1[id] += 1;
        if (r_s < 3) r_tot3[id] += 1;
        if (r_s < 5) r_tot5[id] += 1;

        l_filter_rank_dao[id] += 1.0 / (l_filter_s + 1);
        r_filter_rank_dao[id] += 1.0 / (r_filter_s + 1);
        l_rank_dao[id] += 1.0 / (l_s + 1);
        r_rank_dao[id] += 1.0 / (r_s + 1);

        l_filter_rank[id] += l_filter_s;
        r_filter_rank[id] += r_filter_s;
        l_rank[id] += l_s;
        r_rank[id] += r_s;
    }
}

void* test(void *con) {
    l_filter_tot = (float *)calloc(threads, sizeof(float));
    r_filter_tot = (float *)calloc(threads, sizeof(float));
    l_filter_tot1 = (float *)calloc(threads, sizeof(float));
    r_filter_tot1 = (float *)calloc(threads, sizeof(float));
    l_filter_tot3 = (float *)calloc(threads, sizeof(float));
    r_filter_tot3 = (float *)calloc(threads, sizeof(float));
    l_filter_tot5 = (float *)calloc(threads, sizeof(float));
    r_filter_tot5 = (float *)calloc(threads, sizeof(float));

    l_tot = (float *)calloc(threads, sizeof(float));
    r_tot = (float *)calloc(threads, sizeof(float));
    l_tot1 = (float *)calloc(threads, sizeof(float));
    r_tot1 = (float *)calloc(threads, sizeof(float));
    l_tot3 = (float *)calloc(threads, sizeof(float));
    r_tot3 = (float *)calloc(threads, sizeof(float));
    l_tot5 = (float *)calloc(threads, sizeof(float));
    r_tot5 = (float *)calloc(threads, sizeof(float));

    l_filter_rank = (float *)calloc(threads, sizeof(float));
    r_filter_rank = (float *)calloc(threads, sizeof(float));
    l_filter_rank_dao = (float *)calloc(threads, sizeof(float));
    r_filter_rank_dao = (float *)calloc(threads, sizeof(float));
    
    l_rank = (float *)calloc(threads, sizeof(float));
    r_rank = (float *)calloc(threads, sizeof(float));
    l_rank_dao = (float *)calloc(threads, sizeof(float));
    r_rank_dao = (float *)calloc(threads, sizeof(float));

    pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
    for (long a = 0; a < threads; a++)
        pthread_create(&pt[a], NULL, testMode,  (void*)a);
    for (long a = 0; a < threads; a++)
        pthread_join(pt[a], NULL);
    free(pt);
    for (long a = 1; a < threads; a++) {
        l_filter_tot[a] += l_filter_tot[a - 1];
        r_filter_tot[a] += r_filter_tot[a - 1];
        l_filter_tot1[a] += l_filter_tot1[a - 1];
        r_filter_tot1[a] += r_filter_tot1[a - 1];
        l_filter_tot3[a] += l_filter_tot3[a - 1];
        r_filter_tot3[a] += r_filter_tot3[a - 1];
        l_filter_tot5[a] += l_filter_tot5[a - 1];
        r_filter_tot5[a] += r_filter_tot5[a - 1];

        l_tot[a] += l_tot[a - 1];
        r_tot[a] += r_tot[a - 1];
        l_tot1[a] += l_tot1[a - 1];
        r_tot1[a] += r_tot1[a - 1];
        l_tot3[a] += l_tot3[a - 1];
        r_tot3[a] += r_tot3[a - 1];
        l_tot5[a] += l_tot5[a - 1];
        r_tot5[a] += r_tot5[a - 1];

        l_filter_rank[a] += l_filter_rank[a - 1];
        r_filter_rank[a] += r_filter_rank[a - 1];
        l_rank[a] += l_rank[a - 1];
        r_rank[a] += r_rank[a - 1];

        l_filter_rank_dao[a] += l_filter_rank_dao[a - 1];
        r_filter_rank_dao[a] += r_filter_rank_dao[a - 1];
        l_rank_dao[a] += l_rank_dao[a - 1];
        r_rank_dao[a] += r_rank_dao[a - 1];
    }

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@5  \t hit@3  \t hit@1 \n");
    printf("averaged(raw):\t\t %.3f \t %.1f \t %.3f \t %.3f \t %.3f \t %.3f \n",
            (l_rank_dao[threads - 1] / testTotal + r_rank_dao[threads - 1] / testTotal)/2, 
            (l_rank[threads - 1] / testTotal + r_rank[threads - 1] / testTotal)/2, 
            (l_tot[threads - 1] / testTotal + r_tot[threads - 1] / testTotal)/2,
            (l_tot5[threads - 1] / testTotal + r_tot5[threads - 1] / testTotal)/2, 
            (l_tot3[threads - 1] / testTotal + r_tot3[threads - 1] / testTotal)/2,
            (l_tot1[threads - 1] / testTotal + r_tot1[threads - 1] / testTotal)/2);
    printf("averaged(filter):\t %.3f \t %.1f \t %.3f \t %.3f \t %.3f \t %.3f \n",
            (l_filter_rank_dao[threads - 1] / testTotal + r_filter_rank_dao[threads - 1] / testTotal)/2, 
            (l_filter_rank[threads - 1] / testTotal + r_filter_rank[threads - 1] / testTotal)/2, 
            (l_filter_tot[threads - 1] / testTotal + r_filter_tot[threads - 1] / testTotal)/2,
            (l_filter_tot5[threads - 1] / testTotal + r_filter_tot5[threads - 1] / testTotal)/2,
            (l_filter_tot3[threads - 1] / testTotal + r_filter_tot3[threads - 1] / testTotal)/2,
            (l_filter_tot1[threads - 1] / testTotal + r_filter_tot1[threads - 1] / testTotal)/2);
}

long ArgPos(char *str, long argc, char **argv) {
  long a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}


int main(int argc, char**argv) {
    int i;
	if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) dataSet = argv[i + 1];
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) threads = atoi(argv[i + 1]);
    cout << "dimension = " << dimension << endl;
    cout << "data = " << dataSet << endl;
    cout << "threads = " << threads << endl;
    init();
    prepare();
    test(NULL);
    return 0;
}
