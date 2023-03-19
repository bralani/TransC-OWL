#ifndef GLOBALS_H
#define GLOBALS_H

    #define REAL float
    #define INT int

    #include <iostream>
    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <string>
    #include <map>
    #include <list>
    #include <set>
    #include <vector>

    using namespace std;

    static int dimension = 100;         // dimensione dei vettori
    static int bernFlag = 1;


    static const bool debug = true;     // se true crea un solo thread (per permettere il debug)
    static int threads = 12;            // numero di thread (funziona solo solo se debug = false)
    static int trainTimes = 1000;       // trainTimes(epoch) - 1000
    static int nbatches = 50;	        // batches - 50/100
    static int epoch;                   // epoch attuale


    /* Costanti per iperparametri numerici */
    static float alpha = 0.001;
    static float margin = 1.0;
    static float margin_instance = 0.4;
    static float margin_subclass = 0.3;
    static float ins_cut = 8.0;
    static float sub_cut = 8.0;
    static float RATE = 0.001;


    /* Costanti per salvataggio e caricamento */
    static string note = "_OWL";       // estensione del file dei vettori (da non modificare)

    static int loadBinaryFlag = 0;    // flag che indica se caricare i dati da file binari
    static string loadPath = "";            // percorso dove caricare i vettori addestrati
    static string inPath = "Train/";        // percorso dove prelevare il training set

    static int outBinaryFlag = 0;     // flag che indica se salvare i dati in file binari
    static string outPath = "Output/";      // percorso dove salvare i vettori addestrati


    // relazione type (ovvero instanceOf)
    static const string typeOf = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";


    static INT *lefHead, *rigHead;
    static INT *lefTail, *rigTail;

    // OWL Variable
    static multimap<INT, INT> ent2class;	// mappa una entità nella classe di appartenzenza (se disponibile) id dbpedia
    static multimap<INT, INT> ent2cls;		// mappa una entità alla classe di appartenenza (per range e domain)
    static multimap<INT, INT> rel2range;	// mappa una relazione nel range, che corrisponde ad una classe
    static multimap<INT, INT> rel2domain;	// mappa una relazione nel domain, che corrisponde ad una classe
    static multimap<INT, INT> cls2false_t; // data una entità, restituisce una classe falsa (per tail corruption)
    static vector<vector<int>> concept_instance;
    static vector<vector<int>> instance_concept;
    static vector<vector<int>> instance_brother;
    static vector<vector<int>> sub_up_concept;
    static vector<vector<int>> up_sub_concept;
    static vector<vector<int>> concept_brother;
    static list<int> functionalRel;
    static map<int, int> inverse;
    static map<int, int> equivalentRel;
    // static map<int,int> disjointWith;
    static int typeOf_id;
    static INT trainSize, tripleTotal;


    static map<pair<int, int>, map<int, int>> ok;
    static map<pair<int, int>, int> subClassOf_ok;
    static map<pair<int, int>, int> instanceOf_ok;
    static vector<pair<int, int>> subClassOf;
    static vector<pair<int, int>> instanceOf;

    static INT relationTotal, entityTotal, conceptTotal;
    static REAL *relationVec, *entityVec;
    static vector<vector<double> > conceptVec;
    static REAL *relationVecDao, *entityVecDao;
    static INT *freqRel, *freqEnt;
    static REAL *left_mean, *right_mean;


    struct Triple
    {
        INT h, r, t;
    };
    static Triple *trainHead, *trainTail, *trainList;

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

#endif