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

    inline int dimension = 100;         // dimensione dei vettori
    inline int bernFlag = 1;
    inline const string typeOf = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"; // relazione type (ovvero instanceOf)


    inline const bool debug = true;     // se true crea un solo thread (per permettere il debug)
    inline int threads = 12;            // numero di thread (funziona solo solo se debug = false)
    inline int trainTimes = 1000;       // trainTimes(epoch) - 1000
    inline int nbatches = 50;	        // batches - 50/100
    inline int epoch;                   // epoch attuale


    /* Costanti per iperparametri numerici */
    inline float alpha = 0.001;
    inline float margin = 1.0;
    inline float margin_instance = 0.4;
    inline float margin_subclass = 0.3;
    inline float ins_cut = 8.0;
    inline float sub_cut = 8.0;
    inline float RATE = 0.001;


    /* Costanti per salvataggio e caricamento */
    inline string note = "_OWL";            // estensione del file dei vettori (da non modificare)
    inline int loadBinaryFlag = 0;          // flag che indica se caricare i dati da file binari
    inline string loadPath = "";            // percorso dove caricare i vettori addestrati
    inline string inPath = "Train/";        // percorso dove prelevare il training set
    inline int outBinaryFlag = 0;           // flag che indica se salvare i dati in file binari
    inline string outPath = "Output/";      // percorso dove salvare i vettori addestrati


    inline INT *lefHead, *rigHead;
    inline INT *lefTail, *rigTail;


    /** Vettori da addestrare(output) */
    inline INT relationTotal, entityTotal, conceptTotal;    // numero di relazioni, entità e classi
    inline REAL *relationVec;                               // vettori delle relazioni
    inline REAL *entityVec;                                 // vettori delle delle entità
    inline vector<vector<double> > conceptVec;              // vettori delle classi
    inline vector<double> concept_r;                        // vettore di supporto per le classi

    /** Vettori di supporto per l'addestramento */
    inline map<pair<int, int>, map<int, int>> ok;
    inline map<pair<int, int>, int> subClassOf_ok;
    inline map<pair<int, int>, int> instanceOf_ok;
    inline vector<pair<int, int>> subClassOf;
    inline vector<pair<int, int>> instanceOf;
    inline vector<vector<int>> concept_instance;
    inline vector<vector<int>> instance_concept;
    inline vector<vector<int>> instance_brother;
    inline vector<vector<int>> sub_up_concept;
    inline vector<vector<int>> up_sub_concept;
    inline vector<vector<int>> concept_brother;

    // OWL Variable
    inline multimap<INT, INT> ent2class;	// mappa una entità nella classe di appartenzenza (se disponibile) id dbpedia
    inline multimap<INT, INT> ent2cls;		// mappa una entità alla classe di appartenenza (per range e domain)
    inline multimap<INT, INT> rel2range;	// mappa una relazione nel range, che corrisponde ad una classe
    inline multimap<INT, INT> rel2domain;	// mappa una relazione nel domain, che corrisponde ad una classe
    inline multimap<INT, INT> cls2false_t;  // data una entità, restituisce una classe falsa (per tail corruption)
    inline list<int> functionalRel;
    inline map<int, int> inverse;
    inline map<int, int> equivalentRel;
    // inline map<int,int> disjointWith;
    inline int typeOf_id;
    inline INT trainSize, tripleTotal;


    inline INT *freqRel, *freqEnt;
    inline REAL *left_mean, *right_mean;


    /** Struct che mappa una tripla */
    struct Triple
    {
        INT h, r, t;
    };

    // Triple
    inline Triple *trainHead, *trainTail, *trainList;

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