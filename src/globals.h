#ifndef GLOBALS_H
#define GLOBALS_H

#include <string>

using namespace std;

static const bool debug = true; // se true crea un solo thread per addestrare il modello (per permettere il debug)

static int threads = 12;
static int bernFlag = 1;
static int loadBinaryFlag = 0;
static int outBinaryFlag = 0;
static int trainTimes = 1000; // epoch - 1000
static int nbatches = 50;	   // batches - 50/100
static int dimension = 100;
static int epoch;
static float alpha = 0.001;
static float margin = 1.0;
static float margin_instance = 0.4;
static float margin_subclass = 0.3;
static float ins_cut = 8.0;
static float sub_cut = 8.0;
static float RATE = 0.001;

static int failed = 0;
static int tot_batches = 0;
static int wrong = 0;

static string inPath = "Train/";
static string outPath = "Output/";
static string loadPath = "";
static string note = "_OWL";

static const string typeOf = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";

#endif