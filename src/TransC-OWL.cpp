
#define REAL float
#define INT int

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
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

#include "math.h"
#include "globals.h"

using namespace std;

void out();


INT *lefHead, *rigHead;
INT *lefTail, *rigTail;

// OWL Variable
multimap<INT, INT> ent2class;	// mappa una entità nella classe di appartenzenza (se disponibile) id dbpedia
multimap<INT, INT> ent2cls;		// mappa una entità alla classe di appartenenza (per range e domain)
multimap<INT, INT> rel2range;	// mappa una relazione nel range, che corrisponde ad una classe
multimap<INT, INT> rel2domain;	// mappa una relazione nel domain, che corrisponde ad una classe
multimap<INT, INT> cls2false_t; // data una entità, restituisce una classe falsa (per tail corruption)
vector<vector<int>> concept_instance;
vector<vector<int>> instance_concept;
vector<vector<int>> instance_brother;
vector<vector<int>> sub_up_concept;
vector<vector<int>> up_sub_concept;
vector<vector<int>> concept_brother;
list<int> functionalRel;
map<int, int> inverse;
map<int, int> equivalentRel;
// map<int,int> disjointWith;
int typeOf_id;
INT trainSize, tripleTotal;

struct Triple
{
	INT h, r, t;
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

// vero se l'entità index è di classe class_id
bool inRange(int id_rel, int id_obj)
{
	pair<std::multimap<int, int>::iterator, multimap<int, int>::iterator> ret;
	// prendi le classi di id_obj!!
	ret = ent2cls.equal_range(id_obj);
	// prendi le classi di id_obj!!
	set<INT> cls;
	for (multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
	{
		cls.insert(it->second);
	}
	ret = rel2range.equal_range(id_rel);
	for (multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
		if (cls.find(it->second) != cls.end())
			return true;

	return false;
}

bool inDomain(int id_rel, int id_sub)
{
	pair<std::multimap<int, int>::iterator, multimap<int, int>::iterator> ret;
	// prendi le classi di id_obj!!
	ret = ent2cls.equal_range(id_sub);
	// prendi le classi di id_obj!!
	set<INT> cls;
	for (multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
	{
		cls.insert(it->second);
	}
	ret = rel2domain.equal_range(id_rel);
	for (multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
		if (cls.find(it->second) != cls.end())
			return true;

	return false;
}

map<pair<int, int>, map<int, int>> ok;
map<pair<int, int>, int> subClassOf_ok;
map<pair<int, int>, int> instanceOf_ok;
vector<pair<int, int>> subClassOf;
vector<pair<int, int>> instanceOf;

void addSubClassOf(int sub, int parent)
{
	subClassOf.emplace_back(sub, parent);
	subClassOf_ok[make_pair(sub, parent)] = 1;
}

void addInstanceOf(int instance, int concept)
{
	instanceOf.emplace_back(instance, concept);
	instanceOf_ok[make_pair(instance, concept)] = 1;
}

// vero se l'entità index è di classe class_id
bool containClass(int class_id, int index)
{
	pair<std::multimap<int, int>::iterator, multimap<int, int>::iterator> ret;
	ret = ent2class.equal_range(index);
	for (multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
		if (it->second == class_id)
			return true;
	return false;
}

// vero se la relazione è di tipo functional
bool isFunctional(int rel_id)
{
	if (functionalRel.size() == 0)
		return false;
	for (list<int>::iterator it = functionalRel.begin(); it != functionalRel.end(); ++it)
		if ((*it) == rel_id)
			return true;
	return false;
}

// vero se la relazione è di tipo SubclassOf
bool isSubclassOf(int rel_id)
{
	return rel_id == -1;
}

// vero se la relazione è di tipo InstanceOf
bool isInstanceOf(int rel_id)
{
	return rel_id == typeOf_id;
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

// read owl information from files prepared using DBPedia OWL (see dbpediaUtils)
void owlInit()
{
	string tmp;
	map<string, int> rel2id;
	map<string, int> ent2id;
	map<string, int> class2id;

	printf("Loading ontology information...\n");

	// carico le relazioni per il confronto
	ifstream rel_file(inPath + "relation2id.txt");
	getline(rel_file, tmp);
	while (getline(rel_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		string rel = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		rel2id.insert(pair<string, int>(rel, id));
		if (rel == typeOf)
			typeOf_id = id;
	}
	printf("ID TypeOf: %d\n", typeOf_id);
	rel_file.close();

	// carico le entità per il confronto
	ifstream ent_file(inPath + "entity2id.txt");
	getline(ent_file, tmp);
	while (getline(ent_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		string ent = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		ent2id.insert(pair<string, int>(ent, id));
	}
	ent_file.close();

	// carico le classi per il confronto
	ifstream class2id_file(inPath + "class2id.txt");
	getline(class2id_file, tmp);
	while (getline(class2id_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		string classe = tmp.substr(0, pos);
		int id = atoi(tmp.substr(pos + 1).c_str());
		class2id.insert(pair<string, int>(classe, id));
	}
	class2id_file.close();

	ifstream class_file(inPath + "entity2class.txt");
	while (getline(class_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		int entity = atoi(tmp.substr(0, pos).c_str());
		int class_id = atoi(tmp.substr(pos + 1).c_str());
		ent2class.insert(pair<int, int>(entity, class_id));
	}
	class_file.close();

	ifstream instanceOf_file(inPath + "instanceof.txt");
	string tmpStr;
	while (getline(instanceOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		string a1 = tmpStr.substr(0, pos);
		string b1 = tmpStr.substr(pos + 1);
		int a = ent2id.find(a1)->second;
		int b = ent2id.find(b1)->second;
		addInstanceOf(a, b);
		instance_concept[a].push_back(b);
		concept_instance[b].push_back(a);
	}
	instanceOf_file.close();

	ifstream subclassOf_file(inPath + "subclassof.txt");
	while (getline(subclassOf_file, tmpStr))
	{
		int pos = tmpStr.find(' ', 0);
		string a1 = tmpStr.substr(0, pos);
		string b1 = tmpStr.substr(pos + 1);
		int a = class2id.find(a1)->second;
		int b = class2id.find(b1)->second;
		addSubClassOf(a, b);
		sub_up_concept[a].push_back(b);
		up_sub_concept[b].push_back(a);
	}
	subclassOf_file.close();

	trainSize = tripleTotal + instanceOf.size() + subClassOf.size();

	ifstream domain_file(inPath + "rs_domain2id.txt");
	while (getline(domain_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		int relation = atoi(tmp.substr(0, pos).c_str());
		int domain = atoi(tmp.substr(pos + 1).c_str());
		rel2domain.insert(pair<int, int>(relation, domain));
	}
	domain_file.close();

	ifstream range_file(inPath + "rs_range2id.txt");
	while (getline(range_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		int relation = atoi(tmp.substr(0, pos).c_str());
		int range = atoi(tmp.substr(pos + 1).c_str());
		rel2range.insert(pair<int, int>(relation, range));
	}
	range_file.close();

	ifstream false_file(inPath + "falseTypeOf2id.txt");
	while (getline(false_file, tmp))
	{
		string::size_type pos = tmp.find(' ', 0);
		string last_part = tmp.substr(pos + 1);
		int head = atoi(tmp.substr(0, pos).c_str());
		pos = last_part.find(' ', 0);
		int cls = atoi(last_part.substr(0, pos).c_str());
		cls2false_t.insert(pair<int, int>(head, cls));
		// cls2false_h.insert(pair<int,int>(cls,head));
	}

	// trovo le relazioni di tipo functional
	ifstream function_file(inPath + "functionalProperty.txt");
	while (getline(function_file, tmp))
	{
		if (rel2id.find(tmp) != rel2id.end())
		{
			functionalRel.push_front(rel2id.find(tmp)->second);
		}
	}
	function_file.close();

	// trovo le relazioni inverseOf
	ifstream inverse_file(inPath + "inverseOf.txt");
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

	ifstream eqProp_file(inPath + "equivalentProperty.txt");
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

	multimap<int, int> dbp_disj;
	ifstream disj_file(inPath + "disjoint2id.txt");
	while (getline(disj_file, tmp))
	{
		string::size_type pos = tmp.find('\t', 0);
		int cls = atoi(tmp.substr(0, pos).c_str());
		int disjoint = atoi(tmp.substr(pos + 1).c_str());
		dbp_disj.insert(pair<int, int>(cls, disjoint));
	}
	disj_file.close();
}

/*
	Read triples from the training file.
*/

INT relationTotal, entityTotal, conceptTotal;
REAL *relationVec, *entityVec;
vector<vector<double> > conceptVec;
REAL *relationVecDao, *entityVecDao;
INT *freqRel, *freqEnt;
REAL *left_mean, *right_mean;

void init()
{

	FILE *fin;
	INT tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	relationVec = (REAL *)calloc(relationTotal * dimension, sizeof(REAL));
	for (INT i = 0; i < relationTotal; i++)
	{
		for (INT ii = 0; ii < dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	entityVec = (REAL *)calloc(entityTotal * dimension, sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++)
	{
		for (INT ii = 0; ii < dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec + i * dimension);
	}

	fin = fopen((inPath + "class2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &conceptTotal);
	fclose(fin);


	conceptVec.resize(conceptTotal);
	for(auto &i : conceptVec)
		i.resize(dimension);
	for(int i = 0; i < conceptTotal; ++i){
		for(int j = 0; j < dimension; ++j){
			conceptVec[i][j] = randn(0,1.0/dimension,-6/sqrt(dimension),6/sqrt(dimension));
		}
		normV(conceptVec[i]);
	}

	// ! CONTROLLARE SE BISOGNA FARE RELATIONTOTAL + ENTITYTOTAL + CONCEPTTOTAL
	freqRel = (INT *)calloc(relationTotal + entityTotal, sizeof(INT));
	freqEnt = freqRel + relationTotal;

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	for (INT i = 0; i < tripleTotal; i++)
	{
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);
		if (trainList[i].r == typeOf_id)
		{
			ent2cls.insert(pair<int, int>(trainList[i].h, trainList[i].t));
		}
		freqEnt[trainList[i].t]++;
		freqEnt[trainList[i].h]++;
		freqRel[trainList[i].r]++;
		trainHead[i] = trainList[i];
		trainTail[i] = trainList[i];
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	memset(rigHead, -1, sizeof(INT) * entityTotal);
	memset(rigTail, -1, sizeof(INT) * entityTotal);
	for (INT i = 1; i < tripleTotal; i++)
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
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (REAL *)calloc(relationTotal * 2, sizeof(REAL));
	right_mean = left_mean + relationTotal;
	for (INT i = 0; i < entityTotal; i++)
	{
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}

	for (INT i = 0; i < relationTotal; i++)
	{
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}

	relationVecDao = (REAL *)calloc(dimension * relationTotal, sizeof(REAL));
	entityVecDao = (REAL *)calloc(dimension * entityTotal, sizeof(REAL));

	concept_instance.resize(conceptTotal);
	instance_concept.resize(entityTotal);
	sub_up_concept.resize(conceptTotal);
	up_sub_concept.resize(conceptTotal);
	instance_brother.resize(entityTotal);
	concept_brother.resize(conceptTotal);

	for (int i = 0; i < instance_concept.size(); ++i)
	{
		for (int j = 0; j < instance_concept[i].size(); ++j)
		{
			for (int k = 0; k < concept_instance[instance_concept[i][j]].size(); ++k)
			{
				if (concept_instance[instance_concept[i][j]][k] != i)
					instance_brother[i].push_back(concept_instance[instance_concept[i][j]][k]);
			}
		}
	}

	for (int i = 0; i < sub_up_concept.size(); ++i)
	{
		for (int j = 0; j < sub_up_concept[i].size(); ++j)
		{
			for (int k = 0; k < up_sub_concept[sub_up_concept[i][j]].size(); ++k)
			{
				if (up_sub_concept[sub_up_concept[i][j]][k] != i)
				{
					concept_brother[i].push_back(up_sub_concept[sub_up_concept[i][j]][k]);
				}
			}
		}
	}
}

void load_binary()
{
	struct stat statbuf1;
	if (stat((loadPath + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1)
	{
		INT fd = open((loadPath + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
		REAL *entityVecTmp = (REAL *)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
		memcpy(entityVec, entityVecTmp, statbuf1.st_size);
		munmap(entityVecTmp, statbuf1.st_size);
		close(fd);
	}
	struct stat statbuf2;
	if (stat((loadPath + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1)
	{
		INT fd = open((loadPath + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
		REAL *relationVecTmp = (REAL *)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
		memcpy(relationVec, relationVecTmp, statbuf2.st_size);
		munmap(relationVecTmp, statbuf2.st_size);
		close(fd);
	}
}

void load()
{
	if (loadBinaryFlag)
	{
		load_binary();
		return;
	}
	FILE *fin;
	INT tmp;
	fin = fopen((loadPath + "entity2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < entityTotal; i++)
	{
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &entityVec[last + j]);
	}
	fclose(fin);
	fin = fopen((loadPath + "relation2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < relationTotal; i++)
	{
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &relationVec[last + j]);
	}
	fclose(fin);
	/*fin = fopen((loadPath + "concept2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < conceptTotal; i++) {
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &conceptVec[i][j]);
	}
	fclose(fin);*/
}

/*
	Training process of transE.
*/

INT Len;
INT Batch;
REAL res;

INT corrupt_head(INT id, INT h, INT r)
{
	INT lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig)
	{
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r)
			rig = mid;
		else
			lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig)
	{
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r)
			lef = mid;
		else
			rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t)
		return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1)
		return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig)
	{
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r)
{
	INT lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig)
	{
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r)
			rig = mid;
		else
			lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig)
	{
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r)
			lef = mid;
		else
			rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h)
		return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1)
		return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig)
	{
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

REAL calc_sum(INT e1, INT e2, INT rel)
{
	REAL sum = 0;
	INT last1 = e1 * dimension;
	INT last2 = e2 * dimension;
	INT lastr = rel * dimension;
	for (INT ii = 0; ii < dimension; ii++)
		sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
	return sum;
}

void gradient(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b)
{
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimension;
	INT lastb1 = e1_b * dimension;
	INT lastb2 = e2_b * dimension;
	INT lastbr = rel_b * dimension;
	// se entità equivalenti, salva array
	for (INT ii = 0; ii < dimension; ii++)
	{
		REAL x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		if (x > 0)
			x = -alpha;
		else
			x = alpha;
		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;
		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relationVec[lastbr + ii] -= x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;
	}
}


/*
void trainInstanceOf(int head, int tail, int cut, int id)
{
	
	int j = 0;
	if (rand() % 2 == 0)
	{
		do
		{
			if (!instance_brother[head].empty())
			{
				if (rand() % 10 < cut)
				{
					j = rand_max(entityTotal, id);
				}
				else
				{
					j = rand() % (int)instance_brother[head].size();
					j = instance_brother[head][j];
				}
			}
			else
			{
				j = rand_max(entityTotal, id);
			}
		} while (instanceOf_ok.count(make_pair(j, tail)) > 0);
		doTrainInstanceOf(head, tail, j, tail);
		normV(entity_tmp[j]);
	}
	else
	{
		do
		{
			if (!concept_brother[tail].empty())
			{
				if (rand() % 10 < cut)
				{
					j = rand_max(conceptTotal, id);
				}
				else
				{
					j = rand() % (int)concept_brother[tail].size();
					j = concept_brother[tail][j];
				}
			}
			else
			{
				j = rand_max(conceptTotal, id);
			}
		} while (instanceOf_ok.count(make_pair(head, j)) > 0);
		doTrainInstanceOf(head, tail, head, j);
		normV(concept_tmp[j]);
		normR(concept_r_tmp[j]);
	}
	normV(entity_tmp[head]);
	normV(concept_tmp[tail]);
	normR(concept_r_tmp[tail]);
}

void trainSubClassOf(int head, int tail, int cut, int id)
{
	int j = 0;
	if (rand() % 2 == 0)
	{
		do
		{
			if (!concept_brother[head].empty())
			{
				if (rand() % 10 < cut)
				{
					j = rand_max(conceptTotal, id);
				}
				else
				{
					j = rand() % (int)concept_brother[head].size();
					j = concept_brother[head][j];
				}
			}
			else
			{
				j = rand_max(conceptTotal, id);
			}
		} while (subClassOf_ok.count(make_pair(j, tail)) > 0);
		doTrainSubClassOf(head, tail, j, tail);
	}
	else
	{
		do
		{
			if (!concept_brother[tail].empty())
			{
				if (rand() % 10 < cut)
				{
					j = rand_max(conceptTotal, id);
				}
				else
				{
					j = rand() % (int)concept_brother[tail].size();
					j = concept_brother[tail][j];
				}
			}
			else
			{
				j = rand_max(conceptTotal, id);
			}
		} while (subClassOf_ok.count(make_pair(head, j)) > 0);
		doTrainSubClassOf(head, tail, head, j);
	}
	normV(concept_tmp[head]);
	normV(concept_tmp[tail]);
	normV(concept_tmp[j]);
	normR(concept_r_tmp[head]);
	normR(concept_r_tmp[tail]);
	normR(concept_r_tmp[j]);
}
void doTrainInstanceOf(int e_a, int c_a, int e_b, int c_b)
{
	double sum1 = calcSumInstanceOf(e_a, c_a);
	double sum2 = calcSumInstanceOf(e_b, c_b);
	if (sum1 + margin_instance > sum2)
	{
		res += (margin_instance + sum1 - sum2);
		gradientInstanceOf(e_a, c_a, e_b, c_b);
	}
}

void doTrainSubClassOf(int c1_a, int c2_a, int c1_b, int c2_b)
{
	double sum1 = calcSumSubClassOf(c1_a, c2_a);
	double sum2 = calcSumSubClassOf(c1_b, c2_b);
	if (sum1 + margin_subclass > sum2)
	{
		res += (margin_subclass + sum1 - sum2);
		gradientSubClassOf(c1_a, c2_a, c1_b, c2_b);
	}
}


double calcSumInstanceOf(int e, int c){
	double dis = 0;
	for(int i = 0; i < dimension; ++i){
		dis += sqr(entity_vec[e][i] - conceptVec[c][i]);
	}
	if(dis < sqr(concept_r[c])){
		return 0;
	}
	return dis - sqr(concept_r[c]);

}

double calcSumSubClassOf(int c1, int c2){
	double dis = 0;
	for(int i = 0; i < dimension; ++i){
		dis += sqr(conceptVec[c1][i] - conceptVec[c2][i]);
	}
	if(sqrt(dis) < fabs(concept_r[c1] - concept_r[c2])){
		return 0;
	}
	return dis - sqr(concept_r[c2]) + sqr(concept_r[c1]);

}

void gradientInstanceOf(int e_a, int c_a, int e_b, int c_b)
{
	double dis = 0;
	for (int i = 0; i < dimension; ++i)
	{
		dis += sqr(entity_vec[e_a][i] - conceptVec[c_a][i]);
	}
	if (dis > sqr(concept_r[c_a]))
	{
		for (int j = 0; j < dimension; ++j)
		{
			double x = 2 * (entity_vec[e_a][j] - conceptVec[c_a][j]);
			entity_tmp[e_a][j] -= x * RATE;
			concept_tmp[c_a][j] -= -1 * x * RATE;
		}
		concept_r_tmp[c_a] -= -2 * concept_r[c_a] * RATE;
	}

	dis = 0;
	for (int i = 0; i < dimension; ++i)
	{
		dis += sqr(entity_vec[e_b][i] - conceptVec[c_b][i]);
	}
	if (dis > sqr(concept_r[c_b]))
	{
		for (int j = 0; j < dimension; ++j)
		{
			double x = 2 * (entity_vec[e_b][j] - conceptVec[c_b][j]);
			entity_tmp[e_b][j] += x * RATE;
			concept_tmp[c_b][j] += -1 * x * RATE;
		}
		concept_r_tmp[c_b] += -2 * concept_r[c_b] * RATE;
	}
}

void gradientSubClassOf(int c1_a, int c2_a, int c1_b, int c2_b)
{
	double dis = 0;
	for (int i = 0; i < dimension; ++i)
	{
		dis += sqr(conceptVec[c1_a][i] - conceptVec[c2_a][i]);
	}
	if (sqrt(dis) > fabs(concept_r[c1_a] - concept_r[c2_a]))
	{
		for (int i = 0; i < dimension; ++i)
		{
			double x = 2 * (conceptVec[c1_a][i] - conceptVec[c2_a][i]);
			concept_tmp[c1_a][i] -= x * RATE;
			concept_tmp[c2_a][i] -= -x * RATE;
		}
		concept_r_tmp[c1_a] -= 2 * concept_r[c1_a] * RATE;
		concept_r_tmp[c2_a] -= -2 * concept_r[c2_a] * RATE;
	}

	dis = 0;
	for (int i = 0; i < dimension; ++i)
	{
		dis += sqr(conceptVec[c1_b][i] - conceptVec[c2_b][i]);
	}
	if (sqrt(dis) > fabs(concept_r[c1_b] - concept_r[c2_b]))
	{
		for (int i = 0; i < dimension; ++i)
		{
			double x = 2 * (conceptVec[c1_b][i] - conceptVec[c2_b][i]);
			concept_tmp[c1_b][i] += x * RATE;
			concept_tmp[c2_b][i] += -x * RATE;
		}
		concept_r_tmp[c1_b] += 2 * concept_r[c1_b] * RATE;
		concept_r_tmp[c2_b] += -2 * concept_r[c2_b] * RATE;
	}
}
*/

void gradientInverseOf(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b, INT inverseRel)
{
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimension;
	INT lastb1 = e1_b * dimension;
	INT lastb2 = e2_b * dimension;
	INT lastbr = rel_b * dimension;
	INT lastInverse = inverseRel * dimension;
	// se entità equivalenti, salva array
	for (INT ii = 0; ii < dimension; ii++)
	{
		REAL x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		if (x > 0)
			x = -alpha;
		else
			x = alpha;

		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;

		relationVec[lastInverse + ii] -= x;
		entityVec[lasta2 + ii] -= x;
		entityVec[lasta1 + ii] += x;

		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relationVec[lastbr + ii] -= x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;

		relationVec[lastInverse + ii] -= x;
		entityVec[lastb2 + ii] -= x;
		entityVec[lastb1 + ii] += x;
	}
}

void gradientEquivalentProperty(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b, INT equivalent)
{
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimension;
	INT lastb1 = e1_b * dimension;
	INT lastb2 = e2_b * dimension;
	INT lastbr = rel_b * dimension;
	INT lastEquivalent = equivalent * dimension;
	// se entità equivalenti, salva array
	for (INT ii = 0; ii < dimension; ii++)
	{
		REAL x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		if (x > 0)
			x = -alpha;
		else
			x = alpha;
		relationVec[lastar + ii] -= x;
		relationVec[lastEquivalent + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;
		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relationVec[lastbr + ii] -= x;
		relationVec[lastEquivalent + ii] -= x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;
	}
}

int getFalseClass(int entity)
{
	int j = -1;
	pair<std::multimap<int, int>::iterator, multimap<int, int>::iterator> ret;
	ret = cls2false_t.equal_range(entity);
	vector<INT> cls_vec;
	for (multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
	{
		cls_vec.push_back(it->second);
	}
	if (cls_vec.size() > 0)
	{
		int RandIndex = rand() % cls_vec.size();
		j = cls_vec[RandIndex];
	}
	return j;
}

void train_kb(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b, INT id)
{
	if (isInstanceOf(rel_a))
	{
		int cut = 10 - (int)(epoch * ins_cut / trainTimes);
		//trainInstanceOf(e1_a, e2_a, cut, id);
	}
	else if (isSubclassOf(rel_a))
	{
		int cut = 10 - (int)(epoch * sub_cut / trainTimes);
		//trainSubClassOf(e1_a, e2_a, cut, id);
	}

	REAL sum1 = calc_sum(e1_a, e2_a, rel_a);
	REAL sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2)
	{
		res += margin + sum1 - sum2;
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
			gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
		}
	}
}

int getHeadCorrupted(int triple_index, int id)
{
	int j = -1;
	// Formula di Slovin per la dimensione del campione (generati da corrupt_tail)
	float error = 0.2f; // margine di errore del 20%
	int tries = entityTotal / (1 + entityTotal * error * error);
	for (int i = 0; i < tries; i++)
	{
		int corrpt_head = corrupt_tail(id, trainList[triple_index].t, trainList[triple_index].r);
		if (!inDomain(trainList[triple_index].r, corrpt_head))
		{
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
int getTailCorrupted(int triple_index, int id)
{
	int j = -1;
	// Formula di Slovin per la dimensione del campione (generati da corrupt_tail)
	float error = 0.2f; // margine di errore del 20%
	int tries = entityTotal / (1 + entityTotal * error * error);
	for (int i = 0; i < tries; i++)
	{
		int corrpt_tail = corrupt_head(id, trainList[triple_index].h, trainList[triple_index].r);
		if (!inRange(trainList[triple_index].r, corrpt_tail))
		{
			j = corrpt_tail;
			break;
		}
	}
	return j;
}

bool hasRange(int rel)
{
	pair<std::multimap<int, int>::iterator, multimap<int, int>::iterator> ret;
	ret = rel2range.equal_range(rel);
	return (rel2range.count(rel) > 0);
}

bool hasDomain(int rel)
{
	pair<std::multimap<int, int>::iterator, multimap<int, int>::iterator> ret;
	ret = rel2range.equal_range(rel);
	return (rel2domain.count(rel) > 0);
}

// scegli in modo casuale una coda o una testa da corrompere e addestra
int train(int triple_index, int id)
{
	int j = -1;
	uint pr;

	if (triple_index > tripleTotal + instanceOf.size())
	{
		// Triple SubclassOf
		int idx_sub = triple_index - tripleTotal - instanceOf.size();
		int h = subClassOf.at(idx_sub).first;
		int t = subClassOf.at(idx_sub).second;

		train_kb(h, t, -1, 0, 0, -1, id);
	}
	else if (triple_index > tripleTotal)
	{
		// Triple instanceOf
		int idx_ins = triple_index - tripleTotal;
		int h = instanceOf.at(idx_ins).first;
		int t = instanceOf.at(idx_ins).second;

		train_kb(h, t, typeOf_id, 0, 0, typeOf_id, id);
	}

	// Triple del Training Set
	if (bernFlag)
		pr = 1000 * right_mean[trainList[triple_index].r] / (right_mean[trainList[triple_index].r] + left_mean[trainList[triple_index].r]);
	else
		pr = 500;
	if (randd(id) % 1000 < pr || isFunctional(trainList[triple_index].r))
	{
		if (hasRange(trainList[triple_index].r))
		{
			j = getTailCorrupted(triple_index, id);
		}
		if (j == -1)
			j = corrupt_head(id, trainList[triple_index].h, trainList[triple_index].r);
		train_kb(trainList[triple_index].h, trainList[triple_index].t, trainList[triple_index].r, trainList[triple_index].h, j, trainList[triple_index].r, id);
	}
	else
	{
		if (hasDomain(trainList[triple_index].r))
			j = getHeadCorrupted(triple_index, id);
		if (j == -1)
			j = corrupt_tail(id, trainList[triple_index].t, trainList[triple_index].r);

		train_kb(trainList[triple_index].h, trainList[triple_index].t, trainList[triple_index].r, j, trainList[triple_index].t, trainList[triple_index].r, id);
	}
	tot_batches++;
	return j;
}

void *trainMode(void *con)
{
	INT id, index, j;
	id = (unsigned long long)(con);
	set_next_random_id(id);
	for (INT k = Batch / threads; k >= 0; k--)
	{
		index = rand_max(id, trainSize); // ottieni un indice casuale
		j = train(index, id);

		if (index < tripleTotal)
		{
			norm(relationVec + dimension * trainList[index].r);
			norm(entityVec + dimension * trainList[index].h);
			norm(entityVec + dimension * trainList[index].t);
			norm(entityVec + dimension * j);
		}
	}
	pthread_exit(NULL);
}

void train(void *con)
{
	srand(time(0));

	Len = tripleTotal;
	Batch = Len / nbatches;
	set_next_random();
	for (epoch = 0; epoch < trainTimes; epoch++)
	{
		res = 0;
		for (INT batch = 0; batch < nbatches; batch++)
		{
			if (!debug)
			{
				pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
				for (long a = 0; a < threads; a++)
					pthread_create(&pt[a], NULL, trainMode, (void *)a);
				for (long a = 0; a < threads; a++)
					pthread_join(pt[a], NULL);
				free(pt);
			}
			else
			{
				trainMode(0);
			}
		}
		printf("epoch %d %f %d\n", epoch, res, trainTimes);

		// Salva il modello ogni 50 epoch
		if (outPath != "" && epoch % 50 == 0)
		{
			out();
		}
	}
}

/*
	Get the results of transE.
*/

void out_binary()
{
	INT len, tot;
	REAL *head;
	FILE *f2 = fopen((outPath + "relation2vec" + note + ".bin").c_str(), "wb");
	FILE *f3 = fopen((outPath + "entity2vec" + note + ".bin").c_str(), "wb");
	len = relationTotal * dimension;
	tot = 0;
	head = relationVec;
	while (tot < len)
	{
		INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f2);
		tot = tot + sum;
	}
	len = entityTotal * dimension;
	tot = 0;
	head = entityVec;
	while (tot < len)
	{
		INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f3);
		tot = tot + sum;
	}
	fclose(f2);
	fclose(f3);
}

void out()
{
	if (outBinaryFlag)
	{
		out_binary();
		return;
	}
	FILE *f2 = fopen((outPath + "relation2vec" + note + ".vec").c_str(), "w");
	FILE *f3 = fopen((outPath + "entity2vec" + note + ".vec").c_str(), "w");
	FILE *f4 = fopen((outPath + "concept2vec" + note + ".vec").c_str(), "w");
	for (INT i = 0; i < relationTotal; i++)
	{
		INT last = dimension * i;
		for (INT ii = 0; ii < dimension; ii++)
			fprintf(f2, "%.6f\t", relationVec[last + ii]);
		fprintf(f2, "\n");
	}
	for (INT i = 0; i < entityTotal; i++)
	{
		INT last = i * dimension;
		for (INT ii = 0; ii < dimension; ii++)
			fprintf(f3, "%.6f\t", entityVec[last + ii]);
		fprintf(f3, "\n");
	}
	/*for (int i=0; i<conceptTotal; i++)
	{
		for (int ii=0; ii<dimension; ii++)
			fprintf(f4,"%.6lf\t",conceptVec[i][ii]);
		fprintf(f4,"\n");
		fprintf(f4,"%.6lf\t", concept_r[i]);
		fprintf(f4,"\n");
	}*/
	fclose(f2);
	fclose(f3);
	fclose(f4);
}

void CSVout()
{
	if (outBinaryFlag)
	{
		out_binary();
		return;
	}
	FILE *f2 = fopen((outPath + "relation2vec" + note + ".csv").c_str(), "w");
	FILE *f3 = fopen((outPath + "entity2vec" + note + ".csv").c_str(), "w");
	for (INT i = 0; i < relationTotal; i++)
	{
		INT last = dimension * i;
		for (INT ii = 0; ii < dimension; ii++)
			fprintf(f2, "%.6f,", relationVec[last + ii]);
		fprintf(f2, "\n");
	}
	for (INT i = 0; i < entityTotal; i++)
	{
		INT last = i * dimension;
		for (INT ii = 0; ii < dimension; ii++)
			fprintf(f3, "%.6f,", entityVec[last + ii]);
		fprintf(f3, "\n");
	}
	fclose(f2);
	fclose(f3);
}

/*
	Main function
*/

int ArgPos(char *str, int argc, char **argv)
{
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a]))
		{
			if (a == argc - 1)
			{
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}

void setparameters(int argc, char **argv)
{
	int i;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
		dimension = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0)
		inPath = argv[i + 1];
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0)
		outPath = argv[i + 1];
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0)
		loadPath = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0)
		threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0)
		trainTimes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-nbatches", argc, argv)) > 0)
		nbatches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
		alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0)
		margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-load-binary", argc, argv)) > 0)
		loadBinaryFlag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-out-binary", argc, argv)) > 0)
		outBinaryFlag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0)
		note = argv[i + 1];
}

int main(int argc, char **argv)
{
	setparameters(argc, argv);
	printf("Prepare\n");
	init();
	owlInit();
	if (loadPath != "")
		load();
	printf("Train\n");
	train(NULL);
	if (outPath != "")
		out();
	return 0;
}
