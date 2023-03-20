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


#include "globals.h"
#include "math.h"



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


	owlInit();
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
