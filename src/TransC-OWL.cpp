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

#include "io.h"
#include "math.h"
#include "globals.h"

using namespace std;

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
double calcSumInstanceOf(int e, int c){
	double dis = 0;
	for(int i = 0; i < dimension; ++i){
		dis += sqr(entity_vec[e][i] - conceptVec[c][i]);
	}
	if(dis < sqr(concept_r[c])){
		return 0;
	}
	return dis - sqr(concept_r[c]);

}*/

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

/*
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
			conceptVec[c_a][j] -= -1 * x * RATE;
		}
		concept_r[c_a] -= -2 * concept_r[c_a] * RATE;
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
			conceptVec[c_b][j] += -1 * x * RATE;
		}
		concept_r[c_b] += -2 * concept_r[c_b] * RATE;
	}
}*/

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
			conceptVec[c1_a][i] -= x * RATE;
			conceptVec[c2_a][i] -= -x * RATE;
		}
		concept_r[c1_a] -= 2 * concept_r[c1_a] * RATE;
		concept_r[c2_a] -= -2 * concept_r[c2_a] * RATE;
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
			conceptVec[c1_b][i] += x * RATE;
			conceptVec[c2_b][i] += -x * RATE;
		}
		concept_r[c1_b] += 2 * concept_r[c1_b] * RATE;
		concept_r[c2_b] += -2 * concept_r[c2_b] * RATE;
	}
}

/*
void doTrainInstanceOf(int e_a, int c_a, int e_b, int c_b)
{
	double sum1 = calcSumInstanceOf(e_a, c_a);
	double sum2 = calcSumInstanceOf(e_b, c_b);
	if (sum1 + margin_instance > sum2)
	{
		res += (margin_instance + sum1 - sum2);
		gradientInstanceOf(e_a, c_a, e_b, c_b);
	}
}*/

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
					j = rand_max2(entityTotal);
				}
				else
				{
					j = rand() % (int)instance_brother[head].size();
					j = instance_brother[head][j];
				}
			}
			else
			{
				j = rand_max2(entityTotal);
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
					j = rand_max2(conceptTotal);
				}
				else
				{
					j = rand() % (int)concept_brother[tail].size();
					j = concept_brother[tail][j];
				}
			}
			else
			{
				j = rand_max2(conceptTotal);
			}
		} while (instanceOf_ok.count(make_pair(head, j)) > 0);
		doTrainInstanceOf(head, tail, head, j);
		normV(conceptVec[j]);
		normR(concept_r[j]);
	}
	normV(entity_tmp[head]);
	normV(conceptVec[tail]);
	normR(concept_r[tail]);
} */

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
					j = rand_max2(conceptTotal);
				}
				else
				{
					j = rand() % (int)concept_brother[head].size();
					j = concept_brother[head][j];
				}
			}
			else
			{
				j = rand_max2(conceptTotal);
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
					j = rand_max2(conceptTotal);
				}
				else
				{
					j = rand() % (int)concept_brother[tail].size();
					j = concept_brother[tail][j];
				}
			}
			else
			{
				j = rand_max2(conceptTotal);
			}
		} while (subClassOf_ok.count(make_pair(head, j)) > 0);
		doTrainSubClassOf(head, tail, head, j);
	}
	normV(conceptVec[head]);
	normV(conceptVec[tail]);
	normV(conceptVec[j]);
	normR(concept_r[head]);
	normR(concept_r[tail]);
	normR(concept_r[j]);
}


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
	if (isSubclassOf(rel_a)) {
		int cut = 10 - (int)(epoch * sub_cut / trainTimes);
		trainSubClassOf(e1_a, e2_a, cut, id);
	} else if (isInstanceOf(rel_a)) {
		int cut = 10 - (int)(epoch * ins_cut / trainTimes);
		//trainInstanceOf(e1_a, e2_a, cut, id);
	} else {
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
	uint pr;
	int testa = -1, coda = -1, relazione = -1, testaB = -1, codaB = -1;
	int j = -1;
	int instanceOf_size = instanceOf.size();

	// Triple SubclassOf
	if (triple_index > tripleTotal + instanceOf_size)
	{
		int idx_sub = triple_index - tripleTotal - instanceOf_size;
		testa = subClassOf.at(idx_sub).first;
		relazione = -1;
		coda = subClassOf.at(idx_sub).second;
	}
	// Triple instanceOf
	else if (triple_index > tripleTotal)
	{
		int idx_ins = triple_index - tripleTotal - 1;
		testa = instanceOf.at(idx_ins).first;
		relazione = typeOf_id;
		coda = instanceOf.at(idx_ins).second;
	// Triple relazionali
	} else {
		testa = trainList[triple_index].h;
		relazione = trainList[triple_index].r;
		coda = trainList[triple_index].t;

		if (bernFlag)
			pr = 1000 * right_mean[relazione] / (right_mean[relazione] + left_mean[relazione]);
		else
			pr = 500;
		
		if (randd(id) % 1000 < pr || isFunctional(relazione)) {
			if (hasRange(relazione)) {
				j = getTailCorrupted(triple_index, id);
			}
			if (j == -1)
				j = corrupt_head(id, testa, relazione);

			testaB = testa;
			codaB = j;
		} else {
			if (hasDomain(relazione))
				j = getHeadCorrupted(triple_index, id);
			if (j == -1)
				j = corrupt_tail(id, coda, relazione);

			testaB = j;
			codaB = coda;
		}
	}
	

	train_kb(testa, coda, relazione, testaB, codaB, relazione, id);
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
	if(!debug) {
		pthread_exit(NULL);
	} else {
		return NULL;
	}
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

		// Salva il modello ogni 100 epoch
		if (outPath != "" && epoch % 100 == 0)
		{
			out();
		}
	}
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
	if (loadPath != "") load();
	printf("Train\n");
	train(NULL);
	if (outPath != "")
		out();
	return 0;
}
