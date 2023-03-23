
#ifndef MATH_CPP
#define MATH_CPP

#include <cmath>
#include <vector>

#include "globals.h"
#include "math.h"

const float pi = 3.141592653589793238462643383;
unsigned long long *next_random;

unsigned long long randd(int id)
{
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

void set_next_random() 
{
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));
}

void set_next_random_id(int id) 
{
	next_random[id] = rand();
}

int rand_max(int id, int x)
{
	int res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

int rand_max2(int x){
    int res = (rand() * rand()) % x;
    while (res<0)
        res+=x;
    return res;
}

float rand(float min, float max)
{
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu, float sigma)
{
	return 1.0 / sqrt(2 * pi) / sigma * exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma));
}

float randn(float miu, float sigma, float min, float max)
{
	float x, y, dScope;
	do
	{
		x = rand(min, max);
		y = normal(x, miu, sigma);
		dScope = rand(0.0, normal(miu, miu, sigma));
	} while (dScope > y);
	return x;
}

void norm(float *con)
{
	float x = 0;
	for (int ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x > 1)
		for (int ii = 0; ii < dimension; ii++)
			*(con + ii) /= x;
}

double vecLen(vector<double> &a){
    double res = 0;
    for(double i : a)
        res += i * i;
    res = sqrt(res);
    return res;
}

double normV(vector<double> &a) {
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

double sqr(double x){
    return x * x;
}

#endif