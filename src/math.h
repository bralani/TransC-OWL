#ifndef MATH_H
#define MATH_H

    #include <vector>

    using namespace std;

    void set_next_random();
    void set_next_random_id(int id);
    unsigned long long randd(int id);
    int rand_max(int id, int x);
    int rand_max2(int x);
    float rand(float min, float max);
    float normal(float x, float miu, float sigma);
    float randn(float miu, float sigma, float min, float max);
    void norm(float *con);
    double vecLen(vector<double> &a);
    double normV(vector<double> &a);
    void normR(double& r);
    double sqr(double x);

#endif