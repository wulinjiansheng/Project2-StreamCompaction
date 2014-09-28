#include <vector>
#include <iostream>
using namespace std;

void PrefixSum(int* origin,int *result,int n);
void SerialScatter(int* origin,int *result,int n);
int* StreamCompact(int* origin,int n, int &l);
double gettime();