#include "CPUcode.h"
#include <Windows.h>


//Part 1
void PrefixSum(int* origin,int *result,int n)
{
	result[0] = 0;
	for(int i=1;i<n;i++)
		result[i] = result[i-1] + origin[i-1];

	return;
}

//Part 4
void SerialScatter(int* origin,int *result,int n)
{
	for(int i=0;i<n;i++)
	{
	   if(origin[i]>0)
		   origin[i] = 1;
	}

	PrefixSum(origin,result,n);
}

int* StreamCompact(int* origin,int n,int &l)
{
	 int* corigin = new int[n];
	 int *result = new int[n];
	 for(int i=0;i<n;i++)
		 corigin[i] = origin[i];

     SerialScatter(origin,result,n);
	 int *r = new int[result[n-1]+1];
	 int indexmax = result[n-1];
	 for(int i=n-1;i>=0;i--)
	 {
		 if(result[i]==indexmax)
		 {
			 r[indexmax] = corigin[i];
			 indexmax--;
		 }
	 }

	 //Reset Origin Array
	 for(int i=0;i<n;i++)
		 origin[i] = corigin[i];

	 l = result[n-1]+1;
	 return r;
}

double gettime()
{
	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dff =(double)litmp.QuadPart;
	QueryPerformanceCounter(&litmp);
	double t =(double)litmp.QuadPart;
	return t/dff;
}

