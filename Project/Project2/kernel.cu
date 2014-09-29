#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h> 
#include "CPUcode.h"
#include <time.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>

#define blockSize 256
bool print = false;

const int originN = 1000;

//Part2
__global__ void NaivePrefixSum(int *in,int *out,int n,int d)
{
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(k<n)
	{
		if(k>= (int)pow(2.0,d-1))
		    out[k] = in[k-(int)pow(2.0,d-1)] + in[k];
	    else
		    out[k] = in[k];	    
	}	
}

void InitNaivePrefixSum(int *origin,int *result,int N)
{
	float time;
	int *in;
	int *out;
	int* temp;
	cudaMalloc((void**)&in,sizeof(int)*N);
	cudaMalloc((void**)&out,sizeof(int)*N);
	cudaMemcpy(in, origin, sizeof(int)*N, cudaMemcpyHostToDevice);	
	int numBlocks = (int)ceil(N/(float)blockSize);
	double t = gettime();
	for(int d=1;(int)pow(2.0,d-1)<=N;d++)
	{
        NaivePrefixSum<<<numBlocks,blockSize>>> (in,out,N,d);
		//update indata
		temp = in;
        in=out;
		out = temp;
	}
	out = in;
	t = gettime()-t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;

	result[0] = 0;
	cudaMemcpy(result+1, out, sizeof(int)*N, cudaMemcpyDeviceToHost); 
	cudaFree(in);
	cudaFree(out);
}

//Part3 a
__global__ void PrefixSumSharedM(int  *in,int  *out,int n)
{
	int k = threadIdx.x;
	__shared__ int s_in [blockSize];
	__shared__ int s_out[blockSize];

	s_in[k] = in[k];
	__syncthreads();

	for(int d=1;(int)pow(2.0,d-1)<=n;d++)
	{	
		if(k>= (int)pow(2.0,d-1))
			s_out[k] = s_in[k-(int)pow(2.0,d-1)] + s_in[k];
		else
			s_out[k] = s_in[k];	    

		s_in[k] = s_out[k];
		__syncthreads();
	}

	out[k] = s_out[k];
}

void InitPrefixSumSharedM(int *origin,int *result,int N)
{
	float time;
	int *in,*out;
	cudaMalloc((void**)&in,sizeof(int)*N);
	cudaMalloc((void**)&out,sizeof(int)*N);
	cudaMemcpy(in, origin, sizeof(int)*N, cudaMemcpyHostToDevice);
	int numBlocks = (int)ceil(N/(float)blockSize);
	if(numBlocks>1)
	{
		cout<<"  Error, more numbers than blocksize."<<endl;
		return;
	}

	double t = gettime();
    PrefixSumSharedM<<<numBlocks,blockSize>>> (in,out,N);
	t = gettime() - t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;
	result[0] = 0;
	cudaMemcpy(result+1, out, sizeof(int)*N, cudaMemcpyDeviceToHost); 
	cudaFree(in);
	cudaFree(out);
}

//Part3 b
__global__ void OPPrefixSumSharedM(int  *in,int  *out,int n,int *sums)
{
	int gk = blockDim.x * blockIdx.x + threadIdx.x;
	if(gk>=n)
		return;

	int k = threadIdx.x;
	__shared__ int s_in [blockSize];
	__shared__ int s_out[blockSize];

	s_in[k] = in[gk];
	__syncthreads();

	for(int d=1;(int)pow(2.0,d-1)<=blockSize;d++)
	{	
		if(k>= (int)pow(2.0,d-1))
			s_out[k] = s_in[k-(int)pow(2.0,d-1)] + s_in[k];
		else
			s_out[k] = s_in[k];	    

		s_in[k] = s_out[k];

		__syncthreads();
	}

	if(k==blockSize-1)
	   sums[blockIdx.x] = s_out[k]; 
	out[gk] = s_out[k]; 
}

__global__ void AddInc(int  *in,int *incs)
{
	int gk = blockDim.x * blockIdx.x + threadIdx.x;
	if(blockIdx.x>=1)
	    in[gk] += incs[blockIdx.x-1];
}

void InitOPPrefixSumSharedM(int *origin,int *result,int N)
{
	int *in,*out,*sums,*incr;
	int numBlocks = (int)ceil(N/(float)blockSize);
	cudaMalloc((void**)&in,sizeof(int)*N);
	cudaMalloc((void**)&out,sizeof(int)*N);
	cudaMalloc((void**)&sums,sizeof(int)*numBlocks);
	cudaMalloc((void**)&incr,sizeof(int)*numBlocks);
	cudaMemcpy(in, origin, sizeof(int)*N, cudaMemcpyHostToDevice);

	double t = gettime();
	//Get sums for each block
	OPPrefixSumSharedM<<<numBlocks,blockSize>>> (in,out,N,sums);

	if(numBlocks>1)
	{
		//Get incrs
		int *temp;
		int numBlocks2 = (int)ceil(numBlocks/(float)blockSize);
		for(int d=1;(int)pow(2.0,d-1)<=numBlocks;d++)
		{
			NaivePrefixSum<<<numBlocks2,blockSize>>> (sums,incr,numBlocks,d);
			//update indata
			temp = sums;
			sums = incr;
			incr = temp;
		}
		incr = sums;

		//Add to out
		AddInc<<<numBlocks,blockSize>>> (out,incr);
	}	

	t = gettime()-t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;
	result[0] = 0;
	cudaMemcpy(result+1, out, sizeof(int)*(N-1), cudaMemcpyDeviceToHost); 

	cudaFree(in);
	cudaFree(out);
	cudaFree(sums);
	cudaFree(incr);
}

//Part 4
__global__ void GPUScatter(int *in,int *newin,bool *inbool,int n)
{
	int gk = blockDim.x * blockIdx.x + threadIdx.x;
	if(gk>=n)
		return;

	if(in[gk]>0)
	{
		inbool[gk] = true;
		newin[gk] = 1;
	}
	else if(in[gk]==0)
	{
		inbool[gk] = false;
		newin[gk] = 0;
	}
}

__global__ void GetMaxIndex(int *in,int *maxindex,int n)
{
	int gk = blockDim.x * blockIdx.x + threadIdx.x;
	if(gk>=n)
		return;

	if(gk==n-1)
	    maxindex[0] = in[gk];
}

__global__ void StreamCompact(int *in,int *out,bool *inbool,int *final,int n)
{
	int gk = blockDim.x * blockIdx.x + threadIdx.x;
	if(gk>=n)
		return;

	if(out[gk]>0&&inbool[gk]==true)
		final[out[gk]-1] = in[gk];
}

int* InitStreamCompact(int *origin,int N,int &l)
{
	float time;
	int *in,*newin,*out,*final,*sums,*incr,*maxindex;
	bool *inbool;
	int numBlocks = (int)ceil(N/(float)blockSize);
	cudaMalloc((void**)&maxindex,sizeof(int)*1);
	cudaMalloc((void**)&inbool,sizeof(bool)*N);
	cudaMalloc((void**)&in,sizeof(int)*N);
	cudaMalloc((void**)&newin,sizeof(int)*N);
	cudaMalloc((void**)&out,sizeof(int)*N);
	cudaMalloc((void**)&sums,sizeof(int)*numBlocks);
	cudaMalloc((void**)&incr,sizeof(int)*numBlocks);
	cudaMemcpy(in, origin, sizeof(int)*N, cudaMemcpyHostToDevice);
	
	double t = gettime();
	//Scatter
	GPUScatter<<<numBlocks,blockSize>>> (in,newin,inbool,N);

	//Scan
	//Get sums for each block
	OPPrefixSumSharedM<<<numBlocks,blockSize>>> (newin,out,N,sums);

	if(numBlocks>1)
	{
		//Get incrs
		int *temp;
		int numBlocks2 = (int)ceil(numBlocks/(float)blockSize);
		for(int d=1;(int)pow(2.0,d-1)<=numBlocks;d++)
		{
			NaivePrefixSum<<<numBlocks2,blockSize>>> (sums,incr,numBlocks,d);
			//update indata
			temp = sums;
			sums = incr;
			incr = temp;
		}
		incr = sums;

		//Add to out
		AddInc<<<numBlocks,blockSize>>> (out,incr);
	}	

	//Get Maxindex for out array
	GetMaxIndex<<<numBlocks,blockSize>>> (out,maxindex,N);
	int max;
	cudaMemcpy(&max, maxindex, sizeof(int)*1, cudaMemcpyDeviceToHost); 


	//Generate final result
	cudaMalloc((void**)&final,sizeof(int)*max);
	StreamCompact<<<numBlocks,blockSize>>> (in,out,inbool,final,N);

	t = gettime()-t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;
	int *result = new int[max];
	cudaMemcpy(result, final, sizeof(int)*max, cudaMemcpyDeviceToHost); 
	cudaFree(in);
	cudaFree(out);
	cudaFree(sums);
	cudaFree(incr);
	cudaFree(final);
	l = max;
	return result;
}


//Thrust
struct not_Zero
{
	__host__ __device__
		bool operator()(const int x)
	{
		if(x>0) return true;
		else return false;
	}
};


int* ThrustStreamCompact(int *origin,int N,int &l)
{
	//Count how many numbers is not 0
	int finallength = thrust::count_if(origin, origin+N,not_Zero());
	int *result = new int[finallength];
	thrust::copy_if(origin, origin+N,result,not_Zero());
	l = finallength;

	return result;
}


bool Verify(int *r1,int*r2,int l)
{
	for(int i=0;i<l;i++)
	{
		if(r1[i]!=r2[i])
		{
			cout<<"Wrong result at index"<<i<<endl;
			cout<<endl;
			return false;
		}
	}
	cout<<"Virified!"<<endl;
	cout<<endl;
	return true;
}

void main()
{
	int *origin;
	origin = new int[originN];
	for(int i=0;i<originN;i++)
		origin[i] =  rand()%10;
	if(print)
	{
		cout<<"Array:"<<endl;
		for(int i=0;i<originN;i++)
			cout<<origin[i]<<" ";
	}


	cout<<endl;


	//Part1
	cout<<"Serial Version PrefixSum:"<<endl;
	int* result = new int[originN];
	double t = gettime();
	PrefixSum(origin,result,originN);
	t = gettime() - t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;
	cout<<endl;


	//Part2
	cout<<"Naive GPU Version PrefixSum:"<<endl;
	int* result2 = new int[originN+1];
	InitNaivePrefixSum(origin,result2,originN);
	//Verify(result,result2,originN);
	cout<<endl;

	//Part3 a
	cout<<"GPU Version PrefixSum with Shared Memory(Single Block):"<<endl;
	int* result3 = new int[originN+1];
	InitPrefixSumSharedM(origin,result3,originN);
	//Verify(result,result3,originN);
	cout<<endl;

	//Part3 b
	cout<<"GPU Version PrefixSum with Shared Memory(Arbitrary length):"<<endl;
	int* result4 = new int[originN];
	InitOPPrefixSumSharedM(origin,result4,originN);
	//Verify(result,result4,originN);
	cout<<endl;


	//Part4 
	int l = 0; //Will be the length of compact array
	cout<<"Serial Version StreamCompact:"<<endl;
	t = gettime();
	int* test =	StreamCompact(origin,originN,l);
	t = gettime() - t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;
	cout<<endl;


	cout<<"GPU version Stream Compact:"<<endl;
	int* result5 = InitStreamCompact(origin,originN,l);
	//Verify(test,result5,l);
	cout<<endl;


	cout<<"Thrust version Stream Compact:"<<endl;
	t = gettime();
	int* result6 = ThrustStreamCompact(origin,originN,l);
	t = gettime() - t;
	cout<<"  Run time:"<<1000*t<<" ms"<<endl;
	//Verify(test,result6,l);
	cout<<endl;

	delete []origin;
	delete []result;
	delete []result2;
	delete []result3;
	delete []result4;
	delete []result5;
	delete []result6;

	return;
}