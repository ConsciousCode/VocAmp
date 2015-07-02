#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "normalize.hpp"

int main(int argc,char* argv[]){
	if(argc<2){
		puts("test-normalize <size>");
		return 0;
	}
	
	std::mt19937 entropy{std::random_device{}()};
	std::normal_distribution<double> dist{0,1.5};
	
	int samps=atoi(argv[1]);
	if(samps<=4){
		puts("Too small a size");
		return 0;
	}
	double* orig=new double[samps];
	double* test=new double[samps];
	double* ft=new double[samps];
	
	for(int i=0;i<samps;++i){
		test[i]=orig[i]=tanh(dist(entropy));
	}
	
	normalize(test,samps);
	memcpy(ft,test,samps*sizeof(double));
	denormalize(test,samps);
	
	double error=0,merr=0;
	for(int i=0;i<samps;++i){
		double dif=fabs(orig[i]-test[i]);
		if(dif>merr){
			merr=dif;
		}
		error+=dif*dif;
		printf("%f -> % -14e -> %f\n",orig[i],ft[i],test[i]);
	}
	
	printf("Square mean error: %f\nMax error: %f\n",error/samps,merr);
	
	delete[] orig;
	delete[] test;
}
