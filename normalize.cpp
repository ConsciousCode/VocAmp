#include <unordered_map>
#include <cmath>
#include <fftw3.h>
#include "normalize.hpp"

/**
 * Current normalization formula: Fourier transform
**/

struct ManagedPlan{
	fftwf_plan plan;
	
	~ManagedPlan(){
		fftwf_destroy_plan(plan);
	}
};

std::unordered_map<float*,ManagedPlan> fft_plans;
std::unordered_map<float*,ManagedPlan> ifft_plans;

fftwf_plan get_fft_plan(float* data,unsigned size){
	auto it=fft_plans.find(data);
	if(it==fft_plans.end()){
		auto plan=fftwf_plan_r2r_1d(size,data,data,FFTW_R2HC,FFTW_ESTIMATE);
		fft_plans[data]={plan};
		
		return plan;
	}
	
	return it->second.plan;
}

fftwf_plan get_ifft_plan(float* data,unsigned size){
	auto it=ifft_plans.find(data);
	if(it==ifft_plans.end()){
		auto plan=fftwf_plan_r2r_1d(size,data,data,FFTW_HC2R,FFTW_ESTIMATE);
		ifft_plans[data]={plan};
		
		return plan;
	}
	
	return it->second.plan;
}

float logistic(float x){
	return 1/(1+expf(-x));
}

void normalize(float* data,unsigned size){
	const float ftnorm=1/sqrt(size);
	
	fftwf_execute(get_fft_plan(data,size));
	
	data[0]=logistic(data[0]*ftnorm);
	
	//Go over the frequencies and normalize them into polar coordinates
	for(unsigned i=1;i<(size+1)/2;++i){
		float re=data[i]*ftnorm,im=data[size-i]*ftnorm;
		//Real value replaced with distance, guaranteed to be [0, 1]
		data[i]=logistic(sqrt(re*re+im*im));
		
		//Imaginary value replaced with angle - not put through logistic
		// because it isn't used in learning
		data[size-i]=atan2(im,re);
	}
}

float logit(float x){
	return logf(x/(1-x));
}

void denormalize(float* data,unsigned size){
	const float ftnorm=1/sqrt(size);
	
	data[0]=logit(data[0])*ftnorm;
	if(size%2==0){
		//This wasn't adjusted by ftnorm in normalize because it's unused
		// in training
		data[size/2]/=size;
	}
	
	//Go over the polar coordinates and normalize them into frequencies
	for(unsigned i=1;i<(size+1)/2;++i){
		float r=logit(data[i])*ftnorm,theta=data[size-i];
		
		data[i]=r*cos(theta);
		data[size-i]=r*sin(theta);
	}
	
	fftwf_execute(get_ifft_plan(data,size));
}
