#include <random>
#include <cstring>
#include <cmath>
#include "rbm.hpp"

struct Matrix{
	float* data;
	float trash;
	unsigned size;
	
	Matrix(unsigned s){
		data=new float[s*s];
		size=s;
		for(unsigned i=0;i<s*s;++i){
			data[i]=0;
		}
	}
	
	~Matrix(){
		delete[] data;
	}
	
	float& operator()(int x,int y){
		if(x<0 || x>=size || y<0 || y>=size){
			return trash;
		}
		
		return data[y*size+x];//(size-y)*size+x];
	}
};

void rotate(Matrix& in,Matrix& out,float deg){
	do{
		deg+=360;
	}while(deg<0);
	
	deg=fmodf(deg,360);
	
	if(deg>=270){
		for(unsigned y=0;y<in.size;++y){
			for(unsigned x=0;x<in.size;++x){
				out(y,-x)=in(x,y);
			}
		}
		
		deg-=270;
	}
	else if(deg>=180){
		for(unsigned y=0;y<in.size;++y){
			for(unsigned x=0;x<in.size;++x){
				out(-x,-y)=in(x,y);
			}
		}
		
		deg-=180;
	}
	else if(deg>=90){
		for(unsigned y=0;y<in.size;++y){
			for(unsigned x=0;x<in.size;++x){
				out(in.size-y,x)=in(x,y);
			}
		}
		
		deg-=90;
	}
	/*
	float cr=cosf(deg*M_PI/180),sr=sinf(deg*M_PI/180);
	unsigned off=(in.size-out.size)/2;
	
	for(unsigned y=0;y<in.size;++y){
		for(unsigned x=0;x<in.size;++x){
			unsigned ax=x-in.size/2,ay=y-in.size/2;
			
			float nx=ax*cr-ay*sr,ny=ay*cr+ax*sr,
				dx=nx-floorf(nx),dy=ny-floorf(ny),
				val=in(x,y);
			
			unsigned ix=floorf(nx)+in.size/2-off,
				iy=floorf(ny)+in.size/2-off;
			
			//Upper left
			out(ix,iy)+=val*dx*dy;
			//Upper right
			out(ix,iy+1)+=val*(1-dx)*dy;
			//Lower left
			out(ix+1,iy)+=val*dx*(1-dy);
			//Lower right
			out(ix+1,iy+1)+=val*(1-dx)*(1-dy);
		}
	}*/
}

int main(){
	const unsigned isize=16;
	const unsigned osize=unsigned(isize/sqrt(2)/2)*2;
	
	Matrix orig{isize};
	Matrix out{osize};
	
	std::mt19937 entropy{std::random_device{}()};
	std::normal_distribution<float> dist{0,1};
	
	for(unsigned i=0;i<orig.size*orig.size;++i){
		orig.data[i]=tanhf(dist(entropy));
	}
	
	rotate(orig,out,90);
	
	FILE* f=fopen("orig.raw","wb");
	if(!f){
		perror("Couldn't open orig.raw");
		return 1;
	}
	fwrite(orig.data,sizeof(float),orig.size*orig.size,f);
	fclose(f);
	
	f=fopen("out.raw","wb");
	if(!f){
		perror("Couldn't open out.raw");
		return 1;
	}
	fwrite(out.data,sizeof(float),out.size*out.size,f);
	fclose(f);
	
	return 0;
}
