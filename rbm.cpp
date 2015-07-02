#include "rbm.hpp"

using namespace rbm;

float mse(const float* pos,const float* neg,unsigned count){
	float err=0;
	for(size_t i=0;i<count;++i){
		float dif=pos[i]-neg[i];
		err+=dif*dif;
	}
	
	return err/count;
}

RBM::RBM(unsigned nv,unsigned nh){
	nvis=nv;
	nhid=nh;
	bias_w8s=new float[nv*nh+nv];
	bias=bias_w8s;
	w8s=bias_w8s+nh;
}

float& RBM::at(unsigned x,unsigned y){
	return w8s[y*nhid+x];
}

const float& RBM::at(unsigned x,unsigned y) const{
	return w8s[y*nhid+x];
}

void RBM::load(FILE* f){
	unsigned t;
	if(fread(&t,sizeof(unsigned),1,f)!=1){
		throw std::runtime_error("Reached EOF while parsing .rbm file");
	}
	if(t!=nvis){
		char buf[64];
		sprintf(buf,"Hidden size mismatch (got %d, need %d)",t,nvis);
		throw std::runtime_error(buf);
	}
	
	if(fread(&t,sizeof(unsigned),1,f)!=1){
		throw std::runtime_error("Reached EOF while parsing .rbm file");
	}
	if(t!=nhid){
		char buf[64];
		sprintf(buf,"Visible size mismatch (got %d, need %d)",t,nhid);
		throw std::runtime_error(buf);
	}
	
	if(fread(bias,sizeof(float),nhid,f)!=nhid){
		throw std::runtime_error("Reached EOF while parsing .rbm file");
	}
	
	if(fread(w8s,sizeof(float),nvis*nhid,f)!=nvis*nhid){
		throw std::runtime_error("Reached EOF while parsing .rbm file");
	}
}

void RBM::dump(FILE* f){
	fwrite(&nvis,sizeof(unsigned),1,f);
	fwrite(&nhid,sizeof(unsigned),1,f);
	fwrite(bias,sizeof(float),nhid,f);
	fwrite(w8s,sizeof(float),nvis*nhid,f);
}

void RBM::randomize(float sigma){
	static std::mt19937 entropy{std::random_device{}()};
	std::normal_distribution<float> dist{0,sigma};
	
	for(unsigned x=0;x<nhid;++x){
		for(unsigned y=0;y<nvis;++y){
			at(x,y)=dist(entropy);
		}
		
		bias[x]=dist(entropy);
	}
}

RBM::~RBM(){
	delete[] bias_w8s;
}

RBM::CDTrainer::CDTrainer(unsigned nv,unsigned nh,float r){
	negv_posh_negh=new float[nv+nh*2];
	negv=negv_posh_negh;
	posh=negv+nv;
	negh=posh+nh;
	
	rate=r;
}

void RBM::CDTrainer::learn(RBM& m,float* pv,float* ph,float* nv,float* nh){
	for(unsigned x=0;x<m.nhid;++x){
		for(unsigned y=0;y<m.nvis;++y){
			m.at(x,y)+=(pv[y]*ph[x]-nv[y]*nh[x])*rate;
		}
		
		m.bias[x]+=(ph[x]-nh[x])*rate;
	}
}

float BinaryUnit::operator()(float p){
	static std::mt19937 entropy{std::random_device{}()};
	return std::bernoulli_distribution{1/(1+exp(-p))}(entropy);
}

float LinearUnit::operator()(float p){
	return p;
}

LogisticUnit::LogisticUnit(float s):slope(s){}

float LogisticUnit::operator()(float p){
	return 1/(1+exp(-slope*p));
}

float RectifiedUnit::operator()(float p){
	return p>0?p:0;
}

float NoisyUnit::operator()(float p){
	static std::mt19937 entropy{std::random_device{}()};
	
	p=std::normal_distribution<float>{p,1/(1+expf(-p))}(entropy);
	
	return p>0?p:0;
}

LeakyUnit::LeakyUnit(float l):leak(l){}

float LeakyUnit::operator()(float p){
	return p>0?p:p*leak;
}
