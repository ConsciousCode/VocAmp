#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <fts.h>
#include <sndfile.h>
#include "rbm.hpp"

#include "config.hpp"
#include "normalize.hpp"

using namespace rbm;
using namespace config;

auto top_unit=TopUnit{};
auto mid_unit=MidUnit{};
auto bot_unit=BotUnit{};
#define UNITS bot_unit,mid_unit//,top_unit

//Learning rates
const float RATE=0.0001;

//Initial weight distribution
const float INIT_W8=0.0001;

typedef Model::CDTrainer Trainer;

Trainer&& make_trainer(){
	return std::move(Trainer{NVIS,NMID,RATE});
}

//-Ofast triggers -ffast-math which will optimize isnan calls out
// Specifically, this will check for any exotic float value e.g. NaN and Inf
bool isexotic(float f){
	static const uint64_t EXPONENT=0x7FF00000;
	union{
		uint32_t i;
		float d;
	};
	d=f;
	return (i&EXPONENT)>>52==0x7ff;
}

//Utility function to keep training calculation together
float train(Trainer& t,Model& m,float* data,unsigned epochs){
	//normalize(data,NDAT);
	
	t.train(m,data,epochs,UNITS);
	float err=rbm::mse(data,t.negv,NVIS);
	
	//denormalize(data,NDAT);
	
	return err;
}

double train_file(const char* path,Trainer& t,Model& m,unsigned c){
	static const float PHI=(sqrt(5)-1)/2;
	
	static float data[NDAT];
	static unsigned epochs=1;
	
	double time=0;
	unsigned counter=1;
	
	SF_INFO info={0};
	SNDFILE* f=sf_open(path,SFM_READ,&info);
	if(!f){
		printf("Couldn't open file: %s\n",sf_strerror(f));
		return 0;
	}
	
	if(info.samplerate!=SAMP_RATE){
		printf(
			"File is not the right sample rate (got %d, need %d)\n",
			info.samplerate,SAMP_RATE
		);
	}
	
	printf("Training on file #%d, %s\n",c,path);
	
	sf_read_float(f,data,NDAT);
	
	float error=train(t,m,data,epochs);
	static float avgerr=error;
	avgerr=(PHI*error+(1-PHI)*avgerr);
	
	printf(
		"err = %.8f, rate = %.8f, epochs = %d\n",
		error,t.rate,epochs
	);
	
	memmove(data,data+WINDOW,NDAT-WINDOW);
	
	size_t numread;
	while(numread=sf_read_float(f,data+NDAT-WINDOW,WINDOW)){
		//Create a silence buffer for the end of the file
		for(size_t i=NDAT-WINDOW+numread;i<NDAT;++i){
			data[i]=0;
		}
		
		error=train(t,m,data,epochs);
		avgerr=(PHI*error+(1-PHI)*avgerr);
		
		if(isexotic(error) || fabs(error)>1000){
			printf("Bad error %f after %d samples, aborting\n",error,counter);
			exit(0);
		}
		
		if(counter%5000==0){
			FILE* f=fopen("voice.rbm","wb");
			m.dump(f);
			fclose(f);
			
			printf(
				"err = %.8f, slow rate = %.8f, epochs = %d\n",
				(float)avgerr,(float)t.rate,epochs
			);
		}
		++counter;
		time+=WINDOW/(double)SAMP_RATE;
		
		memmove(data,data+WINDOW,NDAT-WINDOW);
	}
	
	printf("Error = %.8f\n",(float)avgerr);
	
	return time;
}

int main(int argc,char* argv[]){
	printf(
		"ndat = %d, nvis = %d, nhid = %d, window = %d, epochs = 1\n",
		NDAT,NVIS,NHID,WINDOW
	);
	
	Model m{NVIS,NMID};
	Trainer t=make_trainer();
	
	FTS* fts;
	
	char files[]="files";
	char* roots[]={files,0};
	if(!(fts=fts_open(roots,FTS_NOCHDIR,0))){
		perror("Couldn't open directory");
		return -1;
	}
	
	if(argc>1 && argv[1][0]=='y'){
		load_model:
		puts("\nUsing old weights");
		FILE* f=fopen("voice.rbm","rb");
		if(!f){
			puts("Couldn't open voice.rbm");
			return 1;
		}
		m.load(f);
		fclose(f);
	}
	else{
		struct stat buffer;
		if(stat("voice.rbm",&buffer)==0){
			printf("Reset any previous progress? (y/n/c) ");
			char c=getchar();
			if(c=='y'){
				goto create_model;
			}
			else if(c=='c'){
				return 0;
			}
			else{
				goto load_model;
			}
		}
		else{
			create_model:
			m.randomize(INIT_W8);
		}
	}
	
	double time=0;
	auto start=std::chrono::system_clock::now();
	
	unsigned c=0;
	FTSENT* p;
	while(p=fts_read(fts)){
		if(p->fts_info&FTS_F){
			time+=train_file(p->fts_path,t,m,++c);
		}
	}
	
	printf("Length of training data: %.3f minutes\nTook: %.3f minutes\n",
		time/60,std::chrono::duration_cast<
			std::chrono::duration<double,std::ratio<60>>
		>(
			std::chrono::system_clock::now()-start
		).count()
	);
	
	fts_close(fts);
}
