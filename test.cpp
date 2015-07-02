#include <sndfile.h>
#include <cstdio>
#include <cstring>

#include "rbm.hpp"
#include "config.hpp"
#include "normalize.hpp"

using config::Model;

static auto top_unit=config::TopUnit{};
static auto mid_unit=config::MidUnit{};
static auto bot_unit=config::BotUnit{};

const size_t WINDOW=config::NDAT-1;
const size_t EPOCHS=1;

using config::SAMP_RATE;
using config::NVIS;
using config::NHID;
using config::NDAT;
using config::NMID;

//Smoothing function for clip transitions
// gaussian: a=2.5/sqrt(2pi), b=0, c=sqrt(2)
double smooth(double v){
	const double a=2.5/sqrt(2*M_PI);
	return 1-a*exp(-v*v/4);
}

void dream(double (&vis)[NVIS],TopModel* top,BotModel* bot){
	static double hid[NHID];
	static double mid[NMID];
	
	for(size_t i=0;i<EPOCHS;++i){
		bot->deconstruct(vis,mid,bot_up);
		#ifdef LAYER2
			top->deconstruct(mid,hid,top_up);
			top->reconstruct(mid,hid,top_down);
		#endif
		bot->reconstruct(vis,mid,bot_down);
	}
}

void smooth_waveform(double (&data)[NDAT]){
	static double left=0;
	
	for(size_t i=0;i<NDAT-1;++i){
		double mid=data[i];
		data[i]=(left+1.618*mid+data[i+1])/6;
		left=mid;
	}
}

int main(){
	SF_INFO in_info={0};
	SNDFILE* in=sf_open("files/dialects/fem-announcer.wav",SFM_READ,&in_info);
	if(!in){
		printf("Couldn't open file: %s\n",sf_strerror(in));
		return 1;
	}
	
	SF_INFO out_info={0,SAMP_RATE,1,SF_FORMAT_WAV|SF_FORMAT_FLOAT,0,0};
	SNDFILE* out=sf_open("out.wav",SFM_WRITE,&out_info);
	if(!out){
		printf("Couldn't open file: %s\n",sf_strerror(out));
		return 1;
	}
	
	#ifdef LAYER2
		FILE* top_w8s=fopen("voice.rbm","rb");
		TopModel* top=TopModel::load(top_w8s);
		fclose(top_w8s);
		
		FILE* bot_w8s=fopen("bottom.rbm","rb");
		BotModel* bot=BotModel::load(bot_w8s);
		fclose(bot_w8s);
	#else
		TopModel* top=nullptr;
		
		FILE* bot_w8s=fopen("voice.rbm","rb");
		BotModel* bot=BotModel::load(bot_w8s);
		fclose(bot_w8s);
	#endif
	
	double data[NDAT];
	double (&vis)[NVIS]=(double(&)[NVIS])data;
	
	sf_read_double(in,data,NDAT);
	normalize(data,NDAT);
	
	dream(vis,top,bot);
	
	denormalize(data,NDAT);
	
	smooth_waveform(data);
	
	sf_write_double(out,data,WINDOW);
	memmove(data,data+WINDOW,NDAT-WINDOW);
	
	while(sf_read_double(in,data+NDAT-WINDOW,WINDOW)==WINDOW){
		normalize(data,NDAT);
		
		dream(vis,top,bot);
		
		denormalize(data,NDAT);
		
		//Smooth transition between windows, based on quadratic function
		data[0]*=smooth(0.5);
		data[1]*=smooth(1.5);
		data[2]*=smooth(2.5);
		data[NDAT-3]*=smooth(2.5);
		data[NDAT-2]*=smooth(1.5);
		data[NDAT-1]*=smooth(0.5);
		
		smooth_waveform(data);
		
		sf_write_double(out,data,WINDOW);
		memmove(data,data+WINDOW,NDAT-WINDOW);
	}
	
	sf_close(in);
	sf_close(out);
	
	#ifdef LAYER2
	delete top;
	#endif
	delete bot;
}
