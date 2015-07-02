#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace config{
	//Sample rate used by the model
	const unsigned SAMP_RATE=44100/2;
	//The lowest frequency we care about modeling (lowest of male). This
	// determines how big the input vector must be
	const unsigned LOW_FREQ=80;
	
	//Estimate of the # of bits needed to represent the ideal model
	const unsigned BEST_BITS=64;
	
	//The size of the buffer used for data
	const unsigned NDAT=2*SAMP_RATE/LOW_FREQ;
	
	//Input vector size
	const unsigned NVIS=NDAT;//(NDAT+1)/2;
	//The size of the window
	const unsigned WINDOW=NDAT/5;
	
	//Hidden vector size
	const unsigned NMID=BEST_BITS*2;
	
	const unsigned NHID=BEST_BITS*3/2;
	
	typedef rbm::RBM Model;
	
	typedef rbm::LogisticUnit TopUnit;
	typedef rbm::RectifiedUnit MidUnit;
	typedef rbm::LinearUnit BotUnit;
}

#endif
