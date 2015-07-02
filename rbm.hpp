#ifndef RBM_HPP
#define RBM_HPP

#include <random>
#include <stdexcept>

namespace rbm{
	/**
	 * Calculate the mean square error between the two given arrays.
	**/
	float mse(const float* pos,const float* neg,unsigned count);
	
	/**
	 * Traditional RBM.
	**/
	struct RBM{
		/**
		 * The biases of the layer followed by the weights.
		**/
		float* bias_w8s;
		
		///Aliases for bias and weights
		float *bias,*w8s;
		
		/**
		 * Sizes
		**/
		unsigned nvis,nhid;
		
		RBM(unsigned nv,unsigned nh);
		~RBM();
		
		void load(FILE* f);
		void dump(FILE* f);
		
		void randomize(float sigma);
		
		float& at(unsigned x,unsigned y);
		const float& at(unsigned x,unsigned y) const;
		
		/**
		 * Deconstruct the given visible units based on learned features
		 *  and put the result in the given hidden layer.
		 * 
		 * @param vis The visible layer.
		 * @param hid The hidden layer.
		 * @param activate A callable for determining how the hidden
		 *  units are activated.
		**/
		template<typename C>
		void deconstruct(const float* vis,float* hid,C& activate) const{
			for(unsigned x=0;x<nhid;++x){
				float prob=0;
				for(unsigned y=0;y<nvis;++y){
					prob+=at(x,y)*vis[y];
				}
				hid[x]=activate(prob+bias[x]);
			}
		}
		
		/**
		 * Reconstruct a representative visible layer sample from the
		 *  given hidden units.
		 * 
		 * @param vis The visible layer.
		 * @param hid The hidden layer.
		 * @param activate A callable for determining how the visible
		 *  units are activated.
		**/
		template<typename C>
		void reconstruct(float* vis,const float* hid,C& activate) const{
			for(unsigned y=0;y<nvis;++y){
				float prob=0;
				for(unsigned x=0;x<nhid;++x){
					prob+=at(x,y)*hid[x];
				}
				
				vis[y]=activate(prob);
			}
		}
		
		/**
		 * Contrastive Divergence trainer
		**/
		struct CDTrainer{
			/**
			 * All the extra buffers needed for CD
			**/
			float* negv_posh_negh;
			
			///Aliases for the buffers
			float *negv,*posh,*negh;
			
			/**
			 * Learning rate
			**/
			float rate;
			
			CDTrainer(unsigned nv,unsigned nh,float r);
			
			void learn(RBM& m,float* posv,float* posh,float* negv,float* negh);
			
			template<typename C,typename D>
			void train(RBM& m,float* vis,unsigned epochs,C& up,D& down){
				m.deconstruct(vis,posh,up);
				
				m.reconstruct(negv,posh,down);
				m.deconstruct(negv,negh,up);
				for(unsigned i=1;i<epochs;++i){
					m.reconstruct(negv,negh,down);
					m.deconstruct(negv,negh,up);
				}
				
				learn(m,vis,posh,negv,negh);
			}
		};
	};
	
	struct BinaryUnit{
		float operator()(float p);
	};
	
	struct LinearUnit{
		float operator()(float p);
	};
	
	struct LogisticUnit{
		float slope;
		
		LogisticUnit(float s=1);
		
		float operator()(float p);
	};
	
	struct RectifiedUnit{
		float operator()(float p);
	};
	
	struct NoisyUnit{
		float operator()(float p);
	};
	
	struct LeakyUnit{
		float leak;
		
		LeakyUnit(float l=0.001);
		
		float operator()(float p);
	};
}

#endif
