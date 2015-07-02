__global__ void vis2hid(float* vis,float* w8s,float* hid){
	hid[threadIdx.x]+=vis[threadIdx.y]*w8s[threadIdx.y*blockDim.x+threadIdx.x];
}

__global__ void hid2vis(float* hid,float* w8s,float* vis){
	vis[threadIdx.y]+=hid[threadIdx.x]*w8s[threadIdx.y*blockDim.x+threadIdx.x];
}

__global__ void learning(
		float* w8s,float* posv,float* posh,float* negv,float* negh
){
	w8s[threadIdx.y*blockDim.x+threadIdx.x]+=
		posv[threadIdx.y]*posh[threadIdx.x]-negv[threadIdx.y]*negh[threadIdx.x];
}

void deconstruct(float* vis,float* w8s,float* hid,unsigned nv,unsigned nh){
	memset(hid,0,nh*sizeof(float));
	vis2hid<<<nv,nh>>>(vis,w8s,hid);
}

void reconstruct(float* vis,float* w8s,float* hid,unsigned nv,unsigned nh){
	memset(vis,0,nv*sizeof(float));
	hid2vis<<<nv,nh>>>(vis,w8s,hid);
}
