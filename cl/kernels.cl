#include "template/common.h"
#include "cl/tools.cl"

__kernel void render( __global uint* pixels, const int offset )
{
	// plot a pixel to outimg
	const int p = get_global_id( 0 );
	const int x = p % 511, red = x / 2 + offset;
	const int y = p / 512, green = y / 2;
	pixels[x + y * 512] = (red << 16) + (green << 8);
}

__kernel void update_positions( const float& magic, __global float* pos, __global float* curpos ) 
{
	const int p = get_global_id(0);
	//const int seed = WangHash(p);
	//const float r1 = RandomFloat(seed);
}


// EOF