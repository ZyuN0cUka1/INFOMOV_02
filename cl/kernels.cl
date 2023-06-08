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

__kernel void update_positions(__global float* curpos, __global float* prevpos, __global float* magic)
{
	int p = get_global_id(0);
	int seed = WangHash(p);
	float r1 = RandomFloat(&seed);
	float r2 = RandomFloat(&seed);
	float r3 = RandomFloat(&seed);


	//float2 curpos{ grid(x, y).pos.x,grid(x, y).pos.y }, prevpos{ grid(x, y).prev_pos.x,grid(x, y).prev_pos.y };
	float x = curpos[2*p]; 
	float y = curpos[2 * p +1];
	float px = prevpos[2*p];
	float py = prevpos[2 * p + 1];
	//grid(x, y).pos = curpos + (curpos - prevpos) + float2(0, 0.003f); // gravity
	curpos[2 * p]    = x + (x - px);
	curpos[2 * p +1] = y + (y - py) + 0.003f;

	//grid(x, y).prev_pos = curpos;
	prevpos[2 * p] = x;
	prevpos[2 * p + 1] = y;

	//if (Rand(10) < 0.03f) 
	//grid(x, y).pos = float2{ grid(x, y).pos.x, grid(x, y).pos.y } + float2(Rand(0.02f + magic), Rand(0.12f));

	if (10 * r1 < 0.03f) {
		curpos[2 * p] = curpos[2 * p] + (0.02f+*magic) * r2;
		curpos[2 * p + 1] = curpos[2 * p + 1] + 0.12f * r3;
	}

}


// EOF