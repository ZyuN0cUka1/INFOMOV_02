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

__kernel void update_positions2( 
	const int& _flag, 
	__global float* pos,
	__global bool* fixflag,
	__global float* fixpos,
	__global float* restright,
	__global float* restleft,
	__global float* restup,
	__global float* restdown )
{
	const int p = get_global_id(0);
	const uint ix = ((p&0x7f)<<1)|(_flag&0x1);
	const uint iy = ((p&0x3f80)>>6)|((_flag>>1)&0x1);
	const uint idx = CalfromXY(ix,iy);
	const uint idxright = CalfromXY(ix+1,iy);
	const uint idxleft = CalfromXY(ix-1,iy);
	const uint idxup = CalfromXY(ix,iy+1);
	const uint idxdown = CalfromXY(ix,iy-1);

	float x = pos[idx*2];
	float y = pos[idx*2+1];
	
	Constraint(x,y,&pos[idxright*2],restright);
	Constraint(x,y,&pos[idxleft*2],restleft);
	Constraint(x,y,&pos[idxup*2],restup);
	Constraint(x,y,&pos[idxdown*2],restdown);

	pos[idx*2] = x;
	pos[idx*2+1] = y;
} 

void Constraint( float& x, float& y, float* neighbour, const float& restlength )
{
	float delx = neighbour[0]-x;
	float dely = neighbour[1]-y;
	float dist = sqrt(delx*delx+dely*dely);
	if(!isfinite(dist)) return;
	if(dist<=restlength) return;
	float extra = dist/restlength-1;
	float dirx = 0.5*delx*extra;
	float diry = 0.5*dely*extra;
	x+=dirx;
	neighbour[0]-=dirx;
	y+=diry;
	neighbour[1]-=diry;
}
uint CalfromXY( uint x, uint y ) { return x+y<<8; }

__kernel void fix_point(__global bool* fixed,__global float* pos,__global float* fix)
{
	const int p = get_global_id(0);
	if(fixed[p])
	{
		pos[p*2]=fix[p*2];
		pos[p*2+1]=fix[p*2+1];
	}
}
// EOF