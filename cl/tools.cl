// random numbers: seed using WangHash((threadidx+1)*17), then use RandomInt / RandomFloat
uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d, s = s ^ (s >> 15); return s; }
uint RandomInt( uint* s ) { *s ^= *s << 13, * s ^= *s >> 17, * s ^= *s << 5; return *s; }
float RandomFloat( uint* s ) { return RandomInt( s ) * 2.3283064365387e-10f; /* = 1 / (2^32-1) */ }
void Constraint( float* x, float* y, float* neighbourx, float* neighboury, const float restlength )
{
	float delx = *neighbourx-*x;
	float dely = *neighboury-*y;
	float dist = sqrt(delx*delx+dely*dely);
	if(!isfinite(dist)) return;
	if(dist<=restlength) return;
	float extra = dist/restlength-1;
	float dirx = 0.5*delx*extra;
	float diry = 0.5*dely*extra;
	*x+=dirx;
	*neighbourx-=dirx;
	*y+=diry;
	*neighboury-=diry;
}	
uint CalfromXY( uint x, uint y ) { return x+(y<<8); }

// EOF