#include "precomp.h"
#include "game.h"

#define GRIDSIZE 256

// VERLET CLOTH SIMULATION DEMO
// High-level concept: a grid consists of points, each connected to four 
// neighbours. For a simulation step, the position of each point is affected
// by its speed, expressed as (current position - previous position), a
// constant gravity force downwards, and random impulses ("wind").
// The final force is provided by the bonds between points, via the four
// connections.
// Together, this simple scheme yields a pretty convincing cloth simulation.
// The algorithm has been used in games since the game "Thief".

// ASSIGNMENT STEPS:
// 1. SIMD, part 1: in Game::Simulation, convert lines 119 to 126 to SIMD.
//    You receive 2 points if the resulting code is faster than the original.
//    This will probably require a reorganization of the data layout, which
//    may in turn require changes to the rest of the code.
// 2. SIMD, part 2: for an additional 4 points, convert the full Simulation
//    function to SSE. This may require additional changes to the data to
//    avoid concurrency issues when operating on neighbouring points.
//    The resulting code must be at least 2 times faster (using SSE) or 4
//    times faster (using AVX) than the original  to receive the full 4 points.
// 3. GPGPU, part 1: modify Game::Simulation so that it sends the cloth data
//    to the GPU, and execute lines 119 to 126 on the GPU. After this, bring
//    back the cloth data to the CPU and execute the remainder of the Verlet
//    simulation code. You receive 2 points if the code *works* correctly;
//    note that this is expected to be slower due to the data transfers.
// 4. GPGPU, part 2: execute the full Game::Simulation function on the GPU.
//    You receive 4 additional points if this yields a correct simulation
//    that is at least 5x faster than the original code. DO NOT draw the
//    cloth on the GPU; this is (for now) outside the scope of the assignment.
// Note that the GPGPU tasks will benefit from the SIMD tasks.
// Also note that your final grade will be capped at 10.

#define idx(x,y) ((x)+(y)*256)
#define idx4o(x,y) ((x)*4+(y)*256)
#define idx4(x,y) ((x)+(y)*64)
#define setfloat2(x,y) _mm256_setr_ps((x),(y),(x),(y),(x),(y),(x),(y))

static union { float2 upos[GRIDSIZE * GRIDSIZE]; __m256 upos4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float2 uprev_pos[GRIDSIZE * GRIDSIZE]; __m256 uprev_pos4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float2 ufix[GRIDSIZE * GRIDSIZE]; __m256 ufix4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float urest[GRIDSIZE * GRIDSIZE][4]; float restlength[GRIDSIZE * GRIDSIZE][4]; };

struct float_rest {
	uint i;
	float_rest(uint idx) :i(idx) {};
	float& operator[](uint idx) { return urest[i][idx]; }
};

static int idx = 0;
struct Point
{
	float2& pos = upos[idx];				// current position of the point
	float2& prev_pos = uprev_pos[idx];		// position of the point in the previous frame
	float2& fix = ufix[idx];				// stationary position; used for the top line of points
	bool fixed = false;				// true if this is a point in the top line of the cloth
	float_rest restlength{ idx++ };	// initial distance to neighbours
};



// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid( const uint x, const uint y ) { return pointGrid[x + y * GRIDSIZE]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init()
{
	//static union {
	//	float2 a[4] = { float2(1,2),float2(3,4), float2(5,6), float2(7,8) };
	//	__m256 a4;
	//};
	//static union {
	//	float2 a0[4] = { float2(2,3),float2(4,5), float2(4,5), float2(6,7) };
	//	__m256 a04;
	//};
	//__m256* ap4 = (__m256*) & a4;
	//__m256 asr4 = _mm256_setr_ps(((float*)&a)[0], ((float*)&a)[1], ((float*)&a)[2], ((float*)&a)[3], ((float*)&a)[4], ((float*)&a)[5], ((float*)&a)[6], ((float*)&a)[7]);
	//__m256 alr4 = _mm256_load_ps((float*)a);
	//float astr[8]; _mm256_store_ps(astr, a4);

	//float* out = (float*)&a4;
	//cout << "a4:\t" << out[0] << '\t' << out[1] << '\t' << out[2] << '\t' << out[3] << '\t' << out[4] << '\t' << out[5] << '\t' << out[6] << '\t' << out[7] << endl;

	//out = (float*)ap4;
	//cout << "ap4:\t" << out[0] << '\t' << out[1] << '\t' << out[2] << '\t' << out[3] << '\t' << out[4] << '\t' << out[5] << '\t' << out[6] << '\t' << out[7] << endl;

	//out = (float*)&asr4;
	//cout << "asr4:\t" << out[0] << '\t' << out[1] << '\t' << out[2] << '\t' << out[3] << '\t' << out[4] << '\t' << out[5] << '\t' << out[6] << '\t' << out[7] << endl;

	//out = (float*)&alr4;
	//cout << "alr4:\t" << out[0] << '\t' << out[1] << '\t' << out[2] << '\t' << out[3] << '\t' << out[4] << '\t' << out[5] << '\t' << out[6] << '\t' << out[7] << endl;

	//out = astr;
	//cout << "astr:\t" << out[0] << '\t' << out[1] << '\t' << out[2] << '\t' << out[3] << '\t' << out[4] << '\t' << out[5] << '\t' << out[6] << '\t' << out[7] << endl;

	////__mmask8 mask0 = _mm256_cmp_ps_mask(a4, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
	//__m256 mask1 = _mm256_cmp_ps(a4, a04, _CMP_GT_OQ);
	////cout << mask0 << endl;
	//out = (float*)&mask1;
	//cout << "mask:\t" << out[0] << '\t' << out[1] << '\t' << out[2] << '\t' << out[3] << '\t' << out[4] << '\t' << out[5] << '\t' << out[6] << '\t' << out[7] << endl;

	//cout << sizeof(upos) << ';' << sizeof(upos) / sizeof(float2) << endl;
	//cout << sizeof(upos4) << ';' << sizeof(upos4) / sizeof(__m256) << endl;

	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
	{
		grid( x, y ).pos.x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand( 2 );
		grid( x, y ).pos.y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand( 2 );
		grid( x, y ).prev_pos = grid( x, y ).pos; // all points start stationary
		if (y == 0)
		{
			grid( x, y ).fixed = true;
			grid( x, y ).fix = grid( x, y ).pos;
		}
		else
		{
			grid( x, y ).fixed = false;
		}
	}
	for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
	{
		// calculate and store distance to four neighbours, allow 15% slack
		for (int c = 0; c < 4; c++)
		{
			grid( x, y ).restlength[c] = length( grid( x, y ).pos - grid( x + xoffset[c], y + yoffset[c] ).pos ) * 1.15f;
		}
	}
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid()
{
	// draw the grid
	screen->Clear( 0 );
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++)
	{
		const float2 p1 = grid( x, y ).pos;
		const float2 p2 = grid( x + 1, y ).pos;
		const float2 p3 = grid( x, y + 1 ).pos;
		screen->Line( p1.x, p1.y, p2.x, p2.y, 0xffffff );
		screen->Line( p1.x, p1.y, p3.x, p3.y, 0xffffff );
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		const float2 p1 = grid( GRIDSIZE - 2, y ).pos;
		const float2 p2 = grid( GRIDSIZE - 2, y + 1 ).pos;
		screen->Line( p1.x, p1.y, p2.x, p2.y, 0xffffff );
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
__m256 dl1 = setfloat2(0, 0.003f);
void Game::Simulation()
{
	// simulation is exected three times per frame; do not change this.
	for( int steps = 0; steps < 3; steps++ )
	{
		// verlet integration; apply gravity
		//for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
		//{
		//	float2 curpos = grid( x, y ).pos, prevpos = grid( x, y ).prev_pos;
		//	grid( x, y ).pos += (curpos - prevpos) + float2( 0, 0.003f ); // gravity
		//	grid( x, y ).prev_pos = curpos;
		//	if (Rand( 10 ) < 0.03f) grid( x, y ).pos += float2( Rand( 0.02f + magic ), Rand( 0.12f ) );
		//}

		for (int y = 0; y < GRIDSIZE; y++)for (int x = 0; x < GRIDSIZE / 4; x++)
		{
			__m256 curpos = upos4[idx4(x, y)];
			__m256 prevpos = uprev_pos4[idx4(x, y)];
			_mm256_store_ps((float*)(&uprev_pos[idx4o(x, y)]), curpos);

			curpos = _mm256_add_ps(dl1, _mm256_add_ps(curpos, _mm256_sub_ps(curpos, prevpos)));
			__m256 mask = _mm256_cmp_ps(setfloat2(Rand(10), Rand(10)), _mm256_set1_ps(0.003f), _CMP_LT_OQ);
			__m256 randl1 = setfloat2(0.12f, 0.02f + magic);
			_mm256_store_ps((float*)(&upos[idx4o(x, y)]), _mm256_add_ps(curpos, _mm256_and_ps(mask, randl1)));
		}

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.
		for (int i = 0; i < 4; i++)
		{
			//for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
			//{
			//	//float2 pointpos = grid( x, y ).pos;
			//	float2 pointpos = upos[idx(x, y)];
			//	// use springs to four neighbouring points
			//	for (int linknr = 0; linknr < 4; linknr++)
			//	{
			//		Point& neighbour = grid( x + xoffset[linknr], y + yoffset[linknr] );
			//		float distance = length( neighbour.pos - pointpos );
			//		if (!isfinite( distance ))
			//		{
			//			// warning: this happens; sometimes vertex positions 'explode'.
			//			continue;
			//		}
			//		if (distance > grid( x, y ).restlength[linknr])
			//		{
			//			// pull points together
			//			float extra = distance / (grid( x, y ).restlength[linknr]) - 1;
			//			float2 dir = neighbour.pos - pointpos;
			//			pointpos += extra * dir * 0.5f;
			//			neighbour.pos -= extra * dir * 0.5f;
			//		}
			//	}
			//	grid( x, y ).pos = pointpos;
			//}
			//// fixed line of points is fixed.
			//for (int x = 0; x < GRIDSIZE; x++) grid( x, 0 ).pos = grid( x, 0 ).fix;


			for (int y = 1; y < GRIDSIZE - 1; y++)
			{
				//-----------------------------------------------SSE
				__m128 pointpos2 = _mm_set_ps(upos[idx(254, y)].y, upos[idx(254, y)].x, upos[idx(1, y)].y, upos[idx(1, y)].x);
				float *p1rl = restlength[idx(1, y)];
				float *p2rl = restlength[idx(254, y)];
				float2* pointpos = (float2*)&pointpos2;
				for (int linknr = 0; linknr < 4; linknr++)
				{
					float2* neig[2] = { &upos[idx(254 + xoffset[linknr], y + yoffset[linknr])], &upos[idx(1 + xoffset[linknr], y + yoffset[linknr])] };
					__m128 restlength4 = _mm_set_ps(p2rl[linknr], p2rl[linknr], p1rl[linknr], p1rl[linknr]);
					__m128 neighbour2 = _mm_set_ps(neig[1]->y, neig[1]->x, neig[0]->y, neig[0]->x);
					float2* neighbour = (float2*)&neighbour2;
					float distance0 = length(neighbour[0]), distance1 = length(neighbour[1]);
					__m128 distance4 = _mm_set_ps(distance1, distance1, distance0, distance0);
					float cmp[4] = { fpclassify(distance0), fpclassify(distance0), fpclassify(distance1), fpclassify(distance1) };
					__m128* cmp4 = (__m128*)cmp;
					__m128 mask = _mm_and_ps(_mm_cmpgt_ps(*cmp4, _mm_set_ps1(0)), _mm_cmpgt_ps(distance4, restlength4));
					__m128 dir = _mm_and_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_div_ps(distance4, restlength4), _mm_set_ps1(1.0f)), _mm_sub_ps(neighbour2, pointpos2)), _mm_set_ps1(0.5f)), mask);
					_mm_store_ps((float*)&pointpos2, _mm_add_ps(pointpos2, dir));
					*neig[0] -= ((float2*)&dir)[0];
					*neig[1] -= ((float2*)&dir)[1];
				}
				upos[idx(1, y)] = pointpos[0];
				upos[idx(254, y)] = pointpos[1];
				//-----------------------------------------------AVX
				for (int x = 0; x < GRIDSIZE / 4 - 1; x++)
				{
					uint aidx[4] = { idx(x + 2, y), idx(x + 65,y), idx(x + 128,y), idx(x + 191,y) };
					float2* po = &upos[aidx[0]];
					float* prl[4] = { restlength[aidx[0]], restlength[aidx[1]], restlength[aidx[2]], restlength[aidx[3]] };
					__m256 pointpos4 = _mm256_setr_ps(upos[aidx[0]].x, upos[aidx[0]].y, upos[aidx[1]].x, upos[aidx[1]].y, upos[aidx[2]].x, upos[aidx[2]].y, upos[aidx[3]].x, upos[aidx[3]].y);
					float2* pointpos = (float2*)&pointpos4;
					for (int linknr = 0; linknr < 4; linknr++)
					{
						float2* pon = po + idx(xoffset[linknr], yoffset[linknr]);
						float2* neig[4] = { pon, pon + 63 ,pon + 126 ,pon + 189 };
						__m256 neighbour4 = _mm256_setr_ps((*neig[0]).x, (*neig[0]).y, (*neig[1]).x, (*neig[1]).y, (*neig[2]).x, (*neig[2]).y, (*neig[3]).x, (*neig[3]).y);
						__m256 restlength4 = _mm256_setr_ps(prl[0][linknr], prl[0][linknr], prl[1][linknr], prl[1][linknr], prl[2][linknr], prl[2][linknr], prl[3][linknr], prl[3][linknr]);
						__m256 del = _mm256_sub_ps(neighbour4, pointpos4);
						__m256 delx4 = _mm256_setr_ps(((float2*)&del)[0].x, ((float2*)&del)[0].x, ((float2*)&del)[1].x, ((float2*)&del)[1].x, ((float2*)&del)[2].x, ((float2*)&del)[2].x, ((float2*)&del)[3].x, ((float2*)&del)[3].x);
						__m256 dely4 = _mm256_setr_ps(((float2*)&del)[0].y, ((float2*)&del)[0].y, ((float2*)&del)[1].y, ((float2*)&del)[1].y, ((float2*)&del)[2].y, ((float2*)&del)[2].y, ((float2*)&del)[3].y, ((float2*)&del)[3].y);
						__m256 x24 = _mm256_mul_ps(delx4, delx4);
						__m256 y24 = _mm256_mul_ps(dely4, dely4);
						__m256 distance4 = _mm256_sqrt_ps(_mm256_add_ps(x24, y24));
						float cmp[8] = { fpclassify(((float*)&distance4)[0]), fpclassify(((float*)&distance4)[1]), fpclassify(((float*)&distance4)[2]), fpclassify(((float*)&distance4)[3]),
										 fpclassify(((float*)&distance4)[4]) ,fpclassify(((float*)&distance4)[5]) ,fpclassify(((float*)&distance4)[6]) ,fpclassify(((float*)&distance4)[7]) };
						__m256* cmp8 = (__m256*)cmp;
						__m256 mask8 = _mm256_and_ps(_mm256_cmp_ps(*cmp8, _mm256_set1_ps(0.0f), _CMP_LE_OQ), _mm256_cmp_ps(distance4, restlength4, _CMP_GT_OQ));
						__m256 extra4 = _mm256_sub_ps(_mm256_div_ps(distance4, restlength4), _mm256_set1_ps(1.0f));
						__m256 dir4 = _mm256_and_ps(mask8, _mm256_mul_ps(extra4, _mm256_mul_ps(del, _mm256_set1_ps(0.5f))));
						_mm256_store_ps((float*)&pointpos4, _mm256_add_ps(pointpos4, dir4));
						*neig[0] -= ((float2*)&dir4)[0];
						*neig[1] -= ((float2*)&dir4)[1];
						*neig[2] -= ((float2*)&dir4)[2];
						*neig[3] -= ((float2*)&dir4)[3];
					}
					upos[aidx[0]] = pointpos[0];
					upos[aidx[1]] = pointpos[1];
					upos[aidx[2]] = pointpos[2];
					upos[aidx[3]] = pointpos[3];
				}
			}
			for (int x = 0; x < GRIDSIZE; x++) grid(x, 0).pos = grid(x, 0).fix;
		}
	}
}

void Game::Tick( float a_DT )
{
	// update the simulation
	Timer tm;
	tm.reset();
	Simulation();
	float elapsed1 = tm.elapsed();

	// draw the grid
	tm.reset();
	DrawGrid();
	float elapsed2 = tm.elapsed();

	// display statistics
	char t[128];
	sprintf( t, "ye olde ruggeth cloth simulation: %5.1f ms", elapsed1 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 24, 0xffffff );
	sprintf( t, "                       rendering: %5.1f ms", elapsed2 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 14, 0xffffff );
}