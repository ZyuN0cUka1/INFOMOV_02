#include "precomp.h"
#include "game.h"
#include "tmpl8math.h"

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

#define idx(x,y) ((x)+((y)<<8))
#define idx8(x,y) (((x)<<3)+((y)<<7))
#define set256(x) (_mm256_setr_ps((x),(x),(x),(x),(x),(x),(x),(x)))
#define calidx(x) (((x) >> 1) | (((x) & 0x1) << 15))
#define calidx8(x) ((((x) >> 1) | (((x) & 0x1) << 15)) >> 3)
#define revidx(x) (((x) >> 15) | (((x) & 0x7fff) << 1))
#define revidx8(x) ((((x) >> 15) | (((x) & 0x7fff) << 1)) >> 3)

static union { float posx[GRIDSIZE * GRIDSIZE]; __m256 posx8[GRIDSIZE * GRIDSIZE / 8]; };
static union { float posy[GRIDSIZE * GRIDSIZE]; __m256 posy8[GRIDSIZE * GRIDSIZE / 8]; };
static union { float prev_posx[GRIDSIZE * GRIDSIZE]; __m256 prev_posx8[GRIDSIZE * GRIDSIZE / 8]; };
static union { float prev_posy[GRIDSIZE * GRIDSIZE]; __m256 prev_posy8[GRIDSIZE * GRIDSIZE / 8]; };
static union { float fixx[GRIDSIZE * GRIDSIZE]; __m256 fixx8[GRIDSIZE * GRIDSIZE / 8]; };
static union { float fixy[GRIDSIZE * GRIDSIZE]; __m256 fixy8[GRIDSIZE * GRIDSIZE / 8]; };
static union { float urest[4][GRIDSIZE * GRIDSIZE]; __m256 restlength8[4][GRIDSIZE * GRIDSIZE / 8]; };

uint preidx = 0;

struct Pospoint {
	float& x;
	float& y;
	void operator=(const Pospoint& v) {
		x = v.x;
		y = v.y;
	}
	void operator=(const float2& v) {
		x = v.x;
		y = v.y;
	}
};

struct float_rest {
	uint i;
	float_rest(uint id) :i(id) {};
	float& operator[](uint idx) { return urest[idx][i]; }
};

struct Point
{
	Pospoint pos{ posx[calidx(preidx)],posy[calidx(preidx)] };					// current position of the point
	Pospoint prev_pos{ prev_posx[calidx(preidx)],prev_posy[calidx(preidx)] };	// position of the point in the previous frame
	Pospoint fix{ fixx[calidx(preidx)],fixy[calidx(preidx)] };					// stationary position; used for the top line of points
	bool fixed;																	// true if this is a point in the top line of the cloth
	float_rest restlength{ calidx(preidx) };									// initial distance to neighbours
	const uint id = preidx++;
};

// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid(const uint x, const uint y) { return pointGrid[idx(x, y)]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init()
{
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
			float2 rest{ grid(x, y).pos.x - grid(x + xoffset[c], y + yoffset[c]).pos.x,grid(x, y).pos.y - grid(x + xoffset[c], y + yoffset[c]).pos.y };
			urest[c][calidx(idx(x, y))] = length(rest) * 1.15f;
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
		const Pospoint p1 = grid( x, y ).pos;
		const Pospoint p2 = grid( x + 1, y ).pos;
		const Pospoint p3 = grid( x, y + 1 ).pos;
		screen->Line( p1.x, p1.y, p2.x, p2.y, 0xffffff );
		screen->Line( p1.x, p1.y, p3.x, p3.y, 0xffffff );
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		const Pospoint p1 = grid( GRIDSIZE - 2, y ).pos;
		const Pospoint p2 = grid( GRIDSIZE - 2, y + 1 ).pos;
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
__m256 gy = _mm256_set1_ps(0.003f);
__m256 xe00mask = _mm256_cmp_ps(_mm256_setr_ps(0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f), _mm256_set1_ps(0.0f), _CMP_GT_OQ);
__m256 xeffmask = _mm256_cmp_ps(_mm256_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f), _mm256_set1_ps(0.0f), _CMP_GT_OQ);
__m256 truemask = _mm256_cmp_ps(_mm256_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f), _mm256_set1_ps(0.0f), _CMP_GT_OQ);

void inline cal_dir(__m256* curx, __m256* cury, const __m256& _mask, const uint& x, const uint& y) {
	uint i0 = y * 16 + x;
	__m256 ppsx = *curx;
	__m256 ppsy = *cury;
	for (int linknr = 0; linknr < 4; linknr++)
	{
		uint i = calidx(revidx(i0 * 8) + xoffset[linknr] + yoffset[linknr] * GRIDSIZE);
		__m256& nbrx = *((__m256*) & (posx[i]));
		__m256& nbry = *((__m256*) & (posy[i]));
		__m256 delx = _mm256_sub_ps(nbrx, ppsx);
		__m256 dely = _mm256_sub_ps(nbry, ppsy);
		__m256 dist = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(delx, delx), _mm256_mul_ps(dely, dely)));
		__m256 mask0 = _mm256_cmp_ps(dist, _mm256_set1_ps(numeric_limits<float>::infinity()), _CMP_NEQ_OQ);
		//__m256 mask1 = _mm256_cmp_ps(dist, _mm256_set1_ps(numeric_limits<float>::quiet_NaN()), _CMP_NEQ_OQ);
		__m256 mask = _mm256_and_ps(_mask, _mm256_and_ps(mask0, _mm256_cmp_ps(dist, restlength8[linknr][i0], _CMP_GT_OQ)));
		//__m256 mask = _mm256_and_ps(_mask, _mm256_and_ps(_mm256_and_ps(mask0, mask1), _mm256_cmp_ps(dist, restlength8[linknr][i0], _CMP_GT_OQ)));
		__m256 extra = _mm256_sub_ps(_mm256_div_ps(dist, restlength8[linknr][i0]), _mm256_set1_ps(1.0f));
		__m256 dirx = _mm256_and_ps(mask, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(delx, extra)));
		__m256 diry = _mm256_and_ps(mask, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(dely, extra)));
		ppsx = _mm256_add_ps(ppsx, dirx);
		ppsy = _mm256_add_ps(ppsy, diry);
		nbrx = _mm256_sub_ps(nbrx, dirx);
		nbry = _mm256_sub_ps(nbry, diry);
	}
	*curx = ppsx;
	*cury = ppsy;
}

void Game::Simulation()
{
	// simulation is exected three times per frame; do not change this.
	for( int steps = 0; steps < 3; steps++ )
	{
		//------------------avx
		__m256* curx = posx8;
		__m256* cury = posy8;
		__m256* prex = prev_posx8;
		__m256* prey = prev_posy8;
		for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE / 8; x++)
		{
			__m256 cx = *curx;
			__m256 cy = *cury;
			__m256 px = *prex;
			__m256 py = *prey;

			_mm256_store_ps((float*)prex, cx);
			_mm256_store_ps((float*)prey, cy);

			*curx = _mm256_add_ps(_mm256_sub_ps(cx, px), cx);
			*cury = _mm256_add_ps(_mm256_sub_ps(cy, py), _mm256_add_ps(cy, gy));
			
			__m256 mask = _mm256_cmp_ps(set256(Rand(10)), _mm256_set1_ps(0.03f), _CMP_LT_OQ);

			*curx = _mm256_add_ps(*curx, _mm256_and_ps(mask, set256(Rand(0.02f + magic))));
			*cury = _mm256_add_ps(*cury, _mm256_and_ps(mask, set256(Rand(0.12f))));

			curx++;
			cury++;
			prex++;
			prey++;
		}
		//------------------avx

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.

		for (int i = 0; i < 4; i++)
		{
			//------------------avx
			__m256* curx = posx8 + 128 / 8;
			__m256* cury = posy8 + 128 / 8;
			for (int y = 1; y < GRIDSIZE - 1; y++)
			{
				cal_dir(curx, cury, xe00mask, 0, y);
				curx++;
				cury++;
				for (int x = 1; x < GRIDSIZE / 16; x++)
				{
					cal_dir(curx, cury, truemask, x, y);
					curx++;
					cury++;
				}
			}
			curx += 128 * 2 / 8;
			cury += 128 * 2 / 8;
			for (int y = GRIDSIZE + 1; y < GRIDSIZE * 2 - 1; y++)
			{
				for (int x = 0; x < GRIDSIZE / 16 - 1; x++)
				{
					cal_dir(curx, cury, truemask, x, y);
					curx++;
					cury++;
				}
				
				cal_dir(curx, cury, xeffmask, GRIDSIZE / 16 - 1, y);
				curx++;
				cury++;
			}
			//------------------avx
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