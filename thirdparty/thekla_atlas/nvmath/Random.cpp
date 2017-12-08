// This code is in the public domain -- castanyo@yahoo.es

#include <nvmath/Random.h>
#include <time.h>

using namespace nv;

// Statics
const uint16 Rand48::a0 = 0xE66D; 
const uint16 Rand48::a1 = 0xDEEC; 
const uint16 Rand48::a2 = 0x0005;
const uint16 Rand48::c0 = 0x000B;


/// Get a random seed based on the current time.
uint Rand::randomSeed()
{
    return (uint)time(NULL);
}


void MTRand::initialize( uint32 seed )
{
    // Initialize generator state with seed
    // See Knuth TAOCP Vol 2, 3rd Ed, p.106 for multiplier.
    // In previous versions, most significant bits (MSBs) of the seed affect
    // only MSBs of the state array.  Modified 9 Jan 2002 by Makoto Matsumoto.
    uint32 *s = state;
    uint32 *r = state;
    int i = 1;
    *s++ = seed & 0xffffffffUL;
    for( ; i < N; ++i )
    {
        *s++ = ( 1812433253UL * ( *r ^ (*r >> 30) ) + i ) & 0xffffffffUL;
        r++;
    }
}


void MTRand::reload()
{
    // Generate N new values in state
    // Made clearer and faster by Matthew Bellew (matthew.bellew@home.com)
    uint32 *p = state;
    int i;
    for( i = N - M; i--; ++p )
        *p = twist( p[M], p[0], p[1] );
    for( i = M; --i; ++p )
        *p = twist( p[M-N], p[0], p[1] );
    *p = twist( p[M-N], p[0], state[0] );

    left = N, next = state;
}

