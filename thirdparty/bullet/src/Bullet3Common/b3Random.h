/*
Copyright (c) 2003-2013 Gino van den Bergen / Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/



#ifndef B3_GEN_RANDOM_H
#define B3_GEN_RANDOM_H

#include "b3Scalar.h"

#ifdef MT19937

#include <limits.h>
#include <mt19937.h>

#define B3_RAND_MAX UINT_MAX

B3_FORCE_INLINE void         b3Srand(unsigned int seed) { init_genrand(seed); }
B3_FORCE_INLINE unsigned int b3rand()                   { return genrand_int32(); }

#else

#include <stdlib.h>

#define B3_RAND_MAX RAND_MAX

B3_FORCE_INLINE void         b3Srand(unsigned int seed) { srand(seed); } 
B3_FORCE_INLINE unsigned int b3rand()                   { return rand(); }

#endif

inline b3Scalar b3RandRange(b3Scalar minRange, b3Scalar maxRange)
{
	return (b3rand() / (b3Scalar(B3_RAND_MAX) + b3Scalar(1.0))) * (maxRange - minRange) + minRange;
}


#endif //B3_GEN_RANDOM_H

