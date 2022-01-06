/*
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_GEN_RANDOM_H
#define BT_GEN_RANDOM_H

#ifdef MT19937

#include <limits.h>
#include <mt19937.h>

#define GEN_RAND_MAX UINT_MAX

SIMD_FORCE_INLINE void GEN_srand(unsigned int seed) { init_genrand(seed); }
SIMD_FORCE_INLINE unsigned int GEN_rand() { return genrand_int32(); }

#else

#include <stdlib.h>

#define GEN_RAND_MAX RAND_MAX

SIMD_FORCE_INLINE void GEN_srand(unsigned int seed) { srand(seed); }
SIMD_FORCE_INLINE unsigned int GEN_rand() { return rand(); }

#endif

#endif  //BT_GEN_RANDOM_H
