#ifndef OPEN_SIMPLEX_NOISE_H__
#define OPEN_SIMPLEX_NOISE_H__

/*
 * OpenSimplex (Simplectic) Noise in C.
 * Ported to C from Kurt Spencer's java implementation by Stephen M. Cameron
 *
 * v1.1 (October 6, 2014) 
 * - Ported to C
 * 
 * v1.1 (October 5, 2014)
 * - Added 2D and 4D implementations.
 * - Proper gradient sets for all dimensions, from a
 *   dimensionally-generalizable scheme with an actual
 *   rhyme and reason behind it.
 * - Removed default permutation array in favor of
 *   default seed.
 * - Changed seed-based constructor to be independent
 *   of any particular randomization library, so results
 *   will be the same when ported to other languages.
 */

#if ((__GNUC_STDC_INLINE__) || (__STDC_VERSION__ >= 199901L))
	#include <stdint.h>
	#define INLINE inline
#elif (defined (_MSC_VER) || defined (__GNUC_GNU_INLINE__))
	#include <stdint.h>
	#define INLINE __inline
#else 
	/* ANSI C doesn't have inline or stdint.h. */
	#define INLINE
#endif

#ifdef __cplusplus
	extern "C" {
#endif

// -- GODOT start --
// Modified to work without allocating memory, also removed some unused function. 

struct osn_context {
	int16_t perm[256];
	int16_t permGradIndex3D[256];
};

int open_simplex_noise(int64_t seed, struct osn_context *ctx);
//int open_simplex_noise_init_perm(struct osn_context *ctx, int16_t p[], int nelements);
// -- GODOT end --
void open_simplex_noise_free(struct osn_context *ctx);
double open_simplex_noise2(const struct osn_context *ctx, double x, double y);
double open_simplex_noise3(const struct osn_context *ctx, double x, double y, double z);
double open_simplex_noise4(const struct osn_context *ctx, double x, double y, double z, double w);

#ifdef __cplusplus
	}
#endif

#endif
