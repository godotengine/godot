/*
 * PCG64 Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Copyright 2015 Robert Kern <robert.kern@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#include "pcg64.h"

extern inline void pcg_setseq_128_step_r(pcg_state_setseq_128* rng);
extern inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state);
extern inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128* rng,
				     pcg128_t initstate, pcg128_t initseq);
extern inline uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128* rng);
extern inline uint64_t
pcg_setseq_128_xsl_rr_64_boundedrand_r(pcg_state_setseq_128* rng,
				       uint64_t bound);
extern inline void pcg_setseq_128_advance_r(pcg_state_setseq_128* rng, pcg128_t delta);

/* Multi-step advance functions (jump-ahead, jump-back)
*
* The method used here is based on Brown, "Random Number Generation
* with Arbitrary Stride,", Transactions of the American Nuclear
* Society (Nov. 1994).  The algorithm is very similar to fast
* exponentiation.
*
* Even though delta is an unsigned integer, we can pass a
* signed integer to go backwards, it just goes "the long way round".
*/

#ifndef PCG_EMULATED_128BIT_MATH

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
			    pcg128_t cur_plus)
{
   pcg128_t acc_mult = 1u;
   pcg128_t acc_plus = 0u;
   while (delta > 0) {
       if (delta & 1) {
	   acc_mult *= cur_mult;
	   acc_plus = acc_plus * cur_mult + cur_plus;
       }
       cur_plus = (cur_mult + 1) * cur_plus;
       cur_mult *= cur_mult;
       delta /= 2;
   }
   return acc_mult * state + acc_plus;
}

#else

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
			    pcg128_t cur_plus)
{
   pcg128_t acc_mult = PCG_128BIT_CONSTANT(0u, 1u);
   pcg128_t acc_plus = PCG_128BIT_CONSTANT(0u, 0u);
   while ((delta.high > 0) || (delta.low > 0)) {
       if (delta.low & 1) {
	   acc_mult = _pcg128_mult(acc_mult, cur_mult);
	   acc_plus = _pcg128_add(_pcg128_mult(acc_plus, cur_mult), cur_plus);
       }
       cur_plus = _pcg128_mult(_pcg128_add(cur_mult, PCG_128BIT_CONSTANT(0u, 1u)), cur_plus);
       cur_mult = _pcg128_mult(cur_mult, cur_mult);
       delta.low >>= 1;
       delta.low += delta.high & 1;
       delta.high >>= 1;
   }
   return _pcg128_add(_pcg128_mult(acc_mult, state), acc_plus);
}

#endif
