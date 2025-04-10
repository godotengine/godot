// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2021 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

#include "astcenc_mathlib.h"

/**
 * @brief 64-bit rotate left.
 *
 * @param val   The value to rotate.
 * @param count The rotation, in bits.
 */
static inline uint64_t rotl(uint64_t val, int count)
{
	return (val << count) | (val >> (64 - count));
}

/* See header for documentation. */
void astc::rand_init(uint64_t state[2])
{
	state[0] = 0xfaf9e171cea1ec6bULL;
	state[1] = 0xf1b318cc06af5d71ULL;
}

/* See header for documentation. */
uint64_t astc::rand(uint64_t state[2])
{
	uint64_t s0 = state[0];
	uint64_t s1 = state[1];
	uint64_t res = s0 + s1;
	s1 ^= s0;
	state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
	state[1] = rotl(s1, 37);
	return res;
}
