/**************************************************************************/
/*  vertex_cache_optimizer.h                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef VERTEX_CACHE_OPTIMIZER_H
#define VERTEX_CACHE_OPTIMIZER_H

// This class is derived from
// https://www.martin.st/thesis/
// Based on Tom Forsyth's vertex cache optimizer

/*
  Copyright (C) 2008 Martin Storsjo
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.
  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:
  1. The origin of this software must not be misrepresented; you must not
	 claim that you wrote the original software. If you use this software
	 in a product, an acknowledgment in the product documentation would be
	 appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
	 misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "core/local_vector.h"
#include "core/math/math_defs.h"

#include <stdint.h>

class VertexCacheOptimizer {
	typedef uint32_t VERTEX_INDEX_TYPE;

	// The size of these data types affect the memory usage.
	typedef uint16_t SCORE_TYPE;
	typedef uint8_t ADJACENCY_TYPE;
	typedef int8_t CACHE_POS_TYPE;
	typedef int32_t TRIANGLE_INDEX_TYPE;
	typedef int32_t ARRAY_INDEX_TYPE;

	struct Constants {
		// The size of the precalculated tables.
		static const int CACHE_SCORE_TABLE_SIZE = 32;
		static const int VALENCE_SCORE_TABLE_SIZE = 32;
		static const int MAX_ADJACENCY = UINT8_MAX;
		static const int SCORE_SCALING = 7281;

		// Score function constants.
		static constexpr float CACHE_DECAY_POWER = 1.5;
		static constexpr float LAST_TRI_SCORE = 0.75;
		static constexpr float VALENCE_BOOST_SCALE = 2.0;
		static constexpr float VALENCE_BOOST_POWER = 0.5;

		// Set these to adjust the performance and result quality.
		static const int VERTEX_CACHE_SIZE = 24;
		static const int CACHE_FUNCTION_LENGTH = 32;

		static_assert(CACHE_SCORE_TABLE_SIZE >= VERTEX_CACHE_SIZE, "Vertex score table too small");
	};

	// Precalculated tables.
	SCORE_TYPE _cache_position_score[Constants::CACHE_SCORE_TABLE_SIZE];
	SCORE_TYPE _valence_score[Constants::VALENCE_SCORE_TABLE_SIZE];

	int is_added(const uint8_t *p_triangle_added, int p_x) const {
		return p_triangle_added[(p_x) >> 3] & (1 << (p_x & 7));
	}

	void set_added(uint8_t *p_triangle_added, int p_x) const {
		p_triangle_added[(p_x) >> 3] |= (1 << (p_x & 7));
	}

	// Precalculate the tables.
	void init();

	// Calculate the score for a vertex.
	SCORE_TYPE find_vertex_score(int p_num_active_tris, int p_cache_position);

	// The main reordering function.
	VERTEX_INDEX_TYPE *_reorder_indices(VERTEX_INDEX_TYPE *r_dest_indices, const VERTEX_INDEX_TYPE *p_source_indices, int p_num_triangles, int p_num_vertices);

public:
	VertexCacheOptimizer() {
		init();
	}

	bool reorder_indices(LocalVector<int> &r_indices, uint32_t p_num_triangles, uint32_t p_num_verts);
	bool reorder_indices_pool(PoolVector<int> &r_indices, uint32_t p_num_triangles, uint32_t p_num_verts);
};

#endif // VERTEX_CACHE_OPTIMIZER_H
