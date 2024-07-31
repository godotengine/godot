/**************************************************************************/
/*  vertex_cache_optimizer.cpp                                            */
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

#include "vertex_cache_optimizer.h"

#include "core/math/geometry.h"
#include "core/math/math_funcs.h"

// Precalculate the tables.
void VertexCacheOptimizer::init() {
	for (int i = 0; i < Constants::CACHE_SCORE_TABLE_SIZE; i++) {
		float score = 0;
		if (i < 3) {
			// This vertex was used in the last triangle,
			// so it has a fixed score, which ever of the three
			// it's in. Otherwise, you can get very different
			// answers depending on whether you add
			// the triangle 1,2,3 or 3,1,2 - which is silly.
			score = Constants::LAST_TRI_SCORE;
		} else {
			// Points for being high in the cache.
			const float scaler = 1.0f / (Constants::CACHE_FUNCTION_LENGTH - 3);
			score = 1.0f - (i - 3) * scaler;
			score = Math::pow(score, Constants::CACHE_DECAY_POWER);
		}
		_cache_position_score[i] = (SCORE_TYPE)(Constants::SCORE_SCALING * score);
	}

	for (int i = 1; i < Constants::VALENCE_SCORE_TABLE_SIZE; i++) {
		// Bonus points for having a low number of tris still to
		// use the vert, so we get rid of lone verts quickly.
		float valence_boost = Math::pow(i, -Constants::VALENCE_BOOST_POWER);
		float score = Constants::VALENCE_BOOST_SCALE * valence_boost;
		_valence_score[i] = (SCORE_TYPE)(Constants::SCORE_SCALING * score);
	}
}

VertexCacheOptimizer::SCORE_TYPE VertexCacheOptimizer::find_vertex_score(int p_num_active_tris, int p_cache_position) {
	if (p_num_active_tris == 0) {
		// No triangles need this vertex!
		return 0;
	}

	SCORE_TYPE score = 0;
	if (p_cache_position < 0) {
		// Vertex is not in LRU cache - no score.
	} else {
		score = _cache_position_score[p_cache_position];
	}

	if (p_num_active_tris < Constants::VALENCE_SCORE_TABLE_SIZE) {
		score += _valence_score[p_num_active_tris];
	}
	return score;
}

VertexCacheOptimizer::VERTEX_INDEX_TYPE *VertexCacheOptimizer::_reorder_indices(VERTEX_INDEX_TYPE *r_dest_indices, const VERTEX_INDEX_TYPE *p_source_indices, int p_num_triangles, int p_num_vertices) {
	ADJACENCY_TYPE *num_active_tris = (ADJACENCY_TYPE *)memalloc(sizeof(ADJACENCY_TYPE) * p_num_vertices);
	memset(num_active_tris, 0, sizeof(ADJACENCY_TYPE) * p_num_vertices);

	// First scan over the vertex data, count the total number of
	// occurrances of each vertex.
	for (int i = 0; i < 3 * p_num_triangles; i++) {
		if (num_active_tris[p_source_indices[i]] == Constants::MAX_ADJACENCY) {
			// Unsupported mesh,
			// vertex shared by too many triangles.
			memfree(num_active_tris);
			return nullptr;
		}
		num_active_tris[p_source_indices[i]]++;
	}

	// Allocate the rest of the arrays.
	ARRAY_INDEX_TYPE *offsets = (ARRAY_INDEX_TYPE *)memalloc(sizeof(ARRAY_INDEX_TYPE) * p_num_vertices);
	SCORE_TYPE *last_score = (SCORE_TYPE *)memalloc(sizeof(SCORE_TYPE) * p_num_vertices);
	CACHE_POS_TYPE *cache_tag = (CACHE_POS_TYPE *)memalloc(sizeof(CACHE_POS_TYPE) * p_num_vertices);

	uint8_t *triangle_added = (uint8_t *)memalloc((p_num_triangles + 7) / 8);
	SCORE_TYPE *triangle_score = (SCORE_TYPE *)memalloc(sizeof(SCORE_TYPE) * p_num_triangles);
	TRIANGLE_INDEX_TYPE *triangle_indices = (TRIANGLE_INDEX_TYPE *)memalloc(sizeof(TRIANGLE_INDEX_TYPE) * 3 * p_num_triangles);
	memset(triangle_added, 0, sizeof(uint8_t) * ((p_num_triangles + 7) / 8));
	memset(triangle_score, 0, sizeof(SCORE_TYPE) * p_num_triangles);
	memset(triangle_indices, 0, sizeof(TRIANGLE_INDEX_TYPE) * 3 * p_num_triangles);

	// Count the triangle array offset for each vertex,
	// initialize the rest of the data.
	int sum = 0;
	for (int i = 0; i < p_num_vertices; i++) {
		offsets[i] = sum;
		sum += num_active_tris[i];
		num_active_tris[i] = 0;
		cache_tag[i] = -1;
	}

	// Fill the vertex data structures with indices to the triangles
	// using each vertex.
	for (int i = 0; i < p_num_triangles; i++) {
		for (int j = 0; j < 3; j++) {
			int v = p_source_indices[3 * i + j];
			triangle_indices[offsets[v] + num_active_tris[v]] = i;
			num_active_tris[v]++;
		}
	}

	// Initialize the score for all vertices.
	for (int i = 0; i < p_num_vertices; i++) {
		last_score[i] = find_vertex_score(num_active_tris[i], cache_tag[i]);
		for (int j = 0; j < num_active_tris[i]; j++) {
			triangle_score[triangle_indices[offsets[i] + j]] += last_score[i];
		}
	}

	// Find the best triangle.
	int best_triangle = -1;
	int best_score = -1;

	for (int i = 0; i < p_num_triangles; i++) {
		if (triangle_score[i] > best_score) {
			best_score = triangle_score[i];
			best_triangle = i;
		}
	}

	// Allocate the output array.
	TRIANGLE_INDEX_TYPE *out_triangles = (TRIANGLE_INDEX_TYPE *)memalloc(sizeof(TRIANGLE_INDEX_TYPE) * p_num_triangles);
	int out_pos = 0;

	// Initialize the cache.
	int cache[Constants::VERTEX_CACHE_SIZE + 3];
	for (int i = 0; i < Constants::VERTEX_CACHE_SIZE + 3; i++) {
		cache[i] = -1;
	}

	int scan_pos = 0;

	// Output the currently best triangle, as long as there
	// are triangles left to output.
	while (best_triangle >= 0) {
		// Mark the triangle as added.
		set_added(triangle_added, best_triangle);
		// Output this triangle.
		out_triangles[out_pos++] = best_triangle;
		for (int i = 0; i < 3; i++) {
			// Update this vertex.
			int v = p_source_indices[3 * best_triangle + i];

			// Check the current cache position, if it
			// is in the cache.
			int endpos = cache_tag[v];
			if (endpos < 0) {
				endpos = Constants::VERTEX_CACHE_SIZE + i;
			}
			if (endpos > i) {
				// Move all cache entries from the previous position
				// in the cache to the new target position (i) one
				// step backwards.
				for (int j = endpos; j > i; j--) {
					cache[j] = cache[j - 1];
					// If this cache slot contains a real
					// vertex, update its cache tag.
					if (cache[j] >= 0) {
						cache_tag[cache[j]]++;
					}
				}
				// Insert the current vertex into its new target
				// slot.
				cache[i] = v;
				cache_tag[v] = i;
			}

			// Find the current triangle in the list of active
			// triangles and remove it (moving the last
			// triangle in the list to the slot of this triangle).
			for (int j = 0; j < num_active_tris[v]; j++) {
				if (triangle_indices[offsets[v] + j] == best_triangle) {
					triangle_indices[offsets[v] + j] = triangle_indices[offsets[v] + num_active_tris[v] - 1];
					break;
				}
			}
			// Shorten the list.
			num_active_tris[v]--;
		}
		// Update the scores of all triangles in the cache.
		for (int i = 0; i < Constants::VERTEX_CACHE_SIZE + 3; i++) {
			int v = cache[i];
			if (v < 0) {
				break;
			}
			// This vertex has been pushed outside of the
			// actual cache.
			if (i >= Constants::VERTEX_CACHE_SIZE) {
				cache_tag[v] = -1;
				cache[i] = -1;
			}
			SCORE_TYPE newScore = find_vertex_score(num_active_tris[v], cache_tag[v]);
			SCORE_TYPE diff = newScore - last_score[v];
			for (int j = 0; j < num_active_tris[v]; j++) {
				triangle_score[triangle_indices[offsets[v] + j]] += diff;
			}
			last_score[v] = newScore;
		}
		// Find the best triangle referenced by vertices in the cache.
		best_triangle = -1;
		best_score = -1;
		for (int i = 0; i < Constants::VERTEX_CACHE_SIZE; i++) {
			if (cache[i] < 0) {
				break;
			}
			int v = cache[i];
			for (int j = 0; j < num_active_tris[v]; j++) {
				int t = triangle_indices[offsets[v] + j];
				if (triangle_score[t] > best_score) {
					best_triangle = t;
					best_score = triangle_score[t];
				}
			}
		}
		// If no active triangle was found at all, continue
		// scanning the whole list of triangles.
		if (best_triangle < 0) {
			for (; scan_pos < p_num_triangles; scan_pos++) {
				if (!is_added(triangle_added, scan_pos)) {
					best_triangle = scan_pos;
					break;
				}
			}
		}
	}

	// Convert the triangle index array into a full triangle list.
	out_pos = 0;
	for (int i = 0; i < p_num_triangles; i++) {
		int t = out_triangles[i];
		for (int j = 0; j < 3; j++) {
			int v = p_source_indices[3 * t + j];
			r_dest_indices[out_pos++] = v;
		}
	}

	// Clean up.
	memfree(triangle_indices);
	memfree(offsets);
	memfree(last_score);
	memfree(num_active_tris);
	memfree(cache_tag);
	memfree(triangle_added);
	memfree(triangle_score);
	memfree(out_triangles);

	return r_dest_indices;
}

bool VertexCacheOptimizer::reorder_indices_pool(PoolVector<int> &r_indices, uint32_t p_num_triangles, uint32_t p_num_verts) {
	LocalVector<int> temp;
	temp = r_indices;
	if (reorder_indices(temp, p_num_triangles, p_num_verts)) {
		r_indices = temp;
		return true;
	}
	return false;
}

bool VertexCacheOptimizer::reorder_indices(LocalVector<int> &r_indices, uint32_t p_num_triangles, uint32_t p_num_verts) {
	// If the mesh contains invalid indices, abort.
	ERR_FAIL_COND_V(!Geometry::verify_indices(r_indices.ptr(), r_indices.size(), p_num_verts), false);

	LocalVector<int> temp;
	temp.resize(r_indices.size());
	if (_reorder_indices((VERTEX_INDEX_TYPE *)temp.ptr(), (VERTEX_INDEX_TYPE *)r_indices.ptr(), p_num_triangles, p_num_verts)) {
#if 0
		uint32_t show = MIN(r_indices.size(), 16);
		for (uint32_t n = 0; n < show; n++) {
			print_line(itos(n) + " : " + itos(r_indices[n]) + " to " + itos(temp[n]));
		}
#endif

		r_indices = temp;
		return true;
	}
	return false;
}
