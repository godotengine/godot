// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

// This work is based on:
// Tom Forsyth. Linear-Speed Vertex Cache Optimisation. 2006
// Pedro Sander, Diego Nehab and Joshua Barczak. Fast Triangle Reordering for Vertex Locality and Reduced Overdraw. 2007
namespace meshopt
{

const size_t kCacheSizeMax = 16;
const size_t kValenceMax = 8;

struct VertexScoreTable
{
	float cache[1 + kCacheSizeMax];
	float live[1 + kValenceMax];
};

// Tuned to minimize the ACMR of a GPU that has a cache profile similar to NVidia and AMD
static const VertexScoreTable kVertexScoreTable = {
    {0.f, 0.779f, 0.791f, 0.789f, 0.981f, 0.843f, 0.726f, 0.847f, 0.882f, 0.867f, 0.799f, 0.642f, 0.613f, 0.600f, 0.568f, 0.372f, 0.234f},
    {0.f, 0.995f, 0.713f, 0.450f, 0.404f, 0.059f, 0.005f, 0.147f, 0.006f},
};

// Tuned to minimize the encoded index buffer size
static const VertexScoreTable kVertexScoreTableStrip = {
    {0.f, 1.000f, 1.000f, 1.000f, 0.453f, 0.561f, 0.490f, 0.459f, 0.179f, 0.526f, 0.000f, 0.227f, 0.184f, 0.490f, 0.112f, 0.050f, 0.131f},
    {0.f, 0.956f, 0.786f, 0.577f, 0.558f, 0.618f, 0.549f, 0.499f, 0.489f},
};

struct TriangleAdjacency
{
	unsigned int* counts;
	unsigned int* offsets;
	unsigned int* data;
};

static void buildTriangleAdjacency(TriangleAdjacency& adjacency, const unsigned int* indices, size_t index_count, size_t vertex_count, meshopt_Allocator& allocator)
{
	size_t face_count = index_count / 3;

	// allocate arrays
	adjacency.counts = allocator.allocate<unsigned int>(vertex_count);
	adjacency.offsets = allocator.allocate<unsigned int>(vertex_count);
	adjacency.data = allocator.allocate<unsigned int>(index_count);

	// fill triangle counts
	memset(adjacency.counts, 0, vertex_count * sizeof(unsigned int));

	for (size_t i = 0; i < index_count; ++i)
	{
		assert(indices[i] < vertex_count);

		adjacency.counts[indices[i]]++;
	}

	// fill offset table
	unsigned int offset = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		adjacency.offsets[i] = offset;
		offset += adjacency.counts[i];
	}

	assert(offset == index_count);

	// fill triangle data
	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];

		adjacency.data[adjacency.offsets[a]++] = unsigned(i);
		adjacency.data[adjacency.offsets[b]++] = unsigned(i);
		adjacency.data[adjacency.offsets[c]++] = unsigned(i);
	}

	// fix offsets that have been disturbed by the previous pass
	for (size_t i = 0; i < vertex_count; ++i)
	{
		assert(adjacency.offsets[i] >= adjacency.counts[i]);

		adjacency.offsets[i] -= adjacency.counts[i];
	}
}

static unsigned int getNextVertexDeadEnd(const unsigned int* dead_end, unsigned int& dead_end_top, unsigned int& input_cursor, const unsigned int* live_triangles, size_t vertex_count)
{
	// check dead-end stack
	while (dead_end_top)
	{
		unsigned int vertex = dead_end[--dead_end_top];

		if (live_triangles[vertex] > 0)
			return vertex;
	}

	// input order
	while (input_cursor < vertex_count)
	{
		if (live_triangles[input_cursor] > 0)
			return input_cursor;

		++input_cursor;
	}

	return ~0u;
}

static unsigned int getNextVertexNeighbour(const unsigned int* next_candidates_begin, const unsigned int* next_candidates_end, const unsigned int* live_triangles, const unsigned int* cache_timestamps, unsigned int timestamp, unsigned int cache_size)
{
	unsigned int best_candidate = ~0u;
	int best_priority = -1;

	for (const unsigned int* next_candidate = next_candidates_begin; next_candidate != next_candidates_end; ++next_candidate)
	{
		unsigned int vertex = *next_candidate;

		// otherwise we don't need to process it
		if (live_triangles[vertex] > 0)
		{
			int priority = 0;

			// will it be in cache after fanning?
			if (2 * live_triangles[vertex] + timestamp - cache_timestamps[vertex] <= cache_size)
			{
				priority = timestamp - cache_timestamps[vertex]; // position in cache
			}

			if (priority > best_priority)
			{
				best_candidate = vertex;
				best_priority = priority;
			}
		}
	}

	return best_candidate;
}

static float vertexScore(const VertexScoreTable* table, int cache_position, unsigned int live_triangles)
{
	assert(cache_position >= -1 && cache_position < int(kCacheSizeMax));

	unsigned int live_triangles_clamped = live_triangles < kValenceMax ? live_triangles : kValenceMax;

	return table->cache[1 + cache_position] + table->live[live_triangles_clamped];
}

static unsigned int getNextTriangleDeadEnd(unsigned int& input_cursor, const unsigned char* emitted_flags, size_t face_count)
{
	// input order
	while (input_cursor < face_count)
	{
		if (!emitted_flags[input_cursor])
			return input_cursor;

		++input_cursor;
	}

	return ~0u;
}

} // namespace meshopt

void meshopt_optimizeVertexCacheTable(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, const meshopt::VertexScoreTable* table)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);

	meshopt_Allocator allocator;

	// guard for empty meshes
	if (index_count == 0 || vertex_count == 0)
		return;

	// support in-place optimization
	if (destination == indices)
	{
		unsigned int* indices_copy = allocator.allocate<unsigned int>(index_count);
		memcpy(indices_copy, indices, index_count * sizeof(unsigned int));
		indices = indices_copy;
	}

	unsigned int cache_size = 16;
	assert(cache_size <= kCacheSizeMax);

	size_t face_count = index_count / 3;

	// build adjacency information
	TriangleAdjacency adjacency = {};
	buildTriangleAdjacency(adjacency, indices, index_count, vertex_count, allocator);

	// live triangle counts
	unsigned int* live_triangles = allocator.allocate<unsigned int>(vertex_count);
	memcpy(live_triangles, adjacency.counts, vertex_count * sizeof(unsigned int));

	// emitted flags
	unsigned char* emitted_flags = allocator.allocate<unsigned char>(face_count);
	memset(emitted_flags, 0, face_count);

	// compute initial vertex scores
	float* vertex_scores = allocator.allocate<float>(vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
		vertex_scores[i] = vertexScore(table, -1, live_triangles[i]);

	// compute triangle scores
	float* triangle_scores = allocator.allocate<float>(face_count);

	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0];
		unsigned int b = indices[i * 3 + 1];
		unsigned int c = indices[i * 3 + 2];

		triangle_scores[i] = vertex_scores[a] + vertex_scores[b] + vertex_scores[c];
	}

	unsigned int cache_holder[2 * (kCacheSizeMax + 3)];
	unsigned int* cache = cache_holder;
	unsigned int* cache_new = cache_holder + kCacheSizeMax + 3;
	size_t cache_count = 0;

	unsigned int current_triangle = 0;
	unsigned int input_cursor = 1;

	unsigned int output_triangle = 0;

	while (current_triangle != ~0u)
	{
		assert(output_triangle < face_count);

		unsigned int a = indices[current_triangle * 3 + 0];
		unsigned int b = indices[current_triangle * 3 + 1];
		unsigned int c = indices[current_triangle * 3 + 2];

		// output indices
		destination[output_triangle * 3 + 0] = a;
		destination[output_triangle * 3 + 1] = b;
		destination[output_triangle * 3 + 2] = c;
		output_triangle++;

		// update emitted flags
		emitted_flags[current_triangle] = true;
		triangle_scores[current_triangle] = 0;

		// new triangle
		size_t cache_write = 0;
		cache_new[cache_write++] = a;
		cache_new[cache_write++] = b;
		cache_new[cache_write++] = c;

		// old triangles
		for (size_t i = 0; i < cache_count; ++i)
		{
			unsigned int index = cache[i];

			if (index != a && index != b && index != c)
			{
				cache_new[cache_write++] = index;
			}
		}

		unsigned int* cache_temp = cache;
		cache = cache_new, cache_new = cache_temp;
		cache_count = cache_write > cache_size ? cache_size : cache_write;

		// update live triangle counts
		live_triangles[a]--;
		live_triangles[b]--;
		live_triangles[c]--;

		// remove emitted triangle from adjacency data
		// this makes sure that we spend less time traversing these lists on subsequent iterations
		for (size_t k = 0; k < 3; ++k)
		{
			unsigned int index = indices[current_triangle * 3 + k];

			unsigned int* neighbours = &adjacency.data[0] + adjacency.offsets[index];
			size_t neighbours_size = adjacency.counts[index];

			for (size_t i = 0; i < neighbours_size; ++i)
			{
				unsigned int tri = neighbours[i];

				if (tri == current_triangle)
				{
					neighbours[i] = neighbours[neighbours_size - 1];
					adjacency.counts[index]--;
					break;
				}
			}
		}

		unsigned int best_triangle = ~0u;
		float best_score = 0;

		// update cache positions, vertex scores and triangle scores, and find next best triangle
		for (size_t i = 0; i < cache_write; ++i)
		{
			unsigned int index = cache[i];

			int cache_position = i >= cache_size ? -1 : int(i);

			// update vertex score
			float score = vertexScore(table, cache_position, live_triangles[index]);
			float score_diff = score - vertex_scores[index];

			vertex_scores[index] = score;

			// update scores of vertex triangles
			const unsigned int* neighbours_begin = &adjacency.data[0] + adjacency.offsets[index];
			const unsigned int* neighbours_end = neighbours_begin + adjacency.counts[index];

			for (const unsigned int* it = neighbours_begin; it != neighbours_end; ++it)
			{
				unsigned int tri = *it;
				assert(!emitted_flags[tri]);

				float tri_score = triangle_scores[tri] + score_diff;
				assert(tri_score > 0);

				if (best_score < tri_score)
				{
					best_triangle = tri;
					best_score = tri_score;
				}

				triangle_scores[tri] = tri_score;
			}
		}

		// step through input triangles in order if we hit a dead-end
		current_triangle = best_triangle;

		if (current_triangle == ~0u)
		{
			current_triangle = getNextTriangleDeadEnd(input_cursor, &emitted_flags[0], face_count);
		}
	}

	assert(input_cursor == face_count);
	assert(output_triangle == face_count);
}

void meshopt_optimizeVertexCache(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count)
{
	meshopt_optimizeVertexCacheTable(destination, indices, index_count, vertex_count, &meshopt::kVertexScoreTable);
}

void meshopt_optimizeVertexCacheStrip(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count)
{
	meshopt_optimizeVertexCacheTable(destination, indices, index_count, vertex_count, &meshopt::kVertexScoreTableStrip);
}

void meshopt_optimizeVertexCacheFifo(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, unsigned int cache_size)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(cache_size >= 3);

	meshopt_Allocator allocator;

	// guard for empty meshes
	if (index_count == 0 || vertex_count == 0)
		return;

	// support in-place optimization
	if (destination == indices)
	{
		unsigned int* indices_copy = allocator.allocate<unsigned int>(index_count);
		memcpy(indices_copy, indices, index_count * sizeof(unsigned int));
		indices = indices_copy;
	}

	size_t face_count = index_count / 3;

	// build adjacency information
	TriangleAdjacency adjacency = {};
	buildTriangleAdjacency(adjacency, indices, index_count, vertex_count, allocator);

	// live triangle counts
	unsigned int* live_triangles = allocator.allocate<unsigned int>(vertex_count);
	memcpy(live_triangles, adjacency.counts, vertex_count * sizeof(unsigned int));

	// cache time stamps
	unsigned int* cache_timestamps = allocator.allocate<unsigned int>(vertex_count);
	memset(cache_timestamps, 0, vertex_count * sizeof(unsigned int));

	// dead-end stack
	unsigned int* dead_end = allocator.allocate<unsigned int>(index_count);
	unsigned int dead_end_top = 0;

	// emitted flags
	unsigned char* emitted_flags = allocator.allocate<unsigned char>(face_count);
	memset(emitted_flags, 0, face_count);

	unsigned int current_vertex = 0;

	unsigned int timestamp = cache_size + 1;
	unsigned int input_cursor = 1; // vertex to restart from in case of dead-end

	unsigned int output_triangle = 0;

	while (current_vertex != ~0u)
	{
		const unsigned int* next_candidates_begin = &dead_end[0] + dead_end_top;

		// emit all vertex neighbours
		const unsigned int* neighbours_begin = &adjacency.data[0] + adjacency.offsets[current_vertex];
		const unsigned int* neighbours_end = neighbours_begin + adjacency.counts[current_vertex];

		for (const unsigned int* it = neighbours_begin; it != neighbours_end; ++it)
		{
			unsigned int triangle = *it;

			if (!emitted_flags[triangle])
			{
				unsigned int a = indices[triangle * 3 + 0], b = indices[triangle * 3 + 1], c = indices[triangle * 3 + 2];

				// output indices
				destination[output_triangle * 3 + 0] = a;
				destination[output_triangle * 3 + 1] = b;
				destination[output_triangle * 3 + 2] = c;
				output_triangle++;

				// update dead-end stack
				dead_end[dead_end_top + 0] = a;
				dead_end[dead_end_top + 1] = b;
				dead_end[dead_end_top + 2] = c;
				dead_end_top += 3;

				// update live triangle counts
				live_triangles[a]--;
				live_triangles[b]--;
				live_triangles[c]--;

				// update cache info
				// if vertex is not in cache, put it in cache
				if (timestamp - cache_timestamps[a] > cache_size)
					cache_timestamps[a] = timestamp++;

				if (timestamp - cache_timestamps[b] > cache_size)
					cache_timestamps[b] = timestamp++;

				if (timestamp - cache_timestamps[c] > cache_size)
					cache_timestamps[c] = timestamp++;

				// update emitted flags
				emitted_flags[triangle] = true;
			}
		}

		// next candidates are the ones we pushed to dead-end stack just now
		const unsigned int* next_candidates_end = &dead_end[0] + dead_end_top;

		// get next vertex
		current_vertex = getNextVertexNeighbour(next_candidates_begin, next_candidates_end, &live_triangles[0], &cache_timestamps[0], timestamp, cache_size);

		if (current_vertex == ~0u)
		{
			current_vertex = getNextVertexDeadEnd(&dead_end[0], dead_end_top, input_cursor, &live_triangles[0], vertex_count);
		}
	}

	assert(output_triangle == face_count);
}
