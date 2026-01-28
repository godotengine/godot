// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

meshopt_VertexCacheStatistics meshopt_analyzeVertexCache(const unsigned int* indices, size_t index_count, size_t vertex_count, unsigned int cache_size, unsigned int warp_size, unsigned int primgroup_size)
{
	assert(index_count % 3 == 0);
	assert(cache_size >= 3);
	assert(warp_size == 0 || warp_size >= 3);

	meshopt_Allocator allocator;

	meshopt_VertexCacheStatistics result = {};

	unsigned int warp_offset = 0;
	unsigned int primgroup_offset = 0;

	unsigned int* cache_timestamps = allocator.allocate<unsigned int>(vertex_count);
	memset(cache_timestamps, 0, vertex_count * sizeof(unsigned int));

	unsigned int timestamp = cache_size + 1;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		bool ac = (timestamp - cache_timestamps[a]) > cache_size;
		bool bc = (timestamp - cache_timestamps[b]) > cache_size;
		bool cc = (timestamp - cache_timestamps[c]) > cache_size;

		// flush cache if triangle doesn't fit into warp or into the primitive buffer
		if ((primgroup_size && primgroup_offset == primgroup_size) || (warp_size && warp_offset + ac + bc + cc > warp_size))
		{
			result.warps_executed += warp_offset > 0;

			warp_offset = 0;
			primgroup_offset = 0;

			// reset cache
			timestamp += cache_size + 1;
		}

		// update cache and add vertices to warp
		for (int j = 0; j < 3; ++j)
		{
			unsigned int index = indices[i + j];

			if (timestamp - cache_timestamps[index] > cache_size)
			{
				cache_timestamps[index] = timestamp++;
				result.vertices_transformed++;
				warp_offset++;
			}
		}

		primgroup_offset++;
	}

	size_t unique_vertex_count = 0;

	for (size_t i = 0; i < vertex_count; ++i)
		unique_vertex_count += cache_timestamps[i] > 0;

	result.warps_executed += warp_offset > 0;

	result.acmr = index_count == 0 ? 0 : float(result.vertices_transformed) / float(index_count / 3);
	result.atvr = unique_vertex_count == 0 ? 0 : float(result.vertices_transformed) / float(unique_vertex_count);

	return result;
}

meshopt_VertexFetchStatistics meshopt_analyzeVertexFetch(const unsigned int* indices, size_t index_count, size_t vertex_count, size_t vertex_size)
{
	assert(index_count % 3 == 0);
	assert(vertex_size > 0 && vertex_size <= 256);

	meshopt_Allocator allocator;

	meshopt_VertexFetchStatistics result = {};

	unsigned char* vertex_visited = allocator.allocate<unsigned char>(vertex_count);
	memset(vertex_visited, 0, vertex_count);

	const size_t kCacheLine = 64;
	const size_t kCacheSize = 128 * 1024;

	// simple direct mapped cache; on typical mesh data this is close to 4-way cache, and this model is a gross approximation anyway
	size_t cache[kCacheSize / kCacheLine] = {};

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		vertex_visited[index] = 1;

		size_t start_address = index * vertex_size;
		size_t end_address = start_address + vertex_size;

		size_t start_tag = start_address / kCacheLine;
		size_t end_tag = (end_address + kCacheLine - 1) / kCacheLine;

		assert(start_tag < end_tag);

		for (size_t tag = start_tag; tag < end_tag; ++tag)
		{
			size_t line = tag % (sizeof(cache) / sizeof(cache[0]));

			// we store +1 since cache is filled with 0 by default
			result.bytes_fetched += (cache[line] != tag + 1) * kCacheLine;
			cache[line] = tag + 1;
		}
	}

	size_t unique_vertex_count = 0;

	for (size_t i = 0; i < vertex_count; ++i)
		unique_vertex_count += vertex_visited[i];

	result.overfetch = unique_vertex_count == 0 ? 0 : float(result.bytes_fetched) / float(unique_vertex_count * vertex_size);

	return result;
}
