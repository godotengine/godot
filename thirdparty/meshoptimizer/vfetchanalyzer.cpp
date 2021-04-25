// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

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
