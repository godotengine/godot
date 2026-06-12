// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <math.h>
#include <string.h>

// This work is based on:
// Pedro Sander, Diego Nehab and Joshua Barczak. Fast Triangle Reordering for Vertex Locality and Reduced Overdraw. 2007
namespace meshopt
{

static void calculateSortData(float* sort_data, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, const unsigned int* clusters, size_t cluster_count)
{
	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	float mesh_centroid[3] = {};

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* p = vertex_positions + vertex_stride_float * i;

		mesh_centroid[0] += p[0];
		mesh_centroid[1] += p[1];
		mesh_centroid[2] += p[2];
	}

	mesh_centroid[0] /= float(vertex_count);
	mesh_centroid[1] /= float(vertex_count);
	mesh_centroid[2] /= float(vertex_count);

	for (size_t cluster = 0; cluster < cluster_count; ++cluster)
	{
		size_t cluster_begin = clusters[cluster] * 3;
		size_t cluster_end = (cluster + 1 < cluster_count) ? clusters[cluster + 1] * 3 : index_count;
		assert(cluster_begin < cluster_end);

		float cluster_area = 0;
		float cluster_centroid[3] = {};
		float cluster_normal[3] = {};

		for (size_t i = cluster_begin; i < cluster_end; i += 3)
		{
			const float* p0 = vertex_positions + vertex_stride_float * indices[i + 0];
			const float* p1 = vertex_positions + vertex_stride_float * indices[i + 1];
			const float* p2 = vertex_positions + vertex_stride_float * indices[i + 2];

			float p10[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
			float p20[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

			float normalx = p10[1] * p20[2] - p10[2] * p20[1];
			float normaly = p10[2] * p20[0] - p10[0] * p20[2];
			float normalz = p10[0] * p20[1] - p10[1] * p20[0];

			float area = sqrtf(normalx * normalx + normaly * normaly + normalz * normalz);

			cluster_centroid[0] += (p0[0] + p1[0] + p2[0]) * (area / 3);
			cluster_centroid[1] += (p0[1] + p1[1] + p2[1]) * (area / 3);
			cluster_centroid[2] += (p0[2] + p1[2] + p2[2]) * (area / 3);
			cluster_normal[0] += normalx;
			cluster_normal[1] += normaly;
			cluster_normal[2] += normalz;
			cluster_area += area;
		}

		float inv_cluster_area = cluster_area == 0 ? 0 : 1 / cluster_area;

		cluster_centroid[0] *= inv_cluster_area;
		cluster_centroid[1] *= inv_cluster_area;
		cluster_centroid[2] *= inv_cluster_area;

		float cluster_normal_length = sqrtf(cluster_normal[0] * cluster_normal[0] + cluster_normal[1] * cluster_normal[1] + cluster_normal[2] * cluster_normal[2]);
		float inv_cluster_normal_length = cluster_normal_length == 0 ? 0 : 1 / cluster_normal_length;

		cluster_normal[0] *= inv_cluster_normal_length;
		cluster_normal[1] *= inv_cluster_normal_length;
		cluster_normal[2] *= inv_cluster_normal_length;

		float centroid_vector[3] = {cluster_centroid[0] - mesh_centroid[0], cluster_centroid[1] - mesh_centroid[1], cluster_centroid[2] - mesh_centroid[2]};

		sort_data[cluster] = centroid_vector[0] * cluster_normal[0] + centroid_vector[1] * cluster_normal[1] + centroid_vector[2] * cluster_normal[2];
	}
}

static void calculateSortOrderRadix(unsigned int* sort_order, const float* sort_data, unsigned short* sort_keys, size_t cluster_count)
{
	// compute sort data bounds and renormalize, using fixed point snorm
	float sort_data_max = 1e-3f;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		float dpa = fabsf(sort_data[i]);

		sort_data_max = (sort_data_max < dpa) ? dpa : sort_data_max;
	}

	const int sort_bits = 11;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		// note that we flip distribution since high dot product should come first
		float sort_key = 0.5f - 0.5f * (sort_data[i] / sort_data_max);

		sort_keys[i] = meshopt_quantizeUnorm(sort_key, sort_bits) & ((1 << sort_bits) - 1);
	}

	// fill histogram for counting sort
	unsigned int histogram[1 << sort_bits];
	memset(histogram, 0, sizeof(histogram));

	for (size_t i = 0; i < cluster_count; ++i)
	{
		histogram[sort_keys[i]]++;
	}

	// compute offsets based on histogram data
	size_t histogram_sum = 0;

	for (size_t i = 0; i < 1 << sort_bits; ++i)
	{
		size_t count = histogram[i];
		histogram[i] = unsigned(histogram_sum);
		histogram_sum += count;
	}

	assert(histogram_sum == cluster_count);

	// compute sort order based on offsets
	for (size_t i = 0; i < cluster_count; ++i)
	{
		sort_order[histogram[sort_keys[i]]++] = unsigned(i);
	}
}

static unsigned int updateCache(unsigned int a, unsigned int b, unsigned int c, unsigned int cache_size, unsigned int* cache_timestamps, unsigned int& timestamp)
{
	unsigned int cache_misses = 0;

	// if vertex is not in cache, put it in cache
	if (timestamp - cache_timestamps[a] > cache_size)
	{
		cache_timestamps[a] = timestamp++;
		cache_misses++;
	}

	if (timestamp - cache_timestamps[b] > cache_size)
	{
		cache_timestamps[b] = timestamp++;
		cache_misses++;
	}

	if (timestamp - cache_timestamps[c] > cache_size)
	{
		cache_timestamps[c] = timestamp++;
		cache_misses++;
	}

	return cache_misses;
}

static size_t generateHardBoundaries(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, unsigned int cache_size, unsigned int* cache_timestamps)
{
	memset(cache_timestamps, 0, vertex_count * sizeof(unsigned int));

	unsigned int timestamp = cache_size + 1;

	size_t face_count = index_count / 3;

	size_t result = 0;

	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int m = updateCache(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], cache_size, &cache_timestamps[0], timestamp);

		// when all three vertices are not in the cache it's usually relatively safe to assume that this is a new patch in the mesh
		// that is disjoint from previous vertices; sometimes it might come back to reference existing vertices but that frequently
		// suggests an inefficiency in the vertex cache optimization algorithm
		// usually the first triangle has 3 misses unless it's degenerate - thus we make sure the first cluster always starts with 0
		if (i == 0 || m == 3)
		{
			destination[result++] = unsigned(i);
		}
	}

	assert(result <= index_count / 3);

	return result;
}

static size_t generateSoftBoundaries(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, const unsigned int* clusters, size_t cluster_count, unsigned int cache_size, float threshold, unsigned int* cache_timestamps)
{
	memset(cache_timestamps, 0, vertex_count * sizeof(unsigned int));

	unsigned int timestamp = 0;

	size_t result = 0;

	for (size_t it = 0; it < cluster_count; ++it)
	{
		size_t start = clusters[it];
		size_t end = (it + 1 < cluster_count) ? clusters[it + 1] : index_count / 3;
		assert(start < end);

		// reset cache
		timestamp += cache_size + 1;

		// measure cluster ACMR
		unsigned int cluster_misses = 0;

		for (size_t i = start; i < end; ++i)
		{
			unsigned int m = updateCache(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], cache_size, &cache_timestamps[0], timestamp);

			cluster_misses += m;
		}

		float cluster_threshold = threshold * (float(cluster_misses) / float(end - start));

		// first cluster always starts from the hard cluster boundary
		destination[result++] = unsigned(start);

		// reset cache
		timestamp += cache_size + 1;

		unsigned int running_misses = 0;
		unsigned int running_faces = 0;

		for (size_t i = start; i < end; ++i)
		{
			unsigned int m = updateCache(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], cache_size, &cache_timestamps[0], timestamp);

			running_misses += m;
			running_faces += 1;

			if (float(running_misses) / float(running_faces) <= cluster_threshold)
			{
				// we have reached the target ACMR with the current triangle so we need to start a new cluster on the next one
				// note that this may mean that we add 'end` to destination for the last triangle, which will imply that the last
				// cluster is empty; however, the 'pop_back' after the loop will clean it up
				destination[result++] = unsigned(i + 1);

				// reset cache
				timestamp += cache_size + 1;

				running_misses = 0;
				running_faces = 0;
			}
		}

		// each time we reach the target ACMR we flush the cluster
		// this means that the last cluster is by definition not very good - there are frequent cases where we are left with a few triangles
		// in the last cluster, producing a very bad ACMR and significantly penalizing the overall results
		// thus we remove the last cluster boundary, merging the last complete cluster with the last incomplete one
		// there are sometimes cases when the last cluster is actually good enough - in which case the code above would have added 'end'
		// to the cluster boundary array which we need to remove anyway - this code will do that automatically
		if (destination[result - 1] != start)
		{
			result--;
		}
	}

	assert(result >= cluster_count);
	assert(result <= index_count / 3);

	return result;
}

} // namespace meshopt

void meshopt_optimizeOverdraw(unsigned int* destination, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, float threshold)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

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

	unsigned int* cache_timestamps = allocator.allocate<unsigned int>(vertex_count);

	// generate hard boundaries from full-triangle cache misses
	unsigned int* hard_clusters = allocator.allocate<unsigned int>(index_count / 3);
	size_t hard_cluster_count = generateHardBoundaries(hard_clusters, indices, index_count, vertex_count, cache_size, cache_timestamps);

	// generate soft boundaries
	unsigned int* soft_clusters = allocator.allocate<unsigned int>(index_count / 3 + 1);
	size_t soft_cluster_count = generateSoftBoundaries(soft_clusters, indices, index_count, vertex_count, hard_clusters, hard_cluster_count, cache_size, threshold, cache_timestamps);

	const unsigned int* clusters = soft_clusters;
	size_t cluster_count = soft_cluster_count;

	// fill sort data
	float* sort_data = allocator.allocate<float>(cluster_count);
	calculateSortData(sort_data, indices, index_count, vertex_positions, vertex_count, vertex_positions_stride, clusters, cluster_count);

	// sort clusters using sort data
	unsigned short* sort_keys = allocator.allocate<unsigned short>(cluster_count);
	unsigned int* sort_order = allocator.allocate<unsigned int>(cluster_count);
	calculateSortOrderRadix(sort_order, sort_data, sort_keys, cluster_count);

	// fill output buffer
	size_t offset = 0;

	for (size_t it = 0; it < cluster_count; ++it)
	{
		unsigned int cluster = sort_order[it];
		assert(cluster < cluster_count);

		size_t cluster_begin = clusters[cluster] * 3;
		size_t cluster_end = (cluster + 1 < cluster_count) ? clusters[cluster + 1] * 3 : index_count;
		assert(cluster_begin < cluster_end);

		memcpy(destination + offset, indices + cluster_begin, (cluster_end - cluster_begin) * sizeof(unsigned int));
		offset += cluster_end - cluster_begin;
	}

	assert(offset == index_count);
}
