// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <string.h>

// This work is based on:
// Fabian Giesen. Decoding Morton codes. 2009
namespace meshopt
{

// "Insert" two 0 bits after each of the 10 low bits of x
inline unsigned int part1By2(unsigned int x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

static void computeOrder(unsigned int* result, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride)
{
	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	float minv[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxv[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* v = vertex_positions_data + i * vertex_stride_float;

		for (int j = 0; j < 3; ++j)
		{
			float vj = v[j];

			minv[j] = minv[j] > vj ? vj : minv[j];
			maxv[j] = maxv[j] < vj ? vj : maxv[j];
		}
	}

	float extent = 0.f;

	extent = (maxv[0] - minv[0]) < extent ? extent : (maxv[0] - minv[0]);
	extent = (maxv[1] - minv[1]) < extent ? extent : (maxv[1] - minv[1]);
	extent = (maxv[2] - minv[2]) < extent ? extent : (maxv[2] - minv[2]);

	float scale = extent == 0 ? 0.f : 1.f / extent;

	// generate Morton order based on the position inside a unit cube
	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* v = vertex_positions_data + i * vertex_stride_float;

		int x = int((v[0] - minv[0]) * scale * 1023.f + 0.5f);
		int y = int((v[1] - minv[1]) * scale * 1023.f + 0.5f);
		int z = int((v[2] - minv[2]) * scale * 1023.f + 0.5f);

		result[i] = part1By2(x) | (part1By2(y) << 1) | (part1By2(z) << 2);
	}
}

static void computeHistogram(unsigned int (&hist)[1024][3], const unsigned int* data, size_t count)
{
	memset(hist, 0, sizeof(hist));

	// compute 3 10-bit histograms in parallel
	for (size_t i = 0; i < count; ++i)
	{
		unsigned int id = data[i];

		hist[(id >> 0) & 1023][0]++;
		hist[(id >> 10) & 1023][1]++;
		hist[(id >> 20) & 1023][2]++;
	}

	unsigned int sumx = 0, sumy = 0, sumz = 0;

	// replace histogram data with prefix histogram sums in-place
	for (int i = 0; i < 1024; ++i)
	{
		unsigned int hx = hist[i][0], hy = hist[i][1], hz = hist[i][2];

		hist[i][0] = sumx;
		hist[i][1] = sumy;
		hist[i][2] = sumz;

		sumx += hx;
		sumy += hy;
		sumz += hz;
	}

	assert(sumx == count && sumy == count && sumz == count);
}

static void radixPass(unsigned int* destination, const unsigned int* source, const unsigned int* keys, size_t count, unsigned int (&hist)[1024][3], int pass)
{
	int bitoff = pass * 10;

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int id = (keys[source[i]] >> bitoff) & 1023;

		destination[hist[id][pass]++] = source[i];
	}
}

} // namespace meshopt

void meshopt_spatialSortRemap(unsigned int* destination, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(vertex_positions_stride > 0 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	meshopt_Allocator allocator;

	unsigned int* keys = allocator.allocate<unsigned int>(vertex_count);
	computeOrder(keys, vertex_positions, vertex_count, vertex_positions_stride);

	unsigned int hist[1024][3];
	computeHistogram(hist, keys, vertex_count);

	unsigned int* scratch = allocator.allocate<unsigned int>(vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
		destination[i] = unsigned(i);

	// 3-pass radix sort computes the resulting order into scratch
	radixPass(scratch, destination, keys, vertex_count, hist, 0);
	radixPass(destination, scratch, keys, vertex_count, hist, 1);
	radixPass(scratch, destination, keys, vertex_count, hist, 2);

	// since our remap table is mapping old=>new, we need to reverse it
	for (size_t i = 0; i < vertex_count; ++i)
		destination[scratch[i]] = unsigned(i);
}

void meshopt_spatialSortTriangles(unsigned int* destination, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride > 0 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	(void)vertex_count;

	size_t face_count = index_count / 3;
	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	meshopt_Allocator allocator;

	float* centroids = allocator.allocate<float>(face_count * 3);

	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		const float* va = vertex_positions + a * vertex_stride_float;
		const float* vb = vertex_positions + b * vertex_stride_float;
		const float* vc = vertex_positions + c * vertex_stride_float;

		centroids[i * 3 + 0] = (va[0] + vb[0] + vc[0]) / 3.f;
		centroids[i * 3 + 1] = (va[1] + vb[1] + vc[1]) / 3.f;
		centroids[i * 3 + 2] = (va[2] + vb[2] + vc[2]) / 3.f;
	}

	unsigned int* remap = allocator.allocate<unsigned int>(face_count);

	meshopt_spatialSortRemap(remap, centroids, face_count, sizeof(float) * 3);

	// support in-order remap
	if (destination == indices)
	{
		unsigned int* indices_copy = allocator.allocate<unsigned int>(index_count);
		memcpy(indices_copy, indices, index_count * sizeof(unsigned int));
		indices = indices_copy;
	}

	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		unsigned int r = remap[i];

		destination[r * 3 + 0] = a;
		destination[r * 3 + 1] = b;
		destination[r * 3 + 2] = c;
	}
}
