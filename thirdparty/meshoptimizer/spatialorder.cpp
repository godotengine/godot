// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <string.h>

// This work is based on:
// Fabian Giesen. Decoding Morton codes. 2009
namespace meshopt
{

// "Insert" two 0 bits after each of the 20 low bits of x
inline unsigned long long part1By2(unsigned long long x)
{
	x &= 0x000fffffull;                          // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- jihg fedc ba98 7654 3210
	x = (x ^ (x << 32)) & 0x000f00000000ffffull; // x = ---- ---- ---- jihg ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x << 16)) & 0x000f0000ff0000ffull; // x = ---- ---- ---- jihg ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x000f00f00f00f00full;  // x = ---- ---- ---- jihg ---- ---- fedc ---- ---- ba98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x00c30c30c30c30c3ull;  // x = ---- ---- ji-- --hg ---- fe-- --dc ---- ba-- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x0249249249249249ull;  // x = ---- --j- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

static void computeOrder(unsigned long long* result, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride, bool morton)
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

	// rescale each axis to 16 bits to get 48-bit Morton codes
	float scale = extent == 0 ? 0.f : 65535.f / extent;

	// generate Morton order based on the position inside a unit cube
	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* v = vertex_positions_data + i * vertex_stride_float;

		int x = int((v[0] - minv[0]) * scale + 0.5f);
		int y = int((v[1] - minv[1]) * scale + 0.5f);
		int z = int((v[2] - minv[2]) * scale + 0.5f);

		if (morton)
			result[i] = part1By2(x) | (part1By2(y) << 1) | (part1By2(z) << 2);
		else
			result[i] = ((unsigned long long)x << 0) | ((unsigned long long)y << 20) | ((unsigned long long)z << 40);
	}
}

static void radixSort10(unsigned int* destination, const unsigned int* source, const unsigned short* keys, size_t count)
{
	unsigned int hist[1024];
	memset(hist, 0, sizeof(hist));

	// compute histogram (assume keys are 10-bit)
	for (size_t i = 0; i < count; ++i)
		hist[keys[i]]++;

	unsigned int sum = 0;

	// replace histogram data with prefix histogram sums in-place
	for (int i = 0; i < 1024; ++i)
	{
		unsigned int h = hist[i];
		hist[i] = sum;
		sum += h;
	}

	assert(sum == count);

	// reorder values
	for (size_t i = 0; i < count; ++i)
	{
		unsigned int id = keys[source[i]];

		destination[hist[id]++] = source[i];
	}
}

static void computeHistogram(unsigned int (&hist)[256][2], const unsigned short* data, size_t count)
{
	memset(hist, 0, sizeof(hist));

	// compute 2 8-bit histograms in parallel
	for (size_t i = 0; i < count; ++i)
	{
		unsigned long long id = data[i];

		hist[(id >> 0) & 255][0]++;
		hist[(id >> 8) & 255][1]++;
	}

	unsigned int sum0 = 0, sum1 = 0;

	// replace histogram data with prefix histogram sums in-place
	for (int i = 0; i < 256; ++i)
	{
		unsigned int h0 = hist[i][0], h1 = hist[i][1];

		hist[i][0] = sum0;
		hist[i][1] = sum1;

		sum0 += h0;
		sum1 += h1;
	}

	assert(sum0 == count && sum1 == count);
}

static void radixPass(unsigned int* destination, const unsigned int* source, const unsigned short* keys, size_t count, unsigned int (&hist)[256][2], int pass)
{
	int bitoff = pass * 8;

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int id = unsigned(keys[source[i]] >> bitoff) & 255;

		destination[hist[id][pass]++] = source[i];
	}
}

static void partitionPoints(unsigned int* target, const unsigned int* order, const unsigned char* sides, size_t split, size_t count)
{
	size_t l = 0, r = split;

	for (size_t i = 0; i < count; ++i)
	{
		unsigned char side = sides[order[i]];
		target[side ? r : l] = order[i];
		l += 1;
		l -= side;
		r += side;
	}

	assert(l == split && r == count);
}

static void splitPoints(unsigned int* destination, unsigned int* orderx, unsigned int* ordery, unsigned int* orderz, const unsigned long long* keys, size_t count, void* scratch, size_t cluster_size)
{
	if (count <= cluster_size)
	{
		memcpy(destination, orderx, count * sizeof(unsigned int));
		return;
	}

	unsigned int* axes[3] = {orderx, ordery, orderz};

	int bestk = -1;
	unsigned int bestdim = 0;

	for (int k = 0; k < 3; ++k)
	{
		const unsigned int mask = (1 << 20) - 1;
		unsigned int dim = (unsigned(keys[axes[k][count - 1]] >> (k * 20)) & mask) - (unsigned(keys[axes[k][0]] >> (k * 20)) & mask);

		if (dim >= bestdim)
		{
			bestk = k;
			bestdim = dim;
		}
	}

	assert(bestk >= 0);

	// split roughly in half, with the left split always being aligned to cluster size
	size_t split = ((count / 2) + cluster_size - 1) / cluster_size * cluster_size;
	assert(split > 0 && split < count);

	// mark sides of split for partitioning
	unsigned char* sides = static_cast<unsigned char*>(scratch) + count * sizeof(unsigned int);

	for (size_t i = 0; i < split; ++i)
		sides[axes[bestk][i]] = 0;

	for (size_t i = split; i < count; ++i)
		sides[axes[bestk][i]] = 1;

	// partition all axes into two sides, maintaining order
	unsigned int* temp = static_cast<unsigned int*>(scratch);

	for (int k = 0; k < 3; ++k)
	{
		if (k == bestk)
			continue;

		unsigned int* axis = axes[k];
		memcpy(temp, axis, sizeof(unsigned int) * count);
		partitionPoints(axis, temp, sides, split, count);
	}

	splitPoints(destination, orderx, ordery, orderz, keys, split, scratch, cluster_size);
	splitPoints(destination + split, orderx + split, ordery + split, orderz + split, keys, count - split, scratch, cluster_size);
}

} // namespace meshopt

void meshopt_spatialSortRemap(unsigned int* destination, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	meshopt_Allocator allocator;

	unsigned long long* keys = allocator.allocate<unsigned long long>(vertex_count);
	computeOrder(keys, vertex_positions, vertex_count, vertex_positions_stride, /* morton= */ true);

	unsigned int* scratch = allocator.allocate<unsigned int>(vertex_count * 2); // 4b for order + 2b for keys
	unsigned short* keyk = (unsigned short*)(scratch + vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
		destination[i] = unsigned(i);

	unsigned int* order[] = {scratch, destination};

	// 5-pass radix sort computes the resulting order into scratch
	for (int k = 0; k < 5; ++k)
	{
		// copy 10-bit key segments into keyk to reduce cache pressure during radix pass
		for (size_t i = 0; i < vertex_count; ++i)
			keyk[i] = (unsigned short)((keys[i] >> (k * 10)) & 1023);

		radixSort10(order[k % 2], order[(k + 1) % 2], keyk, vertex_count);
	}

	// since our remap table is mapping old=>new, we need to reverse it
	for (size_t i = 0; i < vertex_count; ++i)
		destination[scratch[i]] = unsigned(i);
}

void meshopt_spatialSortTriangles(unsigned int* destination, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
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

void meshopt_spatialClusterPoints(unsigned int* destination, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, size_t cluster_size)
{
	using namespace meshopt;

	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);
	assert(cluster_size > 0);

	meshopt_Allocator allocator;

	unsigned long long* keys = allocator.allocate<unsigned long long>(vertex_count);
	computeOrder(keys, vertex_positions, vertex_count, vertex_positions_stride, /* morton= */ false);

	unsigned int* order = allocator.allocate<unsigned int>(vertex_count * 3);
	unsigned int* scratch = allocator.allocate<unsigned int>(vertex_count * 2); // 4b for order + 1b for side or 2b for keys
	unsigned short* keyk = reinterpret_cast<unsigned short*>(scratch + vertex_count);

	for (int k = 0; k < 3; ++k)
	{
		// copy 16-bit key segments into keyk to reduce cache pressure during radix pass
		for (size_t i = 0; i < vertex_count; ++i)
			keyk[i] = (unsigned short)(keys[i] >> (k * 20));

		unsigned int hist[256][2];
		computeHistogram(hist, keyk, vertex_count);

		for (size_t i = 0; i < vertex_count; ++i)
			order[k * vertex_count + i] = unsigned(i);

		radixPass(scratch, order + k * vertex_count, keyk, vertex_count, hist, 0);
		radixPass(order + k * vertex_count, scratch, keyk, vertex_count, hist, 1);
	}

	splitPoints(destination, order, order + vertex_count, order + 2 * vertex_count, keys, vertex_count, scratch, cluster_size);
}
