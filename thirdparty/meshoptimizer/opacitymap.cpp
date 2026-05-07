// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <math.h>
#include <string.h>

namespace meshopt
{

// opacity micromaps use a "bird" traversal order which recursively subdivides the triangles:
// https://docs.vulkan.org/spec/latest/_images/micromap-subd.svg
// note that triangles 0 and 2 have the same winding as the source triangle, however triangles 1 (flipped)
// and 3 (upright) have flipped winding; this is obvious from the level 2 subdivision in the diagram above
inline size_t getLevelSize(int level, int states)
{
	// 1-bit 2-state or 2-bit 4-state per micro triangle, rounded up to whole bytes
	return ((1 << (level * 2)) * (states >> 1) + 7) >> 3;
}

struct Texture
{
	const unsigned char* data;
	size_t stride, pitch;
	unsigned int width, height;
	float widthf, heightf; // width * 256.f, height * 256.f
};

static float sampleTexture(const Texture& texture, float u, float v)
{
	// wrap texture coordinates; floor is expensive so only call it if we're outside of [0, 1] range (+eps)
	u = fabsf(u - 0.5f) > 0.5f ? u - floorf(u) : u;
	v = fabsf(v - 0.5f) > 0.5f ? v - floorf(v) : v;

	// convert from [0, 1] to 16.8 fixed point coordinates (rounded to nearest subpixel) with texel centers on integer grid
	int uf = int(u * texture.widthf - 127.5f);
	int vf = int(v * texture.heightf - 127.5f);

	// clamp to avoid extrapolation past left/top edge since we don't wrap across the edge
	uf = uf < 0 ? 0 : uf;
	vf = vf < 0 ? 0 : vf;

	// x/y are texel coordinates, rx/ry are subpixel offsets
	int x = uf >> 8;
	int y = vf >> 8;
	int rx = uf & 255;
	int ry = vf & 255;

	// safeguard: this should not happen but if it ever does, ensure the accesses are inbounds
	if (unsigned(x) >= texture.width || unsigned(y) >= texture.height)
		return 0.f;

	// clamp the offsets instead of wrapping for simplicity and performance
	size_t offset = size_t(y) * texture.pitch + x * texture.stride;
	size_t offsetx = (x + 1 < int(texture.width)) ? texture.stride : 0;
	size_t offsety = (y + 1 < int(texture.height)) ? texture.pitch : 0;

	unsigned char a00 = texture.data[offset];
	unsigned char a10 = texture.data[offset + offsetx];
	unsigned char a01 = texture.data[offset + offsety];
	unsigned char a11 = texture.data[offset + offsetx + offsety];

	// bilinear interpolation in integer space: result is 8.16 fixed point
	int ax0 = a00 * 256 + (a10 - a00) * rx;
	int ax1 = a01 * 256 + (a11 - a01) * rx;
	int axy = ax0 * 256 + (ax1 - ax0) * ry;

	return float(axy) * (1.f / (255.f * 65536.f));
}

static unsigned int hashUpdate4u(unsigned int h, const unsigned char* key, size_t len)
{
	// MurmurHash2
	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	while (len >= 4)
	{
		unsigned int k;
		memcpy(&k, key, sizeof(k));

		k *= m;
		k ^= k >> r;
		k *= m;

		h *= m;
		h ^= k;

		key += 4;
		len -= 4;
	}

	return h;
}

struct TriangleOMM
{
	int uvs[6];
	int level;
};

struct TriangleOMMHasher
{
	const TriangleOMM* data;

	size_t hash(unsigned int index) const
	{
		const TriangleOMM& tri = data[index];

		return hashUpdate4u(tri.level, reinterpret_cast<const unsigned char*>(tri.uvs), sizeof(tri.uvs));
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		const TriangleOMM& lt = data[lhs];
		const TriangleOMM& rt = data[rhs];

		return lt.level == rt.level && memcmp(lt.uvs, rt.uvs, sizeof(lt.uvs)) == 0;
	}
};

struct OMMHasher
{
	const unsigned char* data;
	const unsigned int* offsets;
	const unsigned char* levels;
	int states;

	size_t hash(unsigned int index) const
	{
		const unsigned char* key = data + offsets[index];
		size_t size = getLevelSize(levels[index], states);

		unsigned int h = levels[index];

		// MurmurHash2 for large keys, simple fold for small; note that size is a power of two
		if (size < 4)
			h ^= key[0] | (key[size - 1] << 8);
		else
			h = hashUpdate4u(h, key, size);

		// MurmurHash2 finalizer
		h ^= h >> 13;
		h *= 0x5bd1e995;
		h ^= h >> 15;
		return h;
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		size_t size = getLevelSize(levels[lhs], states);

		return levels[lhs] == levels[rhs] && memcmp(data + offsets[lhs], data + offsets[rhs], size) == 0;
	}
};

static size_t hashBuckets3(size_t count)
{
	size_t buckets = 1;
	while (buckets < count + count / 4)
		buckets *= 2;

	return buckets;
}

template <typename T, typename Hash>
static T* hashLookup3(T* table, size_t buckets, const Hash& hash, const T& key, const T& empty)
{
	assert(buckets > 0);
	assert((buckets & (buckets - 1)) == 0);

	size_t hashmod = buckets - 1;
	size_t bucket = hash.hash(key) & hashmod;

	for (size_t probe = 0; probe <= hashmod; ++probe)
	{
		T& item = table[bucket];

		if (item == empty)
			return &item;

		if (hash.equal(item, key))
			return &item;

		// hash collision, quadratic probing
		bucket = (bucket + probe + 1) & hashmod;
	}

	assert(false && "Hash table is full"); // unreachable
	return NULL;
}

inline int quantizeSubpixel(float v, unsigned int size)
{
	return int(v * float(int(size) * 4) + (v >= 0 ? 0.5f : -0.5f));
}

static int rasterizeEdge(float u0, float v0, float u1, float v1, int edgeres, const Texture& texture)
{
	float edgestep = 1.f / float(edgeres + 1);

	float ud = (u1 - u0) * edgestep, vd = (v1 - v0) * edgestep;
	float u = u0, v = v0;

	int mask = 0;
	int count = 0;

	for (int i = 0; i < edgeres; ++i)
	{
		u += ud;
		v += vd;

		float a = sampleTexture(texture, u, v);
		mask |= (a >= 0.5f) << i;
		count += a >= 0.5f;
	}

	return mask | (count << 16);
}

template <int States>
static void rasterizeOpacity0(unsigned char* result, size_t index, float a0, float a1, float a2, float ac, int e0, int e1, int e2, int edgeres)
{
	int states = States;

	// basic coverage estimator from center and corner values; trained to minimize error
	float coverage = (a0 + a1 + a2) * 0.12f + ac * 0.64f;

	if (edgeres)
	{
		float edgescale = 1.f / edgeres;

		// if we have edge samples, we can get a better coverage estimate by including them; trained to minimize error
		coverage = ac * 0.22f + float((e0 >> 16) + (e1 >> 16) + (e2 >> 16)) * edgescale * 0.23f + (a0 + a1 + a2) * 0.03f;
	}

	if (states == 2)
	{
		result[index / 8] |= (coverage >= 0.5f) << (index % 8);
		return;
	}

	int transp = (a0 < 0.5f) & (a1 < 0.5f) & (a2 < 0.5f) & (ac < 0.5f);
	int opaque = (a0 > 0.5f) & (a1 > 0.5f) & (a2 > 0.5f) & (ac > 0.5f);

	// treat state as known if thresholding of corners & centers against wider bounds is consistent
	// for unknown states, we currently use the same formula as the 2-state opacity for better consistency with forced 2-state
	int unknown = 2 + (coverage >= 0.5f);
	int state = (transp | opaque) ? opaque : unknown;

	if (edgeres && (transp | opaque))
	{
		// if we have edge samples, ensure they are consistent too, falling back to unknown if not
		int exp = opaque ? (1 << edgeres) - 1 : 0;
		int eok = ((e0 & 0xffff) == exp) & ((e1 & 0xffff) == exp) & ((e2 & 0xffff) == exp);

		state = eok ? state : unknown;
	}

	result[index / 4] |= state << ((index % 4) * 2);
}

template <int States>
static void rasterizeOpacity1(unsigned char* result, size_t index, int edgeres, const float* c0, const float* c1, const float* c2, const Texture& texture)
{
	// compute each edge midpoint & sample
	float c01[3] = {(c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2, 0.f};
	float c12[3] = {(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, 0.f};
	float c20[3] = {(c2[0] + c0[0]) / 2, (c2[1] + c0[1]) / 2, 0.f};

	c01[2] = sampleTexture(texture, c01[0], c01[1]);
	c12[2] = sampleTexture(texture, c12[0], c12[1]);
	c20[2] = sampleTexture(texture, c20[0], c20[1]);

	// corner tables for each edge, and corner + edge tables for each triangle
	// edges are numbered counter clockwise, 6 outer first, 3 inner last; triangle vertex and edge references are in triangle winding order
	static const unsigned char edges[9][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {5, 1}, {1, 3}, {3, 5}};
	static const unsigned char triangles[4][6] = {{0, 1, 5, 0, 6, 5}, {5, 3, 1, 8, 7, 6}, {1, 2, 3, 1, 2, 7}, {3, 5, 4, 8, 4, 3}};

	const float* points[] = {c0, c01, c1, c12, c2, c20};

	int em[9] = {};

	// sample additional points on the edges to improve state estimation
	if (edgeres > 0)
		for (size_t i = 0; i < 9; ++i)
			em[i] = rasterizeEdge(points[edges[i][0]][0], points[edges[i][0]][1], points[edges[i][1]][0], points[edges[i][1]][1], edgeres, texture);

	for (size_t i = 0; i < 4; ++i)
	{
		const unsigned char* tri = triangles[i];
		const float* p0 = points[tri[0]];
		const float* p1 = points[tri[1]];
		const float* p2 = points[tri[2]];

		// compute triangle center & sample
		float uc = (p0[0] + p1[0] + p2[0]) * (1.f / 3.f);
		float vc = (p0[1] + p1[1] + p2[1]) * (1.f / 3.f);
		float ac = sampleTexture(texture, uc, vc);

		// rasterize opacity state based on alpha values in corners and center (and optionally edges)
		rasterizeOpacity0<States>(result, index * 4 + i, p0[2], p1[2], p2[2], ac, em[tri[3]], em[tri[4]], em[tri[5]], edgeres);
	}
}

template <int States>
static void rasterizeOpacityRec(unsigned char* result, size_t index, int level, int edgeres, const float* c0, const float* c1, const float* c2, const Texture& texture)
{
	if (level == 0)
	{
		// compute triangle center & sample
		float uc = (c0[0] + c1[0] + c2[0]) * (1.f / 3.f);
		float vc = (c0[1] + c1[1] + c2[1]) * (1.f / 3.f);
		float ac = sampleTexture(texture, uc, vc);

		int e0 = 0, e1 = 0, e2 = 0;

		if (edgeres > 0)
		{
			// sample additional points on the edges to improve state estimation
			e0 = rasterizeEdge(c0[0], c0[1], c1[0], c1[1], edgeres, texture);
			e1 = rasterizeEdge(c1[0], c1[1], c2[0], c2[1], edgeres, texture);
			e2 = rasterizeEdge(c2[0], c2[1], c0[0], c0[1], edgeres, texture);
		}

		// rasterize opacity state based on alpha values in corners and center (and optionally edges)
		return rasterizeOpacity0<States>(result, index, c0[2], c1[2], c2[2], ac, e0, e1, e2, edgeres);
	}

	// fast path: equivalent to recursive rasterization, but reuses edge data to reduce sample count
	if (level == 1 && edgeres > 0)
		return rasterizeOpacity1<States>(result, index, edgeres, c0, c1, c2, texture);

	// compute each edge midpoint & sample
	float c01[3] = {(c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2, 0.f};
	float c12[3] = {(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, 0.f};
	float c20[3] = {(c2[0] + c0[0]) / 2, (c2[1] + c0[1]) / 2, 0.f};

	c01[2] = sampleTexture(texture, c01[0], c01[1]);
	c12[2] = sampleTexture(texture, c12[0], c12[1]);
	c20[2] = sampleTexture(texture, c20[0], c20[1]);

	// recursively rasterize each triangle
	// note: triangles 1 and 3 have flipped winding, and 1 is flipped upside down
	rasterizeOpacityRec<States>(result, index * 4 + 0, level - 1, edgeres, c0, c01, c20, texture);
	rasterizeOpacityRec<States>(result, index * 4 + 1, level - 1, edgeres, c20, c12, c01, texture);
	rasterizeOpacityRec<States>(result, index * 4 + 2, level - 1, edgeres, c01, c1, c12, texture);
	rasterizeOpacityRec<States>(result, index * 4 + 3, level - 1, edgeres, c12, c20, c2, texture);
}

static int getSpecialIndex(const unsigned char* data, int level, int states)
{
	int first = data[0] & (states == 2 ? 1 : 3);
	int special = -(1 + first);

	// at level 0, every micromap can be converted to a special index
	if (level == 0)
		return special;

	// at level 1 with 2 states, the byte is partially filled so we need a separate check
	if (level == 1 && states == 2)
		return (data[0] & 15) == ((-first) & 15) ? special : 0;

	// otherwise we need to check that all bytes are consistent with the first value and we can do this byte-wise
	int expected = first * (states == 2 ? 0xff : 0x55);
	size_t size = getLevelSize(level, states);

	for (size_t i = 0; i < size; ++i)
		if (data[i] != expected)
			return 0;

	return special;
}

} // namespace meshopt

size_t meshopt_opacityMapMeasure(unsigned char* levels, unsigned int* sources, int* omm_indices, const unsigned int* indices, size_t index_count, const float* vertex_uvs, size_t vertex_count, size_t vertex_uvs_stride, unsigned int texture_width, unsigned int texture_height, int max_level, float target_edge)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_uvs_stride >= 8 && vertex_uvs_stride <= 256);
	assert(vertex_uvs_stride % sizeof(float) == 0);
	assert(unsigned(texture_width - 1) < 16384 && unsigned(texture_height - 1) < 16384);
	assert(max_level >= 0 && max_level <= 12);
	assert(target_edge >= 0);

	(void)vertex_count;

	meshopt_Allocator allocator;

	size_t vertex_stride_float = vertex_uvs_stride / sizeof(float);
	float texture_area = float(texture_width) * float(texture_height);

	// hash map used to deduplicate triangle rasterization requests based on UV
	size_t table_size = hashBuckets3(index_count / 3);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	TriangleOMM* triangles = allocator.allocate<TriangleOMM>(index_count / 3);
	TriangleOMMHasher hasher = {triangles};

	size_t result = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		float u0 = vertex_uvs[a * vertex_stride_float + 0], v0 = vertex_uvs[a * vertex_stride_float + 1];
		float u1 = vertex_uvs[b * vertex_stride_float + 0], v1 = vertex_uvs[b * vertex_stride_float + 1];
		float u2 = vertex_uvs[c * vertex_stride_float + 0], v2 = vertex_uvs[c * vertex_stride_float + 1];

		int level = max_level;

		if (target_edge > 0)
		{
			// compute ratio of edge length (in texels) to target and determine subdivision level
			float uvarea = fabsf((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)) * 0.5f * texture_area;
			float ratio = sqrtf(uvarea) / target_edge;
			float levelf = log2f(ratio > 1 ? ratio : 1);

			// round to nearest and clamp
			level = int(levelf + 0.5f);
			level = level < 0 ? 0 : level;
			level = level < max_level ? level : max_level;
		}

		// deduplicate rasterization requests based on UV
		int su0 = quantizeSubpixel(u0, texture_width), sv0 = quantizeSubpixel(v0, texture_height);
		int su1 = quantizeSubpixel(u1, texture_width), sv1 = quantizeSubpixel(v1, texture_height);
		int su2 = quantizeSubpixel(u2, texture_width), sv2 = quantizeSubpixel(v2, texture_height);

		TriangleOMM tri = {{su0, sv0, su1, sv1, su2, sv2}, level};
		triangles[result] = tri; // speculatively write triangle data to give hasher a way to compare it

		unsigned int* entry = hashLookup3(table, table_size, hasher, unsigned(result), ~0u);

		if (*entry == ~0u)
		{
			*entry = unsigned(result);
			levels[result] = (unsigned char)level;
			sources[result] = unsigned(i / 3);
			result++;
		}

		omm_indices[i / 3] = int(*entry);
	}

	return result;
}

size_t meshopt_opacityMapEntrySize(int level, int states)
{
	assert(level >= 0 && level <= 12);
	assert(states == 2 || states == 4);

	return meshopt::getLevelSize(level, states);
}

void meshopt_opacityMapRasterize(unsigned char* result, int level, int states, const float* uv0, const float* uv1, const float* uv2, const unsigned char* texture_data, size_t texture_stride, size_t texture_pitch, unsigned int texture_width, unsigned int texture_height)
{
	using namespace meshopt;

	assert(level >= 0 && level <= 12);
	assert(states == 2 || states == 4);
	assert(unsigned(texture_width - 1) < 16384 && unsigned(texture_height - 1) < 16384);
	assert(texture_stride >= 1 && texture_stride <= 4);
	assert(texture_pitch >= texture_stride * texture_width);

	memset(result, 0, getLevelSize(level, states));

	Texture texture = {texture_data, texture_stride, texture_pitch, texture_width, texture_height, float(int(texture_width)) * 256.f, float(int(texture_height)) * 256.f};

	// determine number of edge samples for conservative state estimation
	float texture_area = float(int(texture_width)) * float(int(texture_height));
	float uvarea = fabsf((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (uv1[1] - uv0[1])) * 0.5f * texture_area;
	float uvedge = sqrtf(uvarea) / float(1 << level);

	// target ~2px distance between edge samples (assuming equilateral microtriangles)
	int edgeres = int(uvedge * 0.75f);
	edgeres = edgeres < 0 ? 0 : edgeres;
	edgeres = edgeres > 7 ? 7 : edgeres;

	// rasterize all micro triangles recursively, passing corner data down to reduce redundant sampling
	float c0[3] = {uv0[0], uv0[1], sampleTexture(texture, uv0[0], uv0[1])};
	float c1[3] = {uv1[0], uv1[1], sampleTexture(texture, uv1[0], uv1[1])};
	float c2[3] = {uv2[0], uv2[1], sampleTexture(texture, uv2[0], uv2[1])};

	(states == 2 ? rasterizeOpacityRec<2> : rasterizeOpacityRec<4>)(result, 0, level, edgeres, c0, c1, c2, texture);
}

size_t meshopt_opacityMapCompact(unsigned char* data, size_t data_size, unsigned char* levels, unsigned int* offsets, size_t omm_count, int* omm_indices, size_t triangle_count, int states)
{
	using namespace meshopt;

	assert(states == 2 || states == 4);

	meshopt_Allocator allocator;

	unsigned char* data_old = allocator.allocate<unsigned char>(data_size);
	memcpy(data_old, data, data_size);

	size_t table_size = hashBuckets3(omm_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	OMMHasher hasher = {data, offsets, levels, states};

	int* remap = allocator.allocate<int>(omm_count);

	size_t next = 0;
	size_t offset = 0;

	for (size_t i = 0; i < omm_count; ++i)
	{
		int level = levels[i];
		assert(level >= 0 && level <= 12);

		const unsigned char* old = data_old + offsets[i];
		size_t size = getLevelSize(level, states);
		assert(offsets[i] + size <= data_size);

		// try to convert to a special index if all micro-triangle states are the same
		int special = getSpecialIndex(old, level, states);
		if (special < 0)
		{
			remap[i] = special;
			continue;
		}

		// speculatively write data to give hasher a way to compare it
		memcpy(data + offset, old, size);
		offsets[next] = unsigned(offset);
		levels[next] = (unsigned char)level;

		unsigned int* entry = hashLookup3(table, table_size, hasher, unsigned(next), ~0u);

		if (*entry == ~0u)
		{
			*entry = unsigned(next);
			next++;
			offset += size;
		}

		remap[i] = int(*entry);
	}

	// remap triangle indices to new indices or special indices
	for (size_t i = 0; i < triangle_count; ++i)
	{
		assert(omm_indices[i] < 0 || unsigned(omm_indices[i]) < omm_count);
		omm_indices[i] = omm_indices[i] < 0 ? omm_indices[i] : remap[omm_indices[i]];
	}

	return next;
}
