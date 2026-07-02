// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <math.h>
#include <string.h>

// This work is based on:
// Morten Mikkelsen. Simulation of wrinkled surfaces revisited. 2008
// Matthias Teschner, Bruno Heidelberger, Matthias Mueller, Danat Pomeranets, Markus Gross. Optimized Spatial Hashing for Collision Detection of Deformable Objects. 2003
// Cecil Hastings Jr. Approximations for digital computers. 1955
namespace meshopt
{

struct Tangent
{
	float x, y, z, w;
};

static void computeFaceTangents(Tangent* result, size_t triangle_count, const unsigned int* indices, const float* vertex_positions, size_t vertex_positions_stride, const float* vertex_uvs, size_t vertex_uvs_stride)
{
	size_t vertex_position_stride_float = vertex_positions_stride / sizeof(float);
	size_t vertex_uv_stride_float = vertex_uvs_stride / sizeof(float);

	for (size_t i = 0; i < triangle_count; ++i)
	{
		unsigned int a = indices ? indices[i * 3 + 0] : unsigned(i * 3 + 0);
		unsigned int b = indices ? indices[i * 3 + 1] : unsigned(i * 3 + 1);
		unsigned int c = indices ? indices[i * 3 + 2] : unsigned(i * 3 + 2);

		const float* pa = &vertex_positions[a * vertex_position_stride_float];
		const float* pb = &vertex_positions[b * vertex_position_stride_float];
		const float* pc = &vertex_positions[c * vertex_position_stride_float];

		const float* ta = &vertex_uvs[a * vertex_uv_stride_float];
		const float* tb = &vertex_uvs[b * vertex_uv_stride_float];
		const float* tc = &vertex_uvs[c * vertex_uv_stride_float];

		// compute face tangents using chain rule, see eq. 42
		float dp1x = pb[0] - pa[0], dp1y = pb[1] - pa[1], dp1z = pb[2] - pa[2];
		float dp2x = pc[0] - pa[0], dp2y = pc[1] - pa[1], dp2z = pc[2] - pa[2];
		float dt1x = tb[0] - ta[0], dt1y = tb[1] - ta[1];
		float dt2x = tc[0] - ta[0], dt2y = tc[1] - ta[1];

		float rx = dt2y * dp1x - dt1y * dp2x;
		float ry = dt2y * dp1y - dt1y * dp2y;
		float rz = dt2y * dp1z - dt1y * dp2z;

		// .w stores orientation (+1/-1) or 0 for degenerate triangles, see eq. 48
		float det = dt1x * dt2y - dt1y * dt2x;
		float rw = det == 0.f ? 0.f : (det > 0.f ? 1.f : -1.f);

		// treat zero area triangles as degenerate even if their UVs aren't for consistency
		rw = (pa[0] == pb[0] && pa[1] == pb[1] && pa[2] == pb[2]) ? 0.f : rw;
		rw = (pa[0] == pc[0] && pa[1] == pc[1] && pa[2] == pc[2]) ? 0.f : rw;
		rw = (pb[0] == pc[0] && pb[1] == pc[1] && pb[2] == pc[2]) ? 0.f : rw;

		float rl = sqrtf(rx * rx + ry * ry + rz * rz);
		float rs = rl != 0.f ? rw / rl : 0.f;

		result[i].x = rx * rs;
		result[i].y = ry * rs;
		result[i].z = rz * rs;
		result[i].w = rw;
	}
}

struct VertexHasherF
{
	const float* vertex_positions;
	size_t vertex_positions_stride_float;
	const float* vertex_normals;
	size_t vertex_normals_stride_float;
	const float* vertex_uvs;
	size_t vertex_uvs_stride_float;

	size_t hash(unsigned int index) const
	{
		const unsigned int* p = reinterpret_cast<const unsigned int*>(vertex_positions + index * vertex_positions_stride_float);
		const unsigned int* n = reinterpret_cast<const unsigned int*>(vertex_normals + index * vertex_normals_stride_float);
		const unsigned int* t = reinterpret_cast<const unsigned int*>(vertex_uvs + index * vertex_uvs_stride_float);

		unsigned int x = p[0], y = p[1], z = p[2];

		// replace negative zero with zero
		x = (x == 0x80000000) ? 0 : x;
		y = (y == 0x80000000) ? 0 : y;
		z = (z == 0x80000000) ? 0 : z;

		// scramble bits to make sure that integer coordinates have entropy in lower bits
		x ^= x >> 17;
		y ^= y >> 17;
		z ^= z >> 17;

		// mix in normal bits
		x ^= (n[0] == 0x80000000) ? 0 : n[0] >> 15;
		y ^= (n[1] == 0x80000000) ? 0 : n[1] >> 15;
		z ^= (n[2] == 0x80000000) ? 0 : n[2] >> 15;

		// collect texture coordinate bits (simplified -0 handling and scrambling)
		unsigned int w = (t[0] ^ t[1]) & 0x7fffffff;
		w ^= w >> 13;

		// Optimized Spatial Hashing for Collision Detection of Deformable Objects
		return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^ (w * 50331653);
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		const float* lp = vertex_positions + lhs * vertex_positions_stride_float;
		const float* ln = vertex_normals + lhs * vertex_normals_stride_float;
		const float* lt = vertex_uvs + lhs * vertex_uvs_stride_float;
		const float* rp = vertex_positions + rhs * vertex_positions_stride_float;
		const float* rn = vertex_normals + rhs * vertex_normals_stride_float;
		const float* rt = vertex_uvs + rhs * vertex_uvs_stride_float;

		return lp[0] == rp[0] && lp[1] == rp[1] && lp[2] == rp[2] && ln[0] == rn[0] && ln[1] == rn[1] && ln[2] == rn[2] && lt[0] == rt[0] && lt[1] == rt[1];
	}
};

static size_t hashBuckets4(size_t count)
{
	size_t buckets = 1;
	while (buckets < count + count / 4)
		buckets *= 2;

	return buckets;
}

template <typename T, typename Hash>
static T* hashLookup4(T* table, size_t buckets, const Hash& hash, const T& key, const T& empty)
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

static void buildVertexRemap(unsigned int* remap, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, const float* vertex_normals, size_t vertex_normals_stride, const float* vertex_uvs, size_t vertex_uvs_stride, meshopt_Allocator& allocator)
{
	VertexHasherF vertex_hasher = {vertex_positions, vertex_positions_stride / sizeof(float), vertex_normals, vertex_normals_stride / sizeof(float), vertex_uvs, vertex_uvs_stride / sizeof(float)};

	size_t vertex_table_size = hashBuckets4(vertex_count);
	unsigned int* vertex_table = allocator.allocate<unsigned int>(vertex_table_size);
	memset(vertex_table, -1, vertex_table_size * sizeof(unsigned int));

	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int index = unsigned(i);
		unsigned int* entry = hashLookup4(vertex_table, vertex_table_size, vertex_hasher, index, ~0u);

		if (*entry == ~0u)
			*entry = index;

		remap[index] = *entry;
	}

	allocator.deallocate(vertex_table);
}

struct CornerAdjacency
{
	unsigned int* offsets;
	unsigned int* data;
};

static void buildCornerAdjacency(CornerAdjacency& adjacency, const unsigned int* indices, size_t index_count, const unsigned int* remap, size_t vertex_count, meshopt_Allocator& allocator)
{
	adjacency.offsets = allocator.allocate<unsigned int>(vertex_count + 1);
	adjacency.data = allocator.allocate<unsigned int>(index_count);

	size_t face_count = index_count / 3;
	unsigned int* offsets = adjacency.offsets + 1;

	// fill corner counts
	memset(offsets, 0, vertex_count * sizeof(unsigned int));

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices ? remap[indices[i]] : remap[i];
		assert(v < vertex_count);

		offsets[v]++;
	}

	// fill offset table
	unsigned int offset = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int count = offsets[i];
		offsets[i] = offset;
		offset += count;
	}

	assert(offset == index_count);

	// fill corner data
	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices ? remap[indices[i * 3 + 0]] : remap[i * 3 + 0];
		unsigned int b = indices ? remap[indices[i * 3 + 1]] : remap[i * 3 + 1];
		unsigned int c = indices ? remap[indices[i * 3 + 2]] : remap[i * 3 + 2];

		// encode corner index in the low 2 bits
		adjacency.data[offsets[a]++] = unsigned(i << 2) | 0;
		adjacency.data[offsets[b]++] = unsigned(i << 2) | 1;
		adjacency.data[offsets[c]++] = unsigned(i << 2) | 2;
	}

	// finalize offsets
	adjacency.offsets[0] = 0;
	assert(adjacency.offsets[vertex_count] == index_count);
}

static unsigned int follow2(unsigned int* parents, unsigned int index)
{
	while (index != parents[index])
	{
		unsigned int parent = parents[index];
		parents[index] = parents[parent];
		index = parent;
	}

	return index;
}

static void mergeTangentGroups(unsigned int* groups, unsigned int* facegroups, unsigned char* facesign, const unsigned int* data, size_t count, const unsigned int* indices, const unsigned int* remap)
{
	static const int next[4] = {1, 2, 0, 1};

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int ti = data[i] >> 2;
		unsigned int cib = ti * 3 + next[(data[i] & 3) + 0];
		unsigned int cic = ti * 3 + next[(data[i] & 3) + 1];

		unsigned int nib = indices ? remap[indices[cib]] : remap[cib];
		unsigned int nic = indices ? remap[indices[cic]] : remap[cic];

		for (size_t j = i + 1; j < count; ++j)
		{
			unsigned int tj = data[j] >> 2;
			unsigned int cjb = tj * 3 + next[(data[j] & 3) + 0];
			unsigned int cjc = tj * 3 + next[(data[j] & 3) + 1];

			unsigned int njb = indices ? remap[indices[cjb]] : remap[cjb];
			unsigned int njc = indices ? remap[indices[cjc]] : remap[cjc];

			// merge tangent groups if triangles are adjacent and orientations agree or either one is degenerate
			if ((njb == nic || njc == nib) && (facesign[ti] | facesign[tj]) != 3)
			{
				// if either triangle is degenerate, it could have been assigned to a concrete orientation; we need to confirm that via facegroups
				if ((facesign[ti] & facesign[tj]) == 0)
				{
					unsigned int gti = follow2(facegroups, ti);
					unsigned int gtj = follow2(facegroups, tj);

					// skip merge if orientations diverge; otherwise, union the groups with gti as the root
					if (gti != gtj)
					{
						if ((facesign[gti] | facesign[gtj]) == 3)
							continue;

						facegroups[gtj] = gti;
						facesign[gti] |= facesign[gtj];
					}
				}

				// union tangent groups for individual corners with gi as the root; the orientations must match due to facegroups check above
				unsigned int gi = follow2(groups, ti * 3 + (data[i] & 3));
				unsigned int gj = follow2(groups, tj * 3 + (data[j] & 3));

				if (gi != gj)
					groups[gj] = gi;
			}
		}
	}
}

inline float optacos(float x)
{
	// approximation for acos(x) = sqrt(1 - abs(x)) * 2-degree polynomial, max error ~5e-4
	float ax = fabsf(x);
	ax = ax > 1.f ? 1.f : ax;
	float r = 1.570337f + ax * (-0.2053972f + ax * 0.05147786f);
	r *= sqrtf(1.0f - ax);
	return x < 0.f ? 3.1415926f - r : r;
}

static void accumulateTangentGroups(float* result, const unsigned int* groups, const unsigned int* indices, size_t index_count, const unsigned int* remap, const Tangent* face_tangents, const float* vertex_positions, size_t vertex_positions_stride, const float* vertex_normals, size_t vertex_normals_stride, unsigned int options)
{
	static const int next[4] = {1, 2, 0, 1};

	size_t vertex_position_stride_float = vertex_positions_stride / sizeof(float);
	size_t vertex_normal_stride_float = vertex_normals_stride / sizeof(float);

	size_t face_count = index_count / 3;

	for (size_t i = 0; i < face_count; ++i)
	{
		const Tangent& t = face_tangents[i];

		if (t.w == 0.f)
			continue;

		for (int k = 0; k < 3; ++k)
		{
			float* r = &result[size_t(groups[i * 3 + k]) * 4];

			unsigned int a = indices ? remap[indices[i * 3 + k]] : remap[i * 3 + k];
			unsigned int b = indices ? remap[indices[i * 3 + next[k]]] : remap[i * 3 + next[k]];
			unsigned int c = indices ? remap[indices[i * 3 + next[k + 1]]] : remap[i * 3 + next[k + 1]];

			const float* pa = vertex_positions + a * vertex_position_stride_float;
			const float* pb = vertex_positions + b * vertex_position_stride_float;
			const float* pc = vertex_positions + c * vertex_position_stride_float;

			const float* n = vertex_normals + a * vertex_normal_stride_float;

			// reproject tangent vector onto vertex normal
			float sd = t.x * n[0] + t.y * n[1] + t.z * n[2];
			float sx = t.x - n[0] * sd, sy = t.y - n[1] * sd, sz = t.z - n[2] * sd;
			float sl = sqrtf(sx * sx + sy * sy + sz * sz);

			// compute incident angle for weighting, in tangent plane
			// note: this step is absent from the paper, reference implementation computes angle after projecting edges to tangent plane...
			float dp1x = pb[0] - pa[0], dp1y = pb[1] - pa[1], dp1z = pb[2] - pa[2];
			float dp2x = pc[0] - pa[0], dp2y = pc[1] - pa[1], dp2z = pc[2] - pa[2];

			float dp1d = dp1x * n[0] + dp1y * n[1] + dp1z * n[2];
			float dp2d = dp2x * n[0] + dp2y * n[1] + dp2z * n[2];

			dp1x -= n[0] * dp1d, dp1y -= n[1] * dp1d, dp1z -= n[2] * dp1d;
			dp2x -= n[0] * dp2d, dp2y -= n[1] * dp2d, dp2z -= n[2] * dp2d;

			float dp1l = dp1x * dp1x + dp1y * dp1y + dp1z * dp1z;
			float dp2l = dp2x * dp2x + dp2y * dp2y + dp2z * dp2z;
			float dpl = sqrtf(dp1l * dp2l);

			float cosa = (dp1x * dp2x + dp1y * dp2y + dp1z * dp2z) * (dpl == 0.f ? 0.f : 1.f / dpl);
			float angle = optacos(cosa); // optacos handles clamping to [-1..1]

			// accumulate renormalized tangent weighted by angle
			float w = angle * (sl == 0.f ? 0.f : 1.f / sl);

			// weight larger adjacent triangles more to reduce interpolation artifacts from slivers; this deviates from reference implementation for higher quality
			if ((options & meshopt_TangentCompatible) == 0)
				w *= dpl;

			r[0] += sx * w;
			r[1] += sy * w;
			r[2] += sz * w;
		}
	}
}

} // namespace meshopt

void meshopt_generateTangents(float* result, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, const float* vertex_normals, size_t vertex_normals_stride, const float* vertex_uvs, size_t vertex_uvs_stride, unsigned int options)
{
	using namespace meshopt;

	assert(indices || index_count == vertex_count);
	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_normals_stride >= 12 && vertex_normals_stride <= 256);
	assert(vertex_uvs_stride >= 8 && vertex_uvs_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0 && vertex_normals_stride % sizeof(float) == 0 && vertex_uvs_stride % sizeof(float) == 0);

	meshopt_Allocator allocator;

	size_t face_count = index_count / 3;

	// compute vertex remap to unique vertex index
	unsigned int* remap = allocator.allocate<unsigned int>(vertex_count);
	buildVertexRemap(remap, vertex_positions, vertex_count, vertex_positions_stride, vertex_normals, vertex_normals_stride, vertex_uvs, vertex_uvs_stride, allocator);

	// build adjacency information
	CornerAdjacency adjacency = {};
	buildCornerAdjacency(adjacency, indices, index_count, remap, vertex_count, allocator);

	// compute per-triangle tangents
	Tangent* face_tangents = allocator.allocate<Tangent>(face_count);
	computeFaceTangents(face_tangents, face_count, indices, vertex_positions, vertex_positions_stride, vertex_uvs, vertex_uvs_stride);

	// compute per-corner tangent groups: triangles adjacent to the same vertex with the same orientation are merged in strips
	unsigned int* groups = allocator.allocate<unsigned int>(index_count);
	for (size_t i = 0; i < index_count; ++i)
		groups[i] = unsigned(i);

	// compute per-face orientation groups: degenerate triangles are merged with adjacent non-degenerate ones and inherit their orientation, stored as a bitmask
	unsigned int* facegroups = allocator.allocate<unsigned int>(face_count);
	unsigned char* facesign = allocator.allocate<unsigned char>(face_count);

	for (size_t i = 0; i < face_count; ++i)
	{
		facegroups[i] = unsigned(i);
		facesign[i] = (face_tangents[i].w > 0.f ? 1 : 0) | (face_tangents[i].w < 0.f ? 2 : 0);
	}

	for (size_t i = 0; i < vertex_count; ++i)
		if (adjacency.offsets[i + 1] != adjacency.offsets[i])
			mergeTangentGroups(groups, facegroups, facesign, adjacency.data + adjacency.offsets[i], adjacency.offsets[i + 1] - adjacency.offsets[i], indices, remap);

	for (size_t i = 0; i < index_count; ++i)
		groups[i] = follow2(groups, unsigned(i));

	// accumulate tangents into their own respective groups
	memset(result, 0, index_count * sizeof(float) * 4);
	accumulateTangentGroups(result, groups, indices, index_count, remap, face_tangents, vertex_positions, vertex_positions_stride, vertex_normals, vertex_normals_stride, options);

	// finalize tangent signs for each face using facegroups
	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int fg = follow2(facegroups, unsigned(i));
		float w = (facesign[fg] & 1) ? 1.f : -1.f; // note: if face is degenerate, we canonicalize to -1 for consistency

		result[(i * 3 + 0) * 4 + 3] = result[(i * 3 + 1) * 4 + 3] = result[(i * 3 + 2) * 4 + 3] = w;
	}

	// finalize tangent vectors by normalizing roots and propagating the rest
	for (size_t i = 0; i < index_count; ++i)
		if (groups[i] == i)
		{
			float* r = &result[i * 4];
			float l = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
			float s = l == 0.f ? 0.f : 1.f / l;

			r[0] *= s;
			r[1] *= s;
			r[2] *= s;

			if ((options & meshopt_TangentZeroFallback) == 0)
				r[0] = s == 0.f ? 1.f : r[0]; // for isolated degenerate triangles, use (1, 0, 0) tangent for consistency
		}

	for (size_t i = 0; i < index_count; ++i)
		if (groups[i] != i)
			memcpy(&result[i * 4], &result[size_t(groups[i]) * 4], sizeof(float) * 3); // .w was set per face earlier
}
