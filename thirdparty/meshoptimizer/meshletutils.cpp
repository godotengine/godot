// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

// This work is based on:
// Matthaeus Chajdas. GeometryFX 1.2 - Cluster Culling. 2016
// Jack Ritter. An Efficient Bounding Sphere. 1990
// Thomas Larsson. Fast and Tight Fitting Bounding Spheres. 2008
namespace meshopt
{

// This must be <= 256 since meshlet indices are stored as bytes
const size_t kMeshletMaxVertices = 256;

// A reasonable limit is around 2*max_vertices or less
const size_t kMeshletMaxTriangles = 512;

static void computeBoundingSphere(float result[4], const float* points, size_t count, size_t points_stride, const float* radii, size_t radii_stride, size_t axis_count, const unsigned int* indices = NULL)
{
	static const float axes[7][3] = {
	    // X, Y, Z
	    {1, 0, 0},
	    {0, 1, 0},
	    {0, 0, 1},

	    // XYZ, -XYZ, X-YZ, XY-Z; normalized to unit length
	    {0.57735026f, 0.57735026f, 0.57735026f},
	    {-0.57735026f, 0.57735026f, 0.57735026f},
	    {0.57735026f, -0.57735026f, 0.57735026f},
	    {0.57735026f, 0.57735026f, -0.57735026f},
	};

	assert(count > 0);
	assert(axis_count <= sizeof(axes) / sizeof(axes[0]));

	size_t points_stride_float = points_stride / sizeof(float);
	size_t radii_stride_float = radii_stride / sizeof(float);

	// find extremum points along all axes; for each axis we get a pair of points with min/max coordinates
	unsigned int pmin[7], pmax[7];
	float tmin[7], tmax[7];

	for (size_t axis = 0; axis < axis_count; ++axis)
	{
		pmin[axis] = pmax[axis] = 0;
		tmin[axis] = FLT_MAX;
		tmax[axis] = -FLT_MAX;
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int v = indices ? indices[i] : unsigned(i);
		const float* p = points + v * points_stride_float;
		float r = radii[v * radii_stride_float];

		for (size_t axis = 0; axis < axis_count; ++axis)
		{
			const float* ax = axes[axis];

			float tp = ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2];
			float tpmin = tp - r, tpmax = tp + r;

			pmin[axis] = (tpmin < tmin[axis]) ? v : pmin[axis];
			pmax[axis] = (tpmax > tmax[axis]) ? v : pmax[axis];
			tmin[axis] = (tpmin < tmin[axis]) ? tpmin : tmin[axis];
			tmax[axis] = (tpmax > tmax[axis]) ? tpmax : tmax[axis];
		}
	}

	// find the pair of points with largest distance
	size_t paxis = 0;
	float paxisdr = 0;

	for (size_t axis = 0; axis < axis_count; ++axis)
	{
		const float* p1 = points + pmin[axis] * points_stride_float;
		const float* p2 = points + pmax[axis] * points_stride_float;
		float r1 = radii[pmin[axis] * radii_stride_float];
		float r2 = radii[pmax[axis] * radii_stride_float];

		float d2 = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]);
		float dr = sqrtf(d2) + r1 + r2;

		if (dr > paxisdr)
		{
			paxisdr = dr;
			paxis = axis;
		}
	}

	// use the longest segment as the initial sphere diameter
	const float* p1 = points + pmin[paxis] * points_stride_float;
	const float* p2 = points + pmax[paxis] * points_stride_float;
	float r1 = radii[pmin[paxis] * radii_stride_float];
	float r2 = radii[pmax[paxis] * radii_stride_float];

	float paxisd = sqrtf((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]));
	float paxisk = paxisd > 0 ? (paxisd + r2 - r1) / (2 * paxisd) : 0.f;

	float center[3] = {p1[0] + (p2[0] - p1[0]) * paxisk, p1[1] + (p2[1] - p1[1]) * paxisk, p1[2] + (p2[2] - p1[2]) * paxisk};
	float radius = paxisdr / 2;

	// iteratively adjust the sphere up until all points fit
	for (size_t i = 0; i < count; ++i)
	{
		unsigned int v = indices ? indices[i] : unsigned(i);
		const float* p = points + v * points_stride_float;
		float r = radii[v * radii_stride_float];

		float d2 = (p[0] - center[0]) * (p[0] - center[0]) + (p[1] - center[1]) * (p[1] - center[1]) + (p[2] - center[2]) * (p[2] - center[2]);
		float d = sqrtf(d2);

		if (d + r > radius)
		{
			float k = d > 0 ? (d + r - radius) / (2 * d) : 0.f;

			center[0] += k * (p[0] - center[0]);
			center[1] += k * (p[1] - center[1]);
			center[2] += k * (p[2] - center[2]);
			radius = (radius + d + r) / 2;
		}
	}

	result[0] = center[0];
	result[1] = center[1];
	result[2] = center[2];
	result[3] = radius;
}

static meshopt_Bounds computeClusterBounds(const unsigned int* indices, size_t index_count, const unsigned int* corners, size_t corner_count, const float* vertex_positions, size_t vertex_positions_stride)
{
	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	// compute triangle normals (.w completes plane equation)
	float normals[kMeshletMaxTriangles][4];
	size_t triangles = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];

		const float* p0 = vertex_positions + vertex_stride_float * a;
		const float* p1 = vertex_positions + vertex_stride_float * b;
		const float* p2 = vertex_positions + vertex_stride_float * c;

		float p10[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
		float p20[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

		float normalx = p10[1] * p20[2] - p10[2] * p20[1];
		float normaly = p10[2] * p20[0] - p10[0] * p20[2];
		float normalz = p10[0] * p20[1] - p10[1] * p20[0];

		float area = sqrtf(normalx * normalx + normaly * normaly + normalz * normalz);

		// no need to include degenerate triangles - they will be invisible anyway
		if (area == 0.f)
			continue;

		normalx /= area;
		normaly /= area;
		normalz /= area;

		// record triangle normals; normal and corner 0 define a plane equation
		normals[triangles][0] = normalx;
		normals[triangles][1] = normaly;
		normals[triangles][2] = normalz;
		normals[triangles][3] = -(normalx * p0[0] + normaly * p0[1] + normalz * p0[2]);
		triangles++;
	}

	meshopt_Bounds bounds = {};

	// degenerate cluster, no valid triangles => trivial reject (cone data is 0)
	if (triangles == 0)
		return bounds;

	const float rzero = 0.f;

	// compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
	float psphere[4] = {};
	computeBoundingSphere(psphere, vertex_positions, corner_count, vertex_positions_stride, &rzero, 0, 7, corners);

	float center[3] = {psphere[0], psphere[1], psphere[2]};

	// treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
	float nsphere[4] = {};
	computeBoundingSphere(nsphere, normals[0], triangles, sizeof(float) * 4, &rzero, 0, 3);

	float axis[3] = {nsphere[0], nsphere[1], nsphere[2]};
	float axislength = sqrtf(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
	float invaxislength = axislength == 0.f ? 0.f : 1.f / axislength;

	axis[0] *= invaxislength;
	axis[1] *= invaxislength;
	axis[2] *= invaxislength;

	// compute a tight cone around all normals, mindp = cos(angle/2)
	float mindp = 1.f;

	for (size_t i = 0; i < triangles; ++i)
	{
		float dp = normals[i][0] * axis[0] + normals[i][1] * axis[1] + normals[i][2] * axis[2];

		mindp = (dp < mindp) ? dp : mindp;
	}

	// fill bounding sphere info; note that below we can return bounds without cone information for degenerate cones
	bounds.center[0] = center[0];
	bounds.center[1] = center[1];
	bounds.center[2] = center[2];
	bounds.radius = psphere[3];

	// degenerate cluster, normal cone is larger than a hemisphere => trivial accept
	// note that if mindp is positive but close to 0, the triangle intersection code below gets less stable
	// we arbitrarily decide that if a normal cone is ~168 degrees wide or more, the cone isn't useful
	if (mindp <= 0.1f)
	{
		bounds.cone_cutoff = 1;
		bounds.cone_cutoff_s8 = 127;
		return bounds;
	}

	float maxt = 0;

	// we need to find the point on center-t*axis ray that lies in negative half-space of all triangles
	for (size_t i = 0; i < triangles; ++i)
	{
		// dot(center-t*axis-corner, trinormal) = 0
		// dot(center-corner, trinormal) - t * dot(axis, trinormal) = 0
		float dc = center[0] * normals[i][0] + center[1] * normals[i][1] + center[2] * normals[i][2] + normals[i][3];
		float dn = axis[0] * normals[i][0] + axis[1] * normals[i][1] + axis[2] * normals[i][2];

		// dn should be larger than mindp cutoff above
		assert(dn > 0.f);
		float t = dc / dn;

		maxt = (t > maxt) ? t : maxt;
	}

	// cone apex should be in the negative half-space of all cluster triangles by construction
	bounds.cone_apex[0] = center[0] - axis[0] * maxt;
	bounds.cone_apex[1] = center[1] - axis[1] * maxt;
	bounds.cone_apex[2] = center[2] - axis[2] * maxt;

	// note: this axis is the axis of the normal cone, but our test for perspective camera effectively negates the axis
	bounds.cone_axis[0] = axis[0];
	bounds.cone_axis[1] = axis[1];
	bounds.cone_axis[2] = axis[2];

	// cos(a) for normal cone is mindp; we need to add 90 degrees on both sides and invert the cone
	// which gives us -cos(a+90) = -(-sin(a)) = sin(a) = sqrt(1 - cos^2(a))
	bounds.cone_cutoff = sqrtf(1 - mindp * mindp);

	// quantize axis & cutoff to 8-bit SNORM format
	bounds.cone_axis_s8[0] = (signed char)(meshopt_quantizeSnorm(bounds.cone_axis[0], 8));
	bounds.cone_axis_s8[1] = (signed char)(meshopt_quantizeSnorm(bounds.cone_axis[1], 8));
	bounds.cone_axis_s8[2] = (signed char)(meshopt_quantizeSnorm(bounds.cone_axis[2], 8));

	// for the 8-bit test to be conservative, we need to adjust the cutoff by measuring the max. error
	float cone_axis_s8_e0 = fabsf(bounds.cone_axis_s8[0] / 127.f - bounds.cone_axis[0]);
	float cone_axis_s8_e1 = fabsf(bounds.cone_axis_s8[1] / 127.f - bounds.cone_axis[1]);
	float cone_axis_s8_e2 = fabsf(bounds.cone_axis_s8[2] / 127.f - bounds.cone_axis[2]);

	// note that we need to round this up instead of rounding to nearest, hence +1
	int cone_cutoff_s8 = int(127 * (bounds.cone_cutoff + cone_axis_s8_e0 + cone_axis_s8_e1 + cone_axis_s8_e2) + 1);

	bounds.cone_cutoff_s8 = (cone_cutoff_s8 > 127) ? 127 : (signed char)(cone_cutoff_s8);

	return bounds;
}

} // namespace meshopt

meshopt_Bounds meshopt_computeClusterBounds(const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(index_count / 3 <= kMeshletMaxTriangles);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	(void)vertex_count;

	unsigned int cache[512];
	memset(cache, -1, sizeof(cache));

	unsigned int corners[kMeshletMaxTriangles * 3 + 1]; // +1 for branchless slot
	size_t corner_count = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];
		assert(v < vertex_count);

		unsigned int& c = cache[v & (sizeof(cache) / sizeof(cache[0]) - 1)];

		// branchless append if vertex isn't in cache
		corners[corner_count] = v;
		corner_count += (c != v);
		c = v;
	}

	return computeClusterBounds(indices, index_count, corners, corner_count, vertex_positions, vertex_positions_stride);
}

meshopt_Bounds meshopt_computeMeshletBounds(const unsigned int* meshlet_vertices, const unsigned char* meshlet_triangles, size_t triangle_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(triangle_count <= kMeshletMaxTriangles);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	(void)vertex_count;

	unsigned int indices[kMeshletMaxTriangles * 3];
	size_t corner_count = 0;

	for (size_t i = 0; i < triangle_count * 3; ++i)
	{
		unsigned char t = meshlet_triangles[i];
		unsigned int index = meshlet_vertices[t];
		assert(index < vertex_count);

		indices[i] = index;

		// meshlet_vertices[] slice should only contain vertices used by triangle indices, which is the case for any well formed meshlet
		corner_count = t >= corner_count ? t + 1 : corner_count;
	}

	return computeClusterBounds(indices, triangle_count * 3, meshlet_vertices, corner_count, vertex_positions, vertex_positions_stride);
}

meshopt_Bounds meshopt_computeSphereBounds(const float* positions, size_t count, size_t positions_stride, const float* radii, size_t radii_stride)
{
	using namespace meshopt;

	assert(positions_stride >= 12 && positions_stride <= 256);
	assert(positions_stride % sizeof(float) == 0);
	assert((radii_stride >= 4 && radii_stride <= 256) || radii == NULL);
	assert(radii_stride % sizeof(float) == 0);

	meshopt_Bounds bounds = {};

	if (count == 0)
		return bounds;

	const float rzero = 0.f;

	float psphere[4] = {};
	computeBoundingSphere(psphere, positions, count, positions_stride, radii ? radii : &rzero, radii ? radii_stride : 0, 7);

	bounds.center[0] = psphere[0];
	bounds.center[1] = psphere[1];
	bounds.center[2] = psphere[2];
	bounds.radius = psphere[3];

	return bounds;
}

void meshopt_optimizeMeshletLevel(unsigned int* meshlet_vertices, size_t vertex_count, unsigned char* meshlet_triangles, size_t triangle_count, int level)
{
	using namespace meshopt;

	assert(triangle_count <= kMeshletMaxTriangles);
	assert(vertex_count <= kMeshletMaxVertices);
	assert(level >= 0 && level <= 9);

	unsigned char* indices = meshlet_triangles;
	unsigned int* vertices = meshlet_vertices;

	// cache tracks vertex timestamps (corresponding to triangle index! all 3 vertices are added at the same time and never removed)
	unsigned char cache[kMeshletMaxVertices];
	memset(cache, 0, vertex_count);

	// note that we start from a value that means all vertices aren't in cache
	unsigned char cache_last = 128;
	const unsigned char cache_cutoff = 3; // 3 triangles = ~5..9 vertices depending on reuse

	// vertex valence is used to prioritize triangles for level>0
	// note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristic
	unsigned char valence[kMeshletMaxVertices];
	memset(valence, 0, vertex_count);

	for (size_t i = 0; i < triangle_count; ++i)
	{
		unsigned char a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		valence[a]++;
		valence[b]++;
		valence[c]++;
	}

	for (size_t i = 0; i < triangle_count; ++i)
	{
		int next = -1;
		int next_score = -1;
		int edges = 0;

		for (size_t j = i; j < triangle_count; ++j)
		{
			unsigned char a = indices[j * 3 + 0], b = indices[j * 3 + 1], c = indices[j * 3 + 2];
			assert(a < vertex_count && b < vertex_count && c < vertex_count);

			// compute cache distance using unsigned 8-bit subtraction, so cache timestamp overflow is handled gracefully
			unsigned char ad = (unsigned char)(cache_last - cache[a]);
			unsigned char bd = (unsigned char)(cache_last - cache[b]);
			unsigned char cd = (unsigned char)(cache_last - cache[c]);

			int match = (ad < cache_cutoff) + (bd < cache_cutoff) + (cd < cache_cutoff);

			if (level)
			{
				// prefer low minimum valence
				int vmin = valence[a] < valence[b] ? valence[a] : valence[b];
				vmin = valence[c] < vmin ? valence[c] : vmin;

				// prefer vertices with smaller cache distance and valence to improve traversal locality
				int score = match * 1024 + (1023 - ad - bd - cd);
				score = score * 256 + (255 - vmin);

				next = (score > next_score) ? int(j) : next;
				next_score = (score > next_score) ? score : next_score;

				// terminate after finding enough edge matches
				if (match >= 2 && ++edges >= level)
					break;
			}
			else
			{
				int score = match;

				next = (score > next_score) ? int(j) : next;
				next_score = (score > next_score) ? score : next_score;

				// settle for a first edge match, which makes the function ~linear in practice
				if (match >= 2)
					break;
			}
		}

		assert(next >= 0);

		unsigned char a = indices[next * 3 + 0], b = indices[next * 3 + 1], c = indices[next * 3 + 2];

		// shift triangles before the next one forward so that we always keep an ordered partition
		// note: this could have swapped triangles [i] and [next] but that distorts the order and may skew the output sequence
		memmove(indices + (i + 1) * 3, indices + i * 3, (next - i) * 3 * sizeof(unsigned char));

		indices[i * 3 + 0] = a;
		indices[i * 3 + 1] = b;
		indices[i * 3 + 2] = c;

		// cache timestamp is the same between all vertices of each triangle to reduce overflow
		cache_last++;
		cache[a] = cache_last;
		cache[b] = cache_last;
		cache[c] = cache_last;

		// update vertex valences for scoring heuristic
		valence[a]--;
		valence[b]--;
		valence[c]--;
	}

	// rotate triangles to maximize compressibility; only done at level >= 1 for compatibility
	if (level >= 1)
	{
		memset(cache, 0, vertex_count);

		for (size_t i = 0; i < triangle_count; ++i)
		{
			unsigned char a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];

			// if only the middle vertex has been used, rotate triangle to ensure new vertices are always sequential
			if (!cache[a] && cache[b] && !cache[c])
			{
				// abc -> bca
				unsigned char t = a;
				a = b, b = c, c = t;
			}
			else if (!cache[a] && !cache[b] && !cache[c])
			{
				// out of three edges, the edge ab can not be reused by subsequent triangles in some encodings
				// if subsequent triangles don't share edges ca or bc, we can rotate the triangle to fix this
				bool needab = false, needbc = false, needca = false;

				for (size_t j = i + 1; j < triangle_count && j <= i + 3; ++j)
				{
					unsigned char oa = indices[j * 3 + 0], ob = indices[j * 3 + 1], oc = indices[j * 3 + 2];

					// note: edge comparisons are reversed as reused edges are flipped
					needab |= (oa == b && ob == a) || (ob == b && oc == a) || (oc == b && oa == a);
					needbc |= (oa == c && ob == b) || (ob == c && oc == b) || (oc == c && oa == b);
					needca |= (oa == a && ob == c) || (ob == a && oc == c) || (oc == a && oa == c);
				}

				if (needab && !needbc)
				{
					// abc -> bca
					unsigned char t = a;
					a = b, b = c, c = t;
				}
				else if (needab && !needca)
				{
					// abc -> cab
					unsigned char t = c;
					c = b, b = a, a = t;
				}
			}

			indices[i * 3 + 0] = a, indices[i * 3 + 1] = b, indices[i * 3 + 2] = c;

			cache[a] = cache[b] = cache[c] = 1;
		}
	}

	// reorder meshlet vertices for access locality assuming index buffer is scanned sequentially
	unsigned int order[kMeshletMaxVertices];

	short remap[kMeshletMaxVertices];
	memset(remap, -1, vertex_count * sizeof(short));

	size_t vertex_offset = 0;

	for (size_t i = 0; i < triangle_count * 3; ++i)
	{
		short& r = remap[indices[i]];

		if (r < 0)
		{
			r = short(vertex_offset);
			order[vertex_offset] = vertices[indices[i]];
			vertex_offset++;
		}

		indices[i] = (unsigned char)r;
	}

	assert(vertex_offset <= vertex_count);
	memcpy(vertices, order, vertex_offset * sizeof(unsigned int));
}

void meshopt_optimizeMeshlet(unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, size_t triangle_count, size_t vertex_count)
{
	meshopt_optimizeMeshletLevel(meshlet_vertices, vertex_count, meshlet_triangles, triangle_count, 0);
}

size_t meshopt_extractMeshletIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(index_count / 3 <= kMeshletMaxTriangles);

	size_t unique = 0;

	// direct mapped cache for fast lookups based on low index bits; inspired by vk_lod_clusters from NVIDIA
	short cache[1024];
	memset(cache, -1, sizeof(cache));

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];
		unsigned int key = v & (sizeof(cache) / sizeof(cache[0]) - 1);
		short c = cache[key];

		// fast path: vertex has been seen before
		if (c >= 0 && vertices[c] == v)
		{
			triangles[i] = (unsigned char)c;
			continue;
		}

		// fast path: vertex has never been seen before
		if (c < 0)
		{
			assert(unique < kMeshletMaxVertices);
			cache[key] = short(unique);
			triangles[i] = (unsigned char)unique;
			vertices[unique++] = v;
			continue;
		}

		// slow path: collision with a different vertex, so we need to look through all vertices
		int pos = -1;
		for (size_t j = 0; j < unique; ++j)
			if (vertices[j] == v)
			{
				pos = int(j);
				break;
			}

		if (pos < 0)
		{
			assert(unique < kMeshletMaxVertices);
			pos = int(unique);
			vertices[unique++] = v;
		}

		cache[key] = short(pos);
		triangles[i] = (unsigned char)pos;
	}

	assert(unique <= kMeshletMaxVertices);
	return unique;
}
