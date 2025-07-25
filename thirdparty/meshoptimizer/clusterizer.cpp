// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

// This work is based on:
// Graham Wihlidal. Optimizing the Graphics Pipeline with Compute. 2016
// Matthaeus Chajdas. GeometryFX 1.2 - Cluster Culling. 2016
// Jack Ritter. An Efficient Bounding Sphere. 1990
// Thomas Larsson. Fast and Tight Fitting Bounding Spheres. 2008
// Ingo Wald, Vlastimil Havran. On building fast kd-Trees for Ray Tracing, and on doing that in O(N log N). 2006
namespace meshopt
{

// This must be <= 256 since meshlet indices are stored as bytes
const size_t kMeshletMaxVertices = 256;

// A reasonable limit is around 2*max_vertices or less
const size_t kMeshletMaxTriangles = 512;

// We keep a limited number of seed triangles and add a few triangles per finished meshlet
const size_t kMeshletMaxSeeds = 256;
const size_t kMeshletAddSeeds = 4;

// To avoid excessive recursion for malformed inputs, we limit the maximum depth of the tree
const int kMeshletMaxTreeDepth = 50;

struct TriangleAdjacency2
{
	unsigned int* counts;
	unsigned int* offsets;
	unsigned int* data;
};

static void buildTriangleAdjacency(TriangleAdjacency2& adjacency, const unsigned int* indices, size_t index_count, size_t vertex_count, meshopt_Allocator& allocator)
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

static void buildTriangleAdjacencySparse(TriangleAdjacency2& adjacency, const unsigned int* indices, size_t index_count, size_t vertex_count, meshopt_Allocator& allocator)
{
	size_t face_count = index_count / 3;

	// sparse mode can build adjacency more quickly by ignoring unused vertices, using a bit to mark visited vertices
	const unsigned int sparse_seen = 1u << 31;
	assert(index_count < sparse_seen);

	// allocate arrays
	adjacency.counts = allocator.allocate<unsigned int>(vertex_count);
	adjacency.offsets = allocator.allocate<unsigned int>(vertex_count);
	adjacency.data = allocator.allocate<unsigned int>(index_count);

	// fill triangle counts
	for (size_t i = 0; i < index_count; ++i)
		assert(indices[i] < vertex_count);

	for (size_t i = 0; i < index_count; ++i)
		adjacency.counts[indices[i]] = 0;

	for (size_t i = 0; i < index_count; ++i)
		adjacency.counts[indices[i]]++;

	// fill offset table; uses sparse_seen bit to tag visited vertices
	unsigned int offset = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];

		if ((adjacency.counts[v] & sparse_seen) == 0)
		{
			adjacency.offsets[v] = offset;
			offset += adjacency.counts[v];
			adjacency.counts[v] |= sparse_seen;
		}
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
	// also fix counts (that were marked with sparse_seen by the first pass)
	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];

		if (adjacency.counts[v] & sparse_seen)
		{
			adjacency.counts[v] &= ~sparse_seen;

			assert(adjacency.offsets[v] >= adjacency.counts[v]);
			adjacency.offsets[v] -= adjacency.counts[v];
		}
	}
}

static void computeBoundingSphere(float result[4], const float* points, size_t count, size_t points_stride, const float* radii, size_t radii_stride, size_t axis_count)
{
	static const float kAxes[7][3] = {
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
	assert(axis_count <= sizeof(kAxes) / sizeof(kAxes[0]));

	size_t points_stride_float = points_stride / sizeof(float);
	size_t radii_stride_float = radii_stride / sizeof(float);

	// find extremum points along all axes; for each axis we get a pair of points with min/max coordinates
	size_t pmin[7], pmax[7];
	float tmin[7], tmax[7];

	for (size_t axis = 0; axis < axis_count; ++axis)
	{
		pmin[axis] = pmax[axis] = 0;
		tmin[axis] = FLT_MAX;
		tmax[axis] = -FLT_MAX;
	}

	for (size_t i = 0; i < count; ++i)
	{
		const float* p = points + i * points_stride_float;
		float r = radii[i * radii_stride_float];

		for (size_t axis = 0; axis < axis_count; ++axis)
		{
			const float* ax = kAxes[axis];

			float tp = ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2];
			float tpmin = tp - r, tpmax = tp + r;

			pmin[axis] = (tpmin < tmin[axis]) ? i : pmin[axis];
			pmax[axis] = (tpmax > tmax[axis]) ? i : pmax[axis];
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
		const float* p = points + i * points_stride_float;
		float r = radii[i * radii_stride_float];

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

struct Cone
{
	float px, py, pz;
	float nx, ny, nz;
};

static float getDistance(float dx, float dy, float dz, bool aa)
{
	if (!aa)
		return sqrtf(dx * dx + dy * dy + dz * dz);

	float rx = fabsf(dx), ry = fabsf(dy), rz = fabsf(dz);
	float rxy = rx > ry ? rx : ry;
	return rxy > rz ? rxy : rz;
}

static float getMeshletScore(float distance, float spread, float cone_weight, float expected_radius)
{
	if (cone_weight < 0)
		return 1 + distance / expected_radius;

	float cone = 1.f - spread * cone_weight;
	float cone_clamped = cone < 1e-3f ? 1e-3f : cone;

	return (1 + distance / expected_radius * (1 - cone_weight)) * cone_clamped;
}

static Cone getMeshletCone(const Cone& acc, unsigned int triangle_count)
{
	Cone result = acc;

	float center_scale = triangle_count == 0 ? 0.f : 1.f / float(triangle_count);

	result.px *= center_scale;
	result.py *= center_scale;
	result.pz *= center_scale;

	float axis_length = result.nx * result.nx + result.ny * result.ny + result.nz * result.nz;
	float axis_scale = axis_length == 0.f ? 0.f : 1.f / sqrtf(axis_length);

	result.nx *= axis_scale;
	result.ny *= axis_scale;
	result.nz *= axis_scale;

	return result;
}

static float computeTriangleCones(Cone* triangles, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	(void)vertex_count;

	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);
	size_t face_count = index_count / 3;

	float mesh_area = 0;

	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		const float* p0 = vertex_positions + vertex_stride_float * a;
		const float* p1 = vertex_positions + vertex_stride_float * b;
		const float* p2 = vertex_positions + vertex_stride_float * c;

		float p10[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
		float p20[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

		float normalx = p10[1] * p20[2] - p10[2] * p20[1];
		float normaly = p10[2] * p20[0] - p10[0] * p20[2];
		float normalz = p10[0] * p20[1] - p10[1] * p20[0];

		float area = sqrtf(normalx * normalx + normaly * normaly + normalz * normalz);
		float invarea = (area == 0.f) ? 0.f : 1.f / area;

		triangles[i].px = (p0[0] + p1[0] + p2[0]) / 3.f;
		triangles[i].py = (p0[1] + p1[1] + p2[1]) / 3.f;
		triangles[i].pz = (p0[2] + p1[2] + p2[2]) / 3.f;

		triangles[i].nx = normalx * invarea;
		triangles[i].ny = normaly * invarea;
		triangles[i].nz = normalz * invarea;

		mesh_area += area;
	}

	return mesh_area;
}

static void finishMeshlet(meshopt_Meshlet& meshlet, unsigned char* meshlet_triangles)
{
	size_t offset = meshlet.triangle_offset + meshlet.triangle_count * 3;

	// fill 4b padding with 0
	while (offset & 3)
		meshlet_triangles[offset++] = 0;
}

static bool appendMeshlet(meshopt_Meshlet& meshlet, unsigned int a, unsigned int b, unsigned int c, short* used, meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, size_t meshlet_offset, size_t max_vertices, size_t max_triangles, bool split = false)
{
	short& av = used[a];
	short& bv = used[b];
	short& cv = used[c];

	bool result = false;

	int used_extra = (av < 0) + (bv < 0) + (cv < 0);

	if (meshlet.vertex_count + used_extra > max_vertices || meshlet.triangle_count >= max_triangles || split)
	{
		meshlets[meshlet_offset] = meshlet;

		for (size_t j = 0; j < meshlet.vertex_count; ++j)
			used[meshlet_vertices[meshlet.vertex_offset + j]] = -1;

		finishMeshlet(meshlet, meshlet_triangles);

		meshlet.vertex_offset += meshlet.vertex_count;
		meshlet.triangle_offset += (meshlet.triangle_count * 3 + 3) & ~3; // 4b padding
		meshlet.vertex_count = 0;
		meshlet.triangle_count = 0;

		result = true;
	}

	if (av < 0)
	{
		av = short(meshlet.vertex_count);
		meshlet_vertices[meshlet.vertex_offset + meshlet.vertex_count++] = a;
	}

	if (bv < 0)
	{
		bv = short(meshlet.vertex_count);
		meshlet_vertices[meshlet.vertex_offset + meshlet.vertex_count++] = b;
	}

	if (cv < 0)
	{
		cv = short(meshlet.vertex_count);
		meshlet_vertices[meshlet.vertex_offset + meshlet.vertex_count++] = c;
	}

	meshlet_triangles[meshlet.triangle_offset + meshlet.triangle_count * 3 + 0] = (unsigned char)av;
	meshlet_triangles[meshlet.triangle_offset + meshlet.triangle_count * 3 + 1] = (unsigned char)bv;
	meshlet_triangles[meshlet.triangle_offset + meshlet.triangle_count * 3 + 2] = (unsigned char)cv;
	meshlet.triangle_count++;

	return result;
}

static unsigned int getNeighborTriangle(const meshopt_Meshlet& meshlet, const Cone& meshlet_cone, const unsigned int* meshlet_vertices, const unsigned int* indices, const TriangleAdjacency2& adjacency, const Cone* triangles, const unsigned int* live_triangles, const short* used, float meshlet_expected_radius, float cone_weight)
{
	unsigned int best_triangle = ~0u;
	int best_priority = 5;
	float best_score = FLT_MAX;

	for (size_t i = 0; i < meshlet.vertex_count; ++i)
	{
		unsigned int index = meshlet_vertices[meshlet.vertex_offset + i];

		unsigned int* neighbors = &adjacency.data[0] + adjacency.offsets[index];
		size_t neighbors_size = adjacency.counts[index];

		for (size_t j = 0; j < neighbors_size; ++j)
		{
			unsigned int triangle = neighbors[j];
			unsigned int a = indices[triangle * 3 + 0], b = indices[triangle * 3 + 1], c = indices[triangle * 3 + 2];

			int extra = (used[a] < 0) + (used[b] < 0) + (used[c] < 0);
			assert(extra <= 2);

			int priority = -1;

			// triangles that don't add new vertices to meshlets are max. priority
			if (extra == 0)
				priority = 0;
			// artificially increase the priority of dangling triangles as they're expensive to add to new meshlets
			else if (live_triangles[a] == 1 || live_triangles[b] == 1 || live_triangles[c] == 1)
				priority = 1;
			// if two vertices have live count of 2, removing this triangle will make another triangle dangling which is good for overall flow
			else if ((live_triangles[a] == 2) + (live_triangles[b] == 2) + (live_triangles[c] == 2) >= 2)
				priority = 1 + extra;
			// otherwise adjust priority to be after the above cases, 3 or 4 based on used[] count
			else
				priority = 2 + extra;

			// since topology-based priority is always more important than the score, we can skip scoring in some cases
			if (priority > best_priority)
				continue;

			const Cone& tri_cone = triangles[triangle];

			float dx = tri_cone.px - meshlet_cone.px, dy = tri_cone.py - meshlet_cone.py, dz = tri_cone.pz - meshlet_cone.pz;
			float distance = getDistance(dx, dy, dz, cone_weight < 0);
			float spread = tri_cone.nx * meshlet_cone.nx + tri_cone.ny * meshlet_cone.ny + tri_cone.nz * meshlet_cone.nz;

			float score = getMeshletScore(distance, spread, cone_weight, meshlet_expected_radius);

			// note that topology-based priority is always more important than the score
			// this helps maintain reasonable effectiveness of meshlet data and reduces scoring cost
			if (priority < best_priority || score < best_score)
			{
				best_triangle = triangle;
				best_priority = priority;
				best_score = score;
			}
		}
	}

	return best_triangle;
}

static size_t appendSeedTriangles(unsigned int* seeds, const meshopt_Meshlet& meshlet, const unsigned int* meshlet_vertices, const unsigned int* indices, const TriangleAdjacency2& adjacency, const Cone* triangles, const unsigned int* live_triangles, float cornerx, float cornery, float cornerz)
{
	unsigned int best_seeds[kMeshletAddSeeds];
	unsigned int best_live[kMeshletAddSeeds];
	float best_score[kMeshletAddSeeds];

	for (size_t i = 0; i < kMeshletAddSeeds; ++i)
	{
		best_seeds[i] = ~0u;
		best_live[i] = ~0u;
		best_score[i] = FLT_MAX;
	}

	for (size_t i = 0; i < meshlet.vertex_count; ++i)
	{
		unsigned int index = meshlet_vertices[meshlet.vertex_offset + i];

		unsigned int best_neighbor = ~0u;
		unsigned int best_neighbor_live = ~0u;

		// find the neighbor with the smallest live metric
		unsigned int* neighbors = &adjacency.data[0] + adjacency.offsets[index];
		size_t neighbors_size = adjacency.counts[index];

		for (size_t j = 0; j < neighbors_size; ++j)
		{
			unsigned int triangle = neighbors[j];
			unsigned int a = indices[triangle * 3 + 0], b = indices[triangle * 3 + 1], c = indices[triangle * 3 + 2];

			unsigned int live = live_triangles[a] + live_triangles[b] + live_triangles[c];

			if (live < best_neighbor_live)
			{
				best_neighbor = triangle;
				best_neighbor_live = live;
			}
		}

		// add the neighbor to the list of seeds; the list is unsorted and the replacement criteria is approximate
		if (best_neighbor == ~0u)
			continue;

		float best_neighbor_score = getDistance(triangles[best_neighbor].px - cornerx, triangles[best_neighbor].py - cornery, triangles[best_neighbor].pz - cornerz, false);

		for (size_t j = 0; j < kMeshletAddSeeds; ++j)
		{
			// non-strict comparison reduces the number of duplicate seeds (triangles adjacent to multiple vertices)
			if (best_neighbor_live < best_live[j] || (best_neighbor_live == best_live[j] && best_neighbor_score <= best_score[j]))
			{
				best_seeds[j] = best_neighbor;
				best_live[j] = best_neighbor_live;
				best_score[j] = best_neighbor_score;
				break;
			}
		}
	}

	// add surviving seeds to the meshlet
	size_t seed_count = 0;

	for (size_t i = 0; i < kMeshletAddSeeds; ++i)
		if (best_seeds[i] != ~0u)
			seeds[seed_count++] = best_seeds[i];

	return seed_count;
}

static size_t pruneSeedTriangles(unsigned int* seeds, size_t seed_count, const unsigned char* emitted_flags)
{
	size_t result = 0;

	for (size_t i = 0; i < seed_count; ++i)
	{
		unsigned int index = seeds[i];

		seeds[result] = index;
		result += emitted_flags[index] == 0;
	}

	return result;
}

static unsigned int selectSeedTriangle(const unsigned int* seeds, size_t seed_count, const unsigned int* indices, const Cone* triangles, const unsigned int* live_triangles, float cornerx, float cornery, float cornerz)
{
	unsigned int best_seed = ~0u;
	unsigned int best_live = ~0u;
	float best_score = FLT_MAX;

	for (size_t i = 0; i < seed_count; ++i)
	{
		unsigned int index = seeds[i];
		unsigned int a = indices[index * 3 + 0], b = indices[index * 3 + 1], c = indices[index * 3 + 2];

		unsigned int live = live_triangles[a] + live_triangles[b] + live_triangles[c];
		float score = getDistance(triangles[index].px - cornerx, triangles[index].py - cornery, triangles[index].pz - cornerz, false);

		if (live < best_live || (live == best_live && score < best_score))
		{
			best_seed = index;
			best_live = live;
			best_score = score;
		}
	}

	return best_seed;
}

struct KDNode
{
	union
	{
		float split;
		unsigned int index;
	};

	// leaves: axis = 3, children = number of extra points after this one (0 if 'index' is the only point)
	// branches: axis != 3, left subtree = skip 1, right subtree = skip 1+children
	unsigned int axis : 2;
	unsigned int children : 30;
};

static size_t kdtreePartition(unsigned int* indices, size_t count, const float* points, size_t stride, unsigned int axis, float pivot)
{
	size_t m = 0;

	// invariant: elements in range [0, m) are < pivot, elements in range [m, i) are >= pivot
	for (size_t i = 0; i < count; ++i)
	{
		float v = points[indices[i] * stride + axis];

		// swap(m, i) unconditionally
		unsigned int t = indices[m];
		indices[m] = indices[i];
		indices[i] = t;

		// when v >= pivot, we swap i with m without advancing it, preserving invariants
		m += v < pivot;
	}

	return m;
}

static size_t kdtreeBuildLeaf(size_t offset, KDNode* nodes, size_t node_count, unsigned int* indices, size_t count)
{
	assert(offset + count <= node_count);
	(void)node_count;

	KDNode& result = nodes[offset];

	result.index = indices[0];
	result.axis = 3;
	result.children = unsigned(count - 1);

	// all remaining points are stored in nodes immediately following the leaf
	for (size_t i = 1; i < count; ++i)
	{
		KDNode& tail = nodes[offset + i];

		tail.index = indices[i];
		tail.axis = 3;
		tail.children = ~0u >> 2; // bogus value to prevent misuse
	}

	return offset + count;
}

static size_t kdtreeBuild(size_t offset, KDNode* nodes, size_t node_count, const float* points, size_t stride, unsigned int* indices, size_t count, size_t leaf_size)
{
	assert(count > 0);
	assert(offset < node_count);

	if (count <= leaf_size)
		return kdtreeBuildLeaf(offset, nodes, node_count, indices, count);

	float mean[3] = {};
	float vars[3] = {};
	float runc = 1, runs = 1;

	// gather statistics on the points in the subtree using Welford's algorithm
	for (size_t i = 0; i < count; ++i, runc += 1.f, runs = 1.f / runc)
	{
		const float* point = points + indices[i] * stride;

		for (int k = 0; k < 3; ++k)
		{
			float delta = point[k] - mean[k];
			mean[k] += delta * runs;
			vars[k] += delta * (point[k] - mean[k]);
		}
	}

	// split axis is one where the variance is largest
	unsigned int axis = (vars[0] >= vars[1] && vars[0] >= vars[2]) ? 0 : (vars[1] >= vars[2] ? 1 : 2);

	float split = mean[axis];
	size_t middle = kdtreePartition(indices, count, points, stride, axis, split);

	// when the partition is degenerate simply consolidate the points into a single node
	if (middle <= leaf_size / 2 || middle >= count - leaf_size / 2)
		return kdtreeBuildLeaf(offset, nodes, node_count, indices, count);

	KDNode& result = nodes[offset];

	result.split = split;
	result.axis = axis;

	// left subtree is right after our node
	size_t next_offset = kdtreeBuild(offset + 1, nodes, node_count, points, stride, indices, middle, leaf_size);

	// distance to the right subtree is represented explicitly
	result.children = unsigned(next_offset - offset - 1);

	return kdtreeBuild(next_offset, nodes, node_count, points, stride, indices + middle, count - middle, leaf_size);
}

static void kdtreeNearest(KDNode* nodes, unsigned int root, const float* points, size_t stride, const unsigned char* emitted_flags, const float* position, bool aa, unsigned int& result, float& limit)
{
	const KDNode& node = nodes[root];

	if (node.axis == 3)
	{
		// leaf
		for (unsigned int i = 0; i <= node.children; ++i)
		{
			unsigned int index = nodes[root + i].index;

			if (emitted_flags[index])
				continue;

			const float* point = points + index * stride;

			float dx = point[0] - position[0], dy = point[1] - position[1], dz = point[2] - position[2];
			float distance = getDistance(dx, dy, dz, aa);

			if (distance < limit)
			{
				result = index;
				limit = distance;
			}
		}
	}
	else
	{
		// branch; we order recursion to process the node that search position is in first
		float delta = position[node.axis] - node.split;
		unsigned int first = (delta <= 0) ? 0 : node.children;
		unsigned int second = first ^ node.children;

		kdtreeNearest(nodes, root + 1 + first, points, stride, emitted_flags, position, aa, result, limit);

		// only process the other node if it can have a match based on closest distance so far
		if (fabsf(delta) <= limit)
			kdtreeNearest(nodes, root + 1 + second, points, stride, emitted_flags, position, aa, result, limit);
	}
}

struct BVHBox
{
	float min[3];
	float max[3];
};

static void boxMerge(BVHBox& box, const BVHBox& other)
{
	for (int k = 0; k < 3; ++k)
	{
		box.min[k] = other.min[k] < box.min[k] ? other.min[k] : box.min[k];
		box.max[k] = other.max[k] > box.max[k] ? other.max[k] : box.max[k];
	}
}

inline float boxSurface(const BVHBox& box)
{
	float sx = box.max[0] - box.min[0], sy = box.max[1] - box.min[1], sz = box.max[2] - box.min[2];
	return sx * sy + sx * sz + sy * sz;
}

inline unsigned int radixFloat(unsigned int v)
{
	// if sign bit is 0, flip sign bit
	// if sign bit is 1, flip everything
	unsigned int mask = (int(v) >> 31) | 0x80000000;
	return v ^ mask;
}

static void computeHistogram(unsigned int (&hist)[1024][3], const float* data, size_t count)
{
	memset(hist, 0, sizeof(hist));

	const unsigned int* bits = reinterpret_cast<const unsigned int*>(data);

	// compute 3 10-bit histograms in parallel (dropping 2 LSB)
	for (size_t i = 0; i < count; ++i)
	{
		unsigned int id = radixFloat(bits[i]);

		hist[(id >> 2) & 1023][0]++;
		hist[(id >> 12) & 1023][1]++;
		hist[(id >> 22) & 1023][2]++;
	}

	unsigned int sum0 = 0, sum1 = 0, sum2 = 0;

	// replace histogram data with prefix histogram sums in-place
	for (int i = 0; i < 1024; ++i)
	{
		unsigned int hx = hist[i][0], hy = hist[i][1], hz = hist[i][2];

		hist[i][0] = sum0;
		hist[i][1] = sum1;
		hist[i][2] = sum2;

		sum0 += hx;
		sum1 += hy;
		sum2 += hz;
	}

	assert(sum0 == count && sum1 == count && sum2 == count);
}

static void radixPass(unsigned int* destination, const unsigned int* source, const float* keys, size_t count, unsigned int (&hist)[1024][3], int pass)
{
	const unsigned int* bits = reinterpret_cast<const unsigned int*>(keys);
	int bitoff = pass * 10 + 2; // drop 2 LSB to be able to use 3 10-bit passes

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int id = (radixFloat(bits[source[i]]) >> bitoff) & 1023;

		destination[hist[id][pass]++] = source[i];
	}
}

static void bvhPrepare(BVHBox* boxes, float* centroids, const unsigned int* indices, size_t face_count, const float* vertex_positions, size_t vertex_count, size_t vertex_stride_float)
{
	(void)vertex_count;

	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		const float* va = vertex_positions + vertex_stride_float * a;
		const float* vb = vertex_positions + vertex_stride_float * b;
		const float* vc = vertex_positions + vertex_stride_float * c;

		BVHBox& box = boxes[i];

		for (int k = 0; k < 3; ++k)
		{
			box.min[k] = va[k] < vb[k] ? va[k] : vb[k];
			box.min[k] = vc[k] < box.min[k] ? vc[k] : box.min[k];

			box.max[k] = va[k] > vb[k] ? va[k] : vb[k];
			box.max[k] = vc[k] > box.max[k] ? vc[k] : box.max[k];

			centroids[i + face_count * k] = (box.min[k] + box.max[k]) / 2.f;
		}
	}
}

static bool bvhPackLeaf(unsigned char* boundary, const unsigned int* order, size_t count, short* used, const unsigned int* indices, size_t max_vertices)
{
	// count number of unique vertices
	size_t used_vertices = 0;
	for (size_t i = 0; i < count; ++i)
	{
		unsigned int index = order[i];
		unsigned int a = indices[index * 3 + 0], b = indices[index * 3 + 1], c = indices[index * 3 + 2];

		used_vertices += (used[a] < 0) + (used[b] < 0) + (used[c] < 0);
		used[a] = used[b] = used[c] = 1;
	}

	// reset used[] for future invocations
	for (size_t i = 0; i < count; ++i)
	{
		unsigned int index = order[i];
		unsigned int a = indices[index * 3 + 0], b = indices[index * 3 + 1], c = indices[index * 3 + 2];

		used[a] = used[b] = used[c] = -1;
	}

	if (used_vertices > max_vertices)
		return false;

	// mark meshlet boundary for future reassembly
	assert(count > 0);

	boundary[0] = 1;
	memset(boundary + 1, 0, count - 1);

	return true;
}

static void bvhPackTail(unsigned char* boundary, const unsigned int* order, size_t count, short* used, const unsigned int* indices, size_t max_vertices, size_t max_triangles)
{
	for (size_t i = 0; i < count;)
	{
		size_t chunk = i + max_triangles <= count ? max_triangles : count - i;

		if (bvhPackLeaf(boundary + i, order + i, chunk, used, indices, max_vertices))
		{
			i += chunk;
			continue;
		}

		// chunk is vertex bound, split it into smaller meshlets
		assert(chunk > max_vertices / 3);

		bvhPackLeaf(boundary + i, order + i, max_vertices / 3, used, indices, max_vertices);
		i += max_vertices / 3;
	}
}

static bool bvhDivisible(size_t count, size_t min, size_t max)
{
	// count is representable as a sum of values in [min..max] if if it in range of [k*min..k*min+k*(max-min)]
	// equivalent to ceil(count / max) <= floor(count / min), but the form below allows using idiv (see nv_cluster_builder)
	// we avoid expensive integer divisions in the common case where min is <= max/2
	return min * 2 <= max ? count >= min : count % min <= (count / min) * (max - min);
}

static size_t bvhPivot(const BVHBox* boxes, const unsigned int* order, size_t count, void* scratch, size_t step, size_t min, size_t max, float fill, float* out_cost)
{
	BVHBox accuml = boxes[order[0]], accumr = boxes[order[count - 1]];
	float* costs = static_cast<float*>(scratch);

	// accumulate SAH cost in forward and backward directions
	for (size_t i = 0; i < count; ++i)
	{
		boxMerge(accuml, boxes[order[i]]);
		boxMerge(accumr, boxes[order[count - 1 - i]]);

		costs[i] = boxSurface(accuml);
		costs[i + count] = boxSurface(accumr);
	}

	bool aligned = count >= min * 2 && bvhDivisible(count, min, max);
	size_t end = aligned ? count - min : count - 1;

	float rmaxf = 1.f / float(int(max));

	// find best split that minimizes SAH
	size_t bestsplit = 0;
	float bestcost = FLT_MAX;

	for (size_t i = min - 1; i < end; i += step)
	{
		size_t lsplit = i + 1, rsplit = count - (i + 1);

		if (!bvhDivisible(lsplit, min, max))
			continue;
		if (aligned && !bvhDivisible(rsplit, min, max))
			continue;

		// costs[x] = inclusive surface area of boxes[0..x]
		// costs[count-1-x] = inclusive surface area of boxes[x..count-1]
		float larea = costs[i], rarea = costs[(count - 1 - (i + 1)) + count];
		float cost = larea * float(int(lsplit)) + rarea * float(int(rsplit));

		if (cost > bestcost)
			continue;

		// fill cost; use floating point math to avoid expensive integer modulo
		int lrest = int(float(int(lsplit + max - 1)) * rmaxf) * int(max) - int(lsplit);
		int rrest = int(float(int(rsplit + max - 1)) * rmaxf) * int(max) - int(rsplit);

		cost += fill * (float(lrest) * larea + float(rrest) * rarea);

		if (cost < bestcost)
		{
			bestcost = cost;
			bestsplit = i + 1;
		}
	}

	*out_cost = bestcost;
	return bestsplit;
}

static void bvhPartition(unsigned int* target, const unsigned int* order, const unsigned char* sides, size_t split, size_t count)
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

static void bvhSplit(const BVHBox* boxes, unsigned int* orderx, unsigned int* ordery, unsigned int* orderz, unsigned char* boundary, size_t count, int depth, void* scratch, short* used, const unsigned int* indices, size_t max_vertices, size_t min_triangles, size_t max_triangles, float fill_weight)
{
	if (depth >= kMeshletMaxTreeDepth)
		return bvhPackTail(boundary, orderx, count, used, indices, max_vertices, max_triangles);

	if (count <= max_triangles && bvhPackLeaf(boundary, orderx, count, used, indices, max_vertices))
		return;

	unsigned int* axes[3] = {orderx, ordery, orderz};

	// we can use step=1 unconditionally but to reduce the cost for min=max case we use step=max
	size_t step = min_triangles == max_triangles && count > max_triangles ? max_triangles : 1;

	// if we could not pack the meshlet, we must be vertex bound
	size_t mint = count <= max_triangles && max_vertices / 3 < min_triangles ? max_vertices / 3 : min_triangles;

	// only use fill weight if we are optimizing for triangle count
	float fill = count <= max_triangles ? 0.f : fill_weight;

	// find best split that minimizes SAH
	int bestk = -1;
	size_t bestsplit = 0;
	float bestcost = FLT_MAX;

	for (int k = 0; k < 3; ++k)
	{
		float axiscost = FLT_MAX;
		size_t axissplit = bvhPivot(boxes, axes[k], count, scratch, step, mint, max_triangles, fill, &axiscost);

		if (axissplit && axiscost < bestcost)
		{
			bestk = k;
			bestcost = axiscost;
			bestsplit = axissplit;
		}
	}

	// this may happen if SAH costs along the admissible splits are NaN
	if (bestk < 0)
		return bvhPackTail(boundary, orderx, count, used, indices, max_vertices, max_triangles);

	// mark sides of split for partitioning
	unsigned char* sides = static_cast<unsigned char*>(scratch) + count * sizeof(unsigned int);

	for (size_t i = 0; i < bestsplit; ++i)
		sides[axes[bestk][i]] = 0;

	for (size_t i = bestsplit; i < count; ++i)
		sides[axes[bestk][i]] = 1;

	// partition all axes into two sides, maintaining order
	unsigned int* temp = static_cast<unsigned int*>(scratch);

	for (int k = 0; k < 3; ++k)
	{
		if (k == bestk)
			continue;

		unsigned int* axis = axes[k];
		memcpy(temp, axis, sizeof(unsigned int) * count);
		bvhPartition(axis, temp, sides, bestsplit, count);
	}

	bvhSplit(boxes, orderx, ordery, orderz, boundary, bestsplit, depth + 1, scratch, used, indices, max_vertices, min_triangles, max_triangles, fill_weight);
	bvhSplit(boxes, orderx + bestsplit, ordery + bestsplit, orderz + bestsplit, boundary + bestsplit, count - bestsplit, depth + 1, scratch, used, indices, max_vertices, min_triangles, max_triangles, fill_weight);
}

} // namespace meshopt

size_t meshopt_buildMeshletsBound(size_t index_count, size_t max_vertices, size_t max_triangles)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(max_vertices >= 3 && max_vertices <= kMeshletMaxVertices);
	assert(max_triangles >= 1 && max_triangles <= kMeshletMaxTriangles);
	assert(max_triangles % 4 == 0); // ensures the caller will compute output space properly as index data is 4b aligned

	(void)kMeshletMaxVertices;
	(void)kMeshletMaxTriangles;

	// meshlet construction is limited by max vertices and max triangles per meshlet
	// the worst case is that the input is an unindexed stream since this equally stresses both limits
	// note that we assume that in the worst case, we leave 2 vertices unpacked in each meshlet - if we have space for 3 we can pack any triangle
	size_t max_vertices_conservative = max_vertices - 2;
	size_t meshlet_limit_vertices = (index_count + max_vertices_conservative - 1) / max_vertices_conservative;
	size_t meshlet_limit_triangles = (index_count / 3 + max_triangles - 1) / max_triangles;

	return meshlet_limit_vertices > meshlet_limit_triangles ? meshlet_limit_vertices : meshlet_limit_triangles;
}

size_t meshopt_buildMeshletsFlex(meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, size_t max_vertices, size_t min_triangles, size_t max_triangles, float cone_weight, float split_factor)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	assert(max_vertices >= 3 && max_vertices <= kMeshletMaxVertices);
	assert(min_triangles >= 1 && min_triangles <= max_triangles && max_triangles <= kMeshletMaxTriangles);
	assert(min_triangles % 4 == 0 && max_triangles % 4 == 0); // ensures the caller will compute output space properly as index data is 4b aligned

	assert(cone_weight <= 1); // negative cone weight switches metric to optimize for axis-aligned meshlets
	assert(split_factor >= 0);

	if (index_count == 0)
		return 0;

	meshopt_Allocator allocator;

	TriangleAdjacency2 adjacency = {};
	if (vertex_count > index_count && index_count < (1u << 31))
		buildTriangleAdjacencySparse(adjacency, indices, index_count, vertex_count, allocator);
	else
		buildTriangleAdjacency(adjacency, indices, index_count, vertex_count, allocator);

	// live triangle counts; note, we alias adjacency.counts as we remove triangles after emitting them so the counts always match
	unsigned int* live_triangles = adjacency.counts;

	size_t face_count = index_count / 3;

	unsigned char* emitted_flags = allocator.allocate<unsigned char>(face_count);
	memset(emitted_flags, 0, face_count);

	// for each triangle, precompute centroid & normal to use for scoring
	Cone* triangles = allocator.allocate<Cone>(face_count);
	float mesh_area = computeTriangleCones(triangles, indices, index_count, vertex_positions, vertex_count, vertex_positions_stride);

	// assuming each meshlet is a square patch, expected radius is sqrt(expected area)
	float triangle_area_avg = face_count == 0 ? 0.f : mesh_area / float(face_count) * 0.5f;
	float meshlet_expected_radius = sqrtf(triangle_area_avg * max_triangles) * 0.5f;

	// build a kd-tree for nearest neighbor lookup
	unsigned int* kdindices = allocator.allocate<unsigned int>(face_count);
	for (size_t i = 0; i < face_count; ++i)
		kdindices[i] = unsigned(i);

	KDNode* nodes = allocator.allocate<KDNode>(face_count * 2);
	kdtreeBuild(0, nodes, face_count * 2, &triangles[0].px, sizeof(Cone) / sizeof(float), kdindices, face_count, /* leaf_size= */ 8);

	// find a specific corner of the mesh to use as a starting point for meshlet flow
	float cornerx = FLT_MAX, cornery = FLT_MAX, cornerz = FLT_MAX;

	for (size_t i = 0; i < face_count; ++i)
	{
		const Cone& tri = triangles[i];

		cornerx = cornerx > tri.px ? tri.px : cornerx;
		cornery = cornery > tri.py ? tri.py : cornery;
		cornerz = cornerz > tri.pz ? tri.pz : cornerz;
	}

	// index of the vertex in the meshlet, -1 if the vertex isn't used
	short* used = allocator.allocate<short>(vertex_count);
	memset(used, -1, vertex_count * sizeof(short));

	// initial seed triangle is the one closest to the corner
	unsigned int initial_seed = ~0u;
	float initial_score = FLT_MAX;

	for (size_t i = 0; i < face_count; ++i)
	{
		const Cone& tri = triangles[i];

		float score = getDistance(tri.px - cornerx, tri.py - cornery, tri.pz - cornerz, false);

		if (initial_seed == ~0u || score < initial_score)
		{
			initial_seed = unsigned(i);
			initial_score = score;
		}
	}

	// seed triangles to continue meshlet flow
	unsigned int seeds[kMeshletMaxSeeds] = {};
	size_t seed_count = 0;

	meshopt_Meshlet meshlet = {};
	size_t meshlet_offset = 0;

	Cone meshlet_cone_acc = {};

	for (;;)
	{
		Cone meshlet_cone = getMeshletCone(meshlet_cone_acc, meshlet.triangle_count);

		unsigned int best_triangle = ~0u;

		// for the first triangle, we don't have a meshlet cone yet, so we use the initial seed
		// to continue the meshlet, we select an adjacent triangle based on connectivity and spatial scoring
		if (meshlet_offset == 0 && meshlet.triangle_count == 0)
			best_triangle = initial_seed;
		else
			best_triangle = getNeighborTriangle(meshlet, meshlet_cone, meshlet_vertices, indices, adjacency, triangles, live_triangles, used, meshlet_expected_radius, cone_weight);

		bool split = false;

		// when we run out of adjacent triangles we need to switch to spatial search; we currently just pick the closest triangle irrespective of connectivity
		if (best_triangle == ~0u)
		{
			float position[3] = {meshlet_cone.px, meshlet_cone.py, meshlet_cone.pz};
			unsigned int index = ~0u;
			float distance = FLT_MAX;

			kdtreeNearest(nodes, 0, &triangles[0].px, sizeof(Cone) / sizeof(float), emitted_flags, position, cone_weight < 0.f, index, distance);

			best_triangle = index;
			split = meshlet.triangle_count >= min_triangles && split_factor > 0 && distance > meshlet_expected_radius * split_factor;
		}

		if (best_triangle == ~0u)
			break;

		int best_extra = (used[indices[best_triangle * 3 + 0]] < 0) + (used[indices[best_triangle * 3 + 1]] < 0) + (used[indices[best_triangle * 3 + 2]] < 0);

		// if the best triangle doesn't fit into current meshlet, we re-select using seeds to maintain global flow
		if (split || (meshlet.vertex_count + best_extra > max_vertices || meshlet.triangle_count >= max_triangles))
		{
			seed_count = pruneSeedTriangles(seeds, seed_count, emitted_flags);
			seed_count = (seed_count + kMeshletAddSeeds <= kMeshletMaxSeeds) ? seed_count : kMeshletMaxSeeds - kMeshletAddSeeds;
			seed_count += appendSeedTriangles(seeds + seed_count, meshlet, meshlet_vertices, indices, adjacency, triangles, live_triangles, cornerx, cornery, cornerz);

			unsigned int best_seed = selectSeedTriangle(seeds, seed_count, indices, triangles, live_triangles, cornerx, cornery, cornerz);

			// we may not find a valid seed triangle if the mesh is disconnected as seeds are based on adjacency
			best_triangle = best_seed != ~0u ? best_seed : best_triangle;
		}

		unsigned int a = indices[best_triangle * 3 + 0], b = indices[best_triangle * 3 + 1], c = indices[best_triangle * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		// add meshlet to the output; when the current meshlet is full we reset the accumulated bounds
		if (appendMeshlet(meshlet, a, b, c, used, meshlets, meshlet_vertices, meshlet_triangles, meshlet_offset, max_vertices, max_triangles, split))
		{
			meshlet_offset++;
			memset(&meshlet_cone_acc, 0, sizeof(meshlet_cone_acc));
		}

		// remove emitted triangle from adjacency data
		// this makes sure that we spend less time traversing these lists on subsequent iterations
		// live triangle counts are updated as a byproduct of these adjustments
		for (size_t k = 0; k < 3; ++k)
		{
			unsigned int index = indices[best_triangle * 3 + k];

			unsigned int* neighbors = &adjacency.data[0] + adjacency.offsets[index];
			size_t neighbors_size = adjacency.counts[index];

			for (size_t i = 0; i < neighbors_size; ++i)
			{
				unsigned int tri = neighbors[i];

				if (tri == best_triangle)
				{
					neighbors[i] = neighbors[neighbors_size - 1];
					adjacency.counts[index]--;
					break;
				}
			}
		}

		// update aggregated meshlet cone data for scoring subsequent triangles
		meshlet_cone_acc.px += triangles[best_triangle].px;
		meshlet_cone_acc.py += triangles[best_triangle].py;
		meshlet_cone_acc.pz += triangles[best_triangle].pz;
		meshlet_cone_acc.nx += triangles[best_triangle].nx;
		meshlet_cone_acc.ny += triangles[best_triangle].ny;
		meshlet_cone_acc.nz += triangles[best_triangle].nz;

		assert(!emitted_flags[best_triangle]);
		emitted_flags[best_triangle] = 1;
	}

	if (meshlet.triangle_count)
	{
		finishMeshlet(meshlet, meshlet_triangles);

		meshlets[meshlet_offset++] = meshlet;
	}

	assert(meshlet_offset <= meshopt_buildMeshletsBound(index_count, max_vertices, min_triangles));
	return meshlet_offset;
}

size_t meshopt_buildMeshlets(meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, size_t max_vertices, size_t max_triangles, float cone_weight)
{
	assert(cone_weight >= 0); // to use negative cone weight, use meshopt_buildMeshletsFlex

	return meshopt_buildMeshletsFlex(meshlets, meshlet_vertices, meshlet_triangles, indices, index_count, vertex_positions, vertex_count, vertex_positions_stride, max_vertices, max_triangles, max_triangles, cone_weight, 0.0f);
}

size_t meshopt_buildMeshletsScan(meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, const unsigned int* indices, size_t index_count, size_t vertex_count, size_t max_vertices, size_t max_triangles)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);

	assert(max_vertices >= 3 && max_vertices <= kMeshletMaxVertices);
	assert(max_triangles >= 1 && max_triangles <= kMeshletMaxTriangles);
	assert(max_triangles % 4 == 0); // ensures the caller will compute output space properly as index data is 4b aligned

	meshopt_Allocator allocator;

	// index of the vertex in the meshlet, -1 if the vertex isn't used
	short* used = allocator.allocate<short>(vertex_count);
	memset(used, -1, vertex_count * sizeof(short));

	meshopt_Meshlet meshlet = {};
	size_t meshlet_offset = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		// appends triangle to the meshlet and writes previous meshlet to the output if full
		meshlet_offset += appendMeshlet(meshlet, a, b, c, used, meshlets, meshlet_vertices, meshlet_triangles, meshlet_offset, max_vertices, max_triangles);
	}

	if (meshlet.triangle_count)
	{
		finishMeshlet(meshlet, meshlet_triangles);

		meshlets[meshlet_offset++] = meshlet;
	}

	assert(meshlet_offset <= meshopt_buildMeshletsBound(index_count, max_vertices, max_triangles));
	return meshlet_offset;
}

size_t meshopt_buildMeshletsSpatial(struct meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, size_t max_vertices, size_t min_triangles, size_t max_triangles, float fill_weight)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	assert(max_vertices >= 3 && max_vertices <= kMeshletMaxVertices);
	assert(min_triangles >= 1 && min_triangles <= max_triangles && max_triangles <= kMeshletMaxTriangles);
	assert(min_triangles % 4 == 0 && max_triangles % 4 == 0); // ensures the caller will compute output space properly as index data is 4b aligned

	if (index_count == 0)
		return 0;

	size_t face_count = index_count / 3;
	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	meshopt_Allocator allocator;

	// 3 floats plus 1 uint for sorting, or
	// 2 floats for SAH costs, or
	// 1 uint plus 1 byte for partitioning
	float* scratch = allocator.allocate<float>(face_count * 4);

	// compute bounding boxes and centroids for sorting
	BVHBox* boxes = allocator.allocate<BVHBox>(face_count);
	bvhPrepare(boxes, scratch, indices, face_count, vertex_positions, vertex_count, vertex_stride_float);

	unsigned int* axes = allocator.allocate<unsigned int>(face_count * 3);
	unsigned int* temp = reinterpret_cast<unsigned int*>(scratch) + face_count * 3;

	for (int k = 0; k < 3; ++k)
	{
		unsigned int* order = axes + k * face_count;
		const float* keys = scratch + k * face_count;

		unsigned int hist[1024][3];
		computeHistogram(hist, keys, face_count);

		// 3-pass radix sort computes the resulting order into axes
		for (size_t i = 0; i < face_count; ++i)
			temp[i] = unsigned(i);

		radixPass(order, temp, keys, face_count, hist, 0);
		radixPass(temp, order, keys, face_count, hist, 1);
		radixPass(order, temp, keys, face_count, hist, 2);
	}

	// index of the vertex in the meshlet, -1 if the vertex isn't used
	short* used = allocator.allocate<short>(vertex_count);
	memset(used, -1, vertex_count * sizeof(short));

	unsigned char* boundary = allocator.allocate<unsigned char>(face_count);

	bvhSplit(boxes, &axes[0], &axes[face_count], &axes[face_count * 2], boundary, face_count, 0, scratch, used, indices, max_vertices, min_triangles, max_triangles, fill_weight);

	// compute the desired number of meshlets; note that on some meshes with a lot of vertex bound clusters this might go over the bound
	size_t meshlet_count = 0;
	for (size_t i = 0; i < face_count; ++i)
	{
		assert(boundary[i] <= 1);
		meshlet_count += boundary[i];
	}

	size_t meshlet_bound = meshopt_buildMeshletsBound(index_count, max_vertices, min_triangles);

	// pack triangles into meshlets according to the order and boundaries marked by bvhSplit
	meshopt_Meshlet meshlet = {};
	size_t meshlet_offset = 0;
	size_t meshlet_pending = meshlet_count;

	for (size_t i = 0; i < face_count; ++i)
	{
		assert(boundary[i] <= 1);
		bool split = i > 0 && boundary[i] == 1;

		// while we are over the limit, we ignore boundary[] data and disable splits until we free up enough space
		if (split && meshlet_count > meshlet_bound && meshlet_offset + meshlet_pending >= meshlet_bound)
			split = false;

		unsigned int index = axes[i];
		assert(index < face_count);

		unsigned int a = indices[index * 3 + 0], b = indices[index * 3 + 1], c = indices[index * 3 + 2];

		// appends triangle to the meshlet and writes previous meshlet to the output if full
		meshlet_offset += appendMeshlet(meshlet, a, b, c, used, meshlets, meshlet_vertices, meshlet_triangles, meshlet_offset, max_vertices, max_triangles, split);
		meshlet_pending -= boundary[i];
	}

	if (meshlet.triangle_count)
	{
		finishMeshlet(meshlet, meshlet_triangles);

		meshlets[meshlet_offset++] = meshlet;
	}

	assert(meshlet_offset <= meshlet_bound);
	return meshlet_offset;
}

meshopt_Bounds meshopt_computeClusterBounds(const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(index_count / 3 <= kMeshletMaxTriangles);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	(void)vertex_count;

	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	// compute triangle normals and gather triangle corners
	float normals[kMeshletMaxTriangles][3];
	float corners[kMeshletMaxTriangles][3][3];
	size_t triangles = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

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

		// record triangle normals & corners for future use; normal and corner 0 define a plane equation
		normals[triangles][0] = normalx / area;
		normals[triangles][1] = normaly / area;
		normals[triangles][2] = normalz / area;
		memcpy(corners[triangles][0], p0, 3 * sizeof(float));
		memcpy(corners[triangles][1], p1, 3 * sizeof(float));
		memcpy(corners[triangles][2], p2, 3 * sizeof(float));
		triangles++;
	}

	meshopt_Bounds bounds = {};

	// degenerate cluster, no valid triangles => trivial reject (cone data is 0)
	if (triangles == 0)
		return bounds;

	const float rzero = 0.f;

	// compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
	float psphere[4] = {};
	computeBoundingSphere(psphere, corners[0][0], triangles * 3, sizeof(float) * 3, &rzero, 0, 7);

	float center[3] = {psphere[0], psphere[1], psphere[2]};

	// treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
	float nsphere[4] = {};
	computeBoundingSphere(nsphere, normals[0], triangles, sizeof(float) * 3, &rzero, 0, 3);

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
		float cx = center[0] - corners[i][0][0];
		float cy = center[1] - corners[i][0][1];
		float cz = center[2] - corners[i][0][2];

		float dc = cx * normals[i][0] + cy * normals[i][1] + cz * normals[i][2];
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

meshopt_Bounds meshopt_computeMeshletBounds(const unsigned int* meshlet_vertices, const unsigned char* meshlet_triangles, size_t triangle_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(triangle_count <= kMeshletMaxTriangles);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	unsigned int indices[kMeshletMaxTriangles * 3];

	for (size_t i = 0; i < triangle_count * 3; ++i)
	{
		unsigned int index = meshlet_vertices[meshlet_triangles[i]];
		assert(index < vertex_count);

		indices[i] = index;
	}

	return meshopt_computeClusterBounds(indices, triangle_count * 3, vertex_positions, vertex_count, vertex_positions_stride);
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

void meshopt_optimizeMeshlet(unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, size_t triangle_count, size_t vertex_count)
{
	using namespace meshopt;

	assert(triangle_count <= kMeshletMaxTriangles);
	assert(vertex_count <= kMeshletMaxVertices);

	unsigned char* indices = meshlet_triangles;
	unsigned int* vertices = meshlet_vertices;

	// cache tracks vertex timestamps (corresponding to triangle index! all 3 vertices are added at the same time and never removed)
	unsigned char cache[kMeshletMaxVertices];
	memset(cache, 0, vertex_count);

	// note that we start from a value that means all vertices aren't in cache
	unsigned char cache_last = 128;
	const unsigned char cache_cutoff = 3; // 3 triangles = ~5..9 vertices depending on reuse

	for (size_t i = 0; i < triangle_count; ++i)
	{
		int next = -1;
		int next_match = -1;

		for (size_t j = i; j < triangle_count; ++j)
		{
			unsigned char a = indices[j * 3 + 0], b = indices[j * 3 + 1], c = indices[j * 3 + 2];
			assert(a < vertex_count && b < vertex_count && c < vertex_count);

			// score each triangle by how many vertices are in cache
			// note: the distance is computed using unsigned 8-bit values, so cache timestamp overflow is handled gracefully
			int aok = (unsigned char)(cache_last - cache[a]) < cache_cutoff;
			int bok = (unsigned char)(cache_last - cache[b]) < cache_cutoff;
			int cok = (unsigned char)(cache_last - cache[c]) < cache_cutoff;

			if (aok + bok + cok > next_match)
			{
				next = (int)j;
				next_match = aok + bok + cok;

				// note that we could end up with all 3 vertices in the cache, but 2 is enough for ~strip traversal
				if (next_match >= 2)
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
