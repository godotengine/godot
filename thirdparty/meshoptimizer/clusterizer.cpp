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
namespace meshopt
{

// This must be <= 255 since index 0xff is used internally to indice a vertex that doesn't belong to a meshlet
const size_t kMeshletMaxVertices = 255;

// A reasonable limit is around 2*max_vertices or less
const size_t kMeshletMaxTriangles = 512;

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

static void computeBoundingSphere(float result[4], const float points[][3], size_t count)
{
	assert(count > 0);

	// find extremum points along all 3 axes; for each axis we get a pair of points with min/max coordinates
	size_t pmin[3] = {0, 0, 0};
	size_t pmax[3] = {0, 0, 0};

	for (size_t i = 0; i < count; ++i)
	{
		const float* p = points[i];

		for (int axis = 0; axis < 3; ++axis)
		{
			pmin[axis] = (p[axis] < points[pmin[axis]][axis]) ? i : pmin[axis];
			pmax[axis] = (p[axis] > points[pmax[axis]][axis]) ? i : pmax[axis];
		}
	}

	// find the pair of points with largest distance
	float paxisd2 = 0;
	int paxis = 0;

	for (int axis = 0; axis < 3; ++axis)
	{
		const float* p1 = points[pmin[axis]];
		const float* p2 = points[pmax[axis]];

		float d2 = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]);

		if (d2 > paxisd2)
		{
			paxisd2 = d2;
			paxis = axis;
		}
	}

	// use the longest segment as the initial sphere diameter
	const float* p1 = points[pmin[paxis]];
	const float* p2 = points[pmax[paxis]];

	float center[3] = {(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2};
	float radius = sqrtf(paxisd2) / 2;

	// iteratively adjust the sphere up until all points fit
	for (size_t i = 0; i < count; ++i)
	{
		const float* p = points[i];
		float d2 = (p[0] - center[0]) * (p[0] - center[0]) + (p[1] - center[1]) * (p[1] - center[1]) + (p[2] - center[2]) * (p[2] - center[2]);

		if (d2 > radius * radius)
		{
			float d = sqrtf(d2);
			assert(d > 0);

			float k = 0.5f + (radius / d) / 2;

			center[0] = center[0] * k + p[0] * (1 - k);
			center[1] = center[1] * k + p[1] * (1 - k);
			center[2] = center[2] * k + p[2] * (1 - k);
			radius = (radius + d) / 2;
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

static float getMeshletScore(float distance2, float spread, float cone_weight, float expected_radius)
{
	float cone = 1.f - spread * cone_weight;
	float cone_clamped = cone < 1e-3f ? 1e-3f : cone;

	return (1 + sqrtf(distance2) / expected_radius * (1 - cone_weight)) * cone_clamped;
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

static bool appendMeshlet(meshopt_Meshlet& meshlet, unsigned int a, unsigned int b, unsigned int c, unsigned char* used, meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, size_t meshlet_offset, size_t max_vertices, size_t max_triangles)
{
	unsigned char& av = used[a];
	unsigned char& bv = used[b];
	unsigned char& cv = used[c];

	bool result = false;

	int used_extra = (av == 0xff) + (bv == 0xff) + (cv == 0xff);

	if (meshlet.vertex_count + used_extra > max_vertices || meshlet.triangle_count >= max_triangles)
	{
		meshlets[meshlet_offset] = meshlet;

		for (size_t j = 0; j < meshlet.vertex_count; ++j)
			used[meshlet_vertices[meshlet.vertex_offset + j]] = 0xff;

		finishMeshlet(meshlet, meshlet_triangles);

		meshlet.vertex_offset += meshlet.vertex_count;
		meshlet.triangle_offset += (meshlet.triangle_count * 3 + 3) & ~3; // 4b padding
		meshlet.vertex_count = 0;
		meshlet.triangle_count = 0;

		result = true;
	}

	if (av == 0xff)
	{
		av = (unsigned char)meshlet.vertex_count;
		meshlet_vertices[meshlet.vertex_offset + meshlet.vertex_count++] = a;
	}

	if (bv == 0xff)
	{
		bv = (unsigned char)meshlet.vertex_count;
		meshlet_vertices[meshlet.vertex_offset + meshlet.vertex_count++] = b;
	}

	if (cv == 0xff)
	{
		cv = (unsigned char)meshlet.vertex_count;
		meshlet_vertices[meshlet.vertex_offset + meshlet.vertex_count++] = c;
	}

	meshlet_triangles[meshlet.triangle_offset + meshlet.triangle_count * 3 + 0] = av;
	meshlet_triangles[meshlet.triangle_offset + meshlet.triangle_count * 3 + 1] = bv;
	meshlet_triangles[meshlet.triangle_offset + meshlet.triangle_count * 3 + 2] = cv;
	meshlet.triangle_count++;

	return result;
}

static unsigned int getNeighborTriangle(const meshopt_Meshlet& meshlet, const Cone* meshlet_cone, unsigned int* meshlet_vertices, const unsigned int* indices, const TriangleAdjacency2& adjacency, const Cone* triangles, const unsigned int* live_triangles, const unsigned char* used, float meshlet_expected_radius, float cone_weight)
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

			int extra = (used[a] == 0xff) + (used[b] == 0xff) + (used[c] == 0xff);
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

			float score = 0;

			// caller selects one of two scoring functions: geometrical (based on meshlet cone) or topological (based on remaining triangles)
			if (meshlet_cone)
			{
				const Cone& tri_cone = triangles[triangle];

				float distance2 =
				    (tri_cone.px - meshlet_cone->px) * (tri_cone.px - meshlet_cone->px) +
				    (tri_cone.py - meshlet_cone->py) * (tri_cone.py - meshlet_cone->py) +
				    (tri_cone.pz - meshlet_cone->pz) * (tri_cone.pz - meshlet_cone->pz);

				float spread = tri_cone.nx * meshlet_cone->nx + tri_cone.ny * meshlet_cone->ny + tri_cone.nz * meshlet_cone->nz;

				score = getMeshletScore(distance2, spread, cone_weight, meshlet_expected_radius);
			}
			else
			{
				// each live_triangles entry is >= 1 since it includes the current triangle we're processing
				score = float(live_triangles[a] + live_triangles[b] + live_triangles[c] - 3);
			}

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

static void kdtreeNearest(KDNode* nodes, unsigned int root, const float* points, size_t stride, const unsigned char* emitted_flags, const float* position, unsigned int& result, float& limit)
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

			float distance2 =
			    (point[0] - position[0]) * (point[0] - position[0]) +
			    (point[1] - position[1]) * (point[1] - position[1]) +
			    (point[2] - position[2]) * (point[2] - position[2]);
			float distance = sqrtf(distance2);

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

		kdtreeNearest(nodes, root + 1 + first, points, stride, emitted_flags, position, result, limit);

		// only process the other node if it can have a match based on closest distance so far
		if (fabsf(delta) <= limit)
			kdtreeNearest(nodes, root + 1 + second, points, stride, emitted_flags, position, result, limit);
	}
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

size_t meshopt_buildMeshlets(meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, size_t max_vertices, size_t max_triangles, float cone_weight)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	assert(max_vertices >= 3 && max_vertices <= kMeshletMaxVertices);
	assert(max_triangles >= 1 && max_triangles <= kMeshletMaxTriangles);
	assert(max_triangles % 4 == 0); // ensures the caller will compute output space properly as index data is 4b aligned

	assert(cone_weight >= 0 && cone_weight <= 1);

	meshopt_Allocator allocator;

	TriangleAdjacency2 adjacency = {};
	buildTriangleAdjacency(adjacency, indices, index_count, vertex_count, allocator);

	unsigned int* live_triangles = allocator.allocate<unsigned int>(vertex_count);
	memcpy(live_triangles, adjacency.counts, vertex_count * sizeof(unsigned int));

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

	// index of the vertex in the meshlet, 0xff if the vertex isn't used
	unsigned char* used = allocator.allocate<unsigned char>(vertex_count);
	memset(used, -1, vertex_count);

	meshopt_Meshlet meshlet = {};
	size_t meshlet_offset = 0;

	Cone meshlet_cone_acc = {};

	for (;;)
	{
		Cone meshlet_cone = getMeshletCone(meshlet_cone_acc, meshlet.triangle_count);

		unsigned int best_triangle = getNeighborTriangle(meshlet, &meshlet_cone, meshlet_vertices, indices, adjacency, triangles, live_triangles, used, meshlet_expected_radius, cone_weight);
		int best_extra = best_triangle == ~0u ? -1 : (used[indices[best_triangle * 3 + 0]] == 0xff) + (used[indices[best_triangle * 3 + 1]] == 0xff) + (used[indices[best_triangle * 3 + 2]] == 0xff);

		// if the best triangle doesn't fit into current meshlet, the spatial scoring we've used is not very meaningful, so we re-select using topological scoring
		if (best_triangle != ~0u && (meshlet.vertex_count + best_extra > max_vertices || meshlet.triangle_count >= max_triangles))
		{
			best_triangle = getNeighborTriangle(meshlet, NULL, meshlet_vertices, indices, adjacency, triangles, live_triangles, used, meshlet_expected_radius, 0.f);
		}

		// when we run out of neighboring triangles we need to switch to spatial search; we currently just pick the closest triangle irrespective of connectivity
		if (best_triangle == ~0u)
		{
			float position[3] = {meshlet_cone.px, meshlet_cone.py, meshlet_cone.pz};
			unsigned int index = ~0u;
			float limit = FLT_MAX;

			kdtreeNearest(nodes, 0, &triangles[0].px, sizeof(Cone) / sizeof(float), emitted_flags, position, index, limit);

			best_triangle = index;
		}

		if (best_triangle == ~0u)
			break;

		unsigned int a = indices[best_triangle * 3 + 0], b = indices[best_triangle * 3 + 1], c = indices[best_triangle * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		// add meshlet to the output; when the current meshlet is full we reset the accumulated bounds
		if (appendMeshlet(meshlet, a, b, c, used, meshlets, meshlet_vertices, meshlet_triangles, meshlet_offset, max_vertices, max_triangles))
		{
			meshlet_offset++;
			memset(&meshlet_cone_acc, 0, sizeof(meshlet_cone_acc));
		}

		live_triangles[a]--;
		live_triangles[b]--;
		live_triangles[c]--;

		// remove emitted triangle from adjacency data
		// this makes sure that we spend less time traversing these lists on subsequent iterations
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

		emitted_flags[best_triangle] = 1;
	}

	if (meshlet.triangle_count)
	{
		finishMeshlet(meshlet, meshlet_triangles);

		meshlets[meshlet_offset++] = meshlet;
	}

	assert(meshlet_offset <= meshopt_buildMeshletsBound(index_count, max_vertices, max_triangles));
	return meshlet_offset;
}

size_t meshopt_buildMeshletsScan(meshopt_Meshlet* meshlets, unsigned int* meshlet_vertices, unsigned char* meshlet_triangles, const unsigned int* indices, size_t index_count, size_t vertex_count, size_t max_vertices, size_t max_triangles)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);

	assert(max_vertices >= 3 && max_vertices <= kMeshletMaxVertices);
	assert(max_triangles >= 1 && max_triangles <= kMeshletMaxTriangles);
	assert(max_triangles % 4 == 0); // ensures the caller will compute output space properly as index data is 4b aligned

	meshopt_Allocator allocator;

	// index of the vertex in the meshlet, 0xff if the vertex isn't used
	unsigned char* used = allocator.allocate<unsigned char>(vertex_count);
	memset(used, -1, vertex_count);

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

	// compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
	float psphere[4] = {};
	computeBoundingSphere(psphere, corners[0], triangles * 3);

	float center[3] = {psphere[0], psphere[1], psphere[2]};

	// treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
	float nsphere[4] = {};
	computeBoundingSphere(nsphere, normals, triangles);

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

	unsigned char remap[kMeshletMaxVertices];
	memset(remap, -1, vertex_count);

	size_t vertex_offset = 0;

	for (size_t i = 0; i < triangle_count * 3; ++i)
	{
		unsigned char& r = remap[indices[i]];

		if (r == 0xff)
		{
			r = (unsigned char)(vertex_offset);
			order[vertex_offset] = vertices[indices[i]];
			vertex_offset++;
		}

		indices[i] = r;
	}

	assert(vertex_offset <= vertex_count);
	memcpy(vertices, order, vertex_offset * sizeof(unsigned int));
}
