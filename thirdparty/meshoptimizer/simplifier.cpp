// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

#ifndef TRACE
#define TRACE 0
#endif

#if TRACE
#include <stdio.h>
#endif

#if TRACE
#define TRACESTATS(i) stats[i]++;
#else
#define TRACESTATS(i) (void)0
#endif

#define ATTRIBUTES 3

// This work is based on:
// Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. 1997
// Michael Garland. Quadric-based polygonal surface simplification. 1999
// Peter Lindstrom. Out-of-Core Simplification of Large Polygonal Models. 2000
// Matthias Teschner, Bruno Heidelberger, Matthias Mueller, Danat Pomeranets, Markus Gross. Optimized Spatial Hashing for Collision Detection of Deformable Objects. 2003
// Peter Van Sandt, Yannis Chronis, Jignesh M. Patel. Efficiently Searching In-Memory Sorted Arrays: Revenge of the Interpolation Search? 2019
namespace meshopt
{

struct EdgeAdjacency
{
	struct Edge
	{
		unsigned int next;
		unsigned int prev;
	};

	unsigned int* counts;
	unsigned int* offsets;
	Edge* data;
};

static void prepareEdgeAdjacency(EdgeAdjacency& adjacency, size_t index_count, size_t vertex_count, meshopt_Allocator& allocator)
{
	adjacency.counts = allocator.allocate<unsigned int>(vertex_count);
	adjacency.offsets = allocator.allocate<unsigned int>(vertex_count);
	adjacency.data = allocator.allocate<EdgeAdjacency::Edge>(index_count);
}

static void updateEdgeAdjacency(EdgeAdjacency& adjacency, const unsigned int* indices, size_t index_count, size_t vertex_count, const unsigned int* remap)
{
	size_t face_count = index_count / 3;

	// fill edge counts
	memset(adjacency.counts, 0, vertex_count * sizeof(unsigned int));

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = remap ? remap[indices[i]] : indices[i];
		assert(v < vertex_count);

		adjacency.counts[v]++;
	}

	// fill offset table
	unsigned int offset = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		adjacency.offsets[i] = offset;
		offset += adjacency.counts[i];
	}

	assert(offset == index_count);

	// fill edge data
	for (size_t i = 0; i < face_count; ++i)
	{
		unsigned int a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];

		if (remap)
		{
			a = remap[a];
			b = remap[b];
			c = remap[c];
		}

		adjacency.data[adjacency.offsets[a]].next = b;
		adjacency.data[adjacency.offsets[a]].prev = c;
		adjacency.offsets[a]++;

		adjacency.data[adjacency.offsets[b]].next = c;
		adjacency.data[adjacency.offsets[b]].prev = a;
		adjacency.offsets[b]++;

		adjacency.data[adjacency.offsets[c]].next = a;
		adjacency.data[adjacency.offsets[c]].prev = b;
		adjacency.offsets[c]++;
	}

	// fix offsets that have been disturbed by the previous pass
	for (size_t i = 0; i < vertex_count; ++i)
	{
		assert(adjacency.offsets[i] >= adjacency.counts[i]);

		adjacency.offsets[i] -= adjacency.counts[i];
	}
}

struct PositionHasher
{
	const float* vertex_positions;
	size_t vertex_stride_float;

	size_t hash(unsigned int index) const
	{
		const unsigned int* key = reinterpret_cast<const unsigned int*>(vertex_positions + index * vertex_stride_float);

		// scramble bits to make sure that integer coordinates have entropy in lower bits
		unsigned int x = key[0] ^ (key[0] >> 17);
		unsigned int y = key[1] ^ (key[1] >> 17);
		unsigned int z = key[2] ^ (key[2] >> 17);

		// Optimized Spatial Hashing for Collision Detection of Deformable Objects
		return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		return memcmp(vertex_positions + lhs * vertex_stride_float, vertex_positions + rhs * vertex_stride_float, sizeof(float) * 3) == 0;
	}
};

static size_t hashBuckets2(size_t count)
{
	size_t buckets = 1;
	while (buckets < count + count / 4)
		buckets *= 2;

	return buckets;
}

template <typename T, typename Hash>
static T* hashLookup2(T* table, size_t buckets, const Hash& hash, const T& key, const T& empty)
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
	return 0;
}

static void buildPositionRemap(unsigned int* remap, unsigned int* wedge, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride, meshopt_Allocator& allocator)
{
	PositionHasher hasher = {vertex_positions_data, vertex_positions_stride / sizeof(float)};

	size_t table_size = hashBuckets2(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	// build forward remap: for each vertex, which other (canonical) vertex does it map to?
	// we use position equivalence for this, and remap vertices to other existing vertices
	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int index = unsigned(i);
		unsigned int* entry = hashLookup2(table, table_size, hasher, index, ~0u);

		if (*entry == ~0u)
			*entry = index;

		remap[index] = *entry;
	}

	// build wedge table: for each vertex, which other vertex is the next wedge that also maps to the same vertex?
	// entries in table form a (cyclic) wedge loop per vertex; for manifold vertices, wedge[i] == remap[i] == i
	for (size_t i = 0; i < vertex_count; ++i)
		wedge[i] = unsigned(i);

	for (size_t i = 0; i < vertex_count; ++i)
		if (remap[i] != i)
		{
			unsigned int r = remap[i];

			wedge[i] = wedge[r];
			wedge[r] = unsigned(i);
		}
}

enum VertexKind
{
	Kind_Manifold, // not on an attribute seam, not on any boundary
	Kind_Border,   // not on an attribute seam, has exactly two open edges
	Kind_Seam,     // on an attribute seam with exactly two attribute seam edges
	Kind_Complex,  // none of the above; these vertices can move as long as all wedges move to the target vertex
	Kind_Locked,   // none of the above; these vertices can't move

	Kind_Count
};

// manifold vertices can collapse onto anything
// border/seam vertices can only be collapsed onto border/seam respectively
// complex vertices can collapse onto complex/locked
// a rule of thumb is that collapsing kind A into kind B preserves the kind B in the target vertex
// for example, while we could collapse Complex into Manifold, this would mean the target vertex isn't Manifold anymore
const unsigned char kCanCollapse[Kind_Count][Kind_Count] = {
    {1, 1, 1, 1, 1},
    {0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 0, 0},
};

// if a vertex is manifold or seam, adjoining edges are guaranteed to have an opposite edge
// note that for seam edges, the opposite edge isn't present in the attribute-based topology
// but is present if you consider a position-only mesh variant
const unsigned char kHasOpposite[Kind_Count][Kind_Count] = {
    {1, 1, 1, 0, 1},
    {1, 0, 1, 0, 0},
    {1, 1, 1, 0, 1},
    {0, 0, 0, 0, 0},
    {1, 0, 1, 0, 0},
};

static bool hasEdge(const EdgeAdjacency& adjacency, unsigned int a, unsigned int b)
{
	unsigned int count = adjacency.counts[a];
	const EdgeAdjacency::Edge* edges = adjacency.data + adjacency.offsets[a];

	for (size_t i = 0; i < count; ++i)
		if (edges[i].next == b)
			return true;

	return false;
}

static void classifyVertices(unsigned char* result, unsigned int* loop, unsigned int* loopback, size_t vertex_count, const EdgeAdjacency& adjacency, const unsigned int* remap, const unsigned int* wedge)
{
	memset(loop, -1, vertex_count * sizeof(unsigned int));
	memset(loopback, -1, vertex_count * sizeof(unsigned int));

	// incoming & outgoing open edges: ~0u if no open edges, i if there are more than 1
	// note that this is the same data as required in loop[] arrays; loop[] data is only valid for border/seam
	// but here it's okay to fill the data out for other types of vertices as well
	unsigned int* openinc = loopback;
	unsigned int* openout = loop;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int vertex = unsigned(i);

		unsigned int count = adjacency.counts[vertex];
		const EdgeAdjacency::Edge* edges = adjacency.data + adjacency.offsets[vertex];

		for (size_t j = 0; j < count; ++j)
		{
			unsigned int target = edges[j].next;

			if (!hasEdge(adjacency, target, vertex))
			{
				openinc[target] = (openinc[target] == ~0u) ? vertex : target;
				openout[vertex] = (openout[vertex] == ~0u) ? target : vertex;
			}
		}
	}

#if TRACE
	size_t stats[4] = {};
#endif

	for (size_t i = 0; i < vertex_count; ++i)
	{
		if (remap[i] == i)
		{
			if (wedge[i] == i)
			{
				// no attribute seam, need to check if it's manifold
				unsigned int openi = openinc[i], openo = openout[i];

				// note: we classify any vertices with no open edges as manifold
				// this is technically incorrect - if 4 triangles share an edge, we'll classify vertices as manifold
				// it's unclear if this is a problem in practice
				if (openi == ~0u && openo == ~0u)
				{
					result[i] = Kind_Manifold;
				}
				else if (openi != i && openo != i)
				{
					result[i] = Kind_Border;
				}
				else
				{
					result[i] = Kind_Locked;
					TRACESTATS(0);
				}
			}
			else if (wedge[wedge[i]] == i)
			{
				// attribute seam; need to distinguish between Seam and Locked
				unsigned int w = wedge[i];
				unsigned int openiv = openinc[i], openov = openout[i];
				unsigned int openiw = openinc[w], openow = openout[w];

				// seam should have one open half-edge for each vertex, and the edges need to "connect" - point to the same vertex post-remap
				if (openiv != ~0u && openiv != i && openov != ~0u && openov != i &&
				    openiw != ~0u && openiw != w && openow != ~0u && openow != w)
				{
					if (remap[openiv] == remap[openow] && remap[openov] == remap[openiw])
					{
						result[i] = Kind_Seam;
					}
					else
					{
						result[i] = Kind_Locked;
						TRACESTATS(1);
					}
				}
				else
				{
					result[i] = Kind_Locked;
					TRACESTATS(2);
				}
			}
			else
			{
				// more than one vertex maps to this one; we don't have classification available
				result[i] = Kind_Locked;
				TRACESTATS(3);
			}
		}
		else
		{
			assert(remap[i] < i);

			result[i] = result[remap[i]];
		}
	}

#if TRACE
	printf("locked: many open edges %d, disconnected seam %d, many seam edges %d, many wedges %d\n",
	    int(stats[0]), int(stats[1]), int(stats[2]), int(stats[3]));
#endif
}

struct Vector3
{
	float x, y, z;

#if ATTRIBUTES
	float a[ATTRIBUTES];
#endif
};

static float rescalePositions(Vector3* result, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride)
{
	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	float minv[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxv[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* v = vertex_positions_data + i * vertex_stride_float;

		if (result)
		{
			result[i].x = v[0];
			result[i].y = v[1];
			result[i].z = v[2];
		}

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

	if (result)
	{
		float scale = extent == 0 ? 0.f : 1.f / extent;

		for (size_t i = 0; i < vertex_count; ++i)
		{
			result[i].x = (result[i].x - minv[0]) * scale;
			result[i].y = (result[i].y - minv[1]) * scale;
			result[i].z = (result[i].z - minv[2]) * scale;
		}
	}

	return extent;
}

struct Quadric
{
	float a00, a11, a22;
	float a10, a20, a21;
	float b0, b1, b2, c;
	float w;

#if ATTRIBUTES
	float gx[ATTRIBUTES];
	float gy[ATTRIBUTES];
	float gz[ATTRIBUTES];
	float gw[ATTRIBUTES];
#endif
};

struct Collapse
{
	unsigned int v0;
	unsigned int v1;

	union
	{
		unsigned int bidi;
		float error;
		unsigned int errorui;
	};
	float distance_error;
};

static float normalize(Vector3& v)
{
	float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	if (length > 0)
	{
		v.x /= length;
		v.y /= length;
		v.z /= length;
	}

	return length;
}

static void quadricAdd(Quadric& Q, const Quadric& R)
{
	Q.a00 += R.a00;
	Q.a11 += R.a11;
	Q.a22 += R.a22;
	Q.a10 += R.a10;
	Q.a20 += R.a20;
	Q.a21 += R.a21;
	Q.b0 += R.b0;
	Q.b1 += R.b1;
	Q.b2 += R.b2;
	Q.c += R.c;
	Q.w += R.w;

#if ATTRIBUTES
	for (int k = 0; k < ATTRIBUTES; ++k)
	{
		Q.gx[k] += R.gx[k];
		Q.gy[k] += R.gy[k];
		Q.gz[k] += R.gz[k];
		Q.gw[k] += R.gw[k];
	}
#endif
}

static float quadricError(const Quadric& Q, const Vector3& v)
{
	float rx = Q.b0;
	float ry = Q.b1;
	float rz = Q.b2;

	rx += Q.a10 * v.y;
	ry += Q.a21 * v.z;
	rz += Q.a20 * v.x;

	rx *= 2;
	ry *= 2;
	rz *= 2;

	rx += Q.a00 * v.x;
	ry += Q.a11 * v.y;
	rz += Q.a22 * v.z;

	float r = Q.c;
	r += rx * v.x;
	r += ry * v.y;
	r += rz * v.z;

#if ATTRIBUTES
	// see quadricUpdateAttributes for general derivation; here we need to add the parts of (eval(pos) - attr)^2 that depend on attr
	for (int k = 0; k < ATTRIBUTES; ++k)
	{
		float a = v.a[k];

		r += a * a * Q.w;
		r -= 2 * a * (v.x * Q.gx[k] + v.y * Q.gy[k] + v.z * Q.gz[k] + Q.gw[k]);
	}
#endif

	float s = Q.w == 0.f ? 0.f : 1.f / Q.w;

	return fabsf(r) * s;
}

static float quadricErrorNoAttributes(const Quadric& Q, const Vector3& v)
{
	float rx = Q.b0;
	float ry = Q.b1;
	float rz = Q.b2;

	rx += Q.a10 * v.y;
	ry += Q.a21 * v.z;
	rz += Q.a20 * v.x;

	rx *= 2;
	ry *= 2;
	rz *= 2;

	rx += Q.a00 * v.x;
	ry += Q.a11 * v.y;
	rz += Q.a22 * v.z;

	float r = Q.c;
	r += rx * v.x;
	r += ry * v.y;
	r += rz * v.z;

	float s = Q.w == 0.f ? 0.f : 1.f / Q.w;

	return fabsf(r) * s;
}

static void quadricFromPlane(Quadric& Q, float a, float b, float c, float d, float w)
{
	float aw = a * w;
	float bw = b * w;
	float cw = c * w;
	float dw = d * w;

	Q.a00 = a * aw;
	Q.a11 = b * bw;
	Q.a22 = c * cw;
	Q.a10 = a * bw;
	Q.a20 = a * cw;
	Q.a21 = b * cw;
	Q.b0 = a * dw;
	Q.b1 = b * dw;
	Q.b2 = c * dw;
	Q.c = d * dw;
	Q.w = w;

#if ATTRIBUTES
	memset(Q.gx, 0, sizeof(Q.gx));
	memset(Q.gy, 0, sizeof(Q.gy));
	memset(Q.gz, 0, sizeof(Q.gz));
	memset(Q.gw, 0, sizeof(Q.gw));
#endif
}

static void quadricFromPoint(Quadric& Q, float x, float y, float z, float w)
{
	// we need to encode (x - X) ^ 2 + (y - Y)^2 + (z - Z)^2 into the quadric
	Q.a00 = w;
	Q.a11 = w;
	Q.a22 = w;
	Q.a10 = 0.f;
	Q.a20 = 0.f;
	Q.a21 = 0.f;
	Q.b0 = -2.f * x * w;
	Q.b1 = -2.f * y * w;
	Q.b2 = -2.f * z * w;
	Q.c = (x * x + y * y + z * z) * w;
	Q.w = w;
}

static void quadricFromTriangle(Quadric& Q, const Vector3& p0, const Vector3& p1, const Vector3& p2, float weight)
{
	Vector3 p10 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
	Vector3 p20 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};

	// normal = cross(p1 - p0, p2 - p0)
	Vector3 normal = {p10.y * p20.z - p10.z * p20.y, p10.z * p20.x - p10.x * p20.z, p10.x * p20.y - p10.y * p20.x};
	float area = normalize(normal);

	float distance = normal.x * p0.x + normal.y * p0.y + normal.z * p0.z;

	// we use sqrtf(area) so that the error is scaled linearly; this tends to improve silhouettes
	quadricFromPlane(Q, normal.x, normal.y, normal.z, -distance, sqrtf(area) * weight);
}

static void quadricFromTriangleEdge(Quadric& Q, const Vector3& p0, const Vector3& p1, const Vector3& p2, float weight)
{
	Vector3 p10 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
	float length = normalize(p10);

	// p20p = length of projection of p2-p0 onto normalize(p1 - p0)
	Vector3 p20 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
	float p20p = p20.x * p10.x + p20.y * p10.y + p20.z * p10.z;

	// normal = altitude of triangle from point p2 onto edge p1-p0
	Vector3 normal = {p20.x - p10.x * p20p, p20.y - p10.y * p20p, p20.z - p10.z * p20p};
	normalize(normal);

	float distance = normal.x * p0.x + normal.y * p0.y + normal.z * p0.z;

	// note: the weight is scaled linearly with edge length; this has to match the triangle weight
	quadricFromPlane(Q, normal.x, normal.y, normal.z, -distance, length * weight);
}

#if ATTRIBUTES
static void quadricUpdateAttributes(Quadric& Q, const Vector3& p0, const Vector3& p1, const Vector3& p2, float w)
{
	// for each attribute we want to encode the following function into the quadric:
	// (eval(pos) - attr)^2
	// where eval(pos) interpolates attribute across the triangle like so:
	// eval(pos) = pos.x * gx + pos.y * gy + pos.z * gz + gw
	// where gx/gy/gz/gw are gradients
	Vector3 p10 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
	Vector3 p20 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};

	// we compute gradients using barycentric coordinates; barycentric coordinates can be computed as follows:
	// v = (d11 * d20 - d01 * d21) / denom
	// w = (d00 * d21 - d01 * d20) / denom
	// u = 1 - v - w
	// here v0, v1 are triangle edge vectors, v2 is a vector from point to triangle corner, and dij = dot(vi, vj)
	const Vector3& v0 = p10;
	const Vector3& v1 = p20;
	float d00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
	float d01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
	float d11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
	float denom = d00 * d11 - d01 * d01;
	float denomr = denom == 0 ? 0.f : 1.f / denom;

	// precompute gradient factors
	// these are derived by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w and factoring out common factors that are shared between attributes
	float gx1 = (d11 * v0.x - d01 * v1.x) * denomr;
	float gx2 = (d00 * v1.x - d01 * v0.x) * denomr;
	float gy1 = (d11 * v0.y - d01 * v1.y) * denomr;
	float gy2 = (d00 * v1.y - d01 * v0.y) * denomr;
	float gz1 = (d11 * v0.z - d01 * v1.z) * denomr;
	float gz2 = (d00 * v1.z - d01 * v0.z) * denomr;

	for (int k = 0; k < ATTRIBUTES; ++k)
	{
		float a0 = p0.a[k], a1 = p1.a[k], a2 = p2.a[k];

		// compute gradient of eval(pos) for x/y/z/w
		// the formulas below are obtained by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w
		float gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
		float gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
		float gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
		float gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

		// quadric encodes (eval(pos)-attr)^2; this means that the resulting expansion needs to compute, for example, pos.x * pos.y * K
		// since quadrics already encode factors for pos.x * pos.y, we can accumulate almost everything in basic quadric fields
		Q.a00 += w * (gx * gx);
		Q.a11 += w * (gy * gy);
		Q.a22 += w * (gz * gz);

		Q.a10 += w * (gy * gx);
		Q.a20 += w * (gz * gx);
		Q.a21 += w * (gz * gy);

		Q.b0 += w * (gx * gw);
		Q.b1 += w * (gy * gw);
		Q.b2 += w * (gz * gw);

		Q.c += w * (gw * gw);

		// the only remaining sum components are ones that depend on attr; these will be addded during error evaluation, see quadricError
		Q.gx[k] = w * gx;
		Q.gy[k] = w * gy;
		Q.gz[k] = w * gz;
		Q.gw[k] = w * gw;

#if TRACE > 2
		printf("attr%d: %e %e %e\n",
			k,
			(gx * p0.x + gy * p0.y + gz * p0.z + gw - a0),
			(gx * p1.x + gy * p1.y + gz * p1.z + gw - a1),
			(gx * p2.x + gy * p2.y + gz * p2.z + gw - a2)
			);
#endif
	}
}
#endif

static void fillFaceQuadrics(Quadric* vertex_quadrics, Quadric* vertex_no_attrib_quadrics, const unsigned int* indices, size_t index_count, const Vector3* vertex_positions, const unsigned int* remap)
{
	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int i0 = indices[i + 0];
		unsigned int i1 = indices[i + 1];
		unsigned int i2 = indices[i + 2];

		Quadric Q;
		quadricFromTriangle(Q, vertex_positions[i0], vertex_positions[i1], vertex_positions[i2], 1.f);
		quadricAdd(vertex_no_attrib_quadrics[remap[i0]], Q);
		quadricAdd(vertex_no_attrib_quadrics[remap[i1]], Q);
		quadricAdd(vertex_no_attrib_quadrics[remap[i2]], Q);

#if ATTRIBUTES
		quadricUpdateAttributes(Q, vertex_positions[i0], vertex_positions[i1], vertex_positions[i2], Q.w);
#endif
		quadricAdd(vertex_quadrics[remap[i0]], Q);
		quadricAdd(vertex_quadrics[remap[i1]], Q);
		quadricAdd(vertex_quadrics[remap[i2]], Q);
	}
}

static void fillEdgeQuadrics(Quadric* vertex_quadrics, Quadric* vertex_no_attrib_quadrics, const unsigned int* indices, size_t index_count, const Vector3* vertex_positions, const unsigned int* remap, const unsigned char* vertex_kind, const unsigned int* loop, const unsigned int* loopback)
{
	for (size_t i = 0; i < index_count; i += 3)
	{
		static const int next[3] = {1, 2, 0};

		for (int e = 0; e < 3; ++e)
		{
			unsigned int i0 = indices[i + e];
			unsigned int i1 = indices[i + next[e]];

			unsigned char k0 = vertex_kind[i0];
			unsigned char k1 = vertex_kind[i1];

			// check that either i0 or i1 are border/seam and are on the same edge loop
			// note that we need to add the error even for edged that connect e.g. border & locked
			// if we don't do that, the adjacent border->border edge won't have correct errors for corners
			if (k0 != Kind_Border && k0 != Kind_Seam && k1 != Kind_Border && k1 != Kind_Seam)
				continue;

			if ((k0 == Kind_Border || k0 == Kind_Seam) && loop[i0] != i1)
				continue;

			if ((k1 == Kind_Border || k1 == Kind_Seam) && loopback[i1] != i0)
				continue;

			// seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
			if (kHasOpposite[k0][k1] && remap[i1] > remap[i0])
				continue;

			unsigned int i2 = indices[i + next[next[e]]];

			// we try hard to maintain border edge geometry; seam edges can move more freely
			// due to topological restrictions on collapses, seam quadrics slightly improves collapse structure but aren't critical
			const float kEdgeWeightSeam = 1.f;
			const float kEdgeWeightBorder = 10.f;

			float edgeWeight = (k0 == Kind_Border || k1 == Kind_Border) ? kEdgeWeightBorder : kEdgeWeightSeam;

			Quadric Q;
			quadricFromTriangleEdge(Q, vertex_positions[i0], vertex_positions[i1], vertex_positions[i2], edgeWeight);

			quadricAdd(vertex_quadrics[remap[i0]], Q);
			quadricAdd(vertex_quadrics[remap[i1]], Q);

			quadricAdd(vertex_no_attrib_quadrics[remap[i0]], Q);
			quadricAdd(vertex_no_attrib_quadrics[remap[i1]], Q);
		}
	}
}

// does triangle ABC flip when C is replaced with D?
static bool hasTriangleFlip(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d)
{
	Vector3 eb = {b.x - a.x, b.y - a.y, b.z - a.z};
	Vector3 ec = {c.x - a.x, c.y - a.y, c.z - a.z};
	Vector3 ed = {d.x - a.x, d.y - a.y, d.z - a.z};

	Vector3 nbc = {eb.y * ec.z - eb.z * ec.y, eb.z * ec.x - eb.x * ec.z, eb.x * ec.y - eb.y * ec.x};
	Vector3 nbd = {eb.y * ed.z - eb.z * ed.y, eb.z * ed.x - eb.x * ed.z, eb.x * ed.y - eb.y * ed.x};

	return nbc.x * nbd.x + nbc.y * nbd.y + nbc.z * nbd.z < 0;
}

static bool hasTriangleFlips(const EdgeAdjacency& adjacency, const Vector3* vertex_positions, const unsigned int* collapse_remap, unsigned int i0, unsigned int i1)
{
	assert(collapse_remap[i0] == i0);
	assert(collapse_remap[i1] == i1);

	const Vector3& v0 = vertex_positions[i0];
	const Vector3& v1 = vertex_positions[i1];

	const EdgeAdjacency::Edge* edges = &adjacency.data[adjacency.offsets[i0]];
	size_t count = adjacency.counts[i0];

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int a = collapse_remap[edges[i].next];
		unsigned int b = collapse_remap[edges[i].prev];

		// skip triangles that get collapsed
		// note: this is mathematically redundant as if either of these is true, the dot product in hasTriangleFlip should be 0
		if (a == i1 || b == i1)
			continue;

		// early-out when at least one triangle flips due to a collapse
		if (hasTriangleFlip(vertex_positions[a], vertex_positions[b], v0, v1))
			return true;
	}

	return false;
}

static size_t pickEdgeCollapses(Collapse* collapses, const unsigned int* indices, size_t index_count, const unsigned int* remap, const unsigned char* vertex_kind, const unsigned int* loop)
{
	size_t collapse_count = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		static const int next[3] = {1, 2, 0};

		for (int e = 0; e < 3; ++e)
		{
			unsigned int i0 = indices[i + e];
			unsigned int i1 = indices[i + next[e]];

			// this can happen either when input has a zero-length edge, or when we perform collapses for complex
			// topology w/seams and collapse a manifold vertex that connects to both wedges onto one of them
			// we leave edges like this alone since they may be important for preserving mesh integrity
			if (remap[i0] == remap[i1])
				continue;

			unsigned char k0 = vertex_kind[i0];
			unsigned char k1 = vertex_kind[i1];

			// the edge has to be collapsible in at least one direction
			if (!(kCanCollapse[k0][k1] | kCanCollapse[k1][k0]))
				continue;

			// manifold and seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
			if (kHasOpposite[k0][k1] && remap[i1] > remap[i0])
				continue;

			// two vertices are on a border or a seam, but there's no direct edge between them
			// this indicates that they belong to two different edge loops and we should not collapse this edge
			// loop[] tracks half edges so we only need to check i0->i1
			if (k0 == k1 && (k0 == Kind_Border || k0 == Kind_Seam) && loop[i0] != i1)
				continue;

			// edge can be collapsed in either direction - we will pick the one with minimum error
			// note: we evaluate error later during collapse ranking, here we just tag the edge as bidirectional
			if (kCanCollapse[k0][k1] & kCanCollapse[k1][k0])
			{
				Collapse c = {i0, i1, {/* bidi= */ 1}};
				collapses[collapse_count++] = c;
			}
			else
			{
				// edge can only be collapsed in one direction
				unsigned int e0 = kCanCollapse[k0][k1] ? i0 : i1;
				unsigned int e1 = kCanCollapse[k0][k1] ? i1 : i0;

				Collapse c = {e0, e1, {/* bidi= */ 0}};
				collapses[collapse_count++] = c;
			}
		}
	}

	return collapse_count;
}

static void rankEdgeCollapses(Collapse* collapses, size_t collapse_count, const Vector3* vertex_positions, const Quadric* vertex_quadrics, const Quadric* vertex_no_attrib_quadrics, const unsigned int* remap)
{
	for (size_t i = 0; i < collapse_count; ++i)
	{
		Collapse& c = collapses[i];

		unsigned int i0 = c.v0;
		unsigned int i1 = c.v1;

		// most edges are bidirectional which means we need to evaluate errors for two collapses
		// to keep this code branchless we just use the same edge for unidirectional edges
		unsigned int j0 = c.bidi ? i1 : i0;
		unsigned int j1 = c.bidi ? i0 : i1;

		const Quadric& qi = vertex_quadrics[remap[i0]];
		const Quadric& qj = vertex_quadrics[remap[j0]];

		float ei = quadricError(qi, vertex_positions[i1]);
		float ej = quadricError(qj, vertex_positions[j1]);

		const Quadric& naqi = vertex_no_attrib_quadrics[remap[i0]];
		const Quadric& naqj = vertex_no_attrib_quadrics[remap[j0]];

		// pick edge direction with minimal error
		c.v0 = ei <= ej ? i0 : j0;
		c.v1 = ei <= ej ? i1 : j1;
		c.error = ei <= ej ? ei : ej;
		c.distance_error = ei <= ej ? quadricErrorNoAttributes(naqi, vertex_positions[i1]) :  quadricErrorNoAttributes(naqj, vertex_positions[j1]);
	}
}

#if TRACE > 1
static void dumpEdgeCollapses(const Collapse* collapses, size_t collapse_count, const unsigned char* vertex_kind)
{
	size_t ckinds[Kind_Count][Kind_Count] = {};
	float cerrors[Kind_Count][Kind_Count] = {};

	for (int k0 = 0; k0 < Kind_Count; ++k0)
		for (int k1 = 0; k1 < Kind_Count; ++k1)
			cerrors[k0][k1] = FLT_MAX;

	for (size_t i = 0; i < collapse_count; ++i)
	{
		unsigned int i0 = collapses[i].v0;
		unsigned int i1 = collapses[i].v1;

		unsigned char k0 = vertex_kind[i0];
		unsigned char k1 = vertex_kind[i1];

		ckinds[k0][k1]++;
		cerrors[k0][k1] = (collapses[i].error < cerrors[k0][k1]) ? collapses[i].error : cerrors[k0][k1];
	}

	for (int k0 = 0; k0 < Kind_Count; ++k0)
		for (int k1 = 0; k1 < Kind_Count; ++k1)
			if (ckinds[k0][k1])
				printf("collapses %d -> %d: %d, min error %e\n", k0, k1, int(ckinds[k0][k1]), ckinds[k0][k1] ? sqrtf(cerrors[k0][k1]) : 0.f);
}

static void dumpLockedCollapses(const unsigned int* indices, size_t index_count, const unsigned char* vertex_kind)
{
	size_t locked_collapses[Kind_Count][Kind_Count] = {};

	for (size_t i = 0; i < index_count; i += 3)
	{
		static const int next[3] = {1, 2, 0};

		for (int e = 0; e < 3; ++e)
		{
			unsigned int i0 = indices[i + e];
			unsigned int i1 = indices[i + next[e]];

			unsigned char k0 = vertex_kind[i0];
			unsigned char k1 = vertex_kind[i1];

			locked_collapses[k0][k1] += !kCanCollapse[k0][k1] && !kCanCollapse[k1][k0];
		}
	}

	for (int k0 = 0; k0 < Kind_Count; ++k0)
		for (int k1 = 0; k1 < Kind_Count; ++k1)
			if (locked_collapses[k0][k1])
				printf("locked collapses %d -> %d: %d\n", k0, k1, int(locked_collapses[k0][k1]));
}
#endif

static void sortEdgeCollapses(unsigned int* sort_order, const Collapse* collapses, size_t collapse_count)
{
	const int sort_bits = 11;

	// fill histogram for counting sort
	unsigned int histogram[1 << sort_bits];
	memset(histogram, 0, sizeof(histogram));

	for (size_t i = 0; i < collapse_count; ++i)
	{
		// skip sign bit since error is non-negative
		unsigned int key = (collapses[i].errorui << 1) >> (32 - sort_bits);

		histogram[key]++;
	}

	// compute offsets based on histogram data
	size_t histogram_sum = 0;

	for (size_t i = 0; i < 1 << sort_bits; ++i)
	{
		size_t count = histogram[i];
		histogram[i] = unsigned(histogram_sum);
		histogram_sum += count;
	}

	assert(histogram_sum == collapse_count);

	// compute sort order based on offsets
	for (size_t i = 0; i < collapse_count; ++i)
	{
		// skip sign bit since error is non-negative
		unsigned int key = (collapses[i].errorui << 1) >> (32 - sort_bits);

		sort_order[histogram[key]++] = unsigned(i);
	}
}

static size_t performEdgeCollapses(unsigned int* collapse_remap, unsigned char* collapse_locked, Quadric* vertex_quadrics, Quadric* vertex_no_attrib_quadrics, const Collapse* collapses, size_t collapse_count, const unsigned int* collapse_order, const unsigned int* remap, const unsigned int* wedge, const unsigned char* vertex_kind, const Vector3* vertex_positions, const EdgeAdjacency& adjacency, size_t triangle_collapse_goal, float error_limit, float& result_error)
{
	size_t edge_collapses = 0;
	size_t triangle_collapses = 0;

	// most collapses remove 2 triangles; use this to establish a bound on the pass in terms of error limit
	// note that edge_collapse_goal is an estimate; triangle_collapse_goal will be used to actually limit collapses
	size_t edge_collapse_goal = triangle_collapse_goal / 2;

#if TRACE
	size_t stats[4] = {};
#endif

	for (size_t i = 0; i < collapse_count; ++i)
	{
		const Collapse& c = collapses[collapse_order[i]];

		TRACESTATS(0);

		if (c.error > error_limit)
			break;

		if (triangle_collapses >= triangle_collapse_goal)
			break;

		// we limit the error in each pass based on the error of optimal last collapse; since many collapses will be locked
		// as they will share vertices with other successfull collapses, we need to increase the acceptable error by some factor
		float error_goal = edge_collapse_goal < collapse_count ? 1.5f * collapses[collapse_order[edge_collapse_goal]].error : FLT_MAX;

		// on average, each collapse is expected to lock 6 other collapses; to avoid degenerate passes on meshes with odd
		// topology, we only abort if we got over 1/6 collapses accordingly.
		if (c.error > error_goal && triangle_collapses > triangle_collapse_goal / 6)
			break;

		unsigned int i0 = c.v0;
		unsigned int i1 = c.v1;

		unsigned int r0 = remap[i0];
		unsigned int r1 = remap[i1];

		// we don't collapse vertices that had source or target vertex involved in a collapse
		// it's important to not move the vertices twice since it complicates the tracking/remapping logic
		// it's important to not move other vertices towards a moved vertex to preserve error since we don't re-rank collapses mid-pass
		if (collapse_locked[r0] | collapse_locked[r1])
		{
			TRACESTATS(1);
			continue;
		}

		if (hasTriangleFlips(adjacency, vertex_positions, collapse_remap, r0, r1))
		{
			// adjust collapse goal since this collapse is invalid and shouldn't factor into error goal
			edge_collapse_goal++;

			TRACESTATS(2);
			continue;
		}

		assert(collapse_remap[r0] == r0);
		assert(collapse_remap[r1] == r1);

		quadricAdd(vertex_quadrics[r1], vertex_quadrics[r0]);
		quadricAdd(vertex_no_attrib_quadrics[r1], vertex_no_attrib_quadrics[r0]);

		if (vertex_kind[i0] == Kind_Complex)
		{
			unsigned int v = i0;

			do
			{
				collapse_remap[v] = r1;
				v = wedge[v];
			} while (v != i0);
		}
		else if (vertex_kind[i0] == Kind_Seam)
		{
			// remap v0 to v1 and seam pair of v0 to seam pair of v1
			unsigned int s0 = wedge[i0];
			unsigned int s1 = wedge[i1];

			assert(s0 != i0 && s1 != i1);
			assert(wedge[s0] == i0 && wedge[s1] == i1);

			collapse_remap[i0] = i1;
			collapse_remap[s0] = s1;
		}
		else
		{
			assert(wedge[i0] == i0);

			collapse_remap[i0] = i1;
		}

		collapse_locked[r0] = 1;
		collapse_locked[r1] = 1;

		// border edges collapse 1 triangle, other edges collapse 2 or more
		triangle_collapses += (vertex_kind[i0] == Kind_Border) ? 1 : 2;
		edge_collapses++;

		result_error = result_error < c.distance_error ? c.distance_error : result_error;
	}

#if TRACE
	float error_goal_perfect = edge_collapse_goal < collapse_count ? collapses[collapse_order[edge_collapse_goal]].error : 0.f;

	printf("removed %d triangles, error %e (goal %e); evaluated %d/%d collapses (done %d, skipped %d, invalid %d)\n",
	    int(triangle_collapses), sqrtf(result_error), sqrtf(error_goal_perfect),
	    int(stats[0]), int(collapse_count), int(edge_collapses), int(stats[1]), int(stats[2]));
#endif

	return edge_collapses;
}

static size_t remapIndexBuffer(unsigned int* indices, size_t index_count, const unsigned int* collapse_remap)
{
	size_t write = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int v0 = collapse_remap[indices[i + 0]];
		unsigned int v1 = collapse_remap[indices[i + 1]];
		unsigned int v2 = collapse_remap[indices[i + 2]];

		// we never move the vertex twice during a single pass
		assert(collapse_remap[v0] == v0);
		assert(collapse_remap[v1] == v1);
		assert(collapse_remap[v2] == v2);

		if (v0 != v1 && v0 != v2 && v1 != v2)
		{
			indices[write + 0] = v0;
			indices[write + 1] = v1;
			indices[write + 2] = v2;
			write += 3;
		}
	}

	return write;
}

static void remapEdgeLoops(unsigned int* loop, size_t vertex_count, const unsigned int* collapse_remap)
{
	for (size_t i = 0; i < vertex_count; ++i)
	{
		if (loop[i] != ~0u)
		{
			unsigned int l = loop[i];
			unsigned int r = collapse_remap[l];

			// i == r is a special case when the seam edge is collapsed in a direction opposite to where loop goes
			loop[i] = (i == r) ? loop[l] : r;
		}
	}
}

struct CellHasher
{
	const unsigned int* vertex_ids;

	size_t hash(unsigned int i) const
	{
		unsigned int h = vertex_ids[i];

		// MurmurHash2 finalizer
		h ^= h >> 13;
		h *= 0x5bd1e995;
		h ^= h >> 15;
		return h;
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		return vertex_ids[lhs] == vertex_ids[rhs];
	}
};

struct IdHasher
{
	size_t hash(unsigned int id) const
	{
		unsigned int h = id;

		// MurmurHash2 finalizer
		h ^= h >> 13;
		h *= 0x5bd1e995;
		h ^= h >> 15;
		return h;
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		return lhs == rhs;
	}
};

struct TriangleHasher
{
	const unsigned int* indices;

	size_t hash(unsigned int i) const
	{
		const unsigned int* tri = indices + i * 3;

		// Optimized Spatial Hashing for Collision Detection of Deformable Objects
		return (tri[0] * 73856093) ^ (tri[1] * 19349663) ^ (tri[2] * 83492791);
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		const unsigned int* lt = indices + lhs * 3;
		const unsigned int* rt = indices + rhs * 3;

		return lt[0] == rt[0] && lt[1] == rt[1] && lt[2] == rt[2];
	}
};

static void computeVertexIds(unsigned int* vertex_ids, const Vector3* vertex_positions, size_t vertex_count, int grid_size)
{
	assert(grid_size >= 1 && grid_size <= 1024);
	float cell_scale = float(grid_size - 1);

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const Vector3& v = vertex_positions[i];

		int xi = int(v.x * cell_scale + 0.5f);
		int yi = int(v.y * cell_scale + 0.5f);
		int zi = int(v.z * cell_scale + 0.5f);

		vertex_ids[i] = (xi << 20) | (yi << 10) | zi;
	}
}

static size_t countTriangles(const unsigned int* vertex_ids, const unsigned int* indices, size_t index_count)
{
	size_t result = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int id0 = vertex_ids[indices[i + 0]];
		unsigned int id1 = vertex_ids[indices[i + 1]];
		unsigned int id2 = vertex_ids[indices[i + 2]];

		result += (id0 != id1) & (id0 != id2) & (id1 != id2);
	}

	return result;
}

static size_t fillVertexCells(unsigned int* table, size_t table_size, unsigned int* vertex_cells, const unsigned int* vertex_ids, size_t vertex_count)
{
	CellHasher hasher = {vertex_ids};

	memset(table, -1, table_size * sizeof(unsigned int));

	size_t result = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int* entry = hashLookup2(table, table_size, hasher, unsigned(i), ~0u);

		if (*entry == ~0u)
		{
			*entry = unsigned(i);
			vertex_cells[i] = unsigned(result++);
		}
		else
		{
			vertex_cells[i] = vertex_cells[*entry];
		}
	}

	return result;
}

static size_t countVertexCells(unsigned int* table, size_t table_size, const unsigned int* vertex_ids, size_t vertex_count)
{
	IdHasher hasher;

	memset(table, -1, table_size * sizeof(unsigned int));

	size_t result = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int id = vertex_ids[i];
		unsigned int* entry = hashLookup2(table, table_size, hasher, id, ~0u);

		result += (*entry == ~0u);
		*entry = id;
	}

	return result;
}

static void fillCellQuadrics(Quadric* cell_quadrics, const unsigned int* indices, size_t index_count, const Vector3* vertex_positions, const unsigned int* vertex_cells)
{
	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int i0 = indices[i + 0];
		unsigned int i1 = indices[i + 1];
		unsigned int i2 = indices[i + 2];

		unsigned int c0 = vertex_cells[i0];
		unsigned int c1 = vertex_cells[i1];
		unsigned int c2 = vertex_cells[i2];

		bool single_cell = (c0 == c1) & (c0 == c2);

		Quadric Q;
		quadricFromTriangle(Q, vertex_positions[i0], vertex_positions[i1], vertex_positions[i2], single_cell ? 3.f : 1.f);

		if (single_cell)
		{
			quadricAdd(cell_quadrics[c0], Q);
		}
		else
		{
			quadricAdd(cell_quadrics[c0], Q);
			quadricAdd(cell_quadrics[c1], Q);
			quadricAdd(cell_quadrics[c2], Q);
		}
	}
}

static void fillCellQuadrics(Quadric* cell_quadrics, const Vector3* vertex_positions, size_t vertex_count, const unsigned int* vertex_cells)
{
	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int c = vertex_cells[i];
		const Vector3& v = vertex_positions[i];

		Quadric Q;
		quadricFromPoint(Q, v.x, v.y, v.z, 1.f);

		quadricAdd(cell_quadrics[c], Q);
	}
}

static void fillCellRemap(unsigned int* cell_remap, float* cell_errors, size_t cell_count, const unsigned int* vertex_cells, const Quadric* cell_quadrics, const Vector3* vertex_positions, size_t vertex_count)
{
	memset(cell_remap, -1, cell_count * sizeof(unsigned int));

	for (size_t i = 0; i < vertex_count; ++i)
	{
		unsigned int cell = vertex_cells[i];
		float error = quadricError(cell_quadrics[cell], vertex_positions[i]);

		if (cell_remap[cell] == ~0u || cell_errors[cell] > error)
		{
			cell_remap[cell] = unsigned(i);
			cell_errors[cell] = error;
		}
	}
}

static size_t filterTriangles(unsigned int* destination, unsigned int* tritable, size_t tritable_size, const unsigned int* indices, size_t index_count, const unsigned int* vertex_cells, const unsigned int* cell_remap)
{
	TriangleHasher hasher = {destination};

	memset(tritable, -1, tritable_size * sizeof(unsigned int));

	size_t result = 0;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int c0 = vertex_cells[indices[i + 0]];
		unsigned int c1 = vertex_cells[indices[i + 1]];
		unsigned int c2 = vertex_cells[indices[i + 2]];

		if (c0 != c1 && c0 != c2 && c1 != c2)
		{
			unsigned int a = cell_remap[c0];
			unsigned int b = cell_remap[c1];
			unsigned int c = cell_remap[c2];

			if (b < a && b < c)
			{
				unsigned int t = a;
				a = b, b = c, c = t;
			}
			else if (c < a && c < b)
			{
				unsigned int t = c;
				c = b, b = a, a = t;
			}

			destination[result * 3 + 0] = a;
			destination[result * 3 + 1] = b;
			destination[result * 3 + 2] = c;

			unsigned int* entry = hashLookup2(tritable, tritable_size, hasher, unsigned(result), ~0u);

			if (*entry == ~0u)
				*entry = unsigned(result++);
		}
	}

	return result * 3;
}

static float interpolate(float y, float x0, float y0, float x1, float y1, float x2, float y2)
{
	// three point interpolation from "revenge of interpolation search" paper
	float num = (y1 - y) * (x1 - x2) * (x1 - x0) * (y2 - y0);
	float den = (y2 - y) * (x1 - x2) * (y0 - y1) + (y0 - y) * (x1 - x0) * (y1 - y2);
	return x1 + num / den;
}

} // namespace meshopt

#ifndef NDEBUG
// Note: this is only exposed for debug visualization purposes; do *not* use these in debug builds
MESHOPTIMIZER_API unsigned char* meshopt_simplifyDebugKind = 0;
MESHOPTIMIZER_API unsigned int* meshopt_simplifyDebugLoop = 0;
MESHOPTIMIZER_API unsigned int* meshopt_simplifyDebugLoopBack = 0;
#endif

size_t meshopt_simplify(unsigned int* destination, const unsigned int* indices, size_t index_count, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride, size_t target_index_count, float target_error, float* out_result_error)
{
	return meshopt_simplifyWithAttributes(destination, indices, index_count, vertex_positions_data, vertex_count, vertex_positions_stride, target_index_count, target_error, out_result_error, 0, 0, 0);
}

size_t meshopt_simplifyWithAttributes(unsigned int* destination, const unsigned int* indices, size_t index_count, const float* vertex_data, size_t vertex_count, size_t vertex_stride, size_t target_index_count, float target_error, float* out_result_error, const float* attributes, const float* attribute_weights, size_t attribute_count)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_stride > 0 && vertex_stride <= 256);
	assert(vertex_stride % sizeof(float) == 0);
	assert(target_index_count <= index_count);
	assert(attribute_count <= ATTRIBUTES);

	meshopt_Allocator allocator;

	unsigned int* result = destination;

	// build adjacency information
	EdgeAdjacency adjacency = {};
	prepareEdgeAdjacency(adjacency, index_count, vertex_count, allocator);
	updateEdgeAdjacency(adjacency, indices, index_count, vertex_count, NULL);

	// build position remap that maps each vertex to the one with identical position
	unsigned int* remap = allocator.allocate<unsigned int>(vertex_count);
	unsigned int* wedge = allocator.allocate<unsigned int>(vertex_count);
	buildPositionRemap(remap, wedge, vertex_data, vertex_count, vertex_stride, allocator);

	// classify vertices; vertex kind determines collapse rules, see kCanCollapse
	unsigned char* vertex_kind = allocator.allocate<unsigned char>(vertex_count);
	unsigned int* loop = allocator.allocate<unsigned int>(vertex_count);
	unsigned int* loopback = allocator.allocate<unsigned int>(vertex_count);
	classifyVertices(vertex_kind, loop, loopback, vertex_count, adjacency, remap, wedge);

#if TRACE
	size_t unique_positions = 0;
	for (size_t i = 0; i < vertex_count; ++i)
		unique_positions += remap[i] == i;

	printf("position remap: %d vertices => %d positions\n", int(vertex_count), int(unique_positions));

	size_t kinds[Kind_Count] = {};
	for (size_t i = 0; i < vertex_count; ++i)
		kinds[vertex_kind[i]] += remap[i] == i;

	printf("kinds: manifold %d, border %d, seam %d, complex %d, locked %d\n",
	    int(kinds[Kind_Manifold]), int(kinds[Kind_Border]), int(kinds[Kind_Seam]), int(kinds[Kind_Complex]), int(kinds[Kind_Locked]));
#endif

	Vector3* vertex_positions = allocator.allocate<Vector3>(vertex_count);
	rescalePositions(vertex_positions, vertex_data, vertex_count, vertex_stride);

#if ATTRIBUTES
	for (size_t i = 0; i < vertex_count; ++i)
	{
		memset(vertex_positions[i].a, 0, sizeof(vertex_positions[i].a));

		for (size_t k = 0; k < attribute_count; ++k)
		{
			float a = attributes[i * attribute_count + k];

			vertex_positions[i].a[k] = a * attribute_weights[k];
		}
	}
#endif

	Quadric* vertex_quadrics = allocator.allocate<Quadric>(vertex_count);
	memset(vertex_quadrics, 0, vertex_count * sizeof(Quadric));
	Quadric* vertex_no_attrib_quadrics = allocator.allocate<Quadric>(vertex_count);
	memset(vertex_no_attrib_quadrics, 0, vertex_count * sizeof(Quadric));

	fillFaceQuadrics(vertex_quadrics, vertex_no_attrib_quadrics, indices, index_count, vertex_positions, remap);
	fillEdgeQuadrics(vertex_quadrics, vertex_no_attrib_quadrics, indices, index_count, vertex_positions, remap, vertex_kind, loop, loopback);

	if (result != indices)
		memcpy(result, indices, index_count * sizeof(unsigned int));

#if TRACE
	size_t pass_count = 0;
#endif

	Collapse* edge_collapses = allocator.allocate<Collapse>(index_count);
	unsigned int* collapse_order = allocator.allocate<unsigned int>(index_count);
	unsigned int* collapse_remap = allocator.allocate<unsigned int>(vertex_count);
	unsigned char* collapse_locked = allocator.allocate<unsigned char>(vertex_count);

	size_t result_count = index_count;
	float result_error = 0;

	// target_error input is linear; we need to adjust it to match quadricError units
	float error_limit = target_error * target_error;

	while (result_count > target_index_count)
	{
		// note: throughout the simplification process adjacency structure reflects welded topology for result-in-progress
		updateEdgeAdjacency(adjacency, result, result_count, vertex_count, remap);

		size_t edge_collapse_count = pickEdgeCollapses(edge_collapses, result, result_count, remap, vertex_kind, loop);

		// no edges can be collapsed any more due to topology restrictions
		if (edge_collapse_count == 0)
			break;

		rankEdgeCollapses(edge_collapses, edge_collapse_count, vertex_positions, vertex_quadrics, vertex_no_attrib_quadrics, remap);

#if TRACE > 1
		dumpEdgeCollapses(edge_collapses, edge_collapse_count, vertex_kind);
#endif

		sortEdgeCollapses(collapse_order, edge_collapses, edge_collapse_count);

		size_t triangle_collapse_goal = (result_count - target_index_count) / 3;

		for (size_t i = 0; i < vertex_count; ++i)
			collapse_remap[i] = unsigned(i);

		memset(collapse_locked, 0, vertex_count);

#if TRACE
		printf("pass %d: ", int(pass_count++));
#endif

		size_t collapses = performEdgeCollapses(collapse_remap, collapse_locked, vertex_quadrics, vertex_no_attrib_quadrics, edge_collapses, edge_collapse_count, collapse_order, remap, wedge, vertex_kind, vertex_positions, adjacency, triangle_collapse_goal, error_limit, result_error);

		// no edges can be collapsed any more due to hitting the error limit or triangle collapse limit
		if (collapses == 0)
			break;

		remapEdgeLoops(loop, vertex_count, collapse_remap);
		remapEdgeLoops(loopback, vertex_count, collapse_remap);

		size_t new_count = remapIndexBuffer(result, result_count, collapse_remap);
		assert(new_count < result_count);

		result_count = new_count;
	}

#if TRACE
	printf("result: %d triangles, error: %e; total %d passes\n", int(result_count), sqrtf(result_error), int(pass_count));
#endif

#if TRACE > 1
	dumpLockedCollapses(result, result_count, vertex_kind);
#endif

#ifndef NDEBUG
	if (meshopt_simplifyDebugKind)
		memcpy(meshopt_simplifyDebugKind, vertex_kind, vertex_count);

	if (meshopt_simplifyDebugLoop)
		memcpy(meshopt_simplifyDebugLoop, loop, vertex_count * sizeof(unsigned int));

	if (meshopt_simplifyDebugLoopBack)
		memcpy(meshopt_simplifyDebugLoopBack, loopback, vertex_count * sizeof(unsigned int));
#endif

	// result_error is quadratic; we need to remap it back to linear
	if (out_result_error)
	{
		*out_result_error = sqrtf(result_error);
	}

	return result_count;
}

size_t meshopt_simplifySloppy(unsigned int* destination, const unsigned int* indices, size_t index_count, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride, size_t target_index_count, float target_error, float* out_result_error)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride > 0 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);
	assert(target_index_count <= index_count);

	// we expect to get ~2 triangles/vertex in the output
	size_t target_cell_count = target_index_count / 6;

	meshopt_Allocator allocator;

	Vector3* vertex_positions = allocator.allocate<Vector3>(vertex_count);
	rescalePositions(vertex_positions, vertex_positions_data, vertex_count, vertex_positions_stride);

	// find the optimal grid size using guided binary search
#if TRACE
	printf("source: %d vertices, %d triangles\n", int(vertex_count), int(index_count / 3));
	printf("target: %d cells, %d triangles\n", int(target_cell_count), int(target_index_count / 3));
#endif

	unsigned int* vertex_ids = allocator.allocate<unsigned int>(vertex_count);

	const int kInterpolationPasses = 5;

	// invariant: # of triangles in min_grid <= target_count
	int min_grid = int(1.f / (target_error < 1e-3f ? 1e-3f : target_error));
	int max_grid = 1025;
	size_t min_triangles = 0;
	size_t max_triangles = index_count / 3;

	// when we're error-limited, we compute the triangle count for the min. size; this accelerates convergence and provides the correct answer when we can't use a larger grid
	if (min_grid > 1)
	{
		computeVertexIds(vertex_ids, vertex_positions, vertex_count, min_grid);
		min_triangles = countTriangles(vertex_ids, indices, index_count);
	}

	// instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
	int next_grid_size = int(sqrtf(float(target_cell_count)) + 0.5f);

	for (int pass = 0; pass < 10 + kInterpolationPasses; ++pass)
	{
		if (min_triangles >= target_index_count / 3 || max_grid - min_grid <= 1)
			break;

		// we clamp the prediction of the grid size to make sure that the search converges
		int grid_size = next_grid_size;
		grid_size = (grid_size <= min_grid) ? min_grid + 1 : (grid_size >= max_grid) ? max_grid - 1 : grid_size;

		computeVertexIds(vertex_ids, vertex_positions, vertex_count, grid_size);
		size_t triangles = countTriangles(vertex_ids, indices, index_count);

#if TRACE
		printf("pass %d (%s): grid size %d, triangles %d, %s\n",
		    pass, (pass == 0) ? "guess" : (pass <= kInterpolationPasses) ? "lerp" : "binary",
		    grid_size, int(triangles),
		    (triangles <= target_index_count / 3) ? "under" : "over");
#endif

		float tip = interpolate(float(target_index_count / 3), float(min_grid), float(min_triangles), float(grid_size), float(triangles), float(max_grid), float(max_triangles));

		if (triangles <= target_index_count / 3)
		{
			min_grid = grid_size;
			min_triangles = triangles;
		}
		else
		{
			max_grid = grid_size;
			max_triangles = triangles;
		}

		// we start by using interpolation search - it usually converges faster
		// however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
		next_grid_size = (pass < kInterpolationPasses) ? int(tip + 0.5f) : (min_grid + max_grid) / 2;
	}

	if (min_triangles == 0)
	{
		if (out_result_error)
			*out_result_error = 1.f;

		return 0;
	}

	// build vertex->cell association by mapping all vertices with the same quantized position to the same cell
	size_t table_size = hashBuckets2(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);

	unsigned int* vertex_cells = allocator.allocate<unsigned int>(vertex_count);

	computeVertexIds(vertex_ids, vertex_positions, vertex_count, min_grid);
	size_t cell_count = fillVertexCells(table, table_size, vertex_cells, vertex_ids, vertex_count);

	// build a quadric for each target cell
	Quadric* cell_quadrics = allocator.allocate<Quadric>(cell_count);
	memset(cell_quadrics, 0, cell_count * sizeof(Quadric));

	fillCellQuadrics(cell_quadrics, indices, index_count, vertex_positions, vertex_cells);

	// for each target cell, find the vertex with the minimal error
	unsigned int* cell_remap = allocator.allocate<unsigned int>(cell_count);
	float* cell_errors = allocator.allocate<float>(cell_count);

	fillCellRemap(cell_remap, cell_errors, cell_count, vertex_cells, cell_quadrics, vertex_positions, vertex_count);

	// compute error
	float result_error = 0.f;

	for (size_t i = 0; i < cell_count; ++i)
		result_error = result_error < cell_errors[i] ? cell_errors[i] : result_error;

	// collapse triangles!
	// note that we need to filter out triangles that we've already output because we very frequently generate redundant triangles between cells :(
	size_t tritable_size = hashBuckets2(min_triangles);
	unsigned int* tritable = allocator.allocate<unsigned int>(tritable_size);

	size_t write = filterTriangles(destination, tritable, tritable_size, indices, index_count, vertex_cells, cell_remap);

#if TRACE
	printf("result: %d cells, %d triangles (%d unfiltered), error %e\n", int(cell_count), int(write / 3), int(min_triangles), sqrtf(result_error));
#endif

	if (out_result_error)
		*out_result_error = sqrtf(result_error);

	return write;
}

size_t meshopt_simplifyPoints(unsigned int* destination, const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride, size_t target_vertex_count)
{
	using namespace meshopt;

	assert(vertex_positions_stride > 0 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);
	assert(target_vertex_count <= vertex_count);

	size_t target_cell_count = target_vertex_count;

	if (target_cell_count == 0)
		return 0;

	meshopt_Allocator allocator;

	Vector3* vertex_positions = allocator.allocate<Vector3>(vertex_count);
	rescalePositions(vertex_positions, vertex_positions_data, vertex_count, vertex_positions_stride);

	// find the optimal grid size using guided binary search
#if TRACE
	printf("source: %d vertices\n", int(vertex_count));
	printf("target: %d cells\n", int(target_cell_count));
#endif

	unsigned int* vertex_ids = allocator.allocate<unsigned int>(vertex_count);

	size_t table_size = hashBuckets2(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);

	const int kInterpolationPasses = 5;

	// invariant: # of vertices in min_grid <= target_count
	int min_grid = 0;
	int max_grid = 1025;
	size_t min_vertices = 0;
	size_t max_vertices = vertex_count;

	// instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
	int next_grid_size = int(sqrtf(float(target_cell_count)) + 0.5f);

	for (int pass = 0; pass < 10 + kInterpolationPasses; ++pass)
	{
		assert(min_vertices < target_vertex_count);
		assert(max_grid - min_grid > 1);

		// we clamp the prediction of the grid size to make sure that the search converges
		int grid_size = next_grid_size;
		grid_size = (grid_size <= min_grid) ? min_grid + 1 : (grid_size >= max_grid) ? max_grid - 1 : grid_size;

		computeVertexIds(vertex_ids, vertex_positions, vertex_count, grid_size);
		size_t vertices = countVertexCells(table, table_size, vertex_ids, vertex_count);

#if TRACE
		printf("pass %d (%s): grid size %d, vertices %d, %s\n",
		    pass, (pass == 0) ? "guess" : (pass <= kInterpolationPasses) ? "lerp" : "binary",
		    grid_size, int(vertices),
		    (vertices <= target_vertex_count) ? "under" : "over");
#endif

		float tip = interpolate(float(target_vertex_count), float(min_grid), float(min_vertices), float(grid_size), float(vertices), float(max_grid), float(max_vertices));

		if (vertices <= target_vertex_count)
		{
			min_grid = grid_size;
			min_vertices = vertices;
		}
		else
		{
			max_grid = grid_size;
			max_vertices = vertices;
		}

		if (vertices == target_vertex_count || max_grid - min_grid <= 1)
			break;

		// we start by using interpolation search - it usually converges faster
		// however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
		next_grid_size = (pass < kInterpolationPasses) ? int(tip + 0.5f) : (min_grid + max_grid) / 2;
	}

	if (min_vertices == 0)
		return 0;

	// build vertex->cell association by mapping all vertices with the same quantized position to the same cell
	unsigned int* vertex_cells = allocator.allocate<unsigned int>(vertex_count);

	computeVertexIds(vertex_ids, vertex_positions, vertex_count, min_grid);
	size_t cell_count = fillVertexCells(table, table_size, vertex_cells, vertex_ids, vertex_count);

	// build a quadric for each target cell
	Quadric* cell_quadrics = allocator.allocate<Quadric>(cell_count);
	memset(cell_quadrics, 0, cell_count * sizeof(Quadric));

	fillCellQuadrics(cell_quadrics, vertex_positions, vertex_count, vertex_cells);

	// for each target cell, find the vertex with the minimal error
	unsigned int* cell_remap = allocator.allocate<unsigned int>(cell_count);
	float* cell_errors = allocator.allocate<float>(cell_count);

	fillCellRemap(cell_remap, cell_errors, cell_count, vertex_cells, cell_quadrics, vertex_positions, vertex_count);

	// copy results to the output
	assert(cell_count <= target_vertex_count);
	memcpy(destination, cell_remap, sizeof(unsigned int) * cell_count);

#if TRACE
	printf("result: %d cells\n", int(cell_count));
#endif

	return cell_count;
}

float meshopt_simplifyScale(const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(vertex_positions_stride > 0 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);

	float extent = rescalePositions(NULL, vertex_positions, vertex_count, vertex_positions_stride);

	return extent;
}
