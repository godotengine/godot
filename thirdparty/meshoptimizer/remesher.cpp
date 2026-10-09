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

// Note: this is only exposed for development purposes; do *not* use
enum
{
	meshopt_RemeshInternalDebug = 1 << 30
};

// This work is based on:
// William Lorensen, Harvey Cline. Marching Cubes: A High Resolution 3D Surface Construction Algorithm. 1987
// Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. 1997
namespace meshopt
{

// triangles are encoded as three vertex codes in three nibbles
// vertex codes match the bit order XYZ (e.g. 3 maps to X=1 Y=1 Z=0)
struct Case
{
	unsigned char code;
	unsigned char altmode;
	unsigned short triangles[5]; // up to 4 triangles and a null terminator
	unsigned short alternate[5]; // alternate triangulation, restricted to have <= triangles vs primary
};

// the base cases establish a canonical triangulation; unlike classical Marching Cubes, we connect cube *corners* instead of edges
// each case may have an alternate triangulation, used based on deciders; crucially, open boundary edges must stay consistent as decider varies per cell
static const Case kBaseCases[] = {
    {0x00, 0, {}, {}},
    {0x01, 0, {}, {}},
    {0x03, 0, {}, {}},
    {0x06, 0, {}, {}},
    {0x07, 0, {0x120}, {}},
    {0x0f, 2, {0x130, 0x032}, {0x321, 0x120}},
    {0x16, 1, {0x124, 0x142}, {}},
    {0x17, 0, {0x124}, {}},
    {0x18, 0, {}, {}},
    {0x19, 1, {0x043, 0x403}, {}},
    {0x1b, 1, {0x134, 0x430}, {0x410, 0x013}},
    {0x1d, 1, {0x243, 0x340}, {0x320, 0x024}},
    {0x1e, 1, {0x142, 0x432, 0x341}, {0x132}},
    {0x1f, 2, {0x412, 0x213}, {0x134, 0x432}},
    {0x3c, 1, {0x345, 0x324, 0x543, 0x423}, {}},
    {0x3d, 1, {0x253, 0x524, 0x350}, {0x240, 0x504, 0x320}},
    {0x3f, 2, {0x325, 0x524}, {0x243, 0x345}},
    {0x69, 1, {0x035, 0x560, 0x306, 0x536}, {}},
    {0x6b, 1, {0x063, 0x605, 0x365}, {0x305}},
    {0x6f, 1, {0x365, 0x560}, {0x530, 0x036}},
    {0x7e, 0, {0x142, 0x536}, {}},
    {0x7f, 0, {0x365}, {}},
    {0xf0, 2, {0x547, 0x746}, {0x465, 0x567}},
    {0xff, 0, {}, {}},
};

// for each cube case, a triangulation is derived from the base cases via 90 degree rotations
static unsigned short kTriangleTable[256][2][5];
static unsigned char kTriangleCount[256];
static unsigned char kTriangleAlt[256];

static bool buildRemeshTables()
{
	// corner remap for 90 degree rotations around X and Y
	static const unsigned char rotations[2][8] = {{2, 3, 6, 7, 0, 1, 4, 5}, {4, 0, 6, 2, 5, 1, 7, 3}};

	unsigned char filled[256] = {};

	// copy base cases as is; we will generate the rest via rotations
	for (size_t i = 0; i < sizeof(kBaseCases) / sizeof(kBaseCases[0]); ++i)
	{
		const Case& c = kBaseCases[i];
		memcpy(kTriangleTable[c.code][0], c.triangles, sizeof(c.triangles));
		memcpy(kTriangleTable[c.code][1], c.altmode ? c.alternate : c.triangles, sizeof(c.triangles));
		kTriangleAlt[c.code] = c.altmode;
		filled[kBaseCases[i].code] = 1;
	}

	// propagate base cases iteratively to fill all cube cases via rotations
	// note: the worst case is 5 passes but in practice we only need 3 to fill all cases
	for (int pass = 0; pass < 3; ++pass)
		for (int code = 0; code < 256; ++code)
		{
			if (filled[code] != 1)
				continue;

			for (int r = 0; r < 2; ++r)
			{
				unsigned char rotated = 0;
				for (int i = 0; i < 8; ++i)
					rotated |= ((code >> i) & 1) << rotations[r][i];

				if (filled[rotated])
					continue;

				for (int k = 0; k < 2; ++k)
					for (int i = 0; kTriangleTable[code][k][i]; ++i)
					{
						unsigned short tri = kTriangleTable[code][k][i];
						kTriangleTable[rotated][k][i] = (rotations[r][(tri >> 8) & 0xf] << 8) | (rotations[r][(tri >> 4) & 0xf] << 4) | rotations[r][tri & 0xf];
					}

				kTriangleAlt[rotated] = kTriangleAlt[code];
				filled[rotated] = 1; // mark as pending
			}

			filled[code] = 2; // mark as processed
		}

	// finalize auxiliary tables
	for (int code = 0; code < 256; ++code)
	{
		assert(filled[code]);

		int count = 0;
		while (kTriangleTable[code][0][count])
			count++;

		kTriangleCount[code] = (unsigned char)count;

		// counting pass does not use alternate triangulations, so they need to stay at or below primary triangle count
		assert(kTriangleTable[code][1][count] == 0);

		// quadric decider expects a canonical quad encoding (0xabc 0xcbd)
		if (kTriangleAlt[code] == 2)
			assert(count == 2 && (kTriangleTable[code][0][0] & 0xf0) == (kTriangleTable[code][0][1] & 0xf0) && (kTriangleTable[code][0][0] & 0xf) == (kTriangleTable[code][0][1] >> 8));
	}

	return true;
}

static bool gRemeshTablesInitialized = buildRemeshTables();

struct Voxel
{
	unsigned int coord;
	unsigned char octants;

	float px, py, pz;
	float w;

	// a00*x^2 + a11*y^2 + a22*z^2 + 2*a10*xy + 2*a20*xz + 2*a21*yz + 2*b0*x + 2*b1*y + 2*b2*z + c
	float a00, a11, a22;
	float a10, a20, a21;
	float b0, b1, b2, c;
};

static void voxelAccumulate(Voxel& vox, float px, float py, float pz, float w)
{
	vox.px += px * w;
	vox.py += py * w;
	vox.pz += pz * w;
	vox.w += w;
}

static void voxelAccumulateQ(Voxel& vox, float a, float b, float c, float d, float w)
{
	float aw = a * w;
	float bw = b * w;
	float cw = c * w;
	float dw = d * w;

	vox.a00 += a * aw;
	vox.a11 += b * bw;
	vox.a22 += c * cw;
	vox.a10 += a * bw;
	vox.a20 += a * cw;
	vox.a21 += b * cw;
	vox.b0 += a * dw;
	vox.b1 += b * dw;
	vox.b2 += c * dw;
	vox.c += d * dw;
}

static float voxelError(const Voxel& vox, float x, float y, float z)
{
	float rx = (vox.b0 + vox.a10 * y) * 2.f + vox.a00 * x;
	float ry = (vox.b1 + vox.a21 * z) * 2.f + vox.a11 * y;
	float rz = (vox.b2 + vox.a20 * x) * 2.f + vox.a22 * z;

	return fabsf(vox.c + rx * x + ry * y + rz * z);
}

static bool voxelSolve(float& rx, float& ry, float& rz, const Voxel& vox, float lambda)
{
	// solve A*p = -b where A is the quadric matrix and b is the linear term
	float rw = lambda * vox.w;
	float a00 = rw + vox.a00, a11 = rw + vox.a11, a22 = rw + vox.a22;
	float a10 = vox.a10, a20 = vox.a20, a21 = vox.a21;
	float x0 = lambda * vox.px - vox.b0, x1 = lambda * vox.py - vox.b1, x2 = lambda * vox.pz - vox.b2;

	float eps = 1e-6f * vox.w;

	// LDL decomposition: A = LDL^T
	float d0 = a00;
	float l10 = a10 / d0;
	float l20 = a20 / d0;

	float d1 = a11 - a10 * l10;
	float dl21 = a21 - a20 * l10;
	float l21 = dl21 / d1;

	float d2 = a22 - a20 * l20 - dl21 * l21;

	// solve L*y = x
	float y0 = x0;
	float y1 = x1 - l10 * y0;
	float y2 = x2 - l20 * y0 - l21 * y1;

	// solve D*z = y
	float z0 = y0 / d0;
	float z1 = y1 / d1;
	float z2 = y2 / d2;

	// substitute L^T*p = z
	rz = z2;
	ry = z1 - l21 * rz;
	rx = z0 - l10 * ry - l20 * rz;

	return fabsf(d0) > eps && fabsf(d1) > eps && fabsf(d2) > eps;
}

static float measureGrid(const float* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride, int resolution, float* out_offset)
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

	// small subvoxel padding keeps vertices inside the last valid voxel to avoid clamping correctness issues
	extent *= (resolution + 1e-2f) / float(resolution);

	// rescale extents to [0..resolution - 2], because the first and last voxels are used as empty padding
	float scale = extent == 0 ? 0.f : (resolution - 2) / extent;

	// center mesh in the grid; this improves voxelization for symmetric inputs
	for (int j = 0; j < 3; ++j)
		out_offset[j] = minv[j] - (extent - (maxv[j] - minv[j])) * 0.5f;

	return scale;
}

static void voxelize(unsigned char* grid, const unsigned int* rowmap, Voxel* voxels, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, int resolution, float scale, const float offset[3], unsigned int options)
{
	(void)vertex_count;

	size_t vertex_stride_float = vertex_positions_stride / sizeof(float);

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		const float* va = vertex_positions + a * vertex_stride_float;
		const float* vb = vertex_positions + b * vertex_stride_float;
		const float* vc = vertex_positions + c * vertex_stride_float;

		float ex = vb[0] - va[0], ey = vb[1] - va[1], ez = vb[2] - va[2];
		float fx = vc[0] - va[0], fy = vc[1] - va[1], fz = vc[2] - va[2];
		float gx = vc[0] - vb[0], gy = vc[1] - vb[1], gz = vc[2] - vb[2];

		// use maximum edge length to establish sampling rate
		float el2 = ex * ex + ey * ey + ez * ez;
		float fl2 = fx * fx + fy * fy + fz * fz;
		float gl2 = gx * gx + gy * gy + gz * gz;

		float max_edge = el2 > fl2 ? el2 : fl2;
		max_edge = max_edge > gl2 ? max_edge : gl2;
		max_edge = sqrtf(max_edge);

		// we target 2 samples per voxel edge which should be enough to hit all voxels at any rotation
		int samples = int(max_edge * scale * 2.f);
		samples = samples > 1 ? samples : 1;
		samples = samples < resolution * 2 ? samples : resolution * 2;

		// normal is used to compute area for weighting as well as for the quadric
		float nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx;
		float area = sqrtf(nx * nx + ny * ny + nz * nz);

		// skip degenerate triangles; they don't contribute to voxelization and may result in NaN when computing voxel centroid
		if (area == 0.f)
			continue;

		float ns = 1.f / area;
		nx *= ns, ny *= ns, nz *= ns;

		float sx = va[0] - offset[0], sy = va[1] - offset[1], sz = va[2] - offset[2];
		float sr = 1.f;
		float weight = area / float((samples + 1) * (samples + 2));

		// this division is on a critical path, and for dense input meshes the branch is almost never taken so it's worth skipping it
		if (samples > 1)
			sr = 1.f / float(samples);

		for (int u = 0; u <= samples; ++u)
			for (int v = 0; v <= samples - u; ++v)
			{
				float su = float(u) * sr, sv = float(v) * sr;
				float px = sx + su * ex + sv * fx;
				float py = sy + su * ey + sv * fy;
				float pz = sz + su * ez + sv * fz;

				// compute coordinates with an extra bit to use in octant mask
				int hx = int(px * (scale * 2));
				int hy = int(py * (scale * 2));
				int hz = int(pz * (scale * 2));
				int x = hx >> 1, y = hy >> 1, z = hz >> 1;

				// safety: rounding errors and non-finite inputs may produce out of bounds coordinates, so we clamp them
				int cutoff = resolution - 3;
				x = unsigned(x) < unsigned(cutoff) ? x : cutoff;
				y = unsigned(y) < unsigned(cutoff) ? y : cutoff;
				z = unsigned(z) < unsigned(cutoff) ? z : cutoff;

				size_t row = (y + 1) + size_t(resolution) * (z + 1);
				size_t idx = (x + 1) + size_t(resolution) * row;

				if (voxels)
				{
					assert(grid[idx] != 0 && grid[idx] != 0xff);
					Voxel& vox = voxels[rowmap[row] + (grid[idx] - 1)];

					vox.coord = (unsigned(x) << 20) | (unsigned(y) << 10) | unsigned(z);
					vox.octants |= 1 << ((hx & 1) | ((hy & 1) << 1) | ((hz & 1) << 2));

					voxelAccumulate(vox, px, py, pz, weight);

					if (options & meshopt_RemeshSolve)
					{
						float distance = nx * px + ny * py + nz * pz;
						voxelAccumulateQ(vox, nx, ny, nz, -distance, weight);
					}
				}
				else
				{
					grid[idx] = 1;
				}
			}
	}
}

static size_t rowpack(unsigned char* grid, unsigned int* rowmap, int resolution)
{
	size_t result = 0;
	size_t slice = size_t(resolution) * size_t(resolution);

	unsigned char zero[256];
	memset(zero, 0, resolution);

	for (size_t i = 0; i < slice; ++i)
	{
		unsigned char* data = grid + i * size_t(resolution);

		if (memcmp(data, zero, resolution) == 0)
		{
			// mark empty rows with a sentinel to accelerate further processing
			rowmap[i] = ~0u;
			continue;
		}

		int count = 0;

		for (int x = 0; x < resolution; ++x)
		{
			assert(data[x] <= 1); // voxelize only produces 0/1 values

			count += data[x];
			data[x] = (data[x] != 0) ? (unsigned char)count : 0;
		}

		assert(count < 255); // we store offsets in a single byte, with 0 reserved for empty voxels and 0xff reserved for interior voxels

		rowmap[i] = unsigned(result);
		result += count;
	}

	return result;
}

static void solidifyQueue(unsigned int row, unsigned int* worklist, unsigned char* queued, size_t& pending)
{
	if (queued[row])
		return;

	queued[row] = 1;
	worklist[pending++] = row;
}

static void solidify(unsigned char* grid, const unsigned int* rowmap, unsigned int* worklist, unsigned char* queued, int resolution)
{
	size_t pending = 0;
	memset(queued, 0, size_t(resolution) * size_t(resolution));

	// mark the interior empty voxels as 'inside'; we will propagate 'empty' state from the boundary
	for (int z = 1; z < resolution - 1; ++z)
		for (int y = 1; y < resolution - 1; ++y)
		{
			// rows without occupied voxels shortcircuit the interior propagation and are kept as 'empty'
			if (rowmap[y + size_t(resolution) * z] == ~0u)
				continue;

			unsigned char* data = grid + size_t(resolution) * (y + size_t(resolution) * z);

			for (int x = 1; x < resolution - 1; ++x)
				data[x] = (data[x] == 0) ? 0xff : data[x];
		}

	// queue all rows for processing; note that boundary rows are always empty but we need them to propagate into neighboring rows
	for (int z = 0; z < resolution; ++z)
		for (int y = 0; y < resolution; ++y)
			solidifyQueue(y + size_t(resolution) * z, worklist, queued, pending);

	while (pending)
	{
		unsigned int row = worklist[--pending];
		assert(queued[row]);
		queued[row] = 0;

		unsigned char* data = grid + size_t(resolution) * row;

		// propagate outside state to the interior within row; rows without occupied voxels have no 'inside' voxels
		if (rowmap[row] != ~0u)
		{
			for (int x = 1; x < resolution - 1; ++x)
				data[x] = (data[x] == 0xff && data[x - 1] == 0) ? 0 : data[x];

			for (int x = resolution - 2; x >= 1; --x)
				data[x] = (data[x] == 0xff && data[x + 1] == 0) ? 0 : data[x];
		}

		// propagate outside state to the interior into neighboring rows
		int y = row % resolution, z = row / resolution;

		for (int k = 0; k < 4; ++k)
		{
			int yn = y + (k == 0 ? -1 : (k == 1 ? 1 : 0));
			int zn = z + (k == 2 ? -1 : (k == 3 ? 1 : 0));

			if (yn < 1 || yn >= resolution - 1 || zn < 1 || zn >= resolution - 1)
				continue;

			// neighboring rows without occupied voxels have no 'inside' voxels and can be skipped
			if (rowmap[yn + size_t(resolution) * zn] == ~0u)
				continue;

			unsigned char* datan = grid + size_t(resolution) * (yn + size_t(resolution) * zn);
			unsigned char changed = 0;

			for (int x = 1; x < resolution - 1; ++x)
			{
				unsigned char rep = ((data[x] == 0) & (datan[x] == 0xff)) ? 0 : datan[x];
				changed |= rep ^ datan[x];
				datan[x] = rep;
			}

			if (changed)
				solidifyQueue(yn + size_t(resolution) * zn, worklist, queued, pending);
		}
	}
}

static void solve(Voxel* voxels, size_t voxel_count, float scale, unsigned int options)
{
	// regularization factor is a tradeoff between solve stability/uniformity and adherence to the original surface
	float factor = 3e-2f;

	// cutoff is a squared distance in model space; sqrt(3)/scale is the voxel diagonal
	float cutoff = 3.f / (scale * scale);
	float rscale = 1.f / scale;

	size_t solved = 0;

	for (size_t i = 0; i < voxel_count; ++i)
	{
		Voxel& vox = voxels[i];
		float px = vox.px / vox.w, py = vox.py / vox.w, pz = vox.pz / vox.w;

		// voxel corner, used for clamping and debug visualization
		float cx = int((vox.coord >> 20) & 1023) * rscale;
		float cy = int((vox.coord >> 10) & 1023) * rscale;
		float cz = int((vox.coord >> 0) & 1023) * rscale;

		if (options & meshopt_RemeshSolve)
		{
			// compute minimizing point; this includes regularization to stabilize the solve
			float sx, sy, sz;
			if (voxelSolve(sx, sy, sz, vox, factor))
			{
				float d = (sx - px) * (sx - px) + (sy - py) * (sy - py) + (sz - pz) * (sz - pz);

				// reject solutions that move the vertex too far from voxel centroid; this is a safety net in case regularization is insufficient
				if (d < cutoff)
				{
					// use a clamped solution to reduce intersections
					px = sx > cx ? sx : cx;
					px = px < cx + rscale ? px : cx + rscale;
					py = sy > cy ? sy : cy;
					py = py < cy + rscale ? py : cy + rscale;
					pz = sz > cz ? sz : cz;
					pz = pz < cz + rscale ? pz : cz + rscale;

					solved++;
				}
			}
		}

		if (options & meshopt_RemeshInternalDebug)
		{
			px = cx + 0.5f * rscale;
			py = cy + 0.5f * rscale;
			pz = cz + 0.5f * rscale;
		}

		vox.px = px;
		vox.py = py;
		vox.pz = pz;
	}

	(void)solved;

#if TRACE
	if (options & meshopt_RemeshSolve)
		printf("remesher: solved %zu/%zu voxels\n", solved, voxel_count);
#endif
}

static void emitVertex(float* result, int x, int y, int z, int corner, const unsigned char* grid, const unsigned int* rowmap, const Voxel* voxels, int resolution, const float offset[3])
{
	int ox = corner & 1, oy = (corner >> 1) & 1, oz = (corner >> 2) & 1;

	size_t row = (y + oy) + size_t(resolution) * (z + oz);
	size_t idx = (x + ox) + size_t(resolution) * row;

	assert(grid[idx] != 0 && grid[idx] != 0xff);
	const Voxel& vox = voxels[rowmap[row] + (grid[idx] - 1)];

	result[0] = vox.px + offset[0];
	result[1] = vox.py + offset[1];
	result[2] = vox.pz + offset[2];
}

static bool octantDecider(int x, int y, int z, int cube, const unsigned char* grid, const unsigned int* rowmap, const Voxel* voxels, int resolution)
{
	for (int c = 0; c < 8; ++c)
		if (cube & (1 << c))
		{
			int ox = c & 1, oy = (c >> 1) & 1, oz = (c >> 2) & 1;

			size_t row = (y + oy) + size_t(resolution) * (z + oz);
			size_t idx = (x + ox) + size_t(resolution) * row;

			assert(grid[idx] != 0);

			// note: unlike vertex emission, which never consults interior occupied corners, here we need to skip corners without an occupied voxel
			if (grid[idx] == 0xff)
				continue;

			const Voxel& vox = voxels[rowmap[row] + (grid[idx] - 1)];

			// test octant contained within the cell (opposite of the corner index)
			if (vox.octants & (1 << (7 - c)))
				return false;
		}

	// select alternate configuration if *all* octants are empty
	return true;
}

static bool quadricDecider(int x, int y, int z, int cube, const unsigned char* grid, const unsigned int* rowmap, const Voxel* voxels, int resolution, float rscale)
{
	// quads are encoded implicitly as 0xabc 0xcbd
	unsigned int quad = (kTriangleTable[cube][0][0] << 4) | (kTriangleTable[cube][0][1] & 0xf);

	const Voxel* corner[4];

	for (int i = 0; i < 4; ++i)
	{
		int c = (quad >> (12 - i * 4)) & 0xf;
		size_t row = (y + ((c >> 1) & 1)) + size_t(resolution) * (z + ((c >> 2) & 1));
		size_t idx = (x + (c & 1)) + size_t(resolution) * row;

		assert(grid[idx] != 0 && grid[idx] != 0xff);
		corner[i] = &voxels[rowmap[row] + (grid[idx] - 1)];
	}

	// accumulate the quadrics of all four corners; we sum (a+d)+(b+c) to get the same results on both sides of thin sheets
	const Voxel *va = corner[0], *vb = corner[1], *vc = corner[2], *vd = corner[3];

	Voxel sum = {};
	sum.a00 = (va->a00 + vd->a00) + (vb->a00 + vc->a00), sum.a11 = (va->a11 + vd->a11) + (vb->a11 + vc->a11), sum.a22 = (va->a22 + vd->a22) + (vb->a22 + vc->a22);
	sum.a10 = (va->a10 + vd->a10) + (vb->a10 + vc->a10), sum.a20 = (va->a20 + vd->a20) + (vb->a20 + vc->a20), sum.a21 = (va->a21 + vd->a21) + (vb->a21 + vc->a21);
	sum.b0 = (va->b0 + vd->b0) + (vb->b0 + vc->b0), sum.b1 = (va->b1 + vd->b1) + (vb->b1 + vc->b1), sum.b2 = (va->b2 + vd->b2) + (vb->b2 + vc->b2);
	sum.c = (va->c + vd->c) + (vb->c + vc->c), sum.w = (va->w + vd->w) + (vb->w + vc->w);

	// evaluate error for midpoints of primary (bc) and alternate (ad) diagonal
	float error0 = voxelError(sum, (vb->px + vc->px) * 0.5f, (vb->py + vc->py) * 0.5f, (vb->pz + vc->pz) * 0.5f);
	float error1 = voxelError(sum, (va->px + vd->px) * 0.5f, (va->py + vd->py) * 0.5f, (va->pz + vd->pz) * 0.5f);

	// select alternate configuration if it is clearly better and if the primary error is not too small, to avoid excessive flips
	return error0 > sum.w * (0.1f * 0.1f * rscale * rscale) && error1 < error0 * (0.85f * 0.85f);
}

static size_t polygonize(float* destination, size_t max_triangle_count, const unsigned char* grid, const unsigned int* rowmap, const Voxel* voxels, int resolution, float scale, const float offset[3], unsigned int options)
{
	size_t result = 0;
	size_t slice = size_t(resolution) * size_t(resolution);
	float rscale = 1.f / scale;

	assert(gRemeshTablesInitialized);

	for (int z = 0; z < resolution - 1; ++z)
		for (int y = 0; y < resolution - 1; ++y)
		{
			size_t row = y + size_t(resolution) * z;

			// skip processing if all four rows are entirely empty as that guarantees empty output (even three out of four would be enough but that's more expensive to check)
			if ((rowmap[row] & rowmap[row + 1] & rowmap[row + resolution] & rowmap[row + resolution + 1]) == ~0u)
				continue;

			const unsigned char* data = grid + size_t(resolution) * row;

			// we track each slice as a 8-bit code (matching cube indexing) as we iterate through the row to avoid extra lookups
			int last = (data[0] != 0) | ((data[resolution] != 0) << 2) | ((data[slice] != 0) << 4) | ((data[slice + resolution] != 0) << 6);

			for (int x = 0; x < resolution - 1; ++x)
			{
				int next = (data[x + 1] != 0) | ((data[x + 1 + resolution] != 0) << 2) | ((data[x + 1 + slice] != 0) << 4) | ((data[x + 1 + slice + resolution] != 0) << 6);

				// next slice bits are in position for the next iteration so we need to shift them to retrieve current cube code
				int cube = last | (next << 1);
				last = next;

				if (cube == 0 || cube == 0xff)
					continue;

				if (!destination)
				{
					// fast path: we can statically determine the upper bound on triangles the loop below will output based on cube configuration
					result += kTriangleCount[cube];
					continue;
				}

				// deciders are only evaluated for cells with an alternate configuration to reduce overhead
				int alt = 0;

				if (kTriangleAlt[cube] == 1)
					alt = octantDecider(x, y, z, cube, grid, rowmap, voxels, resolution);
				else if (kTriangleAlt[cube] == 2 && (options & meshopt_RemeshSolve))
					alt = quadricDecider(x, y, z, cube, grid, rowmap, voxels, resolution, rscale);

				const unsigned short* tris = kTriangleTable[cube][alt];

				for (int i = 0; tris[i]; ++i)
				{
					// note: we only emit the triangle if we have space for it, but we count it regardless for consistent capacity estimation
					if (result < max_triangle_count)
					{
						unsigned short tri = tris[i];

						emitVertex(&destination[result * 9 + 0], x, y, z, (tri >> 8) & 0xf, grid, rowmap, voxels, resolution, offset);
						emitVertex(&destination[result * 9 + 3], x, y, z, (tri >> 4) & 0xf, grid, rowmap, voxels, resolution, offset);
						emitVertex(&destination[result * 9 + 6], x, y, z, (tri >> 0) & 0xf, grid, rowmap, voxels, resolution, offset);
					}

					result++;
				}
			}
		}

	return result;
}

} // namespace meshopt

size_t meshopt_remesh(float* destination, size_t max_triangle_count, const unsigned int* indices, size_t index_count, const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, int resolution, unsigned int options)
{
	using namespace meshopt;

	assert(destination || max_triangle_count == 0);
	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(float) == 0);
	assert(resolution >= 4 && resolution <= 256);

	meshopt_Allocator allocator;

	// measure voxel grid to compute position => voxel mapping
	float offset[3] = {};
	float scale = measureGrid(vertex_positions, vertex_count, vertex_positions_stride, resolution, offset);

	// rasterize triangles into the voxel grid: in the first pass, this simply tags occupied voxels
	unsigned char* grid = allocator.allocate<unsigned char>(size_t(resolution) * size_t(resolution) * size_t(resolution));
	memset(grid, 0, size_t(resolution) * size_t(resolution) * size_t(resolution));

	voxelize(grid, NULL, NULL, indices, index_count, vertex_positions, vertex_count, vertex_positions_stride, resolution, scale, offset, options);

	// compute row offsets for occupied voxels; also stores ~0u for empty rows to accelerate further processing
	unsigned int* rowmap = allocator.allocate<unsigned int>(size_t(resolution) * size_t(resolution));
	size_t voxel_count = rowpack(grid, rowmap, resolution);

#if TRACE
	printf("remesher: %zu voxels occupied\n", voxel_count);
#endif

	// allocate additional voxel data for each occupied voxel; this can be filled in the second pass to compute positions
	// note that we only do this if we need to compute output triangles; counting runs skip it for performance
	Voxel* voxels = NULL;

	if (destination)
	{
		voxels = allocator.allocate<Voxel>(voxel_count);
		memset(voxels, 0, voxel_count * sizeof(Voxel));
	}

	// fill in empty voxels that are not reachable from the grid boundary; the inside empty voxels are marked with 0xff
	// note that these voxels do *not* have the associated Voxel data as they were never voxelized
	if ((options & meshopt_RemeshShell) == 0)
	{
		unsigned int* worklist = allocator.allocate<unsigned int>(size_t(resolution) * size_t(resolution));
		unsigned char* queued = allocator.allocate<unsigned char>(size_t(resolution) * size_t(resolution));

		solidify(grid, rowmap, worklist, queued, resolution);

#if TRACE
		size_t inside_count = 0, occupied_count = 0;
		for (size_t i = 0; i < size_t(resolution) * size_t(resolution) * size_t(resolution); ++i)
		{
			inside_count += (grid[i] == 0xff);
			occupied_count += (grid[i] != 0 && grid[i] != 0xff);
		}

		printf("remesher: %zu voxels occupied, %zu voxels inside\n", occupied_count, inside_count);
#endif
	}

	// accumulate voxel positions: in the second pass, this computes enough data in each voxel to calculate positions
	if (voxels)
		voxelize(grid, rowmap, voxels, indices, index_count, vertex_positions, vertex_count, vertex_positions_stride, resolution, scale, offset, options);

	// compute final voxel positions; each voxel has a single resulting position that will be emitted during polygonization
	if (voxels)
		solve(voxels, voxel_count, scale, options);

	// output triangles from the voxel grid; if destination is NULL, this still counts the number of triangles that would be generated
	size_t result = polygonize(destination, max_triangle_count, grid, rowmap, voxels, resolution, scale, offset, options);

#if TRACE
	printf("remesher: %zu triangles (%zu capacity)\n", result, max_triangle_count);
#endif

	return result;
}
