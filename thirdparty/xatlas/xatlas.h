/*
MIT License

Copyright (c) 2018-2020 Jonathan Young

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/*
thekla_atlas
MIT License
https://github.com/Thekla/thekla_atlas
Copyright (c) 2013 Thekla, Inc
Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>
*/
#pragma once
#ifndef XATLAS_H
#define XATLAS_H
#include <stddef.h>
#include <stdint.h>

namespace xatlas {

enum class ChartType {
	Planar,
	Ortho,
	LSCM,
	Piecewise,
	Invalid
};

// A group of connected faces, belonging to a single atlas.
struct Chart {
	uint32_t *faceArray;
	uint32_t atlasIndex; // Sub-atlas index.
	uint32_t faceCount;
	ChartType type;
	uint32_t material;
};

// Output vertex.
struct Vertex {
	int32_t atlasIndex; // Sub-atlas index. -1 if the vertex doesn't exist in any atlas.
	int32_t chartIndex; // -1 if the vertex doesn't exist in any chart.
	float uv[2]; // Not normalized - values are in Atlas width and height range.
	uint32_t xref; // Index of input vertex from which this output vertex originated.
};

// Output mesh.
struct Mesh {
	Chart *chartArray;
	uint32_t *indexArray;
	Vertex *vertexArray;
	uint32_t chartCount;
	uint32_t indexCount;
	uint32_t vertexCount;
};

static const uint32_t kImageChartIndexMask = 0x1FFFFFFF;
static const uint32_t kImageHasChartIndexBit = 0x80000000;
static const uint32_t kImageIsBilinearBit = 0x40000000;
static const uint32_t kImageIsPaddingBit = 0x20000000;

// Empty on creation. Populated after charts are packed.
struct Atlas {
	uint32_t *image;
	Mesh *meshes; // The output meshes, corresponding to each AddMesh call.
	float *utilization; // Normalized atlas texel utilization array. E.g. a value of 0.8 means 20% empty space. atlasCount in length.
	uint32_t width; // Atlas width in texels.
	uint32_t height; // Atlas height in texels.
	uint32_t atlasCount; // Number of sub-atlases. Equal to 0 unless PackOptions resolution is changed from default (0).
	uint32_t chartCount; // Total number of charts in all meshes.
	uint32_t meshCount; // Number of output meshes. Equal to the number of times AddMesh was called.
	float texelsPerUnit; // Equal to PackOptions texelsPerUnit if texelsPerUnit > 0, otherwise an estimated value to match PackOptions resolution.
};

// Create an empty atlas.
Atlas *Create();

void Destroy(Atlas *atlas);

enum class IndexFormat {
	UInt16,
	UInt32
};

// Input mesh declaration.
struct MeshDecl {
	const void *vertexPositionData = nullptr;
	const void *vertexNormalData = nullptr; // optional
	const void *vertexUvData = nullptr; // optional. The input UVs are provided as a hint to the chart generator.
	const void *indexData = nullptr; // optional

	// Optional. Must be faceCount in length.
	// Don't atlas faces set to true. Ignored faces still exist in the output meshes, Vertex uv is set to (0, 0) and Vertex atlasIndex to -1.
	const bool *faceIgnoreData = nullptr;

	// Optional. Must be faceCount in length.
	// Only faces with the same material will be assigned to the same chart.
	const uint32_t *faceMaterialData = nullptr;

	// Optional. Must be faceCount in length.
	// Polygon / n-gon support. Faces are assumed to be triangles if this is null.
	const uint8_t *faceVertexCount = nullptr;

	uint32_t vertexCount = 0;
	uint32_t vertexPositionStride = 0;
	uint32_t vertexNormalStride = 0; // optional
	uint32_t vertexUvStride = 0; // optional
	uint32_t indexCount = 0;
	int32_t indexOffset = 0; // optional. Add this offset to all indices.
	uint32_t faceCount = 0; // Optional if faceVertexCount is null. Otherwise assumed to be indexCount / 3.
	IndexFormat indexFormat = IndexFormat::UInt16;

	// Vertex positions within epsilon distance of each other are considered colocal.
	float epsilon = 1.192092896e-07F;
};

enum class AddMeshError {
	Success, // No error.
	Error, // Unspecified error.
	IndexOutOfRange, // An index is >= MeshDecl vertexCount.
	InvalidFaceVertexCount, // Must be >= 3.
	InvalidIndexCount // Not evenly divisible by 3 - expecting triangles.
};

// Add a mesh to the atlas. MeshDecl data is copied, so it can be freed after AddMesh returns.
AddMeshError AddMesh(Atlas *atlas, const MeshDecl &meshDecl, uint32_t meshCountHint = 0);

// Wait for AddMesh async processing to finish. ComputeCharts / Generate call this internally.
void AddMeshJoin(Atlas *atlas);

struct UvMeshDecl {
	const void *vertexUvData = nullptr;
	const void *indexData = nullptr; // optional
	const uint32_t *faceMaterialData = nullptr; // Optional. Overlapping UVs should be assigned a different material. Must be indexCount / 3 in length.
	uint32_t vertexCount = 0;
	uint32_t vertexStride = 0;
	uint32_t indexCount = 0;
	int32_t indexOffset = 0; // optional. Add this offset to all indices.
	IndexFormat indexFormat = IndexFormat::UInt16;
};

AddMeshError AddUvMesh(Atlas *atlas, const UvMeshDecl &decl);

// Custom parameterization function. texcoords initial values are an orthogonal parameterization.
typedef void (*ParameterizeFunc)(const float *positions, float *texcoords, uint32_t vertexCount, const uint32_t *indices, uint32_t indexCount);

struct ChartOptions {
	ParameterizeFunc paramFunc = nullptr;

	float maxChartArea = 0.0f; // Don't grow charts to be larger than this. 0 means no limit.
	float maxBoundaryLength = 0.0f; // Don't grow charts to have a longer boundary than this. 0 means no limit.

	// Weights determine chart growth. Higher weights mean higher cost for that metric.
	float normalDeviationWeight = 2.0f; // Angle between face and average chart normal.
	float roundnessWeight = 0.01f;
	float straightnessWeight = 6.0f;
	float normalSeamWeight = 4.0f; // If > 1000, normal seams are fully respected.
	float textureSeamWeight = 0.5f;

	float maxCost = 2.0f; // If total of all metrics * weights > maxCost, don't grow chart. Lower values result in more charts.
	uint32_t maxIterations = 1; // Number of iterations of the chart growing and seeding phases. Higher values result in better charts.

	bool useInputMeshUvs = false; // Use MeshDecl::vertexUvData for charts.
	bool fixWinding = false; // Enforce consistent texture coordinate winding.
};

// Call after all AddMesh calls. Can be called multiple times to recompute charts with different options.
void ComputeCharts(Atlas *atlas, ChartOptions options = ChartOptions());

struct PackOptions {
	// Charts larger than this will be scaled down. 0 means no limit.
	uint32_t maxChartSize = 0;

	// Number of pixels to pad charts with.
	uint32_t padding = 0;

	// Unit to texel scale. e.g. a 1x1 quad with texelsPerUnit of 32 will take up approximately 32x32 texels in the atlas.
	// If 0, an estimated value will be calculated to approximately match the given resolution.
	// If resolution is also 0, the estimated value will approximately match a 1024x1024 atlas.
	float texelsPerUnit = 0.0f;

	// If 0, generate a single atlas with texelsPerUnit determining the final resolution.
	// If not 0, and texelsPerUnit is not 0, generate one or more atlases with that exact resolution.
	// If not 0, and texelsPerUnit is 0, texelsPerUnit is estimated to approximately match the resolution.
	uint32_t resolution = 0;

	// Leave space around charts for texels that would be sampled by bilinear filtering.
	bool bilinear = true;

	// Align charts to 4x4 blocks. Also improves packing speed, since there are fewer possible chart locations to consider.
	bool blockAlign = false;

	// Slower, but gives the best result. If false, use random chart placement.
	bool bruteForce = false;

	// Create Atlas::image
	bool createImage = false;

	// Rotate charts to the axis of their convex hull.
	bool rotateChartsToAxis = true;

	// Rotate charts to improve packing.
	bool rotateCharts = true;
};

// Call after ComputeCharts. Can be called multiple times to re-pack charts with different options.
void PackCharts(Atlas *atlas, PackOptions packOptions = PackOptions());

// Equivalent to calling ComputeCharts and PackCharts in sequence. Can be called multiple times to regenerate with different options.
void Generate(Atlas *atlas, ChartOptions chartOptions = ChartOptions(), PackOptions packOptions = PackOptions());

// Progress tracking.
enum class ProgressCategory {
	AddMesh,
	ComputeCharts,
	PackCharts,
	BuildOutputMeshes
};

// May be called from any thread. Return false to cancel.
typedef bool (*ProgressFunc)(ProgressCategory category, int progress, void *userData);

void SetProgressCallback(Atlas *atlas, ProgressFunc progressFunc = nullptr, void *progressUserData = nullptr);

// Custom memory allocation.
typedef void *(*ReallocFunc)(void *, size_t);
typedef void (*FreeFunc)(void *);
void SetAlloc(ReallocFunc reallocFunc, FreeFunc freeFunc = nullptr);

// Custom print function.
typedef int (*PrintFunc)(const char *, ...);
void SetPrint(PrintFunc print, bool verbose);

// Helper functions for error messages.
const char *StringForEnum(AddMeshError error);
const char *StringForEnum(ProgressCategory category);

} // namespace xatlas

#endif // XATLAS_H
