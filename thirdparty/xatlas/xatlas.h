/*
MIT License

Copyright (c) 2018-2019 Jonathan Young

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
#include <stdint.h>

namespace xatlas {

struct ChartType
{
	enum Enum
	{
		Planar,
		Ortho,
		LSCM,
		Piecewise
	};
};

// A group of connected faces, belonging to a single atlas.
struct Chart
{
	uint32_t atlasIndex; // Sub-atlas index.
	uint32_t *faceArray;
	uint32_t faceCount;
	uint32_t material;
	ChartType::Enum type;
};

// Output vertex.
struct Vertex
{
	int32_t atlasIndex; // Sub-atlas index. -1 if the vertex doesn't exist in any atlas.
	int32_t chartIndex; // -1 if the vertex doesn't exist in any chart.
	float uv[2]; // Not normalized - values are in Atlas width and height range.
	uint32_t xref; // Index of input vertex from which this output vertex originated.
};

// Output mesh.
struct Mesh
{
	Chart *chartArray;
	uint32_t chartCount;
	uint32_t *indexArray;
	uint32_t indexCount;
	Vertex *vertexArray;
	uint32_t vertexCount;
};

static const uint32_t kImageChartIndexMask = 0x1FFFFFFF;
static const uint32_t kImageHasChartIndexBit = 0x80000000;
static const uint32_t kImageIsBilinearBit = 0x40000000;
static const uint32_t kImageIsPaddingBit = 0x20000000;

// Empty on creation. Populated after charts are packed.
struct Atlas
{
	uint32_t width; // Atlas width in texels.
	uint32_t height; // Atlas height in texels.
	uint32_t atlasCount; // Number of sub-atlases. Equal to 0 unless PackOptions resolution is changed from default (0).
	uint32_t chartCount; // Total number of charts in all meshes.
	uint32_t meshCount; // Number of output meshes. Equal to the number of times AddMesh was called.
	Mesh *meshes; // The output meshes, corresponding to each AddMesh call.
	float *utilization; // Normalized atlas texel utilization array. E.g. a value of 0.8 means 20% empty space. atlasCount in length.
	float texelsPerUnit; // Equal to PackOptions texelsPerUnit if texelsPerUnit > 0, otherwise an estimated value to match PackOptions resolution.
	uint32_t *image;
};

// Create an empty atlas.
Atlas *Create();

void Destroy(Atlas *atlas);

struct IndexFormat
{
	enum Enum
	{
		UInt16,
		UInt32
	};
};

// Input mesh declaration.
struct MeshDecl
{
	uint32_t vertexCount = 0;
	const void *vertexPositionData = nullptr;
	uint32_t vertexPositionStride = 0;
	const void *vertexNormalData = nullptr; // optional
	uint32_t vertexNormalStride = 0; // optional
	const void *vertexUvData = nullptr; // optional. The input UVs are provided as a hint to the chart generator.
	uint32_t vertexUvStride = 0; // optional
	uint32_t indexCount = 0;
	const void *indexData = nullptr; // optional
	int32_t indexOffset = 0; // optional. Add this offset to all indices.
	IndexFormat::Enum indexFormat = IndexFormat::UInt16;
	
	// Optional. indexCount / 3 (triangle count) in length.
	// Don't atlas faces set to true. Ignored faces still exist in the output meshes, Vertex uv is set to (0, 0) and Vertex atlasIndex to -1.
	const bool *faceIgnoreData = nullptr;

	// Vertex positions within epsilon distance of each other are considered colocal.
	float epsilon = 1.192092896e-07F;
};

struct AddMeshError
{
	enum Enum
	{
		Success, // No error.
		Error, // Unspecified error.
		IndexOutOfRange, // An index is >= MeshDecl vertexCount.
		InvalidIndexCount // Not evenly divisible by 3 - expecting triangles.
	};
};

// Add a mesh to the atlas. MeshDecl data is copied, so it can be freed after AddMesh returns.
AddMeshError::Enum AddMesh(Atlas *atlas, const MeshDecl &meshDecl, uint32_t meshCountHint = 0);

// Wait for AddMesh async processing to finish. ComputeCharts / Generate call this internally.
void AddMeshJoin(Atlas *atlas);

struct UvMeshDecl
{
	uint32_t vertexCount = 0;
	uint32_t vertexStride = 0;
	const void *vertexUvData = nullptr;
	uint32_t indexCount = 0;
	const void *indexData = nullptr; // optional
	int32_t indexOffset = 0; // optional. Add this offset to all indices.
	IndexFormat::Enum indexFormat = IndexFormat::UInt16;
	const uint32_t *faceMaterialData = nullptr; // Optional. Faces with different materials won't be assigned to the same chart. Must be indexCount / 3 in length.
	bool rotateCharts = true;
};

AddMeshError::Enum AddUvMesh(Atlas *atlas, const UvMeshDecl &decl);

struct ChartOptions
{
	float maxChartArea = 0.0f; // Don't grow charts to be larger than this. 0 means no limit.
	float maxBoundaryLength = 0.0f; // Don't grow charts to have a longer boundary than this. 0 means no limit.

	// Weights determine chart growth. Higher weights mean higher cost for that metric.
	float proxyFitMetricWeight = 2.0f; // Angle between face and average chart normal.
	float roundnessMetricWeight = 0.01f;
	float straightnessMetricWeight = 6.0f;
	float normalSeamMetricWeight = 4.0f; // If > 1000, normal seams are fully respected.
	float textureSeamMetricWeight = 0.5f;

	float maxThreshold = 2.0f; // If total of all metrics * weights > maxThreshold, don't grow chart. Lower values result in more charts.
	uint32_t maxIterations = 1; // Number of iterations of the chart growing and seeding phases. Higher values result in better charts.
};

// Call after all AddMesh calls. Can be called multiple times to recompute charts with different options.
void ComputeCharts(Atlas *atlas, ChartOptions chartOptions = ChartOptions());

// Custom parameterization function. texcoords initial values are an orthogonal parameterization.
typedef void (*ParameterizeFunc)(const float *positions, float *texcoords, uint32_t vertexCount, const uint32_t *indices, uint32_t indexCount);

// Call after ComputeCharts. Can be called multiple times to re-parameterize charts with a different ParameterizeFunc.
void ParameterizeCharts(Atlas *atlas, ParameterizeFunc func = nullptr);

struct PackOptions
{
	// Leave space around charts for texels that would be sampled by bilinear filtering.
	bool bilinear = true;

	// Align charts to 4x4 blocks. Also improves packing speed, since there are fewer possible chart locations to consider.
	bool blockAlign = false;

	// Slower, but gives the best result. If false, use random chart placement.
	bool bruteForce = false;

	// Create Atlas::image
	bool createImage = false;

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
};

// Call after ParameterizeCharts. Can be called multiple times to re-pack charts with different options.
void PackCharts(Atlas *atlas, PackOptions packOptions = PackOptions());

// Equivalent to calling ComputeCharts, ParameterizeCharts and PackCharts in sequence. Can be called multiple times to regenerate with different options.
void Generate(Atlas *atlas, ChartOptions chartOptions = ChartOptions(), ParameterizeFunc paramFunc = nullptr, PackOptions packOptions = PackOptions());

// Progress tracking.
struct ProgressCategory
{
	enum Enum
	{
		AddMesh,
		ComputeCharts,
		ParameterizeCharts,
		PackCharts,
		BuildOutputMeshes
	};
};

// May be called from any thread. Return false to cancel.
typedef bool (*ProgressFunc)(ProgressCategory::Enum category, int progress, void *userData);

void SetProgressCallback(Atlas *atlas, ProgressFunc progressFunc = nullptr, void *progressUserData = nullptr);

// Custom memory allocation.
typedef void *(*ReallocFunc)(void *, size_t);
typedef void (*FreeFunc)(void *);
void SetAlloc(ReallocFunc reallocFunc, FreeFunc freeFunc = nullptr);

// Custom print function.
typedef int (*PrintFunc)(const char *, ...);
void SetPrint(PrintFunc print, bool verbose);

// Helper functions for error messages.
const char *StringForEnum(AddMeshError::Enum error);
const char *StringForEnum(ProgressCategory::Enum category);

} // namespace xatlas

#endif // XATLAS_H
