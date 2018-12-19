// This code is in the public domain -- castanyo@yahoo.es
#pragma once
#ifndef XATLAS_H
#define XATLAS_H
#include <float.h> // FLT_MAX
#include <limits.h>
#include <stdint.h>
namespace xatlas {

typedef void (*PrintFunc)(const char *, ...);

struct Atlas;

struct CharterOptions {
	float proxyFitMetricWeight;
	float roundnessMetricWeight;
	float straightnessMetricWeight;
	float normalSeamMetricWeight;
	float textureSeamMetricWeight;
	float maxChartArea;
	float maxBoundaryLength;

	CharterOptions() {
		// These are the default values we use on The Witness.
		proxyFitMetricWeight = 2.0f;
		roundnessMetricWeight = 0.01f;
		straightnessMetricWeight = 6.0f;
		normalSeamMetricWeight = 4.0f;
		textureSeamMetricWeight = 0.5f;
		/*
		proxyFitMetricWeight = 1.0f;
		roundnessMetricWeight = 0.1f;
		straightnessMetricWeight = 0.25f;
		normalSeamMetricWeight = 1.0f;
		textureSeamMetricWeight = 0.1f;
		*/
		maxChartArea = FLT_MAX;
		maxBoundaryLength = FLT_MAX;
	}
};

struct PackMethod {
	enum Enum {
		TexelArea, // texel_area determines resolution
		ApproximateResolution, // guess texel_area to approximately match desired resolution
		ExactResolution // run the packer multiple times to exactly match the desired resolution (slow)
	};
};

struct PackerOptions {
	PackMethod::Enum method;

	// 0 - brute force
	// 1 - 4096 attempts
	// 2 - 2048
	// 3 - 1024
	// 4 - 512
	// other - 256
	// Avoid brute force packing, since it can be unusably slow in some situations.
	int quality;

	float texelArea; // This is not really texel area, but 1 / texel width?
	uint32_t resolution;
	bool blockAlign; // Align charts to 4x4 blocks.
	bool conservative; // Pack charts with extra padding.
	int padding;

	PackerOptions() {
		method = PackMethod::ApproximateResolution;
		quality = 1;
		texelArea = 8;
		resolution = 512;
		blockAlign = false;
		conservative = false;
		padding = 0;
	}
};

struct AddMeshErrorCode {
	enum Enum {
		Success,
		AlreadyAddedEdge, // index0 and index1 are the edge indices
		DegenerateColocalEdge, // index0 and index1 are the edge indices
		DegenerateEdge, // index0 and index1 are the edge indices
		DuplicateEdge, // index0 and index1 are the edge indices
		IndexOutOfRange, // index0 is the index
		InvalidIndexCount, // not evenly divisible by 3 - expecting triangles
		ZeroAreaFace,
		ZeroLengthEdge // index0 and index1 are the edge indices
	};
};

struct AddMeshError {
	AddMeshErrorCode::Enum code;
	uint32_t face;
	uint32_t index0, index1;
};

struct IndexFormat {
	enum Enum {
		HalfFloat,
		Float
	};
};

struct InputMesh {
	uint32_t vertexCount;
	const void *vertexPositionData;
	uint32_t vertexPositionStride;
	const void *vertexNormalData; // optional
	uint32_t vertexNormalStride; // optional

	// optional
	// The input UVs are provided as a hint to the chart generator.
	const void *vertexUvData;
	uint32_t vertexUvStride;

	uint32_t indexCount;
	const void *indexData;
	IndexFormat::Enum indexFormat;

	// optional. indexCount / 3 in length.
	// Charter also uses material boundaries as a hint to cut charts.
	const uint16_t *faceMaterialData;
};

struct OutputChart {
	uint32_t *indexArray;
	uint32_t indexCount;
};

struct OutputVertex {
	float uv[2];
	uint32_t xref; // Index of input vertex from which this output vertex originated.
};

struct OutputMesh {
	OutputChart *chartArray;
	uint32_t chartCount;
	uint32_t *indexArray;
	uint32_t indexCount;
	OutputVertex *vertexArray;
	uint32_t vertexCount;
};

void SetPrint(PrintFunc print);
Atlas *Create();
void Destroy(Atlas *atlas);
// useColocalVertices - generates fewer charts (good), but is more sensitive to bad geometry.
AddMeshError AddMesh(Atlas *atlas, const InputMesh &mesh, bool useColocalVertices = true);
void Generate(Atlas *atlas, CharterOptions charterOptions = CharterOptions(), PackerOptions packerOptions = PackerOptions());
uint32_t GetWidth(const Atlas *atlas);
uint32_t GetHeight(const Atlas *atlas);
uint32_t GetNumCharts(const Atlas *atlas);
const OutputMesh *const *GetOutputMeshes(const Atlas *atlas);
const char *StringForEnum(AddMeshErrorCode::Enum error);

} // namespace xatlas

#endif // XATLAS_H
