/**************************************************************************/
/*  register_types.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "register_types.h"

#include "core/crypto/crypto_core.h"

#include <xatlas.h>

extern bool (*array_mesh_lightmap_unwrap_callback)(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, int p_index_count, const uint8_t *p_cache_data, bool *r_use_cache, uint8_t **r_mesh_cache, int *r_mesh_cache_size, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y);

bool xatlas_mesh_lightmap_unwrap_callback(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, int p_index_count, const uint8_t *p_cache_data, bool *r_use_cache, uint8_t **r_mesh_cache, int *r_mesh_cache_size, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y) {
	CryptoCore::MD5Context ctx;
	ctx.start();

	ctx.update((unsigned char *)&p_texel_size, sizeof(float));
	ctx.update((unsigned char *)p_indices, sizeof(int) * p_index_count);
	ctx.update((unsigned char *)p_vertices, sizeof(float) * p_vertex_count * 3);
	ctx.update((unsigned char *)p_normals, sizeof(float) * p_vertex_count * 3);

	unsigned char hash[16];
	ctx.finish(hash);

	bool cached = false;
	unsigned int cache_idx = 0;

	*r_mesh_cache = nullptr;
	*r_mesh_cache_size = 0;

	if (p_cache_data) {
		//Check if hash is in cache data
		int *cache_data = (int *)p_cache_data;
		int n_entries = cache_data[0];
		unsigned int read_idx = 1;
		for (int i = 0; i < n_entries; ++i) {
			if (memcmp(&cache_data[read_idx], hash, 16) == 0) {
				cached = true;
				cache_idx = read_idx;
				break;
			}

			read_idx += 4; // hash
			read_idx += 2; // size hint

			int vertex_count = cache_data[read_idx];
			read_idx += 1; // vertex count
			read_idx += vertex_count; // vertex
			read_idx += vertex_count * 2; // uvs

			int index_count = cache_data[read_idx];
			read_idx += 1; // index count
			read_idx += index_count; // indices
		}
	}

	if (cached) {
		int *cache_data = (int *)p_cache_data;

		cache_idx += 4;

		// Load size
		*r_size_hint_x = cache_data[cache_idx];
		*r_size_hint_y = cache_data[cache_idx + 1];
		cache_idx += 2;

		// Load vertices
		*r_vertex_count = cache_data[cache_idx];
		cache_idx++;
		*r_vertex = &cache_data[cache_idx];
		cache_idx += *r_vertex_count;

		// Load UVs
		*r_uv = (float *)&cache_data[cache_idx];
		cache_idx += *r_vertex_count * 2;

		// Load indices
		*r_index_count = cache_data[cache_idx];
		cache_idx++;
		*r_index = &cache_data[cache_idx];
	} else {
		// set up input mesh
		xatlas::MeshDecl input_mesh;
		input_mesh.indexData = p_indices;
		input_mesh.indexCount = p_index_count;
		input_mesh.indexFormat = xatlas::IndexFormat::UInt32;

		input_mesh.vertexCount = p_vertex_count;
		input_mesh.vertexPositionData = p_vertices;
		input_mesh.vertexPositionStride = sizeof(float) * 3;
		input_mesh.vertexNormalData = p_normals;
		input_mesh.vertexNormalStride = sizeof(uint32_t) * 3;
		input_mesh.vertexUvData = nullptr;
		input_mesh.vertexUvStride = 0;

		xatlas::ChartOptions chart_options;
		chart_options.fixWinding = true;

		ERR_FAIL_COND_V_MSG(p_texel_size <= 0.0f, false, "Texel size must be greater than 0.");

		xatlas::PackOptions pack_options;
		pack_options.padding = 1;
		pack_options.maxChartSize = 4094; // Lightmap atlassing needs 2 for padding between meshes, so 4096-2
		pack_options.blockAlign = true;
		pack_options.texelsPerUnit = 1.0 / p_texel_size;

		xatlas::Atlas *atlas = xatlas::Create();

		xatlas::AddMeshError err = xatlas::AddMesh(atlas, input_mesh, 1);
		ERR_FAIL_COND_V_MSG(err != xatlas::AddMeshError::Success, false, xatlas::StringForEnum(err));

		xatlas::Generate(atlas, chart_options, pack_options);

		*r_size_hint_x = atlas->width;
		*r_size_hint_y = atlas->height;

		float w = *r_size_hint_x;
		float h = *r_size_hint_y;

		if (w == 0 || h == 0) {
			xatlas::Destroy(atlas);
			return false; //could not bake because there is no area
		}

		const xatlas::Mesh &output = atlas->meshes[0];

		*r_vertex = (int *)memalloc(sizeof(int) * output.vertexCount);
		ERR_FAIL_NULL_V_MSG(*r_vertex, false, "Out of memory.");
		*r_uv = (float *)memalloc(sizeof(float) * output.vertexCount * 2);
		ERR_FAIL_NULL_V_MSG(*r_uv, false, "Out of memory.");
		*r_index = (int *)memalloc(sizeof(int) * output.indexCount);
		ERR_FAIL_NULL_V_MSG(*r_index, false, "Out of memory.");

		float max_x = 0;
		float max_y = 0;
		for (uint32_t i = 0; i < output.vertexCount; i++) {
			(*r_vertex)[i] = output.vertexArray[i].xref;
			(*r_uv)[i * 2 + 0] = output.vertexArray[i].uv[0] / w;
			(*r_uv)[i * 2 + 1] = output.vertexArray[i].uv[1] / h;
			max_x = MAX(max_x, output.vertexArray[i].uv[0]);
			max_y = MAX(max_y, output.vertexArray[i].uv[1]);
		}

		*r_vertex_count = output.vertexCount;

		for (uint32_t i = 0; i < output.indexCount; i++) {
			(*r_index)[i] = output.indexArray[i];
		}

		*r_index_count = output.indexCount;

		xatlas::Destroy(atlas);
	}

	if (*r_use_cache) {
		// Build cache data for current mesh

		unsigned int new_cache_size = 4 + 2 + 1 + *r_vertex_count + (*r_vertex_count * 2) + 1 + *r_index_count; // hash + size hint + vertex_count + vertices + uvs + index_count + indices
		new_cache_size *= sizeof(int);
		int *new_cache_data = (int *)memalloc(new_cache_size);
		unsigned int new_cache_idx = 0;

		// hash
		memcpy(&new_cache_data[new_cache_idx], hash, 16);
		new_cache_idx += 4;

		// size hint
		new_cache_data[new_cache_idx] = *r_size_hint_x;
		new_cache_data[new_cache_idx + 1] = *r_size_hint_y;
		new_cache_idx += 2;

		// vertex count
		new_cache_data[new_cache_idx] = *r_vertex_count;
		new_cache_idx++;

		// vertices
		memcpy(&new_cache_data[new_cache_idx], *r_vertex, sizeof(int) * (*r_vertex_count));
		new_cache_idx += *r_vertex_count;

		// uvs
		memcpy(&new_cache_data[new_cache_idx], *r_uv, sizeof(float) * (*r_vertex_count) * 2);
		new_cache_idx += *r_vertex_count * 2;

		// index count
		new_cache_data[new_cache_idx] = *r_index_count;
		new_cache_idx++;

		// indices
		memcpy(&new_cache_data[new_cache_idx], *r_index, sizeof(int) * (*r_index_count));

		// Return cache data to the caller
		*r_mesh_cache = (uint8_t *)new_cache_data;
		*r_mesh_cache_size = new_cache_size;
	}

	*r_use_cache = cached; // Return whether cache was used.

	return true;
}

void initialize_xatlas_unwrap_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	array_mesh_lightmap_unwrap_callback = xatlas_mesh_lightmap_unwrap_callback;
}

void uninitialize_xatlas_unwrap_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
