/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "register_types.h"
#include "thirdparty/thekla_atlas/thekla/thekla_atlas.h"
#include <stdio.h>
#include <stdlib.h>
extern bool (*array_mesh_lightmap_unwrap_callback)(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, const int *p_face_materials, int p_index_count, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y);

bool thekla_mesh_lightmap_unwrap_callback(float p_texel_size, const float *p_vertices, const float *p_normals, int p_vertex_count, const int *p_indices, const int *p_face_materials, int p_index_count, float **r_uv, int **r_vertex, int *r_vertex_count, int **r_index, int *r_index_count, int *r_size_hint_x, int *r_size_hint_y) {

	//set up input mesh
	Thekla::Atlas_Input_Mesh input_mesh;
	input_mesh.face_array = new Thekla::Atlas_Input_Face[p_index_count / 3];
	for (int i = 0; i < p_index_count / 3; i++) {
		input_mesh.face_array[i].vertex_index[0] = p_indices[i * 3 + 0];
		input_mesh.face_array[i].vertex_index[1] = p_indices[i * 3 + 1];
		input_mesh.face_array[i].vertex_index[2] = p_indices[i * 3 + 2];
		//printf("face %i - %i, %i, %i - mat %i\n", i, input_mesh.face_array[i].vertex_index[0], input_mesh.face_array[i].vertex_index[1], input_mesh.face_array[i].vertex_index[2], p_face_materials[i]);
		input_mesh.face_array[i].material_index = p_face_materials[i];
	}
	input_mesh.vertex_array = new Thekla::Atlas_Input_Vertex[p_vertex_count];
	for (int i = 0; i < p_vertex_count; i++) {
		input_mesh.vertex_array[i].first_colocal = i; //wtf
		for (int j = 0; j < 3; j++) {
			input_mesh.vertex_array[i].position[j] = p_vertices[i * 3 + j];
			input_mesh.vertex_array[i].normal[j] = p_normals[i * 3 + j];
		}
		input_mesh.vertex_array[i].uv[0] = 0;
		input_mesh.vertex_array[i].uv[1] = 0;
		//printf("vertex %i - %f, %f, %f\n", i, input_mesh.vertex_array[i].position[0], input_mesh.vertex_array[i].position[1], input_mesh.vertex_array[i].position[2]);
		//printf("normal %i - %f, %f, %f\n", i, input_mesh.vertex_array[i].normal[0], input_mesh.vertex_array[i].normal[1], input_mesh.vertex_array[i].normal[2]);
	}
	input_mesh.face_count = p_index_count / 3;
	input_mesh.vertex_count = p_vertex_count;

	//set up options
	Thekla::Atlas_Options options;
	Thekla::atlas_set_default_options(&options);
	options.packer_options.witness.packing_quality = 1;
	options.packer_options.witness.texel_area = 1.0 / p_texel_size;
	options.packer_options.witness.conservative = false;

	//generate
	Thekla::Atlas_Error err;
	Thekla::Atlas_Output_Mesh *output = atlas_generate(&input_mesh, &options, &err);

	delete[] input_mesh.face_array;
	delete[] input_mesh.vertex_array;

	if (err != Thekla::Atlas_Error_Success) {
		printf("error with atlas\n");
	} else {
		*r_vertex = (int *)malloc(sizeof(int) * output->vertex_count);
		*r_uv = (float *)malloc(sizeof(float) * output->vertex_count * 3);
		*r_index = (int *)malloc(sizeof(int) * output->index_count);

		//		printf("w: %i, h: %i\n", output->atlas_width, output->atlas_height);
		for (int i = 0; i < output->vertex_count; i++) {
			(*r_vertex)[i] = output->vertex_array[i].xref;
			(*r_uv)[i * 2 + 0] = output->vertex_array[i].uv[0] / output->atlas_width;
			(*r_uv)[i * 2 + 1] = output->vertex_array[i].uv[1] / output->atlas_height;
			//			printf("uv: %f,%f\n", (*r_uv)[i * 2 + 0], (*r_uv)[i * 2 + 1]);
		}
		*r_vertex_count = output->vertex_count;

		for (int i = 0; i < output->index_count; i++) {
			(*r_index)[i] = output->index_array[i];
		}

		*r_index_count = output->index_count;

		*r_size_hint_x = output->atlas_height;
		*r_size_hint_y = output->atlas_width;
	}

	if (output) {
		atlas_free(output);
	}

	return err == Thekla::Atlas_Error_Success;
}

void register_thekla_unwrap_types() {

	array_mesh_lightmap_unwrap_callback = thekla_mesh_lightmap_unwrap_callback;
}

void unregister_thekla_unwrap_types() {
}
