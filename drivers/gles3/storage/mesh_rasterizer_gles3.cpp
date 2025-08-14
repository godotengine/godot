/**************************************************************************/
/*  mesh_rasterizer_gles3.cpp                                             */
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

#include "mesh_rasterizer_gles3.h"

#include "drivers/gles3/storage/material_storage.h"
#include "drivers/gles3/storage/mesh_storage.h"
#include "drivers/gles3/storage/texture_storage.h"

namespace GLES3 {

static GLenum _primitive_type_to_render_primitive(RS::PrimitiveType p_primitive) {
	switch (p_primitive) {
		case RS::PRIMITIVE_POINTS:
			return GL_POINTS;
		case RS::PRIMITIVE_LINES:
			return GL_LINES;
		case RS::PRIMITIVE_LINE_STRIP:
			return GL_LINE_STRIP;
		case RS::PRIMITIVE_TRIANGLES:
			return GL_TRIANGLES;
		case RS::PRIMITIVE_TRIANGLE_STRIP:
			return GL_TRIANGLE_STRIP;
		default:
			return -1;
	}
}

void MeshRasterizerGLES3::texture_drawable_blit_mesh_advanced(RID p_texture_drawable, RID p_material, RID p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer) {
	MaterialStorage::get_singleton()->_update_global_shader_uniforms(); //must do before materials, so it can queue them for update
	MaterialStorage::get_singleton()->_update_queued_materials();

	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	MaterialData *md = material_storage->material_get_data(p_material, RS::SHADER_MESH_RASTERIZER);
	ERR_FAIL_NULL(md);
	MeshRasterizerMaterialData *material_data = static_cast<MeshRasterizerMaterialData *>(md);
	MeshRasterizerShaderData *shader_data = static_cast<MeshRasterizerShaderData *>(material_data->shader_data);
	ERR_FAIL_COND(!shader_data->valid);

	TextureStorage *texture_storage = TextureStorage::get_singleton();
	GLuint tex_id = texture_storage->texture_get_texid(p_texture_drawable);
	Vector3i size = texture_storage->texture_get_size(p_texture_drawable);

	MeshStorage *mesh_storage = MeshStorage::get_singleton();
	ERR_FAIL_COND(p_surface_index >= (uint32_t)mesh_storage->mesh_get_surface_count(p_mesh));

	void *surface = mesh_storage->mesh_get_surface(p_mesh, p_surface_index);
	GLenum primitive_gl = _primitive_type_to_render_primitive(mesh_storage->mesh_surface_get_primitive(surface));
	ERR_FAIL_COND(primitive_gl == (GLenum)-1);

	Texture::Type texture_type = texture_storage->texture_get_type(p_texture_drawable);
	ERR_FAIL_COND_MSG(texture_type == Texture::TYPE_3D, "Texture Drawable 3D is not supported.");

	GLuint index_buffer = mesh_storage->mesh_surface_get_index_buffer(surface, 0);
	GLuint vertex_array;
	uint64_t input_mask = shader_data->vertex_input_mask;
	mesh_storage->mesh_surface_get_vertex_arrays_and_format(surface, input_mask, vertex_array);

	bool success = material_storage->shaders.mesh_rasterizer_shader.version_bind_shader(shader_data->version, MeshRasterizerShaderGLES3::MODE_DEFAULT);
	ERR_FAIL_COND(!success);
	material_data->bind_uniforms();

	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	if (texture_type == Texture::TYPE_2D) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_id, 0);
	} else if (texture_type == Texture::TYPE_LAYERED) {
		glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_id, 0, p_layer);
	}
	glViewport(0, 0, size.x, size.y);

	glEnable(GL_BLEND);
	switch (p_blend_mode) {
		case RS::TEXTURE_DRAWABLE_BLEND_CLEAR:
			glDisable(GL_BLEND);
			glClearColor(p_clear_color.r, p_clear_color.g, p_clear_color.b, p_clear_color.a);
			glClear(GL_COLOR_BUFFER_BIT);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_ADD:
			glBlendEquation(GL_FUNC_ADD);
			glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
			break;

		case RS::TEXTURE_DRAWABLE_BLEND_SUB:
			glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
			glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
			break;

		case RS::TEXTURE_DRAWABLE_BLEND_MIX:
			glBlendEquation(GL_FUNC_ADD);
			glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			break;

		case RS::TEXTURE_DRAWABLE_BLEND_MUL:
			glBlendEquation(GL_FUNC_ADD);
			glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_PREMULTIPLIED_ALPHA:
			glBlendEquation(GL_FUNC_ADD);
			glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_DISABLED:
			glDisable(GL_BLEND);
			break;
	}

	RS::CullMode cull_mode = (RS::CullMode)shader_data->cull_modei;
	switch (cull_mode) {
		case RenderingServer::CULL_MODE_DISABLED:
			glDisable(GL_CULL_FACE);
			break;
		case RenderingServer::CULL_MODE_FRONT:
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
			break;
		case RenderingServer::CULL_MODE_BACK:
			glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);
			break;
	}

	GLuint global_buffer = GLES3::MaterialStorage::get_singleton()->global_shader_parameters_get_uniform_buffer();
	glBindBufferBase(GL_UNIFORM_BUFFER, MESH_RASTERIZER_GLOBALS_UNIFORM_LOCATION, global_buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glBindVertexArray(vertex_array);
	uint32_t vertex_count = mesh_storage->mesh_surface_get_vertices_drawn_count(surface);
	if (index_buffer != 0) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
		glDrawElements(primitive_gl, vertex_count, mesh_storage->mesh_surface_get_index_type(surface), nullptr);
	} else {
		glDrawArrays(primitive_gl, 0, vertex_count);
	}

	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, TextureStorage::system_fbo);
	glDeleteFramebuffers(1, &fbo);
}

MeshRasterizerGLES3::MeshRasterizerGLES3() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	String global_defines;
	global_defines += "#define MAX_GLOBAL_SHADER_UNIFORMS 256\n"; // TODO: this is arbitrary for now
	material_storage->shaders.mesh_rasterizer_shader.initialize(global_defines);
}

} //namespace GLES3
