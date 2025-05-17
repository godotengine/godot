/**************************************************************************/
/*  mesh_rasterizer_rd.h                                                  */
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

#pragma once

#include "servers/rendering/mesh_rasterizer.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/rasterize_mesh.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"

namespace RendererRD {

class MeshRasterizerRD : public MeshRasterizer {
private:
	static MeshRasterizerRD *singleton;
	static constexpr int SAMPLERS_BINDING_FIRST_INDEX = 1;

	static MaterialStorage::ShaderData *_create_rasterize_mesh_shader_funcs();
	static MaterialStorage::MaterialData *_create_rasterize_mesh_material_funcs(MaterialStorage::ShaderData *p_shader);

	enum {
		BASE_UNIFORM_SET,
		MATERIAL_UNIFORM_SET
	};

	RD::VertexFormatID vertex_format;
	RD::PipelineColorBlendState pipeline_color_blend_state;

	struct RasterizeMeshShaderData : public RendererRD::MaterialStorage::ShaderData {
		RID version;
		RID shader_rd;
		RID base_uniforms;
		PipelineCacheRD pipeline_cache;
		int cull_modei = RS::CULL_MODE_BACK;

		bool valid = false;
		Vector<ShaderCompiler::GeneratedCode::Texture> texture_uniforms;
		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size = 0;

		String code;

		virtual void set_code(const String &p_code);
		virtual bool is_animated() const { return false; }
		virtual bool casts_shadows() const { return false; }

		~RasterizeMeshShaderData();
	};

	struct RasterizeMeshMaterialData : public RendererRD::MaterialStorage::MaterialData {
		RasterizeMeshShaderData *shader_data = nullptr;
		RID material_uniforms;

		virtual void set_render_priority(int p_priority) {}
		virtual void set_next_pass(RID p_pass) {}
		virtual bool update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);
	};

	RasterizeMeshShaderRD shader_file_rd;
	RID default_shader;
	RID default_material;
	RasterizeMeshShaderData *default_shader_data;
	RasterizeMeshMaterialData *default_material_data;
	ShaderCompiler compiler;

	struct MeshRasterizerData {
		RasterizeMeshMaterialData *material_data = nullptr;
		RasterizeMeshShaderData *shader_data = nullptr;

		Color bg_color = Color(0, 0, 0, 0);

		RID material;
		RID mesh;
		int surface_index = 0;

		RID framebuffer_texture_id;
		RID framebuffer_id;
		RID vertex_array_id;
		RID index_array_id;
		RID index_buffer_id;
		RID vertex_buffer_pos_id;
		RID vertex_buffer_uv_id;
		RID vertex_buffer_color_id;

		void update_vertex();
		void update_material();
		void draw();

		DependencyTracker dependency_tracker;

		MeshRasterizerData();
	};

	mutable RID_Owner<MeshRasterizerData, true> mesh_rasterizer_owner;

	static void _dependency_changed(Dependency::DependencyChangedNotification p_notification, DependencyTracker *p_tracker);
	static void _dependency_deleted(const RID &p_dependency, DependencyTracker *p_tracker);

public:
	RID mesh_rasterizer_allocate();
	void mesh_rasterizer_initialize(RID p_mesh_rasterizer, int p_width, int p_height, RS::RasterizedTextureFormat p_texture_format, bool p_generate_mipmaps);
	void mesh_rasterizer_set_bg_color(RID p_mesh_rasterizer, const Color &p_bg_color);
	void mesh_rasterizer_set_mesh(RID p_mesh_rasterizer, RID p_mesh, int p_surface_index);
	void mesh_rasterizer_set_material(RID p_mesh_rasterizer, RID p_material);
	void mesh_rasterizer_draw(RID p_mesh_rasterizer);
	RID mesh_rasterizer_get_rd_texture(RID p_mesh_rasterizer);
	bool free(RID p_mesh_rasterizer);

	void free_shader();
	static MeshRasterizerRD *get_singleton();
	MeshRasterizerRD();
};
} //namespace RendererRD
