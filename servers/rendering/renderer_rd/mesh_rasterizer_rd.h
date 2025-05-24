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
#include "servers/rendering/renderer_rd/shaders/mesh_rasterizer.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"

namespace RendererRD {

class MeshRasterizerRD : public MeshRasterizer {
private:
	static MeshRasterizerRD *singleton;
	static constexpr int SAMPLERS_BINDING_FIRST_INDEX = 1;

	static MaterialStorage::ShaderData *_create_mesh_rasterizer_shader_funcs();
	static MaterialStorage::MaterialData *_create_mesh_rasterizer_material_funcs(MaterialStorage::ShaderData *p_shader);

	RD::PipelineColorBlendState::Attachment attachment_mix;
	RD::PipelineColorBlendState::Attachment attachment_add;
	RD::PipelineColorBlendState::Attachment attachment_sub;
	RD::PipelineColorBlendState::Attachment attachment_mul;
	RD::PipelineColorBlendState::Attachment attachment_premult_alpha;

	struct RasterizeMeshShaderData : public RendererRD::MaterialStorage::ShaderData {
		RID version;
		RID shader_rd;
		RID base_uniforms;

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

	struct PipelineCacheKey {
		uint64_t shader_id;
		RD::FramebufferFormatID framebuffer_formt;
		RD::RenderPrimitive primitive;
		RD::TextureSamples samples;
		Ref<RasterizerBlendState> blend_state;

		bool operator==(const PipelineCacheKey &b) const {
			if (shader_id != b.shader_id) {
				return false;
			} else if (framebuffer_formt != b.framebuffer_formt) {
				return false;
			} else if (primitive != b.primitive) {
				return false;
			} else if (samples != b.samples) {
				return false;
			} else if (b.blend_state.is_null()) {
				// No need to recreate pipeline if it clears texture.
				return true;
			} else if (blend_state.is_null()) {
				return false;
			} else {
				return blend_state->equal(b.blend_state);
			}
		}
	};

	struct MeshRasterizerData {
		RD::RenderPrimitive primitive = RD::RENDER_PRIMITIVE_TRIANGLES;

		RID mesh;
		int surface_index = 0;

		Pair<RID, RID> rd_texture_samples_cache;
		Pair<PipelineCacheKey, RID> pipeline_cache;

		RID vertex_array_rid;
		RID index_array_rid;
		RID index_buffer_rid;
		RID vertex_buffer_pos_rid;
		RID vertex_buffer_uv_rid;
		RID vertex_buffer_color_rid;

		void update_vertex();
		void update_material();

		DependencyTracker dependency_tracker;

		MeshRasterizerData();
	};

	enum {
		BASE_UNIFORM_SET,
		MATERIAL_UNIFORM_SET
	};

	RD::VertexFormatID vertex_format;
	Vector<RD::FramebufferPass> render_passes;
	MeshRasterizerShaderRD shader_file_rd;
	ShaderCompiler compiler;
	mutable RID_Owner<MeshRasterizerData, true> mesh_rasterizer_owner;

	static void _dependency_changed(Dependency::DependencyChangedNotification p_notification, DependencyTracker *p_tracker);
	static void _dependency_deleted(const RID &p_dependency, DependencyTracker *p_tracker);

public:
	RID mesh_rasterizer_allocate();
	void mesh_rasterizer_initialize(RID p_mesh_rasterizer, RID p_mesh, int surface_index);
	void mesh_rasterizer_draw(RID p_mesh_rasterizer, RID p_material, RID p_texture_drawable, Ref<RasterizerBlendState> p_blend_state, const Color &p_bg_color, RD::TextureSamples p_multisample = RD::TEXTURE_SAMPLES_1);

	bool free(RID p_mesh_rasterizer);

	static MeshRasterizerRD *get_singleton();
	MeshRasterizerRD();
};
} //namespace RendererRD
