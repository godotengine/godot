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

#include "servers/rendering/renderer_rd/shaders/mesh_rasterizer.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/storage/mesh_rasterizer.h"

namespace RendererRD {

class MeshRasterizerRD : public MeshRasterizer {
	static MeshRasterizerRD *singleton;
	static constexpr int SAMPLERS_BINDING_FIRST_INDEX = 1;

	static MaterialStorage::ShaderData *_create_mesh_rasterizer_shader_funcs();
	static MaterialStorage::MaterialData *_create_mesh_rasterizer_material_funcs(MaterialStorage::ShaderData *p_shader);

	struct MeshRasterizerShaderData : public RendererRD::MaterialStorage::ShaderData {
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
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual RS::ShaderNativeSourceCode get_native_source_code() const;
		virtual Pair<ShaderRD *, RID> get_native_shader_and_version() const;

		uint64_t get_vertex_input_mask();

		virtual ~MeshRasterizerShaderData();
	};

	struct MeshRasterizerMaterialData : public RendererRD::MaterialStorage::MaterialData {
		MeshRasterizerShaderData *shader_data = nullptr;
		RID material_uniforms;
		RID material_uniforms_srgb;

		virtual void set_render_priority(int p_priority) {}
		virtual void set_next_pass(RID p_pass) {}
		virtual bool update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);

		virtual ~MeshRasterizerMaterialData();
	};

	enum {
		BASE_UNIFORM_SET,
		MATERIAL_UNIFORM_SET
	};

	MeshRasterizerShaderRD shader_file_rd;
	ShaderCompiler compiler;

public:
	void texture_drawable_blit_mesh_advanced(RID p_texture_drawable, RID p_material, RID p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer = 0) override;

	static MeshRasterizerRD *get_singleton();
	MeshRasterizerRD();
};
} //namespace RendererRD
