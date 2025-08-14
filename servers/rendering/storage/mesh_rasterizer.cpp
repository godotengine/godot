/**************************************************************************/
/*  mesh_rasterizer.cpp                                                   */
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

#include "mesh_rasterizer.h"
#include "servers/rendering/rendering_server_globals.h"

void MeshRasterizer::texture_drawable_blit_texture_rect_region(RID p_texture_drawable, RID p_source_texture, Rect2 p_dst_rect, Rect2 p_src_rect, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer) {
	Vector2 dst_size = RSG::texture_storage->texture_2d_get_size(p_texture_drawable);
	Vector2 src_size = RSG::texture_storage->texture_2d_get_size(p_source_texture);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("src_tex"), p_source_texture);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("src_offset"), p_src_rect.position);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("src_region"), p_src_rect.size);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("src_size"), src_size);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("dst_size"), dst_size);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("dst_offset"), p_dst_rect.position);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("dst_region"), p_dst_rect.size);
	RSG::material_storage->material_set_param(default_blit_material, SNAME("modulate"), p_modulate);

	texture_drawable_blit_mesh_advanced(p_texture_drawable, default_blit_material, default_blit_mesh, 0, p_blend_mode, p_clear_color, p_layer);
}

void MeshRasterizer::initialize() {
	default_blit_shader = RSG::material_storage->shader_allocate();
	default_blit_material = RSG::material_storage->material_allocate();
	RSG::material_storage->shader_initialize(default_blit_shader);
	RSG::material_storage->material_initialize(default_blit_material);
	RSG::material_storage->shader_set_code(default_blit_shader, R"(
		shader_type mesh_rasterizer;

		uniform vec2 dst_size;
		uniform vec2 src_size;
		uniform vec2 dst_offset;
		uniform vec2 src_offset;
		uniform vec2 dst_region;
		uniform vec2 src_region;
		uniform sampler2D src_tex: source_color,repeat_enable;
		uniform vec4 modulate: source_color = vec4(1.0);

		void vertex(){
			POSITION.xy = dst_region / dst_size * POSITION.xy + (dst_offset - dst_size + dst_region) / dst_size;
		}

		void fragment(){
			vec2 uv = src_region / src_size * UV + src_offset / src_size;
			OUTPUT_COLOR = modulate * texture(src_tex, uv);
		}
	)");
	RSG::material_storage->material_set_shader(default_blit_material, default_blit_shader);

	default_blit_mesh = RSG::mesh_storage->mesh_allocate();
	RSG::mesh_storage->mesh_initialize(default_blit_mesh);

	Array surface_arrays;
	surface_arrays.resize(RS::ArrayType::ARRAY_MAX);
	surface_arrays[RS::ARRAY_VERTEX] = Vector<Vector2>{
		Vector2(1, -1),
		Vector2(-1, -1),
		Vector2(1, 1),
		Vector2(-1, 1)
	};
	surface_arrays[RS::ArrayType::ARRAY_INDEX] = Vector<int32_t>{
		0, 1, 2, 1, 3, 2
	};
	surface_arrays[RS::ArrayType::ARRAY_TEX_UV] = Vector<Vector2>{
		Vector2(1, 1),
		Vector2(0, 1),
		Vector2(1, 0),
		Vector2(0, 0),
	};
	RS::SurfaceData surface_data;
	RS::get_singleton()->mesh_create_surface_data_from_arrays(&surface_data, RS::PRIMITIVE_TRIANGLES, surface_arrays);
	RSG::mesh_storage->mesh_add_surface(default_blit_mesh, surface_data);
}

void MeshRasterizer::finalize() {
	RSG::utilities->free(default_blit_material);
	RSG::utilities->free(default_blit_shader);
	RSG::utilities->free(default_blit_mesh);
}
