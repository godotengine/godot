/*************************************************************************/
/*  rasterizer_canvas_gles2.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RASTERIZERCANVASGLES2_H
#define RASTERIZERCANVASGLES2_H

#include "rasterizer_canvas_base_gles2.h"

class RasterizerSceneGLES2;

class RasterizerCanvasGLES2 : public RasterizerCanvasBaseGLES2 {

	// used to determine whether we use hardware transform (none)
	// software transform all verts, or software transform just a translate
	// (no rotate or scale)
	enum TransformMode {
		TM_NONE,
		TM_ALL,
		TM_TRANSLATE,
	};

	// pod versions of vector and color and RID, need to be 32 bit for vertex format
	struct BatchVector2 {
		float x, y;
		void set(const Vector2 &p_o) {
			x = p_o.x;
			y = p_o.y;
		}
		void to(Vector2 &r_o) const {
			r_o.x = x;
			r_o.y = y;
		}
	};

	struct BatchColor {
		float r, g, b, a;
		void set(const Color &p_c) {
			r = p_c.r;
			g = p_c.g;
			b = p_c.b;
			a = p_c.a;
		}
		bool equals(const Color &p_c) const {
			return (r == p_c.r) && (g == p_c.g) && (b == p_c.b) && (a == p_c.a);
		}
		const float *get_data() const { return &r; }
	};

	struct BatchVertex {
		// must be 32 bit pod
		BatchVector2 pos;
		BatchVector2 uv;
	};

	struct BatchVertexColored : public BatchVertex {
		// must be 32 bit pod
		BatchColor col;
	};

	struct Batch {
		enum CommandType : uint32_t {
			BT_DEFAULT,
			BT_RECT,
		};

		CommandType type;
		uint32_t first_command; // also item reference number
		uint32_t num_commands;
		uint32_t first_quad;
		uint32_t batch_texture_id;
		BatchColor color;
	};

	struct BatchTex {
		enum TileMode : uint32_t {
			TILE_OFF,
			TILE_NORMAL,
			TILE_FORCE_REPEAT,
		};
		RID RID_texture;
		RID RID_normal;
		TileMode tile_mode;
		BatchVector2 tex_pixel_size;
	};

	// batch item may represent 1 or more items
	struct BItemJoined {
		uint32_t first_item_ref;
		uint32_t num_item_refs;

		Rect2 bounding_rect;

		// we are always splitting items with lots of commands,
		// and items with unhandled primitives (default)
		bool use_hardware_transform() const { return num_item_refs == 1; }
	};

	struct BItemRef {
		Item *item;
	};

	struct BatchData {
		BatchData();
		void reset_flush() {
			batches.reset();
			batch_textures.reset();
			vertices.reset();

			total_quads = 0;
			total_color_changes = 0;
		}

		GLuint gl_vertex_buffer;
		GLuint gl_index_buffer;

		uint32_t max_quads;
		uint32_t vertex_buffer_size_units;
		uint32_t vertex_buffer_size_bytes;
		uint32_t index_buffer_size_units;
		uint32_t index_buffer_size_bytes;

		RasterizerArrayGLES2<BatchVertex> vertices;
		RasterizerArrayGLES2<BatchVertexColored> vertices_colored;
		RasterizerArrayGLES2<Batch> batches;
		RasterizerArrayGLES2<Batch> batches_temp; // used for translating to colored vertex batches
		RasterizerArray_non_pod_GLES2<BatchTex> batch_textures; // the only reason this is non-POD is because of RIDs

		bool use_colored_vertices;

		RasterizerArrayGLES2<BItemJoined> items_joined;
		RasterizerArrayGLES2<BItemRef> item_refs;

		// counts
		int total_quads;

		// we keep a record of how many color changes caused new batches
		// if the colors are causing an excessive number of batches, we switch
		// to alternate batching method and add color to the vertex format.
		int total_color_changes;

		// measured in pixels, recalculated each frame
		float scissor_threshold_area;

		// global settings
		bool settings_use_batching; // the current use_batching (affected by flash)
		bool settings_use_batching_original_choice; // the choice entered in project settings
		bool settings_flash_batching; // for regression testing, flash between non-batched and batched renderer
		int settings_max_join_item_commands;
		float settings_colored_vertex_format_threshold;
		int settings_batch_buffer_num_verts;
		bool settings_scissor_lights;
		float settings_scissor_threshold; // 0.0 to 1.0
	} bdata;

	struct RenderItemState {
		RenderItemState();
		Item *current_clip;
		RasterizerStorageGLES2::Shader *shader_cache;
		bool rebind_shader;
		bool prev_use_skeleton;
		int last_blend_mode;
		RID canvas_last_material;
		Color final_modulate;

		// 'item group' is data over a single call to canvas_render_items
		int item_group_z;
		Color item_group_modulate;
		Light *item_group_light;
		Transform2D item_group_base_transform;
	};

	struct FillState {
		void reset() {
			curr_batch = 0;
			batch_tex_id = -1;
			use_hardware_transform = true;
			texpixel_size = Vector2(1, 1);
		}
		Batch *curr_batch;
		int batch_tex_id;
		bool use_hardware_transform;
		Vector2 texpixel_size;
	};

public:
	virtual void canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);

private:
	// legacy codepath .. to remove after testing
	void _canvas_render_item(Item *p_ci, RenderItemState &r_ris);
	_FORCE_INLINE_ void _canvas_item_render_commands(Item *p_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	// high level batch funcs
	void canvas_render_items_implementation(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	void render_joined_item(const BItemJoined &p_bij, RenderItemState &r_ris);
	void join_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	bool try_join_item(Item *p_ci, RenderItemState &r_ris, bool &r_batch_break);
	void render_joined_item_commands(const BItemJoined &p_bij, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);
	void render_batches(Item::Command *const *p_commands, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);
	bool prefill_joined_item(FillState &r_fill_state, int &r_command_start, Item *p_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);
	void flush_render_batches(Item *p_first_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	// low level batch funcs
	void _batch_translate_to_colored();
	_FORCE_INLINE_ int _batch_find_or_create_tex(const RID &p_texture, const RID &p_normal, bool p_tile, int p_previous_match);
	RasterizerStorageGLES2::Texture *_get_canvas_texture(const RID &p_texture) const;
	void _batch_upload_buffers();
	void _batch_render_rects(const Batch &p_batch, RasterizerStorageGLES2::Material *p_material);
	BatchVertex *_batch_vertex_request_new() { return bdata.vertices.request(); }
	Batch *_batch_request_new(bool p_blank = true);

	bool _detect_batch_break(Item *p_ci);
	void _software_transform_vertex(BatchVector2 &r_v, const Transform2D &p_tr) const;
	void _software_transform_vertex(Vector2 &r_v, const Transform2D &p_tr) const;
	TransformMode _find_transform_mode(bool p_use_hardware_transform, const Transform2D &p_tr, Transform2D &r_tr) const;

	// light scissoring
	bool _light_find_intersection(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect, Rect2 &r_cliprect) const;
	bool _light_scissor_begin(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect) const;
	void _calculate_scissor_threshold_area();

public:
	void initialize();
	RasterizerCanvasGLES2();
};

//////////////////////////////////////////////////////////////

_FORCE_INLINE_ void RasterizerCanvasGLES2::_software_transform_vertex(BatchVector2 &r_v, const Transform2D &p_tr) const {
	Vector2 vc(r_v.x, r_v.y);
	vc = p_tr.xform(vc);
	r_v.set(vc);
}

_FORCE_INLINE_ void RasterizerCanvasGLES2::_software_transform_vertex(Vector2 &r_v, const Transform2D &p_tr) const {
	r_v = p_tr.xform(r_v);
}

_FORCE_INLINE_ RasterizerCanvasGLES2::TransformMode RasterizerCanvasGLES2::_find_transform_mode(bool p_use_hardware_transform, const Transform2D &p_tr, Transform2D &r_tr) const {
	if (!p_use_hardware_transform) {
		r_tr = p_tr;

		// decided whether to do translate only for software transform
		if ((p_tr.elements[0].x == 1.0) &&
				(p_tr.elements[0].y == 0.0) &&
				(p_tr.elements[1].x == 0.0) &&
				(p_tr.elements[1].y == 1.0)) {
			return TM_TRANSLATE;
		} else {
			return TM_ALL;
		}
	}

	return TM_NONE;
}

#endif // RASTERIZERCANVASGLES2_H
