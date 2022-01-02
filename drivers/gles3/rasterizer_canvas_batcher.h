/*************************************************************************/
/*  rasterizer_canvas_batcher.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RASTERIZER_CANVAS_BATCHER_H
#define RASTERIZER_CANVAS_BATCHER_H

#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "rasterizer_array.h"
#include "rasterizer_asserts.h"
#include "rasterizer_storage_common.h"

#include "core/config/project_settings.h"
#include "servers/rendering/renderer_compositor.h"

// We are using the curiously recurring template pattern
// https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
// For static polymorphism.

// This makes it super easy to access
// data / call funcs in the derived rasterizers from the base without writing and
// maintaining a boatload of virtual functions.
// In addition it assures that vtable will not be used and the function calls can be optimized,
// because it gives compile time static polymorphism.

// These macros makes it simpler and less verbose to define (and redefine) the inline functions
// template preamble
#define T_PREAMBLE template <class T, typename T_STORAGE>
// class preamble
#define C_PREAMBLE RasterizerCanvasBatcher<T, T_STORAGE>
// generic preamble
#define PREAMBLE(RET_T) \
	T_PREAMBLE          \
	RET_T C_PREAMBLE

template <class T, typename T_STORAGE>
class RasterizerCanvasBatcher {
public:
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
		void set(float xx, float yy) {
			x = xx;
			y = yy;
		}
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
		void set_white() {
			r = 1.0f;
			g = 1.0f;
			b = 1.0f;
			a = 1.0f;
		}
		void set(const Color &p_c) {
			r = p_c.r;
			g = p_c.g;
			b = p_c.b;
			a = p_c.a;
		}
		void set(float rr, float gg, float bb, float aa) {
			r = rr;
			g = gg;
			b = bb;
			a = aa;
		}
		bool operator==(const BatchColor &p_c) const {
			return (r == p_c.r) && (g == p_c.g) && (b == p_c.b) && (a == p_c.a);
		}
		bool operator!=(const BatchColor &p_c) const { return (*this == p_c) == false; }
		bool equals(const Color &p_c) const {
			return (r == p_c.r) && (g == p_c.g) && (b == p_c.b) && (a == p_c.a);
		}
		const float *get_data() const { return &r; }
		String to_string() const {
			String sz = "{";
			const float *data = get_data();
			for (int c = 0; c < 4; c++) {
				float f = data[c];
				int val = ((f * 255.0f) + 0.5f);
				sz += String(Variant(val)) + " ";
			}
			sz += "}";
			return sz;
		}
	};

	// simplest FVF - local or baked position
	struct BatchVertex {
		// must be 32 bit pod
		BatchVector2 pos;
		BatchVector2 uv;
	};

	// simple FVF but also incorporating baked color
	struct BatchVertexColored : public BatchVertex {
		// must be 32 bit pod
		BatchColor col;
	};

	// if we are using normal mapping, we need light angles to be sent
	struct BatchVertexLightAngled : public BatchVertexColored {
		// must be pod
		float light_angle;
	};

	// CUSTOM SHADER vertex formats. These are larger but will probably
	// be needed with custom shaders in order to have the data accessible in the shader.

	// if we are using COLOR in vertex shader but not position (VERTEX)
	struct BatchVertexModulated : public BatchVertexLightAngled {
		BatchColor modulate;
	};

	struct BatchTransform {
		BatchVector2 translate;
		BatchVector2 basis[2];
	};

	// last resort, specially for custom shader, we put everything possible into a huge FVF
	// not very efficient, but better than no batching at all.
	struct BatchVertexLarge : public BatchVertexModulated {
		// must be pod
		BatchTransform transform;
	};

	// Batch should be as small as possible, and ideally nicely aligned (is 32 bytes at the moment)
	struct Batch {
		RasterizerStorageCommon::BatchType type; // should be 16 bit
		uint16_t batch_texture_id;

		// also item reference number
		uint32_t first_command;

		// in the case of DEFAULT, this is num commands.
		// with rects, is number of command and rects.
		// with lines, is number of lines
		uint32_t num_commands;

		// first vertex of this batch in the vertex lists
		uint32_t first_vert;

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
		uint32_t flags;
	};

	// items in a list to be sorted prior to joining
	struct BSortItem {
		// have a function to keep as pod, rather than operator
		void assign(const BSortItem &o) {
			item = o.item;
			z_index = o.z_index;
		}
		RendererCanvasRender::Item *item;
		int z_index;
	};

	// batch item may represent 1 or more items
	struct BItemJoined {
		uint32_t first_item_ref;
		uint32_t num_item_refs;

		Rect2 bounding_rect;

		// note the z_index  may only be correct for the first of the joined item references
		// this has implications for light culling with z ranged lights.
		int16_t z_index;

		// these are defined in RasterizerStorageCommon::BatchFlags
		uint16_t flags;

		// we are always splitting items with lots of commands,
		// and items with unhandled primitives (default)
		bool use_hardware_transform() const { return num_item_refs == 1; }
	};

	struct BItemRef {
		RendererCanvasRender::Item *item;
		Color final_modulate;
	};

	struct BLightRegion {
		void reset() {
			light_bitfield = 0;
			shadow_bitfield = 0;
			too_many_lights = false;
		}
		uint64_t light_bitfield;
		uint64_t shadow_bitfield;
		bool too_many_lights; // we can only do light region optimization if there are 64 or less lights
	};

	struct BatchData {
		BatchData() {
			reset_flush();
			reset_joined_item();

			gl_vertex_buffer = 0;
			gl_index_buffer = 0;
			max_quads = 0;
			vertex_buffer_size_units = 0;
			vertex_buffer_size_bytes = 0;
			index_buffer_size_units = 0;
			index_buffer_size_bytes = 0;

			use_colored_vertices = false;

			settings_use_batching = false;
			settings_max_join_item_commands = 0;
			settings_colored_vertex_format_threshold = 0.0f;
			settings_batch_buffer_num_verts = 0;
			scissor_threshold_area = 0.0f;
			joined_item_batch_flags = 0;
			diagnose_frame = false;
			next_diagnose_tick = 10000;
			diagnose_frame_number = 9999999999; // some high number
			join_across_z_indices = true;
			settings_item_reordering_lookahead = 0;

			settings_use_batching_original_choice = false;
			settings_flash_batching = false;
			settings_diagnose_frame = false;
			settings_scissor_lights = false;
			settings_scissor_threshold = -1.0f;
			settings_use_single_rect_fallback = false;
			settings_use_software_skinning = true;
			settings_ninepatch_mode = 0; // default
			settings_light_max_join_items = 16;

			settings_uv_contract = false;
			settings_uv_contract_amount = 0.0f;

			buffer_mode_batch_upload_send_null = true;
			buffer_mode_batch_upload_flag_stream = false;

			stats_items_sorted = 0;
			stats_light_items_joined = 0;
		}

		// called for each joined item
		void reset_joined_item() {
			// noop but left in as a stub
		}

		// called after each flush
		void reset_flush() {
			batches.reset();
			batch_textures.reset();

			vertices.reset();
			light_angles.reset();
			vertex_colors.reset();
			vertex_modulates.reset();
			vertex_transforms.reset();

			total_quads = 0;
			total_verts = 0;
			total_color_changes = 0;

			use_light_angles = false;
			use_modulate = false;
			use_large_verts = false;
			fvf = RasterizerStorageCommon::FVF_REGULAR;
		}

		unsigned int gl_vertex_buffer;
		unsigned int gl_index_buffer;

		uint32_t max_quads;
		uint32_t vertex_buffer_size_units;
		uint32_t vertex_buffer_size_bytes;
		uint32_t index_buffer_size_units;
		uint32_t index_buffer_size_bytes;

		// small vertex FVF type - pos and UV.
		// This will always be written to initially, but can be translated
		// to larger FVFs if necessary.
		RasterizerArray<BatchVertex> vertices;

		// extra data which can be stored during prefilling, for later translation to larger FVFs
		RasterizerArray<float> light_angles;
		RasterizerArray<BatchColor> vertex_colors; // these aren't usually used, but are for polys
		RasterizerArray<BatchColor> vertex_modulates;
		RasterizerArray<BatchTransform> vertex_transforms;

		// instead of having a different buffer for each vertex FVF type
		// we have a special array big enough for the biggest FVF
		// which can have a changeable unit size, and reuse it.
		RasterizerUnitArray unit_vertices;

		RasterizerArray<Batch> batches;
		RasterizerArray<Batch> batches_temp; // used for translating to colored vertex batches
		RasterizerArray_non_pod<BatchTex> batch_textures; // the only reason this is non-POD is because of RIDs

		// SHOULD THESE BE IN FILLSTATE?
		// flexible vertex format.
		// all verts have pos and UV.
		// some have color, some light angles etc.
		RasterizerStorageCommon::FVF fvf;
		bool use_colored_vertices;
		bool use_light_angles;
		bool use_modulate;
		bool use_large_verts;

		// if the shader is using MODULATE, we prevent baking color so the final_modulate can
		// be read in the shader.
		// if the shader is reading VERTEX, we prevent baking vertex positions with extra matrices etc
		// to prevent the read position being incorrect.
		// These flags are defined in RasterizerStorageCommon::BatchFlags
		uint32_t joined_item_batch_flags;

		RasterizerArray<BItemJoined> items_joined;
		RasterizerArray<BItemRef> item_refs;

		// items are sorted prior to joining
		RasterizerArray<BSortItem> sort_items;

		// new for Godot 4 .. the client outputs a linked list so we need to convert this
		// to a linear array
		LocalVector<RendererCanvasRender::Item::Command *> command_shortlist;

		// counts
		int total_quads;
		int total_verts;

		// we keep a record of how many color changes caused new batches
		// if the colors are causing an excessive number of batches, we switch
		// to alternate batching method and add color to the vertex format.
		int total_color_changes;

		// measured in pixels, recalculated each frame
		float scissor_threshold_area;

		// diagnose this frame, every nTh frame when settings_diagnose_frame is on
		bool diagnose_frame;
		String frame_string;
		uint32_t next_diagnose_tick;
		uint64_t diagnose_frame_number;

		// whether to join items across z_indices - this can interfere with z ranged lights,
		// so has to be disabled in some circumstances
		bool join_across_z_indices;

		// global settings
		bool settings_use_batching; // the current use_batching (affected by flash)
		bool settings_use_batching_original_choice; // the choice entered in project settings
		bool settings_flash_batching; // for regression testing, flash between non-batched and batched renderer
		bool settings_diagnose_frame; // print out batches to help optimize / regression test
		int settings_max_join_item_commands;
		float settings_colored_vertex_format_threshold;
		int settings_batch_buffer_num_verts;
		bool settings_scissor_lights;
		float settings_scissor_threshold; // 0.0 to 1.0
		int settings_item_reordering_lookahead;
		bool settings_use_single_rect_fallback;
		bool settings_use_software_skinning;
		int settings_light_max_join_items;
		int settings_ninepatch_mode;

		// buffer orphaning modes
		bool buffer_mode_batch_upload_send_null;
		bool buffer_mode_batch_upload_flag_stream;

		// uv contraction
		bool settings_uv_contract;
		float settings_uv_contract_amount;

		// only done on diagnose frame
		void reset_stats() {
			stats_items_sorted = 0;
			stats_light_items_joined = 0;
		}

		// frame stats (just for monitoring and debugging)
		int stats_items_sorted;
		int stats_light_items_joined;
	} bdata;

	struct FillState {
		void reset_flush() {
			// don't reset members that need to be preserved after flushing
			// half way through a list of commands
			curr_batch = 0;
			batch_tex_id = -1;
			texpixel_size = Vector2(1, 1);
			contract_uvs = false;

			sequence_batch_type_flags = 0;
		}

		void reset_joined_item(bool p_use_hardware_transform) {
			reset_flush();
			use_hardware_transform = p_use_hardware_transform;
			extra_matrix_sent = false;
		}

		// for batching multiple types, we don't allow mixing RECTs / LINEs etc.
		// using flags allows quicker rejection of sequences with different batch types
		uint32_t sequence_batch_type_flags;

		Batch *curr_batch;
		int batch_tex_id;
		bool use_hardware_transform;
		bool contract_uvs;
		Vector2 texpixel_size;
		Color final_modulate;
		TransformMode transform_mode;
		TransformMode orig_transform_mode;

		// support for extra matrices
		bool extra_matrix_sent; // whether sent on this item (in which case software transform can't be used untl end of item)
		int transform_extra_command_number_p1; // plus one to allow fast checking against zero
		Transform2D transform_combined; // final * extra
	};

	// used during try_join
	struct RenderItemState {
		RenderItemState() { reset(); }
		void reset() {
			current_clip = nullptr;
			shader_cache = nullptr;
			rebind_shader = true;
			prev_use_skeleton = false;
			last_blend_mode = -1;
			canvas_last_material = RID();
			item_group_z = 0;
			item_group_light = nullptr;
			final_modulate = Color(-1.0, -1.0, -1.0, -1.0); // just something unlikely

			joined_item_batch_type_flags_curr = 0;
			joined_item_batch_type_flags_prev = 0;

			joined_item = nullptr;
		}

		RendererCanvasRender::Item *current_clip;
		typename T_STORAGE::Shader *shader_cache;
		bool rebind_shader;
		bool prev_use_skeleton;
		bool prev_distance_field;
		int last_blend_mode;
		RID canvas_last_material;
		Color final_modulate;

		// used for joining items only
		BItemJoined *joined_item;
		bool join_batch_break;
		BLightRegion light_region;

		// we need some logic to prevent joining items that have vastly different batch types
		// these are defined in RasterizerStorageCommon::BatchTypeFlags
		uint32_t joined_item_batch_type_flags_curr;
		uint32_t joined_item_batch_type_flags_prev;

		// 'item group' is data over a single call to canvas_render_items
		int item_group_z;
		Color item_group_modulate;
		RendererCanvasRender::Light *item_group_light;
		Transform2D item_group_base_transform;
	} _render_item_state;

	bool use_nvidia_rect_workaround;

	//////////////////////////////////////////////////////////////////////////////
	// End of structs used by the batcher. Beginning of funcs.
private:
	// curiously recurring template pattern - allows access to functions in the DERIVED class
	// this is kind of like using virtual functions but more efficient as they are resolved at compile time
	T_STORAGE *get_storage() { return static_cast<const T *>(this)->storage; }
	const T_STORAGE *get_storage() const { return static_cast<const T *>(this)->storage; }
	T *get_this() { return static_cast<T *>(this); }
	const T *get_this() const { return static_cast<const T *>(this); }

protected:
	// main functions called from the rasterizer canvas
	void batch_constructor();
	void batch_initialize();

	void batch_canvas_begin();
	void batch_canvas_end();
	void batch_canvas_render_items_begin(const Color &p_modulate, RendererCanvasRender::Light *p_light, const Transform2D &p_base_transform);
	void batch_canvas_render_items_end();
	void batch_canvas_render_items(RendererCanvasRender::Item *p_item_list, int p_z, const Color &p_modulate, RendererCanvasRender::Light *p_light, const Transform2D &p_base_transform);

	// recording and sorting items from the initial pass
	void record_items(RendererCanvasRender::Item *p_item_list, int p_z);
	void join_sorted_items();
	void sort_items();
	bool _sort_items_match(const BSortItem &p_a, const BSortItem &p_b) const;
	bool sort_items_from(int p_start);

	// joining logic
	bool _disallow_item_join_if_batch_types_too_different(RenderItemState &r_ris, uint32_t btf_allowed);
	bool _detect_item_batch_break(RenderItemState &r_ris, RendererCanvasRender::Item *p_ci, bool &r_batch_break);

	// drives the loop filling batches and flushing
	void render_joined_item_commands(const BItemJoined &p_bij, RendererCanvasRender::Item *p_current_clip, bool &r_reclip, typename T_STORAGE::Material *p_material, bool p_lit);

private:
	// flush once full or end of joined item
	void flush_render_batches(RendererCanvasRender::Item *p_first_item, RendererCanvasRender::Item *p_current_clip, bool &r_reclip, typename T_STORAGE::Material *p_material, uint32_t p_sequence_batch_type_flags);

	// a single joined item can contain multiple itemrefs, and thus create lots of batches
	// command start given a separate name to make easier to tell apart godot 3 and 4
	bool prefill_joined_item(FillState &r_fill_state, RendererCanvasRender::Item::Command **r_first_command, RendererCanvasRender::Item *p_item, RendererCanvasRender::Item *p_current_clip, bool &r_reclip, typename T_STORAGE::Material *p_material);

	// prefilling different types of batch

	// default batch is an 'unhandled' legacy type batch that will be drawn with the legacy path,
	// all other batches are accelerated.
	void _prefill_default_batch(FillState &r_fill_state, int p_command_num, const RendererCanvasRender::Item &p_item);

	// accelerated batches
	bool _prefill_rect(RendererCanvasRender::Item::CommandRect *rect, FillState &r_fill_state, int &r_command_start, int command_num, int command_count, RendererCanvasRender::Item::Command *const *commands, RendererCanvasRender::Item *p_item, bool multiply_final_modulate);

	// dealing with textures
	int _batch_find_or_create_tex(const RID &p_texture, const RID &p_normal, bool p_tile, int p_previous_match);

protected:
	// legacy support for non batched mode
	void _legacy_canvas_item_render_commands(RendererCanvasRender::Item *p_item, RendererCanvasRender::Item *p_current_clip, bool &r_reclip, typename T_STORAGE::Material *p_material);

	// light scissoring
	bool _light_scissor_begin(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect) const;
	bool _light_find_intersection(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect, Rect2 &r_cliprect) const;
	void _calculate_scissor_threshold_area();

private:
	// translating vertex formats prior to rendering
	void _translate_batches_to_vertex_colored_FVF();
	template <class BATCH_VERTEX_TYPE, bool INCLUDE_LIGHT_ANGLES, bool INCLUDE_MODULATE, bool INCLUDE_LARGE>
	void _translate_batches_to_larger_FVF(uint32_t p_sequence_batch_type_flags);

protected:
	// accessory funcs
	void _software_transform_vertex(BatchVector2 &r_v, const Transform2D &p_tr) const;
	void _software_transform_vertex(Vector2 &r_v, const Transform2D &p_tr) const;
	TransformMode _find_transform_mode(const Transform2D &p_tr) const {
		// decided whether to do translate only for software transform
		if ((p_tr.elements[0].x == 1.0f) &&
				(p_tr.elements[0].y == 0.0f) &&
				(p_tr.elements[1].x == 0.0f) &&
				(p_tr.elements[1].y == 1.0f)) {
			return TM_TRANSLATE;
		}

		return TM_ALL;
	}

	typename T_STORAGE::Texture *_get_canvas_texture(const RID &p_texture) const {
		if (p_texture.is_valid()) {
			typename T_STORAGE::Texture *texture = get_storage()->texture_owner.get_or_null(p_texture);

			if (texture) {
				return texture->get_ptr();
			}
		}

		return 0;
	}

public:
	Batch *_batch_request_new(bool p_blank = true) {
		Batch *batch = bdata.batches.request();
		if (!batch) {
			// grow the batches
			bdata.batches.grow();

			// and the temporary batches (used for color verts)
			bdata.batches_temp.reset();
			bdata.batches_temp.grow();

			// this should always succeed after growing
			batch = bdata.batches.request();
			RAST_DEBUG_ASSERT(batch);
		}

		if (p_blank)
			memset(batch, 0, sizeof(Batch));

		return batch;
	}

	BatchVertex *_batch_vertex_request_new() {
		return bdata.vertices.request();
	}

protected:
	int godot4_commands_count(RendererCanvasRender::Item::Command *p_comm) const {
		int count = 0;
		while (p_comm) {
			count++;
			p_comm = p_comm->next;
		}
		return count;
	}

	unsigned int godot4_commands_to_vector(RendererCanvasRender::Item::Command *p_comm, LocalVector<RendererCanvasRender::Item::Command *> &p_list) {
		p_list.clear();
		while (p_comm) {
			p_list.push_back(p_comm);
			p_comm = p_comm->next;
		}
		return p_list.size();
	}
};

PREAMBLE(void)::batch_canvas_begin() {
	// diagnose_frame?
	bdata.frame_string = ""; // just in case, always set this as we don't want a string leak in release...
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
	if (bdata.settings_diagnose_frame) {
		bdata.diagnose_frame = false;

		uint32_t tick = OS::get_singleton()->get_ticks_msec();
		uint64_t frame = Engine::get_singleton()->get_frames_drawn();

		if (tick >= bdata.next_diagnose_tick) {
			bdata.next_diagnose_tick = tick + 10000;

			// the plus one is prevent starting diagnosis half way through frame
			bdata.diagnose_frame_number = frame + 1;
		}

		if (frame == bdata.diagnose_frame_number) {
			bdata.diagnose_frame = true;
			bdata.reset_stats();
		}

		if (bdata.diagnose_frame) {
			bdata.frame_string = "canvas_begin FRAME " + itos(frame) + "\n";
		}
	}
#endif
}

PREAMBLE(void)::batch_canvas_end() {
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
	if (bdata.diagnose_frame) {
		bdata.frame_string += "canvas_end\n";
		if (bdata.stats_items_sorted) {
			bdata.frame_string += "\titems reordered: " + itos(bdata.stats_items_sorted) + "\n";
		}
		if (bdata.stats_light_items_joined) {
			bdata.frame_string += "\tlight items joined: " + itos(bdata.stats_light_items_joined) + "\n";
		}

		print_line(bdata.frame_string);
	}
#endif
}

PREAMBLE(void)::batch_canvas_render_items_begin(const Color &p_modulate, RendererCanvasRender::Light *p_light, const Transform2D &p_base_transform) {
	// if we are debugging, flash each frame between batching renderer and old version to compare for regressions
	if (bdata.settings_flash_batching) {
		if ((Engine::get_singleton()->get_frames_drawn() % 2) == 0)
			bdata.settings_use_batching = true;
		else
			bdata.settings_use_batching = false;
	}

	if (!bdata.settings_use_batching) {
		return;
	}

	// this only needs to be done when screen size changes, but this should be
	// infrequent enough
	_calculate_scissor_threshold_area();

	// set up render item state for all the z_indexes (this is common to all z_indexes)
	_render_item_state.reset();
	_render_item_state.item_group_modulate = p_modulate;
	_render_item_state.item_group_light = p_light;
	_render_item_state.item_group_base_transform = p_base_transform;
	_render_item_state.light_region.reset();

	// batch break must be preserved over the different z indices,
	// to prevent joining to an item on a previous index if not allowed
	_render_item_state.join_batch_break = false;

	// whether to join across z indices depends on whether there are z ranged lights.
	// joined z_index items can be wrongly classified with z ranged lights.
	bdata.join_across_z_indices = true;

	int light_count = 0;
	while (p_light) {
		light_count++;

		if ((p_light->z_min != RS::CANVAS_ITEM_Z_MIN) || (p_light->z_max != RS::CANVAS_ITEM_Z_MAX)) {
			// prevent joining across z indices. This would have caused visual regressions
			bdata.join_across_z_indices = false;
		}

		p_light = p_light->next_ptr;
	}

	// can't use the light region bitfield if there are too many lights
	// hopefully most games won't blow this limit..
	// if they do they will work but it won't batch join items just in case
	if (light_count > 64) {
		_render_item_state.light_region.too_many_lights = true;
	}
}

PREAMBLE(void)::batch_canvas_render_items_end() {
	if (!bdata.settings_use_batching) {
		return;
	}

	join_sorted_items();

#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
	if (bdata.diagnose_frame) {
		bdata.frame_string += "items\n";
	}
#endif

	// batching render is deferred until after going through all the z_indices, joining all the items
	get_this()->canvas_render_items_implementation(0, 0, _render_item_state.item_group_modulate,
			_render_item_state.item_group_light,
			_render_item_state.item_group_base_transform);

	bdata.items_joined.reset();
	bdata.item_refs.reset();
	bdata.sort_items.reset();
}

PREAMBLE(void)::batch_canvas_render_items(RendererCanvasRender::Item *p_item_list, int p_z, const Color &p_modulate, RendererCanvasRender::Light *p_light, const Transform2D &p_base_transform) {
	// stage 1 : join similar items, so that their state changes are not repeated,
	// and commands from joined items can be batched together
	if (bdata.settings_use_batching) {
		record_items(p_item_list, p_z);
		return;
	}

	// only legacy renders at this stage, batched renderer doesn't render until canvas_render_items_end()
	get_this()->canvas_render_items_implementation(p_item_list, p_z, p_modulate, p_light, p_base_transform);
}

// Default batches will not occur in software transform only items
// EXCEPT IN THE CASE OF SINGLE RECTS (and this may well not occur, check the logic in prefill_join_item TYPE_RECT)
// but can occur where transform commands have been sent during hardware batch
PREAMBLE(void)::_prefill_default_batch(FillState &r_fill_state, int p_command_num, const RendererCanvasRender::Item &p_item) {
	if (r_fill_state.curr_batch->type == RasterizerStorageCommon::BT_DEFAULT) {
		// don't need to flush an extra transform command?
		if (!r_fill_state.transform_extra_command_number_p1) {
			// another default command, just add to the existing batch
			r_fill_state.curr_batch->num_commands++;
		} else {
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
			if (r_fill_state.transform_extra_command_number_p1 != p_command_num) {
				WARN_PRINT_ONCE("_prefill_default_batch : transform_extra_command_number_p1 != p_command_num");
			}
#endif
			// if the first member of the batch is a transform we have to be careful
			if (!r_fill_state.curr_batch->num_commands) {
				// there can be leading useless extra transforms (sometimes happens with debug collision polys)
				// we need to rejig the first_command for the first useful transform
				r_fill_state.curr_batch->first_command += r_fill_state.transform_extra_command_number_p1 - 1;
			}

			// we do have a pending extra transform command to flush
			// either the extra transform is in the prior command, or not, in which case we need 2 batches
			r_fill_state.curr_batch->num_commands += 2;

			r_fill_state.transform_extra_command_number_p1 = 0; // mark as sent
			r_fill_state.extra_matrix_sent = true;

			// the original mode should always be hardware transform ..
			// test this assumption
			//CRASH_COND(r_fill_state.orig_transform_mode != TM_NONE);
			r_fill_state.transform_mode = r_fill_state.orig_transform_mode;

			// do we need to restore anything else?
		}
	} else {
		// end of previous different type batch, so start new default batch

		// first consider whether there is a dirty extra matrix to send
		if (r_fill_state.transform_extra_command_number_p1) {
			// get which command the extra is in, and blank all the records as it no longer is stored CPU side
			int extra_command = r_fill_state.transform_extra_command_number_p1 - 1; // plus 1 based
			r_fill_state.transform_extra_command_number_p1 = 0;
			r_fill_state.extra_matrix_sent = true;

			// send the extra to the GPU in a batch
			r_fill_state.curr_batch = _batch_request_new();
			r_fill_state.curr_batch->type = RasterizerStorageCommon::BT_DEFAULT;
			r_fill_state.curr_batch->first_command = extra_command;
			r_fill_state.curr_batch->num_commands = 1;

			// revert to the original transform mode
			// e.g. go back to NONE if we were in hardware transform mode
			r_fill_state.transform_mode = r_fill_state.orig_transform_mode;

			// reset the original transform if we are going back to software mode,
			// because the extra is now done on the GPU...
			// (any subsequent extras are sent directly to the GPU, no deferring)
			if (r_fill_state.orig_transform_mode != TM_NONE) {
				r_fill_state.transform_combined = p_item.final_transform;
			}

			// can possibly combine batch with the next one in some cases
			// this is more efficient than having an extra batch especially for the extra
			if ((extra_command + 1) == p_command_num) {
				r_fill_state.curr_batch->num_commands = 2;
				return;
			}
		}

		// start default batch
		r_fill_state.curr_batch = _batch_request_new();
		r_fill_state.curr_batch->type = RasterizerStorageCommon::BT_DEFAULT;
		r_fill_state.curr_batch->first_command = p_command_num;
		r_fill_state.curr_batch->num_commands = 1;
	}
}

PREAMBLE(int)::_batch_find_or_create_tex(const RID &p_texture, const RID &p_normal, bool p_tile, int p_previous_match) {
	// optimization .. in 99% cases the last matched value will be the same, so no need to traverse the list
	if (p_previous_match > 0) // if it is zero, it will get hit first in the linear search anyway
	{
		const BatchTex &batch_texture = bdata.batch_textures[p_previous_match];

		// note for future reference, if RID implementation changes, this could become more expensive
		if ((batch_texture.RID_texture == p_texture) && (batch_texture.RID_normal == p_normal)) {
			// tiling mode must also match
			bool tiles = batch_texture.tile_mode != BatchTex::TILE_OFF;

			if (tiles == p_tile)
				// match!
				return p_previous_match;
		}
	}

	// not the previous match .. we will do a linear search ... slower, but should happen
	// not very often except with non-batchable runs, which are going to be slow anyway
	// n.b. could possibly be replaced later by a fast hash table
	for (int n = 0; n < bdata.batch_textures.size(); n++) {
		const BatchTex &batch_texture = bdata.batch_textures[n];
		if ((batch_texture.RID_texture == p_texture) && (batch_texture.RID_normal == p_normal)) {
			// tiling mode must also match
			bool tiles = batch_texture.tile_mode != BatchTex::TILE_OFF;

			if (tiles == p_tile)
				// match!
				return n;
		}
	}

	// pushing back from local variable .. not ideal but has to use a Vector because non pod
	// due to RIDs
	BatchTex new_batch_tex;
	new_batch_tex.RID_texture = p_texture;
	new_batch_tex.RID_normal = p_normal;

	// get the texture
	typename T_STORAGE::Texture *texture = _get_canvas_texture(p_texture);

	if (texture) {
		// special case, there can be textures with no width or height
		int w = texture->width;
		int h = texture->height;

		if (!w || !h) {
			w = 1;
			h = 1;
		}

		new_batch_tex.tex_pixel_size.x = 1.0 / w;
		new_batch_tex.tex_pixel_size.y = 1.0 / h;
		new_batch_tex.flags = texture->flags;
	} else {
		// maybe doesn't need doing...
		new_batch_tex.tex_pixel_size.x = 1.0f;
		new_batch_tex.tex_pixel_size.y = 1.0f;
		new_batch_tex.flags = 0;
	}

	if (p_tile) {
		if (texture) {
			// default
			new_batch_tex.tile_mode = BatchTex::TILE_NORMAL;

			// no hardware support for non power of 2 tiling
			if (!get_storage()->config.support_npot_repeat_mipmap) {
				if (next_power_of_2(texture->alloc_width) != (unsigned int)texture->alloc_width && next_power_of_2(texture->alloc_height) != (unsigned int)texture->alloc_height) {
					new_batch_tex.tile_mode = BatchTex::TILE_FORCE_REPEAT;
				}
			}
		} else {
			// this should not happen?
			new_batch_tex.tile_mode = BatchTex::TILE_OFF;
		}
	} else {
		new_batch_tex.tile_mode = BatchTex::TILE_OFF;
	}

	// push back
	bdata.batch_textures.push_back(new_batch_tex);

	return bdata.batch_textures.size() - 1;
}

PREAMBLE(void)::batch_constructor() {
	bdata.settings_use_batching = false;

#ifdef GLES_OVER_GL
	use_nvidia_rect_workaround = GLOBAL_GET("rendering/quality/2d/use_nvidia_rect_flicker_workaround");
#else
	// Not needed (a priori) on GLES devices
	use_nvidia_rect_workaround = false;
#endif
}

PREAMBLE(void)::batch_initialize() {
#define BATCHING_LOAD_PROJECT_SETTINGS

#ifdef BATCHING_LOAD_PROJECT_SETTINGS
	bdata.settings_use_batching = GLOBAL_GET("rendering/batching/options/use_batching");
	bdata.settings_max_join_item_commands = GLOBAL_GET("rendering/batching/parameters/max_join_item_commands");
	bdata.settings_colored_vertex_format_threshold = GLOBAL_GET("rendering/batching/parameters/colored_vertex_format_threshold");
	bdata.settings_item_reordering_lookahead = GLOBAL_GET("rendering/batching/parameters/item_reordering_lookahead");
	bdata.settings_light_max_join_items = GLOBAL_GET("rendering/batching/lights/max_join_items");
	bdata.settings_use_single_rect_fallback = GLOBAL_GET("rendering/batching/options/single_rect_fallback");
	bdata.settings_use_software_skinning = GLOBAL_GET("rendering/quality/2d/use_software_skinning");
	bdata.settings_ninepatch_mode = GLOBAL_GET("rendering/quality/2d/ninepatch_mode");

	// alternatively only enable uv contract if pixel snap in use,
	// but with this enable bool, it should not be necessary
	bdata.settings_uv_contract = GLOBAL_GET("rendering/batching/precision/uv_contract");
	bdata.settings_uv_contract_amount = (float)GLOBAL_GET("rendering/batching/precision/uv_contract_amount") / 1000000.0f;

	// we can use the threshold to determine whether to turn scissoring off or on
	bdata.settings_scissor_threshold = GLOBAL_GET("rendering/batching/lights/scissor_area_threshold");
#endif

	if (bdata.settings_scissor_threshold > 0.999f) {
		bdata.settings_scissor_lights = false;
	} else {
		bdata.settings_scissor_lights = true;

		// apply power of 4 relationship for the area, as most of the important changes
		// will be happening at low values of scissor threshold
		bdata.settings_scissor_threshold *= bdata.settings_scissor_threshold;
		bdata.settings_scissor_threshold *= bdata.settings_scissor_threshold;
	}

	// The sweet spot on my desktop for cache is actually smaller than the max, and this
	// is the default. This saves memory too so we will use it for now, needs testing to see whether this varies according
	// to device / platform.
#ifdef BATCHING_LOAD_PROJECT_SETTINGS
	bdata.settings_batch_buffer_num_verts = GLOBAL_GET("rendering/batching/parameters/batch_buffer_size");

	// override the use_batching setting in the editor
	// (note that if the editor can't start, you can't change the use_batching project setting!)
	if (Engine::get_singleton()->is_editor_hint()) {
		bool use_in_editor = GLOBAL_GET("rendering/batching/options/use_batching_in_editor");
		bdata.settings_use_batching = use_in_editor;

		// fix some settings in the editor, as the performance not worth the risk
		bdata.settings_use_single_rect_fallback = false;
	}
#endif

	// if we are using batching, we will purposefully disable the nvidia workaround.
	// This is because the only reason to use the single rect fallback is the approx 2x speed
	// of the uniform drawing technique. If we used nvidia workaround, speed would be
	// approx equal to the batcher drawing technique (indexed primitive + VB).
	if (bdata.settings_use_batching) {
		use_nvidia_rect_workaround = false;
	}

	// For debugging, if flash is set in project settings, it will flash on alternate frames
	// between the non-batched renderer and the batched renderer,
	// in order to find regressions.
	// This should not be used except during development.
	// make a note of the original choice in case we are flashing on and off the batching
	bdata.settings_use_batching_original_choice = bdata.settings_use_batching;

#ifdef BATCHING_LOAD_PROJECT_SETTINGS
	bdata.settings_flash_batching = GLOBAL_GET("rendering/batching/debug/flash_batching");
#endif
	if (!bdata.settings_use_batching) {
		// no flash when batching turned off
		bdata.settings_flash_batching = false;
	}

	// frame diagnosis. print out the batches every nth frame
	bdata.settings_diagnose_frame = false;
	if (!Engine::get_singleton()->is_editor_hint() && bdata.settings_use_batching) {
#ifdef BATCHING_LOAD_PROJECT_SETTINGS
		bdata.settings_diagnose_frame = GLOBAL_GET("rendering/batching/debug/diagnose_frame");
#endif
	}

	// the maximum num quads in a batch is limited by GLES2. We can have only 16 bit indices,
	// which means we can address a vertex buffer of max size 65535. 4 vertices are needed per quad.

	// Note this determines the memory use by the vertex buffer vector. max quads (65536/4)-1
	// but can be reduced to save memory if really required (will result in more batches though)
	const int max_possible_quads = (65536 / 4) - 1;
	const int min_possible_quads = 8; // some reasonable small value

	// value from project settings
	int max_quads = bdata.settings_batch_buffer_num_verts / 4;

	// sanity checks
	max_quads = CLAMP(max_quads, min_possible_quads, max_possible_quads);
	bdata.settings_max_join_item_commands = CLAMP(bdata.settings_max_join_item_commands, 0, 65535);
	bdata.settings_colored_vertex_format_threshold = CLAMP(bdata.settings_colored_vertex_format_threshold, 0.0f, 1.0f);
	bdata.settings_scissor_threshold = CLAMP(bdata.settings_scissor_threshold, 0.0f, 1.0f);
	bdata.settings_light_max_join_items = CLAMP(bdata.settings_light_max_join_items, 0, 65535);
	bdata.settings_item_reordering_lookahead = CLAMP(bdata.settings_item_reordering_lookahead, 0, 65535);

	// allow user to override the api usage techniques using project settings
	//	bdata.buffer_mode_batch_upload_send_null = GLOBAL_GET("rendering/options/api_usage_batching/send_null");
	//	bdata.buffer_mode_batch_upload_flag_stream = GLOBAL_GET("rendering/options/api_usage_batching/flag_stream");

	// for debug purposes, output a string with the batching options
	String batching_options_string = "OpenGL ES Batching: ";
	if (bdata.settings_use_batching) {
		batching_options_string += "ON";

		if (OS::get_singleton()->is_stdout_verbose()) {
			batching_options_string += "\n\tOPTIONS\n";
			batching_options_string += "\tmax_join_item_commands " + itos(bdata.settings_max_join_item_commands) + "\n";
			batching_options_string += "\tcolored_vertex_format_threshold " + String(Variant(bdata.settings_colored_vertex_format_threshold)) + "\n";
			batching_options_string += "\tbatch_buffer_size " + itos(bdata.settings_batch_buffer_num_verts) + "\n";
			batching_options_string += "\tlight_scissor_area_threshold " + String(Variant(bdata.settings_scissor_threshold)) + "\n";

			batching_options_string += "\titem_reordering_lookahead " + itos(bdata.settings_item_reordering_lookahead) + "\n";
			batching_options_string += "\tlight_max_join_items " + itos(bdata.settings_light_max_join_items) + "\n";
			batching_options_string += "\tsingle_rect_fallback " + String(Variant(bdata.settings_use_single_rect_fallback)) + "\n";

			batching_options_string += "\tdebug_flash " + String(Variant(bdata.settings_flash_batching)) + "\n";
			batching_options_string += "\tdiagnose_frame " + String(Variant(bdata.settings_diagnose_frame));
		}

		print_line(batching_options_string);
	}

	// special case, for colored vertex format threshold.
	// as the comparison is >=, we want to be able to totally turn on or off
	// conversion to colored vertex format at the extremes, so we will force
	// 1.0 to be just above 1.0
	if (bdata.settings_colored_vertex_format_threshold > 0.995f) {
		bdata.settings_colored_vertex_format_threshold = 1.01f;
	}

	// save memory when batching off
	if (!bdata.settings_use_batching) {
		max_quads = 0;
	}

	uint32_t sizeof_batch_vert = sizeof(BatchVertex);

	bdata.max_quads = max_quads;

	// 4 verts per quad
	bdata.vertex_buffer_size_units = max_quads * 4;

	// the index buffer can be longer than 65535, only the indices need to be within this range
	bdata.index_buffer_size_units = max_quads * 6;

	const int max_verts = bdata.vertex_buffer_size_units;

	// this comes out at approx 64K for non-colored vertex buffer, and 128K for colored vertex buffer
	bdata.vertex_buffer_size_bytes = max_verts * sizeof_batch_vert;
	bdata.index_buffer_size_bytes = bdata.index_buffer_size_units * 2; // 16 bit inds

	// create equal number of normal and (max) unit sized verts (as the normal may need to be translated to a larger FVF)
	bdata.vertices.create(max_verts); // 512k
	bdata.unit_vertices.create(max_verts, sizeof(BatchVertexLarge));

	// extra data per vert needed for larger FVFs
	bdata.light_angles.create(max_verts);
	bdata.vertex_colors.create(max_verts);
	bdata.vertex_modulates.create(max_verts);
	bdata.vertex_transforms.create(max_verts);

	// num batches will be auto increased dynamically if required
	bdata.batches.create(1024);
	bdata.batches_temp.create(bdata.batches.max_size());

	// batch textures can also be increased dynamically
	bdata.batch_textures.create(32);
}

PREAMBLE(bool)::_light_scissor_begin(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect) const {
	float area_item = p_item_rect.size.x * p_item_rect.size.y; // double check these are always positive

	// quick reject .. the area of pixels saved can never be more than the area of the item
	if (area_item < bdata.scissor_threshold_area) {
		return false;
	}

	Rect2 cliprect;
	if (!_light_find_intersection(p_item_rect, p_light_xform, p_light_rect, cliprect)) {
		// should not really occur .. but just in case
		cliprect = Rect2(0, 0, 0, 0);
	} else {
		// some conditions not to scissor
		// determine the area (fill rate) that will be saved
		float area_cliprect = cliprect.size.x * cliprect.size.y;
		float area_saved = area_item - area_cliprect;

		// if area saved is too small, don't scissor
		if (area_saved < bdata.scissor_threshold_area) {
			return false;
		}
	}

	int rh = get_storage()->frame.current_rt->height;

	int y = rh - (cliprect.position.y + cliprect.size.y);
	get_this()->gl_enable_scissor(cliprect.position.x, y, cliprect.size.width, cliprect.size.height);

	return true;
}

PREAMBLE(bool)::_light_find_intersection(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect, Rect2 &r_cliprect) const {
	// transform light to world space (note this is done in the earlier intersection test, so could
	// be made more efficient)
	Vector2 pts[4] = {
		p_light_xform.xform(p_light_rect.position),
		p_light_xform.xform(Vector2(p_light_rect.position.x + p_light_rect.size.x, p_light_rect.position.y)),
		p_light_xform.xform(Vector2(p_light_rect.position.x, p_light_rect.position.y + p_light_rect.size.y)),
		p_light_xform.xform(Vector2(p_light_rect.position.x + p_light_rect.size.x, p_light_rect.position.y + p_light_rect.size.y)),
	};

	// calculate the light bound rect in world space
	Rect2 lrect(pts[0].x, pts[0].y, 0, 0);
	for (int n = 1; n < 4; n++) {
		lrect.expand_to(pts[n]);
	}

	// intersection between the 2 rects
	// they should probably always intersect, because of earlier check, but just in case...
	if (!p_item_rect.intersects(lrect))
		return false;

	// note this does almost the same as Rect2.clip but slightly more efficient for our use case
	r_cliprect.position.x = MAX(p_item_rect.position.x, lrect.position.x);
	r_cliprect.position.y = MAX(p_item_rect.position.y, lrect.position.y);

	Point2 item_rect_end = p_item_rect.position + p_item_rect.size;
	Point2 lrect_end = lrect.position + lrect.size;

	r_cliprect.size.x = MIN(item_rect_end.x, lrect_end.x) - r_cliprect.position.x;
	r_cliprect.size.y = MIN(item_rect_end.y, lrect_end.y) - r_cliprect.position.y;

	return true;
}

PREAMBLE(void)::_calculate_scissor_threshold_area() {
	if (!bdata.settings_scissor_lights) {
		return;
	}

	// scissor area threshold is 0.0 to 1.0 in the settings for ease of use.
	// we need to translate to an absolute area to determine quickly whether
	// to scissor.
	if (bdata.settings_scissor_threshold < 0.0001f) {
		bdata.scissor_threshold_area = -1.0f; // will always pass
	} else {
		// in pixels
		int w = get_storage()->frame.current_rt->width;
		int h = get_storage()->frame.current_rt->height;

		int screen_area = w * h;

		bdata.scissor_threshold_area = bdata.settings_scissor_threshold * screen_area;
	}
}

PREAMBLE(void)::render_joined_item_commands(const BItemJoined &p_bij, RendererCanvasRender::Item *p_current_clip, bool &r_reclip, typename T_STORAGE::Material *p_material, bool p_lit) {
	RendererCanvasRender::Item *item = 0;
	RendererCanvasRender::Item *first_item = bdata.item_refs[p_bij.first_item_ref].item;

	// fill_state and bdata have once off setup per joined item, and a smaller reset on flush
	FillState fill_state;
	fill_state.reset_joined_item(p_bij.use_hardware_transform());

	bdata.reset_joined_item();

	// should this joined item be using large FVF?
	if (p_bij.flags & RasterizerStorageCommon::USE_MODULATE_FVF) {
		bdata.use_modulate = true;
		bdata.fvf = RasterizerStorageCommon::FVF_MODULATED;
	}
	if (p_bij.flags & RasterizerStorageCommon::USE_LARGE_FVF) {
		bdata.use_modulate = true;
		bdata.use_large_verts = true;
		bdata.fvf = RasterizerStorageCommon::FVF_LARGE;
	}

	// in the special case of custom shaders that read from VERTEX (i.e. vertex position)
	// we want to disable software transform of extra matrix
	if (bdata.joined_item_batch_flags & RasterizerStorageCommon::PREVENT_VERTEX_BAKING) {
		fill_state.extra_matrix_sent = true;
	}

	for (unsigned int i = 0; i < p_bij.num_item_refs; i++) {
		const BItemRef &ref = bdata.item_refs[p_bij.first_item_ref + i];
		item = ref.item;

		if (!p_lit) {
			// if not lit we use the complex calculated final modulate
			fill_state.final_modulate = ref.final_modulate;
		} else {
			// if lit we ignore canvas modulate and just use the item modulate
			fill_state.final_modulate = item->final_modulate;
		}

		// ONCE OFF fill state setup, that will be retained over multiple calls to
		// prefill_joined_item()
		fill_state.transform_combined = item->final_transform;

		// decide the initial transform mode, and make a backup
		// in orig_transform_mode in case we need to switch back
		if (!fill_state.use_hardware_transform) {
			fill_state.transform_mode = _find_transform_mode(fill_state.transform_combined);
		} else {
			fill_state.transform_mode = TM_NONE;
		}
		fill_state.orig_transform_mode = fill_state.transform_mode;

		// keep track of when we added an extra matrix
		// so we can defer sending until we see a default command
		fill_state.transform_extra_command_number_p1 = 0;

		RendererCanvasRender::Item::Command *current_command = item->commands;
		while (current_command) {
			// fill as many batches as possible (until all done, or the vertex buffer is full)
			bool bFull = get_this()->prefill_joined_item(fill_state, current_command, item, p_current_clip, r_reclip, p_material);

			if (bFull) {
				// always pass first item (commands for default are always first item)
				flush_render_batches(first_item, p_current_clip, r_reclip, p_material, fill_state.sequence_batch_type_flags);

				// zero all the batch data ready for a new run
				bdata.reset_flush();

				// don't zero all the fill state, some may need to be preserved
				fill_state.reset_flush();
			}
		}
	}

	// flush if any left
	flush_render_batches(first_item, p_current_clip, r_reclip, p_material, fill_state.sequence_batch_type_flags);

	// zero all the batch data ready for a new run
	bdata.reset_flush();
}

PREAMBLE(void)::_legacy_canvas_item_render_commands(RendererCanvasRender::Item *p_item, RendererCanvasRender::Item *p_current_clip, bool &r_reclip, typename T_STORAGE::Material *p_material) {
	// reuse the same list each time to prevent needless dynamic allocations
	unsigned int command_count = godot4_commands_to_vector(p_item->commands, bdata.command_shortlist);
	RendererCanvasRender::Item::Command *const *commands = nullptr;
	if (command_count) {
		commands = &bdata.command_shortlist[0];
	}

	// legacy .. just create one massive batch and render everything as before
	bdata.batches.reset();
	Batch *batch = _batch_request_new();
	batch->type = RasterizerStorageCommon::BT_DEFAULT;
	batch->num_commands = command_count;

	get_this()->render_batches(commands, p_current_clip, r_reclip, p_material);
	bdata.reset_flush();
}

PREAMBLE(void)::record_items(RendererCanvasRender::Item *p_item_list, int p_z) {
	while (p_item_list) {
		BSortItem *s = bdata.sort_items.request_with_grow();

		s->item = p_item_list;
		s->z_index = p_z;

		p_item_list = p_item_list->next;
	}
}

PREAMBLE(void)::join_sorted_items() {
}

PREAMBLE(void)::_software_transform_vertex(BatchVector2 &r_v, const Transform2D &p_tr) const {
	Vector2 vc(r_v.x, r_v.y);
	vc = p_tr.xform(vc);
	r_v.set(vc);
}

PREAMBLE(void)::_software_transform_vertex(Vector2 &r_v, const Transform2D &p_tr) const {
	r_v = p_tr.xform(r_v);
}

PREAMBLE(void)::_translate_batches_to_vertex_colored_FVF() {
	// zeros the size and sets up how big each unit is
	bdata.unit_vertices.prepare(sizeof(BatchVertexColored));

	const BatchColor *source_vertex_colors = &bdata.vertex_colors[0];
	RAST_DEBUG_ASSERT(bdata.vertex_colors.size() == bdata.vertices.size());

	int num_verts = bdata.vertices.size();

	for (int n = 0; n < num_verts; n++) {
		const BatchVertex &bv = bdata.vertices[n];

		BatchVertexColored *cv = (BatchVertexColored *)bdata.unit_vertices.request();

		cv->pos = bv.pos;
		cv->uv = bv.uv;
		cv->col = *source_vertex_colors++;
	}
}

// Translation always involved adding color to the FVF, which enables
// joining of batches that have different colors.
// There is a trade off. Non colored verts are smaller so work faster, but
// there comes a point where it is better to just use colored verts to avoid lots of
// batches.
// In addition this can optionally add light angles to the FVF, necessary for normal mapping.
T_PREAMBLE
template <class BATCH_VERTEX_TYPE, bool INCLUDE_LIGHT_ANGLES, bool INCLUDE_MODULATE, bool INCLUDE_LARGE>
void C_PREAMBLE::_translate_batches_to_larger_FVF(uint32_t p_sequence_batch_type_flags) {
	bool include_poly_color = false;

	// we ONLY want to include the color verts in translation when using polys,
	// as rects do not write vertex colors, only colors per batch.
	if (p_sequence_batch_type_flags & RasterizerStorageCommon::BTF_POLY) {
		include_poly_color = INCLUDE_LIGHT_ANGLES | INCLUDE_MODULATE | INCLUDE_LARGE;
	}

	// zeros the size and sets up how big each unit is
	bdata.unit_vertices.prepare(sizeof(BATCH_VERTEX_TYPE));
	bdata.batches_temp.reset();

	// As the vertices_colored and batches_temp are 'mirrors' of the non-colored version,
	// the sizes should be equal, and allocations should never fail. Hence the use of debug
	// asserts to check program flow, these should not occur at runtime unless the allocation
	// code has been altered.
	RAST_DEBUG_ASSERT(bdata.unit_vertices.max_size() == bdata.vertices.max_size());
	RAST_DEBUG_ASSERT(bdata.batches_temp.max_size() == bdata.batches.max_size());

	Color curr_col(-1.0f, -1.0f, -1.0f, -1.0f);

	Batch *dest_batch = nullptr;

	const BatchColor *source_vertex_colors = &bdata.vertex_colors[0];
	const float *source_light_angles = &bdata.light_angles[0];
	const BatchColor *source_vertex_modulates = &bdata.vertex_modulates[0];
	const BatchTransform *source_vertex_transforms = &bdata.vertex_transforms[0];

	// translate the batches into vertex colored batches
	for (int n = 0; n < bdata.batches.size(); n++) {
		const Batch &source_batch = bdata.batches[n];

		// does source batch use light angles?
		const BatchTex &btex = bdata.batch_textures[source_batch.batch_texture_id];
		bool source_batch_uses_light_angles = btex.RID_normal != RID();

		bool needs_new_batch = true;

		if (dest_batch) {
			if (dest_batch->type == source_batch.type) {
				if (source_batch.type == RasterizerStorageCommon::BT_RECT) {
					if (dest_batch->batch_texture_id == source_batch.batch_texture_id) {
						// add to previous batch
						dest_batch->num_commands += source_batch.num_commands;
						needs_new_batch = false;

						// create the colored verts (only if not default)
						//int first_vert = source_batch.first_quad * 4;
						//int end_vert = 4 * (source_batch.first_quad + source_batch.num_commands);
						int first_vert = source_batch.first_vert;
						int end_vert = first_vert + (4 * source_batch.num_commands);

						for (int v = first_vert; v < end_vert; v++) {
							RAST_DEV_DEBUG_ASSERT(bdata.vertices.size());
							const BatchVertex &bv = bdata.vertices[v];
							BATCH_VERTEX_TYPE *cv = (BATCH_VERTEX_TYPE *)bdata.unit_vertices.request();
							RAST_DEBUG_ASSERT(cv);
							cv->pos = bv.pos;
							cv->uv = bv.uv;
							cv->col = source_batch.color;

							if (INCLUDE_LIGHT_ANGLES) {
								RAST_DEV_DEBUG_ASSERT(bdata.light_angles.size());
								// this is required to allow compilation with non light angle vertex.
								// it should be compiled out.
								BatchVertexLightAngled *lv = (BatchVertexLightAngled *)cv;
								if (source_batch_uses_light_angles)
									lv->light_angle = *source_light_angles++;
								else
									lv->light_angle = 0.0f; // dummy, unused in vertex shader (could possibly be left uninitialized, but probably bad idea)
							} // if including light angles

							if (INCLUDE_MODULATE) {
								RAST_DEV_DEBUG_ASSERT(bdata.vertex_modulates.size());
								BatchVertexModulated *mv = (BatchVertexModulated *)cv;
								mv->modulate = *source_vertex_modulates++;
							} // including modulate

							if (INCLUDE_LARGE) {
								RAST_DEV_DEBUG_ASSERT(bdata.vertex_transforms.size());
								BatchVertexLarge *lv = (BatchVertexLarge *)cv;
								lv->transform = *source_vertex_transforms++;
							} // if including large
						}
					} // textures match
				} else {
					// default
					// we can still join, but only under special circumstances
					// does this ever happen? not sure at this stage, but left for future expansion
					uint32_t source_last_command = source_batch.first_command + source_batch.num_commands;
					if (source_last_command == dest_batch->first_command) {
						dest_batch->num_commands += source_batch.num_commands;
						needs_new_batch = false;
					} // if the commands line up exactly
				}
			} // if both batches are the same type

		} // if dest batch is valid

		if (needs_new_batch) {
			dest_batch = bdata.batches_temp.request();
			RAST_DEBUG_ASSERT(dest_batch);

			*dest_batch = source_batch;

			// create the colored verts (only if not default)
			if (source_batch.type != RasterizerStorageCommon::BT_DEFAULT) {
				//					int first_vert = source_batch.first_quad * 4;
				//					int end_vert = 4 * (source_batch.first_quad + source_batch.num_commands);
				int first_vert = source_batch.first_vert;
				int end_vert = first_vert + (4 * source_batch.num_commands);

				for (int v = first_vert; v < end_vert; v++) {
					RAST_DEV_DEBUG_ASSERT(bdata.vertices.size());
					const BatchVertex &bv = bdata.vertices[v];
					BATCH_VERTEX_TYPE *cv = (BATCH_VERTEX_TYPE *)bdata.unit_vertices.request();
					RAST_DEBUG_ASSERT(cv);
					cv->pos = bv.pos;
					cv->uv = bv.uv;

					// polys are special, they can have per vertex colors
					if (!include_poly_color) {
						cv->col = source_batch.color;
					} else {
						RAST_DEV_DEBUG_ASSERT(bdata.vertex_colors.size());
						cv->col = *source_vertex_colors++;
					}

					if (INCLUDE_LIGHT_ANGLES) {
						RAST_DEV_DEBUG_ASSERT(bdata.light_angles.size());
						// this is required to allow compilation with non light angle vertex.
						// it should be compiled out.
						BatchVertexLightAngled *lv = (BatchVertexLightAngled *)cv;
						if (source_batch_uses_light_angles)
							lv->light_angle = *source_light_angles++;
						else
							lv->light_angle = 0.0f; // dummy, unused in vertex shader (could possibly be left uninitialized, but probably bad idea)
					} // if using light angles

					if (INCLUDE_MODULATE) {
						RAST_DEV_DEBUG_ASSERT(bdata.vertex_modulates.size());
						BatchVertexModulated *mv = (BatchVertexModulated *)cv;
						mv->modulate = *source_vertex_modulates++;
					} // including modulate

					if (INCLUDE_LARGE) {
						RAST_DEV_DEBUG_ASSERT(bdata.vertex_transforms.size());
						BatchVertexLarge *lv = (BatchVertexLarge *)cv;
						lv->transform = *source_vertex_transforms++;
					} // if including large
				}
			}
		}
	}

	// copy the temporary batches to the master batch list (this could be avoided but it makes the code cleaner)
	bdata.batches.copy_from(bdata.batches_temp);
}

PREAMBLE(bool)::_disallow_item_join_if_batch_types_too_different(RenderItemState &r_ris, uint32_t btf_allowed) {
	r_ris.joined_item_batch_type_flags_curr |= btf_allowed;

	bool disallow = false;

	if (r_ris.joined_item_batch_type_flags_prev & (~btf_allowed))
		disallow = true;

	return disallow;
}

#undef PREAMBLE
#undef T_PREAMBLE
#undef C_PREAMBLE

#endif // RASTERIZER_CANVAS_BATCHER_H
