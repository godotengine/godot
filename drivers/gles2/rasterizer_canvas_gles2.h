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
		bool operator==(const BatchColor &p_c) const {
			return (r == p_c.r) && (g == p_c.g) && (b == p_c.b) && (a == p_c.a);
		}
		bool operator!=(const BatchColor &p_c) const { return (*this == p_c) == false; }
		bool equals(const Color &p_c) const {
			return (r == p_c.r) && (g == p_c.g) && (b == p_c.b) && (a == p_c.a);
		}
		const float *get_data() const { return &r; }
		String to_string() const;
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

	struct BatchVertexLightAngled : public BatchVertexColored {
		// must be pod
		float light_angle;
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
		uint32_t flags;
	};

	// items in a list to be sorted prior to joining
	struct BSortItem {
		// have a function to keep as pod, rather than operator
		void assign(const BSortItem &o) {
			item = o.item;
			z_index = o.z_index;
		}
		Item *item;
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

		// these are defined in RasterizerStorageGLES2::Shader::CanvasItem::BatchFlags
		uint16_t flags;

		// we are always splitting items with lots of commands,
		// and items with unhandled primitives (default)
		bool use_hardware_transform() const { return num_item_refs == 1; }
	};

	struct BItemRef {
		Item *item;
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
		BatchData();
		void reset_flush() {
			batches.reset();
			batch_textures.reset();

			vertices.reset();
			light_angles.reset();

			total_quads = 0;
			total_color_changes = 0;
			use_light_angles = false;
		}

		GLuint gl_vertex_buffer;
		GLuint gl_index_buffer;

		uint32_t max_quads;
		uint32_t vertex_buffer_size_units;
		uint32_t vertex_buffer_size_bytes;
		uint32_t index_buffer_size_units;
		uint32_t index_buffer_size_bytes;

		// small vertex FVF type - pos and UV.
		// This will always be written to initially, but can be translated
		// to larger FVFs if necessary.
		RasterizerArrayGLES2<BatchVertex> vertices;

		// extra data which can be stored during prefilling, for later translation to larger FVFs
		RasterizerArrayGLES2<float> light_angles;

		// instead of having a different buffer for each vertex FVF type
		// we have a special array big enough for the biggest FVF
		// which can have a changeable unit size, and reuse it.
		RasterizerUnitArrayGLES2 unit_vertices;

		RasterizerArrayGLES2<Batch> batches;
		RasterizerArrayGLES2<Batch> batches_temp; // used for translating to colored vertex batches
		RasterizerArray_non_pod_GLES2<BatchTex> batch_textures; // the only reason this is non-POD is because of RIDs

		// flexible vertex format.
		// all verts have pos and UV.
		// some have color, some light angles etc.
		bool use_colored_vertices;
		bool use_light_angles;

		RasterizerArrayGLES2<BItemJoined> items_joined;
		RasterizerArrayGLES2<BItemRef> item_refs;

		// items are sorted prior to joining
		RasterizerArrayGLES2<BSortItem> sort_items;

		// counts
		int total_quads;

		// we keep a record of how many color changes caused new batches
		// if the colors are causing an excessive number of batches, we switch
		// to alternate batching method and add color to the vertex format.
		int total_color_changes;

		// if the shader is using MODULATE, we prevent baking color so the final_modulate can
		// be read in the shader.
		// if the shader is reading VERTEX, we prevent baking vertex positions with extra matrices etc
		// to prevent the read position being incorrect.
		// These flags are defined in RasterizerStorageGLES2::Shader::CanvasItem::BatchFlags
		uint32_t joined_item_batch_flags;

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
		int settings_light_max_join_items;

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

	struct RenderItemState {
		RenderItemState() { reset(); }
		void reset();
		Item *current_clip;
		RasterizerStorageGLES2::Shader *shader_cache;
		bool rebind_shader;
		bool prev_use_skeleton;
		int last_blend_mode;
		RID canvas_last_material;
		Color final_modulate;

		// used for joining items only
		BItemJoined *joined_item;
		bool join_batch_break;
		BLightRegion light_region;

		// 'item group' is data over a single call to canvas_render_items
		int item_group_z;
		Color item_group_modulate;
		Light *item_group_light;
		Transform2D item_group_base_transform;
	} _render_item_state;

	struct FillState {
		void reset() {
			// don't reset members that need to be preserved after flushing
			// half way through a list of commands
			curr_batch = 0;
			batch_tex_id = -1;
			texpixel_size = Vector2(1, 1);
			contract_uvs = false;
		}
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

public:
	virtual void canvas_render_items_begin(const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	virtual void canvas_render_items_end();
	virtual void canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	virtual void canvas_begin();
	virtual void canvas_end();

private:
	// legacy codepath .. to remove after testing
	void _canvas_render_item(Item *p_ci, RenderItemState &r_ris);
	void _canvas_item_render_commands(Item *p_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	// high level batch funcs
	void canvas_render_items_implementation(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	void render_joined_item(const BItemJoined &p_bij, RenderItemState &r_ris);
	void record_items(Item *p_item_list, int p_z);
	void join_items(Item *p_item_list, int p_z);
	void join_sorted_items();
	bool try_join_item(Item *p_ci, RenderItemState &r_ris, bool &r_batch_break);
	void render_joined_item_commands(const BItemJoined &p_bij, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material, bool p_lit);
	void render_batches(Item::Command *const *p_commands, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	bool prefill_joined_item(FillState &r_fill_state, int &r_command_start, Item *p_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	void flush_render_batches(Item *p_first_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	// low level batch funcs
	int _batch_find_or_create_tex(const RID &p_texture, const RID &p_normal, bool p_tile, int p_previous_match);
	RasterizerStorageGLES2::Texture *_get_canvas_texture(const RID &p_texture) const;
	void _batch_upload_buffers();
	void _batch_render_rects(const Batch &p_batch, RasterizerStorageGLES2::Material *p_material);
	BatchVertex *_batch_vertex_request_new() { return bdata.vertices.request(); }
	Batch *_batch_request_new(bool p_blank = true);

	bool _detect_batch_break(Item *p_ci);
	void _software_transform_vertex(BatchVector2 &r_v, const Transform2D &p_tr) const;
	void _software_transform_vertex(Vector2 &r_v, const Transform2D &p_tr) const;
	TransformMode _find_transform_mode(const Transform2D &p_tr) const;
	void _prefill_default_batch(FillState &r_fill_state, int p_command_num, const Item &p_item);

	// sorting
	void sort_items();
	bool sort_items_from(int p_start);
	bool _sort_items_match(const BSortItem &p_a, const BSortItem &p_b) const;

	// light scissoring
	bool _light_find_intersection(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect, Rect2 &r_cliprect) const;
	bool _light_scissor_begin(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect) const;
	void _calculate_scissor_threshold_area();

	// no need to compile these in in release, they are unneeded outside the editor and only add to executable size
#ifdef DEBUG_ENABLED
	void diagnose_batches(Item::Command *const *p_commands);
	String get_command_type_string(const Item::Command &p_command) const;
#endif

public:
	void initialize();
	RasterizerCanvasGLES2();

private:
	template <bool SEND_LIGHT_ANGLES>
	bool prefill_rect(Item::CommandRect *rect, FillState &r_fill_state, int &r_command_start, int command_num, int command_count, Item::Command *const *commands, Item *p_item, bool multiply_final_modulate);

	template <class BATCH_VERTEX_TYPE, bool INCLUDE_LIGHT_ANGLES>
	void _translate_batches_to_larger_FVF();
};

//////////////////////////////////////////////////////////////

// Default batches will not occur in software transform only items
// EXCEPT IN THE CASE OF SINGLE RECTS (and this may well not occur, check the logic in prefill_join_item TYPE_RECT)
// but can occur where transform commands have been sent during hardware batch
inline void RasterizerCanvasGLES2::_prefill_default_batch(FillState &r_fill_state, int p_command_num, const Item &p_item) {
	if (r_fill_state.curr_batch->type == Batch::BT_DEFAULT) {
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
			r_fill_state.curr_batch->type = Batch::BT_DEFAULT;
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
		r_fill_state.curr_batch->type = Batch::BT_DEFAULT;
		r_fill_state.curr_batch->first_command = p_command_num;
		r_fill_state.curr_batch->num_commands = 1;
	}
}

inline void RasterizerCanvasGLES2::_software_transform_vertex(BatchVector2 &r_v, const Transform2D &p_tr) const {
	Vector2 vc(r_v.x, r_v.y);
	vc = p_tr.xform(vc);
	r_v.set(vc);
}

inline void RasterizerCanvasGLES2::_software_transform_vertex(Vector2 &r_v, const Transform2D &p_tr) const {
	r_v = p_tr.xform(r_v);
}

inline RasterizerCanvasGLES2::TransformMode RasterizerCanvasGLES2::_find_transform_mode(const Transform2D &p_tr) const {
	// decided whether to do translate only for software transform
	if ((p_tr.elements[0].x == 1.0) &&
			(p_tr.elements[0].y == 0.0) &&
			(p_tr.elements[1].x == 0.0) &&
			(p_tr.elements[1].y == 1.0)) {
		return TM_TRANSLATE;
	}

	return TM_ALL;
}

inline bool RasterizerCanvasGLES2::_sort_items_match(const BSortItem &p_a, const BSortItem &p_b) const {
	const Item *a = p_a.item;
	const Item *b = p_b.item;

	if (b->commands.size() != 1)
		return false;

	// tested outside function
	//	if (a->commands.size() != 1)
	//		return false;

	const Item::Command &cb = *b->commands[0];
	if (cb.type != Item::Command::TYPE_RECT)
		return false;

	const Item::Command &ca = *a->commands[0];
	// tested outside function
	//	if (ca.type != Item::Command::TYPE_RECT)
	//		return false;

	const Item::CommandRect *rect_a = static_cast<const Item::CommandRect *>(&ca);
	const Item::CommandRect *rect_b = static_cast<const Item::CommandRect *>(&cb);

	if (rect_a->texture != rect_b->texture)
		return false;

	return true;
}

//////////////////////////////////////////////////////////////
// TEMPLATE FUNCS

// Translation always involved adding color to the FVF, which enables
// joining of batches that have different colors.
// There is a trade off. Non colored verts are smaller so work faster, but
// there comes a point where it is better to just use colored verts to avoid lots of
// batches.
// In addition this can optionally add light angles to the FVF, necessary for normal mapping.
template <class BATCH_VERTEX_TYPE, bool INCLUDE_LIGHT_ANGLES>
void RasterizerCanvasGLES2::_translate_batches_to_larger_FVF() {

	// zeros the size and sets up how big each unit is
	bdata.unit_vertices.prepare(sizeof(BATCH_VERTEX_TYPE));
	bdata.batches_temp.reset();

	// As the vertices_colored and batches_temp are 'mirrors' of the non-colored version,
	// the sizes should be equal, and allocations should never fail. Hence the use of debug
	// asserts to check program flow, these should not occur at runtime unless the allocation
	// code has been altered.
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
	CRASH_COND(bdata.unit_vertices.max_size() != bdata.vertices.max_size());
	CRASH_COND(bdata.batches_temp.max_size() != bdata.batches.max_size());
#endif

	Color curr_col(-1.0, -1.0, -1.0, -1.0);

	Batch *dest_batch = 0;

	const float *source_light_angles = &bdata.light_angles[0];

	// translate the batches into vertex colored batches
	for (int n = 0; n < bdata.batches.size(); n++) {
		const Batch &source_batch = bdata.batches[n];

		// does source batch use light angles?
		const BatchTex &btex = bdata.batch_textures[source_batch.batch_texture_id];
		bool source_batch_uses_light_angles = btex.RID_normal != RID();

		bool needs_new_batch = true;

		if (dest_batch) {
			if (dest_batch->type == source_batch.type) {
				if (source_batch.type == Batch::BT_RECT) {
					if (dest_batch->batch_texture_id == source_batch.batch_texture_id) {
						// add to previous batch
						dest_batch->num_commands += source_batch.num_commands;
						needs_new_batch = false;

						// create the colored verts (only if not default)
						int first_vert = source_batch.first_quad * 4;
						int end_vert = 4 * (source_batch.first_quad + source_batch.num_commands);

						for (int v = first_vert; v < end_vert; v++) {
							const BatchVertex &bv = bdata.vertices[v];
							BATCH_VERTEX_TYPE *cv = (BatchVertexLightAngled *)bdata.unit_vertices.request();
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
							CRASH_COND(!cv);
#endif
							cv->pos = bv.pos;
							cv->uv = bv.uv;
							cv->col = source_batch.color;

							if (INCLUDE_LIGHT_ANGLES) {
								// this is required to allow compilation with non light angle vertex.
								// it should be compiled out.
								BatchVertexLightAngled *lv = (BatchVertexLightAngled *)cv;
								if (source_batch_uses_light_angles)
									lv->light_angle = *source_light_angles++;
								else
									lv->light_angle = 0.0f; // dummy, unused in vertex shader (could possibly be left uninitialized, but probably bad idea)
							}
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
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
			CRASH_COND(!dest_batch);
#endif

			*dest_batch = source_batch;

			// create the colored verts (only if not default)
			if (source_batch.type != Batch::BT_DEFAULT) {
				int first_vert = source_batch.first_quad * 4;
				int end_vert = 4 * (source_batch.first_quad + source_batch.num_commands);

				for (int v = first_vert; v < end_vert; v++) {
					const BatchVertex &bv = bdata.vertices[v];
					BATCH_VERTEX_TYPE *cv = (BatchVertexLightAngled *)bdata.unit_vertices.request();
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
					CRASH_COND(!cv);
#endif
					cv->pos = bv.pos;
					cv->uv = bv.uv;
					cv->col = source_batch.color;

					if (INCLUDE_LIGHT_ANGLES) {
						// this is required to allow compilation with non light angle vertex.
						// it should be compiled out.
						BatchVertexLightAngled *lv = (BatchVertexLightAngled *)cv;
						if (source_batch_uses_light_angles)
							lv->light_angle = *source_light_angles++;
						else
							lv->light_angle = 0.0f; // dummy, unused in vertex shader (could possibly be left uninitialized, but probably bad idea)
					} // if using light angles
				}
			}
		}
	}

	// copy the temporary batches to the master batch list (this could be avoided but it makes the code cleaner)
	bdata.batches.copy_from(bdata.batches_temp);
}

// return true if buffer full up, else return false
template <bool SEND_LIGHT_ANGLES>
bool RasterizerCanvasGLES2::prefill_rect(Item::CommandRect *rect, FillState &r_fill_state, int &r_command_start, int command_num, int command_count, Item::Command *const *commands, Item *p_item, bool multiply_final_modulate) {
	bool change_batch = false;

	// conditions for creating a new batch
	if (r_fill_state.curr_batch->type != Batch::BT_RECT) {
		change_batch = true;

		// check for special case if there is only a single or small number of rects,
		// in which case we will use the legacy default rect renderer
		// because it is faster for single rects

		// we only want to do this if not a joined item with more than 1 item,
		// because joined items with more than 1, the command * will be incorrect
		// NOTE - this is assuming that use_hardware_transform means that it is a non-joined item!!
		// If that assumption is incorrect this will go horribly wrong.
		if (bdata.settings_use_single_rect_fallback && r_fill_state.use_hardware_transform) {
			bool is_single_rect = false;
			int command_num_next = command_num + 1;
			if (command_num_next < command_count) {
				Item::Command *command_next = commands[command_num_next];
				if ((command_next->type != Item::Command::TYPE_RECT) && (command_next->type != Item::Command::TYPE_TRANSFORM)) {
					is_single_rect = true;
				}
			} else {
				is_single_rect = true;
			}
			// if it is a rect on its own, do exactly the same as the default routine
			if (is_single_rect) {
				_prefill_default_batch(r_fill_state, command_num, *p_item);
				return false;
			}
		} // if use hardware transform
	}

	Color col = rect->modulate;
	if (multiply_final_modulate) {
		col *= r_fill_state.final_modulate;
	}

	// instead of doing all the texture preparation for EVERY rect,
	// we build a list of texture combinations and do this once off.
	// This means we have a potentially rather slow step to identify which texture combo
	// using the RIDs.
	int old_batch_tex_id = r_fill_state.batch_tex_id;
	r_fill_state.batch_tex_id = _batch_find_or_create_tex(rect->texture, rect->normal_map, rect->flags & CANVAS_RECT_TILE, old_batch_tex_id);

	//r_fill_state.use_light_angles = send_light_angles;
	if (SEND_LIGHT_ANGLES)
		bdata.use_light_angles = true;

	// try to create vertices BEFORE creating a batch,
	// because if the vertex buffer is full, we need to finish this
	// function, draw what we have so far, and then start a new set of batches

	// request FOUR vertices at a time, this is more efficient
	BatchVertex *bvs = bdata.vertices.request(4);
	if (!bvs) {
		// run out of space in the vertex buffer .. finish this function and draw what we have so far
		// return where we got to
		r_command_start = command_num;
		return true;
	}

	// conditions for creating a new batch
	if (old_batch_tex_id != r_fill_state.batch_tex_id) {
		change_batch = true;
	}

	// we need to treat color change separately because we need to count these
	// to decide whether to switch on the fly to colored vertices.
	if (!r_fill_state.curr_batch->color.equals(col)) {
		change_batch = true;
		bdata.total_color_changes++;
	}

	if (change_batch) {
		// put the tex pixel size  in a local (less verbose and can be a register)
		const BatchTex &batchtex = bdata.batch_textures[r_fill_state.batch_tex_id];
		batchtex.tex_pixel_size.to(r_fill_state.texpixel_size);

		if (bdata.settings_uv_contract) {
			r_fill_state.contract_uvs = (batchtex.flags & VS::TEXTURE_FLAG_FILTER) == 0;
		}

		// need to preserve texpixel_size between items
		r_fill_state.texpixel_size = r_fill_state.texpixel_size;

		// open new batch (this should never fail, it dynamically grows)
		r_fill_state.curr_batch = _batch_request_new(false);

		r_fill_state.curr_batch->type = Batch::BT_RECT;
		r_fill_state.curr_batch->color.set(col);
		r_fill_state.curr_batch->batch_texture_id = r_fill_state.batch_tex_id;
		r_fill_state.curr_batch->first_command = command_num;
		r_fill_state.curr_batch->num_commands = 1;
		r_fill_state.curr_batch->first_quad = bdata.total_quads;
	} else {
		// we could alternatively do the count when closing a batch .. perhaps more efficient
		r_fill_state.curr_batch->num_commands++;
	}

	// fill the quad geometry
	Vector2 mins = rect->rect.position;

	if (r_fill_state.transform_mode == TM_TRANSLATE) {
		_software_transform_vertex(mins, r_fill_state.transform_combined);
	}

	Vector2 maxs = mins + rect->rect.size;

	// just aliases
	BatchVertex *bA = &bvs[0];
	BatchVertex *bB = &bvs[1];
	BatchVertex *bC = &bvs[2];
	BatchVertex *bD = &bvs[3];

	bA->pos.x = mins.x;
	bA->pos.y = mins.y;

	bB->pos.x = maxs.x;
	bB->pos.y = mins.y;

	bC->pos.x = maxs.x;
	bC->pos.y = maxs.y;

	bD->pos.x = mins.x;
	bD->pos.y = maxs.y;

	// possibility of applying flips here for normal mapping .. but they don't seem to be used
	if (rect->rect.size.x < 0) {
		SWAP(bA->pos, bB->pos);
		SWAP(bC->pos, bD->pos);
	}
	if (rect->rect.size.y < 0) {
		SWAP(bA->pos, bD->pos);
		SWAP(bB->pos, bC->pos);
	}

	if (r_fill_state.transform_mode == TM_ALL) {
		_software_transform_vertex(bA->pos, r_fill_state.transform_combined);
		_software_transform_vertex(bB->pos, r_fill_state.transform_combined);
		_software_transform_vertex(bC->pos, r_fill_state.transform_combined);
		_software_transform_vertex(bD->pos, r_fill_state.transform_combined);
	}

	// uvs
	Vector2 src_min;
	Vector2 src_max;
	if (rect->flags & CANVAS_RECT_REGION) {
		src_min = rect->source.position;
		src_max = src_min + rect->source.size;

		src_min *= r_fill_state.texpixel_size;
		src_max *= r_fill_state.texpixel_size;

		const float uv_epsilon = bdata.settings_uv_contract_amount;

		// nudge offset for the maximum to prevent precision error on GPU reading into line outside the source rect
		// this is very difficult to get right.
		if (r_fill_state.contract_uvs) {
			src_min.x += uv_epsilon;
			src_min.y += uv_epsilon;
			src_max.x -= uv_epsilon;
			src_max.y -= uv_epsilon;
		}
	} else {
		src_min = Vector2(0, 0);
		src_max = Vector2(1, 1);
	}

	// 10% faster calculating the max first
	Vector2 uvs[4] = {
		src_min,
		Vector2(src_max.x, src_min.y),
		src_max,
		Vector2(src_min.x, src_max.y),
	};

	// for encoding in light angle
	// flips should be optimized out when not being used for light angle.
	bool flip_h = false;
	bool flip_v = false;

	if (rect->flags & CANVAS_RECT_TRANSPOSE) {
		SWAP(uvs[1], uvs[3]);
	}

	if (rect->flags & CANVAS_RECT_FLIP_H) {
		SWAP(uvs[0], uvs[1]);
		SWAP(uvs[2], uvs[3]);
		flip_h = !flip_h;
		flip_v = !flip_v;
	}
	if (rect->flags & CANVAS_RECT_FLIP_V) {
		SWAP(uvs[0], uvs[3]);
		SWAP(uvs[1], uvs[2]);
		flip_v = !flip_v;
	}

	bA->uv.set(uvs[0]);
	bB->uv.set(uvs[1]);
	bC->uv.set(uvs[2]);
	bD->uv.set(uvs[3]);

	if (SEND_LIGHT_ANGLES) {
		// we can either keep the light angles in sync with the verts when writing,
		// or sync them up during translation. We are syncing in translation.
		// N.B. There may be batches that don't require light_angles between batches that do.
		float *angles = bdata.light_angles.request(4);
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
		CRASH_COND(angles == nullptr);
#endif

		float angle = 0.0f;
		const float TWO_PI = Math_PI * 2;

		if (r_fill_state.transform_mode != TM_NONE) {

			const Transform2D &tr = r_fill_state.transform_combined;

			// apply to an x axis
			// the x axis and y axis can be taken directly from the transform (no need to xform identity vectors)
			Vector2 x_axis(tr.elements[0][0], tr.elements[1][0]);

			// have to do a y axis to check for scaling flips
			// this is hassle and extra slowness. We could only allow flips via the flags.
			Vector2 y_axis(tr.elements[0][1], tr.elements[1][1]);

			// has the x / y axis flipped due to scaling?
			float cross = x_axis.cross(y_axis);
			if (cross < 0.0f) {
				flip_v = !flip_v;
			}

			// passing an angle is smaller than a vector, it can be reconstructed in the shader
			angle = x_axis.angle();

			// we don't want negative angles, as negative is used to encode flips.
			// This moves range from -PI to PI to 0 to TWO_PI
			if (angle < 0.0f)
				angle += TWO_PI;

		} // if transform needed

		// if horizontal flip, angle is shifted by 180 degrees
		if (flip_h) {
			angle += Math_PI;

			// mod to get back to 0 to TWO_PI range
			angle = fmodf(angle, TWO_PI);
		}

		// add 1 (to take care of zero floating point error with sign)
		angle += 1.0f;

		// flip if necessary to indicate a vertical flip in the shader
		if (flip_v)
			angle *= -1.0f;

		// light angle must be sent for each vert, instead as a single uniform in the uniform draw method
		// this has the benefit of enabling batching with light angles.
		for (int n = 0; n < 4; n++) {
			angles[n] = angle;
		}
	}

	// increment quad count
	bdata.total_quads++;

	return false;
}

#endif // RASTERIZERCANVASGLES2_H
