/*************************************************************************/
/*  renderer_viewport.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RENDERER_VIEWPORT_H
#define RENDERER_VIEWPORT_H

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/renderer_scene.h"
#include "servers/rendering_server.h"
#include "servers/xr/xr_interface.h"

class RendererViewport {
public:
	struct CanvasBase {
	};

	struct Viewport {
		RID self;
		RID parent;

		bool use_xr; /* use xr interface to override camera positioning and projection matrices and control output */

		Size2i internal_size;
		Size2i size;
		RID camera;
		RID scenario;

		RS::ViewportScaling3DMode scaling_3d_mode;
		float scaling_3d_scale = 1.0;
		float fsr_sharpness = 0.2f;
		float fsr_mipmap_bias = 0.0f;
		bool fsr_enabled;
		RS::ViewportUpdateMode update_mode;
		RID render_target;
		RID render_target_texture;
		RID render_buffers;

		RS::ViewportMSAA msaa;
		RS::ViewportScreenSpaceAA screen_space_aa;
		bool use_debanding;

		bool use_occlusion_culling;
		bool occlusion_buffer_dirty;

		DisplayServer::WindowID viewport_to_screen;
		Rect2 viewport_to_screen_rect;
		bool viewport_render_direct_to_screen;

		bool disable_2d = false;
		bool disable_environment = false;
		bool disable_3d = false;
		bool measure_render_time;

		bool snap_2d_transforms_to_pixel;
		bool snap_2d_vertices_to_pixel;

		uint64_t time_cpu_begin;
		uint64_t time_cpu_end;

		uint64_t time_gpu_begin;
		uint64_t time_gpu_end;

		RID shadow_atlas;
		int shadow_atlas_size;
		bool shadow_atlas_16_bits = false;

		bool sdf_active;

		float lod_threshold = 1.0;

		uint64_t last_pass = 0;

		RS::ViewportDebugDraw debug_draw;

		RS::ViewportClearMode clear_mode;

		RS::CanvasItemTextureFilter texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
		RS::CanvasItemTextureRepeat texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;

		bool transparent_bg;

		struct CanvasKey {
			int64_t stacking;
			RID canvas;
			bool operator<(const CanvasKey &p_canvas) const {
				if (stacking == p_canvas.stacking) {
					return canvas < p_canvas.canvas;
				}
				return stacking < p_canvas.stacking;
			}
			CanvasKey() {
				stacking = 0;
			}
			CanvasKey(const RID &p_canvas, int p_layer, int p_sublayer) {
				canvas = p_canvas;
				int64_t sign = p_layer < 0 ? -1 : 1;
				stacking = sign * (((int64_t)ABS(p_layer)) << 32) + p_sublayer;
			}
			int get_layer() const { return stacking >> 32; }
		};

		struct CanvasData {
			CanvasBase *canvas;
			Transform2D transform;
			int layer;
			int sublayer;
		};

		Transform2D global_transform;

		Map<RID, CanvasData> canvas_map;

		RendererScene::RenderInfo render_info;

		Viewport() {
			update_mode = RS::VIEWPORT_UPDATE_WHEN_VISIBLE;
			clear_mode = RS::VIEWPORT_CLEAR_ALWAYS;
			transparent_bg = false;

			viewport_to_screen = DisplayServer::INVALID_WINDOW_ID;
			shadow_atlas_size = 0;
			measure_render_time = false;

			debug_draw = RS::VIEWPORT_DEBUG_DRAW_DISABLED;
			msaa = RS::VIEWPORT_MSAA_DISABLED;
			screen_space_aa = RS::VIEWPORT_SCREEN_SPACE_AA_DISABLED;
			use_debanding = false;
			use_occlusion_culling = false;
			occlusion_buffer_dirty = true;

			snap_2d_transforms_to_pixel = false;
			snap_2d_vertices_to_pixel = false;

			use_xr = false;
			sdf_active = false;

			time_cpu_begin = 0;
			time_cpu_end = 0;

			time_gpu_begin = 0;
			time_gpu_end = 0;
		}

		uint32_t get_view_count();
	};

	HashMap<String, RID> timestamp_vp_map;

	uint64_t draw_viewports_pass = 0;

	mutable RID_Owner<Viewport, true> viewport_owner;

	struct ViewportSort {
		_FORCE_INLINE_ bool operator()(const Viewport *p_left, const Viewport *p_right) const {
			bool left_to_screen = p_left->viewport_to_screen_rect.size != Size2();
			bool right_to_screen = p_right->viewport_to_screen_rect.size != Size2();

			if (left_to_screen == right_to_screen) {
				return p_right->parent == p_left->self;
			}
			return (right_to_screen ? 0 : 1) < (left_to_screen ? 0 : 1);
		}
	};

	Vector<Viewport *> active_viewports;

	int total_objects_drawn = 0;
	int total_vertices_drawn = 0;
	int total_draw_calls_used = 0;

private:
	void _configure_3d_render_buffers(Viewport *p_viewport);
	void _draw_3d(Viewport *p_viewport);
	void _draw_viewport(Viewport *p_viewport);

	int occlusion_rays_per_thread = 512;

	void _resize_occlusion_culling_buffer(const Size2i &p_size);

public:
	RID viewport_allocate();
	void viewport_initialize(RID p_rid);

	void viewport_set_use_xr(RID p_viewport, bool p_use_xr);

	void viewport_set_size(RID p_viewport, int p_width, int p_height);

	void viewport_attach_to_screen(RID p_viewport, const Rect2 &p_rect = Rect2(), DisplayServer::WindowID p_screen = DisplayServer::MAIN_WINDOW_ID);
	void viewport_set_render_direct_to_screen(RID p_viewport, bool p_enable);

	void viewport_set_active(RID p_viewport, bool p_active);
	void viewport_set_parent_viewport(RID p_viewport, RID p_parent_viewport);

	void viewport_set_scaling_3d_mode(RID p_viewport, RS::ViewportScaling3DMode p_mode);
	void viewport_set_scaling_3d_scale(RID p_viewport, float p_scaling_3d_scale);
	void viewport_set_fsr_sharpness(RID p_viewport, float p_sharpness);
	void viewport_set_fsr_mipmap_bias(RID p_viewport, float p_mipmap_bias);

	void viewport_set_update_mode(RID p_viewport, RS::ViewportUpdateMode p_mode);
	void viewport_set_vflip(RID p_viewport, bool p_enable);

	void viewport_set_clear_mode(RID p_viewport, RS::ViewportClearMode p_clear_mode);

	RID viewport_get_texture(RID p_viewport) const;
	RID viewport_get_occluder_debug_texture(RID p_viewport) const;

	void viewport_set_disable_2d(RID p_viewport, bool p_disable);
	void viewport_set_disable_environment(RID p_viewport, bool p_disable);
	void viewport_set_disable_3d(RID p_viewport, bool p_disable);

	void viewport_attach_camera(RID p_viewport, RID p_camera);
	void viewport_set_scenario(RID p_viewport, RID p_scenario);
	void viewport_attach_canvas(RID p_viewport, RID p_canvas);
	void viewport_remove_canvas(RID p_viewport, RID p_canvas);
	void viewport_set_canvas_transform(RID p_viewport, RID p_canvas, const Transform2D &p_offset);
	void viewport_set_transparent_background(RID p_viewport, bool p_enabled);

	void viewport_set_global_canvas_transform(RID p_viewport, const Transform2D &p_transform);
	void viewport_set_canvas_stacking(RID p_viewport, RID p_canvas, int p_layer, int p_sublayer);

	void viewport_set_shadow_atlas_size(RID p_viewport, int p_size, bool p_16_bits = false);
	void viewport_set_shadow_atlas_quadrant_subdivision(RID p_viewport, int p_quadrant, int p_subdiv);

	void viewport_set_msaa(RID p_viewport, RS::ViewportMSAA p_msaa);
	void viewport_set_screen_space_aa(RID p_viewport, RS::ViewportScreenSpaceAA p_mode);
	void viewport_set_use_debanding(RID p_viewport, bool p_use_debanding);
	void viewport_set_use_occlusion_culling(RID p_viewport, bool p_use_occlusion_culling);
	void viewport_set_occlusion_rays_per_thread(int p_rays_per_thread);
	void viewport_set_occlusion_culling_build_quality(RS::ViewportOcclusionCullingBuildQuality p_quality);
	void viewport_set_lod_threshold(RID p_viewport, float p_pixels);

	virtual int viewport_get_render_info(RID p_viewport, RS::ViewportRenderInfoType p_type, RS::ViewportRenderInfo p_info);
	virtual void viewport_set_debug_draw(RID p_viewport, RS::ViewportDebugDraw p_draw);

	void viewport_set_measure_render_time(RID p_viewport, bool p_enable);
	float viewport_get_measured_render_time_cpu(RID p_viewport) const;
	float viewport_get_measured_render_time_gpu(RID p_viewport) const;

	void viewport_set_snap_2d_transforms_to_pixel(RID p_viewport, bool p_enabled);
	void viewport_set_snap_2d_vertices_to_pixel(RID p_viewport, bool p_enabled);

	void viewport_set_default_canvas_item_texture_filter(RID p_viewport, RS::CanvasItemTextureFilter p_filter);
	void viewport_set_default_canvas_item_texture_repeat(RID p_viewport, RS::CanvasItemTextureRepeat p_repeat);

	void viewport_set_sdf_oversize_and_scale(RID p_viewport, RS::ViewportSDFOversize p_over_size, RS::ViewportSDFScale p_scale);

	void handle_timestamp(String p_timestamp, uint64_t p_cpu_time, uint64_t p_gpu_time);

	void set_default_clear_color(const Color &p_color);
	void draw_viewports();

	bool free(RID p_rid);

	int get_total_objects_drawn() const;
	int get_total_vertices_drawn() const;
	int get_total_draw_calls_used() const;

	// Workaround for setting this on thread.
	void call_set_vsync_mode(DisplayServer::VSyncMode p_mode, DisplayServer::WindowID p_window);

	RendererViewport();
	virtual ~RendererViewport() {}
};

#endif // RENDERER_VIEWPORT_H
