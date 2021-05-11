/*************************************************************************/
/*  renderer_compositor_rd.h                                             */
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

#ifndef RENDERING_SERVER_COMPOSITOR_RD_H
#define RENDERING_SERVER_COMPOSITOR_RD_H

#include "core/os/os.h"
#include "core/templates/thread_work_pool.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.h"
#include "servers/rendering/renderer_rd/forward_mobile/render_forward_mobile.h"
#include "servers/rendering/renderer_rd/renderer_canvas_render_rd.h"
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"
#include "servers/rendering/renderer_rd/shaders/blit.glsl.gen.h"

class RendererCompositorRD : public RendererCompositor {
protected:
	RendererCanvasRenderRD *canvas;
	RendererStorageRD *storage;
	RendererSceneRenderRD *scene;

	enum BlitMode {
		BLIT_MODE_NORMAL,
		BLIT_MODE_USE_LAYER,
		BLIT_MODE_LENS,
		BLIT_MODE_MAX
	};

	struct BlitPushConstant {
		float rect[4];

		float eye_center[2];
		float k1;
		float k2;

		float upscale;
		float aspect_ratio;
		uint32_t layer;
		uint32_t pad1;
	};

	struct Blit {
		BlitPushConstant push_constant;
		BlitShaderRD shader;
		RID shader_version;
		RID pipelines[BLIT_MODE_MAX];
		RID index_buffer;
		RID array;
		RID sampler;
	} blit;

	Map<RID, RID> render_target_descriptors;

	double time;
	float delta;

	static uint64_t frame;

public:
	RendererStorage *get_storage() { return storage; }
	RendererCanvasRender *get_canvas() { return canvas; }
	RendererSceneRender *get_scene() { return scene; }

	void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {}

	void initialize();
	void begin_frame(double frame_step);
	void prepare_for_blitting_render_targets();
	void blit_render_targets_to_screen(DisplayServer::WindowID p_screen, const BlitToScreen *p_render_targets, int p_amount);

	void end_frame(bool p_swap_buffers);
	void finalize();

	_ALWAYS_INLINE_ uint64_t get_frame_number() const { return frame; }
	_ALWAYS_INLINE_ float get_frame_delta_time() const { return delta; }
	_ALWAYS_INLINE_ double get_total_time() const { return time; }

	static Error is_viable() {
		return OK;
	}

	static RendererCompositor *_create_current() {
		return memnew(RendererCompositorRD);
	}

	static void make_current() {
		_create_func = _create_current;
	}

	virtual bool is_low_end() const { return false; }

	static RendererCompositorRD *singleton;
	RendererCompositorRD();
	~RendererCompositorRD() {}
};
#endif // RASTERIZER_RD_H
