/*************************************************************************/
/*  renderer_compositor.h                                                */
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

#ifndef RENDERING_SERVER_COMPOSITOR_H
#define RENDERING_SERVER_COMPOSITOR_H

#include "core/math/camera_matrix.h"
#include "core/templates/pair.h"
#include "core/templates/self_list.h"
#include "servers/rendering/renderer_canvas_render.h"
#include "servers/rendering/renderer_scene.h"
#include "servers/rendering/renderer_storage.h"
#include "servers/rendering_server.h"
class RendererSceneRender;
struct BlitToScreen {
	RID render_target;
	Rect2 src_rect = Rect2(0.0, 0.0, 1.0, 1.0);
	Rect2i dst_rect;

	struct {
		bool use_layer = false;
		uint32_t layer = 0;
	} multi_view;

	struct {
		//lens distorted parameters for VR
		bool apply = false;
		Vector2 eye_center;
		float k1 = 0.0;
		float k2 = 0.0;

		float upscale = 1.0;
		float aspect_ratio = 1.0;
	} lens_distortion;
};

class RendererCompositor {
private:
	bool xr_enabled = false;

protected:
	static RendererCompositor *(*_create_func)();
	bool back_end = false;

public:
	static RendererCompositor *create();

	virtual RendererStorage *get_storage() = 0;
	virtual RendererCanvasRender *get_canvas() = 0;
	virtual RendererSceneRender *get_scene() = 0;

	virtual void set_boot_image(const Ref<Image> &p_image, const Color &p_color, RenderingServer::SplashStretchMode p_stretch_mode, bool p_use_filter = true) = 0;

	virtual void initialize() = 0;
	virtual void begin_frame(double frame_step) = 0;

	virtual void prepare_for_blitting_render_targets() = 0;
	virtual void blit_render_targets_to_screen(DisplayServer::WindowID p_screen, const BlitToScreen *p_render_targets, int p_amount) = 0;

	virtual void end_frame(bool p_swap_buffers) = 0;
	virtual void finalize() = 0;
	virtual uint64_t get_frame_number() const = 0;
	virtual double get_frame_delta_time() const = 0;

	_FORCE_INLINE_ virtual bool is_low_end() const { return back_end; };
	virtual bool is_xr_enabled() const;

	RendererCompositor();
	virtual ~RendererCompositor() {}
};

#endif // RASTERIZER_H
