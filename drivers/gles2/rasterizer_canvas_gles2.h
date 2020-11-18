/*************************************************************************/
/*  rasterizer_canvas_gles2.h                                            */
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

#ifndef RASTERIZERCANVASGLES2_H
#define RASTERIZERCANVASGLES2_H

#include "drivers/gles_common/rasterizer_platforms.h"
#ifdef GLES2_BACKEND_ENABLED

#include "drivers/gles_common/rasterizer_canvas_batcher.h"
#include "drivers/gles_common/rasterizer_version.h"
#include "rasterizer_canvas_base_gles2.h"

class RasterizerSceneGLES2;

class RasterizerCanvasGLES2 : public RasterizerCanvasBaseGLES2, public RasterizerCanvasBatcher<RasterizerCanvasGLES2, RasterizerStorageGLES2> {
	friend class RasterizerCanvasBatcher<RasterizerCanvasGLES2, RasterizerStorageGLES2>;

public:
	virtual void canvas_render_items_begin(const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	virtual void canvas_render_items_end();
	void canvas_render_items_internal(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	virtual void canvas_begin() override;
	virtual void canvas_end() override;

	void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) override {
		storage->frame.current_rt = nullptr;

		//if (p_to_render_target.is_valid())
		//		print_line("canvas_render_items " + itos(p_to_render_target.get_id()) );
		//		print_line("canvas_render_items ");

		// first set the current render target
		storage->_set_current_render_target(p_to_render_target);

		// binds the render target (framebuffer)
		canvas_begin();

		canvas_render_items_begin(p_modulate, p_light_list, p_canvas_transform);
		canvas_render_items_internal(p_item_list, 0, p_modulate, p_light_list, p_canvas_transform);
		canvas_render_items_end();

		canvas_end();

		// not sure why these are needed to get frame to render?
		storage->_set_current_render_target(RID());
		//		storage->frame.current_rt = nullptr;
		//		canvas_begin();
		//		canvas_end();
	}

private:
	// legacy codepath .. to remove after testing
	void _legacy_canvas_render_item(Item *p_ci, RenderItemState &r_ris);

	// high level batch funcs
	void canvas_render_items_implementation(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	//void render_joined_item(const BItemJoined &p_bij, RenderItemState &r_ris);
	//bool try_join_item(Item *p_ci, RenderItemState &r_ris, bool &r_batch_break);
	void render_batches(Item::Command *const *p_commands, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material);

	// low level batch funcs
	//	void _batch_upload_buffers();
	//	void _batch_render_generic(const Batch &p_batch, RasterizerStorageGLES2::Material *p_material);
	//	void _batch_render_lines(const Batch &p_batch, RasterizerStorageGLES2::Material *p_material, bool p_anti_alias);

	// funcs used from rasterizer_canvas_batcher template
	void gl_enable_scissor(int p_x, int p_y, int p_width, int p_height) const;
	void gl_disable_scissor() const;

public:
	void initialize();
	RasterizerCanvasGLES2();
};

#endif // GLES2_BACKEND_ENABLED
#endif // RASTERIZERCANVASGLES2_H
