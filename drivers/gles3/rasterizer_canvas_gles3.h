/*************************************************************************/
/*  rasterizer_canvas_gles3.h                                            */
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

#ifndef RASTERIZER_CANVAS_OPENGL_H
#define RASTERIZER_CANVAS_OPENGL_H

#include "drivers/gles3/rasterizer_platforms.h"
#ifdef GLES3_BACKEND_ENABLED

#include "drivers/gles3/rasterizer_canvas_batcher.h"
#include "rasterizer_canvas_base_gles3.h"

class RasterizerSceneGLES3;

class RasterizerCanvasGLES3 : public RasterizerCanvasBaseGLES3, public RasterizerCanvasBatcher<RasterizerCanvasGLES3, RasterizerStorageGLES3> {
	friend class RasterizerCanvasBatcher<RasterizerCanvasGLES3, RasterizerStorageGLES3>;

private:
	// legacy codepath .. to remove after testing
	void _legacy_canvas_render_item(Item *p_ci, RenderItemState &r_ris);

	// high level batch funcs
	void canvas_render_items_implementation(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	void render_batches(Item::Command *const *p_commands, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES3::Material *p_material);

	// funcs used from rasterizer_canvas_batcher template
	void gl_enable_scissor(int p_x, int p_y, int p_width, int p_height) const;
	void gl_disable_scissor() const;

public:
	void canvas_render_items_begin(const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	void canvas_render_items_end();
	void canvas_render_items_internal(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	void canvas_begin() override;
	void canvas_end() override;

	void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) override;

	void initialize();
	RasterizerCanvasGLES3();
};

#endif // GLES3_BACKEND_ENABLED
#endif // RASTERIZER_CANVAS_OPENGL_H
