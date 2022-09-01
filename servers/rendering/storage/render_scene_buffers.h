/*************************************************************************/
/*  render_scene_buffers.h                                               */
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

#ifndef RENDER_SCENE_BUFFERS_H
#define RENDER_SCENE_BUFFERS_H

#include "core/object/ref_counted.h"
#include "servers/rendering_server.h"

class RenderSceneBuffers : public RefCounted {
	GDCLASS(RenderSceneBuffers, RefCounted);

protected:
	static void _bind_methods();

	GDVIRTUAL10(_configure, RID, Size2i, Size2i, float, float, RS::ViewportMSAA, RenderingServer::ViewportScreenSpaceAA, bool, bool, uint32_t)
	GDVIRTUAL1(_set_fsr_sharpness, float)
	GDVIRTUAL1(_set_texture_mipmap_bias, float)
	GDVIRTUAL1(_set_use_debanding, bool)

public:
	RenderSceneBuffers(){};
	virtual ~RenderSceneBuffers(){};

	virtual void configure(RID p_render_target, const Size2i p_internal_size, const Size2i p_target_size, float p_fsr_sharpness, float p_texture_mipmap_bias, RS::ViewportMSAA p_msaa_3d, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count);

	// for those settings that are unlikely to require buffers to be recreated, we'll add setters
	virtual void set_fsr_sharpness(float p_fsr_sharpness);
	virtual void set_texture_mipmap_bias(float p_texture_mipmap_bias);
	virtual void set_use_debanding(bool p_use_debanding);
};

#endif // RENDER_SCENE_BUFFERS_H
