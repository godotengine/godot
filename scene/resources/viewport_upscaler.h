/**************************************************************************/
/*  viewport_upscaler.h                                                   */
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

#pragma once

#include "core/object/gdvirtual.gen.inc"
#include "servers/rendering/storage/render_data.h"

class ViewportUpscaler : public Resource {
	GDCLASS(ViewportUpscaler, Resource);

private:
	RID rid;
	bool requires_motion_vectors = false;
	float mipmap_bias = 0.0;
	uint32_t jitter_phase_count = 0;

protected:
	static void _bind_methods();

	GDVIRTUAL1(_render_callback, const RenderData *)

public:
	virtual void render_callback(const RenderData *p_render_data);
	virtual RID get_rid() const override { return rid; }

	void set_requires_motion_vectors(bool p_enabled);
	bool get_requires_motion_vectors() const;

	void set_mipmap_bias(float p_mipmap_bias);
	float get_mipmap_bias() const;

	void set_jitter_phase_count(uint32_t p_jitter_phase_count);
	uint32_t get_jitter_phase_count() const;

	ViewportUpscaler();
	~ViewportUpscaler();
};
