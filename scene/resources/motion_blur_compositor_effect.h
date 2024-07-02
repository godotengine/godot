/**************************************************************************/
/*  motion_blur_compositor_effect.h                                       */
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

#ifndef MOTION_BLUR_COMPOSITOR_EFFECT_H
#define MOTION_BLUR_COMPOSITOR_EFFECT_H

#include "scene/resources/compositor.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"

class MotionBlurCompositorEffect : public CompositorEffect {
	GDCLASS(MotionBlurCompositorEffect, CompositorEffect);

private:
	RID linear_sampler;

	RID motion_blur_shader;
	RID motion_blur_pipeline;

	RID overlay_shader;
	RID overlay_pipeline;

	int motion_blur_samples = 8;
	float motion_blur_intensity = 0.25f;
	float motion_blur_center_fade = 0.0;

protected:
	static void _bind_methods();
	void _render_callback(int p_effect_callback_type, const RenderDataRD *p_render_data);

public:
	MotionBlurCompositorEffect();
	virtual ~MotionBlurCompositorEffect();
	int get_motion_blur_samples() const;
	void set_motion_blur_samples(int p_samples);

	float get_motion_blur_intensity() const;
	void set_motion_blur_intensity(float p_intensity);

	void _initialize_compute();
	float get_motion_blur_center_fade() const;
	void set_motion_blur_center_fade(float p_value);
	RD::Uniform get_image_uniform(RID p_image, int p_binding);
	RD::Uniform get_sampler_uniform(RID p_image, int p_binding);
};

#endif // MOTION_BLUR_COMPOSITOR_EFFECT_H
