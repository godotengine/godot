/*************************************************************************/
/*  color_transform.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef COLOR_TRANSFORM_H
#define COLOR_TRANSFORM_H

#include "core/image.h"
#include "core/reference.h"
#include "lcms2.h"

#include "color_profile.h"

class ColorTransform : public Reference {
	GDCLASS(ColorTransform, Reference)

public:
	enum Intent { // re-export LCMS macro constants for safety and scope complications
		CM_INTENT_PERCEPTUAL = INTENT_PERCEPTUAL,
		CM_INTENT_RELATIVE = INTENT_RELATIVE_COLORIMETRIC,
		CM_INTENT_SATURATION = INTENT_SATURATION,
		CM_INTENT_ABSOLUTE = INTENT_ABSOLUTE_COLORIMETRIC,
	};

private:
	Ref<ColorProfile> src_profile;
	Ref<ColorProfile> dst_profile;
	cmsUInt32Number intent;
	bool use_bpc;

public:
	bool is_valid();

	void set_src_profile(Ref<ColorProfile> p_profile);
	void set_dst_profile(Ref<ColorProfile> p_profile);
	void set_intent(Intent p_intent);
	void set_black_point_compensation(bool p_black_point_compensation);

	bool apply_screen_lut();

	ColorTransform();
	ColorTransform(Ref<ColorProfile> p_src, Ref<ColorProfile> p_dst, Intent p_intent = CM_INTENT_PERCEPTUAL, bool p_black_point_compensation = true);
	~ColorTransform();
};

#endif // COLOR_TRANSFORM_H
