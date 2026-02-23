/**************************************************************************/
/*  rasterizer_util_gles3.h                                               */
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

#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_commons.h"
#include <cstdint>

#include "platform_gl.h"

// This class is meant to hold static utility methods with minimal dependencies.

class RasterizerUtilGLES3 {
private:
	static bool gles_over_gl;

public:
	static void set_gles_over_gl(bool p_gles_over_gl) {
		gles_over_gl = p_gles_over_gl;
	}
	static bool is_gles_over_gl() {
		return gles_over_gl;
	}

	static void clear_depth(float p_depth);
	static void clear_stencil(int32_t p_stencil);

	static GLuint RD_TO_GL_BLEND_FUNC(RD::BlendOperation op) {
		switch (op) {
			case RD::BLEND_OP_ADD:
				return GL_FUNC_ADD;
			case RD::BLEND_OP_SUBTRACT:
				return GL_FUNC_SUBTRACT;
			case RD::BLEND_OP_REVERSE_SUBTRACT:
				return GL_FUNC_REVERSE_SUBTRACT;
			case RD::BLEND_OP_MINIMUM:
				return GL_MIN;
			case RD::BLEND_OP_MAXIMUM:
				return GL_MAX;
			case RD::BLEND_OP_MAX:
				return GL_MAX;
			default:
				break;
		}

		return GL_FUNC_ADD;
	}

	static GLuint RD_TO_GL_BLEND_FACTOR(RD::BlendFactor f) {
		switch (f) {
			case RD::BLEND_FACTOR_ZERO:
				return GL_ZERO;
			case RD::BLEND_FACTOR_ONE:
				return GL_ONE;
			case RD::BLEND_FACTOR_SRC_COLOR:
				return GL_SRC_COLOR;
			case RD::BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
				return GL_ONE_MINUS_SRC_COLOR;
			case RD::BLEND_FACTOR_DST_COLOR:
				return GL_DST_COLOR;
			case RD::BLEND_FACTOR_ONE_MINUS_DST_COLOR:
				return GL_ONE_MINUS_DST_COLOR;
			case RD::BLEND_FACTOR_SRC_ALPHA:
				return GL_SRC_ALPHA;
			case RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
				return GL_ONE_MINUS_SRC_ALPHA;
			case RD::BLEND_FACTOR_DST_ALPHA:
				return GL_ONE_MINUS_DST_ALPHA;
			case RD::BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
				return GL_ONE_MINUS_DST_ALPHA;
			case RD::BLEND_FACTOR_CONSTANT_COLOR:
				return GL_CONSTANT_COLOR;
			case RD::BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
				return GL_ONE_MINUS_CONSTANT_COLOR;
			case RD::BLEND_FACTOR_CONSTANT_ALPHA:
				return GL_CONSTANT_ALPHA;
			case RD::BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
				return GL_ONE_MINUS_CONSTANT_ALPHA;
			case RD::BLEND_FACTOR_SRC_ALPHA_SATURATE:
				return GL_SRC_ALPHA_SATURATE;
// blend factors only supported on desktop opengl
#ifdef GL_API_ENABLED
			case RD::BLEND_FACTOR_SRC1_COLOR:
				return GL_SRC1_COLOR;
			case RD::BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
				return GL_ONE_MINUS_SRC1_COLOR;
			case RD::BLEND_FACTOR_SRC1_ALPHA:
				return GL_SRC1_ALPHA;
			case RD::BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
				return GL_ONE_MINUS_SRC1_ALPHA;
#endif
			case RD::BLEND_FACTOR_MIN:
				return GL_MIN;
			case RD::BLEND_FACTOR_MAX:
				return GL_MAX;
			default:
				break;
		}

		return GL_ONE;
	}
};
