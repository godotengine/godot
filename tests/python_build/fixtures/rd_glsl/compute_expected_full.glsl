/**************************************************************************/
/*  compute.glsl.gen.h                                                    */
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

/* THIS FILE IS GENERATED. EDITS WILL BE LOST. */

#ifndef COMPUTE_RD_GLSL_GEN_H
#define COMPUTE_RD_GLSL_GEN_H

#include "servers/rendering/renderer_rd/shader_rd.h"

class ComputeShaderRD : public ShaderRD {
public:
	ComputeShaderRD() {
		static const char _compute_code[] = {
			10, 35, 118, 101, 114, 115, 105, 111, 110, 32, 52, 53, 48, 10, 10, 35, 86, 69, 82, 83, 73, 79, 78, 95, 68, 69, 70, 73, 78, 69, 83, 10, 10, 35, 100, 101, 102, 105, 110, 101, 32, 66, 76, 79, 67, 75, 95, 83, 73, 90, 69, 32, 56, 10, 10, 35, 100, 101, 102, 105, 110, 101, 32, 77, 95, 80, 73, 32, 51, 46, 49, 52, 49, 53, 57, 50, 54, 53, 51, 53, 57, 10, 10, 118, 111, 105, 100, 32, 109, 97, 105, 110, 40, 41, 32, 123, 10, 9, 117, 105, 110, 116, 32, 116, 32, 61, 32, 66, 76, 79, 67, 75, 95, 83, 73, 90, 69, 32, 43, 32, 49, 59, 10, 125, 10, 0
		};
		setup(nullptr, nullptr, _compute_code, "ComputeShaderRD");
	}
};

#endif // COMPUTE_RD_GLSL_GEN_H
