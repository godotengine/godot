/**************************************************************************/
/*  test_shader_language.h                                                */
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

#ifndef TEST_SHADER_LANGUAGE_H
#define TEST_SHADER_LANGUAGE_H

#include "servers/rendering/shader_language.h"

#include "tests/test_macros.h"

#include <servers/rendering/shader_types.h>

namespace TestShaderLanguage {

TEST_CASE("[ShaderLanguage] Function call trailing commas") {
	String code(
			"shader_type canvas_item;\n"
			"void fragment() {\n"
			"  clamp(\n"
			"    5.0,\n"
			"    0.0,\n"
			"    1.0,\n"
			"  );\n"
			"}\n");
	ShaderLanguage sl;
	sl.set_warning_flags(0);
	ShaderLanguage::ShaderCompileInfo comp_info;
	comp_info.functions = ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(RenderingServer::SHADER_CANVAS_ITEM));
	comp_info.render_modes = ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(RenderingServer::SHADER_CANVAS_ITEM));
	comp_info.shader_types = ShaderTypes::get_singleton()->get_types();
	Error last_compile_result = sl.compile(code, comp_info);
	CHECK_EQ(last_compile_result, OK);
}

} // namespace TestShaderLanguage

#endif // TEST_SHADER_LANGUAGE_H
