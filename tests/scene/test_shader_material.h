/**************************************************************************/
/*  test_shader_material.h                                                */
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

#include "scene/resources/material.h"
#include "tests/test_macros.h"

namespace TestShaderMaterial {

TEST_CASE("[ShaderMaterial] get/set") {
	Ref<ShaderMaterial> shader_material = memnew(ShaderMaterial);

	shader_material->set_shader_parameter("param", Variant(2));
	CHECK(shader_material->get_shader_parameter("param") == Variant(2));
}

TEST_CASE("[ShaderMaterial] implicit int/float conversion") {
	Ref<ShaderMaterial> shader_material = memnew(ShaderMaterial);

	shader_material->set_shader_parameter("float", 2.4);
	shader_material->set_shader_parameter("float", 2);
	CHECK(shader_material->get_shader_parameter("float") == Variant(2.0));

	shader_material->set_shader_parameter("int", 2);
	shader_material->set_shader_parameter("int", 2.4);
	CHECK(shader_material->get_shader_parameter("int") == Variant(2));
}

} // namespace TestShaderMaterial
