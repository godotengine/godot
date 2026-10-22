/**************************************************************************/
/*  test_visual_shader_input.h                                            */
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

#ifndef TEST_VISUAL_SHADER_INPUT_H
#define TEST_VISUAL_SHADER_INPUT_H

#include "../visual_shader.h"
#include "../visual_shader_group.h"

#include "tests/test_macros.h"

namespace TestVisualShaderInput {

TEST_CASE("[SceneTree][VisualShader] Group input availability") {
	Ref<VisualShaderNodeInput> input;
	input.instantiate();
	input->set_shader_mode(Shader::MODE_SPATIAL);
	input->set_shader_type(VisualShader::TYPE_MAX);

	SUBCASE("Global variables and constants remain available") {
		HashSet<String> names;
		for (int i = 0; i < input->get_input_index_count(); i++) {
			CHECK_FALSE(names.has(input->get_input_index_name(i)));
			names.insert(input->get_input_index_name(i));
		}
		CHECK(names.has("time"));
		CHECK(names.has("clip_space_far"));
		CHECK(names.has("output_is_srgb"));

		for (const char *name : { "time", "clip_space_far", "output_is_srgb" }) {
			input->set_input_name(name);
			CHECK(input->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_MAX));
			CHECK_FALSE(input->get_input_real_name().is_empty());
		}
	}

	SUBCASE("Stage-specific inputs are unavailable in groups") {
		for (const char *name : { "vertex", "normal", "view" }) {
			input->set_input_name(name);
			CHECK_FALSE(input->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_MAX));
			CHECK(input->get_input_real_name().is_empty());
			CHECK(input->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT));
		}
	}

	SUBCASE("Copied inputs are checked by name in the target context") {
		input->set_shader_type(VisualShader::TYPE_FRAGMENT);
		input->set_input_name("time");
		Ref<VisualShaderNodeInput> copy = input->duplicate();
		CHECK(copy->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_MAX));
		CHECK(copy->is_available(Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_MAX));

		copy->set_input_name("normal");
		CHECK_FALSE(copy->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_MAX));
		CHECK(copy->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT));
		CHECK_FALSE(copy->is_available(Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX));
	}

	SUBCASE("Standalone groups have no parent shader mode") {
		input->set_shader_mode(Shader::MODE_MAX);
		input->set_input_name("time");
		CHECK(input->is_available(Shader::MODE_MAX, VisualShader::TYPE_MAX));
		CHECK(input->get_input_index_count() > 0);
		input->set_input_name("clip_space_far");
		CHECK(input->is_available(Shader::MODE_MAX, VisualShader::TYPE_MAX));
		input->set_input_name("normal");
		CHECK_FALSE(input->is_available(Shader::MODE_MAX, VisualShader::TYPE_MAX));
	}

	SUBCASE("Unconfigured and unknown input names") {
		CHECK(input->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_MAX));
		input->set_input_name("not_a_shader_input");
		CHECK_FALSE(input->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_MAX));
	}

	SUBCASE("Group compatibility checks its inputs in group context") {
		Ref<VisualShaderGroup> group;
		group.instantiate();
		group->add_node(input, Vector2(), group->get_valid_node_id());
		Ref<VisualShaderNodeGroup> group_node;
		group_node.instantiate();
		group_node->set_group(group);

		input->set_input_name("clip_space_far");
		CHECK(group_node->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT));
		CHECK_FALSE(group_node->is_available(Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT));
		input->set_input_name("time");
		CHECK(group_node->is_available(Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT));
		input->set_input_name("normal");
		CHECK_FALSE(group_node->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT));

		Ref<VisualShaderGroup> outer_group;
		outer_group.instantiate();
		outer_group->add_node(group_node, Vector2(), outer_group->get_valid_node_id());
		Ref<VisualShaderNodeGroup> outer_node;
		outer_node.instantiate();
		outer_node->set_group(outer_group);
		CHECK_FALSE(outer_node->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT));
		input->set_input_name("time");
		CHECK(outer_node->is_available(Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT));
	}
}

} // namespace TestVisualShaderInput

#endif // TEST_VISUAL_SHADER_INPUT_H
