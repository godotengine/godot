/**************************************************************************/
/*  visual_shader_language_plugin.cpp                                     */
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

#include "visual_shader_language_plugin.h"

#include "visual_shader_editor_plugin.h"

bool VisualShaderLanguagePlugin::handles_shader(const Ref<Shader> &p_shader) const {
	return Object::cast_to<VisualShader>(p_shader.ptr()) != nullptr;
}

ShaderEditor *VisualShaderLanguagePlugin::edit_shader(const Ref<Shader> &p_shader) {
	const Ref<VisualShader> shader = p_shader;
	ERR_FAIL_COND_V(shader.is_null(), nullptr);
	VisualShaderEditor *editor = memnew(VisualShaderEditor);
	editor->edit_shader(shader);
	return editor;
}

Ref<Shader> VisualShaderLanguagePlugin::create_new_shader(int p_variation_index, Shader::Mode p_shader_mode, int p_template_index) {
	Ref<VisualShader> shader;
	shader.instantiate();
	shader->set_mode(p_shader_mode);
	return shader;
}

PackedStringArray VisualShaderLanguagePlugin::get_language_variations() const {
	PackedStringArray variations;
	variations.push_back("VisualShader");
	return variations;
}
