/**************************************************************************/
/*  text_shader_language_plugin.cpp                                       */
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

#include "text_shader_language_plugin.h"

#include "text_shader_editor.h"

#include "core/string/string_builder.h"
#include "servers/rendering/shader_types.h"

bool TextShaderLanguagePlugin::handles_shader(const Ref<Shader> &p_shader) const {
	// The text shader editor only edits the base Shader class,
	// not classes that inherit from it like VisualShader.
	return p_shader->get_class_name() == Shader::get_class_static();
}

bool TextShaderLanguagePlugin::handles_shader_include(const Ref<ShaderInclude> &p_shader_inc) const {
	return p_shader_inc->get_class_static() == ShaderInclude::get_class_static();
}

ShaderEditor *TextShaderLanguagePlugin::edit_shader(const Ref<Shader> &p_shader) {
	TextShaderEditor *editor = memnew(TextShaderEditor);
	editor->edit_shader(p_shader);
	return editor;
}

ShaderEditor *TextShaderLanguagePlugin::edit_shader_include(const Ref<ShaderInclude> &p_shader_inc) {
	TextShaderEditor *editor = memnew(TextShaderEditor);
	editor->edit_shader_include(p_shader_inc);
	return editor;
}

Ref<Shader> TextShaderLanguagePlugin::create_new_shader(int p_variation_index, Shader::Mode p_shader_mode, int p_template_index) {
	Ref<Shader> shader;
	shader.instantiate();

	StringBuilder code;
	const String &shader_type = ShaderTypes::get_singleton()->get_types_list().get(p_shader_mode);
	code += vformat("shader_type %s;\n", shader_type);

	if (p_template_index == 0) { // Default template.
		switch (p_shader_mode) {
			case Shader::MODE_SPATIAL: {
				code += R"(
void vertex() {
	// Called for every vertex the material is visible on.
}

void fragment() {
	// Called for every pixel the material is visible on.
}

//void light() {
//	// Called for every pixel for every light affecting the material.
//	// Uncomment to replace the default light processing function with this one.
//}
)";
			} break;
			case Shader::MODE_CANVAS_ITEM: {
				code += R"(
void vertex() {
	// Called for every vertex the material is visible on.
}

void fragment() {
	// Called for every pixel the material is visible on.
}

//void light() {
//	// Called for every pixel for every light affecting the CanvasItem.
//	// Uncomment to replace the default light processing function with this one.
//}
)";
			} break;
			case Shader::MODE_PARTICLES: {
				code += R"(
void start() {
	// Called when a particle is spawned.
}

void process() {
	// Called every frame on existing particles (according to the Fixed FPS property).
}
)";
			} break;
			case Shader::MODE_SKY: {
				code += R"(
void sky() {
	// Called for every visible pixel in the sky background, as well as all pixels
	// in the radiance cubemap.
}
)";
			} break;
			case Shader::MODE_FOG: {
				code += R"(
void fog() {
	// Called once for every froxel that is touched by an axis-aligned bounding box
	// of the associated FogVolume. This means that froxels that just barely touch
	// a given FogVolume will still be used.
}
)";
			} break;
			case Shader::MODE_MAX: {
				ERR_FAIL_V_MSG(Ref<Shader>(), "Invalid shader mode for text shader editor.");
			} break;
		}
	}
	shader->set_code(code.as_string());
	return shader;
}

Ref<ShaderInclude> TextShaderLanguagePlugin::create_new_shader_include() {
	Ref<ShaderInclude> shader_inc;
	shader_inc.instantiate();
	return shader_inc;
}

PackedStringArray TextShaderLanguagePlugin::get_language_variations() const {
	return PackedStringArray{ "Shader", "ShaderInclude" };
}

String TextShaderLanguagePlugin::get_file_extension(int p_variation_index) const {
	if (p_variation_index == 0) {
		return "gdshader";
	} else if (p_variation_index == 1) {
		return "gdshaderinc";
	}
	return "tres";
}
