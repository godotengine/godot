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

#include "core/string/string_builder.h"
#include "editor/shader/shader_text_editor.h"
#include "scene/gui/option_button.h"
#include "servers/rendering/shader_types.h"

Ref<Shader> TextShaderLanguagePlugin::create_new_shader(int p_variation_index, Shader::Mode p_shader_mode, int p_template_index) {
	Ref<Shader> shader;
	shader.instantiate();

	StringBuilder code;
	const String &shader_type = ShaderTypes::get_singleton()->get_types_list().get(p_shader_mode);
	code += vformat("shader_type %s;\n", shader_type);

	if (p_template_index == DEFAULT_TEMPLATE) { // Default template.
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
			case Shader::MODE_TEXTURE_BLIT: {
				code += R"(
void blit() {
	// Called for each pixel inside the given rect on the DrawableTexture.
}
)";
			} break;
			case Shader::MODE_MAX: {
				ERR_FAIL_V_MSG(Ref<Shader>(), "Invalid shader mode for text shader editor.");
			} break;
		}
	} else {
		switch (p_shader_mode) {
			case Shader::MODE_SPATIAL: {
				switch (p_template_index) {
					case TEMPLATE_SPATIAL_SPRITE_3D: {
						code += R"(render_mode unshaded;

uniform sampler2D texture_albedo : source_color; // Required for Sprite3D.
uniform ivec2 albedo_texture_size; // Required for Sprite3D.

void vertex() {
	// Called for every vertex the material is visible on.

	// Convert vertex color to SRGB.
	COLOR.rgb = mix(
		pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)),
		COLOR.rgb * (1.0 / 12.92),
		lessThan(COLOR.rgb, vec3(0.04045)));
}

void fragment() {
	// Called for every pixel the material is visible on.
	vec4 col = COLOR * texture(texture_albedo, UV);
	ALBEDO = col.rgb;
	ALPHA = col.a;
}

//void light() {
//	// Called for every pixel for every light affecting the material.
//	// Uncomment to replace the default light processing function with this one.
//}
)";
					} break;
				}
			} break;
			default: {
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

String TextShaderLanguagePlugin::get_file_extension(int p_variation_index) const {
	if (p_variation_index == 0) {
		return "gdshader";
	} else if (p_variation_index == 1) {
		return "gdshaderinc";
	}
	return "tres";
}

int TextShaderLanguagePlugin::get_template_count(Shader::Mode p_shader_mode) const {
	switch (p_shader_mode) {
		case Shader::MODE_SPATIAL: {
			return TEMPLATE_SPATIAL_MAX;
		} break;
		default: {
			return DEFAULT_TEMPLATE_MAX;
		} break;
	}
}

bool TextShaderLanguagePlugin::get_template_option(Shader::Mode p_shader_mode, int p_template_index, StringName &r_icon_name, String &r_label) const {
	switch (p_shader_mode) {
		case Shader::MODE_SPATIAL: {
			switch (p_template_index) {
				case TEMPLATE_SPATIAL_SPRITE_3D: {
					r_icon_name = SNAME("Sprite3D");
					r_label = TTR("Spatial: Sprite3D");
					return true;
				} break;
			}
		} break;
		default: {
			return false;
		} break;
	}
	return false;
}
