/**************************************************************************/
/*  visual_shader_node_standard_surface.cpp                               */
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

#include "visual_shader_node_standard_surface.h"

#include "core/variant/variant.h"
#include "scene/resources/visual_shader.h"

String VisualShaderNodeStandardSurface::get_caption() const {
	return "Standard Surface Output";
}

int VisualShaderNodeStandardSurface::get_input_port_count() const {
	return 42;
}

VisualShaderNode::PortType VisualShaderNodeStandardSurface::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_SCALAR; // base float
		case 1:
			return PORT_TYPE_VECTOR_3D; // base_color color3
		case 2:
			return PORT_TYPE_SCALAR; // diffuse_roughness float
		case 3:
			return PORT_TYPE_SCALAR; // metalness float

		case 4:
			return PORT_TYPE_SCALAR; // specular float
		case 5:
			return PORT_TYPE_VECTOR_3D; // specular_color color3
		case 6:
			return PORT_TYPE_SCALAR; // specular_roughness float
		case 7:
			return PORT_TYPE_SCALAR; // specular_IOR float
		case 8:
			return PORT_TYPE_SCALAR; // specular_anisotropy float
		case 9:
			return PORT_TYPE_SCALAR; // specular_rotation float

		case 10:
			return PORT_TYPE_SCALAR; // transmission float
		case 11:
			return PORT_TYPE_VECTOR_3D; // transmission_color color3
		case 12:
			return PORT_TYPE_SCALAR; // transmission_depth float
		case 13:
			return PORT_TYPE_VECTOR_3D; // transmission_scatter color3
		case 14:
			return PORT_TYPE_SCALAR; // transmission_scatter_anisotropy float
		case 15:
			return PORT_TYPE_SCALAR; // transmission_dispersion float
		case 16:
			return PORT_TYPE_SCALAR; // transmission_extra_roughness float

		case 17:
			return PORT_TYPE_SCALAR; // subsurface float
		case 18:
			return PORT_TYPE_VECTOR_3D; // subsurface_color color3
		case 19:
			return PORT_TYPE_VECTOR_3D; // subsurface_radius color3
		case 20:
			return PORT_TYPE_SCALAR; // subsurface_scale float
		case 21:
			return PORT_TYPE_SCALAR; // subsurface_anisotropy float

		case 22:
			return PORT_TYPE_SCALAR; // sheen float
		case 23:
			return PORT_TYPE_VECTOR_3D; // sheen_color color3
		case 24:
			return PORT_TYPE_SCALAR; // sheen_roughness float

		case 25:
			return PORT_TYPE_SCALAR; // coat float
		case 26:
			return PORT_TYPE_VECTOR_3D; // coat_color color3
		case 27:
			return PORT_TYPE_SCALAR; // coat_roughness float
		case 28:
			return PORT_TYPE_SCALAR; // coat_anisotropy float
		case 29:
			return PORT_TYPE_SCALAR; // coat_rotation float
		case 30:
			return PORT_TYPE_SCALAR; // coat_IOR float
		case 31:
			return PORT_TYPE_VECTOR_3D; // coat_normal vector3
		case 32:
			return PORT_TYPE_SCALAR; // coat_affect_color float
		case 33:
			return PORT_TYPE_SCALAR; // coat_affect_roughness float

		case 34:
			return PORT_TYPE_SCALAR; // thin_film_thickness float
		case 35:
			return PORT_TYPE_SCALAR; // thin_film_IOR float

		case 36:
			return PORT_TYPE_SCALAR; // emission float
		case 37:
			return PORT_TYPE_VECTOR_3D; // emission_color color3

		case 38:
			return PORT_TYPE_VECTOR_3D; // opacity color3

		case 39:
			return PORT_TYPE_BOOLEAN; // thin_walled boolean

		case 40:
			return PORT_TYPE_VECTOR_3D; // normal vector3
		case 41:
			return PORT_TYPE_VECTOR_3D; // tangent vector3

		default:
			return PORT_TYPE_MAX;
	}
}

String VisualShaderNodeStandardSurface::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "base";
		case 1:
			return "base_color";
		case 2:
			return "diffuse_roughness";
		case 3:
			return "metalness";

		case 4:
			return "specular";
		case 5:
			return "specular_color";
		case 6:
			return "specular_roughness";
		case 7:
			return "specular_IOR";
		case 8:
			return "specular_anisotropy";
		case 9:
			return "specular_rotation";

		case 10:
			return "transmission";
		case 11:
			return "transmission_color";
		case 12:
			return "transmission_depth";
		case 13:
			return "transmission_scatter";
		case 14:
			return "transmission_scatter_anisotropy";
		case 15:
			return "transmission_dispersion";
		case 16:
			return "transmission_extra_roughness";

		case 17:
			return "subsurface";
		case 18:
			return "subsurface_color";
		case 19:
			return "subsurface_radius";
		case 20:
			return "subsurface_scale";
		case 21:
			return "subsurface_anisotropy";

		case 22:
			return "sheen";
		case 23:
			return "sheen_color";
		case 24:
			return "sheen_roughness";

		case 25:
			return "coat";
		case 26:
			return "coat_color";
		case 27:
			return "coat_roughness";
		case 28:
			return "coat_anisotropy";
		case 29:
			return "coat_rotation";
		case 30:
			return "coat_IOR";
		case 31:
			return "coat_normal";
		case 32:
			return "coat_affect_color";
		case 33:
			return "coat_affect_roughness";

		case 34:
			return "thin_film_thickness";
		case 35:
			return "thin_film_IOR";

		case 36:
			return "emission";
		case 37:
			return "emission_color";

		case 38:
			return "opacity";

		case 39:
			return "thin_walled";

		case 40:
			return "normal";
		case 41:
			return "tangent";
		default:
			return "";
	}
}

int VisualShaderNodeStandardSurface::get_output_port_count() const {
	return 1;
}

VisualShaderNode::PortType VisualShaderNodeStandardSurface::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeStandardSurface::get_output_port_name(int p_port) const {
	return "metallic";
}

String VisualShaderNodeStandardSurface::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;

	code += vformat("float hasCoatColor = dot(%s, vec3(1, 1, 1));\n", p_input_vars[26]);

	String transmission_cond =
			"	// transmission_depth\n"
			"	if (%s > 0.0) {\n"
			"		ALBEDO = vec3(1, 1, 1);\n"
			"	} else {\n"
			"		// transmission_color\n"
			"		ALBEDO = %s;\n"
			"	}\n";

	String transmission_compiled = vformat(transmission_cond, p_input_vars[12], p_input_vars[11]);

	String base_color_cond =
			"	// base, base_color\n"
			"	vec3 base_color = %s * %s;\n"
			"	// coat_color\n"
			"	if (hasCoatColor == 0.0) {\n"
			"		ALBEDO = base_color;\n"
			"	} else {\n"
			"		//coat_color, coat\n"
			"		vec3 coatAttenuation = mix(%s, vec3(1, 1, 1), %s);\n"
			"		ALBEDO = base_color * coatAttenuation;\n"
			"	}\n";

	String base_color_compiled = vformat(base_color_cond, p_input_vars[0], p_input_vars[1], p_input_vars[26], p_input_vars[25]);

	String condition =
			"// transmission \n"
			"if (%s > 0.0) {\n" +
			transmission_compiled +
			"} else {\n" +
			base_color_compiled +
			"}\n";

	code += vformat(condition, p_input_vars[10]);

	code += vformat("%s = %s;\n", p_output_vars[0], p_input_vars[3]);
	code += vformat("ROUGHNESS = %s;\n", p_input_vars[6]);
	code += vformat("SPECULAR = %s;\n", p_input_vars[4]);

	//?
	//code += vformat("TRANSMISSION = %s;\n", p_input_vars[10]);
	//code += vformat("THICKNESS = %s;", p_input_vars[12]);
	//code += vformat("attenuation_color = %s;", p_input_vars[12]);
	//code += vformat("sheen_color = %s;", p_input_vars[12]);
	//code += vformat("sheen_roughness = %s;", p_input_vars[12]);

	String clearcoat =
			"if (hasCoatColor != 0.0) {\n"
			"	//coat, coat_color\n"
			"	vec3 coatColorScaled = %s * %s;\n"
			"	float one_third = 1.0/3.0;\n"
			"	CLEARCOAT = dot(coatColorScaled, vec3(one_third, one_third, one_third));\n"
			"} else {\n"
			"	// coat\n"
			"	CLEARCOAT = %s;\n"
			"}\n";

	code += vformat(clearcoat, p_input_vars[25], p_input_vars[26], p_input_vars[25]);
	code += vformat("CLEARCOAT_ROUGHNESS = %s;\n", p_input_vars[27]);
	code += vformat("EMISSION = %s * %s;\n", p_input_vars[36], p_input_vars[37]);

	code += vformat("NORMAL_MAP = %s;\n", p_input_vars[40]);
	code += vformat("TANGENT = %s;\n", p_input_vars[41]);

	return code;
}

VisualShaderNodeStandardSurface::VisualShaderNodeStandardSurface() {
}
