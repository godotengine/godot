/**************************************************************************/
/*  gltf_light.cpp                                                        */
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

#include "gltf_light.h"

#include "scene/3d/light.h"

void GLTFLight::_bind_methods() {
	ClassDB::bind_method(D_METHOD("to_node"), &GLTFLight::to_node);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFLight::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_color"), &GLTFLight::get_color);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &GLTFLight::set_color);
	ClassDB::bind_method(D_METHOD("get_intensity"), &GLTFLight::get_intensity);
	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &GLTFLight::set_intensity);
	ClassDB::bind_method(D_METHOD("get_type"), &GLTFLight::get_type);
	ClassDB::bind_method(D_METHOD("set_type", "type"), &GLTFLight::set_type);
	ClassDB::bind_method(D_METHOD("get_range"), &GLTFLight::get_range);
	ClassDB::bind_method(D_METHOD("set_range", "range"), &GLTFLight::set_range);
	ClassDB::bind_method(D_METHOD("get_inner_cone_angle"), &GLTFLight::get_inner_cone_angle);
	ClassDB::bind_method(D_METHOD("set_inner_cone_angle", "inner_cone_angle"), &GLTFLight::set_inner_cone_angle);
	ClassDB::bind_method(D_METHOD("get_outer_cone_angle"), &GLTFLight::get_outer_cone_angle);
	ClassDB::bind_method(D_METHOD("set_outer_cone_angle", "outer_cone_angle"), &GLTFLight::set_outer_cone_angle);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color"); // Color
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "intensity"), "set_intensity", "get_intensity"); // float
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "type"), "set_type", "get_type"); // String
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "range"), "set_range", "get_range"); // float
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "inner_cone_angle"), "set_inner_cone_angle", "get_inner_cone_angle"); // float
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "outer_cone_angle"), "set_outer_cone_angle", "get_outer_cone_angle"); // float
}

Color GLTFLight::get_color() {
	return color;
}

void GLTFLight::set_color(Color p_color) {
	color = p_color;
}

float GLTFLight::get_intensity() {
	return intensity;
}

void GLTFLight::set_intensity(float p_intensity) {
	intensity = p_intensity;
}

String GLTFLight::get_type() {
	return type;
}

void GLTFLight::set_type(String p_type) {
	type = p_type;
}

float GLTFLight::get_range() {
	return range;
}

void GLTFLight::set_range(float p_range) {
	range = p_range;
}

float GLTFLight::get_inner_cone_angle() {
	return inner_cone_angle;
}

void GLTFLight::set_inner_cone_angle(float p_inner_cone_angle) {
	inner_cone_angle = p_inner_cone_angle;
}

float GLTFLight::get_outer_cone_angle() {
	return outer_cone_angle;
}

void GLTFLight::set_outer_cone_angle(float p_outer_cone_angle) {
	outer_cone_angle = p_outer_cone_angle;
}

Ref<GLTFLight> GLTFLight::from_node(const Light *p_light) {
	Ref<GLTFLight> l;
	l.instance();
	ERR_FAIL_COND_V_MSG(!p_light, l, "Tried to create a GLTFLight from a Light node, but the given node was null.");
	l->color = p_light->get_color();
	if (cast_to<const DirectionalLight>(p_light)) {
		l->type = "directional";
		const DirectionalLight *light = cast_to<const DirectionalLight>(p_light);
		l->intensity = light->get_param(DirectionalLight::PARAM_ENERGY);
		l->range = FLT_MAX; // Range for directional lights is infinite in Godot.
	} else if (cast_to<const OmniLight>(p_light)) {
		l->type = "point";
		const OmniLight *light = cast_to<const OmniLight>(p_light);
		l->range = light->get_param(OmniLight::PARAM_RANGE);
		l->intensity = light->get_param(OmniLight::PARAM_ENERGY);
	} else if (cast_to<const SpotLight>(p_light)) {
		l->type = "spot";
		const SpotLight *light = cast_to<const SpotLight>(p_light);
		l->range = light->get_param(SpotLight::PARAM_RANGE);
		l->intensity = light->get_param(SpotLight::PARAM_ENERGY);
		l->outer_cone_angle = Math::deg2rad(light->get_param(SpotLight::PARAM_SPOT_ANGLE));
		// This equation is the inverse of the import equation (which has a desmos link).
		float angle_ratio = 1 - (0.2 / (0.1 + light->get_param(SpotLight::PARAM_SPOT_ATTENUATION)));
		angle_ratio = MAX(0, angle_ratio);
		l->inner_cone_angle = l->outer_cone_angle * angle_ratio;
	}
	return l;
}

Light *GLTFLight::to_node() const {
	if (type == "directional") {
		DirectionalLight *light = memnew(DirectionalLight);
		light->set_param(Light::PARAM_ENERGY, intensity);
		light->set_color(color);
		return light;
	}
	if (type == "point") {
		OmniLight *light = memnew(OmniLight);
		light->set_param(OmniLight::PARAM_ENERGY, intensity);
		light->set_param(OmniLight::PARAM_RANGE, CLAMP(range, 0, 4096));
		light->set_color(color);
		return light;
	}
	if (type == "spot") {
		SpotLight *light = memnew(SpotLight);
		light->set_param(SpotLight::PARAM_ENERGY, intensity);
		light->set_param(SpotLight::PARAM_RANGE, CLAMP(range, 0, 4096));
		light->set_param(SpotLight::PARAM_SPOT_ANGLE, Math::rad2deg(outer_cone_angle));
		light->set_color(color);
		// Line of best fit derived from guessing, see https://www.desmos.com/calculator/biiflubp8b
		// The points in desmos are not exact, except for (1, infinity).
		float angle_ratio = inner_cone_angle / outer_cone_angle;
		float angle_attenuation = 0.2 / (1 - angle_ratio) - 0.1;
		light->set_param(SpotLight::PARAM_SPOT_ATTENUATION, angle_attenuation);
		return light;
	}
	return memnew(Light);
}

Ref<GLTFLight> GLTFLight::from_dictionary(const Dictionary p_dictionary) {
	ERR_FAIL_COND_V_MSG(!p_dictionary.has("type"), Ref<GLTFLight>(), "Failed to parse GLTF light, missing required field 'type'.");
	Ref<GLTFLight> light;
	light.instance();
	const String &type = p_dictionary["type"];
	light->type = type;

	if (p_dictionary.has("color")) {
		const Array &arr = p_dictionary["color"];
		if (arr.size() == 3) {
			light->color = Color(arr[0], arr[1], arr[2]).to_srgb();
		} else {
			ERR_PRINT("Error parsing GLTF light: The color must have exactly 3 numbers.");
		}
	}
	if (p_dictionary.has("intensity")) {
		light->intensity = p_dictionary["intensity"];
	}
	if (p_dictionary.has("range")) {
		light->range = p_dictionary["range"];
	}
	if (type == "spot") {
		const Dictionary &spot = p_dictionary["spot"];
		light->inner_cone_angle = spot["innerConeAngle"];
		light->outer_cone_angle = spot["outerConeAngle"];
		if (light->inner_cone_angle >= light->outer_cone_angle) {
			ERR_PRINT("Error parsing GLTF light: The inner angle must be smaller than the outer angle.");
		}
	} else if (type != "point" && type != "directional") {
		ERR_PRINT("Error parsing GLTF light: Light type '" + type + "' is unknown.");
	}
	return light;
}

Dictionary GLTFLight::to_dictionary() const {
	Dictionary d;
	Array color_array;
	color_array.resize(3);
	color_array[0] = color.r;
	color_array[1] = color.g;
	color_array[2] = color.b;
	d["color"] = color_array;
	d["type"] = type;
	if (type == "spot") {
		Dictionary spot_dict;
		spot_dict["innerConeAngle"] = inner_cone_angle;
		spot_dict["outerConeAngle"] = outer_cone_angle;
		d["spot"] = spot_dict;
	}
	d["intensity"] = intensity;
	d["range"] = range;
	return d;
}
