/**************************************************************************/
/*  usd_light.cpp                                                         */
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

#include "usd_light.h"

void USDLight::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_type"), &USDLight::get_type);
	ClassDB::bind_method(D_METHOD("set_type", "type"), &USDLight::set_type);
	ClassDB::bind_method(D_METHOD("get_color"), &USDLight::get_color);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &USDLight::set_color);
	ClassDB::bind_method(D_METHOD("get_intensity"), &USDLight::get_intensity);
	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &USDLight::set_intensity);
	ClassDB::bind_method(D_METHOD("get_exposure"), &USDLight::get_exposure);
	ClassDB::bind_method(D_METHOD("set_exposure", "exposure"), &USDLight::set_exposure);
	ClassDB::bind_method(D_METHOD("get_radius"), &USDLight::get_radius);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &USDLight::set_radius);
	ClassDB::bind_method(D_METHOD("get_width"), &USDLight::get_width);
	ClassDB::bind_method(D_METHOD("set_width", "width"), &USDLight::set_width);
	ClassDB::bind_method(D_METHOD("get_height"), &USDLight::get_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &USDLight::set_height);
	ClassDB::bind_method(D_METHOD("get_cone_angle"), &USDLight::get_cone_angle);
	ClassDB::bind_method(D_METHOD("set_cone_angle", "cone_angle"), &USDLight::set_cone_angle);
	ClassDB::bind_method(D_METHOD("get_cone_softness"), &USDLight::get_cone_softness);
	ClassDB::bind_method(D_METHOD("set_cone_softness", "cone_softness"), &USDLight::set_cone_softness);
	ClassDB::bind_method(D_METHOD("get_cast_shadows"), &USDLight::get_cast_shadows);
	ClassDB::bind_method(D_METHOD("set_cast_shadows", "cast_shadows"), &USDLight::set_cast_shadows);
	ClassDB::bind_method(D_METHOD("get_dome_texture"), &USDLight::get_dome_texture);
	ClassDB::bind_method(D_METHOD("set_dome_texture", "path"), &USDLight::set_dome_texture);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, "Distant,Sphere,Disk,Rect,Cylinder,Dome"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "intensity"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "exposure"), "set_exposure", "get_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cone_angle"), "set_cone_angle", "get_cone_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cone_softness"), "set_cone_softness", "get_cone_softness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cast_shadows"), "set_cast_shadows", "get_cast_shadows");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "dome_texture"), "set_dome_texture", "get_dome_texture");

	BIND_ENUM_CONSTANT(DISTANT);
	BIND_ENUM_CONSTANT(SPHERE);
	BIND_ENUM_CONSTANT(DISK);
	BIND_ENUM_CONSTANT(RECT);
	BIND_ENUM_CONSTANT(CYLINDER);
	BIND_ENUM_CONSTANT(DOME);
}

USDLight::LightType USDLight::get_type() const {
	return type;
}

void USDLight::set_type(LightType p_type) {
	type = p_type;
}

Color USDLight::get_color() const {
	return color;
}

void USDLight::set_color(const Color &p_color) {
	color = p_color;
}

float USDLight::get_intensity() const {
	return intensity;
}

void USDLight::set_intensity(float p_intensity) {
	intensity = p_intensity;
}

float USDLight::get_exposure() const {
	return exposure;
}

void USDLight::set_exposure(float p_exposure) {
	exposure = p_exposure;
}

float USDLight::get_radius() const {
	return radius;
}

void USDLight::set_radius(float p_radius) {
	radius = p_radius;
}

float USDLight::get_width() const {
	return width;
}

void USDLight::set_width(float p_width) {
	width = p_width;
}

float USDLight::get_height() const {
	return height;
}

void USDLight::set_height(float p_height) {
	height = p_height;
}

float USDLight::get_cone_angle() const {
	return cone_angle;
}

void USDLight::set_cone_angle(float p_cone_angle) {
	cone_angle = p_cone_angle;
}

float USDLight::get_cone_softness() const {
	return cone_softness;
}

void USDLight::set_cone_softness(float p_cone_softness) {
	cone_softness = p_cone_softness;
}

bool USDLight::get_cast_shadows() const {
	return cast_shadows;
}

void USDLight::set_cast_shadows(bool p_cast_shadows) {
	cast_shadows = p_cast_shadows;
}

String USDLight::get_dome_texture() const {
	return dome_texture;
}

void USDLight::set_dome_texture(const String &p_path) {
	dome_texture = p_path;
}
