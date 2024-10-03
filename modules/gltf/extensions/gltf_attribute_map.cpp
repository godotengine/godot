/**************************************************************************/
/*  gltf_attribute_map.cpp                                                */
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

#include "gltf_attribute_map.h"

void GLTFAttributeMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_color"), &GLTFAttributeMap::get_color);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &GLTFAttributeMap::set_color);
	ClassDB::bind_method(D_METHOD("get_uv"), &GLTFAttributeMap::get_uv);
	ClassDB::bind_method(D_METHOD("set_uv", "uv"), &GLTFAttributeMap::set_uv);
	ClassDB::bind_method(D_METHOD("get_uv2"), &GLTFAttributeMap::get_uv2);
	ClassDB::bind_method(D_METHOD("set_uv2", "uv2"), &GLTFAttributeMap::set_uv2);
	ClassDB::bind_method(D_METHOD("get_custom0"), &GLTFAttributeMap::get_custom0);
	ClassDB::bind_method(D_METHOD("set_custom0", "custom0"), &GLTFAttributeMap::set_custom0);
	ClassDB::bind_method(D_METHOD("get_custom0_mux"), &GLTFAttributeMap::get_custom0_mux);
	ClassDB::bind_method(D_METHOD("set_custom0_mux", "custom0_mux"), &GLTFAttributeMap::set_custom0_mux);
	ClassDB::bind_method(D_METHOD("get_custom1"), &GLTFAttributeMap::get_custom1);
	ClassDB::bind_method(D_METHOD("set_custom1", "custom1"), &GLTFAttributeMap::set_custom1);
	ClassDB::bind_method(D_METHOD("get_custom1_mux"), &GLTFAttributeMap::get_custom1_mux);
	ClassDB::bind_method(D_METHOD("set_custom1_mux", "custom1_mux"), &GLTFAttributeMap::set_custom1_mux);
	ClassDB::bind_method(D_METHOD("get_custom2"), &GLTFAttributeMap::get_custom2);
	ClassDB::bind_method(D_METHOD("set_custom2", "custom2"), &GLTFAttributeMap::set_custom2);
	ClassDB::bind_method(D_METHOD("get_custom2_mux"), &GLTFAttributeMap::get_custom2_mux);
	ClassDB::bind_method(D_METHOD("set_custom2_mux", "custom2_mux"), &GLTFAttributeMap::set_custom2_mux);
	ClassDB::bind_method(D_METHOD("get_custom3"), &GLTFAttributeMap::get_custom3);
	ClassDB::bind_method(D_METHOD("set_custom3", "custom3"), &GLTFAttributeMap::set_custom3);
	ClassDB::bind_method(D_METHOD("get_custom3_mux"), &GLTFAttributeMap::get_custom3_mux);
	ClassDB::bind_method(D_METHOD("set_custom3_mux", "custom3_mux"), &GLTFAttributeMap::set_custom3_mux);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uv"), "set_uv", "get_uv");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uv2"), "set_uv2", "get_uv2");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom0"), "set_custom0", "get_custom0");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom0_mux"), "set_custom0_mux", "get_custom0_mux");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom1"), "set_custom1", "get_custom1");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom1_mux"), "set_custom1_mux", "get_custom1_mux");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom2"), "set_custom2", "get_custom2");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom2_mux"), "set_custom2_mux", "get_custom2_mux");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom3"), "set_custom3", "get_custom3");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom3_mux"), "set_custom3_mux", "get_custom3_mux");
}

String GLTFAttributeMap::get_color() const {
	return _color;
}

void GLTFAttributeMap::set_color(const String &p_color) {
	_color = p_color;
}

String GLTFAttributeMap::get_uv() const {
	return _uv;
}

void GLTFAttributeMap::set_uv(const String &p_uv) {
	_uv = p_uv;
}

String GLTFAttributeMap::get_uv2() const {
	return _uv2;
}

void GLTFAttributeMap::set_uv2(const String &p_uv2) {
	_uv2 = p_uv2;
}

String GLTFAttributeMap::get_custom0() const {
	return _custom[0];
}

void GLTFAttributeMap::set_custom0(const String &p_custom0) {
	_custom[0] = p_custom0;
}

String GLTFAttributeMap::get_custom0_mux() const {
	return _custom_mux[0];
}

void GLTFAttributeMap::set_custom0_mux(const String &p_custom0_mux) {
	_custom_mux[0] = p_custom0_mux;
}

String GLTFAttributeMap::get_custom1() const {
	return _custom[1];
}

void GLTFAttributeMap::set_custom1(const String &p_custom1) {
	_custom[1] = p_custom1;
}

String GLTFAttributeMap::get_custom1_mux() const {
	return _custom_mux[1];
}

void GLTFAttributeMap::set_custom1_mux(const String &p_custom1_mux) {
	_custom_mux[1] = p_custom1_mux;
}

String GLTFAttributeMap::get_custom2() const {
	return _custom[2];
}

void GLTFAttributeMap::set_custom2(const String &p_custom2) {
	_custom[2] = p_custom2;
}

String GLTFAttributeMap::get_custom2_mux() const {
	return _custom_mux[2];
}

void GLTFAttributeMap::set_custom2_mux(const String &p_custom2_mux) {
	_custom_mux[2] = p_custom2_mux;
}

String GLTFAttributeMap::get_custom3() const {
	return _custom[3];
}

void GLTFAttributeMap::set_custom3(const String &p_custom3) {
	_custom[3] = p_custom3;
}

String GLTFAttributeMap::get_custom3_mux() const {
	return _custom_mux[3];
}

void GLTFAttributeMap::set_custom3_mux(const String &p_custom3_mux) {
	_custom_mux[3] = p_custom3_mux;
}
