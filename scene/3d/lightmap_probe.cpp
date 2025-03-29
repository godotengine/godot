/**************************************************************************/
/*  lightmap_probe.cpp                                                    */
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

#include "lightmap_probe.h"

void LightmapProbe::set_size(Vector3 p_size) {
	size = Vector3(MAX(0.001, p_size.x), MAX(0.001, p_size.y), MAX(0.001, p_size.z));
	update_gizmos();
}

Vector3 LightmapProbe::get_size() const {
	return size;
}

void LightmapProbe::set_cell_size(Vector3 p_cell_size) {
	cell_size = Vector3(MAX(0.001, p_cell_size.x), MAX(0.001, p_cell_size.y), MAX(0.001, p_cell_size.z));
	update_gizmos();
}

Vector3 LightmapProbe::get_cell_size() const {
	return cell_size;
}

void LightmapProbe::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &LightmapProbe::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &LightmapProbe::get_size);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &LightmapProbe::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &LightmapProbe::get_cell_size);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "cell_size"), "set_cell_size", "get_cell_size");
}

LightmapProbe::LightmapProbe() {
}
