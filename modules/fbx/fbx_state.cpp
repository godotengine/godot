/**************************************************************************/
/*  fbx_state.cpp                                                         */
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

#include "fbx_state.h"

void FBXState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_allow_geometry_helper_nodes"), &FBXState::get_allow_geometry_helper_nodes);
	ClassDB::bind_method(D_METHOD("set_allow_geometry_helper_nodes", "allow"), &FBXState::set_allow_geometry_helper_nodes);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_geometry_helper_nodes"), "set_allow_geometry_helper_nodes", "get_allow_geometry_helper_nodes");
}

bool FBXState::get_allow_geometry_helper_nodes() {
	return allow_geometry_helper_nodes;
}

void FBXState::set_allow_geometry_helper_nodes(bool p_allow_geometry_helper_nodes) {
	allow_geometry_helper_nodes = p_allow_geometry_helper_nodes;
}
