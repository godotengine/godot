/**************************************************************************/
/*  test_node_3d_editor.cpp                                               */
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

#pragma once

#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "tests/test_macros.h"
TEST_FORCE_LINK(test_node_3d_editor)

#include "scene/main/window.h"

namespace TestNode3dEditor {

TEST_CASE("[Node3dEditor][Bottom grid]") {
	SUBCASE("[Bottom grid] Bottom grid should not be drawn when camera is under y=0 and the bottom grid is deactivated") {
		Node3DEditor node_3d_editor;
		node_3d_editor.set_grid_bottom_enabled(false);
		CHECK(node_3d_editor.should_draw_bottom_grid(-1.0f) == false);
	}

	SUBCASE("[Bottom grid] Bottom grid should be drawn when camera is over y=0 and the bottom grid is deactivated") {
		Node3DEditor node_3d_editor;
		node_3d_editor.set_grid_bottom_enabled(false);
		CHECK(node_3d_editor.should_draw_bottom_grid(1.0f) == true);
	}

	SUBCASE("[Bottom grid] Bottom grid should be drawn when camera is under y=0 and the bottom grid is activated") {
		Node3DEditor node_3d_editor;
		node_3d_editor.set_grid_bottom_enabled(true);
		CHECK(node_3d_editor.should_draw_bottom_grid(-1.0f) == true);
	}
}

} // namespace TestNode3dEditor
