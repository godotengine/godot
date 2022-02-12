/*************************************************************************/
/*  test_node_path.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TEST_NODE_PATH_H
#define TEST_NODE_PATH_H

#include "core/string/node_path.h"

#include "tests/test_macros.h"

namespace TestNodePath {

TEST_CASE("[NodePath] Relative path") {
	const NodePath node_path_relative = NodePath("Path2D/PathFollow2D/Sprite2D:position:x");

	CHECK_MESSAGE(
			node_path_relative.get_as_property_path() == NodePath(":Path2D/PathFollow2D/Sprite2D:position:x"),
			"The returned property path should match the expected value.");
	CHECK_MESSAGE(
			node_path_relative.get_concatenated_subnames() == "position:x",
			"The returned concatenated subnames should match the expected value.");

	CHECK_MESSAGE(
			node_path_relative.get_name(0) == "Path2D",
			"The returned name at index 0 should match the expected value.");
	CHECK_MESSAGE(
			node_path_relative.get_name(1) == "PathFollow2D",
			"The returned name at index 1 should match the expected value.");
	CHECK_MESSAGE(
			node_path_relative.get_name(2) == "Sprite2D",
			"The returned name at index 2 should match the expected value.");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			node_path_relative.get_name(3) == "",
			"The returned name at invalid index 3 should match the expected value.");
	CHECK_MESSAGE(
			node_path_relative.get_name(-1) == "",
			"The returned name at invalid index -1 should match the expected value.");
	ERR_PRINT_ON;

	CHECK_MESSAGE(
			node_path_relative.get_name_count() == 3,
			"The returned number of names should match the expected value.");

	CHECK_MESSAGE(
			node_path_relative.get_subname(0) == "position",
			"The returned subname at index 0 should match the expected value.");
	CHECK_MESSAGE(
			node_path_relative.get_subname(1) == "x",
			"The returned subname at index 1 should match the expected value.");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			node_path_relative.get_subname(2) == "",
			"The returned subname at invalid index 2 should match the expected value.");
	CHECK_MESSAGE(
			node_path_relative.get_subname(-1) == "",
			"The returned subname at invalid index -1 should match the expected value.");
	ERR_PRINT_ON;

	CHECK_MESSAGE(
			node_path_relative.get_subname_count() == 2,
			"The returned number of subnames should match the expected value.");

	CHECK_MESSAGE(
			!node_path_relative.is_absolute(),
			"The node path should be considered relative.");

	CHECK_MESSAGE(
			!node_path_relative.is_empty(),
			"The node path shouldn't be considered empty.");
}

TEST_CASE("[NodePath] Absolute path") {
	const NodePath node_path_aboslute = NodePath("/root/Sprite2D");

	CHECK_MESSAGE(
			node_path_aboslute.get_as_property_path() == NodePath(":root/Sprite2D"),
			"The returned property path should match the expected value.");
	CHECK_MESSAGE(
			node_path_aboslute.get_concatenated_subnames() == "",
			"The returned concatenated subnames should match the expected value.");

	CHECK_MESSAGE(
			node_path_aboslute.get_name(0) == "root",
			"The returned name at index 0 should match the expected value.");
	CHECK_MESSAGE(
			node_path_aboslute.get_name(1) == "Sprite2D",
			"The returned name at index 1 should match the expected value.");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			node_path_aboslute.get_name(2) == "",
			"The returned name at invalid index 2 should match the expected value.");
	CHECK_MESSAGE(
			node_path_aboslute.get_name(-1) == "",
			"The returned name at invalid index -1 should match the expected value.");
	ERR_PRINT_ON;

	CHECK_MESSAGE(
			node_path_aboslute.get_name_count() == 2,
			"The returned number of names should match the expected value.");

	CHECK_MESSAGE(
			node_path_aboslute.get_subname_count() == 0,
			"The returned number of subnames should match the expected value.");

	CHECK_MESSAGE(
			node_path_aboslute.is_absolute(),
			"The node path should be considered absolute.");

	CHECK_MESSAGE(
			!node_path_aboslute.is_empty(),
			"The node path shouldn't be considered empty.");
}

TEST_CASE("[NodePath] Empty path") {
	const NodePath node_path_empty = NodePath();

	CHECK_MESSAGE(
			node_path_empty.get_as_property_path() == NodePath(),
			"The returned property path should match the expected value.");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			node_path_empty.get_concatenated_subnames() == "",
			"The returned concatenated subnames should match the expected value.");
	ERR_PRINT_ON;

	CHECK_MESSAGE(
			node_path_empty.get_name_count() == 0,
			"The returned number of names should match the expected value.");

	CHECK_MESSAGE(
			node_path_empty.get_subname_count() == 0,
			"The returned number of subnames should match the expected value.");

	CHECK_MESSAGE(
			!node_path_empty.is_absolute(),
			"The node path shouldn't be considered absolute.");

	CHECK_MESSAGE(
			node_path_empty.is_empty(),
			"The node path should be considered empty.");
}
} // namespace TestNodePath

#endif // TEST_NODE_PATH_H
