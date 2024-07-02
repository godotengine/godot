/**************************************************************************/
/*  test_mesh_library.h                                                   */
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

#ifndef TEST_MESH_LIBRARY_H
#define TEST_MESH_LIBRARY_H

#include "core/variant/variant.h"
#include "scene/resources/3d/mesh_library.h"

#include "tests/test_macros.h"

namespace TestMeshLibrary {

TEST_CASE("[SceneTree][MeshLibrary]") {
	SUBCASE("[MeshLibrary][Custom Data]") {
		MeshLibrary *test_node = memnew(MeshLibrary);

		test_node->create_item(0);
		test_node->add_custom_data_layer();
		test_node->set_custom_data_layer_name(0, "test_name");
		test_node->set_custom_data_layer_type(0, Variant::Type::STRING);

		test_node->set_custom_data(0, "test_name", "test_value");

		CHECK_EQ(test_node->get_custom_data_layers_count(), 1);
		CHECK_EQ(test_node->get_custom_data_layer_name(0), "test_name");
		CHECK_EQ(test_node->get_custom_data_layer_type(0), Variant::Type::STRING);
		CHECK_EQ(test_node->get_custom_data(0, "test_name"), "test_value");

		memdelete(test_node);
	}
}

} // namespace TestMeshLibrary

#endif // TEST_MESH_LIBRARY_H
