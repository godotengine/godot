/*************************************************************************/
/*  test_string.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef TEST_FBX_IMPORT_H
#define TEST_FBX_IMPORT_H
#include "core/io/ip_address.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include "main/main.h"
#include <thirdparty/doctest/doctest.h>

#include "modules/regex/regex.h"

#include <wchar.h>
//#include "core/math/math_funcs.h"
#include "test_render.h"
#include <stdio.h>

#include "core/io/resource_loader.h"
#include "core/math/math_funcs.h"
#include "core/math/quick_hull.h"
#include "core/node_path.h"
#include "core/os/keyboard.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "core/project_settings.h"
#include "scene/3d/skeleton.h"
#include "scene/resources/packed_scene.h"
#include "servers/visual_server.h"

namespace TestFbxImport {

class GodotEngineTestFixture : public MainLoop {
	float ofs;
	bool quit;

protected:
public:
	virtual void input_event(const Ref<InputEvent> &p_event) {
		if (p_event->is_pressed())
			quit = true;
	}

	virtual void init() {
		print_line("Starting up test renderer");
		ofs = 0;
		quit = false;
	}

	virtual bool iteration(float p_time) {
		// do stuff
		ofs += p_time * 0.05;
		return quit;
	}

	virtual bool idle(float p_time) {
		return quit;
	}

	virtual void finish() {
	}
};

class EngineTestFixture {
protected:
	static MainLoop *main;

public:
	EngineTestFixture() {
		// not required unless you do not supply unit-test-project --test doctest
		/// initialize the godot engine test fixture!
		//main = memnew(GodotEngineTestFixture);
	}
};

// todo: make testset which can be run without a test project - using pck on CI.
// in order to open project you must use --path unit-test-project --test doctest

TEST_CASE("[Model import] Simple bone count test") {
	Error err;
	String path_to_load = String("");
	String path_to_armature = String("");
	int expected_bone_count = 0;

	// note: every sub case causes a re-execution of test
	SUBCASE("") {
		path_to_load = String("res://Models/ANA-SingleRoot.fbx");
		path_to_armature = String("ANA ARMATURE/ROOT/Skeleton");
		expected_bone_count = 10;
	}

	// we can declare as many models as we like to test.
	SUBCASE("") {
		path_to_load = String("res://Models/BIRD.fbx");
		path_to_armature = String("Armature/Skeleton");
		expected_bone_count = 13;
	}

	Ref<PackedScene> scene = ResourceLoader::load(path_to_load, "PackedScene", false, &err);

	// check that resource can load
	CHECK(err == OK);

	// instance scene
	Node *ptr = scene->instance();
	ptr->print_tree_pretty();
	OS::get_singleton()->print("load successful filename %ls\n", ptr->get_filename().c_str());

	Node *skeleton_node = ptr->get_node(NodePath(path_to_armature));
	Skeleton *skeleton = Object::cast_to<Skeleton>(skeleton_node);
	CHECK(skeleton);

	OS::get_singleton()->print("bone count: %d\n", skeleton->get_bone_count());

	// ana has 10 bones
	CHECK(skeleton->get_bone_count() == expected_bone_count); // has bone check
}

} // namespace TestFbxImport

#endif // TEST_FBX_IMPORT_H
