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
#include "main/main.h"
#include "core/io/ip_address.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include <thirdparty/doctest/doctest.h>

#include "modules/regex/regex.h"

#include <wchar.h>
//#include "core/math/math_funcs.h"
#include <stdio.h>
#include "test_render.h"

#include "core/math/math_funcs.h"
#include "core/math/quick_hull.h"
#include "core/os/keyboard.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "servers/visual_server.h"
#include "core/io/resource_loader.h"
#include "core/project_settings.h"
#include "core/node_path.h"
#include "scene/resources/packed_scene.h"
#include "scene/3d/skeleton.h"

namespace TestFbxImport {

class GodotEngineTestFixture : public MainLoop {

	RID test_cube;
	RID instance;
	RID camera;
	RID viewport;
	RID light;
	RID scenario;

	struct InstanceInfo {

		RID instance;
		Transform base;
		Vector3 rot_axis;
	};

	List<InstanceInfo> instances;

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
		VisualServer *vs = VisualServer::get_singleton();
		test_cube = vs->get_test_cube();
		scenario = vs->scenario_create();
		
		ofs = 0;
		quit = false;
	}

	virtual bool iteration(float p_time) {

		VisualServer *vs = VisualServer::get_singleton();
		//Transform t;
		//t.rotate(Vector3(0, 1, 0), ofs);
		//t.translate(Vector3(0,0,20 ));
		//vs->camera_set_transform(camera, t);

		ofs += p_time * 0.05;

		//return quit;

		for (List<InstanceInfo>::Element *E = instances.front(); E; E = E->next()) {

			Transform pre(Basis(E->get().rot_axis, ofs), Vector3());
			vs->instance_set_transform(E->get().instance, pre * E->get().base);
			/*
			if( !E->next() ) {

				vs->free( E->get().instance );
				instances.erase(E );
			}*/
		}

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

TEST_CASE("[Model import] Godot initialisation test") {
	// Do some stuff
	OS::get_singleton()->print("This test should always pass\n");

	// Load some resource / FBX
	//Ref<PackedScene> scene = ResourceLoader::get_singleton()->load("/home/gordon/Projects/CorpSquad/CorpSquad.ModelTest/Models/ANA-SingleRoot.fbx", "", false);
	//Error e = ProjectSettings::get_singleton()->setup("/home/gordon/Projects/CorpSquad/CorpSquad.ModelTest/project.godot", String(""), false);
	//OS::get_singleton()->print("Error code: %d", (uint)e);
	//CHECK(e == Error::OK);
	Error err;
	Ref<PackedScene> scene = ResourceLoader::load("res://Models/ANA-SingleRoot.fbx", "PackedScene", false, &err);
	
	// check that resource can load
	CHECK(err == OK);

	// instance scene
	Node *ptr = scene->instance();
	ptr->print_tree_pretty();
	OS::get_singleton()->print("load successful filename %ls\n", ptr->get_filename().c_str() );

	Node * skeletonNode = ptr->get_node(NodePath(String("ANA ARMATURE/ROOT/Skeleton")));
	Skeleton * skeleton = Object::cast_to<Skeleton>(skeletonNode);
	CHECK(skeleton);

	OS::get_singleton()->print("bone count: %d\n", skeleton->get_bone_count());

	// ana has 10 bones
	CHECK( skeleton->get_bone_count() == 10 ); // has bone check

	
	//CHECK(ptr->data.filename == "blah"); // expect fail
	// ensure object is valid
	//CHECK(object);			
}

} // namespace TestFbxImport

#endif // TEST_FBX_IMPORT_H
