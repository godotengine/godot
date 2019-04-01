/*************************************************************************/
/*  register_types.cpp                                                   */
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

#include "register_types.h"
#include "test_case.h"
#include "test_config.h"
#include "test_plugin.h"
#include "test_result.h"
#include "test_runner.h"
#include "text_test_result.h"

#include "core/class_db.h"
#include "editor/editor_node.h"

static void _editor_init() {
	TestPlugin *test_plugin = memnew(TestPlugin);
	EditorNode::get_singleton()->add_editor_plugin(test_plugin);
}

static TestConfig *test_config;

void register_unittest_types() {
	ClassDB::register_class<TestCase>();
	ClassDB::register_class<TestConfig>();
	ClassDB::register_class<TestResult>();
	ClassDB::register_class<TestRunner>();
	ClassDB::register_class<TextTestResult>();

	test_config = memnew(TestConfig);

	Engine::get_singleton()->add_singleton(Engine::Singleton("TestConfig", TestConfig::get_singleton()));

#ifdef TOOLS_ENABLED
	EditorNode::add_init_callback(_editor_init);
#endif
}

void unregister_unittest_types() {
	memfree(test_config);
}
