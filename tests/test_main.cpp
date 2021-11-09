/*************************************************************************/
/*  test_main.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "test_main.h"

#include "tests/core/io/test_config_file.h"
#include "tests/core/io/test_file_access.h"
#include "tests/core/io/test_image.h"
#include "tests/core/io/test_json.h"
#include "tests/core/io/test_marshalls.h"
#include "tests/core/io/test_pck_packer.h"
#include "tests/core/io/test_resource.h"
#include "tests/core/io/test_xml_parser.h"
#include "tests/core/math/test_aabb.h"
#include "tests/core/math/test_astar.h"
#include "tests/core/math/test_basis.h"
#include "tests/core/math/test_color.h"
#include "tests/core/math/test_expression.h"
#include "tests/core/math/test_geometry_2d.h"
#include "tests/core/math/test_geometry_3d.h"
#include "tests/core/math/test_math.h"
#include "tests/core/math/test_random_number_generator.h"
#include "tests/core/math/test_rect2.h"
#include "tests/core/object/test_class_db.h"
#include "tests/core/object/test_method_bind.h"
#include "tests/core/object/test_object.h"
#include "tests/core/string/test_node_path.h"
#include "tests/core/string/test_string.h"
#include "tests/core/string/test_translation.h"
#include "tests/core/templates/test_command_queue.h"
#include "tests/core/templates/test_list.h"
#include "tests/core/templates/test_local_vector.h"
#include "tests/core/templates/test_lru.h"
#include "tests/core/templates/test_oa_hash_map.h"
#include "tests/core/templates/test_ordered_hash_map.h"
#include "tests/core/templates/test_paged_array.h"
#include "tests/core/templates/test_vector.h"
#include "tests/core/test_crypto.h"
#include "tests/core/test_hashing_context.h"
#include "tests/core/test_time.h"
#include "tests/core/variant/test_array.h"
#include "tests/core/variant/test_dictionary.h"
#include "tests/core/variant/test_variant.h"
#include "tests/scene/test_code_edit.h"
#include "tests/scene/test_curve.h"
#include "tests/scene/test_gradient.h"
#include "tests/scene/test_gui.h"
#include "tests/scene/test_path_3d.h"
#include "tests/servers/test_physics_2d.h"
#include "tests/servers/test_physics_3d.h"
#include "tests/servers/test_render.h"
#include "tests/servers/test_shader_lang.h"
#include "tests/servers/test_text_server.h"
#include "tests/test_validate_testing.h"

#include "modules/modules_tests.gen.h"

#include "tests/test_macros.h"

#include "scene/resources/default_theme/default_theme.h"

int test_main(int argc, char *argv[]) {
	bool run_tests = true;

	// Convert arguments to Godot's command-line.
	List<String> args;

	for (int i = 0; i < argc; i++) {
		args.push_back(String::utf8(argv[i]));
	}
	OS::get_singleton()->set_cmdline("", args);

	// Run custom test tools.
	if (test_commands) {
		for (Map<String, TestFunc>::Element *E = test_commands->front(); E; E = E->next()) {
			if (args.find(E->key())) {
				const TestFunc &test_func = E->get();
				test_func();
				run_tests = false;
				break;
			}
		}
		if (!run_tests) {
			delete test_commands;
			return 0;
		}
	}
	// Doctest runner.
	doctest::Context test_context;
	List<String> test_args;

	// Clean arguments of "--test" from the args.
	for (int x = 0; x < argc; x++) {
		String arg = String(argv[x]);
		if (arg != "--test") {
			test_args.push_back(arg);
		}
	}

	if (test_args.size() > 0) {
		// Convert Godot command line arguments back to standard arguments.
		char **doctest_args = new char *[test_args.size()];
		for (int x = 0; x < test_args.size(); x++) {
			// Operation to convert Godot string to non wchar string.
			CharString cs = test_args[x].utf8();
			const char *str = cs.get_data();
			// Allocate the string copy.
			doctest_args[x] = new char[strlen(str) + 1];
			// Copy this into memory.
			memcpy(doctest_args[x], str, strlen(str) + 1);
		}

		test_context.applyCommandLine(test_args.size(), doctest_args);

		for (int x = 0; x < test_args.size(); x++) {
			delete[] doctest_args[x];
		}
		delete[] doctest_args;
	}

	return test_context.run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering/rendering_server_default.h"

struct GodotTestCaseListener : public doctest::IReporter {
	GodotTestCaseListener(const doctest::ContextOptions &p_in) {}

	SignalWatcher *signal_watcher = nullptr;

	PhysicsServer3D *physics_3d_server = nullptr;
	PhysicsServer2D *physics_2d_server = nullptr;
	NavigationServer3D *navigation_3d_server = nullptr;
	NavigationServer2D *navigation_2d_server = nullptr;

	void test_case_start(const doctest::TestCaseData &p_in) override {
		SignalWatcher::get_singleton()->_clear_signals();

		String name = String(p_in.m_name);

		if (name.find("[SceneTree]") != -1) {
			GLOBAL_DEF("memory/limits/multithreaded_server/rid_pool_prealloc", 60);
			memnew(MessageQueue);

			GLOBAL_DEF("internationalization/rendering/force_right_to_left_layout_direction", false);

			Error err = OK;
			OS::get_singleton()->set_has_server_feature_callback(nullptr);
			for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
				if (String("headless") == DisplayServer::get_create_function_name(i)) {
					DisplayServer::create(i, "", DisplayServer::WindowMode::WINDOW_MODE_MINIMIZED, DisplayServer::VSyncMode::VSYNC_ENABLED, 0, Vector2i(0, 0), err);
					break;
				}
			}
			memnew(RenderingServerDefault());
			RenderingServerDefault::get_singleton()->init();
			RenderingServerDefault::get_singleton()->set_render_loop_enabled(false);

			physics_3d_server = PhysicsServer3DManager::new_default_server();
			physics_3d_server->init();

			physics_2d_server = PhysicsServer2DManager::new_default_server();
			physics_2d_server->init();

			navigation_3d_server = NavigationServer3DManager::new_default_server();
			navigation_2d_server = memnew(NavigationServer2D);

			memnew(InputMap);
			InputMap::get_singleton()->load_default();

			make_default_theme(false, Ref<Font>());

			memnew(SceneTree);
			SceneTree::get_singleton()->initialize();
			return;
		}
	}

	void test_case_end(const doctest::CurrentTestCaseStats &) override {
		if (SceneTree::get_singleton()) {
			SceneTree::get_singleton()->finalize();
		}

		if (MessageQueue::get_singleton()) {
			MessageQueue::get_singleton()->flush();
		}

		if (SceneTree::get_singleton()) {
			memdelete(SceneTree::get_singleton());
		}

		clear_default_theme();

		if (navigation_3d_server) {
			memdelete(navigation_3d_server);
			navigation_3d_server = nullptr;
		}

		if (navigation_2d_server) {
			memdelete(navigation_2d_server);
			navigation_2d_server = nullptr;
		}

		if (physics_3d_server) {
			physics_3d_server->finish();
			memdelete(physics_3d_server);
			physics_3d_server = nullptr;
		}

		if (physics_2d_server) {
			physics_2d_server->finish();
			memdelete(physics_2d_server);
			physics_2d_server = nullptr;
		}

		if (RenderingServer::get_singleton()) {
			RenderingServer::get_singleton()->sync();
			RenderingServer::get_singleton()->global_variables_clear();
			RenderingServer::get_singleton()->finish();
			memdelete(RenderingServer::get_singleton());
		}

		if (DisplayServer::get_singleton()) {
			memdelete(DisplayServer::get_singleton());
		}

		if (InputMap::get_singleton()) {
			memdelete(InputMap::get_singleton());
		}

		if (MessageQueue::get_singleton()) {
			MessageQueue::get_singleton()->flush();
			memdelete(MessageQueue::get_singleton());
		}
	}

	void test_run_start() override {
		signal_watcher = memnew(SignalWatcher);
	}

	void test_run_end(const doctest::TestRunStats &) override {
		memdelete(signal_watcher);
	}

	void test_case_reenter(const doctest::TestCaseData &) override {
		SignalWatcher::get_singleton()->_clear_signals();
	}

	void subcase_start(const doctest::SubcaseSignature &) override {
		SignalWatcher::get_singleton()->_clear_signals();
	}

	void report_query(const doctest::QueryData &) override {}
	void test_case_exception(const doctest::TestCaseException &) override {}
	void subcase_end() override {}

	void log_assert(const doctest::AssertData &in) override {}
	void log_message(const doctest::MessageData &) override {}
	void test_case_skipped(const doctest::TestCaseData &) override {}
};

REGISTER_LISTENER("GodotTestCaseListener", 1, GodotTestCaseListener);
