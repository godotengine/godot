/*************************************************************************/
/*  test_main.cpp                                                        */
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

#include "test_main.h"

#include "tests/core/input/test_input_event_key.h"
#include "tests/core/input/test_shortcut.h"
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
#include "tests/core/math/test_plane.h"
#include "tests/core/math/test_quaternion.h"
#include "tests/core/math/test_random_number_generator.h"
#include "tests/core/math/test_rect2.h"
#include "tests/core/math/test_rect2i.h"
#include "tests/core/math/test_transform_2d.h"
#include "tests/core/math/test_transform_3d.h"
#include "tests/core/math/test_vector2.h"
#include "tests/core/math/test_vector2i.h"
#include "tests/core/math/test_vector3.h"
#include "tests/core/math/test_vector3i.h"
#include "tests/core/math/test_vector4.h"
#include "tests/core/math/test_vector4i.h"
#include "tests/core/object/test_class_db.h"
#include "tests/core/object/test_method_bind.h"
#include "tests/core/object/test_object.h"
#include "tests/core/os/test_os.h"
#include "tests/core/string/test_node_path.h"
#include "tests/core/string/test_string.h"
#include "tests/core/string/test_translation.h"
#include "tests/core/templates/test_command_queue.h"
#include "tests/core/templates/test_hash_map.h"
#include "tests/core/templates/test_hash_set.h"
#include "tests/core/templates/test_list.h"
#include "tests/core/templates/test_local_vector.h"
#include "tests/core/templates/test_lru.h"
#include "tests/core/templates/test_paged_array.h"
#include "tests/core/templates/test_rid.h"
#include "tests/core/templates/test_vector.h"
#include "tests/core/test_crypto.h"
#include "tests/core/test_hashing_context.h"
#include "tests/core/test_time.h"
#include "tests/core/threads/test_worker_thread_pool.h"
#include "tests/core/variant/test_array.h"
#include "tests/core/variant/test_dictionary.h"
#include "tests/core/variant/test_variant.h"
#include "tests/scene/test_animation.h"
#include "tests/scene/test_audio_stream_wav.h"
#include "tests/scene/test_bit_map.h"
#include "tests/scene/test_code_edit.h"
#include "tests/scene/test_curve.h"
#include "tests/scene/test_gradient.h"
#include "tests/scene/test_path_3d.h"
#include "tests/scene/test_sprite_frames.h"
#include "tests/scene/test_text_edit.h"
#include "tests/scene/test_theme.h"
#include "tests/servers/test_text_server.h"
#include "tests/test_validate_testing.h"

#include "modules/modules_tests.gen.h"

#include "tests/test_macros.h"

#include "scene/theme/theme_db.h"
#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering/rendering_server_default.h"

int test_main(int argc, char *argv[]) {
	bool run_tests = true;

	// Convert arguments to Godot's command-line.
	List<String> args;

	for (int i = 0; i < argc; i++) {
		args.push_back(String::utf8(argv[i]));
	}
	OS::get_singleton()->set_cmdline("", args, List<String>());

	// Run custom test tools.
	if (test_commands) {
		for (const KeyValue<String, TestFunc> &E : (*test_commands)) {
			if (args.find(E.key)) {
				const TestFunc &test_func = E.value;
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

struct GodotTestCaseListener : public doctest::IReporter {
	GodotTestCaseListener(const doctest::ContextOptions &p_in) {}

	SignalWatcher *signal_watcher = nullptr;

	PhysicsServer3D *physics_server_3d = nullptr;
	PhysicsServer2D *physics_server_2d = nullptr;
	NavigationServer3D *navigation_server_3d = nullptr;
	NavigationServer2D *navigation_server_2d = nullptr;
	ThemeDB *theme_db = nullptr;

	void test_case_start(const doctest::TestCaseData &p_in) override {
		SignalWatcher::get_singleton()->_clear_signals();

		String name = String(p_in.m_name);

		if (name.find("[SceneTree]") != -1) {
			GLOBAL_DEF("memory/limits/multithreaded_server/rid_pool_prealloc", 60);
			memnew(MessageQueue);

			GLOBAL_DEF("internationalization/rendering/force_right_to_left_layout_direction", false);

			memnew(Input);

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

			physics_server_3d = PhysicsServer3DManager::new_default_server();
			physics_server_3d->init();

			physics_server_2d = PhysicsServer2DManager::new_default_server();
			physics_server_2d->init();

			navigation_server_3d = NavigationServer3DManager::new_default_server();
			navigation_server_2d = memnew(NavigationServer2D);

			memnew(InputMap);
			InputMap::get_singleton()->load_default();

			theme_db = memnew(ThemeDB);
			theme_db->initialize_theme_noproject();

			memnew(SceneTree);
			SceneTree::get_singleton()->initialize();
			return;
		}

		if (name.find("Audio") != -1) {
			// The last driver index should always be the dummy driver.
			int dummy_idx = AudioDriverManager::get_driver_count() - 1;
			AudioDriverManager::initialize(dummy_idx);
			AudioServer *audio_server = memnew(AudioServer);
			audio_server->init();
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

		if (theme_db) {
			memdelete(theme_db);
			theme_db = nullptr;
		}

		if (navigation_server_3d) {
			memdelete(navigation_server_3d);
			navigation_server_3d = nullptr;
		}

		if (navigation_server_2d) {
			memdelete(navigation_server_2d);
			navigation_server_2d = nullptr;
		}

		if (physics_server_3d) {
			physics_server_3d->finish();
			memdelete(physics_server_3d);
			physics_server_3d = nullptr;
		}

		if (physics_server_2d) {
			physics_server_2d->finish();
			memdelete(physics_server_2d);
			physics_server_2d = nullptr;
		}

		if (Input::get_singleton()) {
			memdelete(Input::get_singleton());
		}

		if (RenderingServer::get_singleton()) {
			RenderingServer::get_singleton()->sync();
			RenderingServer::get_singleton()->global_shader_uniforms_clear();
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

		if (AudioServer::get_singleton()) {
			AudioServer::get_singleton()->finish();
			memdelete(AudioServer::get_singleton());
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
