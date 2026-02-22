/**************************************************************************/
/*  test_main.cpp                                                         */
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

#include "test_main.h"

#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/io/dir_access.h"
#include "core/string/translation_server.h"
#include "scene/main/window.h"
#include "scene/theme/theme_db.h"
#include "servers/audio/audio_server.h"
#include "servers/rendering/rendering_server_default.h"
#include "tests/display_server_mock.h"
#include "tests/force_link.gen.h"
#include "tests/signal_watcher.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

#ifdef TOOLS_ENABLED
#include "editor/file_system/editor_paths.h"
#include "editor/settings/editor_settings.h"
#endif // TOOLS_ENABLED

#ifndef NAVIGATION_2D_DISABLED
#include "servers/navigation_2d/navigation_server_2d.h"
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
#include "servers/navigation_3d/navigation_server_3d.h"
#endif // NAVIGATION_3D_DISABLED

#ifndef PHYSICS_2D_DISABLED
#include "servers/physics_2d/physics_server_2d.h"
#include "servers/physics_2d/physics_server_2d_dummy.h"
#endif // PHYSICS_2D_DISABLED
#ifndef PHYSICS_3D_DISABLED
#include "servers/physics_3d/physics_server_3d.h"
#include "servers/physics_3d/physics_server_3d_dummy.h"
#endif // PHYSICS_3D_DISABLED

#include "modules/modules_tests.gen.h" // IWYU pragma: keep // TODO: Migrate module tests to compilation files.

int test_main(int argc, char *argv[]) {
	ForceLink::force_link_tests();

	bool run_tests = true;

	// Convert arguments to Godot's command-line.
	List<String> args;

	for (int i = 0; i < argc; i++) {
		args.push_back(String::utf8(argv[i]));
	}
	OS::get_singleton()->set_cmdline("", args, List<String>());
	DisplayServerMock::register_mock_driver();

	WorkerThreadPool::get_singleton()->init();

	{
		const String test_path = TestUtils::get_temp_path("");
		Ref<DirAccess> da = DirAccess::open(test_path); // get_temp_path() automatically creates the folder.
		ERR_FAIL_COND_V(da.is_null(), 0);
		ERR_FAIL_COND_V_MSG(da->erase_contents_recursive() != OK, 0, "Failed to delete files");
	}

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
	LocalVector<String> test_args;

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
		for (uint32_t x = 0; x < test_args.size(); x++) {
			// Operation to convert Godot string to non wchar string.
			CharString cs = test_args[x].utf8();
			const char *str = cs.get_data();
			// Allocate the string copy.
			doctest_args[x] = new char[strlen(str) + 1];
			// Copy this into memory.
			memcpy(doctest_args[x], str, strlen(str) + 1);
		}

		test_context.applyCommandLine(test_args.size(), doctest_args);

		for (uint32_t x = 0; x < test_args.size(); x++) {
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

#ifndef PHYSICS_2D_DISABLED
	PhysicsServer2D *physics_server_2d = nullptr;
#endif // PHYSICS_2D_DISABLED
#ifndef PHYSICS_3D_DISABLED
	PhysicsServer3D *physics_server_3d = nullptr;
#endif // PHYSICS_3D_DISABLED

#ifndef NAVIGATION_2D_DISABLED
	NavigationServer2D *navigation_server_2d = nullptr;
#endif // NAVIGATION_2D_DISABLED
#ifndef NAVIGATION_3D_DISABLED
	NavigationServer3D *navigation_server_3d = nullptr;
#endif // NAVIGATION_3D_DISABLED

	void test_case_start(const doctest::TestCaseData &p_in) override {
		reinitialize();

		String name = String(p_in.m_name);
		String suite_name = String(p_in.m_test_suite);

		if (name.contains("[SceneTree]") || name.contains("[Editor]")) {
			memnew(MessageQueue);

			memnew(Input);
			Input::get_singleton()->set_use_accumulated_input(false);

			Error err = OK;
			OS::get_singleton()->set_has_server_feature_callback(nullptr);
			for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
				if (String("mock") == DisplayServer::get_create_function_name(i)) {
					DisplayServer::create(i, "", DisplayServer::WindowMode::WINDOW_MODE_MINIMIZED, DisplayServer::VSyncMode::VSYNC_ENABLED, 0, nullptr, Vector2i(0, 0), DisplayServer::SCREEN_PRIMARY, DisplayServer::CONTEXT_EDITOR, 0, err);
					break;
				}
			}
			memnew(RenderingServerDefault());
			RenderingServerDefault::get_singleton()->init();
			RenderingServerDefault::get_singleton()->set_render_loop_enabled(false);

			// ThemeDB requires RenderingServer to initialize the default theme.
			// So we have to do this for each test case. Also make sure there is
			// no residual theme from something else.
			ThemeDB::get_singleton()->finalize_theme();
			ThemeDB::get_singleton()->initialize_theme();

#ifndef PHYSICS_3D_DISABLED
			physics_server_3d = PhysicsServer3DManager::get_singleton()->new_default_server();
			if (!physics_server_3d) {
				physics_server_3d = memnew(PhysicsServer3DDummy);
			}
			physics_server_3d->init();
#endif // PHYSICS_3D_DISABLED

#ifndef PHYSICS_2D_DISABLED
			physics_server_2d = PhysicsServer2DManager::get_singleton()->new_default_server();
			if (!physics_server_2d) {
				physics_server_2d = memnew(PhysicsServer2DDummy);
			}
			physics_server_2d->init();
#endif // PHYSICS_2D_DISABLED

			ERR_PRINT_OFF;
#ifndef NAVIGATION_3D_DISABLED
			navigation_server_3d = NavigationServer3DManager::get_singleton()->new_default_server();
#endif // NAVIGATION_3D_DISABLED
#ifndef NAVIGATION_2D_DISABLED
			navigation_server_2d = NavigationServer2DManager::get_singleton()->new_default_server();
#endif // NAVIGATION_2D_DISABLED
			ERR_PRINT_ON;

			memnew(InputMap);
			InputMap::get_singleton()->load_default();

			memnew(SceneTree);
			SceneTree::get_singleton()->initialize();
			if (!DisplayServer::get_singleton()->has_feature(DisplayServer::Feature::FEATURE_SUBWINDOWS)) {
				SceneTree::get_singleton()->get_root()->set_embedding_subwindows(true);
			}

#ifdef TOOLS_ENABLED
			if (name.contains("[Editor]")) {
				Engine::get_singleton()->set_editor_hint(true);
				EditorPaths::create();
				EditorSettings::create();
			}
#endif // TOOLS_ENABLED

			return;
		}

		if (name.contains("[Audio]")) {
			// The last driver index should always be the dummy driver.
			int dummy_idx = AudioDriverManager::get_driver_count() - 1;
			AudioDriverManager::initialize(dummy_idx);
			AudioServer *audio_server = memnew(AudioServer);
			audio_server->init();
			return;
		}

#ifndef NAVIGATION_3D_DISABLED
		if (suite_name.contains("[Navigation3D]") && navigation_server_3d == nullptr) {
			ERR_PRINT_OFF;
			navigation_server_3d = NavigationServer3DManager::get_singleton()->new_default_server();
			ERR_PRINT_ON;
			return;
		}
#endif // NAVIGATION_3D_DISABLED

#ifndef NAVIGATION_2D_DISABLED
		if (suite_name.contains("[Navigation2D]") && navigation_server_2d == nullptr) {
			ERR_PRINT_OFF;
			navigation_server_2d = NavigationServer2DManager::get_singleton()->new_default_server();
			ERR_PRINT_ON;
			return;
		}
#endif // NAVIGATION_2D_DISABLED
	}

	void test_case_end(const doctest::CurrentTestCaseStats &) override {
#ifdef TOOLS_ENABLED
		if (EditorSettings::get_singleton()) {
			EditorSettings::destroy();

			// Instantiating the EditorSettings singleton sets the locale to the editor's language.
			TranslationServer::get_singleton()->set_locale("en");
		}
		if (EditorPaths::get_singleton()) {
			EditorPaths::free();
		}
#endif // TOOLS_ENABLED

		Engine::get_singleton()->set_editor_hint(false);

		if (SceneTree::get_singleton()) {
			SceneTree::get_singleton()->finalize();
		}

		if (MessageQueue::get_singleton()) {
			MessageQueue::get_singleton()->flush();
		}

		if (SceneTree::get_singleton()) {
			memdelete(SceneTree::get_singleton());
		}

#ifndef NAVIGATION_3D_DISABLED
		if (navigation_server_3d) {
			memdelete(navigation_server_3d);
			navigation_server_3d = nullptr;
		}
#endif // NAVIGATION_3D_DISABLED

#ifndef NAVIGATION_2D_DISABLED
		if (navigation_server_2d) {
			memdelete(navigation_server_2d);
			navigation_server_2d = nullptr;
		}
#endif // NAVIGATION_2D_DISABLED

#ifndef PHYSICS_3D_DISABLED
		if (physics_server_3d) {
			physics_server_3d->finish();
			memdelete(physics_server_3d);
			physics_server_3d = nullptr;
		}
#endif // PHYSICS_3D_DISABLED

#ifndef PHYSICS_2D_DISABLED
		if (physics_server_2d) {
			physics_server_2d->finish();
			memdelete(physics_server_2d);
			physics_server_2d = nullptr;
		}
#endif // PHYSICS_2D_DISABLED

		if (Input::get_singleton()) {
			memdelete(Input::get_singleton());
		}

		if (RenderingServer::get_singleton()) {
			// ThemeDB requires RenderingServer to finalize the default theme.
			// So we have to do this for each test case.
			ThemeDB::get_singleton()->finalize_theme();

			RenderingServer::get_singleton()->sync();
			RenderingServer::get_singleton()->global_shader_parameters_clear();
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
		reinitialize();
	}

	void subcase_start(const doctest::SubcaseSignature &) override {
		reinitialize();
	}

	void report_query(const doctest::QueryData &) override {}
	void test_case_exception(const doctest::TestCaseException &) override {}
	void subcase_end() override {}

	void log_assert(const doctest::AssertData &in) override {}
	void log_message(const doctest::MessageData &) override {}
	void test_case_skipped(const doctest::TestCaseData &) override {}

private:
	void reinitialize() {
		Math::seed(0x60d07);
		SignalWatcher::get_singleton()->_clear_signals();
	}
};

REGISTER_LISTENER("GodotTestCaseListener", 1, GodotTestCaseListener);
