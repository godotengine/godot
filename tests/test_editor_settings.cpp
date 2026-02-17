#include "core/io/project_settings.h"
#include "editor/settings/editor_settings.h"
#include "editor/editor_node.h"
#include "core/input/shortcut.h"
#include "core/string/translation.h"

// This file contains a few basic unit tests for the new `erase_group`
// functionality added in PR-116393.  The tests are tagged with `[Editor]`
// so that the editor environment (paths/settings) is automatically
// initialized by the test harness.

TEST_CASE("[Editor][Settings] erase_group removes settings and shortcuts") {
	EditorSettings *es = EditorSettings::get_singleton();
	REQUIRE(es != nullptr);

	// Prepare a couple of settings that would normally be created by a
	// plugin using its folder name as prefix.
	es->set_setting("myplugin/option", 42);
	Ref<Shortcut> sc;
	sc.instantiate();
	es->add_shortcut("myplugin/action", sc);

	REQUIRE(es->has_setting("myplugin/option"));
	REQUIRE(!es->get_shortcut("myplugin/action").is_null());

	// Erasing the "myplugin" group should wipe both the normal setting and
	// the shortcut.
	es->erase_group("myplugin");

	CHECK(!es->has_setting("myplugin/option"));
	CHECK(es->get_shortcut("myplugin/action").is_null());
}

TEST_CASE("[Editor][Settings] _remove_plugin_from_enabled triggers cleanup") {
	EditorSettings *es = EditorSettings::get_singleton();
	REQUIRE(es != nullptr);

	// Create a dummy editor node so we can call the private method. The
	// constructor is heavy but safe in the test environment because
	// `Engine::set_editor_hint(true)` has already been called by the harness.
	auto *node = memnew(EditorNode);

	// Simulate some leftover settings from an addon that has been deleted on
	// disk.  The cleanup code should remove them when the addon is dropped from
	// the enabled list.
	es->set_setting("removed/one", 2);
	REQUIRE(es->has_setting("removed/one"));

	node->_remove_plugin_from_enabled("res://addons/removed/plugin.cfg");

	CHECK(!es->has_setting("removed/one"));

	memdelete(node);
}

TEST_CASE("[Editor][Settings] update_plugins drops missing enabled addons") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	PackedStringArray enabled;
	enabled.push_back("res://addons/nonexistent/plugin.cfg");
	ps->set("editor_plugins/enabled", enabled);

	EditorPluginSettings eps;
	eps.update_plugins();

	PackedStringArray new_enabled = ps->get("editor_plugins/enabled");
	CHECK(!new_enabled.has("res://addons/nonexistent/plugin.cfg"));
}
