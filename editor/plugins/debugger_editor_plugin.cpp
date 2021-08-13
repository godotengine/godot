/*************************************************************************/
/*  debugger_editor_plugin.cpp                                           */
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

#include "debugger_editor_plugin.h"

#include "core/os/keyboard.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/editor_debugger_server.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/fileserver/editor_file_server.h"
#include "scene/gui/menu_button.h"

DebuggerEditorPlugin::DebuggerEditorPlugin(EditorNode *p_editor, MenuButton *p_debug_menu) {
	EditorDebuggerServer::initialize();

	ED_SHORTCUT("debugger/step_into", TTR("Step Into"), Key::F11);
	ED_SHORTCUT("debugger/step_over", TTR("Step Over"), Key::F10);
	ED_SHORTCUT("debugger/break", TTR("Break"));
	ED_SHORTCUT("debugger/continue", TTR("Continue"), Key::F12);
	ED_SHORTCUT("debugger/keep_debugger_open", TTR("Keep Debugger Open"));
	ED_SHORTCUT("debugger/debug_with_external_editor", TTR("Debug with External Editor"));

	// File Server for deploy with remote filesystem.
	file_server = memnew(EditorFileServer);

	EditorDebuggerNode *debugger = memnew(EditorDebuggerNode);
	Button *db = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Debugger"), debugger);
	// Add separation for the warning/error icon that is displayed later.
	db->add_theme_constant_override("hseparation", 6 * EDSCALE);
	debugger->set_tool_button(db);

	// Main editor debug menu.
	debug_menu = p_debug_menu;
	PopupMenu *p = debug_menu->get_popup();
	p->set_hide_on_checkable_item_selection(false);
	p->add_check_shortcut(ED_SHORTCUT("editor/deploy_with_remote_debug", TTR("Deploy with Remote Debug")), RUN_DEPLOY_REMOTE_DEBUG);
	p->set_item_tooltip(
			p->get_item_count() - 1,
			TTR("When this option is enabled, using one-click deploy will make the executable attempt to connect to this computer's IP so the running project can be debugged.\nThis option is intended to be used for remote debugging (typically with a mobile device).\nYou don't need to enable it to use the GDScript debugger locally."));
	p->add_check_shortcut(ED_SHORTCUT("editor/small_deploy_with_network_fs", TTR("Small Deploy with Network Filesystem")), RUN_FILE_SERVER);
	p->set_item_tooltip(
			p->get_item_count() - 1,
			TTR("When this option is enabled, using one-click deploy for Android will only export an executable without the project data.\nThe filesystem will be provided from the project by the editor over the network.\nOn Android, deploying will use the USB cable for faster performance. This option speeds up testing for projects with large assets."));
	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("editor/visible_collision_shapes", TTR("Visible Collision Shapes")), RUN_DEBUG_COLLISONS);
	p->set_item_tooltip(
			p->get_item_count() - 1,
			TTR("When this option is enabled, collision shapes and raycast nodes (for 2D and 3D) will be visible in the running project."));
	p->add_check_shortcut(ED_SHORTCUT("editor/visible_navigation", TTR("Visible Navigation")), RUN_DEBUG_NAVIGATION);
	p->set_item_tooltip(
			p->get_item_count() - 1,
			TTR("When this option is enabled, navigation meshes and polygons will be visible in the running project."));
	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("editor/sync_scene_changes", TTR("Synchronize Scene Changes")), RUN_LIVE_DEBUG);
	p->set_item_tooltip(
			p->get_item_count() - 1,
			TTR("When this option is enabled, any changes made to the scene in the editor will be replicated in the running project.\nWhen used remotely on a device, this is more efficient when the network filesystem option is enabled."));
	p->add_check_shortcut(ED_SHORTCUT("editor/sync_script_changes", TTR("Synchronize Script Changes")), RUN_RELOAD_SCRIPTS);
	p->set_item_tooltip(
			p->get_item_count() - 1,
			TTR("When this option is enabled, any script that is saved will be reloaded in the running project.\nWhen used remotely on a device, this is more efficient when the network filesystem option is enabled."));

	// Multi-instance, start/stop
	instances_menu = memnew(PopupMenu);
	instances_menu->set_name("run_instances");
	instances_menu->set_hide_on_checkable_item_selection(false);

	p->add_child(instances_menu);
	p->add_separator();
	p->add_submenu_item(TTR("Run Multiple Instances"), "run_instances");

	instances_menu->add_radio_check_item(TTR("Run 1 Instance"));
	instances_menu->set_item_metadata(0, 1);
	instances_menu->add_radio_check_item(TTR("Run 2 Instances"));
	instances_menu->set_item_metadata(1, 2);
	instances_menu->add_radio_check_item(TTR("Run 3 Instances"));
	instances_menu->set_item_metadata(2, 3);
	instances_menu->add_radio_check_item(TTR("Run 4 Instances"));
	instances_menu->set_item_metadata(3, 4);
	instances_menu->set_item_checked(0, true);
	instances_menu->connect("index_pressed", callable_mp(this, &DebuggerEditorPlugin::_select_run_count));
	p->connect("id_pressed", callable_mp(this, &DebuggerEditorPlugin::_menu_option));
}

DebuggerEditorPlugin::~DebuggerEditorPlugin() {
	EditorDebuggerServer::deinitialize();
	memdelete(file_server);
}

void DebuggerEditorPlugin::_select_run_count(int p_index) {
	int len = instances_menu->get_item_count();
	for (int idx = 0; idx < len; idx++) {
		instances_menu->set_item_checked(idx, idx == p_index);
	}
	EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_instances", instances_menu->get_item_metadata(p_index));
}

void DebuggerEditorPlugin::_menu_option(int p_option) {
	switch (p_option) {
		case RUN_FILE_SERVER: {
			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_FILE_SERVER));

			if (ischecked) {
				file_server->stop();
			} else {
				file_server->start();
			}

			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_FILE_SERVER), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_file_server", !ischecked);

		} break;
		case RUN_LIVE_DEBUG: {
			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_LIVE_DEBUG));

			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_LIVE_DEBUG), !ischecked);
			EditorDebuggerNode::get_singleton()->set_live_debugging(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_live_debug", !ischecked);

		} break;
		case RUN_DEPLOY_REMOTE_DEBUG: {
			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEPLOY_REMOTE_DEBUG));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEPLOY_REMOTE_DEBUG), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_deploy_remote_debug", !ischecked);

		} break;
		case RUN_DEBUG_COLLISONS: {
			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_COLLISONS));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_COLLISONS), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_collisons", !ischecked);

		} break;
		case RUN_DEBUG_NAVIGATION: {
			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_NAVIGATION));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_NAVIGATION), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_navigation", !ischecked);

		} break;
		case RUN_RELOAD_SCRIPTS: {
			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_RELOAD_SCRIPTS));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_RELOAD_SCRIPTS), !ischecked);

			ScriptEditor::get_singleton()->set_live_auto_reload_running_scripts(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_reload_scripts", !ischecked);

		} break;
	}
}

void DebuggerEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_update_debug_options();
	}
}

void DebuggerEditorPlugin::_update_debug_options() {
	bool check_deploy_remote = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_deploy_remote_debug", false);
	bool check_file_server = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_file_server", false);
	bool check_debug_collisions = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_collisons", false);
	bool check_debug_navigation = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_navigation", false);
	bool check_live_debug = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_live_debug", true);
	bool check_reload_scripts = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_reload_scripts", true);
	int instances = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_instances", 1);

	if (check_deploy_remote) {
		_menu_option(RUN_DEPLOY_REMOTE_DEBUG);
	}
	if (check_file_server) {
		_menu_option(RUN_FILE_SERVER);
	}
	if (check_debug_collisions) {
		_menu_option(RUN_DEBUG_COLLISONS);
	}
	if (check_debug_navigation) {
		_menu_option(RUN_DEBUG_NAVIGATION);
	}
	if (check_live_debug) {
		_menu_option(RUN_LIVE_DEBUG);
	}
	if (check_reload_scripts) {
		_menu_option(RUN_RELOAD_SCRIPTS);
	}

	int len = instances_menu->get_item_count();
	for (int idx = 0; idx < len; idx++) {
		bool checked = (int)instances_menu->get_item_metadata(idx) == instances;
		instances_menu->set_item_checked(idx, checked);
	}
}
