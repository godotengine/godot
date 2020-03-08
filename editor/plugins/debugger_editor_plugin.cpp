/*************************************************************************/
/*  debugger_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "editor/editor_node.h"
#include "editor/fileserver/editor_file_server.h"
#include "scene/gui/menu_button.h"

DebuggerEditorPlugin::DebuggerEditorPlugin(EditorNode *p_editor, MenuButton *p_debug_menu) {
	ED_SHORTCUT("debugger/step_into", TTR("Step Into"), KEY_F11);
	ED_SHORTCUT("debugger/step_over", TTR("Step Over"), KEY_F10);
	ED_SHORTCUT("debugger/break", TTR("Break"));
	ED_SHORTCUT("debugger/continue", TTR("Continue"), KEY_F12);
	ED_SHORTCUT("debugger/keep_debugger_open", TTR("Keep Debugger Open"));
	ED_SHORTCUT("debugger/debug_with_external_editor", TTR("Debug with External Editor"));

	// File Server for deploy with remote fs.
	file_server = memnew(EditorFileServer);

	EditorDebuggerNode *debugger = memnew(EditorDebuggerNode);
	Button *db = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Debugger"), debugger);
	debugger->set_tool_button(db);

	// Main editor debug menu.
	debug_menu = p_debug_menu;
	PopupMenu *p = debug_menu->get_popup();
	p->set_hide_on_window_lose_focus(true);
	p->set_hide_on_checkable_item_selection(false);
	p->add_check_shortcut(ED_SHORTCUT("editor/deploy_with_remote_debug", TTR("Deploy with Remote Debug")), RUN_DEPLOY_REMOTE_DEBUG);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When exporting or deploying, the resulting executable will attempt to connect to the IP of this computer in order to be debugged."));
	p->add_check_shortcut(ED_SHORTCUT("editor/small_deploy_with_network_fs", TTR("Small Deploy with Network FS")), RUN_FILE_SERVER);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When this option is enabled, export or deploy will produce a minimal executable.\nThe filesystem will be provided from the project by the editor over the network.\nOn Android, deploy will use the USB cable for faster performance. This option speeds up testing for games with a large footprint."));
	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("editor/visible_collision_shapes", TTR("Visible Collision Shapes")), RUN_DEBUG_COLLISONS);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("Collision shapes and raycast nodes (for 2D and 3D) will be visible on the running game if this option is turned on."));
	p->add_check_shortcut(ED_SHORTCUT("editor/visible_navigation", TTR("Visible Navigation")), RUN_DEBUG_NAVIGATION);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("Navigation meshes and polygons will be visible on the running game if this option is turned on."));
	p->add_separator();
	//those are now on by default, since they are harmless
	p->add_check_shortcut(ED_SHORTCUT("editor/sync_scene_changes", TTR("Sync Scene Changes")), RUN_LIVE_DEBUG);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When this option is turned on, any changes made to the scene in the editor will be replicated in the running game.\nWhen used remotely on a device, this is more efficient with network filesystem."));
	p->set_item_checked(p->get_item_count() - 1, true);
	p->add_check_shortcut(ED_SHORTCUT("editor/sync_script_changes", TTR("Sync Script Changes")), RUN_RELOAD_SCRIPTS);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When this option is turned on, any script that is saved will be reloaded on the running game.\nWhen used remotely on a device, this is more efficient with network filesystem."));
	p->set_item_checked(p->get_item_count() - 1, true);

	// Multi-instance, start/stop
	p->add_separator();
	p->add_radio_check_item(TTR("Debug 1 instance"), RUN_DEBUG_ONE);
	p->add_radio_check_item(TTR("Debug 2 instances"), RUN_DEBUG_TWO);
	p->set_item_checked(p->get_item_index(RUN_DEBUG_ONE), true);

	p->connect("id_pressed", callable_mp(this, &DebuggerEditorPlugin::_menu_option));
}

DebuggerEditorPlugin::~DebuggerEditorPlugin() {
	memdelete(file_server);
	// Should delete debugger?
}

void DebuggerEditorPlugin::_menu_option(int p_option) {
	switch (p_option) {
		case RUN_DEBUG_ONE: {
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_ONE), true);
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_TWO), false);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_instances", 1);

		} break;
		case RUN_DEBUG_TWO: {
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_TWO), true);
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_ONE), false);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_instances", 2);

		} break;
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
	switch (p_what) {
		case NOTIFICATION_READY:
			_update_debug_options();
			break;
		default:
			break;
	}
}

void DebuggerEditorPlugin::_update_debug_options() {
	bool check_deploy_remote = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_deploy_remote_debug", false);
	bool check_file_server = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_file_server", false);
	bool check_debug_collisions = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_collisons", false);
	bool check_debug_navigation = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_navigation", false);
	bool check_live_debug = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_live_debug", false);
	bool check_reload_scripts = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_reload_scripts", false);
	int instances = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_instances", 1);

	if (check_deploy_remote) _menu_option(RUN_DEPLOY_REMOTE_DEBUG);
	if (check_file_server) _menu_option(RUN_FILE_SERVER);
	if (check_debug_collisions) _menu_option(RUN_DEBUG_COLLISONS);
	if (check_debug_navigation) _menu_option(RUN_DEBUG_NAVIGATION);
	if (check_live_debug) _menu_option(RUN_LIVE_DEBUG);
	if (check_reload_scripts) _menu_option(RUN_RELOAD_SCRIPTS);
	int one = false;
	int two = false;
	if (instances == 2)
		two = true;
	else
		one = true;
	debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_ONE), one);
	debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_TWO), two);
}
