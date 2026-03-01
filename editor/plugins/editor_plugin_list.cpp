/**************************************************************************/
/*  editor_plugin_list.cpp                                                */
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

#include "editor_plugin_list.h"

bool EditorPluginList::forward_gui_input(const Ref<InputEvent> &p_event) const {
	bool discard = false;

	for (EditorPlugin *plugin : plugins_list) {
		if (plugin->forward_canvas_gui_input(p_event)) {
			discard = true;
		}
	}

	return discard;
}

EditorPlugin::AfterGUIInput EditorPluginList::forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event, bool p_serve_when_force_input_enabled) const {
	EditorPlugin::AfterGUIInput after = EditorPlugin::AFTER_GUI_INPUT_PASS;

	for (EditorPlugin *plugin : plugins_list) {
		if (!p_serve_when_force_input_enabled && plugin->is_input_event_forwarding_always_enabled()) {
			continue;
		}

		EditorPlugin::AfterGUIInput current_after = plugin->forward_3d_gui_input(p_camera, p_event);
		if (current_after == EditorPlugin::AFTER_GUI_INPUT_STOP) {
			after = EditorPlugin::AFTER_GUI_INPUT_STOP;
		}
		if (after != EditorPlugin::AFTER_GUI_INPUT_STOP && current_after == EditorPlugin::AFTER_GUI_INPUT_CUSTOM) {
			after = EditorPlugin::AFTER_GUI_INPUT_CUSTOM;
		}
	}

	return after;
}

void EditorPluginList::forward_canvas_draw_over_viewport(Control *p_overlay) const {
	for (EditorPlugin *plugin : plugins_list) {
		plugin->forward_canvas_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::forward_canvas_force_draw_over_viewport(Control *p_overlay) const {
	for (EditorPlugin *plugin : plugins_list) {
		plugin->forward_canvas_force_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::forward_3d_draw_over_viewport(Control *p_overlay) const {
	for (EditorPlugin *plugin : plugins_list) {
		plugin->forward_3d_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::forward_3d_force_draw_over_viewport(Control *p_overlay) const {
	for (EditorPlugin *plugin : plugins_list) {
		plugin->forward_3d_force_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::add_plugin(EditorPlugin *p_plugin) {
	ERR_FAIL_COND(plugins_list.has(p_plugin));
	plugins_list.push_back(p_plugin);
}

void EditorPluginList::remove_plugin(EditorPlugin *p_plugin) {
	plugins_list.erase(p_plugin);
}
