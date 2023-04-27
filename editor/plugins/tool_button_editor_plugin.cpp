/**************************************************************************/
/*  tool_button_editor_plugin.cpp                                         */
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

#include "tool_button_editor_plugin.h"
#include "editor/editor_node.h"
#include "editor/editor_property_name_processor.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/button.h"

bool ToolButtonInspectorPlugin::can_handle(Object *p_object) {
	Ref<Script> scr = Object::cast_to<Script>(p_object->get_script());
	return scr.is_valid() && scr->is_tool();
}

void ToolButtonInspectorPlugin::update_action_icon(Button *p_action_button) {
	p_action_button->set_icon(p_action_button->get_theme_icon(action_icon, SNAME("EditorIcons")));
}

void ToolButtonInspectorPlugin::call_action(Object *p_object, StringName p_do_method_name, StringName p_undo_method_name) {
	if (!p_object->has_method(p_do_method_name)) {
		print_error(vformat("Tool button do method is invalid. Could not find method '%s' on %s", p_do_method_name, p_object->get_class_name()));
		return;
	}

	if (!p_object->has_method(p_undo_method_name)) {
		print_error(vformat("Tool button undo method is invalid. Could not find method '%s' on %s", p_undo_method_name, p_object->get_class_name()));
		return;
	}

	EditorUndoRedoManager *undo_redo_manager = EditorUndoRedoManager::get_singleton();
	int history_id = undo_redo_manager->get_history_id_for_object(p_object);
	UndoRedo *undo_redo = undo_redo_manager->get_history_undo_redo(history_id);

	Callable do_callable(p_object, p_do_method_name);
	Callable undo_callable(p_object, p_undo_method_name);

	undo_redo->create_action(TTR("Call Tool Button Action"));
	undo_redo->add_do_method(do_callable);
	undo_redo->add_undo_method(undo_callable);
	undo_redo->commit_action();
}

bool ToolButtonInspectorPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	if (p_type == Variant::CALLABLE) {
		if (p_usage.has_flag(PROPERTY_USAGE_INTERNAL) && p_usage.has_flag(PROPERTY_USAGE_EDITOR)) {
			Button *action_button = EditorInspector::create_inspector_action_button(EditorPropertyNameProcessor::get_singleton()->process_name(p_path, EditorPropertyNameProcessor::STYLE_CAPITALIZED));

			PackedStringArray split = p_hint_text.split(",");
			if (split.size() > 2) {
				String icon = split[2];
				action_icon = StringName(icon);
				action_button->connect(SNAME("theme_changed"), callable_mp(this, &ToolButtonInspectorPlugin::update_action_icon).bind(action_button));
			}
			add_custom_control(action_button);

			action_button->connect(SNAME("pressed"), callable_mp(this, &ToolButtonInspectorPlugin::call_action).bind(p_object, split[1]));
			return true;
		}
	}
	return false;
}

ToolButtonEditorPlugin::ToolButtonEditorPlugin() {
	Ref<ToolButtonInspectorPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
