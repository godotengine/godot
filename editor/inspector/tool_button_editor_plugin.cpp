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
#include "editor/inspector/multi_node_edit.h"

void EditorInspectorToolButtonPlugin::_call_action(const Variant &p_object, const StringName &p_property) {
	Object *object = p_object.get_validated_object();
	ERR_FAIL_NULL_MSG(object, vformat(R"(Failed to get property "%s" on a previously freed instance.)", p_property));

	const Variant value = object->get(p_property);
	ERR_FAIL_COND_MSG(value.get_type() != Variant::CALLABLE, vformat(R"(The value of property "%s" is %s, but Callable was expected.)", p_property, Variant::get_type_name(value.get_type())));

	const Callable callable = value;
	ERR_FAIL_COND_MSG(!callable.is_valid(), vformat(R"(Tool button action "%s" is an invalid callable.)", callable));

	Ref<MultiNodeEdit> multinode = p_object;
	if (multinode.is_valid()) {
		Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
		if (edited_scene) {
			const StringName method = callable.get_method();
			for (int i = 0; i < multinode->get_node_count(); i++) {
				Callable retarget(edited_scene->get_node(multinode->get_node(i)), method);
				_invoke_callable(retarget);
			}
		}
	} else {
		_invoke_callable(callable);
	}
}

void EditorInspectorToolButtonPlugin::_invoke_callable(const Callable &p_callable) {
	Variant ret;
	Callable::CallError ce;
	p_callable.callp(nullptr, 0, ret, ce);
	ERR_FAIL_COND_MSG(ce.error != Callable::CallError::CALL_OK, vformat(R"(Error calling tool button action "%s": %s)", p_callable, Variant::get_call_error_text(p_callable.get_method(), nullptr, 0, ce)));
}

bool EditorInspectorToolButtonPlugin::can_handle(Object *p_object) {
	return true;
}

bool EditorInspectorToolButtonPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	if (p_type != Variant::CALLABLE || p_hint != PROPERTY_HINT_TOOL_BUTTON || !p_usage.has_flag(PROPERTY_USAGE_EDITOR)) {
		return false;
	}

	const PackedStringArray splits = p_hint_text.rsplit(",", true, 1);
	const String &hint_text = splits[0]; // Safe since `splits` cannot be empty.
	const String &hint_icon = splits.size() > 1 ? splits[1] : "Callable";

	EditorInspectorActionButton *action_button = memnew(EditorInspectorActionButton(hint_text, hint_icon));
	action_button->set_auto_translate_mode(Node::AUTO_TRANSLATE_MODE_DISABLED);
	bool is_multinode = Object::cast_to<MultiNodeEdit>(p_object);
	if (p_usage & PROPERTY_USAGE_READ_ONLY) {
		action_button->set_disabled(true);
	} else if (is_multinode) {
		const Callable callback = p_object->get(p_path);
		if (callback.is_custom()) {
			action_button->set_disabled(true);
			action_button->set_tooltip_text(TTR("The assigned tool button callback can't be used with multiple nodes."));
		}
	}
	action_button->connect(SceneStringName(pressed), callable_mp(this, &EditorInspectorToolButtonPlugin::_call_action).bind(p_object, p_path));

	add_custom_control(action_button);
	return true;
}

ToolButtonEditorPlugin::ToolButtonEditorPlugin() {
	Ref<EditorInspectorToolButtonPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
