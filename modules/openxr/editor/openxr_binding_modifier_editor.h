/**************************************************************************/
/*  openxr_binding_modifier_editor.h                                      */
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

#ifndef OPENXR_BINDING_MODIFIER_EDITOR_H
#define OPENXR_BINDING_MODIFIER_EDITOR_H

#include "../action_map/openxr_action_map.h"
#include "../action_map/openxr_binding_modifier.h"
#include "editor/editor_properties.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"

class OpenXRBindingModifierEditor : public PanelContainer {
	GDCLASS(OpenXRBindingModifierEditor, PanelContainer);

private:
	HBoxContainer *header_hb = nullptr;
	Label *binding_modifier_title = nullptr;
	Button *rem_binding_modifier_btn = nullptr;

protected:
	VBoxContainer *main_vb = nullptr;
	HashMap<StringName, EditorProperty *> property_editors;

	EditorUndoRedoManager *undo_redo;
	Ref<OpenXRBindingModifier> binding_modifier;
	Ref<OpenXRActionMap> action_map;

	static void _bind_methods();
	void _notification(int p_what);

	void add_property_editor(const String &p_property, EditorProperty *p_editor);

	void _on_remove_binding_modifier();

public:
	Ref<OpenXRBindingModifier> get_binding_modifier() const { return binding_modifier; }

	void _on_property_changed(const String &p_property, const Variant &p_value, const String &p_name, bool p_changing);

	virtual void set_binding_modifier(Ref<OpenXRActionMap> p_action_map, Ref<OpenXRBindingModifier> p_binding_modifier);

	OpenXRBindingModifierEditor();
};

#endif // OPENXR_BINDING_MODIFIER_EDITOR_H
