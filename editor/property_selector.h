/*************************************************************************/
/*  property_selector.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef PROPERTYSELECTOR_H
#define PROPERTYSELECTOR_H

#include "editor/property_editor.h"
#include "editor_help.h"
#include "scene/gui/rich_text_label.h"

class PropertySelector : public ConfirmationDialog {
	GDCLASS(PropertySelector, ConfirmationDialog)

	LineEdit *search_box;
	Tree *search_options;

	void _update_search();

	void _sbox_input(const InputEvent &p_ie);

	void _confirmed();
	void _text_changed(const String &p_newtext);

	EditorHelpBit *help_bit;

	bool properties;
	String selected;
	Variant::Type type;
	InputEvent::Type event_type;
	String base_type;
	ObjectID script;
	Object *instance;

	void _item_selected();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void select_method_from_base_type(const String &p_base, const String &p_current = "");
	void select_method_from_script(const Ref<Script> &p_script, const String &p_current = "");
	void select_method_from_basic_type(Variant::Type p_type, const String &p_current = "");
	void select_method_from_instance(Object *p_instance, const String &p_current = "");

	void select_property_from_base_type(const String &p_base, const String &p_current = "");
	void select_property_from_script(const Ref<Script> &p_script, const String &p_current = "");
	void select_property_from_basic_type(Variant::Type p_type, InputEvent::Type p_event_type, const String &p_current = "");
	void select_property_from_instance(Object *p_instance, const String &p_current = "");

	PropertySelector();
};

#endif // PROPERTYSELECTOR_H
