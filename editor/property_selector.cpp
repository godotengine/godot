/**************************************************************************/
/*  property_selector.cpp                                                 */
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

#include "property_selector.h"

#include "core/os/keyboard.h"
#include "editor/doc_tools.h"
#include "editor/editor_help.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tree.h"

void PropertySelector::_text_changed(const String &p_newtext) {
	_update_search();
}

void PropertySelector::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				search_options->gui_input(k);
				search_box->accept_event();

				TreeItem *root = search_options->get_root();
				if (!root->get_first_child()) {
					break;
				}

				TreeItem *current = search_options->get_selected();

				TreeItem *item = search_options->get_next_selected(root);
				while (item) {
					item->deselect(0);
					item = search_options->get_next_selected(item);
				}

				current->select(0);

			} break;
			default:
				break;
		}
	}
}

void PropertySelector::_update_search() {
	if (properties) {
		set_title(TTR("Select Property"));
	} else if (virtuals_only) {
		set_title(TTR("Select Virtual Method"));
	} else {
		set_title(TTR("Select Method"));
	}

	search_options->clear();
	help_bit->set_text("");

	TreeItem *root = search_options->create_item();

	// Allow using spaces in place of underscores in the search string (makes the search more fault-tolerant).
	const String search_text = search_box->get_text().replace(" ", "_");

	if (properties) {
		List<PropertyInfo> props;

		if (instance) {
			instance->get_property_list(&props, true);
		} else if (type != Variant::NIL) {
			Variant v;
			Callable::CallError ce;
			Variant::construct(type, v, nullptr, 0, ce);

			v.get_property_list(&props);
		} else {
			Object *obj = ObjectDB::get_instance(script);
			if (Object::cast_to<Script>(obj)) {
				props.push_back(PropertyInfo(Variant::NIL, "Script Variables", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
				Object::cast_to<Script>(obj)->get_script_property_list(&props);
			}

			StringName base = base_type;
			while (base) {
				props.push_back(PropertyInfo(Variant::NIL, base, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
				ClassDB::get_property_list(base, &props, true);
				base = ClassDB::get_parent_class(base);
			}
		}

		TreeItem *category = nullptr;

		bool found = false;

		Ref<Texture2D> type_icons[] = {
			search_options->get_editor_theme_icon(SNAME("Variant")),
			search_options->get_editor_theme_icon(SNAME("bool")),
			search_options->get_editor_theme_icon(SNAME("int")),
			search_options->get_editor_theme_icon(SNAME("float")),
			search_options->get_editor_theme_icon(SNAME("String")),
			search_options->get_editor_theme_icon(SNAME("Vector2")),
			search_options->get_editor_theme_icon(SNAME("Vector2i")),
			search_options->get_editor_theme_icon(SNAME("Rect2")),
			search_options->get_editor_theme_icon(SNAME("Rect2i")),
			search_options->get_editor_theme_icon(SNAME("Vector3")),
			search_options->get_editor_theme_icon(SNAME("Vector3i")),
			search_options->get_editor_theme_icon(SNAME("Transform2D")),
			search_options->get_editor_theme_icon(SNAME("Vector4")),
			search_options->get_editor_theme_icon(SNAME("Vector4i")),
			search_options->get_editor_theme_icon(SNAME("Plane")),
			search_options->get_editor_theme_icon(SNAME("Quaternion")),
			search_options->get_editor_theme_icon(SNAME("AABB")),
			search_options->get_editor_theme_icon(SNAME("Basis")),
			search_options->get_editor_theme_icon(SNAME("Transform3D")),
			search_options->get_editor_theme_icon(SNAME("Projection")),
			search_options->get_editor_theme_icon(SNAME("Color")),
			search_options->get_editor_theme_icon(SNAME("StringName")),
			search_options->get_editor_theme_icon(SNAME("NodePath")),
			search_options->get_editor_theme_icon(SNAME("RID")),
			search_options->get_editor_theme_icon(SNAME("MiniObject")),
			search_options->get_editor_theme_icon(SNAME("Callable")),
			search_options->get_editor_theme_icon(SNAME("Signal")),
			search_options->get_editor_theme_icon(SNAME("Dictionary")),
			search_options->get_editor_theme_icon(SNAME("Array")),
			search_options->get_editor_theme_icon(SNAME("PackedByteArray")),
			search_options->get_editor_theme_icon(SNAME("PackedInt32Array")),
			search_options->get_editor_theme_icon(SNAME("PackedInt64Array")),
			search_options->get_editor_theme_icon(SNAME("PackedFloat32Array")),
			search_options->get_editor_theme_icon(SNAME("PackedFloat64Array")),
			search_options->get_editor_theme_icon(SNAME("PackedStringArray")),
			search_options->get_editor_theme_icon(SNAME("PackedVector2Array")),
			search_options->get_editor_theme_icon(SNAME("PackedVector3Array")),
			search_options->get_editor_theme_icon(SNAME("PackedColorArray"))
		};
		static_assert((sizeof(type_icons) / sizeof(type_icons[0])) == Variant::VARIANT_MAX, "Number of type icons doesn't match the number of Variant types.");

		for (const PropertyInfo &E : props) {
			if (E.usage == PROPERTY_USAGE_CATEGORY) {
				if (category && category->get_first_child() == nullptr) {
					memdelete(category); //old category was unused
				}
				category = search_options->create_item(root);
				category->set_text(0, E.name);
				category->set_selectable(0, false);

				Ref<Texture2D> icon;
				if (E.name == "Script Variables") {
					icon = search_options->get_editor_theme_icon(SNAME("Script"));
				} else {
					icon = EditorNode::get_singleton()->get_class_icon(E.name);
				}
				category->set_icon(0, icon);
				continue;
			}

			if (!(E.usage & PROPERTY_USAGE_EDITOR) && !(E.usage & PROPERTY_USAGE_SCRIPT_VARIABLE)) {
				continue;
			}

			if (!search_box->get_text().is_empty() && E.name.findn(search_text) == -1) {
				continue;
			}

			if (type_filter.size() && !type_filter.has(E.type)) {
				continue;
			}

			TreeItem *item = search_options->create_item(category ? category : root);
			item->set_text(0, E.name);
			item->set_metadata(0, E.name);
			item->set_icon(0, type_icons[E.type]);

			if (!found && !search_box->get_text().is_empty() && E.name.findn(search_text) != -1) {
				item->select(0);
				found = true;
			}

			item->set_selectable(0, true);
		}

		if (category && category->get_first_child() == nullptr) {
			memdelete(category); //old category was unused
		}
	} else {
		List<MethodInfo> methods;

		if (type != Variant::NIL) {
			Variant v;
			Callable::CallError ce;
			Variant::construct(type, v, nullptr, 0, ce);
			v.get_method_list(&methods);
		} else {
			Ref<Script> script_ref = Object::cast_to<Script>(ObjectDB::get_instance(script));
			if (script_ref.is_valid()) {
				methods.push_back(MethodInfo("*Script Methods"));
				if (script_ref->is_built_in()) {
					script_ref->reload(true);
				}
				script_ref->get_script_method_list(&methods);
			}

			StringName base = base_type;
			while (base) {
				methods.push_back(MethodInfo("*" + String(base)));
				ClassDB::get_method_list(base, &methods, true, true);
				base = ClassDB::get_parent_class(base);
			}
		}

		TreeItem *category = nullptr;

		bool found = false;
		bool script_methods = false;

		for (MethodInfo &mi : methods) {
			if (mi.name.begins_with("*")) {
				if (category && category->get_first_child() == nullptr) {
					memdelete(category); //old category was unused
				}
				category = search_options->create_item(root);
				category->set_text(0, mi.name.replace_first("*", ""));
				category->set_selectable(0, false);

				Ref<Texture2D> icon;
				script_methods = false;
				String rep = mi.name.replace("*", "");
				if (mi.name == "*Script Methods") {
					icon = search_options->get_editor_theme_icon(SNAME("Script"));
					script_methods = true;
				} else {
					icon = EditorNode::get_singleton()->get_class_icon(rep);
				}
				category->set_icon(0, icon);

				continue;
			}

			String name = mi.name.get_slice(":", 0);
			if (!script_methods && name.begins_with("_") && !(mi.flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (virtuals_only && !(mi.flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (!virtuals_only && (mi.flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (!search_box->get_text().is_empty() && name.findn(search_text) == -1) {
				continue;
			}

			TreeItem *item = search_options->create_item(category ? category : root);

			String desc;
			if (mi.name.contains(":")) {
				desc = mi.name.get_slice(":", 1) + " ";
				mi.name = mi.name.get_slice(":", 0);
			} else if (mi.return_val.type != Variant::NIL) {
				desc = Variant::get_type_name(mi.return_val.type);
			} else {
				desc = "void";
			}

			desc += vformat(" %s(", mi.name);

			for (int i = 0; i < mi.arguments.size(); i++) {
				if (i > 0) {
					desc += ", ";
				}

				desc += mi.arguments[i].name;

				if (mi.arguments[i].type == Variant::NIL) {
					desc += ": Variant";
				} else if (mi.arguments[i].name.contains(":")) {
					desc += vformat(": %s", mi.arguments[i].name.get_slice(":", 1));
					mi.arguments[i].name = mi.arguments[i].name.get_slice(":", 0);
				} else {
					desc += vformat(": %s", Variant::get_type_name(mi.arguments[i].type));
				}
			}

			desc += ")";

			if (mi.flags & METHOD_FLAG_CONST) {
				desc += " const";
			}

			if (mi.flags & METHOD_FLAG_VIRTUAL) {
				desc += " virtual";
			}

			item->set_text(0, desc);
			item->set_metadata(0, name);
			item->set_selectable(0, true);

			if (!found && !search_box->get_text().is_empty() && name.findn(search_text) != -1) {
				item->select(0);
				found = true;
			}
		}

		if (category && category->get_first_child() == nullptr) {
			memdelete(category); //old category was unused
		}
	}

	get_ok_button()->set_disabled(root->get_first_child() == nullptr);
}

void PropertySelector::_confirmed() {
	TreeItem *ti = search_options->get_selected();
	if (!ti) {
		return;
	}
	emit_signal(SNAME("selected"), ti->get_metadata(0));
	hide();
}

void PropertySelector::_item_selected() {
	help_bit->set_text("");

	TreeItem *item = search_options->get_selected();
	if (!item) {
		return;
	}
	String name = item->get_metadata(0);

	String class_type;
	if (type != Variant::NIL) {
		class_type = Variant::get_type_name(type);
	} else if (!base_type.is_empty()) {
		class_type = base_type;
	} else if (instance) {
		class_type = instance->get_class();
	}

	String text;
	while (!class_type.is_empty()) {
		text = properties ? help_bit->get_property_description(class_type, name) : help_bit->get_method_description(class_type, name);
		if (!text.is_empty()) {
			break;
		}

		// It may be from a parent class, keep looking.
		class_type = ClassDB::get_parent_class(class_type);
	}

	if (!text.is_empty()) {
		// Display both property name and description, since the help bit may be displayed
		// far away from the location (especially if the dialog was resized to be taller).
		help_bit->set_text(vformat("[b]%s[/b]: %s", name, text));
		help_bit->get_rich_text()->set_self_modulate(Color(1, 1, 1, 1));
	} else {
		// Use nested `vformat()` as translators shouldn't interfere with BBCode tags.
		help_bit->set_text(vformat(TTR("No description available for %s."), vformat("[b]%s[/b]", name)));
		help_bit->get_rich_text()->set_self_modulate(Color(1, 1, 1, 0.5));
	}
}

void PropertySelector::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void PropertySelector::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &PropertySelector::_confirmed));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			disconnect("confirmed", callable_mp(this, &PropertySelector::_confirmed));
		} break;
	}
}

void PropertySelector::select_method_from_base_type(const String &p_base, const String &p_current, bool p_virtuals_only) {
	base_type = p_base;
	selected = p_current;
	type = Variant::NIL;
	script = ObjectID();
	properties = false;
	instance = nullptr;
	virtuals_only = p_virtuals_only;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_method_from_script(const Ref<Script> &p_script, const String &p_current) {
	ERR_FAIL_COND(p_script.is_null());
	base_type = p_script->get_instance_base_type();
	selected = p_current;
	type = Variant::NIL;
	script = p_script->get_instance_id();
	properties = false;
	instance = nullptr;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_method_from_basic_type(Variant::Type p_type, const String &p_current) {
	ERR_FAIL_COND(p_type == Variant::NIL);
	base_type = "";
	selected = p_current;
	type = p_type;
	script = ObjectID();
	properties = false;
	instance = nullptr;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_method_from_instance(Object *p_instance, const String &p_current) {
	base_type = p_instance->get_class();
	selected = p_current;
	type = Variant::NIL;
	script = ObjectID();
	{
		Ref<Script> scr = p_instance->get_script();
		if (scr.is_valid()) {
			script = scr->get_instance_id();
		}
	}
	properties = false;
	instance = nullptr;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_property_from_base_type(const String &p_base, const String &p_current) {
	base_type = p_base;
	selected = p_current;
	type = Variant::NIL;
	script = ObjectID();
	properties = true;
	instance = nullptr;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_property_from_script(const Ref<Script> &p_script, const String &p_current) {
	ERR_FAIL_COND(p_script.is_null());

	base_type = p_script->get_instance_base_type();
	selected = p_current;
	type = Variant::NIL;
	script = p_script->get_instance_id();
	properties = true;
	instance = nullptr;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_property_from_basic_type(Variant::Type p_type, const String &p_current) {
	ERR_FAIL_COND(p_type == Variant::NIL);
	base_type = "";
	selected = p_current;
	type = p_type;
	script = ObjectID();
	properties = true;
	instance = nullptr;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::select_property_from_instance(Object *p_instance, const String &p_current) {
	base_type = "";
	selected = p_current;
	type = Variant::NIL;
	script = ObjectID();
	properties = true;
	instance = p_instance;
	virtuals_only = false;

	popup_centered_ratio(0.6);
	search_box->set_text("");
	search_box->grab_focus();
	_update_search();
}

void PropertySelector::set_type_filter(const Vector<Variant::Type> &p_type_filter) {
	type_filter = p_type_filter;
}

void PropertySelector::_bind_methods() {
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::STRING, "name")));
}

PropertySelector::PropertySelector() {
	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	//set_child_rect(vbc);
	search_box = memnew(LineEdit);
	vbc->add_margin_child(TTR("Search:"), search_box);
	search_box->connect("text_changed", callable_mp(this, &PropertySelector::_text_changed));
	search_box->connect("gui_input", callable_mp(this, &PropertySelector::_sbox_input));
	search_options = memnew(Tree);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);
	set_ok_button_text(TTR("Open"));
	get_ok_button()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", callable_mp(this, &PropertySelector::_confirmed));
	search_options->connect("cell_selected", callable_mp(this, &PropertySelector::_item_selected));
	search_options->set_hide_root(true);
	search_options->set_hide_folding(true);

	help_bit = memnew(EditorHelpBit);
	vbc->add_margin_child(TTR("Description:"), help_bit);
	help_bit->get_rich_text()->set_fit_content(false);
	help_bit->get_rich_text()->set_custom_minimum_size(Size2(help_bit->get_rich_text()->get_minimum_size().x, 135 * EDSCALE));
	help_bit->connect("request_hide", callable_mp(this, &PropertySelector::_hide_requested));
}
