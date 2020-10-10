/*************************************************************************/
/*  property_selector.cpp                                                */
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

#include "property_selector.h"

#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor_scale.h"

void PropertySelector::_text_changed(const String &p_newtext) {
	_update_search();
}

void PropertySelector::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case KEY_UP:
			case KEY_DOWN:
			case KEY_PAGEUP:
			case KEY_PAGEDOWN: {
				search_options->call("_gui_input", k);
				search_box->accept_event();

				TreeItem *root = search_options->get_root();
				if (!root->get_children()) {
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
			v = Variant::construct(type, nullptr, 0, ce);

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

		Ref<Texture2D> type_icons[Variant::VARIANT_MAX] = {
			search_options->get_theme_icon("Variant", "EditorIcons"),
			search_options->get_theme_icon("bool", "EditorIcons"),
			search_options->get_theme_icon("int", "EditorIcons"),
			search_options->get_theme_icon("float", "EditorIcons"),
			search_options->get_theme_icon("String", "EditorIcons"),
			search_options->get_theme_icon("Vector2", "EditorIcons"),
			search_options->get_theme_icon("Rect2", "EditorIcons"),
			search_options->get_theme_icon("Vector3", "EditorIcons"),
			search_options->get_theme_icon("Transform2D", "EditorIcons"),
			search_options->get_theme_icon("Plane", "EditorIcons"),
			search_options->get_theme_icon("Quat", "EditorIcons"),
			search_options->get_theme_icon("AABB", "EditorIcons"),
			search_options->get_theme_icon("Basis", "EditorIcons"),
			search_options->get_theme_icon("Transform", "EditorIcons"),
			search_options->get_theme_icon("Color", "EditorIcons"),
			search_options->get_theme_icon("Path", "EditorIcons"),
			search_options->get_theme_icon("RID", "EditorIcons"),
			search_options->get_theme_icon("Object", "EditorIcons"),
			search_options->get_theme_icon("Dictionary", "EditorIcons"),
			search_options->get_theme_icon("Array", "EditorIcons"),
			search_options->get_theme_icon("PackedByteArray", "EditorIcons"),
			search_options->get_theme_icon("PackedInt32Array", "EditorIcons"),
			search_options->get_theme_icon("PackedFloat32Array", "EditorIcons"),
			search_options->get_theme_icon("PackedStringArray", "EditorIcons"),
			search_options->get_theme_icon("PackedVector2Array", "EditorIcons"),
			search_options->get_theme_icon("PackedVector3Array", "EditorIcons"),
			search_options->get_theme_icon("PackedColorArray", "EditorIcons")
		};

		for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
			if (E->get().usage == PROPERTY_USAGE_CATEGORY) {
				if (category && category->get_children() == nullptr) {
					memdelete(category); //old category was unused
				}
				category = search_options->create_item(root);
				category->set_text(0, E->get().name);
				category->set_selectable(0, false);

				Ref<Texture2D> icon;
				if (E->get().name == "Script Variables") {
					icon = search_options->get_theme_icon("Script", "EditorIcons");
				} else {
					icon = EditorNode::get_singleton()->get_class_icon(E->get().name);
				}
				category->set_icon(0, icon);
				continue;
			}

			if (!(E->get().usage & PROPERTY_USAGE_EDITOR) && !(E->get().usage & PROPERTY_USAGE_SCRIPT_VARIABLE)) {
				continue;
			}

			if (search_box->get_text() != String() && E->get().name.findn(search_text) == -1) {
				continue;
			}

			if (type_filter.size() && type_filter.find(E->get().type) == -1) {
				continue;
			}

			TreeItem *item = search_options->create_item(category ? category : root);
			item->set_text(0, E->get().name);
			item->set_metadata(0, E->get().name);
			item->set_icon(0, type_icons[E->get().type]);

			if (!found && search_box->get_text() != String() && E->get().name.findn(search_text) != -1) {
				item->select(0);
				found = true;
			}

			item->set_selectable(0, true);
		}

		if (category && category->get_children() == nullptr) {
			memdelete(category); //old category was unused
		}
	} else {
		List<MethodInfo> methods;

		if (type != Variant::NIL) {
			Variant v;
			Callable::CallError ce;
			v = Variant::construct(type, nullptr, 0, ce);
			v.get_method_list(&methods);
		} else {
			Object *obj = ObjectDB::get_instance(script);
			if (Object::cast_to<Script>(obj)) {
				methods.push_back(MethodInfo("*Script Methods"));
				Object::cast_to<Script>(obj)->get_script_method_list(&methods);
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

		for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
			if (E->get().name.begins_with("*")) {
				if (category && category->get_children() == nullptr) {
					memdelete(category); //old category was unused
				}
				category = search_options->create_item(root);
				category->set_text(0, E->get().name.replace_first("*", ""));
				category->set_selectable(0, false);

				Ref<Texture2D> icon;
				script_methods = false;
				String rep = E->get().name.replace("*", "");
				if (E->get().name == "*Script Methods") {
					icon = search_options->get_theme_icon("Script", "EditorIcons");
					script_methods = true;
				} else {
					icon = EditorNode::get_singleton()->get_class_icon(rep);
				}
				category->set_icon(0, icon);

				continue;
			}

			String name = E->get().name.get_slice(":", 0);
			if (!script_methods && name.begins_with("_") && !(E->get().flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (virtuals_only && !(E->get().flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (!virtuals_only && (E->get().flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (search_box->get_text() != String() && name.findn(search_text) == -1) {
				continue;
			}

			TreeItem *item = search_options->create_item(category ? category : root);

			MethodInfo mi = E->get();

			String desc;
			if (mi.name.find(":") != -1) {
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
				} else if (mi.arguments[i].name.find(":") != -1) {
					desc += vformat(": %s", mi.arguments[i].name.get_slice(":", 1));
					mi.arguments[i].name = mi.arguments[i].name.get_slice(":", 0);
				} else {
					desc += vformat(": %s", Variant::get_type_name(mi.arguments[i].type));
				}
			}

			desc += ")";

			if (E->get().flags & METHOD_FLAG_CONST) {
				desc += " const";
			}

			if (E->get().flags & METHOD_FLAG_VIRTUAL) {
				desc += " virtual";
			}

			item->set_text(0, desc);
			item->set_metadata(0, name);
			item->set_selectable(0, true);

			if (!found && search_box->get_text() != String() && name.findn(search_text) != -1) {
				item->select(0);
				found = true;
			}
		}

		if (category && category->get_children() == nullptr) {
			memdelete(category); //old category was unused
		}
	}

	get_ok()->set_disabled(root->get_children() == nullptr);
}

void PropertySelector::_confirmed() {
	TreeItem *ti = search_options->get_selected();
	if (!ti) {
		return;
	}
	emit_signal("selected", ti->get_metadata(0));
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

	} else {
		class_type = base_type;
	}

	DocData *dd = EditorHelp::get_doc_data();
	String text;

	if (properties) {
		String at_class = class_type;

		while (at_class != String()) {
			Map<String, DocData::ClassDoc>::Element *E = dd->class_list.find(at_class);
			if (E) {
				for (int i = 0; i < E->get().properties.size(); i++) {
					if (E->get().properties[i].name == name) {
						text = DTR(E->get().properties[i].description);
					}
				}
			}

			at_class = ClassDB::get_parent_class(at_class);
		}
	} else {
		String at_class = class_type;

		while (at_class != String()) {
			Map<String, DocData::ClassDoc>::Element *E = dd->class_list.find(at_class);
			if (E) {
				for (int i = 0; i < E->get().methods.size(); i++) {
					if (E->get().methods[i].name == name) {
						text = DTR(E->get().methods[i].description);
					}
				}
			}

			at_class = ClassDB::get_parent_class(at_class);
		}
	}

	if (text == String()) {
		return;
	}

	help_bit->set_text(text);
}

void PropertySelector::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void PropertySelector::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		connect("confirmed", callable_mp(this, &PropertySelector::_confirmed));
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		disconnect("confirmed", callable_mp(this, &PropertySelector::_confirmed));
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
	get_ok()->set_text(TTR("Open"));
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", callable_mp(this, &PropertySelector::_confirmed));
	search_options->connect("cell_selected", callable_mp(this, &PropertySelector::_item_selected));
	search_options->set_hide_root(true);
	search_options->set_hide_folding(true);
	virtuals_only = false;

	help_bit = memnew(EditorHelpBit);
	vbc->add_margin_child(TTR("Description:"), help_bit);
	help_bit->connect("request_hide", callable_mp(this, &PropertySelector::_hide_requested));
}
