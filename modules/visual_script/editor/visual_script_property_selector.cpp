#include "visual_script_property_selector.h"
/*************************************************************************/
/*  visual_script_property_selector.cpp                                  */
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

#include "visual_script_property_selector.h"

#include "../visual_script.h"
#include "../visual_script_builtin_funcs.h"
#include "../visual_script_flow_control.h"
#include "../visual_script_func_nodes.h"
#include "../visual_script_nodes.h"
#include "core/os/keyboard.h"
#include "editor/doc_tools.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/main/node.h"
#include "scene/main/window.h"

void VisualScriptPropertySelector::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				results_tree->gui_input(k);
				search_box->accept_event();

				TreeItem *root = results_tree->get_root();
				if (!root->get_first_child()) {
					break;
				}

				TreeItem *current = results_tree->get_selected();

				TreeItem *item = results_tree->get_next_selected(root);
				while (item) {
					item->deselect(0);
					item = results_tree->get_next_selected(item);
				}

				current->select(0);

			} break;
			default:
				break;
		}
	}
}

void VisualScriptPropertySelector::_update_search_i(int p_int) {
	_update_search();
}

void VisualScriptPropertySelector::_update_search_s(String p_string) {
	_update_search();
}

void VisualScriptPropertySelector::_update_search() {
	set_title(TTR("Search VisualScript"));

	results_tree->clear();
	help_bit->set_text("");

	TreeItem *root = results_tree->create_item();
	bool found = false;
	StringName base = base_type;
	List<StringName> base_list;
	while (base) {
		base_list.push_back(base);
		base = ClassDB::get_parent_class_nocheck(base);
	}

	for (const StringName &E : base_list) {
		List<MethodInfo> methods;
		List<PropertyInfo> props;
		TreeItem *category = nullptr;
		Ref<Texture2D> type_icons[Variant::VARIANT_MAX] = {
			vbox->get_theme_icon(SNAME("Variant"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("bool"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("int"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("float"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("String"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Vector2"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Vector2i"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Rect2"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Rect2i"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Vector3"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Vector3i"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Transform2D"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Plane"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Quaternion"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("AABB"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Basis"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Transform3D"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Color"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("StringName"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("NodePath"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("RID"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("MiniObject"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Callable"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Signal"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Dictionary"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedByteArray"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedInt32Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedInt64Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedFloat32Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedFloat64Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedStringArray"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedVector2Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedVector3Array"), SNAME("EditorIcons")),
			vbox->get_theme_icon(SNAME("PackedColorArray"), SNAME("EditorIcons"))
		};
		{
			String b = String(E);
			category = results_tree->create_item(root);
			if (category) {
				category->set_text(0, b.replace_first("*", ""));
				category->set_selectable(0, false);
				Ref<Texture2D> icon;
				String rep = b.replace("*", "");
				icon = EditorNode::get_singleton()->get_class_icon(rep);
				category->set_icon(0, icon);
			}
		}
		if (properties || seq_connect) {
			if (instance) {
				instance->get_property_list(&props, true);
			} else {
				Object *obj = ObjectDB::get_instance(script);
				if (Object::cast_to<Script>(obj)) {
					Object::cast_to<Script>(obj)->get_script_property_list(&props);
				} else {
					ClassDB::get_property_list(E, &props, true);
				}
			}
			for (const PropertyInfo &F : props) {
				if (!(F.usage & PROPERTY_USAGE_EDITOR) && !(F.usage & PROPERTY_USAGE_SCRIPT_VARIABLE)) {
					continue;
				}

				if (type_filter.size() && type_filter.find(F.type) == -1) {
					continue;
				}

				// capitalize() also converts underscore to space, we'll match again both possible styles
				String get_text_raw = String(vformat(TTR("Get %s"), F.name));
				String get_text = get_text_raw.capitalize();
				String set_text_raw = String(vformat(TTR("Set %s"), F.name));
				String set_text = set_text_raw.capitalize();
				String input = search_box->get_text().capitalize();

				if (input == String() || get_text_raw.findn(input) != -1 || get_text.findn(input) != -1) {
					TreeItem *item = results_tree->create_item(category ? category : root);
					item->set_text(0, get_text);
					item->set_metadata(0, F.name);
					item->set_icon(0, type_icons[F.type]);
					item->set_metadata(1, "get");
					item->set_collapsed(true);
					item->set_selectable(0, true);
					item->set_selectable(1, false);
					item->set_selectable(2, false);
					item->set_metadata(2, connecting);
				}

				if (input == String() || set_text_raw.findn(input) != -1 || set_text.findn(input) != -1) {
					TreeItem *item = results_tree->create_item(category ? category : root);
					item->set_text(0, set_text);
					item->set_metadata(0, F.name);
					item->set_icon(0, type_icons[F.type]);
					item->set_metadata(1, "set");
					item->set_selectable(0, true);
					item->set_selectable(1, false);
					item->set_selectable(2, false);
					item->set_metadata(2, connecting);
				}
			}
		}
		{
			if (type != Variant::NIL) {
				Variant v;
				Callable::CallError ce;
				Variant::construct(type, v, nullptr, 0, ce);
				v.get_method_list(&methods);
			} else {
				Object *obj = ObjectDB::get_instance(script);
				if (Object::cast_to<Script>(obj)) {
					Object::cast_to<Script>(obj)->get_script_method_list(&methods);
				}

				ClassDB::get_method_list(E, &methods, true, true);
			}
		}
		for (List<MethodInfo>::Element *M = methods.front(); M; M = M->next()) {
			String name = M->get().name.get_slice(":", 0);
			if (name.begins_with("_") && !(M->get().flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (virtuals_only && !(M->get().flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			if (!virtuals_only && (M->get().flags & METHOD_FLAG_VIRTUAL)) {
				continue;
			}

			MethodInfo mi = M->get();
			String desc_arguments;
			if (mi.arguments.size() > 0) {
				desc_arguments = "(";
				for (int i = 0; i < mi.arguments.size(); i++) {
					if (i > 0) {
						desc_arguments += ", ";
					}
					if (mi.arguments[i].type == Variant::NIL) {
						desc_arguments += "var";
					} else if (mi.arguments[i].name.find(":") != -1) {
						desc_arguments += mi.arguments[i].name.get_slice(":", 1);
						mi.arguments[i].name = mi.arguments[i].name.get_slice(":", 0);
					} else {
						desc_arguments += Variant::get_type_name(mi.arguments[i].type);
					}
				}
				desc_arguments += ")";
			}
			String desc_raw = mi.name + desc_arguments;
			String desc = desc_raw.capitalize().replace("( ", "(");

			if (search_box->get_text() != String() &&
					name.findn(search_box->get_text()) == -1 &&
					desc.findn(search_box->get_text()) == -1 &&
					desc_raw.findn(search_box->get_text()) == -1) {
				continue;
			}

			TreeItem *item = results_tree->create_item(category ? category : root);
			item->set_text(0, desc);
			item->set_icon(0, vbox->get_theme_icon(SNAME("MemberMethod"), SNAME("EditorIcons")));
			item->set_metadata(0, name);
			item->set_selectable(0, true);

			item->set_metadata(1, "method");
			item->set_collapsed(true);
			item->set_selectable(1, false);

			item->set_selectable(2, false);
			item->set_metadata(2, connecting);
		}

		if (category && category->get_first_child() == nullptr) {
			memdelete(category); //old category was unused
		}
	}
	if (properties) {
		if (!seq_connect && !visual_script_generic) {
			get_visual_node_names("flow_control/type_cast", Set<String>(), found, root, search_box);
			get_visual_node_names("functions/built_in/print", Set<String>(), found, root, search_box);
			get_visual_node_names("functions/by_type/" + Variant::get_type_name(type), Set<String>(), found, root, search_box);
			get_visual_node_names("functions/deconstruct/" + Variant::get_type_name(type), Set<String>(), found, root, search_box);
			get_visual_node_names("operators/compare/", Set<String>(), found, root, search_box);
			if (type == Variant::INT) {
				get_visual_node_names("operators/bitwise/", Set<String>(), found, root, search_box);
			}
			if (type == Variant::BOOL) {
				get_visual_node_names("operators/logic/", Set<String>(), found, root, search_box);
			}
			if (type == Variant::BOOL || type == Variant::INT || type == Variant::FLOAT || type == Variant::VECTOR2 || type == Variant::VECTOR3) {
				get_visual_node_names("operators/math/", Set<String>(), found, root, search_box);
			}
		}
	}

	if (seq_connect && !visual_script_generic) {
		String text = search_box->get_text();
		create_visualscript_item(String("VisualScriptCondition"), root, text, String("Condition"));
		create_visualscript_item(String("VisualScriptSwitch"), root, text, String("Switch"));
		create_visualscript_item(String("VisualScriptSequence"), root, text, String("Sequence"));
		create_visualscript_item(String("VisualScriptIterator"), root, text, String("Iterator"));
		create_visualscript_item(String("VisualScriptWhile"), root, text, String("While"));
		create_visualscript_item(String("VisualScriptReturn"), root, text, String("Return"));
		get_visual_node_names("flow_control/type_cast", Set<String>(), found, root, search_box);
		get_visual_node_names("functions/built_in/print", Set<String>(), found, root, search_box);
	}

	if ((properties || seq_connect) && visual_script_generic) {
		get_visual_node_names("", Set<String>(), found, root, search_box);
	}

	TreeItem *selected_item = results_tree->search_item_text(search_box->get_text());
	if (!found && selected_item != nullptr) {
		selected_item->select(0);
		found = true;
	}

	get_ok_button()->set_disabled(root->get_first_child() == nullptr);
}

void VisualScriptPropertySelector::create_visualscript_item(const String &name, TreeItem *const root, const String &search_input, const String &text) {
	if (search_input == String() || text.findn(search_input) != -1) {
		TreeItem *item = results_tree->create_item(root);
		item->set_text(0, text);
		item->set_icon(0, vbox->get_theme_icon(SNAME("VisualScript"), SNAME("EditorIcons")));
		item->set_metadata(0, name);
		item->set_metadata(1, "action");
		item->set_selectable(0, true);
		item->set_collapsed(true);
		item->set_selectable(1, false);
		item->set_selectable(2, false);
		item->set_metadata(2, connecting);
	}
}

void VisualScriptPropertySelector::get_visual_node_names(const String &root_filter, const Set<String> &p_modifiers, bool &found, TreeItem *const root, LineEdit *const search_box) {
	Map<String, TreeItem *> path_cache;

	List<String> fnodes;
	VisualScriptLanguage::singleton->get_registered_node_names(&fnodes);

	for (const String &E : fnodes) {
		if (!E.begins_with(root_filter)) {
			continue;
		}
		Vector<String> path = E.split("/");

		// check if the name has the filter
		bool in_filter = false;
		Vector<String> tx_filters = search_box->get_text().split(" ");
		for (int i = 0; i < tx_filters.size(); i++) {
			if (tx_filters[i] == "") {
				in_filter = true;
			} else {
				in_filter = false;
			}
			if (E.findn(tx_filters[i]) != -1) {
				in_filter = true;
				break;
			}
		}
		if (!in_filter) {
			continue;
		}

		bool in_modifier = p_modifiers.is_empty();
		for (Set<String>::Element *F = p_modifiers.front(); F && in_modifier; F = F->next()) {
			if (E.findn(F->get()) != -1) {
				in_modifier = true;
			}
		}
		if (!in_modifier) {
			continue;
		}

		TreeItem *item = results_tree->create_item(root);
		Ref<VisualScriptNode> vnode = VisualScriptLanguage::singleton->create_node_from_name(E);
		Ref<VisualScriptOperator> vnode_operator = vnode;
		String type_name;
		if (vnode_operator.is_valid()) {
			String type;
			if (path.size() >= 2) {
				type = path[1];
			}
			type_name = type.capitalize() + " ";
		}
		Ref<VisualScriptFunctionCall> vnode_function_call = vnode;
		if (vnode_function_call.is_valid()) {
			String basic_type = Variant::get_type_name(vnode_function_call->get_basic_type());
			type_name = basic_type.capitalize() + " ";
		}
		Ref<VisualScriptConstructor> vnode_constructor = vnode;
		if (vnode_constructor.is_valid()) {
			type_name = "Construct ";
		}
		Ref<VisualScriptDeconstruct> vnode_deconstruct = vnode;
		if (vnode_deconstruct.is_valid()) {
			type_name = "Deconstruct ";
		}
		Vector<String> desc = path[path.size() - 1].replace("(", " ").replace(")", " ").replace(",", " ").split(" ");
		for (int i = 0; i < desc.size(); i++) {
			desc.write[i] = desc[i].capitalize();
			if (desc[i].ends_with(",")) {
				desc.write[i] = desc[i].replace(",", ", ");
			}
		}

		item->set_text(0, type_name + String("").join(desc));
		item->set_icon(0, vbox->get_theme_icon(SNAME("VisualScript"), SNAME("EditorIcons")));
		item->set_selectable(0, true);
		item->set_metadata(0, E);
		item->set_selectable(0, true);
		item->set_metadata(1, "visualscript");
		item->set_selectable(1, false);
		item->set_selectable(2, false);
		item->set_metadata(2, connecting);
	}
}

void VisualScriptPropertySelector::_confirmed() {
	TreeItem *ti = results_tree->get_selected();
	if (!ti) {
		return;
	}
	emit_signal(SNAME("selected"), ti->get_metadata(0), ti->get_metadata(1), ti->get_metadata(2));
	set_visible(false);
}

void VisualScriptPropertySelector::_item_selected() {
	help_bit->set_text("");

	TreeItem *item = results_tree->get_selected();
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

	DocTools *dd = EditorHelp::get_doc_data();
	String text;

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

		at_class = ClassDB::get_parent_class_nocheck(at_class);
	}
	at_class = class_type;

	while (at_class != String()) {
		Map<String, DocData::ClassDoc>::Element *C = dd->class_list.find(at_class);
		if (C) {
			for (int i = 0; i < C->get().methods.size(); i++) {
				if (C->get().methods[i].name == name) {
					text = DTR(C->get().methods[i].description);
				}
			}
		}

		at_class = ClassDB::get_parent_class_nocheck(at_class);
	}
	Vector<String> functions = name.rsplit("/", false);
	at_class = functions.size() > 3 ? functions[functions.size() - 2] : class_type;
	Map<String, DocData::ClassDoc>::Element *T = dd->class_list.find(at_class);
	if (T) {
		for (int i = 0; i < T->get().methods.size(); i++) {
			if (T->get().methods[i].name == functions[functions.size() - 1]) {
				text = DTR(T->get().methods[i].description);
			}
		}
	}

	List<String> *names = memnew(List<String>);
	VisualScriptLanguage::singleton->get_registered_node_names(names);
	if (names->find(name) != nullptr) {
		Ref<VisualScriptOperator> operator_node = VisualScriptLanguage::singleton->create_node_from_name(name);
		if (operator_node.is_valid()) {
			Map<String, DocData::ClassDoc>::Element *F = dd->class_list.find(operator_node->get_class_name());
			if (F) {
				text = Variant::get_operator_name(operator_node->get_operator());
			}
		}
		Ref<VisualScriptTypeCast> typecast_node = VisualScriptLanguage::singleton->create_node_from_name(name);
		if (typecast_node.is_valid()) {
			Map<String, DocData::ClassDoc>::Element *F = dd->class_list.find(typecast_node->get_class_name());
			if (F) {
				text = DTR(F->get().description);
			}
		}

		Ref<VisualScriptBuiltinFunc> builtin_node = VisualScriptLanguage::singleton->create_node_from_name(name);
		if (builtin_node.is_valid()) {
			Map<String, DocData::ClassDoc>::Element *F = dd->class_list.find(builtin_node->get_class_name());
			if (F) {
				for (int i = 0; i < F->get().constants.size(); i++) {
					if (F->get().constants[i].value.to_int() == int(builtin_node->get_func())) {
						text = DTR(F->get().constants[i].description);
					}
				}
			}
		}
	}

	memdelete(names);

	if (text == String()) {
		return;
	}

	help_bit->set_text(text);
}

void VisualScriptPropertySelector::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void VisualScriptPropertySelector::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		connect("confirmed", callable_mp(this, &VisualScriptPropertySelector::_confirmed));
	}
}

void VisualScriptPropertySelector::select_method_from_base_type(const String &p_base, const String &p_current, const bool p_virtuals_only, const bool p_connecting, bool clear_text) {
	base_type = p_base;
	selected = p_current;
	type = Variant::NIL;
	properties = false;
	instance = nullptr;
	virtuals_only = p_virtuals_only;

	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::set_type_filter(const Vector<Variant::Type> &p_type_filter) {
	type_filter = p_type_filter;
}

void VisualScriptPropertySelector::select_from_base_type(const String &p_base, const String &p_current, bool p_virtuals_only, bool p_seq_connect, const bool p_connecting, bool clear_text) {
	base_type = p_base;
	selected = p_current;
	type = Variant::NIL;
	properties = true;
	visual_script_generic = false;
	instance = nullptr;
	virtuals_only = p_virtuals_only;

	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	seq_connect = p_seq_connect;
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::select_from_script(const Ref<Script> &p_script, const String &p_current, const bool p_connecting, bool clear_text) {
	ERR_FAIL_COND(p_script.is_null());

	base_type = p_script->get_instance_base_type();
	selected = p_current;
	type = Variant::NIL;
	script = p_script->get_instance_id();
	properties = true;
	visual_script_generic = false;
	instance = nullptr;
	virtuals_only = false;

	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	seq_connect = false;
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::select_from_basic_type(Variant::Type p_type, const String &p_current, const bool p_connecting, bool clear_text) {
	ERR_FAIL_COND(p_type == Variant::NIL);
	base_type = "";
	selected = p_current;
	type = p_type;
	properties = true;
	visual_script_generic = false;
	instance = nullptr;
	virtuals_only = false;

	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	seq_connect = false;
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::select_from_action(const String &p_type, const String &p_current, const bool p_connecting, bool clear_text) {
	base_type = p_type;
	selected = p_current;
	type = Variant::NIL;
	properties = false;
	visual_script_generic = false;
	instance = nullptr;
	virtuals_only = false;

	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	seq_connect = true;
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::select_from_instance(Object *p_instance, const String &p_current, const bool p_connecting, const String &p_basetype, bool clear_text) {
	base_type = p_basetype;
	selected = p_current;
	type = Variant::NIL;
	properties = true;
	visual_script_generic = false;
	instance = p_instance;
	virtuals_only = false;

	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	seq_connect = false;
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::select_from_visual_script(const String &p_base, const bool p_connecting, bool clear_text) {
	base_type = p_base;
	selected = "";
	type = Variant::NIL;
	properties = true;
	visual_script_generic = true;
	instance = nullptr;
	virtuals_only = false;
	show_window(.5f);
	if (clear_text) {
		search_box->set_text("");
	} else {
		search_box->select_all();
	}
	search_box->grab_focus();
	connecting = p_connecting;

	_update_search();
}

void VisualScriptPropertySelector::show_window(float p_screen_ratio) {
	popup_centered_ratio(p_screen_ratio);
}

void VisualScriptPropertySelector::_bind_methods() {
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::STRING, "category"), PropertyInfo(Variant::BOOL, "connecting")));
}

VisualScriptPropertySelector::VisualScriptPropertySelector() {
	virtuals_only = false;
	seq_connect = false;

	vbox = memnew(VBoxContainer);
	add_child(vbox);

	// Create the search box and filter controls (at the top).
	HBoxContainer *hbox = memnew(HBoxContainer);
	vbox->add_margin_child(TTR("Search:"), hbox);

	search_box = memnew(LineEdit);
	search_box->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->connect("text_changed", callable_mp(this, &VisualScriptPropertySelector::_update_search_s));
	search_box->connect("gui_input", callable_mp(this, &VisualScriptPropertySelector::_sbox_input));
	register_text_enter(search_box);
	hbox->add_child(search_box);

	case_sensitive_button = memnew(Button);
	//	case_sensitive_button->set_flat(true); comented until update icon is working
	case_sensitive_button->set_tooltip(TTR("Case Sensitive"));
	case_sensitive_button->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_search));
	case_sensitive_button->set_toggle_mode(true);
	case_sensitive_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(case_sensitive_button);

	hierarchy_button = memnew(Button);
	//	hierarchy_button->set_flat(true); comented until update icon is working
	hierarchy_button->set_tooltip(TTR("Show Hierarchy"));
	hierarchy_button->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_search));
	hierarchy_button->set_toggle_mode(true);
	hierarchy_button->set_pressed(true);
	hierarchy_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(hierarchy_button);

	filter_combo = memnew(OptionButton);
	filter_combo->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	filter_combo->set_stretch_ratio(0); // Fixed width.
	filter_combo->add_item(TTR("Display All"), SEARCH_ALL);
	filter_combo->add_separator();
	filter_combo->add_item(TTR("Classes Only"), SEARCH_CLASSES);
	filter_combo->add_item(TTR("Constructors Only"), SEARCH_CONSTRUCTORS);
	filter_combo->add_item(TTR("Methods Only"), SEARCH_METHODS);
	filter_combo->add_item(TTR("Operators Only"), SEARCH_OPERATORS);
	filter_combo->add_item(TTR("Signals Only"), SEARCH_SIGNALS);
	filter_combo->add_item(TTR("Constants Only"), SEARCH_CONSTANTS);
	filter_combo->add_item(TTR("Properties Only"), SEARCH_PROPERTIES);
	filter_combo->add_item(TTR("Theme Properties Only"), SEARCH_THEME_ITEMS);
	filter_combo->connect("item_selected", callable_mp(this, &VisualScriptPropertySelector::_update_search_i));
	hbox->add_child(filter_combo);

	scope_combo = memnew(OptionButton);
	scope_combo->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	scope_combo->set_stretch_ratio(0); // Fixed width.
	scope_combo->add_item(TTR("Search Related"), SCOPE_RELATED);
	scope_combo->add_separator();
	scope_combo->add_item(TTR("Search Base"), SCOPE_BASE);
	scope_combo->add_item(TTR("Search Inheriters"), SCOPE_INHERITERS);
	scope_combo->add_item(TTR("Search Unrelated"), SCOPE_UNRELATED);
	scope_combo->connect("item_selected", callable_mp(this, &VisualScriptPropertySelector::_update_search_i));
	hbox->add_child(scope_combo);

	results_tree = memnew(Tree);
	results_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	results_tree->set_hide_root(true);
	//	results_tree->set_hide_folding(true);
	results_tree->set_columns(3); // needed for original _update_search()
	//	results_tree->set_columns(2);
	results_tree->set_column_title(0, TTR("Name"));
	results_tree->set_column_clip_content(0, true);
	results_tree->set_column_title(1, TTR("Member Type"));
	results_tree->set_column_expand(1, false);
	results_tree->set_column_custom_minimum_width(1, 150 * EDSCALE);
	results_tree->set_column_clip_content(1, true);
	results_tree->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	results_tree->set_select_mode(Tree::SELECT_ROW);
	//	results_tree->set_column_expand(2, false);
	results_tree->connect("item_activated", callable_mp(this, &VisualScriptPropertySelector::_confirmed));
	results_tree->connect("cell_selected", callable_mp(this, &VisualScriptPropertySelector::_item_selected));
	vbox->add_margin_child(TTR("Matches:"), results_tree, true);

	help_bit = memnew(EditorHelpBit);
	vbox->add_margin_child(TTR("Description:"), help_bit);
	help_bit->connect("request_hide", callable_mp(this, &VisualScriptPropertySelector::_hide_requested));
	get_ok_button()->set_text(TTR("Open"));
	get_ok_button()->set_disabled(true);
	set_hide_on_ok(false);
}
