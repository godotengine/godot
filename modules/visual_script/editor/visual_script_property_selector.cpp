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
#include "editor/editor_feature_profile.h"
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

void VisualScriptPropertySelector::_update_results_i(int p_int) {
	_update_results();
}

void VisualScriptPropertySelector::_update_results_s(String p_string) {
	_update_results();
}

void VisualScriptPropertySelector::_update_results() {
	//	node_runner = Ref<NodeRunner>(memnew(NodeRunner(vbox, &result_nodes)));
	doc_runner = Ref<DocRunner>(memnew(DocRunner(&result_nodes, &result_class_list)));
	search_runner = Ref<SearchRunner>(memnew(SearchRunner(this, results_tree, &result_class_list)));
	set_process(true);
	//// Get all nodes and atach them to there categorys
	//List<String> fnodes;
	//VisualScriptLanguage::singleton->get_registered_node_names(&fnodes);
}

void VisualScriptPropertySelector::_update_search() {
	//	return;
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
	switch (p_what) {
		//case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
		//	_update_icons();
		//} break;
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &VisualScriptPropertySelector::_confirmed));
			//_update_icons();
		} break;
		case NOTIFICATION_PROCESS: {
			// Update background search.
			if (doc_runner.is_valid() || search_runner.is_valid()) {
				if (doc_runner->work() && search_runner->work()) {
					// Search done.
					get_ok_button()->set_disabled(!results_tree->get_selected());

					doc_runner = Ref<DocRunner>();
					search_runner = Ref<SearchRunner>();
					set_process(false);
				}
			} else {
				// if one is valid
				set_process(false);
			}
		} break;
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
	_update_results();
	//_update_search();
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
	search_box->connect("text_changed", callable_mp(this, &VisualScriptPropertySelector::_update_results_s));
	search_box->connect("gui_input", callable_mp(this, &VisualScriptPropertySelector::_sbox_input));
	register_text_enter(search_box);
	hbox->add_child(search_box);

	case_sensitive_button = memnew(Button);
	//	case_sensitive_button->set_flat(true); comented until update icon is working
	case_sensitive_button->set_tooltip(TTR("Case Sensitive"));
	case_sensitive_button->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
	case_sensitive_button->set_toggle_mode(true);
	case_sensitive_button->set_focus_mode(Control::FOCUS_NONE);
	hbox->add_child(case_sensitive_button);

	hierarchy_button = memnew(Button);
	//	hierarchy_button->set_flat(true); comented until update icon is working
	hierarchy_button->set_tooltip(TTR("Show Hierarchy"));
	hierarchy_button->connect("pressed", callable_mp(this, &VisualScriptPropertySelector::_update_results));
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
	filter_combo->connect("item_selected", callable_mp(this, &VisualScriptPropertySelector::_update_results_i));
	hbox->add_child(filter_combo);

	scope_combo = memnew(OptionButton);
	scope_combo->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	scope_combo->set_stretch_ratio(0); // Fixed width.
	scope_combo->add_item(TTR("Search Related"), SCOPE_RELATED);
	scope_combo->add_separator();
	scope_combo->add_item(TTR("Search Base"), SCOPE_BASE);
	scope_combo->add_item(TTR("Search Inheriters"), SCOPE_INHERITERS);
	scope_combo->add_item(TTR("Search Unrelated"), SCOPE_UNRELATED);
	scope_combo->connect("item_selected", callable_mp(this, &VisualScriptPropertySelector::_update_results_i));
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

bool VisualScriptPropertySelector::DocRunner::_slice() {
	// Return true when fases are completed, otherwise false.
	bool phase_done = false;
	switch (phase) {
		case PHASE_INIT_SEARCH:
			phase_done = _phase_init_search();
			break;
		case PHASE_GET_ALL_FOLDER_PATHS:
			phase_done = _phase_get_all_folder_paths();
			break;
		case PHASE_GET_ALL_FILE_PATHS:
			phase_done = _phase_get_all_file_paths();
			break;
		case PHASE_MAX:
			return true;
		default:
			WARN_PRINT("Invalid or unhandled phase in EditorHelpSearch::Runner, aborting search.");
			return true;
	};

	if (phase_done) {
		phase++;
	}
	return false;
}

bool VisualScriptPropertySelector::DocRunner::_phase_init_search() {
	// Reset data
	result_nodes->clear();
	*result_class_list = EditorHelp::get_doc_data()->class_list;

	// Config
	_extension_filter.clear();
	_extension_filter.append("gd");
	_extension_filter.append("vs");

	// State
	_current_dir = "";
	PackedStringArray init_folder;
	init_folder.push_back("");
	_folders_stack.clear();
	_folders_stack.push_back(init_folder);
	_initial_files_count = 0;

	return true;
}

bool VisualScriptPropertySelector::DocRunner::_phase_get_all_folder_paths() {
	if (_folders_stack.size() != 0) {
		// Scan folders first so we can build a list of files and have progress info later.

		PackedStringArray &folders_to_scan = _folders_stack.write[_folders_stack.size() - 1];

		if (folders_to_scan.size() != 0) {
			// Scan one folder below.

			String folder_name = folders_to_scan[folders_to_scan.size() - 1];
			folders_to_scan.resize(folders_to_scan.size() - 1); //pop_back(...);

			_current_dir = _current_dir.plus_file(folder_name);

			PackedStringArray sub_dirs;
			_scan_dir("res://" + _current_dir, sub_dirs);

			_folders_stack.push_back(sub_dirs);
		} else {
			// Go back one level.
			_folders_stack.resize(_folders_stack.size() - 1); //pop_back(...);
			_current_dir = _current_dir.get_base_dir();

			if (_folders_stack.size() == 0) {
				// All folders scanned.
				_initial_files_count = _files_to_scan.size();
			}
		}
		return false;
	}
	return true;
}

bool VisualScriptPropertySelector::DocRunner::_phase_get_all_file_paths() {
	if (_files_to_scan.size() != 0) {
		String fpath = _files_to_scan[_files_to_scan.size() - 1];
		_files_to_scan.resize(_files_to_scan.size() - 1); //pop_back(...);
		_scan_file(fpath);
		return false;
	}
	return true;
}

void VisualScriptPropertySelector::DocRunner::_scan_dir(String path, PackedStringArray &out_folders) {
	DirAccessRef dir = DirAccess::open(path);
	if (!dir) {
		print_verbose("Cannot open directory! " + path);
		return;
	}

	dir->list_dir_begin();

	for (int i = 0; i < 1000; ++i) {
		String file = dir->get_next();

		if (file == "") {
			break;
		}

		// If there is a .gdignore file in the directory, skip searching the directory.
		if (file == ".gdignore") {
			break;
		}

		// Ignore special directories (such as those beginning with . and the project data directory).
		String project_data_dir_name = ProjectSettings::get_singleton()->get_project_data_dir_name();
		if (file.begins_with(".") || file == project_data_dir_name) {
			continue;
		}
		if (dir->current_is_hidden()) {
			continue;
		}

		if (dir->current_is_dir()) {
			out_folders.push_back(file);

		} else {
			String file_ext = file.get_extension();
			if (_extension_filter.has(file_ext)) {
				_files_to_scan.push_back(path.plus_file(file));
			}
		}
	}
}

void VisualScriptPropertySelector::DocRunner::_scan_file(String fpath) {
	Ref<Script> script;
	script = ResourceLoader::load(fpath);

	if (script->get_instance_base_type() == "VisualScriptCustomNode") {
		Ref<VisualScriptCustomNode> vs_c_node;
		vs_c_node.instantiate();
		vs_c_node->set_script(script);
		result_nodes->push_back(vs_c_node);
		return;
	}
	DocData::ClassDoc class_doc = DocData::ClassDoc();
	class_doc.name = fpath;
	class_doc.inherits = script->get_instance_base_type();
	class_doc.brief_description = "a project script (brief_description)";
	class_doc.description = "a project script (long_description)";

	Object *obj = ObjectDB::get_instance(script->get_instance_id());
	if (Object::cast_to<Script>(obj)) {
		List<MethodInfo> methods;
		Object::cast_to<Script>(obj)->get_script_method_list(&methods);
		for (List<MethodInfo>::Element *M = methods.front(); M; M = M->next()) {
			class_doc.methods.push_back(_get_method_doc(M->get()));
		}

		List<MethodInfo> signals;
		Object::cast_to<Script>(obj)->get_script_signal_list(&signals);
		for (List<MethodInfo>::Element *S = signals.front(); S; S = S->next()) {
			class_doc.signals.push_back(_get_method_doc(S->get()));
		}

		List<PropertyInfo> propertys;
		Object::cast_to<Script>(obj)->get_script_property_list(&propertys);
		for (List<PropertyInfo>::Element *P = propertys.front(); P; P = P->next()) {
			DocData::PropertyDoc pd = DocData::PropertyDoc();
			pd.name = P->get().name;
			pd.type = Variant::get_type_name(P->get().type);
			class_doc.properties.push_back(pd);
		}
	}
	result_class_list->insert(class_doc.name, class_doc);
}

DocData::MethodDoc VisualScriptPropertySelector::DocRunner::_get_method_doc(MethodInfo method_info) {
	DocData::MethodDoc method_doc = DocData::MethodDoc();
	method_doc.name = method_info.name;
	method_doc.return_type = Variant::get_type_name(method_info.return_val.type);
	method_doc.description = "No description available";
	for (List<PropertyInfo>::Element *P = method_info.arguments.front(); P; P = P->next()) {
		DocData::ArgumentDoc argument_doc = DocData::ArgumentDoc();
		argument_doc.name = P->get().name;
		argument_doc.type = Variant::get_type_name(P->get().type);
		method_doc.arguments.push_back(argument_doc);
	}
	return method_doc;
}

bool VisualScriptPropertySelector::DocRunner::work(uint64_t slot) {
	// Return true when the search has been completed, otherwise false.
	const uint64_t until = OS::get_singleton()->get_ticks_usec() + slot;
	while (!_slice()) {
		if (OS::get_singleton()->get_ticks_usec() > until) {
			return false;
		}
	}
	return true;
}

VisualScriptPropertySelector::DocRunner::DocRunner(Vector<Ref<VisualScriptNode>> *p_result_nodes, Map<String, DocData::ClassDoc> *p_result_class_list) :
		result_nodes(p_result_nodes),
		result_class_list(p_result_class_list) {
}

bool VisualScriptPropertySelector::SearchRunner::_is_class_disabled_by_feature_profile(const StringName &p_class) {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (profile.is_null()) {
		return false;
	}

	StringName class_name = p_class;
	while (class_name != StringName()) {
		if (!ClassDB::class_exists(class_name)) {
			return false;
		}

		if (profile->is_class_disabled(class_name)) {
			return true;
		}
		class_name = ClassDB::get_parent_class(class_name);
	}

	return false;
}

bool VisualScriptPropertySelector::SearchRunner::_slice() {
	bool phase_done = false;
	switch (phase) {
		case PHASE_MATCH_CLASSES_INIT:
			phase_done = _phase_match_classes_init();
			break;
		case PHASE_MATCH_CLASSES:
			phase_done = _phase_match_classes();
			break;
		case PHASE_CLASS_ITEMS_INIT:
			phase_done = _phase_class_items_init();
			break;
		case PHASE_CLASS_ITEMS:
			phase_done = _phase_class_items();
			break;
		case PHASE_MEMBER_ITEMS_INIT:
			phase_done = _phase_member_items_init();
			break;
		case PHASE_MEMBER_ITEMS:
			phase_done = _phase_member_items();
			break;
		case PHASE_SELECT_MATCH:
			phase_done = _phase_select_match();
			break;
		case PHASE_MAX:
			return true;
		default:
			WARN_PRINT("Invalid or unhandled phase in EditorHelpSearch::Runner, aborting search.");
			return true;
	};

	if (phase_done) {
		phase++;
	}
	return false;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_match_classes_init() {
	iterator_doc = class_docs->front();
	matches.clear();
	matched_item = nullptr;
	match_highest_score = 0;

	search_flags = selector_ui->filter_combo->get_selected_id();
	if (selector_ui->case_sensitive_button->is_pressed()) {
		search_flags |= SEARCH_CASE_SENSITIVE;
	}
	if (selector_ui->hierarchy_button->is_pressed()) {
		search_flags |= SEARCH_SHOW_HIERARCHY;
	}

	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_match_classes() {
	DocData::ClassDoc &class_doc = iterator_doc->value();
	if (!_is_class_disabled_by_feature_profile(class_doc.name)) {
		matches[class_doc.name] = ClassMatch();
		ClassMatch &match = matches[class_doc.name];

		match.doc = &class_doc;

		// Match class name.
		if (search_flags & SEARCH_CLASSES) {
			match.name = term == "" || _match_string(term, class_doc.name);
		}

		// Match members if the term is long enough.
		if (term.length() > 1) {
			if (search_flags & SEARCH_CONSTRUCTORS) {
				for (int i = 0; i < class_doc.constructors.size(); i++) {
					String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? class_doc.constructors[i].name : class_doc.constructors[i].name.to_lower();
					if (method_name.find(term) > -1 ||
							(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
							(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
							(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
						match.constructors.push_back(const_cast<DocData::MethodDoc *>(&class_doc.constructors[i]));
					}
				}
			}
			if (search_flags & SEARCH_METHODS) {
				for (int i = 0; i < class_doc.methods.size(); i++) {
					String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? class_doc.methods[i].name : class_doc.methods[i].name.to_lower();
					if (method_name.find(term) > -1 ||
							(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
							(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
							(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
						match.methods.push_back(const_cast<DocData::MethodDoc *>(&class_doc.methods[i]));
					}
				}
			}
			if (search_flags & SEARCH_OPERATORS) {
				for (int i = 0; i < class_doc.operators.size(); i++) {
					String method_name = (search_flags & SEARCH_CASE_SENSITIVE) ? class_doc.operators[i].name : class_doc.operators[i].name.to_lower();
					if (method_name.find(term) > -1 ||
							(term.begins_with(".") && method_name.begins_with(term.substr(1))) ||
							(term.ends_with("(") && method_name.ends_with(term.left(term.length() - 1).strip_edges())) ||
							(term.begins_with(".") && term.ends_with("(") && method_name == term.substr(1, term.length() - 2).strip_edges())) {
						match.operators.push_back(const_cast<DocData::MethodDoc *>(&class_doc.operators[i]));
					}
				}
			}
			if (search_flags & SEARCH_SIGNALS) {
				for (int i = 0; i < class_doc.signals.size(); i++) {
					if (_match_string(term, class_doc.signals[i].name)) {
						match.signals.push_back(const_cast<DocData::MethodDoc *>(&class_doc.signals[i]));
					}
				}
			}
			if (search_flags & SEARCH_CONSTANTS) {
				for (int i = 0; i < class_doc.constants.size(); i++) {
					if (_match_string(term, class_doc.constants[i].name)) {
						match.constants.push_back(const_cast<DocData::ConstantDoc *>(&class_doc.constants[i]));
					}
				}
			}
			if (search_flags & SEARCH_PROPERTIES) {
				for (int i = 0; i < class_doc.properties.size(); i++) {
					if (_match_string(term, class_doc.properties[i].name) || _match_string(term, class_doc.properties[i].getter) || _match_string(term, class_doc.properties[i].setter)) {
						match.properties.push_back(const_cast<DocData::PropertyDoc *>(&class_doc.properties[i]));
					}
				}
			}
			if (search_flags & SEARCH_THEME_ITEMS) {
				for (int i = 0; i < class_doc.theme_properties.size(); i++) {
					if (_match_string(term, class_doc.theme_properties[i].name)) {
						match.theme_properties.push_back(const_cast<DocData::ThemeItemDoc *>(&class_doc.theme_properties[i]));
					}
				}
			}
		}
	}

	iterator_doc = iterator_doc->next();
	return !iterator_doc;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_class_items_init() {
	iterator_match = matches.front();

	results_tree->clear();
	root_item = results_tree->create_item();
	class_items.clear();

	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_class_items() {
	ClassMatch &match = iterator_match->value();

	if (search_flags & SEARCH_SHOW_HIERARCHY) {
		if (match.required()) {
			_create_class_hierarchy(match);
		}
	} else {
		if (match.name) {
			_create_class_item(root_item, match.doc, false);
		}
	}

	iterator_match = iterator_match->next();
	return !iterator_match;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_member_items_init() {
	iterator_match = matches.front();

	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_member_items() {
	ClassMatch &match = iterator_match->value();

	TreeItem *parent = (search_flags & SEARCH_SHOW_HIERARCHY) ? class_items[match.doc->name] : root_item;
	bool constructor_created = false;
	for (int i = 0; i < match.methods.size(); i++) {
		String text = match.methods[i]->name;
		if (!constructor_created) {
			if (match.doc->name == match.methods[i]->name) {
				text += " " + TTR("(constructors)");
				constructor_created = true;
			}
		} else {
			if (match.doc->name == match.methods[i]->name) {
				continue;
			}
		}
		_create_method_item(parent, match.doc, text, match.methods[i]);
	}
	for (int i = 0; i < match.signals.size(); i++) {
		_create_signal_item(parent, match.doc, match.signals[i]);
	}
	for (int i = 0; i < match.constants.size(); i++) {
		_create_constant_item(parent, match.doc, match.constants[i]);
	}
	for (int i = 0; i < match.properties.size(); i++) {
		_create_property_item(parent, match.doc, match.properties[i]);
	}
	for (int i = 0; i < match.theme_properties.size(); i++) {
		_create_theme_property_item(parent, match.doc, match.theme_properties[i]);
	}

	iterator_match = iterator_match->next();
	return !iterator_match;
}

bool VisualScriptPropertySelector::SearchRunner::_phase_select_match() {
	if (matched_item) {
		matched_item->select(0);
	}
	return true;
}

bool VisualScriptPropertySelector::SearchRunner::_match_string(const String &p_term, const String &p_string) const {
	if (search_flags & SEARCH_CASE_SENSITIVE) {
		return p_string.find(p_term) > -1;
	} else {
		return p_string.findn(p_term) > -1;
	}
}

void VisualScriptPropertySelector::SearchRunner::_match_item(TreeItem *p_item, const String &p_text) {
	float inverse_length = 1.f / float(p_text.length());

	// Favor types where search term is a substring close to the start of the type.
	float w = 0.5f;
	int pos = p_text.findn(term);
	float score = (pos > -1) ? 1.0f - w * MIN(1, 3 * pos * inverse_length) : MAX(0.f, .9f - w);

	// Favor shorter items: they resemble the search term more.
	w = 0.1f;
	score *= (1 - w) + w * (term.length() * inverse_length);

	if (match_highest_score == 0 || score > match_highest_score) {
		matched_item = p_item;
		match_highest_score = score;
	}
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_class_hierarchy(const ClassMatch &p_match) {
	if (class_items.has(p_match.doc->name)) {
		return class_items[p_match.doc->name];
	}

	// Ensure parent nodes are created first.
	TreeItem *parent = root_item;
	if (p_match.doc->inherits != "") {
		if (class_items.has(p_match.doc->inherits)) {
			parent = class_items[p_match.doc->inherits];
		} else {
			ClassMatch &base_match = matches[p_match.doc->inherits];
			parent = _create_class_hierarchy(base_match);
		}
	}

	TreeItem *class_item = _create_class_item(parent, p_match.doc, !p_match.name);
	class_items[p_match.doc->name] = class_item;
	return class_item;
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_class_item(TreeItem *p_parent, const DocData::ClassDoc *p_doc, bool p_gray) {
	Ref<Texture2D> icon = empty_icon;
	if (ui_service->has_theme_icon(p_doc->name, "EditorIcons")) {
		icon = ui_service->get_theme_icon(p_doc->name, "EditorIcons");
	} else if (ClassDB::class_exists(p_doc->name) && ClassDB::is_parent_class(p_doc->name, "Object")) {
		icon = ui_service->get_theme_icon(SNAME("Object"), SNAME("EditorIcons"));
	}
	String tooltip = p_doc->brief_description.strip_edges();

	TreeItem *item = results_tree->create_item(p_parent);
	item->set_icon(0, icon);
	item->set_text(0, p_doc->name);
	item->set_text(1, TTR("Class"));
	item->set_tooltip(0, tooltip);
	item->set_tooltip(1, tooltip);
	item->set_metadata(0, "class_name:" + p_doc->name);
	if (p_gray) {
		item->set_custom_color(0, disabled_color);
		item->set_custom_color(1, disabled_color);
	}

	_match_item(item, p_doc->name);

	return item;
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_method_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc) {
	String tooltip = p_doc->return_type + " " + p_class_doc->name + "." + p_doc->name + "(";
	for (int i = 0; i < p_doc->arguments.size(); i++) {
		const DocData::ArgumentDoc &arg = p_doc->arguments[i];
		tooltip += arg.type + " " + arg.name;
		if (arg.default_value != "") {
			tooltip += " = " + arg.default_value;
		}
		if (i < p_doc->arguments.size() - 1) {
			tooltip += ", ";
		}
	}
	tooltip += ")";
	return _create_member_item(p_parent, p_class_doc->name, "MemberMethod", p_doc->name, p_text, TTRC("Method"), "method", tooltip);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_signal_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc) {
	String tooltip = p_doc->return_type + " " + p_class_doc->name + "." + p_doc->name + "(";
	for (int i = 0; i < p_doc->arguments.size(); i++) {
		const DocData::ArgumentDoc &arg = p_doc->arguments[i];
		tooltip += arg.type + " " + arg.name;
		if (arg.default_value != "") {
			tooltip += " = " + arg.default_value;
		}
		if (i < p_doc->arguments.size() - 1) {
			tooltip += ", ";
		}
	}
	tooltip += ")";
	return _create_member_item(p_parent, p_class_doc->name, "MemberSignal", p_doc->name, p_doc->name, TTRC("Signal"), "signal", tooltip);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_constant_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ConstantDoc *p_doc) {
	String tooltip = p_class_doc->name + "." + p_doc->name;
	return _create_member_item(p_parent, p_class_doc->name, "MemberConstant", p_doc->name, p_doc->name, TTRC("Constant"), "constant", tooltip);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::PropertyDoc *p_doc) {
	String tooltip = p_doc->type + " " + p_class_doc->name + "." + p_doc->name;
	tooltip += "\n    " + p_class_doc->name + "." + p_doc->setter + "(value) setter";
	tooltip += "\n    " + p_class_doc->name + "." + p_doc->getter + "() getter";
	return _create_member_item(p_parent, p_class_doc->name, "MemberProperty", p_doc->name, p_doc->name, TTRC("Property"), "property", tooltip);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_theme_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ThemeItemDoc *p_doc) {
	String tooltip = p_doc->type + " " + p_class_doc->name + "." + p_doc->name;
	return _create_member_item(p_parent, p_class_doc->name, "MemberTheme", p_doc->name, p_doc->name, TTRC("Theme Property"), "theme_item", tooltip);
}

TreeItem *VisualScriptPropertySelector::SearchRunner::_create_member_item(TreeItem *p_parent, const String &p_class_name, const String &p_icon, const String &p_name, const String &p_text, const String &p_type, const String &p_metatype, const String &p_tooltip) {
	Ref<Texture2D> icon;
	String text;
	if (search_flags & SEARCH_SHOW_HIERARCHY) {
		icon = ui_service->get_theme_icon(p_icon, SNAME("EditorIcons"));
		text = p_text;
	} else {
		icon = ui_service->get_theme_icon(p_icon, SNAME("EditorIcons"));
		/*// In flat mode, show the class icon.
if (ui_service->has_icon(p_class_name, "EditorIcons"))
icon = ui_service->get_icon(p_class_name, "EditorIcons");
else if (ClassDB::is_parent_class(p_class_name, "Object"))
icon = ui_service->get_icon("Object", "EditorIcons");*/
		text = p_class_name + "." + p_text;
	}

	TreeItem *item = results_tree->create_item(p_parent);
	item->set_icon(0, icon);
	item->set_text(0, text);
	item->set_text(1, TTRGET(p_type));
	item->set_tooltip(0, p_tooltip);
	item->set_tooltip(1, p_tooltip);
	item->set_metadata(0, "class_" + p_metatype + ":" + p_class_name + ":" + p_name);

	_match_item(item, p_name);

	return item;
}

bool VisualScriptPropertySelector::SearchRunner::work(uint64_t slot) {
	// Return true when the search has been completed, otherwise false.
	const uint64_t until = OS::get_singleton()->get_ticks_usec() + slot;
	while (!_slice()) {
		if (OS::get_singleton()->get_ticks_usec() > until) {
			return false;
		}
	}
	return true;
}

VisualScriptPropertySelector::SearchRunner::SearchRunner(VisualScriptPropertySelector *p_selector_ui, Tree *p_results_tree, Map<String, DocData::ClassDoc> *p_class_docs) :
		selector_ui(p_selector_ui),
		ui_service(p_selector_ui->vbox),
		results_tree(p_results_tree),
		term(p_selector_ui->search_box->get_text()),
		class_docs(p_class_docs) {
}
