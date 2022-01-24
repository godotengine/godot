/*************************************************************************/
/*  property_editor.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "property_editor.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/io/image_loader.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "core/math/expression.h"
#include "core/object/class_db.h"
#include "core/os/keyboard.h"
#include "core/string/print_string.h"
#include "core/templates/pair.h"
#include "editor/array_property_edit.h"
#include "editor/create_dialog.h"
#include "editor/dictionary_property_edit.h"
#include "editor/editor_export.h"
#include "editor/editor_file_system.h"
#include "editor/editor_help.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/filesystem_dock.h"
#include "editor/multi_node_edit.h"
#include "editor/property_selector.h"
#include "scene/gui/label.h"
#include "scene/main/window.h"
#include "scene/resources/font.h"
#include "scene/resources/packed_scene.h"
#include "scene/scene_string_names.h"

void EditorResourceConversionPlugin::_bind_methods() {
	GDVIRTUAL_BIND(_converts_to);
	GDVIRTUAL_BIND(_handles, "resource");
	GDVIRTUAL_BIND(_convert, "resource");
}

String EditorResourceConversionPlugin::converts_to() const {
	String ret;
	if (GDVIRTUAL_CALL(_converts_to, ret)) {
		return ret;
	}

	return "";
}

bool EditorResourceConversionPlugin::handles(const Ref<Resource> &p_resource) const {
	bool ret;
	if (GDVIRTUAL_CALL(_handles, p_resource, ret)) {
		return ret;
	}

	return false;
}

Ref<Resource> EditorResourceConversionPlugin::convert(const Ref<Resource> &p_resource) const {
	RES ret;
	if (GDVIRTUAL_CALL(_convert, p_resource, ret)) {
		return ret;
	}

	return Ref<Resource>();
}

void CustomPropertyEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_WM_CLOSE_REQUEST) {
		hide();
	}
}

void CustomPropertyEditor::_menu_option(int p_which) {
	switch (type) {
		case Variant::INT: {
			if (hint == PROPERTY_HINT_FLAGS) {
				int val = v;

				if (val & (1 << p_which)) {
					val &= ~(1 << p_which);
				} else {
					val |= (1 << p_which);
				}

				v = val;
				emit_signal(SNAME("variant_changed"));
			} else if (hint == PROPERTY_HINT_ENUM) {
				v = menu->get_item_metadata(p_which);
				emit_signal(SNAME("variant_changed"));
			}
		} break;
		case Variant::STRING: {
			if (hint == PROPERTY_HINT_ENUM) {
				v = hint_text.get_slice(",", p_which);
				emit_signal(SNAME("variant_changed"));
			}
		} break;
		case Variant::OBJECT: {
			switch (p_which) {
				case OBJ_MENU_LOAD: {
					file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
					String type = (hint == PROPERTY_HINT_RESOURCE_TYPE) ? hint_text : String();

					List<String> extensions;
					for (int i = 0; i < type.get_slice_count(","); i++) {
						ResourceLoader::get_recognized_extensions_for_type(type.get_slice(",", i), &extensions);
					}

					Set<String> valid_extensions;
					for (const String &E : extensions) {
						valid_extensions.insert(E);
					}

					file->clear_filters();
					for (Set<String>::Element *E = valid_extensions.front(); E; E = E->next()) {
						file->add_filter("*." + E->get() + " ; " + E->get().to_upper());
					}

					file->popup_file_dialog();
				} break;

				case OBJ_MENU_EDIT: {
					REF r = v;

					if (!r.is_null()) {
						emit_signal(SNAME("resource_edit_request"));
						hide();
					}
				} break;
				case OBJ_MENU_CLEAR: {
					v = Variant();
					emit_signal(SNAME("variant_changed"));
					hide();
				} break;

				case OBJ_MENU_MAKE_UNIQUE: {
					Ref<Resource> res_orig = v;
					if (res_orig.is_null()) {
						return;
					}

					List<PropertyInfo> property_list;
					res_orig->get_property_list(&property_list);
					List<Pair<String, Variant>> propvalues;

					for (const PropertyInfo &pi : property_list) {
						Pair<String, Variant> p;
						if (pi.usage & PROPERTY_USAGE_STORAGE) {
							p.first = pi.name;
							p.second = res_orig->get(pi.name);
						}

						propvalues.push_back(p);
					}

					String orig_type = res_orig->get_class();

					Object *inst = ClassDB::instantiate(orig_type);

					Ref<Resource> res = Ref<Resource>(Object::cast_to<Resource>(inst));

					ERR_FAIL_COND(res.is_null());

					for (const Pair<String, Variant> &p : propvalues) {
						res->set(p.first, p.second);
					}

					v = res;
					emit_signal(SNAME("variant_changed"));
					hide();
				} break;

				case OBJ_MENU_COPY: {
					EditorSettings::get_singleton()->set_resource_clipboard(v);

				} break;
				case OBJ_MENU_PASTE: {
					v = EditorSettings::get_singleton()->get_resource_clipboard();
					emit_signal(SNAME("variant_changed"));

				} break;
				case OBJ_MENU_NEW_SCRIPT: {
					if (Object::cast_to<Node>(owner)) {
						SceneTreeDock::get_singleton()->open_script_dialog(Object::cast_to<Node>(owner), false);
					}

				} break;
				case OBJ_MENU_EXTEND_SCRIPT: {
					if (Object::cast_to<Node>(owner)) {
						SceneTreeDock::get_singleton()->open_script_dialog(Object::cast_to<Node>(owner), true);
					}

				} break;
				case OBJ_MENU_SHOW_IN_FILE_SYSTEM: {
					RES r = v;
					FileSystemDock *file_system_dock = FileSystemDock::get_singleton();
					file_system_dock->navigate_to_path(r->get_path());
					// Ensure that the FileSystem dock is visible.
					TabContainer *tab_container = (TabContainer *)file_system_dock->get_parent_control();
					tab_container->set_current_tab(file_system_dock->get_index());
				} break;
				default: {
					if (p_which >= CONVERT_BASE_ID) {
						int to_type = p_which - CONVERT_BASE_ID;

						Vector<Ref<EditorResourceConversionPlugin>> conversions = EditorNode::get_singleton()->find_resource_conversion_plugin(RES(v));

						ERR_FAIL_INDEX(to_type, conversions.size());

						Ref<Resource> new_res = conversions[to_type]->convert(v);

						v = new_res;
						emit_signal(SNAME("variant_changed"));
						break;
					}
					ERR_FAIL_COND(inheritors_array.is_empty());

					String intype = inheritors_array[p_which - TYPE_BASE_ID];

					if (intype == "ViewportTexture") {
						scene_tree->set_title(TTR("Pick a Viewport"));
						scene_tree->popup_scenetree_dialog();
						picking_viewport = true;
						return;
					}

					Variant obj = ClassDB::instantiate(intype);

					if (!obj) {
						if (ScriptServer::is_global_class(intype)) {
							obj = EditorNode::get_editor_data().script_class_instance(intype);
						} else {
							obj = EditorNode::get_editor_data().instance_custom_type(intype, "Resource");
						}
					}

					ERR_BREAK(!obj);
					Resource *res = Object::cast_to<Resource>(obj);
					ERR_BREAK(!res);
					if (owner && hint == PROPERTY_HINT_RESOURCE_TYPE && hint_text == "Script") {
						//make visual script the right type
						res->call("set_instance_base_type", owner->get_class());
					}

					v = obj;
					emit_signal(SNAME("variant_changed"));

				} break;
			}

		} break;
		default: {
		}
	}
}

void CustomPropertyEditor::hide_menu() {
	menu->hide();
}

Variant CustomPropertyEditor::get_variant() const {
	return v;
}

String CustomPropertyEditor::get_name() const {
	return name;
}

bool CustomPropertyEditor::edit(Object *p_owner, const String &p_name, Variant::Type p_type, const Variant &p_variant, int p_hint, String p_hint_text) {
	owner = p_owner;
	updating = true;
	name = p_name;
	v = p_variant;
	field_names.clear();
	hint = p_hint;
	hint_text = p_hint_text;
	type_button->hide();
	if (color_picker) {
		color_picker->hide();
	}
	texture_preview->hide();
	inheritors_array.clear();
	text_edit->hide();
	easing_draw->hide();
	spinbox->hide();
	slider->hide();
	menu->clear();
	menu->reset_size();

	for (int i = 0; i < MAX_VALUE_EDITORS; i++) {
		if (i < MAX_VALUE_EDITORS / 4) {
			value_hboxes[i]->hide();
		}
		value_editor[i]->hide();
		value_label[i]->hide();
		if (i < 4) {
			scroll[i]->hide();
		}
	}

	for (int i = 0; i < MAX_ACTION_BUTTONS; i++) {
		action_buttons[i]->hide();
	}

	checks20gc->hide();
	for (int i = 0; i < 20; i++) {
		checks20[i]->hide();
	}

	type = (p_variant.get_type() != Variant::NIL && p_variant.get_type() != Variant::RID && p_type != Variant::OBJECT) ? p_variant.get_type() : p_type;

	switch (type) {
		case Variant::BOOL: {
			checks20gc->show();

			CheckBox *c = checks20[0];
			c->set_text("True");
			checks20gc->set_position(Vector2(4, 4) * EDSCALE);
			c->set_pressed(v);
			c->show();

			checks20gc->set_size(checks20gc->get_minimum_size());
			set_size(checks20gc->get_position() + checks20gc->get_size() + c->get_size() + Vector2(4, 4) * EDSCALE);

		} break;
		case Variant::INT:
		case Variant::FLOAT: {
			if (hint == PROPERTY_HINT_RANGE) {
				int c = hint_text.get_slice_count(",");
				float min = 0, max = 100, step = type == Variant::FLOAT ? .01 : 1;
				if (c >= 1) {
					if (!hint_text.get_slice(",", 0).is_empty()) {
						min = hint_text.get_slice(",", 0).to_float();
					}
				}
				if (c >= 2) {
					if (!hint_text.get_slice(",", 1).is_empty()) {
						max = hint_text.get_slice(",", 1).to_float();
					}
				}

				if (c >= 3) {
					if (!hint_text.get_slice(",", 2).is_empty()) {
						step = hint_text.get_slice(",", 2).to_float();
					}
				}

				if (c >= 4 && hint_text.get_slice(",", 3) == "slider") {
					slider->set_min(min);
					slider->set_max(max);
					slider->set_step(step);
					slider->set_value(v);
					slider->show();
					set_size(Size2(110, 30) * EDSCALE);
				} else {
					spinbox->set_min(min);
					spinbox->set_max(max);
					spinbox->set_step(step);
					spinbox->set_value(v);
					spinbox->show();
					set_size(Size2(70, 35) * EDSCALE);
				}

			} else if (hint == PROPERTY_HINT_ENUM) {
				Vector<String> options = hint_text.split(",");
				int current_val = 0;
				for (int i = 0; i < options.size(); i++) {
					Vector<String> text_split = options[i].split(":");
					if (text_split.size() != 1) {
						current_val = text_split[1].to_int();
					}
					menu->add_item(text_split[0]);
					menu->set_item_metadata(i, current_val);
					current_val += 1;
				}
				menu->set_position(get_position());
				menu->popup();
				hide();
				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_LAYERS_2D_PHYSICS ||
					hint == PROPERTY_HINT_LAYERS_2D_RENDER ||
					hint == PROPERTY_HINT_LAYERS_2D_NAVIGATION ||
					hint == PROPERTY_HINT_LAYERS_3D_PHYSICS ||
					hint == PROPERTY_HINT_LAYERS_3D_RENDER ||
					hint == PROPERTY_HINT_LAYERS_3D_NAVIGATION) {
				String basename;
				switch (hint) {
					case PROPERTY_HINT_LAYERS_2D_RENDER:
						basename = "layer_names/2d_render";
						break;
					case PROPERTY_HINT_LAYERS_2D_PHYSICS:
						basename = "layer_names/2d_physics";
						break;
					case PROPERTY_HINT_LAYERS_2D_NAVIGATION:
						basename = "layer_names/2d_navigation";
						break;
					case PROPERTY_HINT_LAYERS_3D_RENDER:
						basename = "layer_names/3d_render";
						break;
					case PROPERTY_HINT_LAYERS_3D_PHYSICS:
						basename = "layer_names/3d_physics";
						break;
					case PROPERTY_HINT_LAYERS_3D_NAVIGATION:
						basename = "layer_names/3d_navigation";
						break;
				}

				checks20gc->show();
				uint32_t flgs = v;
				for (int i = 0; i < 2; i++) {
					Point2 ofs(4, 4);
					ofs.y += 22 * i;
					for (int j = 0; j < 10; j++) {
						int idx = i * 10 + j;
						CheckBox *c = checks20[idx];
						c->set_text(ProjectSettings::get_singleton()->get(basename + "/layer_" + itos(idx + 1)));
						c->set_pressed(flgs & (1 << (i * 10 + j)));
						c->show();
					}
				}

				show();

				checks20gc->set_position(Vector2(4, 4) * EDSCALE);
				checks20gc->set_size(checks20gc->get_minimum_size());

				set_size(Vector2(4, 4) * EDSCALE + checks20gc->get_position() + checks20gc->get_size());

			} else if (hint == PROPERTY_HINT_EXP_EASING) {
				easing_draw->set_anchor_and_offset(SIDE_LEFT, Control::ANCHOR_BEGIN, 5 * EDSCALE);
				easing_draw->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, -5 * EDSCALE);
				easing_draw->set_anchor_and_offset(SIDE_TOP, Control::ANCHOR_BEGIN, 5 * EDSCALE);
				easing_draw->set_anchor_and_offset(SIDE_BOTTOM, Control::ANCHOR_END, -30 * EDSCALE);
				type_button->set_anchor_and_offset(SIDE_LEFT, Control::ANCHOR_BEGIN, 3 * EDSCALE);
				type_button->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, -3 * EDSCALE);
				type_button->set_anchor_and_offset(SIDE_TOP, Control::ANCHOR_END, -25 * EDSCALE);
				type_button->set_anchor_and_offset(SIDE_BOTTOM, Control::ANCHOR_END, -7 * EDSCALE);
				type_button->set_text(TTR("Preset..."));
				type_button->get_popup()->clear();
				type_button->get_popup()->add_item(TTR("Linear"), EASING_LINEAR);
				type_button->get_popup()->add_item(TTR("Ease In"), EASING_EASE_IN);
				type_button->get_popup()->add_item(TTR("Ease Out"), EASING_EASE_OUT);
				if (hint_text != "attenuation") {
					type_button->get_popup()->add_item(TTR("Zero"), EASING_ZERO);
					type_button->get_popup()->add_item(TTR("Easing In-Out"), EASING_IN_OUT);
					type_button->get_popup()->add_item(TTR("Easing Out-In"), EASING_OUT_IN);
				}

				type_button->show();
				easing_draw->show();
				set_size(Size2(200, 150) * EDSCALE);
			} else if (hint == PROPERTY_HINT_FLAGS) {
				Vector<String> flags = hint_text.split(",");
				for (int i = 0; i < flags.size(); i++) {
					String flag = flags[i];
					if (flag.is_empty()) {
						continue;
					}
					menu->add_check_item(flag, i);
					int f = v;
					if (f & (1 << i)) {
						menu->set_item_checked(menu->get_item_index(i), true);
					}
				}
				menu->set_position(get_position());
				menu->popup();
				hide();
				updating = false;
				return false;

			} else {
				List<String> names;
				names.push_back("value:");
				config_value_editors(1, 1, 50, names);
				value_editor[0]->set_text(TS->format_number(String::num(v)));
			}

		} break;
		case Variant::STRING: {
			if (hint == PROPERTY_HINT_LOCALE_ID) {
				List<String> names;
				names.push_back(TTR("Locale..."));
				names.push_back(TTR("Clear"));
				config_action_buttons(names);
			} else if (hint == PROPERTY_HINT_FILE || hint == PROPERTY_HINT_GLOBAL_FILE) {
				List<String> names;
				names.push_back(TTR("File..."));
				names.push_back(TTR("Clear"));
				config_action_buttons(names);
			} else if (hint == PROPERTY_HINT_DIR || hint == PROPERTY_HINT_GLOBAL_DIR) {
				List<String> names;
				names.push_back(TTR("Dir..."));
				names.push_back(TTR("Clear"));
				config_action_buttons(names);
			} else if (hint == PROPERTY_HINT_ENUM) {
				Vector<String> options = hint_text.split(",");
				for (int i = 0; i < options.size(); i++) {
					menu->add_item(options[i], i);
				}
				menu->set_position(get_position());
				menu->popup();
				hide();
				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_MULTILINE_TEXT) {
				text_edit->show();
				text_edit->set_text(v);
				text_edit->deselect();

				int button_margin = text_edit->get_theme_constant(SNAME("button_margin"), SNAME("Dialogs"));
				int margin = text_edit->get_theme_constant(SNAME("margin"), SNAME("Dialogs"));

				action_buttons[0]->set_anchor(SIDE_LEFT, Control::ANCHOR_END);
				action_buttons[0]->set_anchor(SIDE_TOP, Control::ANCHOR_END);
				action_buttons[0]->set_anchor(SIDE_RIGHT, Control::ANCHOR_END);
				action_buttons[0]->set_anchor(SIDE_BOTTOM, Control::ANCHOR_END);
				action_buttons[0]->set_begin(Point2(-70 * EDSCALE, -button_margin + 5 * EDSCALE));
				action_buttons[0]->set_end(Point2(-margin, -margin));
				action_buttons[0]->set_text(TTR("Close"));
				action_buttons[0]->show();

			} else if (hint == PROPERTY_HINT_TYPE_STRING) {
				if (!create_dialog) {
					create_dialog = memnew(CreateDialog);
					create_dialog->connect("create", callable_mp(this, &CustomPropertyEditor::_create_dialog_callback));
					add_child(create_dialog);
				}

				if (!hint_text.is_empty()) {
					create_dialog->set_base_type(hint_text);
				} else {
					create_dialog->set_base_type("Object");
				}

				create_dialog->popup_create(false);
				hide();
				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_METHOD_OF_VARIANT_TYPE) {
#define MAKE_PROPSELECT                                                                                            \
	if (!property_select) {                                                                                        \
		property_select = memnew(PropertySelector);                                                                \
		property_select->connect("selected", callable_mp(this, &CustomPropertyEditor::_create_selected_property)); \
		add_child(property_select);                                                                                \
	}                                                                                                              \
	hide();

				MAKE_PROPSELECT;

				Variant::Type type = Variant::NIL;
				for (int i = 0; i < Variant::VARIANT_MAX; i++) {
					if (hint_text == Variant::get_type_name(Variant::Type(i))) {
						type = Variant::Type(i);
					}
				}
				if (type != Variant::NIL) {
					property_select->select_method_from_basic_type(type, v);
				}
				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_METHOD_OF_BASE_TYPE) {
				MAKE_PROPSELECT

				property_select->select_method_from_base_type(hint_text, v);

				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_METHOD_OF_INSTANCE) {
				MAKE_PROPSELECT

				Object *instance = ObjectDB::get_instance(ObjectID(hint_text.to_int()));
				if (instance) {
					property_select->select_method_from_instance(instance, v);
				}
				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_METHOD_OF_SCRIPT) {
				MAKE_PROPSELECT

				Object *obj = ObjectDB::get_instance(ObjectID(hint_text.to_int()));
				if (Object::cast_to<Script>(obj)) {
					property_select->select_method_from_script(Object::cast_to<Script>(obj), v);
				}

				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE) {
				MAKE_PROPSELECT
				Variant::Type type = Variant::NIL;
				String tname = hint_text;
				if (tname.find(".") != -1) {
					tname = tname.get_slice(".", 0);
				}
				for (int i = 0; i < Variant::VARIANT_MAX; i++) {
					if (tname == Variant::get_type_name(Variant::Type(i))) {
						type = Variant::Type(Variant::Type(i));
					}
				}

				if (type != Variant::NIL) {
					property_select->select_property_from_basic_type(type, v);
				}

				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_PROPERTY_OF_BASE_TYPE) {
				MAKE_PROPSELECT

				property_select->select_property_from_base_type(hint_text, v);

				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_PROPERTY_OF_INSTANCE) {
				MAKE_PROPSELECT

				Object *instance = ObjectDB::get_instance(ObjectID(hint_text.to_int()));
				if (instance) {
					property_select->select_property_from_instance(instance, v);
				}

				updating = false;
				return false;

			} else if (hint == PROPERTY_HINT_PROPERTY_OF_SCRIPT) {
				MAKE_PROPSELECT

				Object *obj = ObjectDB::get_instance(ObjectID(hint_text.to_int()));
				if (Object::cast_to<Script>(obj)) {
					property_select->select_property_from_script(Object::cast_to<Script>(obj), v);
				}

				updating = false;
				return false;

			} else {
				List<String> names;
				names.push_back("string:");
				config_value_editors(1, 1, 50, names);
				value_editor[0]->set_text(v);
			}

		} break;
		case Variant::VECTOR2: {
			field_names.push_back("x");
			field_names.push_back("y");
			config_value_editors(2, 2, 10, field_names);
			Vector2 vec = v;
			value_editor[0]->set_text(String::num(vec.x));
			value_editor[1]->set_text(String::num(vec.y));
		} break;
		case Variant::RECT2: {
			field_names.push_back("x");
			field_names.push_back("y");
			field_names.push_back("w");
			field_names.push_back("h");
			config_value_editors(4, 4, 10, field_names);
			Rect2 r = v;
			value_editor[0]->set_text(String::num(r.position.x));
			value_editor[1]->set_text(String::num(r.position.y));
			value_editor[2]->set_text(String::num(r.size.x));
			value_editor[3]->set_text(String::num(r.size.y));
		} break;
		case Variant::VECTOR3: {
			field_names.push_back("x");
			field_names.push_back("y");
			field_names.push_back("z");
			config_value_editors(3, 3, 10, field_names);
			Vector3 vec = v;
			value_editor[0]->set_text(String::num(vec.x));
			value_editor[1]->set_text(String::num(vec.y));
			value_editor[2]->set_text(String::num(vec.z));
		} break;
		case Variant::PLANE: {
			field_names.push_back("x");
			field_names.push_back("y");
			field_names.push_back("z");
			field_names.push_back("d");
			config_value_editors(4, 4, 10, field_names);
			Plane plane = v;
			value_editor[0]->set_text(String::num(plane.normal.x));
			value_editor[1]->set_text(String::num(plane.normal.y));
			value_editor[2]->set_text(String::num(plane.normal.z));
			value_editor[3]->set_text(String::num(plane.d));

		} break;
		case Variant::QUATERNION: {
			field_names.push_back("x");
			field_names.push_back("y");
			field_names.push_back("z");
			field_names.push_back("w");
			config_value_editors(4, 4, 10, field_names);
			Quaternion q = v;
			value_editor[0]->set_text(String::num(q.x));
			value_editor[1]->set_text(String::num(q.y));
			value_editor[2]->set_text(String::num(q.z));
			value_editor[3]->set_text(String::num(q.w));

		} break;
		case Variant::AABB: {
			field_names.push_back("px");
			field_names.push_back("py");
			field_names.push_back("pz");
			field_names.push_back("sx");
			field_names.push_back("sy");
			field_names.push_back("sz");
			config_value_editors(6, 3, 16, field_names);

			AABB aabb = v;
			value_editor[0]->set_text(String::num(aabb.position.x));
			value_editor[1]->set_text(String::num(aabb.position.y));
			value_editor[2]->set_text(String::num(aabb.position.z));
			value_editor[3]->set_text(String::num(aabb.size.x));
			value_editor[4]->set_text(String::num(aabb.size.y));
			value_editor[5]->set_text(String::num(aabb.size.z));

		} break;
		case Variant::TRANSFORM2D: {
			field_names.push_back("xx");
			field_names.push_back("xy");
			field_names.push_back("yx");
			field_names.push_back("yy");
			field_names.push_back("ox");
			field_names.push_back("oy");
			config_value_editors(6, 2, 16, field_names);

			Transform2D basis = v;
			for (int i = 0; i < 6; i++) {
				value_editor[i]->set_text(String::num(basis.elements[i / 2][i % 2]));
			}

		} break;
		case Variant::BASIS: {
			field_names.push_back("xx");
			field_names.push_back("xy");
			field_names.push_back("xz");
			field_names.push_back("yx");
			field_names.push_back("yy");
			field_names.push_back("yz");
			field_names.push_back("zx");
			field_names.push_back("zy");
			field_names.push_back("zz");
			config_value_editors(9, 3, 16, field_names);

			Basis basis = v;
			for (int i = 0; i < 9; i++) {
				value_editor[i]->set_text(String::num(basis.elements[i / 3][i % 3]));
			}

		} break;
		case Variant::TRANSFORM3D: {
			field_names.push_back("xx");
			field_names.push_back("xy");
			field_names.push_back("xz");
			field_names.push_back("xo");
			field_names.push_back("yx");
			field_names.push_back("yy");
			field_names.push_back("yz");
			field_names.push_back("yo");
			field_names.push_back("zx");
			field_names.push_back("zy");
			field_names.push_back("zz");
			field_names.push_back("zo");
			config_value_editors(12, 4, 16, field_names);

			Transform3D tr = v;
			for (int i = 0; i < 9; i++) {
				value_editor[(i / 3) * 4 + i % 3]->set_text(String::num(tr.basis.elements[i / 3][i % 3]));
			}

			value_editor[3]->set_text(String::num(tr.origin.x));
			value_editor[7]->set_text(String::num(tr.origin.y));
			value_editor[11]->set_text(String::num(tr.origin.z));

		} break;
		case Variant::COLOR: {
			if (!color_picker) {
				//late init for performance
				color_picker = memnew(ColorPicker);
				color_picker->set_deferred_mode(true);
				value_vbox->add_child(color_picker);
				color_picker->hide();
				color_picker->connect("color_changed", callable_mp(this, &CustomPropertyEditor::_color_changed));

				// get default color picker mode from editor settings
				int default_color_mode = EDITOR_GET("interface/inspector/default_color_picker_mode");
				if (default_color_mode == 1) {
					color_picker->set_hsv_mode(true);
				} else if (default_color_mode == 2) {
					color_picker->set_raw_mode(true);
				}

				int picker_shape = EDITOR_GET("interface/inspector/default_color_picker_shape");
				color_picker->set_picker_shape((ColorPicker::PickerShapeType)picker_shape);
			}

			color_picker->show();
			color_picker->set_edit_alpha(hint != PROPERTY_HINT_COLOR_NO_ALPHA);
			color_picker->set_pick_color(v);
			color_picker->set_focus_on_line_edit();

		} break;

		case Variant::NODE_PATH: {
			List<String> names;
			names.push_back(TTR("Assign"));
			names.push_back(TTR("Clear"));

			if (owner && owner->is_class("Node") && (v.get_type() == Variant::NODE_PATH) && Object::cast_to<Node>(owner)->has_node(v)) {
				names.push_back(TTR("Select Node"));
			}

			config_action_buttons(names);

		} break;
		case Variant::OBJECT: {
			if (hint != PROPERTY_HINT_RESOURCE_TYPE) {
				break;
			}

			if (p_name == "script" && hint_text == "Script" && Object::cast_to<Node>(owner)) {
				menu->add_item(TTR("New Script"), OBJ_MENU_NEW_SCRIPT);
				menu->add_separator();
			} else if (!hint_text.is_empty()) {
				int idx = 0;

				Vector<EditorData::CustomType> custom_resources;

				if (EditorNode::get_editor_data().get_custom_types().has("Resource")) {
					custom_resources = EditorNode::get_editor_data().get_custom_types()["Resource"];
				}

				for (int i = 0; i < hint_text.get_slice_count(","); i++) {
					String base = hint_text.get_slice(",", i);

					Set<String> valid_inheritors;
					valid_inheritors.insert(base);
					List<StringName> inheritors;
					ClassDB::get_inheriters_from_class(base.strip_edges(), &inheritors);

					for (int j = 0; j < custom_resources.size(); j++) {
						inheritors.push_back(custom_resources[j].name);
					}

					List<StringName>::Element *E = inheritors.front();
					while (E) {
						valid_inheritors.insert(E->get());
						E = E->next();
					}

					for (Set<String>::Element *j = valid_inheritors.front(); j; j = j->next()) {
						const String &t = j->get();

						bool is_custom_resource = false;
						Ref<Texture2D> icon;
						if (!custom_resources.is_empty()) {
							for (int k = 0; k < custom_resources.size(); k++) {
								if (custom_resources[k].name == t) {
									is_custom_resource = true;
									if (custom_resources[k].icon.is_valid()) {
										icon = custom_resources[k].icon;
									}
									break;
								}
							}
						}

						if (!is_custom_resource && !ClassDB::can_instantiate(t)) {
							continue;
						}

						inheritors_array.push_back(t);

						int id = TYPE_BASE_ID + idx;

						menu->add_item(vformat(TTR("New %s"), t), id);

						idx++;
					}
				}

				if (menu->get_item_count()) {
					menu->add_separator();
				}
			}

			menu->add_item(TTR("Load"), OBJ_MENU_LOAD);

			if (!RES(v).is_null()) {
				menu->add_item(TTR("Edit"), OBJ_MENU_EDIT);
				menu->add_item(TTR("Clear"), OBJ_MENU_CLEAR);
				menu->add_item(TTR("Make Unique"), OBJ_MENU_MAKE_UNIQUE);

				RES r = v;
				if (r.is_valid() && r->get_path().is_resource_file()) {
					menu->add_separator();
					menu->add_item(TTR("Show in FileSystem"), OBJ_MENU_SHOW_IN_FILE_SYSTEM);
				}
			}

			RES cb = EditorSettings::get_singleton()->get_resource_clipboard();
			bool paste_valid = false;
			if (cb.is_valid()) {
				if (hint_text.is_empty()) {
					paste_valid = true;
				} else {
					for (int i = 0; i < hint_text.get_slice_count(","); i++) {
						if (ClassDB::is_parent_class(cb->get_class(), hint_text.get_slice(",", i))) {
							paste_valid = true;
							break;
						}
					}
				}
			}

			if (!RES(v).is_null() || paste_valid) {
				menu->add_separator();

				if (!RES(v).is_null()) {
					menu->add_item(TTR("Copy"), OBJ_MENU_COPY);
				}

				if (paste_valid) {
					menu->add_item(TTR("Paste"), OBJ_MENU_PASTE);
				}
			}

			if (!RES(v).is_null()) {
				Vector<Ref<EditorResourceConversionPlugin>> conversions = EditorNode::get_singleton()->find_resource_conversion_plugin(RES(v));
				if (conversions.size()) {
					menu->add_separator();
				}
				for (int i = 0; i < conversions.size(); i++) {
					String what = conversions[i]->converts_to();
					menu->add_item(vformat(TTR("Convert to %s"), what), CONVERT_BASE_ID + i);
				}
			}

			menu->set_position(get_position());
			menu->popup();
			hide();
			updating = false;
			return false;
		} break;
		case Variant::DICTIONARY: {
		} break;
		case Variant::PACKED_BYTE_ARRAY: {
		} break;
		case Variant::PACKED_INT32_ARRAY: {
		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
		} break;
		case Variant::PACKED_INT64_ARRAY: {
		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
		} break;
		case Variant::PACKED_STRING_ARRAY: {
		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
		} break;
		case Variant::PACKED_COLOR_ARRAY: {
		} break;
		default: {
		}
	}

	updating = false;
	return true;
}

void CustomPropertyEditor::_file_selected(String p_file) {
	switch (type) {
		case Variant::STRING: {
			if (hint == PROPERTY_HINT_FILE || hint == PROPERTY_HINT_DIR) {
				v = ProjectSettings::get_singleton()->localize_path(p_file);
				emit_signal(SNAME("variant_changed"));
				hide();
			}

			if (hint == PROPERTY_HINT_GLOBAL_FILE || hint == PROPERTY_HINT_GLOBAL_DIR) {
				v = p_file;
				emit_signal(SNAME("variant_changed"));
				hide();
			}

		} break;
		case Variant::OBJECT: {
			String type = (hint == PROPERTY_HINT_RESOURCE_TYPE) ? hint_text : String();

			RES res = ResourceLoader::load(p_file, type);
			if (res.is_null()) {
				error->set_text(TTR("Error loading file: Not a resource!"));
				error->popup_centered();
				break;
			}
			v = res;
			emit_signal(SNAME("variant_changed"));
			hide();
		} break;
		default: {
		}
	}
}

void CustomPropertyEditor::_locale_selected(String p_locale) {
	if (type == Variant::STRING && hint == PROPERTY_HINT_LOCALE_ID) {
		v = p_locale;
		emit_signal(SNAME("variant_changed"));
		hide();
	}
}

void CustomPropertyEditor::_type_create_selected(int p_idx) {
	if (type == Variant::INT || type == Variant::FLOAT) {
		float newval = 0;
		switch (p_idx) {
			case EASING_LINEAR: {
				newval = 1;
			} break;
			case EASING_EASE_IN: {
				newval = 2.0;
			} break;
			case EASING_EASE_OUT: {
				newval = 0.5;
			} break;
			case EASING_ZERO: {
				newval = 0;
			} break;
			case EASING_IN_OUT: {
				newval = -0.5;
			} break;
			case EASING_OUT_IN: {
				newval = -2.0;
			} break;
		}

		v = newval;
		emit_signal(SNAME("variant_changed"));
		easing_draw->update();

	} else if (type == Variant::OBJECT) {
		ERR_FAIL_INDEX(p_idx, inheritors_array.size());

		String intype = inheritors_array[p_idx];

		Variant obj = ClassDB::instantiate(intype);

		if (!obj) {
			if (ScriptServer::is_global_class(intype)) {
				obj = EditorNode::get_editor_data().script_class_instance(intype);
			} else {
				obj = EditorNode::get_editor_data().instance_custom_type(intype, "Resource");
			}
		}

		ERR_FAIL_COND(!obj);
		ERR_FAIL_COND(!Object::cast_to<Resource>(obj));

		v = obj;
		emit_signal(SNAME("variant_changed"));
		hide();
	}
}

void CustomPropertyEditor::_color_changed(const Color &p_color) {
	v = p_color;
	emit_signal(SNAME("variant_changed"));
}

void CustomPropertyEditor::_node_path_selected(NodePath p_path) {
	if (picking_viewport) {
		Node *to_node = get_node(p_path);
		if (!Object::cast_to<Viewport>(to_node)) {
			EditorNode::get_singleton()->show_warning(TTR("Selected node is not a Viewport!"));
			return;
		}

		Ref<ViewportTexture> vt;
		vt.instantiate();
		vt->set_viewport_path_in_scene(get_tree()->get_edited_scene_root()->get_path_to(to_node));
		vt->setup_local_to_scene();
		v = vt;
		emit_signal(SNAME("variant_changed"));
		return;
	}

	if (hint == PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE && !hint_text.is_empty()) {
		Node *node = get_node(hint_text);
		if (node) {
			Node *tonode = node->get_node(p_path);
			if (tonode) {
				p_path = node->get_path_to(tonode);
			}
		}

	} else if (owner) {
		Node *node = nullptr;

		if (owner->is_class("Node")) {
			node = Object::cast_to<Node>(owner);
		} else if (owner->is_class("ArrayPropertyEdit")) {
			node = Object::cast_to<ArrayPropertyEdit>(owner)->get_node();
		} else if (owner->is_class("DictionaryPropertyEdit")) {
			node = Object::cast_to<DictionaryPropertyEdit>(owner)->get_node();
		}
		if (!node) {
			v = p_path;
			emit_signal(SNAME("variant_changed"));
			call_deferred(SNAME("hide")); //to not mess with dialogs
			return;
		}

		Node *tonode = node->get_node(p_path);
		if (tonode) {
			p_path = node->get_path_to(tonode);
		}
	}

	v = p_path;
	emit_signal(SNAME("variant_changed"));
	call_deferred(SNAME("hide")); //to not mess with dialogs
}

void CustomPropertyEditor::_action_pressed(int p_which) {
	if (updating) {
		return;
	}

	switch (type) {
		case Variant::BOOL: {
			v = checks20[0]->is_pressed();
			emit_signal(SNAME("variant_changed"));
		} break;
		case Variant::INT: {
			if (hint == PROPERTY_HINT_LAYERS_2D_PHYSICS ||
					hint == PROPERTY_HINT_LAYERS_2D_RENDER ||
					hint == PROPERTY_HINT_LAYERS_2D_NAVIGATION ||
					hint == PROPERTY_HINT_LAYERS_3D_PHYSICS ||
					hint == PROPERTY_HINT_LAYERS_3D_RENDER ||
					hint == PROPERTY_HINT_LAYERS_3D_NAVIGATION) {
				uint32_t f = v;
				if (checks20[p_which]->is_pressed()) {
					f |= (1 << p_which);
				} else {
					f &= ~(1 << p_which);
				}

				v = f;
				emit_signal(SNAME("variant_changed"));
			}

		} break;
		case Variant::STRING: {
			if (hint == PROPERTY_HINT_MULTILINE_TEXT) {
				hide();
			} else if (hint == PROPERTY_HINT_LOCALE_ID) {
				locale->popup_locale_dialog();
			} else if (hint == PROPERTY_HINT_FILE || hint == PROPERTY_HINT_GLOBAL_FILE) {
				if (p_which == 0) {
					if (hint == PROPERTY_HINT_FILE) {
						file->set_access(EditorFileDialog::ACCESS_RESOURCES);
					} else {
						file->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
					}

					file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
					file->clear_filters();

					file->clear_filters();

					if (!hint_text.is_empty()) {
						Vector<String> extensions = hint_text.split(",");
						for (int i = 0; i < extensions.size(); i++) {
							String filter = extensions[i];
							if (filter.begins_with(".")) {
								filter = "*" + extensions[i];
							} else if (!filter.begins_with("*")) {
								filter = "*." + extensions[i];
							}

							file->add_filter(filter + " ; " + extensions[i].to_upper());
						}
					}
					file->popup_file_dialog();
				} else {
					v = "";
					emit_signal(SNAME("variant_changed"));
					hide();
				}

			} else if (hint == PROPERTY_HINT_DIR || hint == PROPERTY_HINT_GLOBAL_DIR) {
				if (p_which == 0) {
					if (hint == PROPERTY_HINT_DIR) {
						file->set_access(EditorFileDialog::ACCESS_RESOURCES);
					} else {
						file->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
					}
					file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
					file->clear_filters();
					file->popup_file_dialog();
				} else {
					v = "";
					emit_signal(SNAME("variant_changed"));
					hide();
				}
			}

		} break;
		case Variant::NODE_PATH: {
			if (p_which == 0) {
				picking_viewport = false;
				scene_tree->set_title(TTR("Pick a Node"));
				scene_tree->popup_scenetree_dialog();

			} else if (p_which == 1) {
				v = NodePath();
				emit_signal(SNAME("variant_changed"));
				hide();
			} else if (p_which == 2) {
				if (owner->is_class("Node") && (v.get_type() == Variant::NODE_PATH) && Object::cast_to<Node>(owner)->has_node(v)) {
					Node *target_node = Object::cast_to<Node>(owner)->get_node(v);
					EditorNode::get_singleton()->get_editor_selection()->clear();
					SceneTreeDock::get_singleton()->set_selected(target_node);
				}

				hide();
			}

		} break;
		case Variant::OBJECT: {
			if (p_which == 0) {
				ERR_FAIL_COND(inheritors_array.is_empty());

				String intype = inheritors_array[0];

				if (hint == PROPERTY_HINT_RESOURCE_TYPE) {
					Variant obj = ClassDB::instantiate(intype);

					if (!obj) {
						if (ScriptServer::is_global_class(intype)) {
							obj = EditorNode::get_editor_data().script_class_instance(intype);
						} else {
							obj = EditorNode::get_editor_data().instance_custom_type(intype, "Resource");
						}
					}

					ERR_BREAK(!obj);
					ERR_BREAK(!Object::cast_to<Resource>(obj));

					v = obj;
					emit_signal(SNAME("variant_changed"));
					hide();
				}
			} else if (p_which == 1) {
				file->set_access(EditorFileDialog::ACCESS_RESOURCES);
				file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
				List<String> extensions;
				String type = (hint == PROPERTY_HINT_RESOURCE_TYPE) ? hint_text : String();

				ResourceLoader::get_recognized_extensions_for_type(type, &extensions);
				file->clear_filters();
				for (const String &E : extensions) {
					file->add_filter("*." + E + " ; " + E.to_upper());
				}

				file->popup_file_dialog();

			} else if (p_which == 2) {
				RES r = v;

				if (!r.is_null()) {
					emit_signal(SNAME("resource_edit_request"));
					hide();
				}

			} else if (p_which == 3) {
				v = Variant();
				emit_signal(SNAME("variant_changed"));
				hide();
			} else if (p_which == 4) {
				Ref<Resource> res_orig = v;
				if (res_orig.is_null()) {
					return;
				}

				List<PropertyInfo> property_list;
				res_orig->get_property_list(&property_list);
				List<Pair<String, Variant>> propvalues;

				for (const PropertyInfo &pi : property_list) {
					Pair<String, Variant> p;
					if (pi.usage & PROPERTY_USAGE_STORAGE) {
						p.first = pi.name;
						p.second = res_orig->get(pi.name);
					}

					propvalues.push_back(p);
				}

				Ref<Resource> res = Ref<Resource>(ClassDB::instantiate(res_orig->get_class()));

				ERR_FAIL_COND(res.is_null());

				for (const Pair<String, Variant> &p : propvalues) {
					res->set(p.first, p.second);
				}

				v = res;
				emit_signal(SNAME("variant_changed"));
				hide();
			}

		} break;

		default: {
		};
	}
}

void CustomPropertyEditor::_drag_easing(const Ref<InputEvent> &p_ev) {
	Ref<InputEventMouseMotion> mm = p_ev;

	if (mm.is_valid() && (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
		float rel = mm->get_relative().x;
		if (rel == 0) {
			return;
		}

		bool flip = hint_text == "attenuation";

		if (flip) {
			rel = -rel;
		}

		float val = v;
		if (val == 0) {
			return;
		}
		bool sg = val < 0;
		val = Math::absf(val);

		val = Math::log(val) / Math::log((float)2.0);
		//logspace
		val += rel * 0.05;

		val = Math::pow(2.0f, val);
		if (sg) {
			val = -val;
		}

		v = val;
		easing_draw->update();
		emit_signal(SNAME("variant_changed"));
	}
}

void CustomPropertyEditor::_draw_easing() {
	RID ci = easing_draw->get_canvas_item();

	Size2 s = easing_draw->get_size();
	Rect2 r(Point2(), s);
	r = r.grow(3);
	easing_draw->get_theme_stylebox(SNAME("normal"), SNAME("LineEdit"))->draw(ci, r);

	int points = 48;

	float prev = 1.0;
	float exp = v;
	bool flip = hint_text == "attenuation";

	Ref<Font> f = easing_draw->get_theme_font(SNAME("font"), SNAME("Label"));
	int font_size = easing_draw->get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	Color color = easing_draw->get_theme_color(SNAME("font_color"), SNAME("Label"));

	for (int i = 1; i <= points; i++) {
		float ifl = i / float(points);
		float iflp = (i - 1) / float(points);

		float h = 1.0 - Math::ease(ifl, exp);

		if (flip) {
			ifl = 1.0 - ifl;
			iflp = 1.0 - iflp;
		}

		RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(iflp * s.width, prev * s.height), Point2(ifl * s.width, h * s.height), color);
		prev = h;
	}

	f->draw_string(ci, Point2(10, 10 + f->get_ascent(font_size)), String::num(exp, 2), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color);
}

void CustomPropertyEditor::_text_edit_changed() {
	v = text_edit->get_text();
	emit_signal(SNAME("variant_changed"));
}

void CustomPropertyEditor::_create_dialog_callback() {
	v = create_dialog->get_selected_type();
	emit_signal(SNAME("variant_changed"));
}

void CustomPropertyEditor::_create_selected_property(const String &p_prop) {
	v = p_prop;
	emit_signal(SNAME("variant_changed"));
}

void CustomPropertyEditor::_modified(String p_string) {
	if (updating) {
		return;
	}

	Variant prev_v = v;

	updating = true;
	switch (type) {
		case Variant::INT: {
			String text = TS->parse_number(value_editor[0]->get_text());
			Ref<Expression> expr;
			expr.instantiate();
			Error err = expr->parse(text);
			if (err != OK) {
				v = value_editor[0]->get_text().to_int();
				return;
			} else {
				v = expr->execute(Array(), nullptr, false);
			}

			if (v != prev_v) {
				emit_signal(SNAME("variant_changed"));
			}
		} break;
		case Variant::FLOAT: {
			if (hint != PROPERTY_HINT_EXP_EASING) {
				String text = TS->parse_number(value_editor[0]->get_text());
				v = _parse_real_expression(text);
				if (v != prev_v) {
					emit_signal(SNAME("variant_changed"));
				}
			}

		} break;
		case Variant::STRING: {
			v = value_editor[0]->get_text();
			emit_signal(SNAME("variant_changed"));
		} break;
		case Variant::VECTOR2: {
			Vector2 vec;
			vec.x = _parse_real_expression(value_editor[0]->get_text());
			vec.y = _parse_real_expression(value_editor[1]->get_text());
			v = vec;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::RECT2: {
			Rect2 r2;

			r2.position.x = _parse_real_expression(value_editor[0]->get_text());
			r2.position.y = _parse_real_expression(value_editor[1]->get_text());
			r2.size.x = _parse_real_expression(value_editor[2]->get_text());
			r2.size.y = _parse_real_expression(value_editor[3]->get_text());
			v = r2;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;

		case Variant::VECTOR3: {
			Vector3 vec;
			vec.x = _parse_real_expression(value_editor[0]->get_text());
			vec.y = _parse_real_expression(value_editor[1]->get_text());
			vec.z = _parse_real_expression(value_editor[2]->get_text());
			v = vec;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::PLANE: {
			Plane pl;
			pl.normal.x = _parse_real_expression(value_editor[0]->get_text());
			pl.normal.y = _parse_real_expression(value_editor[1]->get_text());
			pl.normal.z = _parse_real_expression(value_editor[2]->get_text());
			pl.d = _parse_real_expression(value_editor[3]->get_text());
			v = pl;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::QUATERNION: {
			Quaternion q;
			q.x = _parse_real_expression(value_editor[0]->get_text());
			q.y = _parse_real_expression(value_editor[1]->get_text());
			q.z = _parse_real_expression(value_editor[2]->get_text());
			q.w = _parse_real_expression(value_editor[3]->get_text());
			v = q;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::AABB: {
			Vector3 pos;
			Vector3 size;

			pos.x = _parse_real_expression(value_editor[0]->get_text());
			pos.y = _parse_real_expression(value_editor[1]->get_text());
			pos.z = _parse_real_expression(value_editor[2]->get_text());
			size.x = _parse_real_expression(value_editor[3]->get_text());
			size.y = _parse_real_expression(value_editor[4]->get_text());
			size.z = _parse_real_expression(value_editor[5]->get_text());
			v = AABB(pos, size);
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::TRANSFORM2D: {
			Transform2D m;
			for (int i = 0; i < 6; i++) {
				m.elements[i / 2][i % 2] = _parse_real_expression(value_editor[i]->get_text());
			}

			v = m;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::BASIS: {
			Basis m;
			for (int i = 0; i < 9; i++) {
				m.elements[i / 3][i % 3] = _parse_real_expression(value_editor[i]->get_text());
			}

			v = m;
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::TRANSFORM3D: {
			Basis basis;
			for (int i = 0; i < 9; i++) {
				basis.elements[i / 3][i % 3] = _parse_real_expression(value_editor[(i / 3) * 4 + i % 3]->get_text());
			}

			Vector3 origin;

			origin.x = _parse_real_expression(value_editor[3]->get_text());
			origin.y = _parse_real_expression(value_editor[7]->get_text());
			origin.z = _parse_real_expression(value_editor[11]->get_text());

			v = Transform3D(basis, origin);
			if (v != prev_v) {
				_emit_changed_whole_or_field();
			}

		} break;
		case Variant::COLOR: {
		} break;

		case Variant::NODE_PATH: {
			v = NodePath(value_editor[0]->get_text());
			if (v != prev_v) {
				emit_signal(SNAME("variant_changed"));
			}
		} break;
		case Variant::DICTIONARY: {
		} break;
		case Variant::PACKED_BYTE_ARRAY: {
		} break;
		case Variant::PACKED_INT32_ARRAY: {
		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
		} break;
		case Variant::PACKED_STRING_ARRAY: {
		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
		} break;
		case Variant::PACKED_COLOR_ARRAY: {
		} break;
		default: {
		}
	}

	updating = false;
}

real_t CustomPropertyEditor::_parse_real_expression(String text) {
	Ref<Expression> expr;
	expr.instantiate();
	Error err = expr->parse(text);
	real_t out;
	if (err != OK) {
		out = value_editor[0]->get_text().to_float();
	} else {
		out = expr->execute(Array(), nullptr, false);
	}
	return out;
}

void CustomPropertyEditor::_emit_changed_whole_or_field() {
	if (!Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		emit_signal(SNAME("variant_changed"));
	} else {
		emit_signal(SNAME("variant_field_changed"), field_names[focused_value_editor]);
	}
}

void CustomPropertyEditor::_range_modified(double p_value) {
	v = p_value;
	emit_signal(SNAME("variant_changed"));
}

void CustomPropertyEditor::_focus_enter() {
	switch (type) {
		case Variant::FLOAT:
		case Variant::STRING:
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
		case Variant::PLANE:
		case Variant::QUATERNION:
		case Variant::AABB:
		case Variant::TRANSFORM2D:
		case Variant::BASIS:
		case Variant::TRANSFORM3D: {
			for (int i = 0; i < MAX_VALUE_EDITORS; ++i) {
				if (value_editor[i]->has_focus()) {
					focused_value_editor = i;
					value_editor[i]->select_all();
					break;
				}
			}
		} break;
		default: {
		}
	}
}

void CustomPropertyEditor::_focus_exit() {
	_modified(String());
}

void CustomPropertyEditor::config_action_buttons(const List<String> &p_strings) {
	Ref<StyleBox> sb = action_buttons[0]->get_theme_stylebox(SNAME("button"));
	int margin_top = sb->get_margin(SIDE_TOP);
	int margin_left = sb->get_margin(SIDE_LEFT);
	int margin_bottom = sb->get_margin(SIDE_BOTTOM);
	int margin_right = sb->get_margin(SIDE_RIGHT);

	int max_width = 0;
	int height = 0;

	for (int i = 0; i < MAX_ACTION_BUTTONS; i++) {
		if (i < p_strings.size()) {
			action_buttons[i]->show();
			action_buttons[i]->set_text(p_strings[i]);

			Size2 btn_m_size = action_buttons[i]->get_minimum_size();
			if (btn_m_size.width > max_width) {
				max_width = btn_m_size.width;
			}

		} else {
			action_buttons[i]->hide();
		}
	}

	for (int i = 0; i < p_strings.size(); i++) {
		Size2 btn_m_size = action_buttons[i]->get_size();
		action_buttons[i]->set_position(Point2(0, height) + Point2(margin_left, margin_top));
		action_buttons[i]->set_size(Size2(max_width, btn_m_size.height));

		height += btn_m_size.height;
	}
	set_size(Size2(max_width, height) + Size2(margin_left + margin_right, margin_top + margin_bottom));
}

void CustomPropertyEditor::config_value_editors(int p_amount, int p_columns, int p_label_w, const List<String> &p_strings) {
	int cell_width = 95;
	int cell_height = 25;
	int cell_margin = 5;
	int rows = ((p_amount - 1) / p_columns) + 1;

	set_size(Size2(cell_margin + p_label_w + (cell_width + cell_margin + p_label_w) * p_columns, cell_margin + (cell_height + cell_margin) * rows) * EDSCALE);

	for (int i = 0; i < MAX_VALUE_EDITORS; i++) {
		value_label[i]->get_parent()->remove_child(value_label[i]);
		value_editor[i]->get_parent()->remove_child(value_editor[i]);

		int box_id = i / p_columns;
		value_hboxes[box_id]->add_child(value_label[i]);
		value_hboxes[box_id]->add_child(value_editor[i]);

		if (i < MAX_VALUE_EDITORS / 4) {
			if (i <= p_amount / 4) {
				value_hboxes[i]->show();
			} else {
				value_hboxes[i]->hide();
			}
		}

		if (i < p_amount) {
			value_editor[i]->show();
			value_label[i]->show();
			value_label[i]->set_text(i < p_strings.size() ? p_strings[i] : String(""));
			value_editor[i]->set_editable(!read_only);
		} else {
			value_editor[i]->hide();
			value_label[i]->hide();
		}
	}
}

void CustomPropertyEditor::_bind_methods() {
	ADD_SIGNAL(MethodInfo("variant_changed"));
	ADD_SIGNAL(MethodInfo("variant_field_changed", PropertyInfo(Variant::STRING, "field")));
	ADD_SIGNAL(MethodInfo("resource_edit_request"));
}

CustomPropertyEditor::CustomPropertyEditor() {
	read_only = false;
	updating = false;

	value_vbox = memnew(VBoxContainer);
	add_child(value_vbox);

	for (int i = 0; i < MAX_VALUE_EDITORS; i++) {
		if (i < MAX_VALUE_EDITORS / 4) {
			value_hboxes[i] = memnew(HBoxContainer);
			value_vbox->add_child(value_hboxes[i]);
			value_hboxes[i]->hide();
		}
		int hbox_idx = i / 4;
		value_label[i] = memnew(Label);
		value_hboxes[hbox_idx]->add_child(value_label[i]);
		value_label[i]->hide();
		value_editor[i] = memnew(LineEdit);
		value_hboxes[hbox_idx]->add_child(value_editor[i]);
		value_editor[i]->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		value_editor[i]->hide();
		value_editor[i]->connect("text_submitted", callable_mp(this, &CustomPropertyEditor::_modified));
		value_editor[i]->connect("focus_entered", callable_mp(this, &CustomPropertyEditor::_focus_enter));
		value_editor[i]->connect("focus_exited", callable_mp(this, &CustomPropertyEditor::_focus_exit));
	}
	focused_value_editor = -1;

	for (int i = 0; i < 4; i++) {
		scroll[i] = memnew(HScrollBar);
		scroll[i]->hide();
		scroll[i]->set_min(0);
		scroll[i]->set_max(1.0);
		scroll[i]->set_step(0.01);
		add_child(scroll[i]);
	}

	checks20gc = memnew(GridContainer);
	add_child(checks20gc);
	checks20gc->set_columns(11);

	for (int i = 0; i < 20; i++) {
		if (i == 5 || i == 15) {
			Control *space = memnew(Control);
			space->set_custom_minimum_size(Size2(20, 0) * EDSCALE);
			checks20gc->add_child(space);
		}

		checks20[i] = memnew(CheckBox);
		checks20[i]->set_toggle_mode(true);
		checks20[i]->set_focus_mode(Control::FOCUS_NONE);
		checks20gc->add_child(checks20[i]);
		checks20[i]->hide();
		checks20[i]->connect("pressed", callable_mp(this, &CustomPropertyEditor::_action_pressed), make_binds(i));
		checks20[i]->set_tooltip(vformat(TTR("Bit %d, val %d."), i, 1 << i));
	}

	text_edit = memnew(TextEdit);
	value_vbox->add_child(text_edit);
	text_edit->set_anchors_and_offsets_preset(Control::PRESET_WIDE, Control::PRESET_MODE_MINSIZE, 5);
	text_edit->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	text_edit->set_offset(SIDE_BOTTOM, -30);

	text_edit->hide();
	text_edit->connect("text_changed", callable_mp(this, &CustomPropertyEditor::_text_edit_changed));

	color_picker = nullptr;

	file = memnew(EditorFileDialog);
	value_vbox->add_child(file);
	file->hide();

	file->connect("file_selected", callable_mp(this, &CustomPropertyEditor::_file_selected));
	file->connect("dir_selected", callable_mp(this, &CustomPropertyEditor::_file_selected));

	locale = memnew(EditorLocaleDialog);
	value_vbox->add_child(locale);
	locale->hide();

	locale->connect("locale_selected", callable_mp(this, &CustomPropertyEditor::_locale_selected));

	error = memnew(ConfirmationDialog);
	error->set_title(TTR("Error!"));
	value_vbox->add_child(error);

	scene_tree = memnew(SceneTreeDialog);
	value_vbox->add_child(scene_tree);
	scene_tree->connect("selected", callable_mp(this, &CustomPropertyEditor::_node_path_selected));
	scene_tree->get_scene_tree()->set_show_enabled_subscene(true);

	texture_preview = memnew(TextureRect);
	value_vbox->add_child(texture_preview);
	texture_preview->hide();

	easing_draw = memnew(Control);
	value_vbox->add_child(easing_draw);
	easing_draw->hide();
	easing_draw->connect("draw", callable_mp(this, &CustomPropertyEditor::_draw_easing));
	easing_draw->connect("gui_input", callable_mp(this, &CustomPropertyEditor::_drag_easing));
	easing_draw->set_default_cursor_shape(Control::CURSOR_MOVE);

	type_button = memnew(MenuButton);
	value_vbox->add_child(type_button);
	type_button->hide();
	type_button->get_popup()->connect("id_pressed", callable_mp(this, &CustomPropertyEditor::_type_create_selected));

	menu = memnew(PopupMenu);
	//	menu->set_pass_on_modal_close_click(false);
	value_vbox->add_child(menu);
	menu->connect("id_pressed", callable_mp(this, &CustomPropertyEditor::_menu_option));

	evaluator = nullptr;

	spinbox = memnew(SpinBox);
	value_vbox->add_child(spinbox);
	spinbox->set_anchors_and_offsets_preset(Control::PRESET_WIDE, Control::PRESET_MODE_MINSIZE, 5);
	spinbox->connect("value_changed", callable_mp(this, &CustomPropertyEditor::_range_modified));

	slider = memnew(HSlider);
	value_vbox->add_child(slider);
	slider->set_anchors_and_offsets_preset(Control::PRESET_WIDE, Control::PRESET_MODE_MINSIZE, 5);
	slider->connect("value_changed", callable_mp(this, &CustomPropertyEditor::_range_modified));

	action_hboxes = memnew(HBoxContainer);
	action_hboxes->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	value_vbox->add_child(action_hboxes);
	for (int i = 0; i < MAX_ACTION_BUTTONS; i++) {
		action_buttons[i] = memnew(Button);
		action_buttons[i]->hide();
		action_hboxes->add_child(action_buttons[i]);
		Vector<Variant> binds;
		binds.push_back(i);
		action_buttons[i]->connect("pressed", callable_mp(this, &CustomPropertyEditor::_action_pressed), binds);
	}

	create_dialog = nullptr;
	property_select = nullptr;
}
