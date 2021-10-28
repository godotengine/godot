/*************************************************************************/
/*  visual_shader_editor_plugin.cpp                                      */
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

#include "visual_shader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "editor/editor_log.h"
#include "editor/editor_properties.h"
#include "editor/editor_scale.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/viewport.h"
#include "scene/resources/visual_shader_nodes.h"
#include "servers/visual/shader_types.h"

Control *VisualShaderNodePlugin::create_editor(const Ref<Resource> &p_parent_resource, const Ref<VisualShaderNode> &p_node) {
	if (get_script_instance()) {
		return get_script_instance()->call("create_editor", p_parent_resource, p_node);
	}
	return nullptr;
}

void VisualShaderNodePlugin::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::OBJECT, "create_editor", PropertyInfo(Variant::OBJECT, "parent_resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource"), PropertyInfo(Variant::OBJECT, "for_node", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderNode")));
}

///////////////////

void VisualShaderEditor::edit(VisualShader *p_visual_shader) {
	bool changed = false;
	if (p_visual_shader) {
		if (visual_shader.is_null()) {
			changed = true;
		} else {
			if (visual_shader.ptr() != p_visual_shader) {
				changed = true;
			}
		}
		visual_shader = Ref<VisualShader>(p_visual_shader);
		if (!visual_shader->is_connected("changed", this, "_update_preview")) {
			visual_shader->connect("changed", this, "_update_preview");
		}
		visual_shader->set_graph_offset(graph->get_scroll_ofs() / EDSCALE);
	} else {
		if (visual_shader.is_valid()) {
			if (visual_shader->is_connected("changed", this, "")) {
				visual_shader->disconnect("changed", this, "_update_preview");
			}
		}
		visual_shader.unref();
	}

	if (visual_shader.is_null()) {
		hide();
	} else {
		if (changed) { // to avoid tree collapse
			_clear_buffer();
			_update_options_menu();
			_update_preview();
		}
		_update_graph();
	}
}

void VisualShaderEditor::add_plugin(const Ref<VisualShaderNodePlugin> &p_plugin) {
	if (plugins.find(p_plugin) != -1) {
		return;
	}
	plugins.push_back(p_plugin);
}

void VisualShaderEditor::remove_plugin(const Ref<VisualShaderNodePlugin> &p_plugin) {
	plugins.erase(p_plugin);
}

void VisualShaderEditor::clear_custom_types() {
	for (int i = 0; i < add_options.size(); i++) {
		if (add_options[i].is_custom) {
			add_options.remove(i);
			i--;
		}
	}
}

void VisualShaderEditor::add_custom_type(const String &p_name, const Ref<Script> &p_script, const String &p_description, int p_return_icon_type, const String &p_category, const String &p_subcategory) {
	ERR_FAIL_COND(!p_name.is_valid_identifier());
	ERR_FAIL_COND(!p_script.is_valid());

	for (int i = 0; i < add_options.size(); i++) {
		if (add_options[i].is_custom) {
			if (add_options[i].script == p_script) {
				return;
			}
		}
	}

	AddOption ao;
	ao.name = p_name;
	ao.script = p_script;
	ao.return_type = p_return_icon_type;
	ao.description = p_description;
	ao.category = p_category;
	ao.sub_category = p_subcategory;
	ao.is_custom = true;

	bool begin = false;

	for (int i = 0; i < add_options.size(); i++) {
		if (add_options[i].is_custom) {
			if (add_options[i].category == p_category) {
				if (!begin) {
					begin = true;
				}
			} else {
				if (begin) {
					add_options.insert(i, ao);
					return;
				}
			}
		}
	}
	add_options.push_back(ao);
}

bool VisualShaderEditor::_is_available(int p_mode) {
	int current_mode = edit_type->get_selected();

	if (p_mode != -1) {
		switch (current_mode) {
			case VisualShader::TYPE_VERTEX:
				current_mode = 1;
				break;
			case VisualShader::TYPE_FRAGMENT:
				current_mode = 2;
				break;
			case VisualShader::TYPE_LIGHT:
				current_mode = 4;
				break;
			default:
				break;
		}

		int temp_mode = 0;

		if (p_mode & VisualShader::TYPE_FRAGMENT) {
			temp_mode |= 2;
		}

		if (p_mode & VisualShader::TYPE_LIGHT) {
			temp_mode |= 4;
		}

		if (temp_mode == 0) {
			temp_mode |= 1;
		}

		p_mode = temp_mode;
	}

	return (p_mode == -1 || (p_mode & current_mode) != 0);
}

void VisualShaderEditor::update_custom_nodes() {
	if (members_dialog->is_visible()) {
		return;
	}
	clear_custom_types();
	List<StringName> class_list;
	ScriptServer::get_global_class_list(&class_list);
	Dictionary added;
	for (int i = 0; i < class_list.size(); i++) {
		if (ScriptServer::get_global_class_native_base(class_list[i]) == "VisualShaderNodeCustom") {
			String script_path = ScriptServer::get_global_class_path(class_list[i]);
			Ref<Resource> res = ResourceLoader::load(script_path);
			ERR_FAIL_COND(res.is_null());
			ERR_FAIL_COND(!res->is_class("Script"));
			Ref<Script> script = Ref<Script>(res);

			Ref<VisualShaderNodeCustom> ref;
			ref.instance();
			ref->set_script(script.get_ref_ptr());

			String name;
			if (ref->has_method("_get_name")) {
				name = (String)ref->call("_get_name");
			} else {
				name = "Unnamed";
			}

			String description = "";
			if (ref->has_method("_get_description")) {
				description = (String)ref->call("_get_description");
			}

			int return_icon_type = -1;
			if (ref->has_method("_get_return_icon_type")) {
				return_icon_type = (int)ref->call("_get_return_icon_type");
			}

			String category = "";
			if (ref->has_method("_get_category")) {
				category = (String)ref->call("_get_category");
			}
			if (category == "") {
				category = "Custom";
			}

			String subcategory = "";
			if (ref->has_method("_get_subcategory")) {
				subcategory = (String)ref->call("_get_subcategory");
			}

			Dictionary dict;
			dict["name"] = name;
			dict["script"] = script;
			dict["description"] = description;
			dict["return_icon_type"] = return_icon_type;
			dict["category"] = category;
			dict["subcategory"] = subcategory;

			String key;
			key = category;
			key += "/";
			if (subcategory != "") {
				key += subcategory;
				key += "/";
			}
			key += name;

			added[key] = dict;
		}
	}

	Array keys = added.keys();
	keys.sort();

	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys.get(i);

		const Dictionary &value = (Dictionary)added[key];

		add_custom_type(value["name"], value["script"], value["description"], value["return_icon_type"], value["category"], value["subcategory"]);
	}

	_update_options_menu();
}

String VisualShaderEditor::_get_description(int p_idx) {
	if (add_options[p_idx].highend) {
		return TTR("(GLES3 only)") + " " + add_options[p_idx].description; // TODO: change it to (Vulkan Only) when its ready
	} else {
		return add_options[p_idx].description;
	}
}

void VisualShaderEditor::_update_options_menu() {
	node_desc->set_text("");
	members_dialog->get_ok()->set_disabled(true);

	String prev_category;
	String prev_sub_category;

	members->clear();
	TreeItem *root = members->create_item();
	TreeItem *category = nullptr;
	TreeItem *sub_category = nullptr;

	String filter = node_filter->get_text().strip_edges();
	bool use_filter = !filter.empty();

	Vector<String> categories;
	Vector<String> sub_categories;

	int item_count = 0;
	int item_count2 = 0;
	bool is_first_item = true;

	Color unsupported_color = get_color("error_color", "Editor");
	Color supported_color = get_color("warning_color", "Editor");

	static bool low_driver = ProjectSettings::get_singleton()->get("rendering/quality/driver/driver_name") == "GLES2";

	int current_func = -1;

	if (!visual_shader.is_null()) {
		current_func = visual_shader->get_mode();
	}

	for (int i = 0; i < add_options.size() + 1; i++) {
		if (i == add_options.size()) {
			if (sub_category != nullptr && item_count2 == 0) {
				memdelete(sub_category);
				--item_count;
			}
			if (category != nullptr && item_count == 0) {
				memdelete(category);
			}
			break;
		}

		if (!use_filter || add_options[i].name.findn(filter) != -1) {
			if ((add_options[i].func != current_func && add_options[i].func != -1) || !_is_available(add_options[i].mode)) {
				continue;
			}

			if (prev_category != add_options[i].category) {
				if (category != nullptr && item_count == 0) {
					memdelete(category);
				}

				item_count = 0;
				prev_sub_category = "";
				category = members->create_item(root);
				category->set_text(0, add_options[i].category);
				category->set_selectable(0, false);
				if (!use_filter) {
					category->set_collapsed(true);
				}
			}

			if (add_options[i].sub_category != "") {
				if (prev_sub_category != add_options[i].sub_category) {
					if (category != nullptr) {
						if (sub_category != nullptr && item_count2 == 0) {
							memdelete(sub_category);
							--item_count;
						}
						++item_count;
						item_count2 = 0;
						sub_category = members->create_item(category);
						sub_category->set_text(0, add_options[i].sub_category);
						sub_category->set_selectable(0, false);
						if (!use_filter) {
							sub_category->set_collapsed(true);
						}
					}
				}
			} else {
				sub_category = nullptr;
			}

			TreeItem *p_category = nullptr;

			if (sub_category != nullptr) {
				p_category = sub_category;
				++item_count2;
			} else if (category != nullptr) {
				p_category = category;
				++item_count;
			}

			if (p_category != nullptr) {
				TreeItem *item = members->create_item(p_category);
				if (add_options[i].highend && low_driver) {
					item->set_custom_color(0, unsupported_color);
				} else if (add_options[i].highend) {
					item->set_custom_color(0, supported_color);
				}
				item->set_text(0, add_options[i].name);
				if (is_first_item && use_filter) {
					item->select(0);
					node_desc->set_text(_get_description(i));
					is_first_item = false;
				}
				switch (add_options[i].return_type) {
					case VisualShaderNode::PORT_TYPE_SCALAR:
						item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_icon("float", "EditorIcons"));
						break;
					case VisualShaderNode::PORT_TYPE_VECTOR:
						item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_icon("Vector3", "EditorIcons"));
						break;
					case VisualShaderNode::PORT_TYPE_BOOLEAN:
						item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_icon("bool", "EditorIcons"));
						break;
					case VisualShaderNode::PORT_TYPE_TRANSFORM:
						item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_icon("Transform", "EditorIcons"));
						break;
					case VisualShaderNode::PORT_TYPE_SAMPLER:
						item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_icon("ImageTexture", "EditorIcons"));
						break;
					default:
						break;
				}
				item->set_meta("id", i);
			}

			prev_sub_category = add_options[i].sub_category;
			prev_category = add_options[i].category;
		}
	}
}

Size2 VisualShaderEditor::get_minimum_size() const {
	return Size2(10, 200);
}

void VisualShaderEditor::_draw_color_over_button(Object *obj, Color p_color) {
	Button *button = Object::cast_to<Button>(obj);
	if (!button) {
		return;
	}

	Ref<StyleBox> normal = get_stylebox("normal", "Button");
	button->draw_rect(Rect2(normal->get_offset(), button->get_size() - normal->get_minimum_size()), p_color);
}

static Ref<StyleBoxEmpty> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBoxEmpty> style(memnew(StyleBoxEmpty));
	style->set_default_margin(MARGIN_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(MARGIN_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(MARGIN_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(MARGIN_TOP, p_margin_top * EDSCALE);
	return style;
}

void VisualShaderEditor::_update_created_node(GraphNode *node) {
	if (EditorSettings::get_singleton()->get("interface/theme/use_graph_node_headers")) {
		Ref<StyleBoxFlat> sb = node->get_stylebox("frame", "GraphNode");
		Color c = sb->get_border_color();
		Color mono_color = ((c.r + c.g + c.b) / 3) < 0.7 ? Color(1.0, 1.0, 1.0) : Color(0.0, 0.0, 0.0);
		mono_color.a = 0.85;
		c = mono_color;

		node->add_color_override("title_color", c);
		c.a = 0.7;
		node->add_color_override("close_color", c);
		node->add_color_override("resizer_color", c);
	}
}

void VisualShaderEditor::_update_graph() {
	if (updating) {
		return;
	}

	if (visual_shader.is_null()) {
		return;
	}

	graph->set_scroll_ofs(visual_shader->get_graph_offset() * EDSCALE);

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	graph->clear_connections();
	//erase all nodes
	for (int i = 0; i < graph->get_child_count(); i++) {
		if (Object::cast_to<GraphNode>(graph->get_child(i))) {
			Node *node = graph->get_child(i);
			graph->remove_child(node);
			memdelete(node);
			i--;
		}
	}

	static const Color type_color[5] = {
		Color(0.38, 0.85, 0.96), // scalar
		Color(0.84, 0.49, 0.93), // vector
		Color(0.55, 0.65, 0.94), // boolean
		Color(0.96, 0.66, 0.43), // transform
		Color(1.0, 1.0, 0.0) // sampler
	};

	List<VisualShader::Connection> connections;
	visual_shader->get_node_connections(type, &connections);

	Ref<StyleBoxEmpty> label_style = make_empty_stylebox(2, 1, 2, 1);

	Vector<int> nodes = visual_shader->get_node_list(type);

	VisualShaderNodeUniformRef::clear_uniforms();

	// scan for all uniforms

	for (int t = 0; t < VisualShader::TYPE_MAX; t++) {
		Vector<int> tnodes = visual_shader->get_node_list((VisualShader::Type)t);
		for (int i = 0; i < tnodes.size(); i++) {
			Ref<VisualShaderNode> vsnode = visual_shader->get_node((VisualShader::Type)t, tnodes[i]);
			Ref<VisualShaderNodeUniform> uniform = vsnode;

			if (uniform.is_valid()) {
				Ref<VisualShaderNodeScalarUniform> scalar_uniform = vsnode;
				Ref<VisualShaderNodeVec3Uniform> vec3_uniform = vsnode;
				Ref<VisualShaderNodeColorUniform> color_uniform = vsnode;
				Ref<VisualShaderNodeBooleanUniform> bool_uniform = vsnode;
				Ref<VisualShaderNodeTransformUniform> transform_uniform = vsnode;

				VisualShaderNodeUniformRef::UniformType uniform_type;
				if (scalar_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_SCALAR;
				} else if (bool_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_BOOLEAN;
				} else if (vec3_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_VECTOR;
				} else if (transform_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_TRANSFORM;
				} else if (color_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_COLOR;
				} else {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_SAMPLER;
				}
				VisualShaderNodeUniformRef::add_uniform(uniform->get_uniform_name(), uniform_type);
			}
		}
	}

	Control *offset;

	for (int n_i = 0; n_i < nodes.size(); n_i++) {
		Vector2 position = visual_shader->get_node_position(type, nodes[n_i]);
		Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, nodes[n_i]);

		Ref<VisualShaderNodeGroupBase> group_node = Object::cast_to<VisualShaderNodeGroupBase>(vsnode.ptr());
		bool is_group = !group_node.is_null();
		Size2 size = Size2(0, 0);

		VisualShaderNodeCustom *custom = Object::cast_to<VisualShaderNodeCustom>(vsnode.ptr());
		if (custom) {
			custom->_set_initialized(true);
		}

		Ref<VisualShaderNodeExpression> expression_node = Object::cast_to<VisualShaderNodeExpression>(group_node.ptr());
		bool is_expression = !expression_node.is_null();
		String expression = "";

		GraphNode *node = memnew(GraphNode);

		if (is_group) {
			size = group_node->get_size();

			node->set_resizable(true);
			node->connect("resize_request", this, "_node_resized", varray((int)type, nodes[n_i]));
		}
		if (is_expression) {
			expression = expression_node->get_expression();
		}

		/*if (!vsnode->is_connected("changed", this, "_node_changed")) {
			vsnode->connect("changed", this, "_node_changed", varray(vsnode->get_instance_id()), CONNECT_DEFERRED);
		}*/

		node->set_offset(position);

		node->set_title(vsnode->get_caption());
		node->set_name(itos(nodes[n_i]));

		if (nodes[n_i] >= 2) {
			node->set_show_close_button(true);
			node->connect("close_request", this, "_delete_request", varray(nodes[n_i]), CONNECT_DEFERRED);
		}

		node->connect("dragged", this, "_node_dragged", varray(nodes[n_i]));

		Control *custom_editor = nullptr;
		int port_offset = 0;

		if (is_group) {
			port_offset += 2;
		}

		Ref<VisualShaderNodeUniform> uniform = vsnode;
		if (uniform.is_valid()) {
			graph->add_child(node);
			_update_created_node(node);

			LineEdit *uniform_name = memnew(LineEdit);
			uniform_name->set_text(uniform->get_uniform_name());
			node->add_child(uniform_name);
			uniform_name->connect("text_entered", this, "_line_edit_changed", varray(uniform_name, nodes[n_i]));
			uniform_name->connect("focus_exited", this, "_line_edit_focus_out", varray(uniform_name, nodes[n_i]));

			if (vsnode->get_input_port_count() == 0 && vsnode->get_output_port_count() == 1 && vsnode->get_output_port_name(0) == "") {
				//shortcut
				VisualShaderNode::PortType port_right = vsnode->get_output_port_type(0);
				node->set_slot(0, false, VisualShaderNode::PORT_TYPE_SCALAR, Color(), true, port_right, type_color[port_right]);
				continue;
			}
			port_offset++;
		}

		for (int i = 0; i < plugins.size(); i++) {
			custom_editor = plugins.write[i]->create_editor(visual_shader, vsnode);
			if (custom_editor) {
				break;
			}
		}

		if (custom_editor && vsnode->get_output_port_count() > 0 && vsnode->get_output_port_name(0) == "" && (vsnode->get_input_port_count() == 0 || vsnode->get_input_port_name(0) == "")) {
			//will be embedded in first port
		} else if (custom_editor) {
			port_offset++;
			node->add_child(custom_editor);
			custom_editor = nullptr;
		}

		if (is_group) {
			offset = memnew(Control);
			offset->set_custom_minimum_size(Size2(0, 6 * EDSCALE));
			node->add_child(offset);

			if (group_node->is_editable()) {
				HBoxContainer *hb2 = memnew(HBoxContainer);

				Button *add_input_btn = memnew(Button);
				add_input_btn->set_text(TTR("Add Input"));
				add_input_btn->connect("pressed", this, "_add_input_port", varray(nodes[n_i], group_node->get_free_input_port_id(), VisualShaderNode::PORT_TYPE_VECTOR, "input" + itos(group_node->get_free_input_port_id())), CONNECT_DEFERRED);
				hb2->add_child(add_input_btn);

				hb2->add_spacer();

				Button *add_output_btn = memnew(Button);
				add_output_btn->set_text(TTR("Add Output"));
				add_output_btn->connect("pressed", this, "_add_output_port", varray(nodes[n_i], group_node->get_free_output_port_id(), VisualShaderNode::PORT_TYPE_VECTOR, "output" + itos(group_node->get_free_output_port_id())), CONNECT_DEFERRED);
				hb2->add_child(add_output_btn);

				node->add_child(hb2);
			}
		}

		for (int i = 0; i < MAX(vsnode->get_input_port_count(), vsnode->get_output_port_count()); i++) {
			if (vsnode->is_port_separator(i)) {
				node->add_child(memnew(HSeparator));
				port_offset++;
			}

			bool valid_left = i < vsnode->get_input_port_count();
			VisualShaderNode::PortType port_left = VisualShaderNode::PORT_TYPE_SCALAR;
			bool port_left_used = false;
			String name_left;
			if (valid_left) {
				name_left = vsnode->get_input_port_name(i);
				port_left = vsnode->get_input_port_type(i);
				for (List<VisualShader::Connection>::Element *E = connections.front(); E; E = E->next()) {
					if (E->get().to_node == nodes[n_i] && E->get().to_port == i) {
						port_left_used = true;
					}
				}
			}

			bool valid_right = i < vsnode->get_output_port_count();
			VisualShaderNode::PortType port_right = VisualShaderNode::PORT_TYPE_SCALAR;
			String name_right;
			if (valid_right) {
				name_right = vsnode->get_output_port_name(i);
				port_right = vsnode->get_output_port_type(i);
			}

			HBoxContainer *hb = memnew(HBoxContainer);
			hb->add_constant_override("separation", 7 * EDSCALE);

			Variant default_value;

			if (valid_left && !port_left_used) {
				default_value = vsnode->get_input_port_default_value(i);
			}

			if (default_value.get_type() != Variant::NIL) { // only a label
				Button *button = memnew(Button);
				hb->add_child(button);
				button->connect("pressed", this, "_edit_port_default_input", varray(button, nodes[n_i], i));

				switch (default_value.get_type()) {
					case Variant::COLOR: {
						button->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
						button->connect("draw", this, "_draw_color_over_button", varray(button, default_value));
					} break;
					case Variant::BOOL: {
						button->set_text(((bool)default_value) ? "true" : "false");
					} break;
					case Variant::INT:
					case Variant::REAL: {
						button->set_text(String::num(default_value, 4));
					} break;
					case Variant::VECTOR3: {
						Vector3 v = default_value;
						button->set_text(String::num(v.x, 3) + "," + String::num(v.y, 3) + "," + String::num(v.z, 3));
					} break;
					default: {
					}
				}
			}

			if (i == 0 && custom_editor) {
				hb->add_child(custom_editor);
				custom_editor->set_h_size_flags(SIZE_EXPAND_FILL);
			} else {
				if (valid_left) {
					if (is_group) {
						OptionButton *type_box = memnew(OptionButton);
						hb->add_child(type_box);
						type_box->add_item(TTR("Scalar"));
						type_box->add_item(TTR("Vector"));
						type_box->add_item(TTR("Boolean"));
						type_box->add_item(TTR("Transform"));
						type_box->add_item(TTR("Sampler"));
						type_box->select(group_node->get_input_port_type(i));
						type_box->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
						type_box->connect("item_selected", this, "_change_input_port_type", varray(nodes[n_i], i), CONNECT_DEFERRED);

						LineEdit *name_box = memnew(LineEdit);
						hb->add_child(name_box);
						name_box->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
						name_box->set_h_size_flags(SIZE_EXPAND_FILL);
						name_box->set_text(name_left);
						name_box->connect("text_entered", this, "_change_input_port_name", varray(name_box, nodes[n_i], i));
						name_box->connect("focus_exited", this, "_port_name_focus_out", varray(name_box, nodes[n_i], i, false));

						Button *remove_btn = memnew(Button);
						remove_btn->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Remove", "EditorIcons"));
						remove_btn->set_tooltip(TTR("Remove") + " " + name_left);
						remove_btn->connect("pressed", this, "_remove_input_port", varray(nodes[n_i], i), CONNECT_DEFERRED);
						hb->add_child(remove_btn);
					} else {
						Label *label = memnew(Label);
						label->set_text(name_left);
						label->add_style_override("normal", label_style); //more compact
						hb->add_child(label);

						if (vsnode->get_input_port_default_hint(i) != "" && !port_left_used) {
							Label *hint_label = memnew(Label);
							hint_label->set_text("[" + vsnode->get_input_port_default_hint(i) + "]");
							hint_label->add_color_override("font_color", get_color("font_color_readonly", "TextEdit"));
							hint_label->add_style_override("normal", label_style);
							hb->add_child(hint_label);
						}
					}
				}

				if (!is_group) {
					hb->add_spacer();
				}

				if (valid_right) {
					if (is_group) {
						Button *remove_btn = memnew(Button);
						remove_btn->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Remove", "EditorIcons"));
						remove_btn->set_tooltip(TTR("Remove") + " " + name_left);
						remove_btn->connect("pressed", this, "_remove_output_port", varray(nodes[n_i], i), CONNECT_DEFERRED);
						hb->add_child(remove_btn);

						LineEdit *name_box = memnew(LineEdit);
						hb->add_child(name_box);
						name_box->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
						name_box->set_h_size_flags(SIZE_EXPAND_FILL);
						name_box->set_text(name_right);
						name_box->connect("text_entered", this, "_change_output_port_name", varray(name_box, nodes[n_i], i));
						name_box->connect("focus_exited", this, "_port_name_focus_out", varray(name_box, nodes[n_i], i, true));

						OptionButton *type_box = memnew(OptionButton);
						hb->add_child(type_box);
						type_box->add_item(TTR("Scalar"));
						type_box->add_item(TTR("Vector"));
						type_box->add_item(TTR("Boolean"));
						type_box->add_item(TTR("Transform"));
						type_box->select(group_node->get_output_port_type(i));
						type_box->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
						type_box->connect("item_selected", this, "_change_output_port_type", varray(nodes[n_i], i), CONNECT_DEFERRED);
					} else {
						Label *label = memnew(Label);
						label->set_text(name_right);
						label->add_style_override("normal", label_style); //more compact
						hb->add_child(label);
					}
				}
			}

			if (valid_right && edit_type->get_selected() == VisualShader::TYPE_FRAGMENT && port_right != VisualShaderNode::PORT_TYPE_TRANSFORM && port_right != VisualShaderNode::PORT_TYPE_SAMPLER) {
				TextureButton *preview = memnew(TextureButton);
				preview->set_toggle_mode(true);
				preview->set_normal_texture(get_icon("GuiVisibilityHidden", "EditorIcons"));
				preview->set_pressed_texture(get_icon("GuiVisibilityVisible", "EditorIcons"));
				preview->set_v_size_flags(SIZE_SHRINK_CENTER);

				if (vsnode->get_output_port_for_preview() == i) {
					preview->set_pressed(true);
				}

				preview->connect("pressed", this, "_preview_select_port", varray(nodes[n_i], i), CONNECT_DEFERRED);
				hb->add_child(preview);
			}

			if (is_group) {
				offset = memnew(Control);
				offset->set_custom_minimum_size(Size2(0, 5 * EDSCALE));
				node->add_child(offset);
				port_offset++;
			}

			node->add_child(hb);

			node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], valid_right, port_right, type_color[port_right]);
		}

		if (vsnode->get_output_port_for_preview() >= 0) {
			int port_type = vsnode->get_output_port_type(vsnode->get_output_port_for_preview());

			if (port_type != VisualShaderNode::PORT_TYPE_TRANSFORM && port_type != VisualShaderNode::PORT_TYPE_SAMPLER) {
				offset = memnew(Control);
				offset->set_custom_minimum_size(Size2(0, 5 * EDSCALE));
				node->add_child(offset);

				VisualShaderNodePortPreview *port_preview = memnew(VisualShaderNodePortPreview);
				port_preview->setup(visual_shader, type, nodes[n_i], vsnode->get_output_port_for_preview());
				port_preview->set_h_size_flags(SIZE_SHRINK_CENTER);
				node->add_child(port_preview);
			}
		}

		offset = memnew(Control);
		offset->set_custom_minimum_size(Size2(0, 4 * EDSCALE));
		node->add_child(offset);

		String error = vsnode->get_warning(visual_shader->get_mode(), type);
		if (error != String()) {
			Label *error_label = memnew(Label);
			error_label->add_color_override("font_color", get_color("error_color", "Editor"));
			error_label->set_text(error);
			node->add_child(error_label);
		}

		if (is_expression) {
			TextEdit *expression_box = memnew(TextEdit);
			expression_node->set_control(expression_box, 0);
			node->add_child(expression_box);

			Color background_color = EDITOR_GET("text_editor/highlighting/background_color");
			Color text_color = EDITOR_GET("text_editor/highlighting/text_color");
			Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");
			Color control_flow_keyword_color = EDITOR_GET("text_editor/highlighting/control_flow_keyword_color");
			Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");
			Color symbol_color = EDITOR_GET("text_editor/highlighting/symbol_color");

			expression_box->set_syntax_coloring(true);
			expression_box->add_color_override("background_color", background_color);

			for (List<String>::Element *E = keyword_list.front(); E; E = E->next()) {
				if (ShaderLanguage::is_control_flow_keyword(E->get())) {
					expression_box->add_keyword_color(E->get(), control_flow_keyword_color);
				} else {
					expression_box->add_keyword_color(E->get(), keyword_color);
				}
			}

			expression_box->add_font_override("font", get_font("expression", "EditorFonts"));
			expression_box->add_color_override("font_color", text_color);
			expression_box->add_color_override("symbol_color", symbol_color);
			expression_box->add_color_region("/*", "*/", comment_color, false);
			expression_box->add_color_region("//", "", comment_color, false);

			expression_box->set_text(expression);
			expression_box->set_context_menu_enabled(false);
			expression_box->set_show_line_numbers(true);

			expression_box->connect("focus_exited", this, "_expression_focus_out", varray(expression_box, nodes[n_i]));
		}

		if (!uniform.is_valid()) {
			graph->add_child(node);
			_update_created_node(node);
			if (is_group) {
				call_deferred("_set_node_size", (int)type, nodes[n_i], size);
			}
		}
	}

	for (List<VisualShader::Connection>::Element *E = connections.front(); E; E = E->next()) {
		int from = E->get().from_node;
		int from_idx = E->get().from_port;
		int to = E->get().to_node;
		int to_idx = E->get().to_port;

		graph->connect_node(itos(from), from_idx, itos(to), to_idx);
	}

	float graph_minimap_opacity = EditorSettings::get_singleton()->get("editors/visual_editors/minimap_opacity");
	graph->set_minimap_opacity(graph_minimap_opacity);
}

void VisualShaderEditor::_add_input_port(int p_node, int p_port, int p_port_type, const String &p_name) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeExpression> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Add input port"));
	undo_redo->add_do_method(node.ptr(), "add_input_port", p_port, p_port_type, p_name);
	undo_redo->add_undo_method(node.ptr(), "remove_input_port", p_port);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_add_output_port(int p_node, int p_port, int p_port_type, const String &p_name) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Add output port"));
	undo_redo->add_do_method(node.ptr(), "add_output_port", p_port, p_port_type, p_name);
	undo_redo->add_undo_method(node.ptr(), "remove_output_port", p_port);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_input_port_type(int p_type, int p_node, int p_port) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Change input port type"));
	undo_redo->add_do_method(node.ptr(), "set_input_port_type", p_port, p_type);
	undo_redo->add_undo_method(node.ptr(), "set_input_port_type", p_port, node->get_input_port_type(p_port));
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_output_port_type(int p_type, int p_node, int p_port) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Change output port type"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_type", p_port, p_type);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_type", p_port, node->get_output_port_type(p_port));
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_input_port_name(const String &p_text, Object *line_edit, int p_node_id, int p_port_id) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	undo_redo->create_action(TTR("Change input port name"));
	undo_redo->add_do_method(node.ptr(), "set_input_port_name", p_port_id, p_text);
	undo_redo->add_undo_method(node.ptr(), "set_input_port_name", p_port_id, node->get_input_port_name(p_port_id));
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_output_port_name(const String &p_text, Object *line_edit, int p_node_id, int p_port_id) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	undo_redo->create_action(TTR("Change output port name"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_name", p_port_id, p_text);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_name", p_port_id, node->get_output_port_name(p_port_id));
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_remove_input_port(int p_node, int p_port) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Remove input port"));

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);
	for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
		int from_node = E->get().from_node;
		int from_port = E->get().from_port;
		int to_node = E->get().to_node;
		int to_port = E->get().to_port;

		if (to_node == p_node) {
			if (to_port == p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port);
			} else if (to_port > p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port);

				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port - 1);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port - 1);
			}
		}
	}

	undo_redo->add_do_method(node.ptr(), "remove_input_port", p_port);
	undo_redo->add_undo_method(node.ptr(), "add_input_port", p_port, (int)node->get_input_port_type(p_port), node->get_input_port_name(p_port));

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");

	undo_redo->commit_action();
}

void VisualShaderEditor::_remove_output_port(int p_node, int p_port) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Remove output port"));

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);
	for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
		int from_node = E->get().from_node;
		int from_port = E->get().from_port;
		int to_node = E->get().to_node;
		int to_port = E->get().to_port;

		if (from_node == p_node) {
			if (from_port == p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port);
			} else if (from_port > p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port);

				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port - 1, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port - 1, to_node, to_port);
			}
		}
	}

	undo_redo->add_do_method(node.ptr(), "remove_output_port", p_port);
	undo_redo->add_undo_method(node.ptr(), "add_output_port", p_port, (int)node->get_output_port_type(p_port), node->get_output_port_name(p_port));

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");

	undo_redo->commit_action();
}

void VisualShaderEditor::_expression_focus_out(Object *text_edit, int p_node) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNodeExpression> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	TextEdit *expression_box = Object::cast_to<TextEdit>(text_edit);

	if (node->get_expression() == expression_box->get_text()) {
		return;
	}

	undo_redo->create_action(TTR("Set expression"));
	undo_redo->add_do_method(node.ptr(), "set_expression", expression_box->get_text());
	undo_redo->add_undo_method(node.ptr(), "set_expression", node->get_expression());
	undo_redo->add_do_method(this, "_rebuild");
	undo_redo->add_undo_method(this, "_rebuild");
	undo_redo->commit_action();
}

void VisualShaderEditor::_rebuild() {
	if (visual_shader != nullptr) {
		EditorNode::get_singleton()->get_log()->clear();
		visual_shader->rebuild();
	}
}

void VisualShaderEditor::_set_node_size(int p_type, int p_node, const Vector2 &p_size) {
	VisualShader::Type type = VisualShader::Type(p_type);
	Ref<VisualShaderNode> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	Ref<VisualShaderNodeGroupBase> group_node = Object::cast_to<VisualShaderNodeGroupBase>(node.ptr());

	if (group_node.is_null()) {
		return;
	}

	Vector2 size = p_size;

	group_node->set_size(size);

	GraphNode *gn = nullptr;
	if (edit_type->get_selected() == p_type) { // check - otherwise the error will be emitted
		Node *node2 = graph->get_node(itos(p_node));
		gn = Object::cast_to<GraphNode>(node2);
		if (!gn) {
			return;
		}

		gn->set_custom_minimum_size(size);
		gn->set_size(Size2(1, 1));
	}

	Ref<VisualShaderNodeExpression> expression_node = Object::cast_to<VisualShaderNodeExpression>(node.ptr());
	if (!expression_node.is_null()) {
		Control *text_box = expression_node->get_control(0);
		Size2 box_size = size;
		if (gn != nullptr) {
			if (box_size.x < 150 * EDSCALE || box_size.y < 0) {
				box_size.x = gn->get_size().x;
			}
		}
		box_size.x -= text_box->get_margin(MARGIN_LEFT);
		box_size.x -= 28 * EDSCALE;
		box_size.y -= text_box->get_margin(MARGIN_TOP);
		box_size.y -= 28 * EDSCALE;
		text_box->set_custom_minimum_size(Size2(box_size.x, box_size.y));
		text_box->set_size(Size2(1, 1));
	}
}

void VisualShaderEditor::_node_resized(const Vector2 &p_new_size, int p_type, int p_node) {
	VisualShader::Type type = VisualShader::Type(p_type);
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Resize VisualShader node"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(this, "_set_node_size", p_type, p_node, p_new_size);
	undo_redo->add_undo_method(this, "_set_node_size", p_type, p_node, node->get_size());
	undo_redo->commit_action();
}

void VisualShaderEditor::_preview_select_port(int p_node, int p_port) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNode> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	if (node->get_output_port_for_preview() == p_port) {
		p_port = -1; //toggle it
	}
	undo_redo->create_action(TTR("Set Uniform Name"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", p_port);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", node->get_output_port_for_preview());
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void VisualShaderEditor::_line_edit_changed(const String &p_text, Object *line_edit, int p_node_id) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Ref<VisualShaderNodeUniform> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	String validated_name = visual_shader->validate_uniform_name(p_text, node);

	updating = true;
	undo_redo->create_action(TTR("Set Uniform Name"));
	undo_redo->add_do_method(node.ptr(), "set_uniform_name", validated_name);
	undo_redo->add_undo_method(node.ptr(), "set_uniform_name", node->get_uniform_name());
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;

	Object::cast_to<LineEdit>(line_edit)->set_text(validated_name);
}

void VisualShaderEditor::_line_edit_focus_out(Object *line_edit, int p_node_id) {
	String text = Object::cast_to<LineEdit>(line_edit)->get_text();
	_line_edit_changed(text, line_edit, p_node_id);
}

void VisualShaderEditor::_port_name_focus_out(Object *line_edit, int p_node_id, int p_port_id, bool p_output) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	String text = Object::cast_to<LineEdit>(line_edit)->get_text();

	if (!p_output) {
		if (node->get_input_port_name(p_port_id) == text) {
			return;
		}
	} else {
		if (node->get_output_port_name(p_port_id) == text) {
			return;
		}
	}

	List<String> input_names;
	List<String> output_names;

	for (int i = 0; i < node->get_input_port_count(); i++) {
		if (!p_output && i == p_port_id) {
			continue;
		}
		input_names.push_back(node->get_input_port_name(i));
	}
	for (int i = 0; i < node->get_output_port_count(); i++) {
		if (p_output && i == p_port_id) {
			continue;
		}
		output_names.push_back(node->get_output_port_name(i));
	}

	String validated_name = visual_shader->validate_port_name(text, input_names, output_names);
	if (validated_name == "") {
		if (!p_output) {
			Object::cast_to<LineEdit>(line_edit)->set_text(node->get_input_port_name(p_port_id));
		} else {
			Object::cast_to<LineEdit>(line_edit)->set_text(node->get_output_port_name(p_port_id));
		}
		return;
	}

	if (!p_output) {
		_change_input_port_name(validated_name, line_edit, p_node_id, p_port_id);
	} else {
		_change_output_port_name(validated_name, line_edit, p_node_id, p_port_id);
	}
}

void VisualShaderEditor::_port_edited() {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Variant value = property_editor->get_variant();
	Ref<VisualShaderNode> vsn = visual_shader->get_node(type, editing_node);
	ERR_FAIL_COND(!vsn.is_valid());
	undo_redo->create_action(TTR("Set Input Default Port"));
	Ref<VisualShaderNodeCustom> custom = Object::cast_to<VisualShaderNodeCustom>(vsn.ptr());
	if (custom.is_valid()) {
		undo_redo->add_do_method(custom.ptr(), "_set_input_port_default_value", editing_port, value);
		undo_redo->add_undo_method(custom.ptr(), "_set_input_port_default_value", editing_port, vsn->get_input_port_default_value(editing_port));
	} else {
		undo_redo->add_do_method(vsn.ptr(), "set_input_port_default_value", editing_port, value);
		undo_redo->add_undo_method(vsn.ptr(), "set_input_port_default_value", editing_port, vsn->get_input_port_default_value(editing_port));
	}
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();

	property_editor->hide();
}

void VisualShaderEditor::_edit_port_default_input(Object *p_button, int p_node, int p_port) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Ref<VisualShaderNode> vsn = visual_shader->get_node(type, p_node);

	Button *button = Object::cast_to<Button>(p_button);
	ERR_FAIL_COND(!button);
	Variant value = vsn->get_input_port_default_value(p_port);
	property_editor->set_global_position(button->get_global_position() + Vector2(0, button->get_size().height));
	property_editor->edit(nullptr, "", value.get_type(), value, 0, "");
	property_editor->popup();
	editing_node = p_node;
	editing_port = p_port;
}

void VisualShaderEditor::_add_custom_node(const String &p_path) {
	int idx = -1;

	for (int i = custom_node_option_idx; i < add_options.size(); i++) {
		if (add_options[i].script.is_valid()) {
			if (add_options[i].script->get_path() == p_path) {
				idx = i;
				break;
			}
		}
	}
	if (idx != -1) {
		_add_node(idx);
	}
}

void VisualShaderEditor::_add_texture_node(const String &p_path) {
	VisualShaderNodeTexture *texture = (VisualShaderNodeTexture *)_add_node(texture_node_option_idx, -1);
	texture->set_texture(ResourceLoader::load(p_path));
}

VisualShaderNode *VisualShaderEditor::_add_node(int p_idx, int p_op_idx) {
	ERR_FAIL_INDEX_V(p_idx, add_options.size(), nullptr);

	Ref<VisualShaderNode> vsnode;

	bool is_custom = add_options[p_idx].is_custom;

	if (!is_custom && add_options[p_idx].type != String()) {
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instance(add_options[p_idx].type));
		ERR_FAIL_COND_V(!vsn, nullptr);

		VisualShaderNodeScalarConstant *constant = Object::cast_to<VisualShaderNodeScalarConstant>(vsn);

		if (constant) {
			if ((int)add_options[p_idx].value != -1) {
				constant->set_constant(add_options[p_idx].value);
			}
		}

		if (p_op_idx != -1) {
			VisualShaderNodeInput *input = Object::cast_to<VisualShaderNodeInput>(vsn);

			if (input) {
				input->set_input_name(add_options[p_idx].sub_func_str);
			}

			VisualShaderNodeIs *is = Object::cast_to<VisualShaderNodeIs>(vsn);

			if (is) {
				is->set_function((VisualShaderNodeIs::Function)p_op_idx);
			}

			VisualShaderNodeCompare *cmp = Object::cast_to<VisualShaderNodeCompare>(vsn);

			if (cmp) {
				cmp->set_function((VisualShaderNodeCompare::Function)p_op_idx);
			}

			VisualShaderNodeColorOp *colorOp = Object::cast_to<VisualShaderNodeColorOp>(vsn);

			if (colorOp) {
				colorOp->set_operator((VisualShaderNodeColorOp::Operator)p_op_idx);
			}

			VisualShaderNodeColorFunc *colorFunc = Object::cast_to<VisualShaderNodeColorFunc>(vsn);

			if (colorFunc) {
				colorFunc->set_function((VisualShaderNodeColorFunc::Function)p_op_idx);
			}

			VisualShaderNodeScalarOp *scalarOp = Object::cast_to<VisualShaderNodeScalarOp>(vsn);

			if (scalarOp) {
				scalarOp->set_operator((VisualShaderNodeScalarOp::Operator)p_op_idx);
			}

			VisualShaderNodeScalarFunc *scalarFunc = Object::cast_to<VisualShaderNodeScalarFunc>(vsn);

			if (scalarFunc) {
				scalarFunc->set_function((VisualShaderNodeScalarFunc::Function)p_op_idx);
			}

			VisualShaderNodeVectorOp *vecOp = Object::cast_to<VisualShaderNodeVectorOp>(vsn);

			if (vecOp) {
				vecOp->set_operator((VisualShaderNodeVectorOp::Operator)p_op_idx);
			}

			VisualShaderNodeVectorFunc *vecFunc = Object::cast_to<VisualShaderNodeVectorFunc>(vsn);

			if (vecFunc) {
				vecFunc->set_function((VisualShaderNodeVectorFunc::Function)p_op_idx);
			}

			VisualShaderNodeTransformFunc *matFunc = Object::cast_to<VisualShaderNodeTransformFunc>(vsn);

			if (matFunc) {
				matFunc->set_function((VisualShaderNodeTransformFunc::Function)p_op_idx);
			}

			VisualShaderNodeScalarDerivativeFunc *sderFunc = Object::cast_to<VisualShaderNodeScalarDerivativeFunc>(vsn);

			if (sderFunc) {
				sderFunc->set_function((VisualShaderNodeScalarDerivativeFunc::Function)p_op_idx);
			}

			VisualShaderNodeVectorDerivativeFunc *vderFunc = Object::cast_to<VisualShaderNodeVectorDerivativeFunc>(vsn);

			if (vderFunc) {
				vderFunc->set_function((VisualShaderNodeVectorDerivativeFunc::Function)p_op_idx);
			}
		}

		vsnode = Ref<VisualShaderNode>(vsn);
	} else {
		ERR_FAIL_COND_V(add_options[p_idx].script.is_null(), nullptr);
		String base_type = add_options[p_idx].script->get_instance_base_type();
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instance(base_type));
		ERR_FAIL_COND_V(!vsn, nullptr);
		vsnode = Ref<VisualShaderNode>(vsn);
		vsnode->set_script(add_options[p_idx].script.get_ref_ptr());
	}

	Point2 position = graph->get_scroll_ofs();

	if (saved_node_pos_dirty) {
		position += saved_node_pos;
	} else {
		position += graph->get_size() * 0.5;
		position /= EDSCALE;
	}
	position /= graph->get_zoom();
	saved_node_pos_dirty = false;

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	int id_to_use = visual_shader->get_valid_node_id(type);

	undo_redo->create_action(TTR("Add Node to Visual Shader"));
	undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, vsnode, position, id_to_use);
	undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_to_use);

	VisualShaderNodeExpression *expr = Object::cast_to<VisualShaderNodeExpression>(vsnode.ptr());
	if (expr) {
		undo_redo->add_do_method(expr, "set_size", Size2(250 * EDSCALE, 150 * EDSCALE));
	}

	if (to_node != -1 && to_slot != -1) {
		if (vsnode->get_output_port_count() > 0) {
			int _from_node = id_to_use;
			int _from_slot = 0;

			if (visual_shader->is_port_types_compatible(vsnode->get_output_port_type(_from_slot), visual_shader->get_node(type, to_node)->get_input_port_type(to_slot))) {
				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, _from_node, _from_slot, to_node, to_slot);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, _from_node, _from_slot, to_node, to_slot);
			}
		}
	} else if (from_node != -1 && from_slot != -1) {
		if (vsnode->get_input_port_count() > 0) {
			int _to_node = id_to_use;
			int _to_slot = 0;

			if (visual_shader->is_port_types_compatible(visual_shader->get_node(type, from_node)->get_output_port_type(from_slot), vsnode->get_input_port_type(_to_slot))) {
				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
			}
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	return vsnode.ptr();
}

void VisualShaderEditor::_node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_node) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	drag_buffer.push_back({ type, p_node, p_from, p_to });
	if (!drag_dirty) {
		call_deferred("_nodes_dragged");
	}
	drag_dirty = true;
}

void VisualShaderEditor::_nodes_dragged() {
	drag_dirty = false;

	undo_redo->create_action(TTR("Node(s) Moved"));

	for (List<DragOp>::Element *E = drag_buffer.front(); E; E = E->next()) {
		undo_redo->add_do_method(visual_shader.ptr(), "set_node_position", E->get().type, E->get().node, E->get().to);
		undo_redo->add_undo_method(visual_shader.ptr(), "set_node_position", E->get().type, E->get().node, E->get().from);
	}
	updating = true;
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	drag_buffer.clear();
	undo_redo->commit_action();
	updating = false;
}

void VisualShaderEditor::_connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	int from = p_from.to_int();
	int to = p_to.to_int();

	if (!visual_shader->can_connect_nodes(type, from, p_from_index, to, p_to_index)) {
		return;
	}

	undo_redo->create_action(TTR("Nodes Connected"));

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
		if (E->get().to_node == to && E->get().to_port == p_to_index) {
			undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
		}
	}

	undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void VisualShaderEditor::_disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {
	graph->disconnect_node(p_from, p_from_index, p_to, p_to_index);

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	int from = p_from.to_int();
	int to = p_to.to_int();

	//updating = true; seems graph edit can handle this, no need to protect
	undo_redo->create_action(TTR("Nodes Disconnected"));
	undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	//updating = false;
}

void VisualShaderEditor::_connection_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_position) {
	from_node = p_from.to_int();
	from_slot = p_from_slot;
	_show_members_dialog(true);
}

void VisualShaderEditor::_connection_from_empty(const String &p_to, int p_to_slot, const Vector2 &p_release_position) {
	to_node = p_to.to_int();
	to_slot = p_to_slot;
	_show_members_dialog(true);
}

void VisualShaderEditor::_delete_request(int which) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	Ref<VisualShaderNode> node = Ref<VisualShaderNode>(visual_shader->get_node(type, which));

	undo_redo->create_action(TTR("Delete Node"));
	undo_redo->add_do_method(visual_shader.ptr(), "remove_node", type, which);
	undo_redo->add_undo_method(visual_shader.ptr(), "add_node", type, node, visual_shader->get_node_position(type, which), which);

	undo_redo->add_do_method(this, "_clear_buffer");
	undo_redo->add_undo_method(this, "_clear_buffer");

	// restore size, inputs and outputs if node is group
	VisualShaderNodeGroupBase *group = Object::cast_to<VisualShaderNodeGroupBase>(node.ptr());
	if (group) {
		undo_redo->add_undo_method(group, "set_size", group->get_size());
		undo_redo->add_undo_method(group, "set_inputs", group->get_inputs());
		undo_redo->add_undo_method(group, "set_outputs", group->get_outputs());
	}

	// restore expression text if node is expression
	VisualShaderNodeExpression *expression = Object::cast_to<VisualShaderNodeExpression>(node.ptr());
	if (expression) {
		undo_redo->add_undo_method(expression, "set_expression", expression->get_expression());
	}

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
		if (E->get().from_node == which || E->get().to_node == which) {
			undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void VisualShaderEditor::_node_selected(Object *p_node) {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	GraphNode *gn = Object::cast_to<GraphNode>(p_node);
	ERR_FAIL_COND(!gn);

	int id = String(gn->get_name()).to_int();

	Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, id);
	ERR_FAIL_COND(!vsnode.is_valid());

	//do not rely on this, makes editor more complex
	//EditorNode::get_singleton()->push_item(vsnode.ptr(), "", true);
}

void VisualShaderEditor::_graph_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) {
		_show_members_dialog(true);
	}
}

void VisualShaderEditor::_show_members_dialog(bool at_mouse_pos) {
	if (at_mouse_pos) {
		saved_node_pos_dirty = true;
		saved_node_pos = graph->get_local_mouse_position();

		Point2 gpos = Input::get_singleton()->get_mouse_position();
		members_dialog->popup();
		members_dialog->set_position(gpos);
	} else {
		members_dialog->popup();
		saved_node_pos_dirty = false;
		members_dialog->set_position(graph->get_global_position() + Point2(5 * EDSCALE, 65 * EDSCALE));
	}

	// keep dialog within window bounds
	Size2 window_size = OS::get_singleton()->get_window_size();
	Rect2 dialog_rect = members_dialog->get_global_rect();
	if (dialog_rect.position.y + dialog_rect.size.y > window_size.y) {
		int difference = dialog_rect.position.y + dialog_rect.size.y - window_size.y;
		members_dialog->set_position(members_dialog->get_position() - Point2(0, difference));
	}
	if (dialog_rect.position.x + dialog_rect.size.x > window_size.x) {
		int difference = dialog_rect.position.x + dialog_rect.size.x - window_size.x;
		members_dialog->set_position(members_dialog->get_position() - Point2(difference, 0));
	}

	node_filter->call_deferred("grab_focus"); // still not visible
	node_filter->select_all();
}

void VisualShaderEditor::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> ie = p_ie;
	if (ie.is_valid() && (ie->get_scancode() == KEY_UP || ie->get_scancode() == KEY_DOWN || ie->get_scancode() == KEY_ENTER || ie->get_scancode() == KEY_KP_ENTER)) {
		members->call("_gui_input", ie);
		node_filter->accept_event();
	}
}

void VisualShaderEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		node_filter->set_clear_button_enabled(true);

		// collapse tree by default

		TreeItem *category = members->get_root()->get_children();
		while (category) {
			category->set_collapsed(true);
			TreeItem *sub_category = category->get_children();
			while (sub_category) {
				sub_category->set_collapsed(true);
				sub_category = sub_category->get_next();
			}
			category = category->get_next();
		}
	}

	if (p_what == NOTIFICATION_DRAG_BEGIN) {
		Dictionary dd = get_viewport()->gui_get_drag_data();
		if (members->is_visible_in_tree() && dd.has("id")) {
			members->set_drop_mode_flags(Tree::DROP_MODE_ON_ITEM);
		}
	} else if (p_what == NOTIFICATION_DRAG_END) {
		members->set_drop_mode_flags(0);
	}

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		error_panel->add_style_override("panel", get_stylebox("bg", "Tree"));
		error_label->add_color_override("font_color", get_color("error_color", "Editor"));

		node_filter->set_right_icon(Control::get_icon("Search", "EditorIcons"));

		preview_shader->set_icon(Control::get_icon("Shader", "EditorIcons"));

		{
			Color background_color = EDITOR_GET("text_editor/highlighting/background_color");
			Color text_color = EDITOR_GET("text_editor/highlighting/text_color");
			Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");
			Color control_flow_keyword_color = EDITOR_GET("text_editor/highlighting/control_flow_keyword_color");
			Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");
			Color symbol_color = EDITOR_GET("text_editor/highlighting/symbol_color");

			preview_text->add_color_override("background_color", background_color);

			for (List<String>::Element *E = keyword_list.front(); E; E = E->next()) {
				if (ShaderLanguage::is_control_flow_keyword(E->get())) {
					preview_text->add_keyword_color(E->get(), control_flow_keyword_color);
				} else {
					preview_text->add_keyword_color(E->get(), keyword_color);
				}
			}

			preview_text->add_font_override("font", get_font("expression", "EditorFonts"));
			preview_text->add_color_override("font_color", text_color);
			preview_text->add_color_override("symbol_color", symbol_color);
			preview_text->add_color_region("/*", "*/", comment_color, false);
			preview_text->add_color_region("//", "", comment_color, false);

			error_text->add_font_override("font", get_font("status_source", "EditorFonts"));
			error_text->add_color_override("font_color", get_color("error_color", "Editor"));
		}

		tools->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Tools", "EditorIcons"));

		if (p_what == NOTIFICATION_THEME_CHANGED && is_visible_in_tree()) {
			_update_graph();
		}
	}
}

void VisualShaderEditor::_scroll_changed(const Vector2 &p_scroll) {
	if (updating) {
		return;
	}
	updating = true;
	visual_shader->set_graph_offset(p_scroll / EDSCALE);
	updating = false;
}

void VisualShaderEditor::_node_changed(int p_id) {
	if (updating) {
		return;
	}

	if (is_visible_in_tree()) {
		_update_graph();
	}
}

void VisualShaderEditor::_dup_update_excluded(int p_type, Set<int> &r_excluded) {
	r_excluded.clear();
	VisualShader::Type type = (VisualShader::Type)p_type;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			int id = String(gn->get_name()).to_int();
			Ref<VisualShaderNode> node = visual_shader->get_node(type, id);
			Ref<VisualShaderNodeOutput> output = node;
			if (output.is_valid()) {
				r_excluded.insert(id);
				continue;
			}
			r_excluded.insert(id);
		}
	}
}

void VisualShaderEditor::_dup_copy_nodes(int p_type, List<int> &r_nodes, Set<int> &r_excluded) {
	VisualShader::Type type = (VisualShader::Type)p_type;

	selection_center.x = 0.0f;
	selection_center.y = 0.0f;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			int id = String(gn->get_name()).to_int();
			Ref<VisualShaderNode> node = visual_shader->get_node(type, id);
			Ref<VisualShaderNodeOutput> output = node;
			if (output.is_valid()) { // can't duplicate output
				r_excluded.insert(id);
				continue;
			}
			if (node.is_valid() && gn->is_selected()) {
				Vector2 pos = visual_shader->get_node_position(type, id);
				selection_center += pos;
				r_nodes.push_back(id);
			}
			r_excluded.insert(id);
		}
	}

	selection_center /= (float)r_nodes.size();
}

void VisualShaderEditor::_dup_paste_nodes(int p_type, int p_pasted_type, List<int> &r_nodes, Set<int> &r_excluded, const Vector2 &p_offset, bool p_select) {
	VisualShader::Type type = (VisualShader::Type)p_type;
	VisualShader::Type pasted_type = (VisualShader::Type)p_pasted_type;

	int base_id = visual_shader->get_valid_node_id(type);
	int id_from = base_id;
	Map<int, int> connection_remap;
	Set<int> unsupported_set;

	for (List<int>::Element *E = r_nodes.front(); E; E = E->next()) {
		connection_remap[E->get()] = id_from;
		Ref<VisualShaderNode> node = visual_shader->get_node(pasted_type, E->get());

		bool unsupported = false;
		for (int i = 0; i < add_options.size(); i++) {
			if (add_options[i].type == node->get_class_name()) {
				if (!_is_available(add_options[i].mode)) {
					unsupported = true;
				}
				break;
			}
		}
		if (unsupported) {
			unsupported_set.insert(E->get());
			continue;
		}

		Ref<VisualShaderNode> dupli = node->duplicate();

		undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, dupli, visual_shader->get_node_position(pasted_type, E->get()) + p_offset, id_from);
		undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_from);

		// duplicate size, inputs and outputs if node is group
		Ref<VisualShaderNodeGroupBase> group = Object::cast_to<VisualShaderNodeGroupBase>(node.ptr());
		if (!group.is_null()) {
			undo_redo->add_do_method(dupli.ptr(), "set_size", group->get_size());
			undo_redo->add_do_method(dupli.ptr(), "set_inputs", group->get_inputs());
			undo_redo->add_do_method(dupli.ptr(), "set_outputs", group->get_outputs());
		}
		// duplicate expression text if node is expression
		Ref<VisualShaderNodeExpression> expression = Object::cast_to<VisualShaderNodeExpression>(node.ptr());
		if (!expression.is_null()) {
			undo_redo->add_do_method(dupli.ptr(), "set_expression", expression->get_expression());
		}

		id_from++;
	}

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(pasted_type, &conns);

	for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
		if (unsupported_set.has(E->get().from_node) || unsupported_set.has(E->get().to_node)) {
			continue;
		}
		if (connection_remap.has(E->get().from_node) && connection_remap.has(E->get().to_node)) {
			undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, connection_remap[E->get().from_node], E->get().from_port, connection_remap[E->get().to_node], E->get().to_port);
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();

	if (p_select) {
		// reselect duplicated nodes by excluding the other ones
		for (int i = 0; i < graph->get_child_count(); i++) {
			GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
			if (gn) {
				int id = String(gn->get_name()).to_int();
				if (!r_excluded.has(id)) {
					gn->set_selected(true);
				} else {
					gn->set_selected(false);
				}
			}
		}
	}
}

void VisualShaderEditor::_clear_buffer() {
	copy_nodes_buffer.clear();
	copy_nodes_excluded_buffer.clear();
}

void VisualShaderEditor::_duplicate_nodes() {
	int type = edit_type->get_selected();

	List<int> nodes;
	Set<int> excluded;

	_dup_copy_nodes(type, nodes, excluded);

	if (nodes.empty()) {
		return;
	}

	undo_redo->create_action(TTR("Duplicate Nodes"));

	_dup_paste_nodes(type, type, nodes, excluded, Vector2(10, 10) * EDSCALE, true);
}

void VisualShaderEditor::_copy_nodes() {
	copy_type = edit_type->get_selected();

	_clear_buffer();

	_dup_copy_nodes(copy_type, copy_nodes_buffer, copy_nodes_excluded_buffer);
}

void VisualShaderEditor::_paste_nodes() {
	if (copy_nodes_buffer.empty()) {
		return;
	}

	int type = edit_type->get_selected();

	undo_redo->create_action(TTR("Paste Nodes"));

	float scale = graph->get_zoom();

	_dup_paste_nodes(type, copy_type, copy_nodes_buffer, copy_nodes_excluded_buffer, (graph->get_scroll_ofs() / scale + graph->get_local_mouse_position() / scale - selection_center), false);

	_dup_update_excluded(type, copy_nodes_excluded_buffer); // to prevent selection of previous copies at new paste
}

void VisualShaderEditor::_on_nodes_delete() {
	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	List<int> to_erase;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			if (gn->is_selected() && gn->is_close_button_visible()) {
				to_erase.push_back(gn->get_name().operator String().to_int());
			}
		}
	}

	if (to_erase.empty()) {
		return;
	}

	undo_redo->create_action(TTR("Delete Nodes"));

	for (List<int>::Element *F = to_erase.front(); F; F = F->next()) {
		Ref<VisualShaderNode> node = visual_shader->get_node(type, F->get());

		undo_redo->add_do_method(visual_shader.ptr(), "remove_node", type, F->get());
		undo_redo->add_undo_method(visual_shader.ptr(), "add_node", type, node, visual_shader->get_node_position(type, F->get()), F->get());

		undo_redo->add_do_method(this, "_clear_buffer");
		undo_redo->add_undo_method(this, "_clear_buffer");

		// restore size, inputs and outputs if node is group
		VisualShaderNodeGroupBase *group = Object::cast_to<VisualShaderNodeGroupBase>(node.ptr());
		if (group) {
			undo_redo->add_undo_method(group, "set_size", group->get_size());
			undo_redo->add_undo_method(group, "set_inputs", group->get_inputs());
			undo_redo->add_undo_method(group, "set_outputs", group->get_outputs());
		}

		// restore expression text if node is expression
		VisualShaderNodeExpression *expression = Object::cast_to<VisualShaderNodeExpression>(node.ptr());
		if (expression) {
			undo_redo->add_undo_method(expression, "set_expression", expression->get_expression());
		}
	}

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	List<VisualShader::Connection> used_conns;
	for (List<int>::Element *F = to_erase.front(); F; F = F->next()) {
		for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
			if (E->get().from_node == F->get() || E->get().to_node == F->get()) {
				bool cancel = false;
				for (List<VisualShader::Connection>::Element *R = used_conns.front(); R; R = R->next()) {
					if (R->get().from_node == E->get().from_node && R->get().from_port == E->get().from_port && R->get().to_node == E->get().to_node && R->get().to_port == E->get().to_port) {
						cancel = true; // to avoid ERR_ALREADY_EXISTS warning
						break;
					}
				}
				if (!cancel) {
					undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
					used_conns.push_back(E->get());
				}
			}
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void VisualShaderEditor::_mode_selected(int p_id) {
	_update_options_menu();
	_update_graph();
}

void VisualShaderEditor::_input_select_item(Ref<VisualShaderNodeInput> input, String name) {
	String prev_name = input->get_input_name();

	if (name == prev_name) {
		return;
	}

	bool type_changed = input->get_input_type_by_name(name) != input->get_input_type_by_name(prev_name);

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("Visual Shader Input Type Changed"));

	undo_redo->add_do_method(input.ptr(), "set_input_name", name);
	undo_redo->add_undo_method(input.ptr(), "set_input_name", prev_name);

	if (type_changed) {
		//restore connections if type changed
		VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
		int id = visual_shader->find_node_id(type, input);
		List<VisualShader::Connection> conns;
		visual_shader->get_node_connections(type, &conns);
		for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
			if (E->get().from_node == id) {
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}
		}
	}

	undo_redo->add_do_method(VisualShaderEditor::get_singleton(), "_update_graph");
	undo_redo->add_undo_method(VisualShaderEditor::get_singleton(), "_update_graph");

	undo_redo->commit_action();
}

void VisualShaderEditor::_uniform_select_item(Ref<VisualShaderNodeUniformRef> p_uniform_ref, String p_name) {
	String prev_name = p_uniform_ref->get_uniform_name();

	if (p_name == prev_name) {
		return;
	}

	bool type_changed = p_uniform_ref->get_uniform_type_by_name(p_name) != p_uniform_ref->get_uniform_type_by_name(prev_name);

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("UniformRef Name Changed"));

	undo_redo->add_do_method(p_uniform_ref.ptr(), "set_uniform_name", p_name);
	undo_redo->add_undo_method(p_uniform_ref.ptr(), "set_uniform_name", prev_name);

	if (type_changed) {
		//restore connections if type changed
		VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
		int id = visual_shader->find_node_id(type, p_uniform_ref);
		List<VisualShader::Connection> conns;
		visual_shader->get_node_connections(type, &conns);
		for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
			if (E->get().from_node == id) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}
		}
	}

	undo_redo->add_do_method(VisualShaderEditor::get_singleton(), "_update_graph");
	undo_redo->add_undo_method(VisualShaderEditor::get_singleton(), "_update_graph");

	undo_redo->commit_action();
}

void VisualShaderEditor::_member_filter_changed(const String &p_text) {
	_update_options_menu();
}

void VisualShaderEditor::_member_selected() {
	TreeItem *item = members->get_selected();

	if (item != nullptr && item->has_meta("id")) {
		members_dialog->get_ok()->set_disabled(false);
		node_desc->set_text(_get_description(item->get_meta("id")));
	} else {
		members_dialog->get_ok()->set_disabled(true);
		node_desc->set_text("");
	}
}

void VisualShaderEditor::_member_unselected() {
}

void VisualShaderEditor::_member_create() {
	TreeItem *item = members->get_selected();
	if (item != nullptr && item->has_meta("id")) {
		int idx = members->get_selected()->get_meta("id");
		_add_node(idx, add_options[idx].sub_func);
		members_dialog->hide();
	}
}

void VisualShaderEditor::_member_cancel() {
	to_node = -1;
	to_slot = -1;
	from_node = -1;
	from_slot = -1;
}

void VisualShaderEditor::_tools_menu_option(int p_idx) {
	TreeItem *category = members->get_root()->get_children();

	switch (p_idx) {
		case EXPAND_ALL:

			while (category) {
				category->set_collapsed(false);
				TreeItem *sub_category = category->get_children();
				while (sub_category) {
					sub_category->set_collapsed(false);
					sub_category = sub_category->get_next();
				}
				category = category->get_next();
			}

			break;

		case COLLAPSE_ALL:

			while (category) {
				category->set_collapsed(true);
				TreeItem *sub_category = category->get_children();
				while (sub_category) {
					sub_category->set_collapsed(true);
					sub_category = sub_category->get_next();
				}
				category = category->get_next();
			}

			break;
		default:
			break;
	}
}

Variant VisualShaderEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (p_from == members) {
		TreeItem *it = members->get_item_at_position(p_point);
		if (!it) {
			return Variant();
		}
		if (!it->has_meta("id")) {
			return Variant();
		}

		int id = it->get_meta("id");
		AddOption op = add_options[id];

		Dictionary d;
		d["id"] = id;
		if (op.sub_func == -1) {
			d["sub_func"] = op.sub_func_str;
		} else {
			d["sub_func"] = op.sub_func;
		}

		Label *label = memnew(Label);
		label->set_text(it->get_text(0));
		set_drag_preview(label);
		return d;
	}
	return Variant();
}

bool VisualShaderEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	if (p_from == graph) {
		Dictionary d = p_data;

		if (d.has("id")) {
			return true;
		}
		if (d.has("files")) {
			return true;
		}
	}

	return false;
}

void VisualShaderEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (p_from == graph) {
		Dictionary d = p_data;

		if (d.has("id")) {
			int idx = d["id"];
			saved_node_pos = p_point;
			saved_node_pos_dirty = true;
			_add_node(idx, add_options[idx].sub_func);
		} else if (d.has("files")) {
			if (d["files"].get_type() == Variant::POOL_STRING_ARRAY) {
				int j = 0;
				PoolStringArray arr = d["files"];
				for (int i = 0; i < arr.size(); i++) {
					String type = ResourceLoader::get_resource_type(arr[i]);
					if (type == "GDScript") {
						Ref<Script> script = ResourceLoader::load(arr[i]);
						if (script->get_instance_base_type() == "VisualShaderNodeCustom") {
							saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
							saved_node_pos_dirty = true;
							_add_custom_node(arr[i]);
							j++;
						}
					} else if (ClassDB::get_parent_class(type) == "Texture") {
						saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_texture_node(arr[i]);
						j++;
					}
				}
			}
		}
	}
}

void VisualShaderEditor::_show_preview_text() {
	preview_showed = !preview_showed;
	preview_vbox->set_visible(preview_showed);
	if (preview_showed) {
		if (pending_update_preview) {
			_update_preview();
			pending_update_preview = false;
		}
	}
}

void VisualShaderEditor::_update_preview() {
	if (!preview_showed) {
		pending_update_preview = true;
		return;
	}

	String code = visual_shader->get_code();

	preview_text->set_text(code);

	ShaderLanguage sl;

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(visual_shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(visual_shader->get_mode())), ShaderTypes::get_singleton()->get_types());

	for (int i = 0; i < preview_text->get_line_count(); i++) {
		preview_text->set_line_as_marked(i, false);
	}
	if (err != OK) {
		preview_text->set_line_as_marked(sl.get_error_line() - 1, true);
		error_text->set_visible(true);

		String text = "error(" + itos(sl.get_error_line()) + "): " + sl.get_error_text();
		error_text->set_text(text);
		shader_error = true;
	} else {
		error_text->set_visible(false);
		shader_error = false;
	}
}

void VisualShaderEditor::_bind_methods() {
	ClassDB::bind_method("_rebuild", &VisualShaderEditor::_rebuild);
	ClassDB::bind_method("_update_graph", &VisualShaderEditor::_update_graph);
	ClassDB::bind_method("_update_options_menu", &VisualShaderEditor::_update_options_menu);
	ClassDB::bind_method("_expression_focus_out", &VisualShaderEditor::_expression_focus_out);
	ClassDB::bind_method("_add_node", &VisualShaderEditor::_add_node);
	ClassDB::bind_method("_node_dragged", &VisualShaderEditor::_node_dragged);
	ClassDB::bind_method("_connection_request", &VisualShaderEditor::_connection_request);
	ClassDB::bind_method("_disconnection_request", &VisualShaderEditor::_disconnection_request);
	ClassDB::bind_method("_node_selected", &VisualShaderEditor::_node_selected);
	ClassDB::bind_method("_scroll_changed", &VisualShaderEditor::_scroll_changed);
	ClassDB::bind_method("_delete_request", &VisualShaderEditor::_delete_request);
	ClassDB::bind_method("_on_nodes_delete", &VisualShaderEditor::_on_nodes_delete);
	ClassDB::bind_method("_node_changed", &VisualShaderEditor::_node_changed);
	ClassDB::bind_method("_edit_port_default_input", &VisualShaderEditor::_edit_port_default_input);
	ClassDB::bind_method("_port_edited", &VisualShaderEditor::_port_edited);
	ClassDB::bind_method("_connection_to_empty", &VisualShaderEditor::_connection_to_empty);
	ClassDB::bind_method("_connection_from_empty", &VisualShaderEditor::_connection_from_empty);
	ClassDB::bind_method("_line_edit_focus_out", &VisualShaderEditor::_line_edit_focus_out);
	ClassDB::bind_method("_line_edit_changed", &VisualShaderEditor::_line_edit_changed);
	ClassDB::bind_method("_port_name_focus_out", &VisualShaderEditor::_port_name_focus_out);
	ClassDB::bind_method("_duplicate_nodes", &VisualShaderEditor::_duplicate_nodes);
	ClassDB::bind_method("_copy_nodes", &VisualShaderEditor::_copy_nodes);
	ClassDB::bind_method("_paste_nodes", &VisualShaderEditor::_paste_nodes);
	ClassDB::bind_method("_mode_selected", &VisualShaderEditor::_mode_selected);
	ClassDB::bind_method("_input_select_item", &VisualShaderEditor::_input_select_item);
	ClassDB::bind_method("_uniform_select_item", &VisualShaderEditor::_uniform_select_item);
	ClassDB::bind_method("_preview_select_port", &VisualShaderEditor::_preview_select_port);
	ClassDB::bind_method("_graph_gui_input", &VisualShaderEditor::_graph_gui_input);
	ClassDB::bind_method("_add_input_port", &VisualShaderEditor::_add_input_port);
	ClassDB::bind_method("_change_input_port_type", &VisualShaderEditor::_change_input_port_type);
	ClassDB::bind_method("_change_input_port_name", &VisualShaderEditor::_change_input_port_name);
	ClassDB::bind_method("_remove_input_port", &VisualShaderEditor::_remove_input_port);
	ClassDB::bind_method("_add_output_port", &VisualShaderEditor::_add_output_port);
	ClassDB::bind_method("_change_output_port_type", &VisualShaderEditor::_change_output_port_type);
	ClassDB::bind_method("_change_output_port_name", &VisualShaderEditor::_change_output_port_name);
	ClassDB::bind_method("_remove_output_port", &VisualShaderEditor::_remove_output_port);
	ClassDB::bind_method("_node_resized", &VisualShaderEditor::_node_resized);
	ClassDB::bind_method("_set_node_size", &VisualShaderEditor::_set_node_size);
	ClassDB::bind_method("_clear_buffer", &VisualShaderEditor::_clear_buffer);
	ClassDB::bind_method("_show_preview_text", &VisualShaderEditor::_show_preview_text);
	ClassDB::bind_method("_update_preview", &VisualShaderEditor::_update_preview);
	ClassDB::bind_method("_nodes_dragged", &VisualShaderEditor::_nodes_dragged);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &VisualShaderEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &VisualShaderEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &VisualShaderEditor::drop_data_fw);

	ClassDB::bind_method("_is_available", &VisualShaderEditor::_is_available);
	ClassDB::bind_method("_tools_menu_option", &VisualShaderEditor::_tools_menu_option);
	ClassDB::bind_method("_show_members_dialog", &VisualShaderEditor::_show_members_dialog);
	ClassDB::bind_method("_sbox_input", &VisualShaderEditor::_sbox_input);
	ClassDB::bind_method("_member_filter_changed", &VisualShaderEditor::_member_filter_changed);
	ClassDB::bind_method("_member_selected", &VisualShaderEditor::_member_selected);
	ClassDB::bind_method("_member_unselected", &VisualShaderEditor::_member_unselected);
	ClassDB::bind_method("_member_create", &VisualShaderEditor::_member_create);
	ClassDB::bind_method("_member_cancel", &VisualShaderEditor::_member_cancel);
}

VisualShaderEditor *VisualShaderEditor::singleton = nullptr;

VisualShaderEditor::VisualShaderEditor() {
	singleton = this;
	updating = false;
	saved_node_pos_dirty = false;
	saved_node_pos = Point2(0, 0);
	ShaderLanguage::get_keyword_list(&keyword_list);

	preview_showed = false;
	pending_update_preview = false;
	shader_error = false;

	to_node = -1;
	to_slot = -1;
	from_node = -1;
	from_slot = -1;

	main_box = memnew(HSplitContainer);
	main_box->set_v_size_flags(SIZE_EXPAND_FILL);
	main_box->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(main_box);

	graph = memnew(GraphEdit);
	graph->get_zoom_hbox()->set_h_size_flags(SIZE_EXPAND_FILL);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->set_h_size_flags(SIZE_EXPAND_FILL);
	main_box->add_child(graph);
	graph->set_drag_forwarding(this);
	float graph_minimap_opacity = EditorSettings::get_singleton()->get("editors/visual_editors/minimap_opacity");
	graph->set_minimap_opacity(graph_minimap_opacity);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_TRANSFORM);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SAMPLER);
	//graph->add_valid_left_disconnect_type(0);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->connect("connection_request", this, "_connection_request", varray(), CONNECT_DEFERRED);
	graph->connect("disconnection_request", this, "_disconnection_request", varray(), CONNECT_DEFERRED);
	graph->connect("node_selected", this, "_node_selected");
	graph->connect("scroll_offset_changed", this, "_scroll_changed");
	graph->connect("duplicate_nodes_request", this, "_duplicate_nodes");
	graph->connect("copy_nodes_request", this, "_copy_nodes");
	graph->connect("paste_nodes_request", this, "_paste_nodes");
	graph->connect("delete_nodes_request", this, "_on_nodes_delete");
	graph->connect("gui_input", this, "_graph_gui_input");
	graph->connect("connection_to_empty", this, "_connection_to_empty");
	graph->connect("connection_from_empty", this, "_connection_from_empty");
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShaderNode::PORT_TYPE_TRANSFORM);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SAMPLER, VisualShaderNode::PORT_TYPE_SAMPLER);

	VSeparator *vs = memnew(VSeparator);
	graph->get_zoom_hbox()->add_child(vs);
	graph->get_zoom_hbox()->move_child(vs, 0);

	edit_type = memnew(OptionButton);
	edit_type->add_item(TTR("Vertex"));
	edit_type->add_item(TTR("Fragment"));
	edit_type->add_item(TTR("Light"));
	edit_type->select(1);
	edit_type->connect("item_selected", this, "_mode_selected");
	graph->get_zoom_hbox()->add_child(edit_type);
	graph->get_zoom_hbox()->move_child(edit_type, 0);

	add_node = memnew(ToolButton);
	graph->get_zoom_hbox()->add_child(add_node);
	add_node->set_text(TTR("Add Node..."));
	graph->get_zoom_hbox()->move_child(add_node, 0);
	add_node->connect("pressed", this, "_show_members_dialog", varray(false));

	preview_shader = memnew(ToolButton);
	preview_shader->set_toggle_mode(true);
	preview_shader->set_tooltip(TTR("Show resulted shader code."));
	graph->get_zoom_hbox()->add_child(preview_shader);
	preview_shader->connect("pressed", this, "_show_preview_text");

	///////////////////////////////////////
	// PREVIEW PANEL
	///////////////////////////////////////

	preview_vbox = memnew(VBoxContainer);
	preview_vbox->set_visible(preview_showed);
	main_box->add_child(preview_vbox);
	preview_text = memnew(TextEdit);
	preview_vbox->add_child(preview_text);
	preview_text->set_h_size_flags(SIZE_EXPAND_FILL);
	preview_text->set_v_size_flags(SIZE_EXPAND_FILL);
	preview_text->set_custom_minimum_size(Size2(400 * EDSCALE, 0));
	preview_text->set_syntax_coloring(true);
	preview_text->set_show_line_numbers(true);
	preview_text->set_readonly(true);

	error_text = memnew(Label);
	preview_vbox->add_child(error_text);
	error_text->set_visible(false);

	///////////////////////////////////////
	// SHADER NODES TREE
	///////////////////////////////////////

	VBoxContainer *members_vb = memnew(VBoxContainer);
	members_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *filter_hb = memnew(HBoxContainer);
	members_vb->add_child(filter_hb);

	node_filter = memnew(LineEdit);
	filter_hb->add_child(node_filter);
	node_filter->connect("text_changed", this, "_member_filter_changed");
	node_filter->connect("gui_input", this, "_sbox_input");
	node_filter->set_h_size_flags(SIZE_EXPAND_FILL);
	node_filter->set_placeholder(TTR("Search"));

	tools = memnew(MenuButton);
	filter_hb->add_child(tools);
	tools->set_tooltip(TTR("Options"));
	tools->get_popup()->connect("id_pressed", this, "_tools_menu_option");
	tools->get_popup()->add_item(TTR("Expand All"), EXPAND_ALL);
	tools->get_popup()->add_item(TTR("Collapse All"), COLLAPSE_ALL);

	members = memnew(Tree);
	members_vb->add_child(members);
	members->set_drag_forwarding(this);
	members->set_h_size_flags(SIZE_EXPAND_FILL);
	members->set_v_size_flags(SIZE_EXPAND_FILL);
	members->set_hide_root(true);
	members->set_allow_reselect(true);
	members->set_hide_folding(false);
	members->set_custom_minimum_size(Size2(180 * EDSCALE, 200 * EDSCALE));
	members->connect("item_activated", this, "_member_create");
	members->connect("item_selected", this, "_member_selected");
	members->connect("nothing_selected", this, "_member_unselected");

	Label *desc_label = memnew(Label);
	members_vb->add_child(desc_label);
	desc_label->set_text(TTR("Description:"));

	node_desc = memnew(RichTextLabel);
	members_vb->add_child(node_desc);
	node_desc->set_h_size_flags(SIZE_EXPAND_FILL);
	node_desc->set_v_size_flags(SIZE_FILL);
	node_desc->set_custom_minimum_size(Size2(0, 70 * EDSCALE));

	members_dialog = memnew(ConfirmationDialog);
	members_dialog->set_title(TTR("Create Shader Node"));
	members_dialog->add_child(members_vb);
	members_dialog->get_ok()->set_text(TTR("Create"));
	members_dialog->get_ok()->connect("pressed", this, "_member_create");
	members_dialog->get_ok()->set_disabled(true);
	members_dialog->set_resizable(true);
	members_dialog->set_as_minsize();
	members_dialog->connect("hide", this, "_member_cancel");
	add_child(members_dialog);

	alert = memnew(AcceptDialog);
	alert->set_as_minsize();
	alert->get_label()->set_autowrap(true);
	alert->get_label()->set_align(Label::ALIGN_CENTER);
	alert->get_label()->set_valign(Label::VALIGN_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(400, 60) * EDSCALE);
	add_child(alert);

	///////////////////////////////////////
	// SHADER NODES TREE OPTIONS
	///////////////////////////////////////

	// COLOR

	add_options.push_back(AddOption("ColorFunc", "Color", "Common", "VisualShaderNodeColorFunc", TTR("Color function."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ColorOp", "Color", "Common", "VisualShaderNodeColorOp", TTR("Color operator."), -1, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("Grayscale", "Color", "Functions", "VisualShaderNodeColorFunc", TTR("Grayscale function."), VisualShaderNodeColorFunc::FUNC_GRAYSCALE, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("HSV2RGB", "Color", "Functions", "VisualShaderNodeVectorFunc", TTR("Converts HSV vector to RGB equivalent."), VisualShaderNodeVectorFunc::FUNC_HSV2RGB, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("RGB2HSV", "Color", "Functions", "VisualShaderNodeVectorFunc", TTR("Converts RGB vector to HSV equivalent."), VisualShaderNodeVectorFunc::FUNC_RGB2HSV, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Sepia", "Color", "Functions", "VisualShaderNodeColorFunc", TTR("Sepia function."), VisualShaderNodeColorFunc::FUNC_SEPIA, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("Burn", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Burn operator."), VisualShaderNodeColorOp::OP_BURN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Darken", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Darken operator."), VisualShaderNodeColorOp::OP_DARKEN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Difference", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Difference operator."), VisualShaderNodeColorOp::OP_DIFFERENCE, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Dodge", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Dodge operator."), VisualShaderNodeColorOp::OP_DODGE, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("HardLight", "Color", "Operators", "VisualShaderNodeColorOp", TTR("HardLight operator."), VisualShaderNodeColorOp::OP_HARD_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Lighten", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Lighten operator."), VisualShaderNodeColorOp::OP_LIGHTEN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Overlay", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Overlay operator."), VisualShaderNodeColorOp::OP_OVERLAY, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Screen", "Color", "Operators", "VisualShaderNodeColorOp", TTR("Screen operator."), VisualShaderNodeColorOp::OP_SCREEN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("SoftLight", "Color", "Operators", "VisualShaderNodeColorOp", TTR("SoftLight operator."), VisualShaderNodeColorOp::OP_SOFT_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("ColorConstant", "Color", "Variables", "VisualShaderNodeColorConstant", TTR("Color constant."), -1, -1));
	add_options.push_back(AddOption("ColorUniform", "Color", "Variables", "VisualShaderNodeColorUniform", TTR("Color uniform."), -1, -1));

	// CONDITIONAL

	const String &compare_func_desc = TTR("Returns the boolean result of the %s comparison between two parameters.");

	add_options.push_back(AddOption("Equal", "Conditional", "Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Equal (==)")), VisualShaderNodeCompare::FUNC_EQUAL, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("GreaterThan", "Conditional", "Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Greater Than (>)")), VisualShaderNodeCompare::FUNC_GREATER_THAN, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("GreaterThanEqual", "Conditional", "Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Greater Than or Equal (>=)")), VisualShaderNodeCompare::FUNC_GREATER_THAN_EQUAL, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("If", "Conditional", "Functions", "VisualShaderNodeIf", TTR("Returns an associated vector if the provided scalars are equal, greater or less."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("IsInf", "Conditional", "Functions", "VisualShaderNodeIs", TTR("Returns the boolean result of the comparison between INF and a scalar parameter."), VisualShaderNodeIs::FUNC_IS_INF, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("IsNaN", "Conditional", "Functions", "VisualShaderNodeIs", TTR("Returns the boolean result of the comparison between NaN and a scalar parameter."), VisualShaderNodeIs::FUNC_IS_NAN, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("LessThan", "Conditional", "Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Less Than (<)")), VisualShaderNodeCompare::FUNC_LESS_THAN, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("LessThanEqual", "Conditional", "Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Less Than or Equal (<=)")), VisualShaderNodeCompare::FUNC_LESS_THAN_EQUAL, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("NotEqual", "Conditional", "Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Not Equal (!=)")), VisualShaderNodeCompare::FUNC_NOT_EQUAL, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("Switch", "Conditional", "Functions", "VisualShaderNodeSwitch", TTR("Returns an associated vector if the provided boolean value is true or false."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("SwitchS", "Conditional", "Functions", "VisualShaderNodeScalarSwitch", TTR("Returns an associated scalar if the provided boolean value is true or false."), -1, VisualShaderNode::PORT_TYPE_SCALAR));

	add_options.push_back(AddOption("Compare", "Conditional", "Common", "VisualShaderNodeCompare", TTR("Returns the boolean result of the comparison between two parameters."), -1, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("Is", "Conditional", "Common", "VisualShaderNodeIs", TTR("Returns the boolean result of the comparison between INF (or NaN) and a scalar parameter."), -1, VisualShaderNode::PORT_TYPE_BOOLEAN));

	add_options.push_back(AddOption("BooleanConstant", "Conditional", "Variables", "VisualShaderNodeBooleanConstant", TTR("Boolean constant."), -1, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("BooleanUniform", "Conditional", "Variables", "VisualShaderNodeBooleanUniform", TTR("Boolean uniform."), -1, VisualShaderNode::PORT_TYPE_BOOLEAN));

	// INPUT

	// SPATIAL-FOR-ALL
	const String input_param_shader_modes = TTR("'%s' input parameter for all shader modes.");
	add_options.push_back(AddOption("Camera", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "camera"), "camera", VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("InvCamera", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "inv_camera"), "inv_camera", VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("InvProjection", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "inv_projection"), "inv_projection", VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Normal", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "normal"), "normal", VisualShaderNode::PORT_TYPE_VECTOR, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("OutputIsSRGB", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "output_is_srgb"), "output_is_srgb", VisualShaderNode::PORT_TYPE_BOOLEAN, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Projection", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "camera"), "projection", VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Time", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewportSize", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "viewport_size"), "viewport_size", VisualShaderNode::PORT_TYPE_VECTOR, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("World", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "world"), "world", VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));

	// CANVASITEM-FOR-ALL

	add_options.push_back(AddOption("Alpha", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Color", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ModulateAlpha", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "modulate_alpha"), "modulate_alpha", VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ModulateColor", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "modulate_color"), "modulate_color", VisualShaderNode::PORT_TYPE_VECTOR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("TexturePixelSize", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "texture_pixel_size"), "texture_pixel_size", VisualShaderNode::PORT_TYPE_VECTOR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Time", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("UV", "Input", "All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "uv"), "uv", VisualShaderNode::PORT_TYPE_VECTOR, -1, Shader::MODE_CANVAS_ITEM));

	/////////////////

	add_options.push_back(AddOption("Input", "Input", "Common", "VisualShaderNodeInput", TTR("Input parameter.")));

	// SPATIAL INPUTS

	const String input_param_for_vertex_and_fragment_shader_modes = TTR("'%s' input parameter for vertex and fragment shader modes.");
	const String input_param_for_fragment_and_light_shader_modes = TTR("'%s' input parameter for fragment and light shader modes.");
	const String input_param_for_fragment_shader_mode = TTR("'%s' input parameter for fragment shader mode.");
	const String input_param_for_light_shader_mode = TTR("'%s' input parameter for light shader mode.");
	const String input_param_for_vertex_shader_mode = TTR("'%s' input parameter for vertex shader mode.");
	const String input_param_for_vertex_and_fragment_shader_mode = TTR("'%s' input parameter for vertex and fragment shader mode.");

	add_options.push_back(AddOption("Alpha", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Binormal", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "binormal"), "binormal", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Color", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("DepthTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "depth_texture"), "depth_texture", VisualShaderNode::PORT_TYPE_SAMPLER, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FragCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FrontFacing", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "front_facing"), "front_facing", VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("PointCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "point_coord"), "point_coord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ScreenTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_texture"), "screen_texture", VisualShaderNode::PORT_TYPE_SAMPLER, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ScreenUV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Side", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "side"), "side", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Tangent", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "tangent"), "tangent", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv"), "uv", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV2", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv2"), "uv2", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Vertex", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "vertex"), "vertex", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("View", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "view"), "view", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_SPATIAL));

	add_options.push_back(AddOption("Albedo", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "albedo"), "albedo", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Attenuation", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "attenuation"), "attenuation", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Diffuse", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "diffuse"), "diffuse", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FragCoord", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Light", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light"), "light", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("LightColor", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_color"), "light_color", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Metallic", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "metallic"), "metallic", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Roughness", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "roughness"), "roughness", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Specular", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "specular"), "specular", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Transmission", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "transmission"), "transmission", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("View", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "view"), "view", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));

	add_options.push_back(AddOption("Alpha", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Binormal", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "binormal"), "binormal", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Color", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ModelView", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "modelview"), "modelview", VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("PointSize", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "point_size"), "point_size", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Tangent", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_mode, "tangent"), "tangent", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv"), "uv", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV2", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv2"), "uv2", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Vertex", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "vertex"), "vertex", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_SPATIAL));

	// CANVASITEM INPUTS

	add_options.push_back(AddOption("FragCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightPass", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "light_pass"), "light_pass", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("NormalTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "normal_texture"), "normal_texture", VisualShaderNode::PORT_TYPE_SAMPLER, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "point_coord"), "point_coord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenPixelSize", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_pixel_size"), "screen_pixel_size", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_texture"), "screen_texture", VisualShaderNode::PORT_TYPE_SAMPLER, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenUV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Texture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "texture"), "texture", VisualShaderNode::PORT_TYPE_SAMPLER, VisualShader::TYPE_FRAGMENT, Shader::MODE_CANVAS_ITEM));

	add_options.push_back(AddOption("FragCoord", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightAlpha", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_alpha"), "light_alpha", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightColor", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_color"), "light_color", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightHeight", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_height"), "light_height", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightUV", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_uv"), "light_uv", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightVector", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_vec"), "light_vec", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Normal", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "normal"), "normal", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointCoord", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "point_coord"), "point_coord", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenUV", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ShadowAlpha", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow_alpha"), "shadow_alpha", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ShadowColor", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow_color"), "shadow_color", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ShadowVec", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow_vec"), "shadow_vec", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Texture", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "texture"), "texture", VisualShaderNode::PORT_TYPE_SAMPLER, VisualShader::TYPE_LIGHT, Shader::MODE_CANVAS_ITEM));

	add_options.push_back(AddOption("Extra", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "extra"), "extra", VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShader::TYPE_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightPass", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "light_pass"), "light_pass", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointSize", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "point_size"), "point_size", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Projection", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "projection"), "projection", VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShader::TYPE_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Vertex", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "vertex"), "vertex", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("World", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "world"), "world", VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShader::TYPE_VERTEX, Shader::MODE_CANVAS_ITEM));

	// PARTICLES INPUTS

	add_options.push_back(AddOption("Active", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "active"), "active", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Alpha", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Color", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Custom", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "custom"), "custom", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("CustomAlpha", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "custom_alpha"), "custom_alpha", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Delta", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "delta"), "delta", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("EmissionTransform", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "emission_transform"), "emission_transform", VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Index", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "index"), "index", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("LifeTime", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "lifetime"), "lifetime", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Restart", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "restart"), "restart", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Time", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Transform", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "transform"), "transform", VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Velocity", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "velocity"), "velocity", VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_VERTEX, Shader::MODE_PARTICLES));

	// SCALAR

	add_options.push_back(AddOption("ScalarFunc", "Scalar", "Common", "VisualShaderNodeScalarFunc", TTR("Scalar function."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ScalarOp", "Scalar", "Common", "VisualShaderNodeScalarOp", TTR("Scalar operator."), -1, VisualShaderNode::PORT_TYPE_SCALAR));

	//CONSTANTS

	add_options.push_back(AddOption("E", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("E constant (2.718282). Represents the base of the natural logarithm."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, Math_E));
	add_options.push_back(AddOption("Epsilon", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Epsilon constant (0.00001). Smallest possible scalar number."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, CMP_EPSILON));
	add_options.push_back(AddOption("Phi", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Phi constant (1.618034). Golden ratio."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, 1.618034f));
	add_options.push_back(AddOption("Pi/4", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Pi/4 constant (0.785398) or 45 degrees."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, Math_PI / 4));
	add_options.push_back(AddOption("Pi/2", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Pi/2 constant (1.570796) or 90 degrees."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, Math_PI / 2));
	add_options.push_back(AddOption("Pi", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Pi constant (3.141593) or 180 degrees."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, Math_PI));
	add_options.push_back(AddOption("Tau", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Tau constant (6.283185) or 360 degrees."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, Math_TAU));
	add_options.push_back(AddOption("Sqrt2", "Scalar", "Constants", "VisualShaderNodeScalarConstant", TTR("Sqrt2 constant (1.414214). Square root of 2."), -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, Math_SQRT2));

	// FUNCTIONS

	add_options.push_back(AddOption("Abs", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the absolute value of the parameter."), VisualShaderNodeScalarFunc::FUNC_ABS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ACos", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the arc-cosine of the parameter."), VisualShaderNodeScalarFunc::FUNC_ACOS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ACosH", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), VisualShaderNodeScalarFunc::FUNC_ACOSH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ASin", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the arc-sine of the parameter."), VisualShaderNodeScalarFunc::FUNC_ASIN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ASinH", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), VisualShaderNodeScalarFunc::FUNC_ASINH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATan", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the arc-tangent of the parameter."), VisualShaderNodeScalarFunc::FUNC_ATAN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATan2", "Scalar", "Functions", "VisualShaderNodeScalarOp", TTR("Returns the arc-tangent of the parameters."), VisualShaderNodeScalarOp::OP_ATAN2, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATanH", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), VisualShaderNodeScalarFunc::FUNC_ATANH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Ceil", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), VisualShaderNodeScalarFunc::FUNC_CEIL, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Clamp", "Scalar", "Functions", "VisualShaderNodeScalarClamp", TTR("Constrains a value to lie between two further values."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Cos", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the cosine of the parameter."), VisualShaderNodeScalarFunc::FUNC_COS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("CosH", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the hyperbolic cosine of the parameter."), VisualShaderNodeScalarFunc::FUNC_COSH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Degrees", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Converts a quantity in radians to degrees."), VisualShaderNodeScalarFunc::FUNC_DEGREES, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Base-e Exponential."), VisualShaderNodeScalarFunc::FUNC_EXP, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp2", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Base-2 Exponential."), VisualShaderNodeScalarFunc::FUNC_EXP2, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Floor", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Finds the nearest integer less than or equal to the parameter."), VisualShaderNodeScalarFunc::FUNC_FLOOR, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Fract", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Computes the fractional part of the argument."), VisualShaderNodeScalarFunc::FUNC_FRAC, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("InverseSqrt", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the inverse of the square root of the parameter."), VisualShaderNodeScalarFunc::FUNC_INVERSE_SQRT, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Natural logarithm."), VisualShaderNodeScalarFunc::FUNC_LOG, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log2", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Base-2 logarithm."), VisualShaderNodeScalarFunc::FUNC_LOG2, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Max", "Scalar", "Functions", "VisualShaderNodeScalarOp", TTR("Returns the greater of two values."), VisualShaderNodeScalarOp::OP_MAX, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Min", "Scalar", "Functions", "VisualShaderNodeScalarOp", TTR("Returns the lesser of two values."), VisualShaderNodeScalarOp::OP_MIN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Mix", "Scalar", "Functions", "VisualShaderNodeScalarInterp", TTR("Linear interpolation between two scalars."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Negate", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the opposite value of the parameter."), VisualShaderNodeScalarFunc::FUNC_NEGATE, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("OneMinus", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("1.0 - scalar"), VisualShaderNodeScalarFunc::FUNC_ONEMINUS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Pow", "Scalar", "Functions", "VisualShaderNodeScalarOp", TTR("Returns the value of the first parameter raised to the power of the second."), VisualShaderNodeScalarOp::OP_POW, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Radians", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Converts a quantity in degrees to radians."), VisualShaderNodeScalarFunc::FUNC_RADIANS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Reciprocal", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("1.0 / scalar"), VisualShaderNodeScalarFunc::FUNC_RECIPROCAL, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Round", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Finds the nearest integer to the parameter."), VisualShaderNodeScalarFunc::FUNC_ROUND, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("RoundEven", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Finds the nearest even integer to the parameter."), VisualShaderNodeScalarFunc::FUNC_ROUNDEVEN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Saturate", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Clamps the value between 0.0 and 1.0."), VisualShaderNodeScalarFunc::FUNC_SATURATE, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sign", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Extracts the sign of the parameter."), VisualShaderNodeScalarFunc::FUNC_SIGN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sin", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the sine of the parameter."), VisualShaderNodeScalarFunc::FUNC_SIN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SinH", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the hyperbolic sine of the parameter."), VisualShaderNodeScalarFunc::FUNC_SINH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sqrt", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the square root of the parameter."), VisualShaderNodeScalarFunc::FUNC_SQRT, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SmoothStep", "Scalar", "Functions", "VisualShaderNodeScalarSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), scalar(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if x is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Step", "Scalar", "Functions", "VisualShaderNodeScalarOp", TTR("Step function( scalar(edge), scalar(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), VisualShaderNodeScalarOp::OP_STEP, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Tan", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the tangent of the parameter."), VisualShaderNodeScalarFunc::FUNC_TAN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("TanH", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Returns the hyperbolic tangent of the parameter."), VisualShaderNodeScalarFunc::FUNC_TANH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Trunc", "Scalar", "Functions", "VisualShaderNodeScalarFunc", TTR("Finds the truncated value of the parameter."), VisualShaderNodeScalarFunc::FUNC_TRUNC, VisualShaderNode::PORT_TYPE_SCALAR));

	add_options.push_back(AddOption("Add", "Scalar", "Operators", "VisualShaderNodeScalarOp", TTR("Adds scalar to scalar."), VisualShaderNodeScalarOp::OP_ADD, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Divide", "Scalar", "Operators", "VisualShaderNodeScalarOp", TTR("Divides scalar by scalar."), VisualShaderNodeScalarOp::OP_DIV, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Multiply", "Scalar", "Operators", "VisualShaderNodeScalarOp", TTR("Multiplies scalar by scalar."), VisualShaderNodeScalarOp::OP_MUL, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Remainder", "Scalar", "Operators", "VisualShaderNodeScalarOp", TTR("Returns the remainder of the two scalars."), VisualShaderNodeScalarOp::OP_MOD, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Subtract", "Scalar", "Operators", "VisualShaderNodeScalarOp", TTR("Subtracts scalar from scalar."), VisualShaderNodeScalarOp::OP_SUB, VisualShaderNode::PORT_TYPE_SCALAR));

	add_options.push_back(AddOption("ScalarConstant", "Scalar", "Variables", "VisualShaderNodeScalarConstant", TTR("Scalar constant."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ScalarUniform", "Scalar", "Variables", "VisualShaderNodeScalarUniform", TTR("Scalar uniform."), -1, VisualShaderNode::PORT_TYPE_SCALAR));

	// TEXTURES

	add_options.push_back(AddOption("CubeMap", "Textures", "Functions", "VisualShaderNodeCubeMap", TTR("Perform the cubic texture lookup."), -1, -1));
	texture_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture", "Textures", "Functions", "VisualShaderNodeTexture", TTR("Perform the texture lookup."), -1, -1));

	add_options.push_back(AddOption("CubeMapUniform", "Textures", "Variables", "VisualShaderNodeCubeMapUniform", TTR("Cubic texture uniform lookup."), -1, -1));
	add_options.push_back(AddOption("TextureUniform", "Textures", "Variables", "VisualShaderNodeTextureUniform", TTR("2D texture uniform lookup."), -1, -1));
	add_options.push_back(AddOption("TextureUniformTriplanar", "Textures", "Variables", "VisualShaderNodeTextureUniformTriplanar", TTR("2D texture uniform lookup with triplanar."), -1, -1, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, Shader::MODE_SPATIAL));

	// TRANSFORM

	add_options.push_back(AddOption("TransformFunc", "Transform", "Common", "VisualShaderNodeTransformFunc", TTR("Transform function."), -1, VisualShaderNode::PORT_TYPE_TRANSFORM));

	add_options.push_back(AddOption("OuterProduct", "Transform", "Composition", "VisualShaderNodeOuterProduct", TTR("Calculate the outer product of a pair of vectors.\n\nOuterProduct treats the first parameter 'c' as a column vector (matrix with one column) and the second parameter 'r' as a row vector (matrix with one row) and does a linear algebraic matrix multiply 'c * r', yielding a matrix whose number of rows is the number of components in 'c' and whose number of columns is the number of components in 'r'."), -1, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformCompose", "Transform", "Composition", "VisualShaderNodeTransformCompose", TTR("Composes transform from four vectors."), -1, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformDecompose", "Transform", "Composition", "VisualShaderNodeTransformDecompose", TTR("Decomposes transform to four vectors.")));

	add_options.push_back(AddOption("Determinant", "Transform", "Functions", "VisualShaderNodeDeterminant", TTR("Calculates the determinant of a transform."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Inverse", "Transform", "Functions", "VisualShaderNodeTransformFunc", TTR("Calculates the inverse of a transform."), VisualShaderNodeTransformFunc::FUNC_INVERSE, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("Transpose", "Transform", "Functions", "VisualShaderNodeTransformFunc", TTR("Calculates the transpose of a transform."), VisualShaderNodeTransformFunc::FUNC_TRANSPOSE, VisualShaderNode::PORT_TYPE_TRANSFORM));

	add_options.push_back(AddOption("TransformMult", "Transform", "Operators", "VisualShaderNodeTransformMult", TTR("Multiplies transform by transform."), -1, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformVectorMult", "Transform", "Operators", "VisualShaderNodeTransformVecMult", TTR("Multiplies vector by transform."), -1, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("TransformConstant", "Transform", "Variables", "VisualShaderNodeTransformConstant", TTR("Transform constant."), -1, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformUniform", "Transform", "Variables", "VisualShaderNodeTransformUniform", TTR("Transform uniform."), -1, VisualShaderNode::PORT_TYPE_TRANSFORM));

	// VECTOR

	add_options.push_back(AddOption("VectorFunc", "Vector", "Common", "VisualShaderNodeVectorFunc", TTR("Vector function."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("VectorOp", "Vector", "Common", "VisualShaderNodeVectorOp", TTR("Vector operator."), -1, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("VectorCompose", "Vector", "Composition", "VisualShaderNodeVectorCompose", TTR("Composes vector from three scalars."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("VectorDecompose", "Vector", "Composition", "VisualShaderNodeVectorDecompose", TTR("Decomposes vector to three scalars.")));

	add_options.push_back(AddOption("Abs", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the absolute value of the parameter."), VisualShaderNodeVectorFunc::FUNC_ABS, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ACos", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-cosine of the parameter."), VisualShaderNodeVectorFunc::FUNC_ACOS, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ACosH", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), VisualShaderNodeVectorFunc::FUNC_ACOSH, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ASin", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-sine of the parameter."), VisualShaderNodeVectorFunc::FUNC_ASIN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ASinH", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), VisualShaderNodeVectorFunc::FUNC_ASINH, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ATan", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-tangent of the parameter."), VisualShaderNodeVectorFunc::FUNC_ATAN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ATan2", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Returns the arc-tangent of the parameters."), VisualShaderNodeVectorOp::OP_ATAN2, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("ATanH", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), VisualShaderNodeVectorFunc::FUNC_ATANH, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Ceil", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), VisualShaderNodeVectorFunc::FUNC_CEIL, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Clamp", "Vector", "Functions", "VisualShaderNodeVectorClamp", TTR("Constrains a value to lie between two further values."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Cos", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the cosine of the parameter."), VisualShaderNodeVectorFunc::FUNC_COS, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("CosH", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic cosine of the parameter."), VisualShaderNodeVectorFunc::FUNC_COSH, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Cross", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Calculates the cross product of two vectors."), VisualShaderNodeVectorOp::OP_CROSS, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Degrees", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in radians to degrees."), VisualShaderNodeVectorFunc::FUNC_DEGREES, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Distance", "Vector", "Functions", "VisualShaderNodeVectorDistance", TTR("Returns the distance between two points."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Dot", "Vector", "Functions", "VisualShaderNodeDotProduct", TTR("Calculates the dot product of two vectors."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Base-e Exponential."), VisualShaderNodeVectorFunc::FUNC_EXP, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Exp2", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 Exponential."), VisualShaderNodeVectorFunc::FUNC_EXP2, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("FaceForward", "Vector", "Functions", "VisualShaderNodeFaceForward", TTR("Returns the vector that points in the same direction as a reference vector. The function has three vector parameters : N, the vector to orient, I, the incident vector, and Nref, the reference vector. If the dot product of I and Nref is smaller than zero the return value is N. Otherwise -N is returned."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Floor", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer less than or equal to the parameter."), VisualShaderNodeVectorFunc::FUNC_FLOOR, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Fract", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Computes the fractional part of the argument."), VisualShaderNodeVectorFunc::FUNC_FRAC, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("InverseSqrt", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse of the square root of the parameter."), VisualShaderNodeVectorFunc::FUNC_INVERSE_SQRT, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Length", "Vector", "Functions", "VisualShaderNodeVectorLen", TTR("Calculates the length of a vector."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Natural logarithm."), VisualShaderNodeVectorFunc::FUNC_LOG, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Log2", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 logarithm."), VisualShaderNodeVectorFunc::FUNC_LOG2, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Max", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Returns the greater of two values."), VisualShaderNodeVectorOp::OP_MAX, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Min", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Returns the lesser of two values."), VisualShaderNodeVectorOp::OP_MIN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Mix", "Vector", "Functions", "VisualShaderNodeVectorInterp", TTR("Linear interpolation between two vectors."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("MixS", "Vector", "Functions", "VisualShaderNodeVectorScalarMix", TTR("Linear interpolation between two vectors using scalar."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Negate", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the opposite value of the parameter."), VisualShaderNodeVectorFunc::FUNC_NEGATE, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Normalize", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Calculates the normalize product of vector."), VisualShaderNodeVectorFunc::FUNC_NORMALIZE, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("OneMinus", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("1.0 - vector"), VisualShaderNodeVectorFunc::FUNC_ONEMINUS, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Pow", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Returns the value of the first parameter raised to the power of the second."), VisualShaderNodeVectorOp::OP_POW, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Radians", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in degrees to radians."), VisualShaderNodeVectorFunc::FUNC_RADIANS, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Reciprocal", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("1.0 / vector"), VisualShaderNodeVectorFunc::FUNC_RECIPROCAL, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Reflect", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Returns the vector that points in the direction of reflection ( a : incident vector, b : normal vector )."), VisualShaderNodeVectorOp::OP_REFLECT, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Refract", "Vector", "Functions", "VisualShaderNodeVectorRefract", TTR("Returns the vector that points in the direction of refraction."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Round", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer to the parameter."), VisualShaderNodeVectorFunc::FUNC_ROUND, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("RoundEven", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest even integer to the parameter."), VisualShaderNodeVectorFunc::FUNC_ROUNDEVEN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Saturate", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Clamps the value between 0.0 and 1.0."), VisualShaderNodeVectorFunc::FUNC_SATURATE, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Sign", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Extracts the sign of the parameter."), VisualShaderNodeVectorFunc::FUNC_SIGN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Sin", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the sine of the parameter."), VisualShaderNodeVectorFunc::FUNC_SIN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("SinH", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic sine of the parameter."), VisualShaderNodeVectorFunc::FUNC_SINH, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Sqrt", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the square root of the parameter."), VisualShaderNodeVectorFunc::FUNC_SQRT, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("SmoothStep", "Vector", "Functions", "VisualShaderNodeVectorSmoothStep", TTR("SmoothStep function( vector(edge0), vector(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("SmoothStepS", "Vector", "Functions", "VisualShaderNodeVectorScalarSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Step", "Vector", "Functions", "VisualShaderNodeVectorOp", TTR("Step function( vector(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), VisualShaderNodeVectorOp::OP_STEP, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("StepS", "Vector", "Functions", "VisualShaderNodeVectorScalarStep", TTR("Step function( scalar(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Tan", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the tangent of the parameter."), VisualShaderNodeVectorFunc::FUNC_TAN, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("TanH", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic tangent of the parameter."), VisualShaderNodeVectorFunc::FUNC_TANH, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Trunc", "Vector", "Functions", "VisualShaderNodeVectorFunc", TTR("Finds the truncated value of the parameter."), VisualShaderNodeVectorFunc::FUNC_TRUNC, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("Add", "Vector", "Operators", "VisualShaderNodeVectorOp", TTR("Adds vector to vector."), VisualShaderNodeVectorOp::OP_ADD, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Divide", "Vector", "Operators", "VisualShaderNodeVectorOp", TTR("Divides vector by vector."), VisualShaderNodeVectorOp::OP_DIV, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Multiply", "Vector", "Operators", "VisualShaderNodeVectorOp", TTR("Multiplies vector by vector."), VisualShaderNodeVectorOp::OP_MUL, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Remainder", "Vector", "Operators", "VisualShaderNodeVectorOp", TTR("Returns the remainder of the two vectors."), VisualShaderNodeVectorOp::OP_MOD, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("Subtract", "Vector", "Operators", "VisualShaderNodeVectorOp", TTR("Subtracts vector from vector."), VisualShaderNodeVectorOp::OP_SUB, VisualShaderNode::PORT_TYPE_VECTOR));

	add_options.push_back(AddOption("VectorConstant", "Vector", "Variables", "VisualShaderNodeVec3Constant", TTR("Vector constant."), -1, VisualShaderNode::PORT_TYPE_VECTOR));
	add_options.push_back(AddOption("VectorUniform", "Vector", "Variables", "VisualShaderNodeVec3Uniform", TTR("Vector uniform."), -1, VisualShaderNode::PORT_TYPE_VECTOR));

	// SPECIAL

	add_options.push_back(AddOption("Expression", "Special", "", "VisualShaderNodeExpression", TTR("Custom Godot Shader Language expression, with custom amount of input and output ports. This is a direct injection of code into the vertex/fragment/light function, do not use it to write the function declarations inside.")));
	add_options.push_back(AddOption("Fresnel", "Special", "", "VisualShaderNodeFresnel", TTR("Returns falloff based on the dot product of surface normal and view direction of camera (pass associated inputs to it)."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("GlobalExpression", "Special", "", "VisualShaderNodeGlobalExpression", TTR("Custom Godot Shader Language expression, which is placed on top of the resulted shader. You can place various function definitions inside and call it later in the Expressions. You can also declare varyings, uniforms and constants.")));
	add_options.push_back(AddOption("UniformRef", "Special", "", "VisualShaderNodeUniformRef", TTR("A reference to an existing uniform.")));

	add_options.push_back(AddOption("ScalarDerivativeFunc", "Special", "Common", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) Scalar derivative function."), -1, VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("VectorDerivativeFunc", "Special", "Common", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) Vector derivative function."), -1, VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));

	add_options.push_back(AddOption("DdX", "Special", "Derivative", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'x' using local differencing."), VisualShaderNodeVectorDerivativeFunc::FUNC_X, VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("DdXS", "Special", "Derivative", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Derivative in 'x' using local differencing."), VisualShaderNodeScalarDerivativeFunc::FUNC_X, VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("DdY", "Special", "Derivative", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'y' using local differencing."), VisualShaderNodeVectorDerivativeFunc::FUNC_Y, VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("DdYS", "Special", "Derivative", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Derivative in 'y' using local differencing."), VisualShaderNodeScalarDerivativeFunc::FUNC_Y, VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("Sum", "Special", "Derivative", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Sum of absolute derivative in 'x' and 'y'."), VisualShaderNodeVectorDerivativeFunc::FUNC_SUM, VisualShaderNode::PORT_TYPE_VECTOR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("SumS", "Special", "Derivative", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Sum of absolute derivative in 'x' and 'y'."), VisualShaderNodeScalarDerivativeFunc::FUNC_SUM, VisualShaderNode::PORT_TYPE_SCALAR, VisualShader::TYPE_FRAGMENT | VisualShader::TYPE_LIGHT, -1, -1, true));
	custom_node_option_idx = add_options.size();

	/////////////////////////////////////////////////////////////////////

	_update_options_menu();

	error_panel = memnew(PanelContainer);
	add_child(error_panel);
	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_label->set_text("eh");
	error_panel->hide();

	undo_redo = EditorNode::get_singleton()->get_undo_redo();

	Ref<VisualShaderNodePluginDefault> default_plugin;
	default_plugin.instance();
	add_plugin(default_plugin);

	property_editor = memnew(CustomPropertyEditor);
	add_child(property_editor);

	property_editor->connect("variant_changed", this, "_port_edited");
}

void VisualShaderEditorPlugin::edit(Object *p_object) {
	visual_shader_editor->edit(Object::cast_to<VisualShader>(p_object));
}

bool VisualShaderEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("VisualShader");
}

void VisualShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		//editor->hide_animation_player_editors();
		//editor->animation_panel_make_visible(true);
		button->show();
		editor->make_bottom_panel_item_visible(visual_shader_editor);
		visual_shader_editor->update_custom_nodes();
		visual_shader_editor->set_process_input(true);
		//visual_shader_editor->set_process(true);
	} else {
		if (visual_shader_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
		button->hide();
		visual_shader_editor->set_process_input(false);
		//visual_shader_editor->set_process(false);
	}
}

VisualShaderEditorPlugin::VisualShaderEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	visual_shader_editor = memnew(VisualShaderEditor);
	visual_shader_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);

	button = editor->add_bottom_panel_item(TTR("VisualShader"), visual_shader_editor);
	button->hide();
}

VisualShaderEditorPlugin::~VisualShaderEditorPlugin() {
}

////////////////

class VisualShaderNodePluginInputEditor : public OptionButton {
	GDCLASS(VisualShaderNodePluginInputEditor, OptionButton);

	Ref<VisualShaderNodeInput> input;

protected:
	static void _bind_methods() {
		ClassDB::bind_method("_item_selected", &VisualShaderNodePluginInputEditor::_item_selected);
	}

public:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			connect("item_selected", this, "_item_selected");
		}
	}

	void _item_selected(int p_item) {
		VisualShaderEditor::get_singleton()->call_deferred("_input_select_item", input, get_item_text(p_item));
	}

	void setup(const Ref<VisualShaderNodeInput> &p_input) {
		input = p_input;
		Ref<Texture> type_icon[5] = {
			EditorNode::get_singleton()->get_gui_base()->get_icon("float", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Vector3", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("bool", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Transform", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("ImageTexture", "EditorIcons"),
		};

		add_item("[None]");
		int to_select = -1;
		for (int i = 0; i < input->get_input_index_count(); i++) {
			if (input->get_input_name() == input->get_input_index_name(i)) {
				to_select = i + 1;
			}
			add_icon_item(type_icon[input->get_input_index_type(i)], input->get_input_index_name(i));
		}

		if (to_select >= 0) {
			select(to_select);
		}
	}
};

////////////////

class VisualShaderNodePluginUniformRefEditor : public OptionButton {
	GDCLASS(VisualShaderNodePluginUniformRefEditor, OptionButton);

	Ref<VisualShaderNodeUniformRef> uniform_ref;

protected:
	static void _bind_methods() {
		ClassDB::bind_method("_item_selected", &VisualShaderNodePluginUniformRefEditor::_item_selected);
	}

public:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			connect("item_selected", this, "_item_selected");
		}
	}

	void _item_selected(int p_item) {
		VisualShaderEditor::get_singleton()->call_deferred("_uniform_select_item", uniform_ref, get_item_text(p_item));
	}

	void setup(const Ref<VisualShaderNodeUniformRef> &p_uniform_ref) {
		uniform_ref = p_uniform_ref;

		Ref<Texture> type_icon[6] = {
			EditorNode::get_singleton()->get_gui_base()->get_icon("float", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("bool", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Vector3", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Transform", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Color", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("ImageTexture", "EditorIcons"),
		};

		add_item("[None]");
		int to_select = -1;
		for (int i = 0; i < p_uniform_ref->get_uniforms_count(); i++) {
			if (p_uniform_ref->get_uniform_name() == p_uniform_ref->get_uniform_name_by_index(i)) {
				to_select = i + 1;
			}
			add_icon_item(type_icon[p_uniform_ref->get_uniform_type_by_index(i)], p_uniform_ref->get_uniform_name_by_index(i));
		}

		if (to_select >= 0) {
			select(to_select);
		}
	}
};

////////////////

class VisualShaderNodePluginDefaultEditor : public VBoxContainer {
	GDCLASS(VisualShaderNodePluginDefaultEditor, VBoxContainer);
	Ref<Resource> parent_resource;

public:
	void _property_changed(const String &prop, const Variant &p_value, const String &p_field, bool p_changing = false) {
		if (p_changing) {
			return;
		}

		UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

		updating = true;
		undo_redo->create_action(TTR("Edit Visual Property:") + " " + prop, UndoRedo::MERGE_ENDS);
		undo_redo->add_do_property(node.ptr(), prop, p_value);
		undo_redo->add_undo_property(node.ptr(), prop, node->get(prop));

		if (p_value.get_type() == Variant::OBJECT) {
			RES prev_res = node->get(prop);
			RES curr_res = p_value;

			if (curr_res.is_null()) {
				undo_redo->add_do_method(this, "_open_inspector", (RES)parent_resource.ptr());
			} else {
				undo_redo->add_do_method(this, "_open_inspector", (RES)curr_res.ptr());
			}
			if (!prev_res.is_null()) {
				undo_redo->add_undo_method(this, "_open_inspector", (RES)prev_res.ptr());
			} else {
				undo_redo->add_undo_method(this, "_open_inspector", (RES)parent_resource.ptr());
			}
			undo_redo->add_do_method(this, "_refresh_request");
			undo_redo->add_undo_method(this, "_refresh_request");
		}
		undo_redo->commit_action();

		updating = false;
	}

	void _node_changed() {
		if (updating) {
			return;
		}
		for (int i = 0; i < properties.size(); i++) {
			properties[i]->update_property();
		}
	}

	void _refresh_request() {
		VisualShaderEditor::get_singleton()->call_deferred("_update_graph");
	}

	void _resource_selected(const String &p_path, RES p_resource) {
		_open_inspector(p_resource);
	}

	void _open_inspector(RES p_resource) {
		EditorNode::get_singleton()->get_inspector()->edit(p_resource.ptr());
	}

	bool updating;
	Ref<VisualShaderNode> node;
	Vector<EditorProperty *> properties;

	void setup(Ref<Resource> p_parent_resource, Vector<EditorProperty *> p_properties, const Vector<StringName> &p_names, Ref<VisualShaderNode> p_node) {
		parent_resource = p_parent_resource;
		updating = false;
		node = p_node;
		properties = p_properties;

		for (int i = 0; i < p_properties.size(); i++) {
			add_child(p_properties[i]);

			bool res_prop = Object::cast_to<EditorPropertyResource>(p_properties[i]);
			if (res_prop) {
				p_properties[i]->connect("resource_selected", this, "_resource_selected");
			}

			properties[i]->connect("property_changed", this, "_property_changed");
			properties[i]->set_object_and_property(node.ptr(), p_names[i]);
			properties[i]->update_property();
			properties[i]->set_name_split_ratio(0);
		}
		node->connect("changed", this, "_node_changed");
		node->connect("editor_refresh_request", this, "_refresh_request", varray(), CONNECT_DEFERRED);
	}

	static void _bind_methods() {
		ClassDB::bind_method("_property_changed", &VisualShaderNodePluginDefaultEditor::_property_changed, DEFVAL(String()), DEFVAL(false));
		ClassDB::bind_method("_node_changed", &VisualShaderNodePluginDefaultEditor::_node_changed);
		ClassDB::bind_method("_refresh_request", &VisualShaderNodePluginDefaultEditor::_refresh_request);
		ClassDB::bind_method("_resource_selected", &VisualShaderNodePluginDefaultEditor::_resource_selected);
		ClassDB::bind_method("_open_inspector", &VisualShaderNodePluginDefaultEditor::_open_inspector);
	}
};

Control *VisualShaderNodePluginDefault::create_editor(const Ref<Resource> &p_parent_resource, const Ref<VisualShaderNode> &p_node) {
	if (p_node->is_class("VisualShaderNodeUniformRef")) {
		//create input
		VisualShaderNodePluginUniformRefEditor *uniform_editor = memnew(VisualShaderNodePluginUniformRefEditor);
		uniform_editor->setup(p_node);
		return uniform_editor;
	}

	if (p_node->is_class("VisualShaderNodeInput")) {
		//create input
		VisualShaderNodePluginInputEditor *input_editor = memnew(VisualShaderNodePluginInputEditor);
		input_editor->setup(p_node);
		return input_editor;
	}

	Vector<StringName> properties = p_node->get_editable_properties();
	if (properties.size() == 0) {
		return nullptr;
	}

	List<PropertyInfo> props;
	p_node->get_property_list(&props);

	Vector<PropertyInfo> pinfo;

	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
		for (int i = 0; i < properties.size(); i++) {
			if (E->get().name == String(properties[i])) {
				pinfo.push_back(E->get());
			}
		}
	}

	if (pinfo.size() == 0) {
		return nullptr;
	}

	properties.clear();

	Ref<VisualShaderNode> node = p_node;
	Vector<EditorProperty *> editors;

	for (int i = 0; i < pinfo.size(); i++) {
		EditorProperty *prop = EditorInspector::instantiate_property_editor(node.ptr(), pinfo[i].type, pinfo[i].name, pinfo[i].hint, pinfo[i].hint_string, pinfo[i].usage);
		if (!prop) {
			return nullptr;
		}

		if (Object::cast_to<EditorPropertyResource>(prop)) {
			Object::cast_to<EditorPropertyResource>(prop)->set_use_sub_inspector(false);
			prop->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyTransform>(prop) || Object::cast_to<EditorPropertyVector3>(prop)) {
			prop->set_custom_minimum_size(Size2(250 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyFloat>(prop)) {
			prop->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyEnum>(prop)) {
			prop->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
			Object::cast_to<EditorPropertyEnum>(prop)->set_option_button_clip(false);
		}

		editors.push_back(prop);
		properties.push_back(pinfo[i].name);
	}
	VisualShaderNodePluginDefaultEditor *editor = memnew(VisualShaderNodePluginDefaultEditor);
	editor->setup(p_parent_resource, editors, properties, p_node);
	return editor;
}

void EditorPropertyShaderMode::_option_selected(int p_which) {
	//will not use this, instead will do all the logic setting manually
	//emit_signal("property_changed", get_edited_property(), p_which);

	Ref<VisualShader> visual_shader(Object::cast_to<VisualShader>(get_edited_object()));

	if (visual_shader->get_mode() == p_which) {
		return;
	}

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("Visual Shader Mode Changed"));
	//do is easy
	undo_redo->add_do_method(visual_shader.ptr(), "set_mode", p_which);
	undo_redo->add_undo_method(visual_shader.ptr(), "set_mode", visual_shader->get_mode());
	//now undo is hell

	//1. restore connections to output
	for (int i = 0; i < VisualShader::TYPE_MAX; i++) {
		VisualShader::Type type = VisualShader::Type(i);
		List<VisualShader::Connection> conns;
		visual_shader->get_node_connections(type, &conns);
		for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
			if (E->get().to_node == VisualShader::NODE_ID_OUTPUT) {
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}
		}
	}
	//2. restore input indices
	for (int i = 0; i < VisualShader::TYPE_MAX; i++) {
		VisualShader::Type type = VisualShader::Type(i);
		Vector<int> nodes = visual_shader->get_node_list(type);
		for (int j = 0; j < nodes.size(); j++) {
			Ref<VisualShaderNodeInput> input = visual_shader->get_node(type, nodes[j]);
			if (!input.is_valid()) {
				continue;
			}

			undo_redo->add_undo_method(input.ptr(), "set_input_name", input->get_input_name());
		}
	}

	//3. restore enums and flags
	List<PropertyInfo> props;
	visual_shader->get_property_list(&props);

	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
		if (E->get().name.begins_with("flags/") || E->get().name.begins_with("modes/")) {
			undo_redo->add_undo_property(visual_shader.ptr(), E->get().name, visual_shader->get(E->get().name));
		}
	}

	undo_redo->add_do_method(VisualShaderEditor::get_singleton(), "_update_options_menu");
	undo_redo->add_undo_method(VisualShaderEditor::get_singleton(), "_update_options_menu");

	//update graph
	undo_redo->add_do_method(VisualShaderEditor::get_singleton(), "_update_graph");
	undo_redo->add_undo_method(VisualShaderEditor::get_singleton(), "_update_graph");

	undo_redo->commit_action();
}

void EditorPropertyShaderMode::update_property() {
	int which = get_edited_object()->get(get_edited_property());
	options->select(which);
}

void EditorPropertyShaderMode::setup(const Vector<String> &p_options) {
	for (int i = 0; i < p_options.size(); i++) {
		options->add_item(p_options[i], i);
	}
}

void EditorPropertyShaderMode::set_option_button_clip(bool p_enable) {
	options->set_clip_text(p_enable);
}

void EditorPropertyShaderMode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_option_selected"), &EditorPropertyShaderMode::_option_selected);
}

EditorPropertyShaderMode::EditorPropertyShaderMode() {
	options = memnew(OptionButton);
	options->set_clip_text(true);
	add_child(options);
	add_focusable(options);
	options->connect("item_selected", this, "_option_selected");
}

bool EditorInspectorShaderModePlugin::can_handle(Object *p_object) {
	return true; //can handle everything
}

void EditorInspectorShaderModePlugin::parse_begin(Object *p_object) {
	//do none
}

bool EditorInspectorShaderModePlugin::parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage) {
	if (p_path == "mode" && p_object->is_class("VisualShader") && p_type == Variant::INT) {
		EditorPropertyShaderMode *editor = memnew(EditorPropertyShaderMode);
		Vector<String> options = p_hint_text.split(",");
		editor->setup(options);
		add_property_editor(p_path, editor);

		return true;
	}

	return false; //can be overridden, although it will most likely be last anyway
}

void EditorInspectorShaderModePlugin::parse_end() {
	//do none
}
//////////////////////////////////

void VisualShaderNodePortPreview::_shader_changed() {
	if (shader.is_null()) {
		return;
	}

	Vector<VisualShader::DefaultTextureParam> default_textures;
	String shader_code = shader->generate_preview_shader(type, node, port, default_textures);

	Ref<Shader> preview_shader;
	preview_shader.instance();
	preview_shader->set_code(shader_code);
	for (int i = 0; i < default_textures.size(); i++) {
		preview_shader->set_default_texture_param(default_textures[i].name, default_textures[i].param);
	}

	Ref<ShaderMaterial> material;
	material.instance();
	material->set_shader(preview_shader);

	//find if a material is also being edited and copy parameters to this one

	for (int i = EditorNode::get_singleton()->get_editor_history()->get_path_size() - 1; i >= 0; i--) {
		Object *object = ObjectDB::get_instance(EditorNode::get_singleton()->get_editor_history()->get_path_object(i));
		ShaderMaterial *src_mat;
		if (!object) {
			continue;
		}
		if (object->has_method("get_material_override")) { // trying getting material from MeshInstance
			src_mat = Object::cast_to<ShaderMaterial>(object->call("get_material_override"));
		} else if (object->has_method("get_material")) { // from CanvasItem/Node2D
			src_mat = Object::cast_to<ShaderMaterial>(object->call("get_material"));
		} else {
			src_mat = Object::cast_to<ShaderMaterial>(object);
		}
		if (src_mat && src_mat->get_shader().is_valid()) {
			List<PropertyInfo> params;
			src_mat->get_shader()->get_param_list(&params);
			for (List<PropertyInfo>::Element *E = params.front(); E; E = E->next()) {
				material->set(E->get().name, src_mat->get(E->get().name));
			}
		}
	}

	set_material(material);
}

void VisualShaderNodePortPreview::setup(const Ref<VisualShader> &p_shader, VisualShader::Type p_type, int p_node, int p_port) {
	shader = p_shader;
	shader->connect("changed", this, "_shader_changed");
	type = p_type;
	port = p_port;
	node = p_node;
	update();
	_shader_changed();
}

Size2 VisualShaderNodePortPreview::get_minimum_size() const {
	return Size2(100, 100) * EDSCALE;
}

void VisualShaderNodePortPreview::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		Vector<Vector2> points;
		Vector<Vector2> uvs;
		Vector<Color> colors;
		points.push_back(Vector2());
		uvs.push_back(Vector2(0, 0));
		colors.push_back(Color(1, 1, 1, 1));
		points.push_back(Vector2(get_size().width, 0));
		uvs.push_back(Vector2(1, 0));
		colors.push_back(Color(1, 1, 1, 1));
		points.push_back(get_size());
		uvs.push_back(Vector2(1, 1));
		colors.push_back(Color(1, 1, 1, 1));
		points.push_back(Vector2(0, get_size().height));
		uvs.push_back(Vector2(0, 1));
		colors.push_back(Color(1, 1, 1, 1));

		draw_primitive(points, colors, uvs);
	}
}

void VisualShaderNodePortPreview::_bind_methods() {
	ClassDB::bind_method("_shader_changed", &VisualShaderNodePortPreview::_shader_changed);
}

VisualShaderNodePortPreview::VisualShaderNodePortPreview() {
}

//////////////////////////////////

String VisualShaderConversionPlugin::converts_to() const {
	return "Shader";
}

bool VisualShaderConversionPlugin::handles(const Ref<Resource> &p_resource) const {
	Ref<VisualShader> vshader = p_resource;
	return vshader.is_valid();
}

Ref<Resource> VisualShaderConversionPlugin::convert(const Ref<Resource> &p_resource) const {
	Ref<VisualShader> vshader = p_resource;
	ERR_FAIL_COND_V(!vshader.is_valid(), Ref<Resource>());

	Ref<Shader> shader;
	shader.instance();

	String code = vshader->get_code();
	shader->set_code(code);

	return shader;
}
