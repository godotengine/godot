/*************************************************************************/
/*  visual_shader_editor_plugin.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "editor/editor_properties.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/viewport.h"

Control *VisualShaderNodePlugin::create_editor(const Ref<VisualShaderNode> &p_node) {

	if (get_script_instance()) {
		return get_script_instance()->call("create_editor", p_node);
	}
	return NULL;
}

void VisualShaderNodePlugin::_bind_methods() {

	BIND_VMETHOD(MethodInfo(Variant::OBJECT, "create_editor", PropertyInfo(Variant::OBJECT, "for_node", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderNode")));
}

///////////////////

void VisualShaderEditor::edit(VisualShader *p_visual_shader) {

	if (p_visual_shader) {
		visual_shader = Ref<VisualShader>(p_visual_shader);
	} else {
		visual_shader.unref();
	}

	if (visual_shader.is_null()) {
		hide();
	} else {
		_update_graph();
	}
}

void VisualShaderEditor::add_plugin(const Ref<VisualShaderNodePlugin> &p_plugin) {
	if (plugins.find(p_plugin) != -1)
		return;
	plugins.push_back(p_plugin);
}

void VisualShaderEditor::remove_plugin(const Ref<VisualShaderNodePlugin> &p_plugin) {
	plugins.erase(p_plugin);
}

void VisualShaderEditor::add_custom_type(const String &p_name, const String &p_category, const Ref<Script> &p_script) {

	for (int i = 0; i < add_options.size(); i++) {
		ERR_FAIL_COND(add_options[i].script == p_script);
	}

	AddOption ao;
	ao.name = p_name;
	ao.script = p_script;
	ao.category = p_category;
	add_options.push_back(ao);

	_update_options_menu();
}

void VisualShaderEditor::remove_custom_type(const Ref<Script> &p_script) {

	for (int i = 0; i < add_options.size(); i++) {
		if (add_options[i].script == p_script) {
			add_options.remove(i);
			return;
		}
	}

	_update_options_menu();
}

void VisualShaderEditor::_update_options_menu() {

	String prev_category;
	add_node->get_popup()->clear();
	for (int i = 0; i < add_options.size(); i++) {
		if (prev_category != add_options[i].category) {
			add_node->get_popup()->add_separator(add_options[i].category);
		}
		add_node->get_popup()->add_item(add_options[i].name, i);
		prev_category = add_options[i].category;
	}
}

Size2 VisualShaderEditor::get_minimum_size() const {

	return Size2(10, 200);
}

void VisualShaderEditor::_draw_color_over_button(Object *obj, Color p_color) {

	Button *button = Object::cast_to<Button>(obj);
	if (!button)
		return;

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

void VisualShaderEditor::_update_graph() {

	if (updating)
		return;

	if (visual_shader.is_null())
		return;

	graph->set_scroll_ofs(visual_shader->get_graph_offset() * EDSCALE);

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());
	graph->clear_connections();
	//erase all nodes
	for (int i = 0; i < graph->get_child_count(); i++) {

		if (Object::cast_to<GraphNode>(graph->get_child(i))) {
			memdelete(graph->get_child(i));
			i--;
		}
	}

	static const Color type_color[3] = {
		Color::html("#61daf4"),
		Color::html("#d67dee"),
		Color::html("#f6a86e")
	};

	List<VisualShader::Connection> connections;
	visual_shader->get_node_connections(type, &connections);

	Ref<StyleBoxEmpty> label_style = make_empty_stylebox(2, 1, 2, 1);

	Vector<int> nodes = visual_shader->get_node_list(type);

	for (int n_i = 0; n_i < nodes.size(); n_i++) {

		Vector2 position = visual_shader->get_node_position(type, nodes[n_i]);
		Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, nodes[n_i]);

		GraphNode *node = memnew(GraphNode);
		graph->add_child(node);

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

		Control *custom_editor = NULL;
		int port_offset = 0;

		Ref<VisualShaderNodeUniform> uniform = vsnode;
		if (uniform.is_valid()) {
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
			custom_editor = plugins.write[i]->create_editor(vsnode);
			if (custom_editor) {
				break;
			}
		}

		if (custom_editor && vsnode->get_output_port_count() > 0 && vsnode->get_output_port_name(0) == "" && (vsnode->get_input_port_count() == 0 || vsnode->get_input_port_name(0) == "")) {
			//will be embedded in first port
		} else if (custom_editor) {
			port_offset++;
			node->add_child(custom_editor);
			custom_editor = NULL;
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
					case Variant::INT:
					case Variant::REAL: {
						button->set_text(String::num(default_value, 4));
					} break;
					case Variant::VECTOR3: {
						Vector3 v = default_value;
						button->set_text(String::num(v.x, 3) + "," + String::num(v.y, 3) + "," + String::num(v.z, 3));
					} break;
					default: {}
				}
			}

			if (i == 0 && custom_editor) {

				hb->add_child(custom_editor);
				custom_editor->set_h_size_flags(SIZE_EXPAND_FILL);
			} else {

				if (valid_left) {

					Label *label = memnew(Label);
					label->set_text(name_left);
					label->add_style_override("normal", label_style); //more compact
					hb->add_child(label);
				}

				hb->add_spacer();

				if (valid_right) {

					Label *label = memnew(Label);
					label->set_text(name_right);
					label->set_align(Label::ALIGN_RIGHT);
					label->add_style_override("normal", label_style); //more compact
					hb->add_child(label);
				}
			}

			if (valid_right && edit_type->get_selected() == VisualShader::TYPE_FRAGMENT) {
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

			node->add_child(hb);

			node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], valid_right, port_right, type_color[port_right]);

			if (EditorSettings::get_singleton()->get("interface/theme/use_graph_node_headers")) {
				Ref<StyleBoxFlat> sb = node->get_stylebox("frame", "GraphNode");
				Color c = sb->get_border_color(MARGIN_TOP);
				Color mono_color = ((c.r + c.g + c.b) / 3) < 0.7 ? Color(1.0, 1.0, 1.0) : Color(0.0, 0.0, 0.0);
				mono_color.a = 0.85;
				c = mono_color;

				node->add_color_override("title_color", c);
				c.a = 0.7;
				node->add_color_override("close_color", c);
			}
		}

		if (vsnode->get_output_port_for_preview() >= 0) {
			VisualShaderNodePortPreview *port_preview = memnew(VisualShaderNodePortPreview);
			port_preview->setup(visual_shader, type, nodes[n_i], vsnode->get_output_port_for_preview());
			port_preview->set_h_size_flags(SIZE_SHRINK_CENTER);
			node->add_child(port_preview);
		}

		String error = vsnode->get_warning(visual_shader->get_mode(), type);
		if (error != String()) {
			Label *error_label = memnew(Label);
			error_label->add_color_override("font_color", get_color("error_color", "Editor"));
			error_label->set_text(error);
			node->add_child(error_label);
		}
	}

	for (List<VisualShader::Connection>::Element *E = connections.front(); E; E = E->next()) {

		int from = E->get().from_node;
		int from_idx = E->get().from_port;
		int to = E->get().to_node;
		int to_idx = E->get().to_port;

		graph->connect_node(itos(from), from_idx, itos(to), to_idx);
	}
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
	undo_redo->create_action("Set Uniform Name");
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
	undo_redo->create_action("Set Uniform Name");
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

void VisualShaderEditor::_port_edited() {

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	Variant value = property_editor->get_variant();
	Ref<VisualShaderNode> vsn = visual_shader->get_node(type, editing_node);
	ERR_FAIL_COND(!vsn.is_valid());

	undo_redo->create_action("Set Input Default Port");
	undo_redo->add_do_method(vsn.ptr(), "set_input_port_default_value", editing_port, value);
	undo_redo->add_undo_method(vsn.ptr(), "set_input_port_default_value", editing_port, vsn->get_input_port_default_value(editing_port));
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
	property_editor->edit(NULL, "", value.get_type(), value, 0, "");
	property_editor->popup();
	editing_node = p_node;
	editing_port = p_port;
}

void VisualShaderEditor::_add_node(int p_idx) {

	ERR_FAIL_INDEX(p_idx, add_options.size());

	Ref<VisualShaderNode> vsnode;

	if (add_options[p_idx].type != String()) {
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instance(add_options[p_idx].type));
		ERR_FAIL_COND(!vsn);
		vsnode = Ref<VisualShaderNode>(vsn);
	} else {
		ERR_FAIL_COND(add_options[p_idx].script.is_null());
		String base_type = add_options[p_idx].script->get_instance_base_type();
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instance(base_type));
		ERR_FAIL_COND(!vsn);
		vsnode = Ref<VisualShaderNode>(vsn);
		vsnode->set_script(add_options[p_idx].script.get_ref_ptr());
	}

	Point2 position = (graph->get_scroll_ofs() + graph->get_size() * 0.5) / EDSCALE;

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	int id_to_use = visual_shader->get_valid_node_id(type);

	undo_redo->create_action("Add Node to Visual Shader");
	undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, vsnode, position, id_to_use);
	undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_to_use);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void VisualShaderEditor::_node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_node) {

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	updating = true;
	undo_redo->create_action("Node Moved");
	undo_redo->add_do_method(visual_shader.ptr(), "set_node_position", type, p_node, p_to);
	undo_redo->add_undo_method(visual_shader.ptr(), "set_node_position", type, p_node, p_from);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;
}

void VisualShaderEditor::_connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	int from = p_from.to_int();
	int to = p_to.to_int();

	if (!visual_shader->can_connect_nodes(type, from, p_from_index, to, p_to_index)) {
		EditorNode::get_singleton()->show_warning(TTR("Unable to connect, port may be in use or connection may be invalid."));
		return;
	}

	undo_redo->create_action("Nodes Connected");

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
	undo_redo->create_action("Nodes Disconnected");
	undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	//updating = false;
}

void VisualShaderEditor::_connection_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_position) {
}

void VisualShaderEditor::_delete_request(int which) {

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	undo_redo->create_action("Delete Node");
	undo_redo->add_do_method(visual_shader.ptr(), "remove_node", type, which);
	undo_redo->add_undo_method(visual_shader.ptr(), "add_node", type, visual_shader->get_node(type, which), visual_shader->get_node_position(type, which), which);

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

void VisualShaderEditor::_input(const Ref<InputEvent> p_event) {
	if (graph->has_focus()) {
		Ref<InputEventMouseButton> mb = p_event;

		if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) {
			add_node->get_popup()->set_position(get_viewport()->get_mouse_position());
			add_node->get_popup()->show_modal();
		}
	}
}

void VisualShaderEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {

		error_panel->add_style_override("panel", get_stylebox("bg", "Tree"));
		error_label->add_color_override("font_color", get_color("error_color", "Editor"));

		if (p_what == NOTIFICATION_THEME_CHANGED && is_visible_in_tree())
			_update_graph();
	}

	if (p_what == NOTIFICATION_PROCESS) {
	}
}

void VisualShaderEditor::_scroll_changed(const Vector2 &p_scroll) {
	if (updating)
		return;
	updating = true;
	visual_shader->set_graph_offset(p_scroll / EDSCALE);
	updating = false;
}

void VisualShaderEditor::_node_changed(int p_id) {
	if (updating)
		return;

	if (is_visible_in_tree()) {
		_update_graph();
	}
}

void VisualShaderEditor::_duplicate_nodes() {

	VisualShader::Type type = VisualShader::Type(edit_type->get_selected());

	List<int> nodes;
	Set<int> excluded;

	for (int i = 0; i < graph->get_child_count(); i++) {

		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			int id = String(gn->get_name()).to_int();
			Ref<VisualShaderNode> node = visual_shader->get_node(type, id);
			Ref<VisualShaderNodeOutput> output = node;
			if (output.is_valid()) { // can't duplicate output
				excluded.insert(id);
				continue;
			}
			if (node.is_valid() && gn->is_selected()) {
				nodes.push_back(id);
			}
			excluded.insert(id);
		}
	}

	if (nodes.empty())
		return;

	undo_redo->create_action("Duplicate Nodes");

	int base_id = visual_shader->get_valid_node_id(type);
	int id_from = base_id;
	Map<int, int> connection_remap;

	for (List<int>::Element *E = nodes.front(); E; E = E->next()) {

		connection_remap[E->get()] = id_from;
		Ref<VisualShaderNode> node = visual_shader->get_node(type, E->get());

		Ref<VisualShaderNode> dupli = node->duplicate();

		undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, dupli, visual_shader->get_node_position(type, E->get()) + Vector2(10, 10) * EDSCALE, id_from);
		undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_from);

		id_from++;
	}

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
		if (connection_remap.has(E->get().from_node) && connection_remap.has(E->get().to_node)) {
			undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, connection_remap[E->get().from_node], E->get().from_port, connection_remap[E->get().to_node], E->get().to_port);
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();

	// reselect duplicated nodes by excluding the other ones
	for (int i = 0; i < graph->get_child_count(); i++) {

		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			int id = String(gn->get_name()).to_int();
			if (!excluded.has(id)) {
				gn->set_selected(true);
			} else {
				gn->set_selected(false);
			}
		}
	}
}

void VisualShaderEditor::_mode_selected(int p_id) {
	_update_graph();
}

void VisualShaderEditor::_input_select_item(Ref<VisualShaderNodeInput> input, String name) {

	String prev_name = input->get_input_name();

	if (name == prev_name)
		return;

	bool type_changed = input->get_input_type_by_name(name) != input->get_input_type_by_name(prev_name);

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action("Visual Shader Input Type Changed");

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

void VisualShaderEditor::_bind_methods() {

	ClassDB::bind_method("_update_graph", &VisualShaderEditor::_update_graph);
	ClassDB::bind_method("_add_node", &VisualShaderEditor::_add_node);
	ClassDB::bind_method("_node_dragged", &VisualShaderEditor::_node_dragged);
	ClassDB::bind_method("_connection_request", &VisualShaderEditor::_connection_request);
	ClassDB::bind_method("_disconnection_request", &VisualShaderEditor::_disconnection_request);
	ClassDB::bind_method("_node_selected", &VisualShaderEditor::_node_selected);
	ClassDB::bind_method("_scroll_changed", &VisualShaderEditor::_scroll_changed);
	ClassDB::bind_method("_delete_request", &VisualShaderEditor::_delete_request);
	ClassDB::bind_method("_node_changed", &VisualShaderEditor::_node_changed);
	ClassDB::bind_method("_edit_port_default_input", &VisualShaderEditor::_edit_port_default_input);
	ClassDB::bind_method("_port_edited", &VisualShaderEditor::_port_edited);
	ClassDB::bind_method("_connection_to_empty", &VisualShaderEditor::_connection_to_empty);
	ClassDB::bind_method("_line_edit_focus_out", &VisualShaderEditor::_line_edit_focus_out);
	ClassDB::bind_method("_line_edit_changed", &VisualShaderEditor::_line_edit_changed);
	ClassDB::bind_method("_duplicate_nodes", &VisualShaderEditor::_duplicate_nodes);
	ClassDB::bind_method("_mode_selected", &VisualShaderEditor::_mode_selected);
	ClassDB::bind_method("_input_select_item", &VisualShaderEditor::_input_select_item);
	ClassDB::bind_method("_preview_select_port", &VisualShaderEditor::_preview_select_port);
	ClassDB::bind_method("_input", &VisualShaderEditor::_input);
}

VisualShaderEditor *VisualShaderEditor::singleton = NULL;

VisualShaderEditor::VisualShaderEditor() {

	singleton = this;
	updating = false;

	graph = memnew(GraphEdit);
	add_child(graph);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_TRANSFORM);
	//graph->add_valid_left_disconnect_type(0);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->connect("connection_request", this, "_connection_request", varray(), CONNECT_DEFERRED);
	graph->connect("disconnection_request", this, "_disconnection_request", varray(), CONNECT_DEFERRED);
	graph->connect("node_selected", this, "_node_selected");
	graph->connect("scroll_offset_changed", this, "_scroll_changed");
	graph->connect("duplicate_nodes_request", this, "_duplicate_nodes");
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShaderNode::PORT_TYPE_TRANSFORM);

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

	add_node = memnew(MenuButton);
	graph->get_zoom_hbox()->add_child(add_node);
	add_node->set_text(TTR("Add Node..."));
	graph->get_zoom_hbox()->move_child(add_node, 0);
	add_node->get_popup()->connect("id_pressed", this, "_add_node");

	add_options.push_back(AddOption("Scalar", "Constants", "VisualShaderNodeScalarConstant"));
	add_options.push_back(AddOption("Vector", "Constants", "VisualShaderNodeVec3Constant"));
	add_options.push_back(AddOption("Color", "Constants", "VisualShaderNodeColorConstant"));
	add_options.push_back(AddOption("Transform", "Constants", "VisualShaderNodeTransformConstant"));
	add_options.push_back(AddOption("Texture", "Constants", "VisualShaderNodeTexture"));
	add_options.push_back(AddOption("CubeMap", "Constants", "VisualShaderNodeCubeMap"));
	add_options.push_back(AddOption("ScalarOp", "Operators", "VisualShaderNodeScalarOp"));
	add_options.push_back(AddOption("VectorOp", "Operators", "VisualShaderNodeVectorOp"));
	add_options.push_back(AddOption("ColorOp", "Operators", "VisualShaderNodeColorOp"));
	add_options.push_back(AddOption("TransformMult", "Operators", "VisualShaderNodeTransformMult"));
	add_options.push_back(AddOption("TransformVectorMult", "Operators", "VisualShaderNodeTransformVecMult"));
	add_options.push_back(AddOption("ScalarFunc", "Functions", "VisualShaderNodeScalarFunc"));
	add_options.push_back(AddOption("VectorFunc", "Functions", "VisualShaderNodeVectorFunc"));
	add_options.push_back(AddOption("DotProduct", "Functions", "VisualShaderNodeDotProduct"));
	add_options.push_back(AddOption("VectorLen", "Functions", "VisualShaderNodeVectorLen"));
	add_options.push_back(AddOption("ScalarInterp", "Interpolation", "VisualShaderNodeScalarInterp"));
	add_options.push_back(AddOption("VectorInterp", "Interpolation", "VisualShaderNodeVectorInterp"));
	add_options.push_back(AddOption("VectorCompose", "Compose", "VisualShaderNodeVectorCompose"));
	add_options.push_back(AddOption("TransformCompose", "Compose", "VisualShaderNodeTransformCompose"));
	add_options.push_back(AddOption("VectorDecompose", "Decompose", "VisualShaderNodeVectorDecompose"));
	add_options.push_back(AddOption("TransformDecompose", "Decompose", "VisualShaderNodeTransformDecompose"));
	add_options.push_back(AddOption("Scalar", "Uniforms", "VisualShaderNodeScalarUniform"));
	add_options.push_back(AddOption("Vector", "Uniforms", "VisualShaderNodeVec3Uniform"));
	add_options.push_back(AddOption("Color", "Uniforms", "VisualShaderNodeColorUniform"));
	add_options.push_back(AddOption("Transform", "Uniforms", "VisualShaderNodeTransformUniform"));
	add_options.push_back(AddOption("Texture", "Uniforms", "VisualShaderNodeTextureUniform"));
	add_options.push_back(AddOption("CubeMap", "Uniforms", "VisualShaderNodeCubeMapUniform"));
	add_options.push_back(AddOption("Input", "Inputs", "VisualShaderNodeInput"));

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
		visual_shader_editor->set_process_input(true);
		//visual_shader_editor->set_process(true);
	} else {

		if (visual_shader_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
		button->hide();
		visual_shader_editor->set_process_input(false);
		//visual_shader_editor->set_process(false);
	}
}

VisualShaderEditorPlugin::VisualShaderEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	visual_shader_editor = memnew(VisualShaderEditor);
	visual_shader_editor->set_custom_minimum_size(Size2(0, 300));

	button = editor->add_bottom_panel_item(TTR("VisualShader"), visual_shader_editor);
	button->hide();
}

VisualShaderEditorPlugin::~VisualShaderEditorPlugin() {
}

////////////////

class VisualShaderNodePluginInputEditor : public OptionButton {
	GDCLASS(VisualShaderNodePluginInputEditor, OptionButton)

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
		Ref<Texture> type_icon[3] = {
			EditorNode::get_singleton()->get_gui_base()->get_icon("float", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Vector3", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_icon("Transform", "EditorIcons"),
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

class VisualShaderNodePluginDefaultEditor : public VBoxContainer {
	GDCLASS(VisualShaderNodePluginDefaultEditor, VBoxContainer)
public:
	void _property_changed(const String &prop, const Variant &p_value, const String &p_field, bool p_changing = false) {

		if (p_changing)
			return;

		UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

		updating = true;
		undo_redo->create_action("Edit Visual Property: " + prop, UndoRedo::MERGE_ENDS);
		undo_redo->add_do_property(node.ptr(), prop, p_value);
		undo_redo->add_undo_property(node.ptr(), prop, node->get(prop));
		undo_redo->commit_action();
		updating = false;
	}

	void _node_changed() {
		if (updating)
			return;
		for (int i = 0; i < properties.size(); i++) {
			properties[i]->update_property();
		}
	}

	void _refresh_request() {
		VisualShaderEditor::get_singleton()->call_deferred("_update_graph");
	}

	bool updating;
	Ref<VisualShaderNode> node;
	Vector<EditorProperty *> properties;

	void setup(Vector<EditorProperty *> p_properties, const Vector<StringName> &p_names, Ref<VisualShaderNode> p_node) {
		updating = false;
		node = p_node;
		properties = p_properties;

		for (int i = 0; i < p_properties.size(); i++) {

			add_child(p_properties[i]);

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
	}
};

Control *VisualShaderNodePluginDefault::create_editor(const Ref<VisualShaderNode> &p_node) {

	if (p_node->is_class("VisualShaderNodeInput")) {
		//create input
		VisualShaderNodePluginInputEditor *input_editor = memnew(VisualShaderNodePluginInputEditor);
		input_editor->setup(p_node);
		return input_editor;
	}

	Vector<StringName> properties = p_node->get_editable_properties();
	if (properties.size() == 0) {
		return NULL;
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

	if (pinfo.size() == 0)
		return NULL;

	properties.clear();

	Ref<VisualShaderNode> node = p_node;
	Vector<EditorProperty *> editors;

	for (int i = 0; i < pinfo.size(); i++) {

		EditorProperty *prop = EditorInspector::instantiate_property_editor(node.ptr(), pinfo[i].type, pinfo[i].name, pinfo[i].hint, pinfo[i].hint_string, pinfo[i].usage);
		if (!prop)
			return NULL;

		if (Object::cast_to<EditorPropertyResource>(prop)) {
			Object::cast_to<EditorPropertyResource>(prop)->set_use_sub_inspector(false);
			prop->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyTransform>(prop)) {
			prop->set_custom_minimum_size(Size2(250 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyFloat>(prop) || Object::cast_to<EditorPropertyVector3>(prop)) {
			prop->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyEnum>(prop)) {
			prop->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
			Object::cast_to<EditorPropertyEnum>(prop)->set_option_button_clip(false);
		}

		editors.push_back(prop);
		properties.push_back(pinfo[i].name);
	}
	VisualShaderNodePluginDefaultEditor *editor = memnew(VisualShaderNodePluginDefaultEditor);
	editor->setup(editors, properties, p_node);
	return editor;
}

void EditorPropertyShaderMode::_option_selected(int p_which) {

	//will not use this, instead will do all the logic setting manually
	//emit_signal("property_changed", get_edited_property(), p_which);

	Ref<VisualShader> visual_shader(Object::cast_to<VisualShader>(get_edited_object()));

	if (visual_shader->get_mode() == p_which)
		return;

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action("Visual Shader Mode Changed");
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
		for (int i = 0; i < nodes.size(); i++) {
			Ref<VisualShaderNodeInput> input = visual_shader->get_node(type, nodes[i]);
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
		if (!object)
			continue;
		ShaderMaterial *src_mat = Object::cast_to<ShaderMaterial>(object);
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
