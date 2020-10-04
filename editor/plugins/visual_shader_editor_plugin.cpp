/*************************************************************************/
/*  visual_shader_editor_plugin.cpp                                      */
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

#include "visual_shader_editor_plugin.h"

#include "core/input/input.h"
#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_log.h"
#include "editor/editor_properties.h"
#include "editor/editor_scale.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/window.h"
#include "scene/resources/visual_shader_nodes.h"
#include "servers/display_server.h"
#include "servers/rendering/shader_types.h"

struct FloatConstantDef {
	String name;
	float value;
	String desc;
};

static FloatConstantDef float_constant_defs[] = {
	{ "E", Math_E, TTR("E constant (2.718282). Represents the base of the natural logarithm.") },
	{ "Epsilon", CMP_EPSILON, TTR("Epsilon constant (0.00001). Smallest possible scalar number.") },
	{ "Phi", 1.618034f, TTR("Phi constant (1.618034). Golden ratio.") },
	{ "Pi/4", Math_PI / 4, TTR("Pi/4 constant (0.785398) or 45 degrees.") },
	{ "Pi/2", Math_PI / 2, TTR("Pi/2 constant (1.570796) or 90 degrees.") },
	{ "Pi", Math_PI, TTR("Pi constant (3.141593) or 180 degrees.") },
	{ "Tau", Math_TAU, TTR("Tau constant (6.283185) or 360 degrees.") },
	{ "Sqrt2", Math_SQRT2, TTR("Sqrt2 constant (1.414214). Square root of 2.") }
};

const int MAX_FLOAT_CONST_DEFS = sizeof(float_constant_defs) / sizeof(FloatConstantDef);

///////////////////

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

static Ref<StyleBoxEmpty> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBoxEmpty> style(memnew(StyleBoxEmpty));
	style->set_default_margin(MARGIN_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(MARGIN_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(MARGIN_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(MARGIN_TOP, p_margin_top * EDSCALE);
	return style;
}

///////////////////

VisualShaderGraphPlugin::VisualShaderGraphPlugin() {
}

void VisualShaderGraphPlugin::_bind_methods() {
	ClassDB::bind_method("add_node", &VisualShaderGraphPlugin::add_node);
	ClassDB::bind_method("remove_node", &VisualShaderGraphPlugin::remove_node);
	ClassDB::bind_method("connect_nodes", &VisualShaderGraphPlugin::connect_nodes);
	ClassDB::bind_method("disconnect_nodes", &VisualShaderGraphPlugin::disconnect_nodes);
	ClassDB::bind_method("set_node_position", &VisualShaderGraphPlugin::set_node_position);
	ClassDB::bind_method("set_node_size", &VisualShaderGraphPlugin::set_node_size);
	ClassDB::bind_method("show_port_preview", &VisualShaderGraphPlugin::show_port_preview);
	ClassDB::bind_method("update_node", &VisualShaderGraphPlugin::update_node);
	ClassDB::bind_method("update_node_deferred", &VisualShaderGraphPlugin::update_node_deferred);
	ClassDB::bind_method("set_input_port_default_value", &VisualShaderGraphPlugin::set_input_port_default_value);
	ClassDB::bind_method("set_uniform_name", &VisualShaderGraphPlugin::set_uniform_name);
	ClassDB::bind_method("set_expression", &VisualShaderGraphPlugin::set_expression);
	ClassDB::bind_method("update_curve", &VisualShaderGraphPlugin::update_curve);
	ClassDB::bind_method("update_constant", &VisualShaderGraphPlugin::update_constant);
}

void VisualShaderGraphPlugin::register_shader(VisualShader *p_shader) {
	visual_shader = Ref<VisualShader>(p_shader);
}

void VisualShaderGraphPlugin::set_connections(List<VisualShader::Connection> &p_connections) {
	connections = p_connections;
}

void VisualShaderGraphPlugin::show_port_preview(VisualShader::Type p_type, int p_node_id, int p_port_id) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_node_id)) {
		for (Map<int, Port>::Element *E = links[p_node_id].output_ports.front(); E; E = E->next()) {
			E->value().preview_button->set_pressed(false);
		}

		if (links[p_node_id].preview_visible && !is_dirty() && links[p_node_id].preview_box != nullptr) {
			links[p_node_id].graph_node->remove_child(links[p_node_id].preview_box);
			memdelete(links[p_node_id].preview_box);
			links[p_node_id].graph_node->set_size(Vector2(-1, -1));
			links[p_node_id].preview_visible = false;
		}

		if (p_port_id != -1) {
			if (is_dirty()) {
				links[p_node_id].preview_pos = links[p_node_id].graph_node->get_child_count();
			}

			VBoxContainer *vbox = memnew(VBoxContainer);
			links[p_node_id].graph_node->add_child(vbox);
			if (links[p_node_id].preview_pos != -1) {
				links[p_node_id].graph_node->move_child(vbox, links[p_node_id].preview_pos);
			}

			Control *offset = memnew(Control);
			offset->set_custom_minimum_size(Size2(0, 5 * EDSCALE));
			vbox->add_child(offset);

			VisualShaderNodePortPreview *port_preview = memnew(VisualShaderNodePortPreview);
			port_preview->setup(visual_shader, visual_shader->get_shader_type(), p_node_id, p_port_id);
			port_preview->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
			vbox->add_child(port_preview);
			links[p_node_id].preview_visible = true;
			links[p_node_id].preview_box = vbox;
			links[p_node_id].output_ports[p_port_id].preview_button->set_pressed(true);
		}
	}
}

void VisualShaderGraphPlugin::update_node_deferred(VisualShader::Type p_type, int p_node_id) {
	call_deferred("update_node", p_type, p_node_id);
}

void VisualShaderGraphPlugin::update_node(VisualShader::Type p_type, int p_node_id) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id)) {
		return;
	}
	remove_node(p_type, p_node_id);
	add_node(p_type, p_node_id);
}

void VisualShaderGraphPlugin::set_input_port_default_value(VisualShader::Type p_type, int p_node_id, int p_port_id, Variant p_value) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id)) {
		return;
	}

	Button *button = links[p_node_id].input_ports[p_port_id].default_input_button;

	switch (p_value.get_type()) {
		case Variant::COLOR: {
			button->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
			if (!button->is_connected("draw", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_draw_color_over_button))) {
				button->connect("draw", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_draw_color_over_button), varray(button, p_value));
			}
		} break;
		case Variant::BOOL: {
			button->set_text(((bool)p_value) ? "true" : "false");
		} break;
		case Variant::INT:
		case Variant::FLOAT: {
			button->set_text(String::num(p_value, 4));
		} break;
		case Variant::VECTOR3: {
			Vector3 v = p_value;
			button->set_text(String::num(v.x, 3) + "," + String::num(v.y, 3) + "," + String::num(v.z, 3));
		} break;
		default: {
		}
	}
}

void VisualShaderGraphPlugin::set_uniform_name(VisualShader::Type p_type, int p_node_id, const String &p_name) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_node_id) && links[p_node_id].uniform_name != nullptr) {
		links[p_node_id].uniform_name->set_text(p_name);
	}
}

void VisualShaderGraphPlugin::update_curve(int p_node_id) {
	if (links.has(p_node_id) && links[p_node_id].curve_editor) {
		if (((VisualShaderNodeCurveTexture *)links[p_node_id].visual_node)->get_texture().is_valid()) {
			links[p_node_id].curve_editor->set_curve(((VisualShaderNodeCurveTexture *)links[p_node_id].visual_node)->get_texture()->get_curve());
		}
	}
}

int VisualShaderGraphPlugin::get_constant_index(float p_constant) const {
	for (int i = 0; i < MAX_FLOAT_CONST_DEFS; i++) {
		if (Math::is_equal_approx(p_constant, float_constant_defs[i].value)) {
			return i + 1;
		}
	}
	return 0;
}

void VisualShaderGraphPlugin::update_constant(VisualShader::Type p_type, int p_node_id) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id) || !links[p_node_id].const_op) {
		return;
	}
	VisualShaderNodeFloatConstant *float_const = Object::cast_to<VisualShaderNodeFloatConstant>(links[p_node_id].visual_node);
	if (!float_const) {
		return;
	}
	links[p_node_id].const_op->select(get_constant_index(float_const->get_constant()));
	links[p_node_id].graph_node->set_size(Size2(-1, -1));
}

void VisualShaderGraphPlugin::set_expression(VisualShader::Type p_type, int p_node_id, const String &p_expression) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id) || !links[p_node_id].expression_edit) {
		return;
	}
	links[p_node_id].expression_edit->set_text(p_expression);
}

void VisualShaderGraphPlugin::update_node_size(int p_node_id) {
	if (!links.has(p_node_id)) {
		return;
	}
	links[p_node_id].graph_node->set_size(Size2(-1, -1));
}

void VisualShaderGraphPlugin::register_default_input_button(int p_node_id, int p_port_id, Button *p_button) {
	links[p_node_id].input_ports.insert(p_port_id, { p_button });
}

void VisualShaderGraphPlugin::register_constant_option_btn(int p_node_id, OptionButton *p_button) {
	links[p_node_id].const_op = p_button;
}

void VisualShaderGraphPlugin::register_expression_edit(int p_node_id, CodeEdit *p_expression_edit) {
	links[p_node_id].expression_edit = p_expression_edit;
}

void VisualShaderGraphPlugin::register_curve_editor(int p_node_id, CurveEditor *p_curve_editor) {
	links[p_node_id].curve_editor = p_curve_editor;
}

void VisualShaderGraphPlugin::update_uniform_refs() {
	for (Map<int, Link>::Element *E = links.front(); E; E = E->next()) {
		VisualShaderNodeUniformRef *ref = Object::cast_to<VisualShaderNodeUniformRef>(E->get().visual_node);
		if (ref) {
			remove_node(E->get().type, E->key());
			add_node(E->get().type, E->key());
		}
	}
}

VisualShader::Type VisualShaderGraphPlugin::get_shader_type() const {
	return visual_shader->get_shader_type();
}

void VisualShaderGraphPlugin::set_node_position(VisualShader::Type p_type, int p_id, const Vector2 &p_position) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_id)) {
		links[p_id].graph_node->set_offset(p_position);
	}
}

void VisualShaderGraphPlugin::set_node_size(VisualShader::Type p_type, int p_id, const Vector2 &p_size) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_id)) {
		links[p_id].graph_node->set_size(p_size);
	}
}

bool VisualShaderGraphPlugin::is_preview_visible(int p_id) const {
	return links[p_id].preview_visible;
}

void VisualShaderGraphPlugin::clear_links() {
	links.clear();
}

bool VisualShaderGraphPlugin::is_dirty() const {
	return dirty;
}

void VisualShaderGraphPlugin::make_dirty(bool p_enabled) {
	dirty = p_enabled;
}

void VisualShaderGraphPlugin::register_link(VisualShader::Type p_type, int p_id, VisualShaderNode *p_visual_node, GraphNode *p_graph_node) {
	links.insert(p_id, { p_type, p_visual_node, p_graph_node, p_visual_node->get_output_port_for_preview() != -1, -1, Map<int, InputPort>(), Map<int, Port>(), nullptr, nullptr, nullptr, nullptr, nullptr });
}

void VisualShaderGraphPlugin::register_output_port(int p_node_id, int p_port, TextureButton *p_button) {
	links[p_node_id].output_ports.insert(p_port, { p_button });
}

void VisualShaderGraphPlugin::register_uniform_name(int p_node_id, LineEdit *p_uniform_name) {
	links[p_node_id].uniform_name = p_uniform_name;
}

void VisualShaderGraphPlugin::add_node(VisualShader::Type p_type, int p_id) {
	if (p_type != visual_shader->get_shader_type()) {
		return;
	}

	Control *offset;

	static Ref<StyleBoxEmpty> label_style = make_empty_stylebox(2, 1, 2, 1);

	static const Color type_color[6] = {
		Color(0.38, 0.85, 0.96), // scalar (float)
		Color(0.49, 0.78, 0.94), // scalar (int)
		Color(0.84, 0.49, 0.93), // vector
		Color(0.55, 0.65, 0.94), // boolean
		Color(0.96, 0.66, 0.43), // transform
		Color(1.0, 1.0, 0.0), // sampler
	};

	Ref<VisualShaderNode> vsnode = visual_shader->get_node(p_type, p_id);

	Ref<VisualShaderNodeResizableBase> resizable_node = Object::cast_to<VisualShaderNodeResizableBase>(vsnode.ptr());
	bool is_resizable = !resizable_node.is_null();
	Size2 size = Size2(0, 0);

	Ref<VisualShaderNodeGroupBase> group_node = Object::cast_to<VisualShaderNodeGroupBase>(vsnode.ptr());
	bool is_group = !group_node.is_null();

	Ref<VisualShaderNodeExpression> expression_node = Object::cast_to<VisualShaderNodeExpression>(group_node.ptr());
	bool is_expression = !expression_node.is_null();
	String expression = "";

	GraphNode *node = memnew(GraphNode);
	register_link(p_type, p_id, vsnode.ptr(), node);

	if (is_resizable) {
		size = resizable_node->get_size();

		node->set_resizable(true);
		node->connect("resize_request", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_node_resized), varray((int)p_type, p_id));
	}
	if (is_expression) {
		expression = expression_node->get_expression();
	}

	node->set_offset(visual_shader->get_node_position(p_type, p_id));
	node->set_title(vsnode->get_caption());
	node->set_name(itos(p_id));

	if (p_id >= 2) {
		node->set_show_close_button(true);
		node->connect("close_request", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_delete_node_request), varray(p_type, p_id), CONNECT_DEFERRED);
	}

	node->connect("dragged", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_node_dragged), varray(p_id));

	Control *custom_editor = nullptr;
	int port_offset = 0;

	if (is_group) {
		port_offset += 2;
	}

	Ref<VisualShaderNodeUniform> uniform = vsnode;
	if (uniform.is_valid()) {
		VisualShaderEditor::get_singleton()->graph->add_child(node);
		VisualShaderEditor::get_singleton()->_update_created_node(node);

		LineEdit *uniform_name = memnew(LineEdit);
		register_uniform_name(p_id, uniform_name);
		uniform_name->set_text(uniform->get_uniform_name());
		node->add_child(uniform_name);
		uniform_name->connect("text_entered", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_uniform_line_edit_changed), varray(p_id));
		uniform_name->connect("focus_exited", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_uniform_line_edit_focus_out), varray(uniform_name, p_id));

		if (vsnode->get_input_port_count() == 0 && vsnode->get_output_port_count() == 1 && vsnode->get_output_port_name(0) == "") {
			//shortcut
			VisualShaderNode::PortType port_right = vsnode->get_output_port_type(0);
			node->set_slot(0, false, VisualShaderNode::PORT_TYPE_SCALAR, Color(), true, port_right, type_color[port_right]);
			if (!vsnode->is_use_prop_slots()) {
				return;
			}
		}
		port_offset++;
	}

	for (int i = 0; i < VisualShaderEditor::get_singleton()->plugins.size(); i++) {
		vsnode->set_meta("id", p_id);
		vsnode->set_meta("shader_type", (int)p_type);
		custom_editor = VisualShaderEditor::get_singleton()->plugins.write[i]->create_editor(visual_shader, vsnode);
		vsnode->remove_meta("id");
		vsnode->remove_meta("shader_type");
		if (custom_editor) {
			if (vsnode->is_show_prop_names()) {
				custom_editor->call_deferred("_show_prop_names", true);
			}
			break;
		}
	}

	Ref<VisualShaderNodeCurveTexture> curve = vsnode;
	if (curve.is_valid()) {
		if (curve->get_texture().is_valid() && !curve->get_texture()->is_connected("changed", callable_mp(VisualShaderEditor::get_singleton()->get_graph_plugin(), &VisualShaderGraphPlugin::update_curve))) {
			curve->get_texture()->connect("changed", callable_mp(VisualShaderEditor::get_singleton()->get_graph_plugin(), &VisualShaderGraphPlugin::update_curve), varray(p_id));
		}

		HBoxContainer *hbox = memnew(HBoxContainer);
		custom_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hbox->add_child(custom_editor);
		custom_editor = hbox;
	}

	Ref<VisualShaderNodeFloatConstant> float_const = vsnode;
	if (float_const.is_valid()) {
		HBoxContainer *hbox = memnew(HBoxContainer);

		hbox->add_child(custom_editor);
		OptionButton *btn = memnew(OptionButton);
		hbox->add_child(btn);
		register_constant_option_btn(p_id, btn);
		btn->add_item("");
		for (int i = 0; i < MAX_FLOAT_CONST_DEFS; i++) {
			btn->add_item(float_constant_defs[i].name);
		}
		btn->select(get_constant_index(float_const->get_constant()));
		btn->connect("item_selected", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_float_constant_selected), varray(p_id));
		custom_editor = hbox;
	}

	if (custom_editor && !vsnode->is_use_prop_slots() && vsnode->get_output_port_count() > 0 && vsnode->get_output_port_name(0) == "" && (vsnode->get_input_port_count() == 0 || vsnode->get_input_port_name(0) == "")) {
		//will be embedded in first port
	} else if (custom_editor) {
		port_offset++;
		node->add_child(custom_editor);

		if (curve.is_valid()) {
			VisualShaderEditor::get_singleton()->graph->add_child(node);
			VisualShaderEditor::get_singleton()->_update_created_node(node);

			CurveEditor *curve_editor = memnew(CurveEditor);
			node->add_child(curve_editor);
			register_curve_editor(p_id, curve_editor);
			curve_editor->set_custom_minimum_size(Size2(300, 0));
			curve_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			if (curve->get_texture().is_valid()) {
				curve_editor->set_curve(curve->get_texture()->get_curve());
			}

			TextureButton *preview = memnew(TextureButton);
			preview->set_toggle_mode(true);
			preview->set_normal_texture(VisualShaderEditor::get_singleton()->get_theme_icon("GuiVisibilityHidden", "EditorIcons"));
			preview->set_pressed_texture(VisualShaderEditor::get_singleton()->get_theme_icon("GuiVisibilityVisible", "EditorIcons"));
			preview->set_v_size_flags(Control::SIZE_SHRINK_CENTER);

			register_output_port(p_id, 0, preview);

			preview->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_preview_select_port), varray(p_id, 0), CONNECT_DEFERRED);
			custom_editor->add_child(preview);

			VisualShaderNode::PortType port_left = vsnode->get_input_port_type(0);
			VisualShaderNode::PortType port_right = vsnode->get_output_port_type(0);
			node->set_slot(0, true, port_left, type_color[port_left], true, port_right, type_color[port_right]);

			VisualShaderEditor::get_singleton()->call_deferred("_set_node_size", (int)p_type, p_id, size);
		}
		if (vsnode->is_use_prop_slots()) {
			return;
		}
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
			add_input_btn->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_add_input_port), varray(p_id, group_node->get_free_input_port_id(), VisualShaderNode::PORT_TYPE_VECTOR, "input" + itos(group_node->get_free_input_port_id())), CONNECT_DEFERRED);
			hb2->add_child(add_input_btn);

			hb2->add_spacer();

			Button *add_output_btn = memnew(Button);
			add_output_btn->set_text(TTR("Add Output"));
			add_output_btn->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_add_output_port), varray(p_id, group_node->get_free_output_port_id(), VisualShaderNode::PORT_TYPE_VECTOR, "output" + itos(group_node->get_free_output_port_id())), CONNECT_DEFERRED);
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
				if (E->get().to_node == p_id && E->get().to_port == i) {
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
		hb->add_theme_constant_override("separation", 7 * EDSCALE);

		Variant default_value;

		if (valid_left && !port_left_used) {
			default_value = vsnode->get_input_port_default_value(i);
		}

		Button *button = memnew(Button);
		hb->add_child(button);
		register_default_input_button(p_id, i, button);
		button->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_edit_port_default_input), varray(button, p_id, i));
		if (default_value.get_type() != Variant::NIL) { // only a label
			set_input_port_default_value(p_type, p_id, i, default_value);
		} else {
			button->hide();
		}

		if (i == 0 && custom_editor) {
			hb->add_child(custom_editor);
			custom_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		} else {
			if (valid_left) {
				if (is_group) {
					OptionButton *type_box = memnew(OptionButton);
					hb->add_child(type_box);
					type_box->add_item(TTR("Float"));
					type_box->add_item(TTR("Int"));
					type_box->add_item(TTR("Vector"));
					type_box->add_item(TTR("Boolean"));
					type_box->add_item(TTR("Transform"));
					type_box->add_item(TTR("Sampler"));
					type_box->select(group_node->get_input_port_type(i));
					type_box->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
					type_box->connect("item_selected", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_change_input_port_type), varray(p_id, i), CONNECT_DEFERRED);

					LineEdit *name_box = memnew(LineEdit);
					hb->add_child(name_box);
					name_box->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
					name_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
					name_box->set_text(name_left);
					name_box->connect("text_entered", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_change_input_port_name), varray(name_box, p_id, i));
					name_box->connect("focus_exited", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_port_name_focus_out), varray(name_box, p_id, i, false));

					Button *remove_btn = memnew(Button);
					remove_btn->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Remove", "EditorIcons"));
					remove_btn->set_tooltip(TTR("Remove") + " " + name_left);
					remove_btn->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_remove_input_port), varray(p_id, i), CONNECT_DEFERRED);
					hb->add_child(remove_btn);
				} else {
					Label *label = memnew(Label);
					label->set_text(name_left);
					label->add_theme_style_override("normal", label_style); //more compact
					hb->add_child(label);

					if (vsnode->get_input_port_default_hint(i) != "" && !port_left_used) {
						Label *hint_label = memnew(Label);
						hint_label->set_text("[" + vsnode->get_input_port_default_hint(i) + "]");
						hint_label->add_theme_color_override("font_color", VisualShaderEditor::get_singleton()->get_theme_color("font_color_readonly", "TextEdit"));
						hint_label->add_theme_style_override("normal", label_style);
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
					remove_btn->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Remove", "EditorIcons"));
					remove_btn->set_tooltip(TTR("Remove") + " " + name_left);
					remove_btn->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_remove_output_port), varray(p_id, i), CONNECT_DEFERRED);
					hb->add_child(remove_btn);

					LineEdit *name_box = memnew(LineEdit);
					hb->add_child(name_box);
					name_box->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
					name_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
					name_box->set_text(name_right);
					name_box->connect("text_entered", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_change_output_port_name), varray(name_box, p_id, i));
					name_box->connect("focus_exited", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_port_name_focus_out), varray(name_box, p_id, i, true));

					OptionButton *type_box = memnew(OptionButton);
					hb->add_child(type_box);
					type_box->add_item(TTR("Float"));
					type_box->add_item(TTR("Int"));
					type_box->add_item(TTR("Vector"));
					type_box->add_item(TTR("Boolean"));
					type_box->add_item(TTR("Transform"));
					type_box->select(group_node->get_output_port_type(i));
					type_box->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
					type_box->connect("item_selected", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_change_output_port_type), varray(p_id, i), CONNECT_DEFERRED);
				} else {
					Label *label = memnew(Label);
					label->set_text(name_right);
					label->add_theme_style_override("normal", label_style); //more compact
					hb->add_child(label);
				}
			}
		}

		if (valid_right && visual_shader->get_shader_type() == VisualShader::TYPE_FRAGMENT && port_right != VisualShaderNode::PORT_TYPE_TRANSFORM && port_right != VisualShaderNode::PORT_TYPE_SAMPLER) {
			TextureButton *preview = memnew(TextureButton);
			preview->set_toggle_mode(true);
			preview->set_normal_texture(VisualShaderEditor::get_singleton()->get_theme_icon("GuiVisibilityHidden", "EditorIcons"));
			preview->set_pressed_texture(VisualShaderEditor::get_singleton()->get_theme_icon("GuiVisibilityVisible", "EditorIcons"));
			preview->set_v_size_flags(Control::SIZE_SHRINK_CENTER);

			register_output_port(p_id, i, preview);

			preview->connect("pressed", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_preview_select_port), varray(p_id, i), CONNECT_DEFERRED);
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
		show_port_preview(p_type, p_id, vsnode->get_output_port_for_preview());
	}

	offset = memnew(Control);
	offset->set_custom_minimum_size(Size2(0, 4 * EDSCALE));
	node->add_child(offset);

	String error = vsnode->get_warning(visual_shader->get_mode(), p_type);
	if (error != String()) {
		Label *error_label = memnew(Label);
		error_label->add_theme_color_override("font_color", VisualShaderEditor::get_singleton()->get_theme_color("error_color", "Editor"));
		error_label->set_text(error);
		node->add_child(error_label);
	}

	if (is_expression) {
		CodeEdit *expression_box = memnew(CodeEdit);
		Ref<CodeHighlighter> expression_syntax_highlighter;
		expression_syntax_highlighter.instance();
		expression_node->set_control(expression_box, 0);
		node->add_child(expression_box);
		register_expression_edit(p_id, expression_box);

		Color background_color = EDITOR_GET("text_editor/highlighting/background_color");
		Color text_color = EDITOR_GET("text_editor/highlighting/text_color");
		Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");
		Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");
		Color symbol_color = EDITOR_GET("text_editor/highlighting/symbol_color");
		Color function_color = EDITOR_GET("text_editor/highlighting/function_color");
		Color number_color = EDITOR_GET("text_editor/highlighting/number_color");
		Color members_color = EDITOR_GET("text_editor/highlighting/member_variable_color");

		expression_box->set_syntax_highlighter(expression_syntax_highlighter);
		expression_box->add_theme_color_override("background_color", background_color);

		for (List<String>::Element *E = VisualShaderEditor::get_singleton()->keyword_list.front(); E; E = E->next()) {
			expression_syntax_highlighter->add_keyword_color(E->get(), keyword_color);
		}

		expression_box->add_theme_font_override("font", VisualShaderEditor::get_singleton()->get_theme_font("expression", "EditorFonts"));
		expression_box->add_theme_color_override("font_color", text_color);
		expression_syntax_highlighter->set_number_color(number_color);
		expression_syntax_highlighter->set_symbol_color(symbol_color);
		expression_syntax_highlighter->set_function_color(function_color);
		expression_syntax_highlighter->set_member_variable_color(members_color);
		expression_syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
		expression_syntax_highlighter->add_color_region("//", "", comment_color, true);

		expression_box->set_text(expression);
		expression_box->set_context_menu_enabled(false);
		expression_box->set_draw_line_numbers(true);

		expression_box->connect("focus_exited", callable_mp(VisualShaderEditor::get_singleton(), &VisualShaderEditor::_expression_focus_out), varray(expression_box, p_id));
	}

	if (!uniform.is_valid()) {
		VisualShaderEditor::get_singleton()->graph->add_child(node);
		VisualShaderEditor::get_singleton()->_update_created_node(node);
		if (is_resizable) {
			VisualShaderEditor::get_singleton()->call_deferred("_set_node_size", (int)p_type, p_id, size);
		}
	}
}

void VisualShaderGraphPlugin::remove_node(VisualShader::Type p_type, int p_id) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_id)) {
		links[p_id].graph_node->get_parent()->remove_child(links[p_id].graph_node);
		memdelete(links[p_id].graph_node);
		links.erase(p_id);
	}
}

void VisualShaderGraphPlugin::connect_nodes(VisualShader::Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	if (visual_shader->get_shader_type() == p_type) {
		VisualShaderEditor::get_singleton()->graph->connect_node(itos(p_from_node), p_from_port, itos(p_to_node), p_to_port);
		if (links[p_to_node].input_ports.has(p_to_port) && links[p_to_node].input_ports[p_to_port].default_input_button != nullptr) {
			links[p_to_node].input_ports[p_to_port].default_input_button->hide();
		}
	}
}

void VisualShaderGraphPlugin::disconnect_nodes(VisualShader::Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	if (visual_shader->get_shader_type() == p_type) {
		VisualShaderEditor::get_singleton()->graph->disconnect_node(itos(p_from_node), p_from_port, itos(p_to_node), p_to_port);
		if (links[p_to_node].input_ports.has(p_to_port) && links[p_to_node].input_ports[p_to_port].default_input_button != nullptr && links[p_to_node].visual_node->get_input_port_default_value(p_to_port).get_type() != Variant::NIL) {
			links[p_to_node].input_ports[p_to_port].default_input_button->show();
			set_input_port_default_value(p_type, p_to_node, p_to_port, links[p_to_node].visual_node->get_input_port_default_value(p_to_port));
		}
	}
}

VisualShaderGraphPlugin::~VisualShaderGraphPlugin() {
}

/////////////////

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
		graph_plugin->register_shader(visual_shader.ptr());
		if (!visual_shader->is_connected("changed", callable_mp(this, &VisualShaderEditor::_update_preview))) {
			visual_shader->connect("changed", callable_mp(this, &VisualShaderEditor::_update_preview));
		}
#ifndef DISABLE_DEPRECATED
		String version = VERSION_BRANCH;
		if (visual_shader->get_version() != version) {
			visual_shader->update_version(version);
		}
#endif
		visual_shader->set_graph_offset(graph->get_scroll_ofs() / EDSCALE);
		_set_mode(visual_shader->get_mode());
	} else {
		if (visual_shader.is_valid()) {
			if (visual_shader->is_connected("changed", callable_mp(this, &VisualShaderEditor::_update_preview))) {
				visual_shader->disconnect("changed", callable_mp(this, &VisualShaderEditor::_update_preview));
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
			_update_graph();
		}
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

void VisualShaderEditor::add_custom_type(const String &p_name, const Ref<Script> &p_script, const String &p_description, int p_return_icon_type, const String &p_category, bool p_highend) {
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
	ao.highend = p_highend;
	ao.is_custom = true;

	bool begin = false;
	String root = p_category.split("/")[0];

	for (int i = 0; i < add_options.size(); i++) {
		if (add_options[i].is_custom) {
			if (add_options[i].category == root) {
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
			case 0: // Vertex or Emit
				current_mode = 1;
				break;
			case 1: // Fragment or Process
				current_mode = 2;
				break;
			case 2: // Light or End
				current_mode = 4;
				break;
			default:
				break;
		}
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
			ref->set_script(script);

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

			String subcategory = "";
			if (ref->has_method("_get_subcategory")) {
				subcategory = (String)ref->call("_get_subcategory");
			}

			bool highend = false;
			if (ref->has_method("_is_highend")) {
				highend = (bool)ref->call("_is_highend");
			}

			Dictionary dict;
			dict["name"] = name;
			dict["script"] = script;
			dict["description"] = description;
			dict["return_icon_type"] = return_icon_type;

			category = category.rstrip("/");
			category = category.lstrip("/");
			category = "Addons/" + category;
			if (subcategory != "") {
				category += "/" + subcategory;
			}

			dict["category"] = category;
			dict["highend"] = highend;

			String key;
			key = category + "/" + name;

			added[key] = dict;
		}
	}

	Array keys = added.keys();
	keys.sort();

	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys.get(i);

		const Dictionary &value = (Dictionary)added[key];

		add_custom_type(value["name"], value["script"], value["description"], value["return_icon_type"], value["category"], value["highend"]);
	}

	_update_options_menu();
}

String VisualShaderEditor::_get_description(int p_idx) {
	return add_options[p_idx].description;
}

void VisualShaderEditor::_update_options_menu() {
	node_desc->set_text("");
	members_dialog->get_ok()->set_disabled(true);

	members->clear();
	TreeItem *root = members->create_item();

	String filter = node_filter->get_text().strip_edges();
	bool use_filter = !filter.empty();

	bool is_first_item = true;

	Color unsupported_color = get_theme_color("error_color", "Editor");
	Color supported_color = get_theme_color("warning_color", "Editor");

	static bool low_driver = ProjectSettings::get_singleton()->get("rendering/quality/driver/driver_name") == "GLES2";

	Map<String, TreeItem *> folders;

	int current_func = -1;

	if (!visual_shader.is_null()) {
		current_func = visual_shader->get_mode();
	}

	Vector<AddOption> custom_options;
	Vector<AddOption> embedded_options;

	for (int i = 0; i < add_options.size(); i++) {
		if (!use_filter || add_options[i].name.findn(filter) != -1) {
			if ((add_options[i].func != current_func && add_options[i].func != -1) || !_is_available(add_options[i].mode)) {
				continue;
			}
			const_cast<AddOption &>(add_options[i]).temp_idx = i; // save valid id
			if (add_options[i].is_custom) {
				custom_options.push_back(add_options[i]);
			} else {
				embedded_options.push_back(add_options[i]);
			}
		}
	}
	Vector<AddOption> options;
	SortArray<AddOption, _OptionComparator> sorter;
	sorter.sort(custom_options.ptrw(), custom_options.size());

	options.append_array(custom_options);
	options.append_array(embedded_options);

	for (int i = 0; i < options.size(); i++) {
		String path = options[i].category;
		Vector<String> subfolders = path.split("/");
		TreeItem *category = nullptr;

		if (!folders.has(path)) {
			category = root;
			String path_temp = "";
			for (int j = 0; j < subfolders.size(); j++) {
				path_temp += subfolders[j];
				if (!folders.has(path_temp)) {
					category = members->create_item(category);
					category->set_selectable(0, false);
					category->set_collapsed(!use_filter);
					category->set_text(0, subfolders[j]);
					folders.insert(path_temp, category);
				} else {
					category = folders[path_temp];
				}
			}
		} else {
			category = folders[path];
		}

		TreeItem *item = members->create_item(category);
		if (options[i].highend && low_driver) {
			item->set_custom_color(0, unsupported_color);
		} else if (options[i].highend) {
			item->set_custom_color(0, supported_color);
		}
		item->set_text(0, options[i].name);
		if (is_first_item && use_filter) {
			item->select(0);
			node_desc->set_text(options[i].description);
			is_first_item = false;
		}
		switch (options[i].return_type) {
			case VisualShaderNode::PORT_TYPE_SCALAR:
				item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_theme_icon("float", "EditorIcons"));
				break;
			case VisualShaderNode::PORT_TYPE_SCALAR_INT:
				item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_theme_icon("int", "EditorIcons"));
				break;
			case VisualShaderNode::PORT_TYPE_VECTOR:
				item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Vector3", "EditorIcons"));
				break;
			case VisualShaderNode::PORT_TYPE_BOOLEAN:
				item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_theme_icon("bool", "EditorIcons"));
				break;
			case VisualShaderNode::PORT_TYPE_TRANSFORM:
				item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Transform", "EditorIcons"));
				break;
			case VisualShaderNode::PORT_TYPE_SAMPLER:
				item->set_icon(0, EditorNode::get_singleton()->get_gui_base()->get_theme_icon("ImageTexture", "EditorIcons"));
				break;
			default:
				break;
		}
		item->set_meta("id", options[i].temp_idx);
	}
}

void VisualShaderEditor::_set_mode(int p_which) {
	if (p_which == VisualShader::MODE_PARTICLES) {
		edit_type_standart->set_visible(false);
		edit_type_particles->set_visible(true);
		edit_type = edit_type_particles;
		particles_mode = true;
	} else {
		edit_type_particles->set_visible(false);
		edit_type_standart->set_visible(true);
		edit_type = edit_type_standart;
		particles_mode = false;
	}
	visual_shader->set_shader_type(get_current_shader_type());
}

Size2 VisualShaderEditor::get_minimum_size() const {
	return Size2(10, 200);
}

void VisualShaderEditor::_draw_color_over_button(Object *obj, Color p_color) {
	Button *button = Object::cast_to<Button>(obj);
	if (!button) {
		return;
	}

	Ref<StyleBox> normal = get_theme_stylebox("normal", "Button");
	button->draw_rect(Rect2(normal->get_offset(), button->get_size() - normal->get_minimum_size()), p_color);
}

void VisualShaderEditor::_update_created_node(GraphNode *node) {
	if (EditorSettings::get_singleton()->get("interface/theme/use_graph_node_headers")) {
		Ref<StyleBoxFlat> sb = node->get_theme_stylebox("frame", "GraphNode");
		Color c = sb->get_border_color();
		Color ic;
		Color mono_color;
		if (((c.r + c.g + c.b) / 3) < 0.7) {
			mono_color = Color(1.0, 1.0, 1.0);
			ic = Color(0.0, 0.0, 0.0, 0.7);
		} else {
			mono_color = Color(0.0, 0.0, 0.0);
			ic = Color(1.0, 1.0, 1.0, 0.7);
		}
		mono_color.a = 0.85;
		c = mono_color;

		node->add_theme_color_override("title_color", c);
		c.a = 0.7;
		node->add_theme_color_override("close_color", c);
		node->add_theme_color_override("resizer_color", ic);
	}
}

void VisualShaderEditor::_update_uniforms(bool p_update_refs) {
	VisualShaderNodeUniformRef::clear_uniforms();

	for (int t = 0; t < VisualShader::TYPE_MAX; t++) {
		Vector<int> tnodes = visual_shader->get_node_list((VisualShader::Type)t);
		for (int i = 0; i < tnodes.size(); i++) {
			Ref<VisualShaderNode> vsnode = visual_shader->get_node((VisualShader::Type)t, tnodes[i]);
			Ref<VisualShaderNodeUniform> uniform = vsnode;

			if (uniform.is_valid()) {
				Ref<VisualShaderNodeFloatUniform> float_uniform = vsnode;
				Ref<VisualShaderNodeIntUniform> int_uniform = vsnode;
				Ref<VisualShaderNodeVec3Uniform> vec3_uniform = vsnode;
				Ref<VisualShaderNodeColorUniform> color_uniform = vsnode;
				Ref<VisualShaderNodeBooleanUniform> bool_uniform = vsnode;
				Ref<VisualShaderNodeTransformUniform> transform_uniform = vsnode;

				VisualShaderNodeUniformRef::UniformType uniform_type;
				if (float_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_FLOAT;
				} else if (int_uniform.is_valid()) {
					uniform_type = VisualShaderNodeUniformRef::UniformType::UNIFORM_TYPE_INT;
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
	if (p_update_refs) {
		graph_plugin->update_uniform_refs();
	}
}

void VisualShaderEditor::_update_uniform_refs(Set<String> &p_deleted_names) {
	for (int i = 0; i < VisualShader::TYPE_MAX; i++) {
		VisualShader::Type type = VisualShader::Type(i);

		Vector<int> nodes = visual_shader->get_node_list(type);
		for (int j = 0; j < nodes.size(); j++) {
			if (j > 0) {
				Ref<VisualShaderNodeUniformRef> ref = visual_shader->get_node(type, nodes[j]);
				if (ref.is_valid()) {
					if (p_deleted_names.has(ref->get_uniform_name())) {
						undo_redo->add_do_method(ref.ptr(), "set_uniform_name", "[None]");
						undo_redo->add_undo_method(ref.ptr(), "set_uniform_name", ref->get_uniform_name());
						undo_redo->add_do_method(graph_plugin.ptr(), "update_node", VisualShader::Type(i), nodes[j]);
						undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", VisualShader::Type(i), nodes[j]);
					}
				}
			}
		}
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

	VisualShader::Type type = get_current_shader_type();

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

	List<VisualShader::Connection> connections;
	visual_shader->get_node_connections(type, &connections);
	graph_plugin->set_connections(connections);

	Vector<int> nodes = visual_shader->get_node_list(type);

	_update_uniforms(false);

	graph_plugin->clear_links();
	graph_plugin->make_dirty(true);

	for (int n_i = 0; n_i < nodes.size(); n_i++) {
		graph_plugin->add_node(type, nodes[n_i]);
	}

	graph_plugin->make_dirty(false);

	for (List<VisualShader::Connection>::Element *E = connections.front(); E; E = E->next()) {
		int from = E->get().from_node;
		int from_idx = E->get().from_port;
		int to = E->get().to_node;
		int to_idx = E->get().to_port;

		graph->connect_node(itos(from), from_idx, itos(to), to_idx);
	}
}

VisualShader::Type VisualShaderEditor::get_current_shader_type() const {
	VisualShader::Type type;
	if (particles_mode) {
		type = VisualShader::Type(edit_type->get_selected() + 3);
	} else {
		type = VisualShader::Type(edit_type->get_selected());
	}
	return type;
}

void VisualShaderEditor::_add_input_port(int p_node, int p_port, int p_port_type, const String &p_name) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeExpression> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Add Input Port"));
	undo_redo->add_do_method(node.ptr(), "add_input_port", p_port, p_port_type, p_name);
	undo_redo->add_undo_method(node.ptr(), "remove_input_port", p_port);
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_add_output_port(int p_node, int p_port, int p_port_type, const String &p_name) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Add Output Port"));
	undo_redo->add_do_method(node.ptr(), "add_output_port", p_port, p_port_type, p_name);
	undo_redo->add_undo_method(node.ptr(), "remove_output_port", p_port);
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_input_port_type(int p_type, int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Change Input Port Type"));
	undo_redo->add_do_method(node.ptr(), "set_input_port_type", p_port, p_type);
	undo_redo->add_undo_method(node.ptr(), "set_input_port_type", p_port, node->get_input_port_type(p_port));
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_output_port_type(int p_type, int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Change Output Port Type"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_type", p_port, p_type);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_type", p_port, node->get_output_port_type(p_port));
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_input_port_name(const String &p_text, Object *line_edit, int p_node_id, int p_port_id) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	undo_redo->create_action(TTR("Change Input Port Name"));
	undo_redo->add_do_method(node.ptr(), "set_input_port_name", p_port_id, p_text);
	undo_redo->add_undo_method(node.ptr(), "set_input_port_name", p_port_id, node->get_input_port_name(p_port_id));
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node_id);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node_id);
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_output_port_name(const String &p_text, Object *line_edit, int p_node_id, int p_port_id) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	undo_redo->create_action(TTR("Change Output Port Name"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_name", p_port_id, p_text);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_name", p_port_id, node->get_output_port_name(p_port_id));
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node_id);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node_id);
	undo_redo->commit_action();
}

void VisualShaderEditor::_remove_input_port(int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Remove Input Port"));

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

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_port, to_node, to_port);
			} else if (to_port > p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_port, to_node, to_port);

				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port - 1);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port - 1);

				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_port, to_node, to_port - 1);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port - 1);
			}
		}
	}

	undo_redo->add_do_method(node.ptr(), "remove_input_port", p_port);
	undo_redo->add_undo_method(node.ptr(), "add_input_port", p_port, (int)node->get_input_port_type(p_port), node->get_input_port_name(p_port));

	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);

	undo_redo->commit_action();
}

void VisualShaderEditor::_remove_output_port(int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Remove Output Port"));

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

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_port, to_node, to_port);
			} else if (from_port > p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port, to_node, to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_port, to_node, to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_port, to_node, to_port);

				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, from_node, from_port - 1, to_node, to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_port - 1, to_node, to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_port - 1, to_node, to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_port - 1, to_node, to_port);
			}
		}
	}

	undo_redo->add_do_method(node.ptr(), "remove_output_port", p_port);
	undo_redo->add_undo_method(node.ptr(), "add_output_port", p_port, (int)node->get_output_port_type(p_port), node->get_output_port_name(p_port));

	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);

	undo_redo->commit_action();
}

void VisualShaderEditor::_expression_focus_out(Object *code_edit, int p_node) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeExpression> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	CodeEdit *expression_box = Object::cast_to<CodeEdit>(code_edit);

	if (node->get_expression() == expression_box->get_text()) {
		return;
	}

	undo_redo->create_action(TTR("Set VisualShader Expression"));
	undo_redo->add_do_method(node.ptr(), "set_expression", expression_box->get_text());
	undo_redo->add_undo_method(node.ptr(), "set_expression", node->get_expression());
	undo_redo->add_do_method(graph_plugin.ptr(), "set_expression", type, p_node, expression_box->get_text());
	undo_redo->add_undo_method(graph_plugin.ptr(), "set_expression", type, p_node, node->get_expression());
	undo_redo->commit_action();
}

void VisualShaderEditor::_set_node_size(int p_type, int p_node, const Vector2 &p_size) {
	VisualShader::Type type = VisualShader::Type(p_type);
	Ref<VisualShaderNodeResizableBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	Size2 size = p_size;
	if (!node->is_allow_v_resize()) {
		size.y = 0;
	}

	node->set_size(size);

	if (get_current_shader_type() == type) {
		Ref<VisualShaderNodeExpression> expression_node = Object::cast_to<VisualShaderNodeExpression>(node.ptr());
		Control *text_box = nullptr;
		if (!expression_node.is_null()) {
			text_box = expression_node->get_control(0);
			if (text_box) {
				text_box->set_custom_minimum_size(Size2(0, 0));
			}
		}

		GraphNode *gn = nullptr;
		Node *node2 = graph->get_node(itos(p_node));
		gn = Object::cast_to<GraphNode>(node2);
		if (!gn) {
			return;
		}

		gn->set_custom_minimum_size(size);
		gn->set_size(Size2(1, 1));

		if (!expression_node.is_null() && text_box) {
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
}

void VisualShaderEditor::_node_resized(const Vector2 &p_new_size, int p_type, int p_node) {
	Ref<VisualShaderNodeResizableBase> node = visual_shader->get_node(VisualShader::Type(p_type), p_node);
	if (node.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Resize VisualShader Node"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(this, "_set_node_size", p_type, p_node, p_new_size);
	undo_redo->add_undo_method(this, "_set_node_size", p_type, p_node, node->get_size());
	undo_redo->commit_action();
}

void VisualShaderEditor::_preview_select_port(int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNode> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}
	int prev_port = node->get_output_port_for_preview();
	if (node->get_output_port_for_preview() == p_port) {
		p_port = -1; //toggle it
	}
	undo_redo->create_action(p_port == -1 ? TTR("Hide Port Preview") : TTR("Show Port Preview"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", p_port);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", prev_port);
	undo_redo->add_do_method(graph_plugin.ptr(), "show_port_preview", (int)type, p_node, p_port);
	undo_redo->add_undo_method(graph_plugin.ptr(), "show_port_preview", (int)type, p_node, prev_port);
	undo_redo->commit_action();
}

void VisualShaderEditor::_uniform_line_edit_changed(const String &p_text, int p_node_id) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNodeUniform> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	String validated_name = visual_shader->validate_uniform_name(p_text, node);

	if (validated_name == node->get_uniform_name()) {
		return;
	}

	undo_redo->create_action(TTR("Set Uniform Name"));
	undo_redo->add_do_method(node.ptr(), "set_uniform_name", validated_name);
	undo_redo->add_undo_method(node.ptr(), "set_uniform_name", node->get_uniform_name());
	undo_redo->add_do_method(graph_plugin.ptr(), "set_uniform_name", type, p_node_id, validated_name);
	undo_redo->add_undo_method(graph_plugin.ptr(), "set_uniform_name", type, p_node_id, node->get_uniform_name());

	undo_redo->add_do_method(this, "_update_uniforms", true);
	undo_redo->add_undo_method(this, "_update_uniforms", true);

	Set<String> changed_names;
	changed_names.insert(node->get_uniform_name());
	_update_uniform_refs(changed_names);

	undo_redo->commit_action();
}

void VisualShaderEditor::_uniform_line_edit_focus_out(Object *line_edit, int p_node_id) {
	_uniform_line_edit_changed(Object::cast_to<LineEdit>(line_edit)->get_text(), p_node_id);
}

void VisualShaderEditor::_port_name_focus_out(Object *line_edit, int p_node_id, int p_port_id, bool p_output) {
	VisualShader::Type type = get_current_shader_type();

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
	VisualShader::Type type = get_current_shader_type();

	Variant value = property_editor->get_variant();
	Ref<VisualShaderNode> vsn = visual_shader->get_node(type, editing_node);
	ERR_FAIL_COND(!vsn.is_valid());

	undo_redo->create_action(TTR("Set Input Default Port"));
	undo_redo->add_do_method(vsn.ptr(), "set_input_port_default_value", editing_port, value);
	undo_redo->add_undo_method(vsn.ptr(), "set_input_port_default_value", editing_port, vsn->get_input_port_default_value(editing_port));
	undo_redo->add_do_method(graph_plugin.ptr(), "set_input_port_default_value", type, editing_node, editing_port, value);
	undo_redo->add_undo_method(graph_plugin.ptr(), "set_input_port_default_value", type, editing_node, editing_port, vsn->get_input_port_default_value(editing_port));
	undo_redo->commit_action();

	property_editor->hide();
}

void VisualShaderEditor::_edit_port_default_input(Object *p_button, int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNode> vsn = visual_shader->get_node(type, p_node);

	Button *button = Object::cast_to<Button>(p_button);
	ERR_FAIL_COND(!button);
	Variant value = vsn->get_input_port_default_value(p_port);
	property_editor->set_position(button->get_screen_position() + Vector2(0, button->get_size().height));
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

void VisualShaderEditor::_add_cubemap_node(const String &p_path) {
	VisualShaderNodeCubemap *cubemap = (VisualShaderNodeCubemap *)_add_node(cubemap_node_option_idx, -1);
	cubemap->set_cube_map(ResourceLoader::load(p_path));
}

void VisualShaderEditor::_add_texture2d_node(const String &p_path) {
	VisualShaderNodeTexture *texture2d = (VisualShaderNodeTexture *)_add_node(texture2d_node_option_idx, -1);
	texture2d->set_texture(ResourceLoader::load(p_path));
}

void VisualShaderEditor::_add_texture2d_array_node(const String &p_path) {
	VisualShaderNodeTexture2DArray *texture2d_array = (VisualShaderNodeTexture2DArray *)_add_node(texture2d_array_node_option_idx, -1);
	texture2d_array->set_texture_array(ResourceLoader::load(p_path));
}

void VisualShaderEditor::_add_texture3d_node(const String &p_path) {
	VisualShaderNodeTexture3D *texture3d = (VisualShaderNodeTexture3D *)_add_node(texture3d_node_option_idx, -1);
	texture3d->set_texture(ResourceLoader::load(p_path));
}

void VisualShaderEditor::_add_curve_node(const String &p_path) {
	VisualShaderNodeCurveTexture *curve = (VisualShaderNodeCurveTexture *)_add_node(curve_node_option_idx, -1);
	curve->set_texture(ResourceLoader::load(p_path));
}

VisualShaderNode *VisualShaderEditor::_add_node(int p_idx, int p_op_idx) {
	ERR_FAIL_INDEX_V(p_idx, add_options.size(), nullptr);

	Ref<VisualShaderNode> vsnode;

	bool is_custom = add_options[p_idx].is_custom;

	if (!is_custom && add_options[p_idx].type != String()) {
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instance(add_options[p_idx].type));
		ERR_FAIL_COND_V(!vsn, nullptr);

		VisualShaderNodeFloatConstant *constant = Object::cast_to<VisualShaderNodeFloatConstant>(vsn);

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

			VisualShaderNodeFloatOp *floatOp = Object::cast_to<VisualShaderNodeFloatOp>(vsn);

			if (floatOp) {
				floatOp->set_operator((VisualShaderNodeFloatOp::Operator)p_op_idx);
			}

			VisualShaderNodeIntOp *intOp = Object::cast_to<VisualShaderNodeIntOp>(vsn);

			if (intOp) {
				intOp->set_operator((VisualShaderNodeIntOp::Operator)p_op_idx);
			}

			VisualShaderNodeFloatFunc *floatFunc = Object::cast_to<VisualShaderNodeFloatFunc>(vsn);

			if (floatFunc) {
				floatFunc->set_function((VisualShaderNodeFloatFunc::Function)p_op_idx);
			}

			VisualShaderNodeIntFunc *intFunc = Object::cast_to<VisualShaderNodeIntFunc>(vsn);

			if (intFunc) {
				intFunc->set_function((VisualShaderNodeIntFunc::Function)p_op_idx);
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

			VisualShaderNodeMultiplyAdd *fmaFunc = Object::cast_to<VisualShaderNodeMultiplyAdd>(vsn);

			if (fmaFunc) {
				fmaFunc->set_op_type((VisualShaderNodeMultiplyAdd::OpType)p_op_idx);
			}
		}

		vsnode = Ref<VisualShaderNode>(vsn);
	} else {
		ERR_FAIL_COND_V(add_options[p_idx].script.is_null(), nullptr);
		String base_type = add_options[p_idx].script->get_instance_base_type();
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instance(base_type));
		ERR_FAIL_COND_V(!vsn, nullptr);
		vsnode = Ref<VisualShaderNode>(vsn);
		vsnode->set_script(add_options[p_idx].script);
	}

	Point2 position = graph->get_scroll_ofs();

	if (saved_node_pos_dirty) {
		position += saved_node_pos;
	} else {
		position += graph->get_size() * 0.5;
		position /= EDSCALE;
	}
	saved_node_pos_dirty = false;

	VisualShader::Type type = get_current_shader_type();

	int id_to_use = visual_shader->get_valid_node_id(type);

	undo_redo->create_action(TTR("Add Node to Visual Shader"));
	undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, vsnode, position, id_to_use);
	undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_to_use);
	undo_redo->add_do_method(graph_plugin.ptr(), "add_node", type, id_to_use);
	undo_redo->add_undo_method(graph_plugin.ptr(), "remove_node", type, id_to_use);

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
				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, _from_node, _from_slot, to_node, to_slot);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, _from_node, _from_slot, to_node, to_slot);
			}
		}
	} else if (from_node != -1 && from_slot != -1) {
		if (vsnode->get_input_port_count() > 0) {
			int _to_node = id_to_use;
			int _to_slot = 0;

			if (visual_shader->is_port_types_compatible(visual_shader->get_node(type, from_node)->get_output_port_type(from_slot), vsnode->get_input_port_type(_to_slot))) {
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
			}
		}
	}

	VisualShaderNodeUniform *uniform = Object::cast_to<VisualShaderNodeUniform>(vsnode.ptr());
	if (uniform) {
		undo_redo->add_do_method(this, "_update_uniforms", true);
		undo_redo->add_undo_method(this, "_update_uniforms", true);
	}

	VisualShaderNodeCurveTexture *curve = Object::cast_to<VisualShaderNodeCurveTexture>(vsnode.ptr());
	if (curve) {
		graph_plugin->call_deferred("update_curve", id_to_use);
	}

	undo_redo->commit_action();
	return vsnode.ptr();
}

void VisualShaderEditor::_node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_node) {
	VisualShader::Type type = get_current_shader_type();
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
		undo_redo->add_do_method(graph_plugin.ptr(), "set_node_position", E->get().type, E->get().node, E->get().to);
		undo_redo->add_undo_method(graph_plugin.ptr(), "set_node_position", E->get().type, E->get().node, E->get().from);
	}

	drag_buffer.clear();
	undo_redo->commit_action();
}

void VisualShaderEditor::_connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {
	VisualShader::Type type = get_current_shader_type();

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
			undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
		}
	}

	undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->commit_action();
}

void VisualShaderEditor::_disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {
	graph->disconnect_node(p_from, p_from_index, p_to, p_to_index);

	VisualShader::Type type = get_current_shader_type();

	int from = p_from.to_int();
	int to = p_to.to_int();

	undo_redo->create_action(TTR("Nodes Disconnected"));
	undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->commit_action();
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

void VisualShaderEditor::_delete_nodes(int p_type, const List<int> &p_nodes) {
	VisualShader::Type type = VisualShader::Type(p_type);
	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	for (const List<int>::Element *F = p_nodes.front(); F; F = F->next()) {
		for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
			if (E->get().from_node == F->get() || E->get().to_node == F->get()) {
				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}
		}
	}

	Set<String> uniform_names;

	for (const List<int>::Element *F = p_nodes.front(); F; F = F->next()) {
		Ref<VisualShaderNode> node = visual_shader->get_node(type, F->get());

		undo_redo->add_do_method(visual_shader.ptr(), "remove_node", type, F->get());
		undo_redo->add_undo_method(visual_shader.ptr(), "add_node", type, node, visual_shader->get_node_position(type, F->get()), F->get());
		undo_redo->add_undo_method(graph_plugin.ptr(), "add_node", type, F->get());

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

		VisualShaderNodeUniform *uniform = Object::cast_to<VisualShaderNodeUniform>(node.ptr());
		if (uniform) {
			uniform_names.insert(uniform->get_uniform_name());
		}
	}

	List<VisualShader::Connection> used_conns;
	for (const List<int>::Element *F = p_nodes.front(); F; F = F->next()) {
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
					undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
					used_conns.push_back(E->get());
				}
			}
		}
	}

	// delete nodes from the graph
	for (const List<int>::Element *F = p_nodes.front(); F; F = F->next()) {
		undo_redo->add_do_method(graph_plugin.ptr(), "remove_node", type, F->get());
	}

	// update uniform refs if any uniform has been deleted
	if (uniform_names.size() > 0) {
		undo_redo->add_do_method(this, "_update_uniforms", true);
		undo_redo->add_undo_method(this, "_update_uniforms", true);

		_update_uniform_refs(uniform_names);
	}
}

void VisualShaderEditor::_delete_node_request(int p_type, int p_node) {
	List<int> to_erase;
	to_erase.push_back(p_node);

	undo_redo->create_action(TTR("Delete VisualShader Node"));
	_delete_nodes(p_type, to_erase);
	undo_redo->commit_action();
}

void VisualShaderEditor::_delete_nodes_request() {
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

	undo_redo->create_action(TTR("Delete VisualShader Node(s)"));
	_delete_nodes(get_current_shader_type(), to_erase);
	undo_redo->commit_action();
}

void VisualShaderEditor::_node_selected(Object *p_node) {
	VisualShader::Type type = get_current_shader_type();

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
		List<int> to_change;
		for (int i = 0; i < graph->get_child_count(); i++) {
			GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
			if (gn) {
				if (gn->is_selected() && gn->is_close_button_visible()) {
					to_change.push_back(gn->get_name().operator String().to_int());
				}
			}
		}
		if (to_change.empty() && copy_nodes_buffer.empty()) {
			_show_members_dialog(true);
		} else {
			popup_menu->set_item_disabled(NodeMenuOptions::COPY, to_change.empty());
			popup_menu->set_item_disabled(NodeMenuOptions::PASTE, copy_nodes_buffer.empty());
			popup_menu->set_item_disabled(NodeMenuOptions::DELETE, to_change.empty());
			popup_menu->set_item_disabled(NodeMenuOptions::DUPLICATE, to_change.empty());
			menu_point = graph->get_local_mouse_position();
			Point2 gpos = Input::get_singleton()->get_mouse_position();
			popup_menu->set_position(gpos);
			popup_menu->popup();
		}
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
	Size2 window_size = DisplayServer::get_singleton()->window_get_size();
	Rect2 dialog_rect = Rect2(members_dialog->get_position(), members_dialog->get_size());
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
	if (ie.is_valid() && (ie->get_keycode() == KEY_UP ||
								 ie->get_keycode() == KEY_DOWN ||
								 ie->get_keycode() == KEY_ENTER ||
								 ie->get_keycode() == KEY_KP_ENTER)) {
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
		highend_label->set_modulate(get_theme_color("vulkan_color", "Editor"));

		error_panel->add_theme_style_override("panel", get_theme_stylebox("bg", "Tree"));
		error_label->add_theme_color_override("font_color", get_theme_color("error_color", "Editor"));

		node_filter->set_right_icon(Control::get_theme_icon("Search", "EditorIcons"));

		preview_shader->set_icon(Control::get_theme_icon("Shader", "EditorIcons"));

		{
			Color background_color = EDITOR_GET("text_editor/highlighting/background_color");
			Color text_color = EDITOR_GET("text_editor/highlighting/text_color");
			Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");
			Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");
			Color symbol_color = EDITOR_GET("text_editor/highlighting/symbol_color");
			Color function_color = EDITOR_GET("text_editor/highlighting/function_color");
			Color number_color = EDITOR_GET("text_editor/highlighting/number_color");
			Color members_color = EDITOR_GET("text_editor/highlighting/member_variable_color");

			preview_text->add_theme_color_override("background_color", background_color);

			for (List<String>::Element *E = keyword_list.front(); E; E = E->next()) {
				syntax_highlighter->add_keyword_color(E->get(), keyword_color);
			}

			preview_text->add_theme_font_override("font", get_theme_font("expression", "EditorFonts"));
			preview_text->add_theme_color_override("font_color", text_color);
			syntax_highlighter->set_number_color(number_color);
			syntax_highlighter->set_symbol_color(symbol_color);
			syntax_highlighter->set_function_color(function_color);
			syntax_highlighter->set_member_variable_color(members_color);
			syntax_highlighter->clear_color_regions();
			syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
			syntax_highlighter->add_color_region("//", "", comment_color, true);

			error_text->add_theme_font_override("font", get_theme_font("status_source", "EditorFonts"));
			error_text->add_theme_color_override("font_color", get_theme_color("error_color", "Editor"));
		}

		tools->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Tools", "EditorIcons"));

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
		undo_redo->add_do_method(graph_plugin.ptr(), "add_node", type, id_from);

		// duplicate size, inputs and outputs if node is group
		Ref<VisualShaderNodeGroupBase> group = Object::cast_to<VisualShaderNodeGroupBase>(node.ptr());
		if (!group.is_null()) {
			undo_redo->add_do_method(dupli.ptr(), "set_size", group->get_size());
			undo_redo->add_do_method(graph_plugin.ptr(), "set_node_size", type, id_from, group->get_size());
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
			undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, connection_remap[E->get().from_node], E->get().from_port, connection_remap[E->get().to_node], E->get().to_port);
			undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, connection_remap[E->get().from_node], E->get().from_port, connection_remap[E->get().to_node], E->get().to_port);
			undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, connection_remap[E->get().from_node], E->get().from_port, connection_remap[E->get().to_node], E->get().to_port);
		}
	}

	id_from = base_id;
	for (List<int>::Element *E = r_nodes.front(); E; E = E->next()) {
		undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_from);
		undo_redo->add_undo_method(graph_plugin.ptr(), "remove_node", type, id_from);
		id_from++;
	}

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
	int type = get_current_shader_type();

	List<int> nodes;
	Set<int> excluded;

	_dup_copy_nodes(type, nodes, excluded);

	if (nodes.empty()) {
		return;
	}

	undo_redo->create_action(TTR("Duplicate VisualShader Node(s)"));

	_dup_paste_nodes(type, type, nodes, excluded, Vector2(10, 10) * EDSCALE, true);
}

void VisualShaderEditor::_copy_nodes() {
	copy_type = get_current_shader_type();

	_clear_buffer();

	_dup_copy_nodes(copy_type, copy_nodes_buffer, copy_nodes_excluded_buffer);
}

void VisualShaderEditor::_paste_nodes(bool p_use_custom_position, const Vector2 &p_custom_position) {
	if (copy_nodes_buffer.empty()) {
		return;
	}

	int type = get_current_shader_type();

	float scale = graph->get_zoom();

	Vector2 mpos;
	if (p_use_custom_position) {
		mpos = p_custom_position;
	} else {
		mpos = graph->get_local_mouse_position();
	}

	undo_redo->create_action(TTR("Paste VisualShader Node(s)"));

	_dup_paste_nodes(type, copy_type, copy_nodes_buffer, copy_nodes_excluded_buffer, (graph->get_scroll_ofs() / scale + mpos / scale - selection_center), false);

	_dup_update_excluded(type, copy_nodes_excluded_buffer); // to prevent selection of previous copies at new paste
}

void VisualShaderEditor::_mode_selected(int p_id) {
	visual_shader->set_shader_type(particles_mode ? VisualShader::Type(p_id + 3) : VisualShader::Type(p_id));
	_update_options_menu();
	_update_graph();
}

void VisualShaderEditor::_input_select_item(Ref<VisualShaderNodeInput> p_input, String p_name) {
	String prev_name = p_input->get_input_name();

	if (p_name == prev_name) {
		return;
	}

	bool type_changed = p_input->get_input_type_by_name(p_name) != p_input->get_input_type_by_name(prev_name);

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("Visual Shader Input Type Changed"));

	undo_redo->add_do_method(p_input.ptr(), "set_input_name", p_name);
	undo_redo->add_undo_method(p_input.ptr(), "set_input_name", prev_name);

	// update output port
	for (int type_id = 0; type_id < VisualShader::TYPE_MAX; type_id++) {
		VisualShader::Type type = VisualShader::Type(type_id);
		int id = visual_shader->find_node_id(type, p_input);
		if (id != VisualShader::NODE_ID_INVALID) {
			if (type_changed) {
				List<VisualShader::Connection> conns;
				visual_shader->get_node_connections(type, &conns);
				for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
					if (E->get().from_node == id) {
						if (visual_shader->is_port_types_compatible(p_input->get_input_type_by_name(p_name), visual_shader->get_node(type, E->get().to_node)->get_input_port_type(E->get().to_port))) {
							undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
							undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
							continue;
						}
						undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
						undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
						undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
						undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
					}
				}
			}
			undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type_id, id);
			undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type_id, id);
			break;
		}
	}

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

	// update output port
	for (int type_id = 0; type_id < VisualShader::TYPE_MAX; type_id++) {
		VisualShader::Type type = VisualShader::Type(type_id);
		int id = visual_shader->find_node_id(type, p_uniform_ref);
		if (id != VisualShader::NODE_ID_INVALID) {
			if (type_changed) {
				List<VisualShader::Connection> conns;
				visual_shader->get_node_connections(type, &conns);
				for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {
					if (E->get().from_node == id) {
						if (visual_shader->is_port_types_compatible(p_uniform_ref->get_uniform_type_by_name(p_name), visual_shader->get_node(type, E->get().to_node)->get_input_port_type(E->get().to_port))) {
							continue;
						}
						undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
						undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
						undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
						undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
					}
				}
			}
			undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type_id, id);
			undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type_id, id);
			break;
		}
	}

	undo_redo->commit_action();
}

void VisualShaderEditor::_float_constant_selected(int p_index, int p_node) {
	if (p_index == 0) {
		graph_plugin->update_node_size(p_node);
		return;
	}

	--p_index;

	ERR_FAIL_INDEX(p_index, MAX_FLOAT_CONST_DEFS);

	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeFloatConstant> node = visual_shader->get_node(type, p_node);
	if (!node.is_valid()) {
		return;
	}

	undo_redo->create_action(TTR("Set constant"));
	undo_redo->add_do_method(node.ptr(), "set_constant", float_constant_defs[p_index].value);
	undo_redo->add_undo_method(node.ptr(), "set_constant", node->get_constant());
	undo_redo->add_do_method(graph_plugin.ptr(), "update_constant", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_constant", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_member_filter_changed(const String &p_text) {
	_update_options_menu();
}

void VisualShaderEditor::_member_selected() {
	TreeItem *item = members->get_selected();

	if (item != nullptr && item->has_meta("id")) {
		members_dialog->get_ok()->set_disabled(false);
		highend_label->set_visible(add_options[item->get_meta("id")].highend);
		node_desc->set_text(_get_description(item->get_meta("id")));
	} else {
		highend_label->set_visible(false);
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

void VisualShaderEditor::_node_menu_id_pressed(int p_idx) {
	switch (p_idx) {
		case NodeMenuOptions::ADD:
			_show_members_dialog(true);
			break;
		case NodeMenuOptions::COPY:
			_copy_nodes();
			break;
		case NodeMenuOptions::PASTE:
			_paste_nodes(true, menu_point);
			break;
		case NodeMenuOptions::DELETE:
			_delete_nodes_request();
			break;
		case NodeMenuOptions::DUPLICATE:
			_duplicate_nodes();
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
			if (d["files"].get_type() == Variant::PACKED_STRING_ARRAY) {
				int j = 0;
				PackedStringArray arr = d["files"];
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
					} else if (type == "CurveTexture") {
						saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_curve_node(arr[i]);
						j++;
					} else if (ClassDB::get_parent_class(type) == "Texture2D") {
						saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_texture2d_node(arr[i]);
						j++;
					} else if (type == "Texture2DArray") {
						saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_texture2d_array_node(arr[i]);
						j++;
					} else if (ClassDB::get_parent_class(type) == "Texture3D") {
						saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_texture3d_node(arr[i]);
						j++;
					} else if (type == "Cubemap") {
						saved_node_pos = p_point + Vector2(0, j * 210 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_cubemap_node(arr[i]);
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

static ShaderLanguage::DataType _get_global_variable_type(const StringName &p_variable) {
	RS::GlobalVariableType gvt = RS::get_singleton()->global_variable_get_type(p_variable);
	return RS::global_variable_type_get_shader_datatype(gvt);
}

void VisualShaderEditor::_update_preview() {
	if (!preview_showed) {
		pending_update_preview = true;
		return;
	}

	String code = visual_shader->get_code();

	preview_text->set_text(code);

	ShaderLanguage sl;

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(visual_shader->get_mode())), ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(visual_shader->get_mode())), ShaderTypes::get_singleton()->get_types(), _get_global_variable_type);

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
	ClassDB::bind_method("_update_graph", &VisualShaderEditor::_update_graph);
	ClassDB::bind_method("_update_options_menu", &VisualShaderEditor::_update_options_menu);
	ClassDB::bind_method("_add_node", &VisualShaderEditor::_add_node);
	ClassDB::bind_method("_node_changed", &VisualShaderEditor::_node_changed);
	ClassDB::bind_method("_input_select_item", &VisualShaderEditor::_input_select_item);
	ClassDB::bind_method("_uniform_select_item", &VisualShaderEditor::_uniform_select_item);
	ClassDB::bind_method("_set_node_size", &VisualShaderEditor::_set_node_size);
	ClassDB::bind_method("_clear_buffer", &VisualShaderEditor::_clear_buffer);
	ClassDB::bind_method("_update_uniforms", &VisualShaderEditor::_update_uniforms);
	ClassDB::bind_method("_set_mode", &VisualShaderEditor::_set_mode);
	ClassDB::bind_method("_nodes_dragged", &VisualShaderEditor::_nodes_dragged);
	ClassDB::bind_method("_float_constant_selected", &VisualShaderEditor::_float_constant_selected);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &VisualShaderEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &VisualShaderEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &VisualShaderEditor::drop_data_fw);

	ClassDB::bind_method("_is_available", &VisualShaderEditor::_is_available);
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
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_TRANSFORM);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SAMPLER);
	//graph->add_valid_left_disconnect_type(0);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->connect("connection_request", callable_mp(this, &VisualShaderEditor::_connection_request), varray(), CONNECT_DEFERRED);
	graph->connect("disconnection_request", callable_mp(this, &VisualShaderEditor::_disconnection_request), varray(), CONNECT_DEFERRED);
	graph->connect("node_selected", callable_mp(this, &VisualShaderEditor::_node_selected));
	graph->connect("scroll_offset_changed", callable_mp(this, &VisualShaderEditor::_scroll_changed));
	graph->connect("duplicate_nodes_request", callable_mp(this, &VisualShaderEditor::_duplicate_nodes));
	graph->connect("copy_nodes_request", callable_mp(this, &VisualShaderEditor::_copy_nodes));
	graph->connect("paste_nodes_request", callable_mp(this, &VisualShaderEditor::_paste_nodes), varray(false, Point2()));
	graph->connect("delete_nodes_request", callable_mp(this, &VisualShaderEditor::_delete_nodes_request));
	graph->connect("gui_input", callable_mp(this, &VisualShaderEditor::_graph_gui_input));
	graph->connect("connection_to_empty", callable_mp(this, &VisualShaderEditor::_connection_to_empty));
	graph->connect("connection_from_empty", callable_mp(this, &VisualShaderEditor::_connection_from_empty));
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_VECTOR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShaderNode::PORT_TYPE_TRANSFORM);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SAMPLER, VisualShaderNode::PORT_TYPE_SAMPLER);

	VSeparator *vs = memnew(VSeparator);
	graph->get_zoom_hbox()->add_child(vs);
	graph->get_zoom_hbox()->move_child(vs, 0);

	edit_type_standart = memnew(OptionButton);
	edit_type_standart->add_item(TTR("Vertex"));
	edit_type_standart->add_item(TTR("Fragment"));
	edit_type_standart->add_item(TTR("Light"));
	edit_type_standart->select(1);
	edit_type_standart->connect("item_selected", callable_mp(this, &VisualShaderEditor::_mode_selected));

	edit_type_particles = memnew(OptionButton);
	edit_type_particles->add_item(TTR("Emit"));
	edit_type_particles->add_item(TTR("Process"));
	edit_type_particles->add_item(TTR("End"));
	edit_type_particles->select(0);
	edit_type_particles->connect("item_selected", callable_mp(this, &VisualShaderEditor::_mode_selected));

	edit_type = edit_type_standart;

	graph->get_zoom_hbox()->add_child(edit_type_particles);
	graph->get_zoom_hbox()->move_child(edit_type_particles, 0);
	graph->get_zoom_hbox()->add_child(edit_type_standart);
	graph->get_zoom_hbox()->move_child(edit_type_standart, 0);

	add_node = memnew(Button);
	add_node->set_flat(true);
	graph->get_zoom_hbox()->add_child(add_node);
	add_node->set_text(TTR("Add Node..."));
	graph->get_zoom_hbox()->move_child(add_node, 0);
	add_node->connect("pressed", callable_mp(this, &VisualShaderEditor::_show_members_dialog), varray(false));

	preview_shader = memnew(Button);
	preview_shader->set_flat(true);
	preview_shader->set_toggle_mode(true);
	preview_shader->set_tooltip(TTR("Show resulted shader code."));
	graph->get_zoom_hbox()->add_child(preview_shader);
	preview_shader->connect("pressed", callable_mp(this, &VisualShaderEditor::_show_preview_text));

	///////////////////////////////////////
	// PREVIEW PANEL
	///////////////////////////////////////

	preview_vbox = memnew(VBoxContainer);
	preview_vbox->set_visible(preview_showed);
	main_box->add_child(preview_vbox);
	preview_text = memnew(CodeEdit);
	syntax_highlighter.instance();
	preview_vbox->add_child(preview_text);
	preview_text->set_h_size_flags(SIZE_EXPAND_FILL);
	preview_text->set_v_size_flags(SIZE_EXPAND_FILL);
	preview_text->set_custom_minimum_size(Size2(400 * EDSCALE, 0));
	preview_text->set_syntax_highlighter(syntax_highlighter);
	preview_text->set_draw_line_numbers(true);
	preview_text->set_readonly(true);

	error_text = memnew(Label);
	preview_vbox->add_child(error_text);
	error_text->set_visible(false);

	///////////////////////////////////////
	// POPUP MENU
	///////////////////////////////////////

	popup_menu = memnew(PopupMenu);
	add_child(popup_menu);
	popup_menu->add_item("Add Node", NodeMenuOptions::ADD);
	popup_menu->add_separator();
	popup_menu->add_item("Copy", NodeMenuOptions::COPY);
	popup_menu->add_item("Paste", NodeMenuOptions::PASTE);
	popup_menu->add_item("Delete", NodeMenuOptions::DELETE);
	popup_menu->add_item("Duplicate", NodeMenuOptions::DUPLICATE);
	popup_menu->connect("id_pressed", callable_mp(this, &VisualShaderEditor::_node_menu_id_pressed));

	///////////////////////////////////////
	// SHADER NODES TREE
	///////////////////////////////////////

	VBoxContainer *members_vb = memnew(VBoxContainer);
	members_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *filter_hb = memnew(HBoxContainer);
	members_vb->add_child(filter_hb);

	node_filter = memnew(LineEdit);
	filter_hb->add_child(node_filter);
	node_filter->connect("text_changed", callable_mp(this, &VisualShaderEditor::_member_filter_changed));
	node_filter->connect("gui_input", callable_mp(this, &VisualShaderEditor::_sbox_input));
	node_filter->set_h_size_flags(SIZE_EXPAND_FILL);
	node_filter->set_placeholder(TTR("Search"));

	tools = memnew(MenuButton);
	filter_hb->add_child(tools);
	tools->set_tooltip(TTR("Options"));
	tools->get_popup()->connect("id_pressed", callable_mp(this, &VisualShaderEditor::_tools_menu_option));
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
	members->connect("item_activated", callable_mp(this, &VisualShaderEditor::_member_create));
	members->connect("item_selected", callable_mp(this, &VisualShaderEditor::_member_selected));
	members->connect("nothing_selected", callable_mp(this, &VisualShaderEditor::_member_unselected));

	HBoxContainer *desc_hbox = memnew(HBoxContainer);
	members_vb->add_child(desc_hbox);

	Label *desc_label = memnew(Label);
	desc_hbox->add_child(desc_label);
	desc_label->set_text(TTR("Description:"));

	desc_hbox->add_spacer();

	highend_label = memnew(Label);
	desc_hbox->add_child(highend_label);
	highend_label->set_visible(false);
	highend_label->set_text("Vulkan");
	highend_label->set_mouse_filter(Control::MOUSE_FILTER_STOP);
	highend_label->set_tooltip(TTR("High-end node"));

	node_desc = memnew(RichTextLabel);
	members_vb->add_child(node_desc);
	node_desc->set_h_size_flags(SIZE_EXPAND_FILL);
	node_desc->set_v_size_flags(SIZE_FILL);
	node_desc->set_custom_minimum_size(Size2(0, 70 * EDSCALE));

	members_dialog = memnew(ConfirmationDialog);
	members_dialog->set_title(TTR("Create Shader Node"));
	members_dialog->set_exclusive(false);
	members_dialog->add_child(members_vb);
	members_dialog->get_ok()->set_text(TTR("Create"));
	members_dialog->get_ok()->connect("pressed", callable_mp(this, &VisualShaderEditor::_member_create));
	members_dialog->get_ok()->set_disabled(true);
	members_dialog->connect("cancelled", callable_mp(this, &VisualShaderEditor::_member_cancel));
	add_child(members_dialog);

	alert = memnew(AcceptDialog);
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
	const String input_param_for_emit_shader_mode = TTR("'%s' input parameter for emit shader mode.");
	const String input_param_for_process_shader_mode = TTR("'%s' input parameter for process shader mode.");
	const String input_param_for_end_shader_mode = TTR("'%s' input parameter for end shader mode.");
	const String input_param_for_emit_and_process_shader_mode = TTR("'%s' input parameter for emit and process shader mode.");
	const String input_param_for_vertex_and_fragment_shader_mode = TTR("'%s' input parameter for vertex and fragment shader mode.");

	add_options.push_back(AddOption("Alpha", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Binormal", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "binormal"), "binormal", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Color", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("DepthTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "depth_texture"), "depth_texture", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FragCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FrontFacing", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "front_facing"), "front_facing", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("PointCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "point_coord"), "point_coord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ScreenTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_texture"), "screen_texture", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ScreenUV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Side", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "side"), "side", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Tangent", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "tangent"), "tangent", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv"), "uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV2", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv2"), "uv2", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Vertex", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "vertex"), "vertex", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("View", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "view"), "view", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));

	add_options.push_back(AddOption("Albedo", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "albedo"), "albedo", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Attenuation", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "attenuation"), "attenuation", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Diffuse", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "diffuse"), "diffuse", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FragCoord", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Light", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light"), "light", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("LightColor", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_color"), "light_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Metallic", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "metallic"), "metallic", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Roughness", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "roughness"), "roughness", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Specular", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "specular"), "specular", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Transmission", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "transmission"), "transmission", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("View", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "view"), "view", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));

	add_options.push_back(AddOption("Alpha", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Binormal", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "binormal"), "binormal", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Color", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ModelView", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "modelview"), "modelview", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("PointSize", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "point_size"), "point_size", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Tangent", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_mode, "tangent"), "tangent", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv"), "uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV2", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "uv2"), "uv2", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Vertex", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "vertex"), "vertex", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));

	// CANVASITEM INPUTS

	add_options.push_back(AddOption("FragCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightPass", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "light_pass"), "light_pass", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("NormalTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "normal_texture"), "normal_texture", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointCoord", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "point_coord"), "point_coord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenPixelSize", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_pixel_size"), "screen_pixel_size", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenTexture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_texture"), "screen_texture", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenUV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Texture", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "texture"), "texture", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));

	add_options.push_back(AddOption("FragCoord", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord"), "fragcoord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightAlpha", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_alpha"), "light_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightColor", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_color"), "light_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightHeight", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_height"), "light_height", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightUV", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_uv"), "light_uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightVector", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_vec"), "light_vec", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Normal", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "normal"), "normal", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointCoord", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "point_coord"), "point_coord", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenUV", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ShadowAlpha", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow_alpha"), "shadow_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ShadowColor", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow_color"), "shadow_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ShadowVec", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow_vec"), "shadow_vec", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Texture", "Input", "Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "texture"), "texture", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));

	add_options.push_back(AddOption("Extra", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "extra"), "extra", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightPass", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "light_pass"), "light_pass", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointSize", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "point_size"), "point_size", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Projection", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "projection"), "projection", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Vertex", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "vertex"), "vertex", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("World", "Input", "Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "world"), "world", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));

	// PARTICLES INPUTS

	add_options.push_back(AddOption("Active", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "active"), "active", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Alpha", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Color", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Custom", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom"), "custom", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("CustomAlpha", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom_alpha"), "custom_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Delta", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "delta"), "delta", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("EmissionTransform", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "emission_transform"), "emission_transform", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Index", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "index"), "index", VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("LifeTime", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "lifetime"), "lifetime", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Restart", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "restart"), "restart", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Time", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Transform", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "transform"), "transform", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Velocity", "Input", "Emit", "VisualShaderNodeInput", vformat(input_param_shader_modes, "velocity"), "velocity", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));

	add_options.push_back(AddOption("Active", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "active"), "active", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Alpha", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Color", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Custom", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom"), "custom", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("CustomAlpha", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom_alpha"), "custom_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Delta", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "delta"), "delta", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("EmissionTransform", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "emission_transform"), "emission_transform", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Index", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "index"), "index", VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("LifeTime", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "lifetime"), "lifetime", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Restart", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "restart"), "restart", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Time", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Transform", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "transform"), "transform", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Velocity", "Input", "Process", "VisualShaderNodeInput", vformat(input_param_shader_modes, "velocity"), "velocity", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));

	add_options.push_back(AddOption("Active", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "active"), "active", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Alpha", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "alpha"), "alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Color", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "color"), "color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Custom", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom"), "custom", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("CustomAlpha", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom_alpha"), "custom_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Delta", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "delta"), "delta", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("EmissionTransform", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "emission_transform"), "emission_transform", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Index", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "index"), "index", VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("LifeTime", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "lifetime"), "lifetime", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Restart", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "restart"), "restart", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Time", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Transform", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "transform"), "transform", VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_END, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Velocity", "Input", "End", "VisualShaderNodeInput", vformat(input_param_shader_modes, "velocity"), "velocity", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_END, Shader::MODE_PARTICLES));

	// SKY INPUTS

	add_options.push_back(AddOption("AtCubeMapPass", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "at_cubemap_pass"), "at_cubemap_pass", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("AtHalfResPass", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "at_half_res_pass"), "at_half_res_pass", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("AtQuarterResPass", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "at_quarter_res_pass"), "at_quarter_res_pass", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("EyeDir", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "eyedir"), "eyedir", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("HalfResColor", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "half_res_color"), "half_res_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("HalfResAlpha", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "half_res_alpha"), "half_res_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Color", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light0_color"), "light0_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Direction", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light0_direction"), "light0_direction", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Enabled", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light0_enabled"), "light0_enabled", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Energy", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light0_energy"), "light0_energy", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Color", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light1_color"), "light1_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Direction", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light1_direction"), "light1_direction", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Enabled", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light1_enabled"), "light1_enabled", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Energy", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light1_energy"), "light1_energy", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Color", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light2_color"), "light2_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Direction", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light2_direction"), "light2_direction", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Enabled", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light2_enabled"), "light2_enabled", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Energy", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light2_energy"), "light2_energy", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Color", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light3_color"), "light3_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Direction", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light3_direction"), "light3_direction", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Enabled", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light3_enabled"), "light3_enabled", VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Energy", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "light3_energy"), "light3_energy", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Position", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "position"), "position", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("QuarterResColor", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "quarter_res_color"), "quarter_res_color", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("QuarterResAlpha", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "quarter_res_alpha"), "quarter_res_alpha", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Radiance", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "radiance"), "radiance", VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("ScreenUV", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_uv"), "screen_uv", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("SkyCoords", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "sky_coords"), "sky_coords", VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));
	add_options.push_back(AddOption("Time", "Input", "Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "time"), "time", VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SKY));

	// SCALAR

	add_options.push_back(AddOption("FloatFunc", "Scalar", "Common", "VisualShaderNodeFloatFunc", TTR("Float function."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntFunc", "Scalar", "Common", "VisualShaderNodeIntFunc", TTR("Integer function."), -1, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("FloatOp", "Scalar", "Common", "VisualShaderNodeFloatOp", TTR("Float operator."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntOp", "Scalar", "Common", "VisualShaderNodeIntOp", TTR("Integer operator."), -1, VisualShaderNode::PORT_TYPE_SCALAR_INT));

	//CONSTANTS

	for (int i = 0; i < MAX_FLOAT_CONST_DEFS; i++) {
		add_options.push_back(AddOption(float_constant_defs[i].name, "Scalar", "Constants", "VisualShaderNodeFloatConstant", float_constant_defs[i].desc, -1, VisualShaderNode::PORT_TYPE_SCALAR, -1, -1, float_constant_defs[i].value));
	}
	// FUNCTIONS

	add_options.push_back(AddOption("Abs", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the absolute value of the parameter."), VisualShaderNodeFloatFunc::FUNC_ABS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Abs", "Scalar", "Functions", "VisualShaderNodeIntFunc", TTR("Returns the absolute value of the parameter."), VisualShaderNodeIntFunc::FUNC_ABS, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("ACos", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the arc-cosine of the parameter."), VisualShaderNodeFloatFunc::FUNC_ACOS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ACosH", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), VisualShaderNodeFloatFunc::FUNC_ACOSH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ASin", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the arc-sine of the parameter."), VisualShaderNodeFloatFunc::FUNC_ASIN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ASinH", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), VisualShaderNodeFloatFunc::FUNC_ASINH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATan", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the arc-tangent of the parameter."), VisualShaderNodeFloatFunc::FUNC_ATAN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATan2", "Scalar", "Functions", "VisualShaderNodeFloatOp", TTR("Returns the arc-tangent of the parameters."), VisualShaderNodeFloatOp::OP_ATAN2, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATanH", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), VisualShaderNodeFloatFunc::FUNC_ATANH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Ceil", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), VisualShaderNodeFloatFunc::FUNC_CEIL, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Clamp", "Scalar", "Functions", "VisualShaderNodeScalarClamp", TTR("Constrains a value to lie between two further values."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Clamp", "Scalar", "Functions", "VisualShaderNodeIntFunc", TTR("Constrains a value to lie between two further values."), VisualShaderNodeIntFunc::FUNC_CLAMP, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Cos", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the cosine of the parameter."), VisualShaderNodeFloatFunc::FUNC_COS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("CosH", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the hyperbolic cosine of the parameter."), VisualShaderNodeFloatFunc::FUNC_COSH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Degrees", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Converts a quantity in radians to degrees."), VisualShaderNodeFloatFunc::FUNC_DEGREES, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Base-e Exponential."), VisualShaderNodeFloatFunc::FUNC_EXP, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp2", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Base-2 Exponential."), VisualShaderNodeFloatFunc::FUNC_EXP2, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Floor", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest integer less than or equal to the parameter."), VisualShaderNodeFloatFunc::FUNC_FLOOR, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Fract", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Computes the fractional part of the argument."), VisualShaderNodeFloatFunc::FUNC_FRAC, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("InverseSqrt", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse of the square root of the parameter."), VisualShaderNodeFloatFunc::FUNC_INVERSE_SQRT, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Natural logarithm."), VisualShaderNodeFloatFunc::FUNC_LOG, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log2", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Base-2 logarithm."), VisualShaderNodeFloatFunc::FUNC_LOG2, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Max", "Scalar", "Functions", "VisualShaderNodeFloatOp", TTR("Returns the greater of two values."), VisualShaderNodeFloatOp::OP_MAX, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Min", "Scalar", "Functions", "VisualShaderNodeFloatOp", TTR("Returns the lesser of two values."), VisualShaderNodeFloatOp::OP_MIN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Mix", "Scalar", "Functions", "VisualShaderNodeScalarInterp", TTR("Linear interpolation between two scalars."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("MultiplyAdd", "Scalar", "Functions", "VisualShaderNodeMultiplyAdd", TTR("Performs a fused multiply-add operation (a * b + c) on scalars."), VisualShaderNodeMultiplyAdd::OP_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Negate", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the opposite value of the parameter."), VisualShaderNodeFloatFunc::FUNC_NEGATE, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Negate", "Scalar", "Functions", "VisualShaderNodeIntFunc", TTR("Returns the opposite value of the parameter."), VisualShaderNodeIntFunc::FUNC_NEGATE, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("OneMinus", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("1.0 - scalar"), VisualShaderNodeFloatFunc::FUNC_ONEMINUS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Pow", "Scalar", "Functions", "VisualShaderNodeFloatOp", TTR("Returns the value of the first parameter raised to the power of the second."), VisualShaderNodeFloatOp::OP_POW, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Radians", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Converts a quantity in degrees to radians."), VisualShaderNodeFloatFunc::FUNC_RADIANS, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Reciprocal", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("1.0 / scalar"), VisualShaderNodeFloatFunc::FUNC_RECIPROCAL, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Round", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest integer to the parameter."), VisualShaderNodeFloatFunc::FUNC_ROUND, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("RoundEven", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest even integer to the parameter."), VisualShaderNodeFloatFunc::FUNC_ROUNDEVEN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Saturate", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Clamps the value between 0.0 and 1.0."), VisualShaderNodeFloatFunc::FUNC_SATURATE, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sign", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Extracts the sign of the parameter."), VisualShaderNodeFloatFunc::FUNC_SIGN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sign", "Scalar", "Functions", "VisualShaderNodeIntFunc", TTR("Extracts the sign of the parameter."), VisualShaderNodeIntFunc::FUNC_SIGN, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Sin", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the sine of the parameter."), VisualShaderNodeFloatFunc::FUNC_SIN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SinH", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the hyperbolic sine of the parameter."), VisualShaderNodeFloatFunc::FUNC_SINH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sqrt", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the square root of the parameter."), VisualShaderNodeFloatFunc::FUNC_SQRT, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SmoothStep", "Scalar", "Functions", "VisualShaderNodeScalarSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), scalar(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if x is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Step", "Scalar", "Functions", "VisualShaderNodeFloatOp", TTR("Step function( scalar(edge), scalar(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), VisualShaderNodeFloatOp::OP_STEP, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Tan", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the tangent of the parameter."), VisualShaderNodeFloatFunc::FUNC_TAN, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("TanH", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Returns the hyperbolic tangent of the parameter."), VisualShaderNodeFloatFunc::FUNC_TANH, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Trunc", "Scalar", "Functions", "VisualShaderNodeFloatFunc", TTR("Finds the truncated value of the parameter."), VisualShaderNodeFloatFunc::FUNC_TRUNC, VisualShaderNode::PORT_TYPE_SCALAR));

	add_options.push_back(AddOption("Add", "Scalar", "Operators", "VisualShaderNodeFloatOp", TTR("Sums two floating-point scalars."), VisualShaderNodeFloatOp::OP_ADD, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Add", "Scalar", "Operators", "VisualShaderNodeIntOp", TTR("Sums two integer scalars."), VisualShaderNodeIntOp::OP_ADD, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Divide", "Scalar", "Operators", "VisualShaderNodeFloatOp", TTR("Divides two floating-point scalars."), VisualShaderNodeFloatOp::OP_DIV, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Divide", "Scalar", "Operators", "VisualShaderNodeIntOp", TTR("Divides two integer scalars."), VisualShaderNodeIntOp::OP_DIV, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Multiply", "Scalar", "Operators", "VisualShaderNodeFloatOp", TTR("Multiplies two floating-point scalars."), VisualShaderNodeFloatOp::OP_MUL, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Multiply", "Scalar", "Operators", "VisualShaderNodeIntOp", TTR("Multiplies two integer scalars."), VisualShaderNodeIntOp::OP_MUL, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Remainder", "Scalar", "Operators", "VisualShaderNodeFloatOp", TTR("Returns the remainder of the two floating-point scalars."), VisualShaderNodeFloatOp::OP_MOD, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Remainder", "Scalar", "Operators", "VisualShaderNodeIntOp", TTR("Returns the remainder of the two integer scalars."), VisualShaderNodeIntOp::OP_MOD, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Subtract", "Scalar", "Operators", "VisualShaderNodeFloatOp", TTR("Subtracts two floating-point scalars."), VisualShaderNodeFloatOp::OP_SUB, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Subtract", "Scalar", "Operators", "VisualShaderNodeIntOp", TTR("Subtracts two integer scalars."), VisualShaderNodeIntOp::OP_SUB, VisualShaderNode::PORT_TYPE_SCALAR_INT));

	add_options.push_back(AddOption("FloatConstant", "Scalar", "Variables", "VisualShaderNodeFloatConstant", TTR("Scalar floating-point constant."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntConstant", "Scalar", "Variables", "VisualShaderNodeIntConstant", TTR("Scalar integer constant."), -1, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("FloatUniform", "Scalar", "Variables", "VisualShaderNodeFloatUniform", TTR("Scalar floating-point uniform."), -1, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntUniform", "Scalar", "Variables", "VisualShaderNodeIntUniform", TTR("Scalar integer uniform."), -1, VisualShaderNode::PORT_TYPE_SCALAR_INT));

	// TEXTURES
	cubemap_node_option_idx = add_options.size();
	add_options.push_back(AddOption("CubeMap", "Textures", "Functions", "VisualShaderNodeCubemap", TTR("Perform the cubic texture lookup."), -1, -1));
	curve_node_option_idx = add_options.size();
	add_options.push_back(AddOption("CurveTexture", "Textures", "Functions", "VisualShaderNodeCurveTexture", TTR("Perform the curve texture lookup."), -1, -1));
	texture2d_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture2D", "Textures", "Functions", "VisualShaderNodeTexture", TTR("Perform the 2D texture lookup."), -1, -1));
	texture2d_array_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture2DArray", "Textures", "Functions", "VisualShaderNodeTexture2DArray", TTR("Perform the 2D-array texture lookup."), -1, -1, -1, -1, -1));
	texture3d_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture3D", "Textures", "Functions", "VisualShaderNodeTexture3D", TTR("Perform the 3D texture lookup."), -1, -1));

	add_options.push_back(AddOption("CubeMapUniform", "Textures", "Variables", "VisualShaderNodeCubemapUniform", TTR("Cubic texture uniform lookup."), -1, -1));
	add_options.push_back(AddOption("TextureUniform", "Textures", "Variables", "VisualShaderNodeTextureUniform", TTR("2D texture uniform lookup."), -1, -1));
	add_options.push_back(AddOption("TextureUniformTriplanar", "Textures", "Variables", "VisualShaderNodeTextureUniformTriplanar", TTR("2D texture uniform lookup with triplanar."), -1, -1, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Texture2DArrayUniform", "Textures", "Variables", "VisualShaderNodeTexture2DArrayUniform", TTR("2D array of textures uniform lookup."), -1, -1, -1, -1, -1));
	add_options.push_back(AddOption("Texture3DUniform", "Textures", "Variables", "VisualShaderNodeTexture3DUniform", TTR("3D texture uniform lookup."), -1, -1, -1, -1, -1));

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
	add_options.push_back(AddOption("MultiplyAdd", "Vector", "Functions", "VisualShaderNodeMultiplyAdd", TTR("Performs a fused multiply-add operation (a * b + c) on vectors."), VisualShaderNodeMultiplyAdd::OP_TYPE_VECTOR, VisualShaderNode::PORT_TYPE_VECTOR));
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

	add_options.push_back(AddOption("ScalarDerivativeFunc", "Special", "Common", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) Scalar derivative function."), -1, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("VectorDerivativeFunc", "Special", "Common", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) Vector derivative function."), -1, VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));

	add_options.push_back(AddOption("DdX", "Special", "Derivative", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'x' using local differencing."), VisualShaderNodeVectorDerivativeFunc::FUNC_X, VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("DdXS", "Special", "Derivative", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Derivative in 'x' using local differencing."), VisualShaderNodeScalarDerivativeFunc::FUNC_X, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("DdY", "Special", "Derivative", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'y' using local differencing."), VisualShaderNodeVectorDerivativeFunc::FUNC_Y, VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("DdYS", "Special", "Derivative", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Derivative in 'y' using local differencing."), VisualShaderNodeScalarDerivativeFunc::FUNC_Y, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("Sum", "Special", "Derivative", "VisualShaderNodeVectorDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Sum of absolute derivative in 'x' and 'y'."), VisualShaderNodeVectorDerivativeFunc::FUNC_SUM, VisualShaderNode::PORT_TYPE_VECTOR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
	add_options.push_back(AddOption("SumS", "Special", "Derivative", "VisualShaderNodeScalarDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Sum of absolute derivative in 'x' and 'y'."), VisualShaderNodeScalarDerivativeFunc::FUNC_SUM, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, -1, true));
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

	graph_plugin.instance();

	property_editor = memnew(CustomPropertyEditor);
	add_child(property_editor);

	property_editor->connect("variant_changed", callable_mp(this, &VisualShaderEditor::_port_edited));
}

/////////////////

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

public:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			connect("item_selected", callable_mp(this, &VisualShaderNodePluginInputEditor::_item_selected));
		}
	}

	void _item_selected(int p_item) {
		VisualShaderEditor::get_singleton()->call_deferred("_input_select_item", input, get_item_text(p_item));
	}

	void setup(const Ref<VisualShaderNodeInput> &p_input) {
		input = p_input;
		Ref<Texture2D> type_icon[6] = {
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("float", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("int", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Vector3", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("bool", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Transform", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("ImageTexture", "EditorIcons"),
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

public:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			connect("item_selected", callable_mp(this, &VisualShaderNodePluginUniformRefEditor::_item_selected));
		}
	}

	void _item_selected(int p_item) {
		VisualShaderEditor::get_singleton()->call_deferred("_uniform_select_item", uniform_ref, get_item_text(p_item));
	}

	void setup(const Ref<VisualShaderNodeUniformRef> &p_uniform_ref) {
		uniform_ref = p_uniform_ref;

		Ref<Texture2D> type_icon[7] = {
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("float", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("int", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("bool", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Vector3", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Transform", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("Color", "EditorIcons"),
			EditorNode::get_singleton()->get_gui_base()->get_theme_icon("ImageTexture", "EditorIcons"),
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
	int node_id;
	VisualShader::Type shader_type;

public:
	void _property_changed(const String &p_property, const Variant &p_value, const String &p_field = "", bool p_changing = false) {
		if (p_changing) {
			return;
		}

		UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

		updating = true;
		undo_redo->create_action(TTR("Edit Visual Property") + ": " + p_property, UndoRedo::MERGE_ENDS);
		undo_redo->add_do_property(node.ptr(), p_property, p_value);
		undo_redo->add_undo_property(node.ptr(), p_property, node->get(p_property));

		if (p_value.get_type() == Variant::OBJECT) {
			RES prev_res = node->get(p_property);
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
		}
		if (p_property != "constant") {
			undo_redo->add_do_method(VisualShaderEditor::get_singleton()->get_graph_plugin(), "update_node_deferred", shader_type, node_id);
			undo_redo->add_undo_method(VisualShaderEditor::get_singleton()->get_graph_plugin(), "update_node_deferred", shader_type, node_id);
		} else {
			undo_redo->add_do_method(VisualShaderEditor::get_singleton()->get_graph_plugin(), "update_constant", shader_type, node_id);
			undo_redo->add_undo_method(VisualShaderEditor::get_singleton()->get_graph_plugin(), "update_constant", shader_type, node_id);
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

	void _resource_selected(const String &p_path, RES p_resource) {
		_open_inspector(p_resource);
	}

	void _open_inspector(RES p_resource) {
		EditorNode::get_singleton()->get_inspector()->edit(p_resource.ptr());
	}

	bool updating;
	Ref<VisualShaderNode> node;
	Vector<EditorProperty *> properties;
	Vector<Label *> prop_names;

	void _show_prop_names(bool p_show) {
		for (int i = 0; i < prop_names.size(); i++) {
			prop_names[i]->set_visible(p_show);
		}
	}

	void setup(Ref<Resource> p_parent_resource, Vector<EditorProperty *> p_properties, const Vector<StringName> &p_names, Ref<VisualShaderNode> p_node) {
		parent_resource = p_parent_resource;
		updating = false;
		node = p_node;
		properties = p_properties;

		node_id = (int)p_node->get_meta("id");
		shader_type = VisualShader::Type((int)p_node->get_meta("shader_type"));

		for (int i = 0; i < p_properties.size(); i++) {
			HBoxContainer *hbox = memnew(HBoxContainer);
			hbox->set_h_size_flags(SIZE_EXPAND_FILL);
			add_child(hbox);

			Label *prop_name = memnew(Label);
			String prop_name_str = p_names[i];
			prop_name_str = prop_name_str.capitalize() + ":";
			prop_name->set_text(prop_name_str);
			prop_name->set_visible(false);
			hbox->add_child(prop_name);
			prop_names.push_back(prop_name);

			p_properties[i]->set_h_size_flags(SIZE_EXPAND_FILL);
			hbox->add_child(p_properties[i]);

			bool res_prop = Object::cast_to<EditorPropertyResource>(p_properties[i]);
			if (res_prop) {
				p_properties[i]->connect("resource_selected", callable_mp(this, &VisualShaderNodePluginDefaultEditor::_resource_selected));
			}

			properties[i]->connect("property_changed", callable_mp(this, &VisualShaderNodePluginDefaultEditor::_property_changed));
			properties[i]->set_object_and_property(node.ptr(), p_names[i]);
			properties[i]->update_property();
			properties[i]->set_name_split_ratio(0);
		}
		node->connect("changed", callable_mp(this, &VisualShaderNodePluginDefaultEditor::_node_changed));
	}

	static void _bind_methods() {
		ClassDB::bind_method("_open_inspector", &VisualShaderNodePluginDefaultEditor::_open_inspector); // Used by UndoRedo.
		ClassDB::bind_method("_show_prop_names", &VisualShaderNodePluginDefaultEditor::_show_prop_names); // Used with call_deferred.
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

	undo_redo->add_do_method(VisualShaderEditor::get_singleton(), "_set_mode", p_which);
	undo_redo->add_undo_method(VisualShaderEditor::get_singleton(), "_set_mode", visual_shader->get_mode());

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
}

EditorPropertyShaderMode::EditorPropertyShaderMode() {
	options = memnew(OptionButton);
	options->set_clip_text(true);
	add_child(options);
	add_focusable(options);
	options->connect("item_selected", callable_mp(this, &EditorPropertyShaderMode::_option_selected));
}

bool EditorInspectorShaderModePlugin::can_handle(Object *p_object) {
	return true; //can handle everything
}

void EditorInspectorShaderModePlugin::parse_begin(Object *p_object) {
	//do none
}

bool EditorInspectorShaderModePlugin::parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage, bool p_wide) {
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
		if (!object) {
			continue;
		}
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
	shader->connect("changed", callable_mp(this, &VisualShaderNodePortPreview::_shader_changed));
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
