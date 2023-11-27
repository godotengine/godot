/**************************************************************************/
/*  visual_shader_editor_plugin.cpp                                       */
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

#include "visual_shader_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/editor_properties_vector.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/filesystem_dock.h"
#include "editor/inspector_dock.h"
#include "editor/plugins/curve_editor_plugin.h"
#include "editor/plugins/shader_editor_plugin.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/code_edit.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/separator.h"
#include "scene/gui/tree.h"
#include "scene/gui/view_panner.h"
#include "scene/main/window.h"
#include "scene/resources/curve_texture.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/visual_shader_nodes.h"
#include "scene/resources/visual_shader_particle_nodes.h"
#include "servers/display_server.h"
#include "servers/rendering/shader_preprocessor.h"
#include "servers/rendering/shader_types.h"

struct FloatConstantDef {
	String name;
	float value = 0;
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

void VisualShaderNodePlugin::set_editor(VisualShaderEditor *p_editor) {
	vseditor = p_editor;
}

Control *VisualShaderNodePlugin::create_editor(const Ref<Resource> &p_parent_resource, const Ref<VisualShaderNode> &p_node) {
	Object *ret = nullptr;
	GDVIRTUAL_CALL(_create_editor, p_parent_resource, p_node, ret);
	return Object::cast_to<Control>(ret);
}

void VisualShaderNodePlugin::_bind_methods() {
	GDVIRTUAL_BIND(_create_editor, "parent_resource", "visual_shader_node");
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
	ClassDB::bind_method("update_node", &VisualShaderGraphPlugin::update_node);
	ClassDB::bind_method("update_node_deferred", &VisualShaderGraphPlugin::update_node_deferred);
	ClassDB::bind_method("set_input_port_default_value", &VisualShaderGraphPlugin::set_input_port_default_value);
	ClassDB::bind_method("set_parameter_name", &VisualShaderGraphPlugin::set_parameter_name);
	ClassDB::bind_method("set_expression", &VisualShaderGraphPlugin::set_expression);
	ClassDB::bind_method("update_curve", &VisualShaderGraphPlugin::update_curve);
	ClassDB::bind_method("update_curve_xyz", &VisualShaderGraphPlugin::update_curve_xyz);
}

void VisualShaderGraphPlugin::set_editor(VisualShaderEditor *p_editor) {
	editor = p_editor;
}

void VisualShaderGraphPlugin::register_shader(VisualShader *p_shader) {
	visual_shader = Ref<VisualShader>(p_shader);
}

void VisualShaderGraphPlugin::set_connections(const List<VisualShader::Connection> &p_connections) {
	connections = p_connections;
}

void VisualShaderGraphPlugin::show_port_preview(VisualShader::Type p_type, int p_node_id, int p_port_id, bool p_is_valid) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_node_id) && links[p_node_id].output_ports.has(p_port_id)) {
		Link &link = links[p_node_id];

		for (const KeyValue<int, Port> &E : link.output_ports) {
			if (E.value.preview_button != nullptr) {
				E.value.preview_button->set_pressed(false);
			}
		}
		bool is_dirty = link.preview_pos < 0;

		if (!is_dirty && link.preview_visible && link.preview_box != nullptr) {
			link.graph_element->remove_child(link.preview_box);
			memdelete(link.preview_box);
			link.preview_box = nullptr;
			link.graph_element->reset_size();
			link.preview_visible = false;
		}

		if (p_port_id != -1 && link.output_ports[p_port_id].preview_button != nullptr) {
			if (is_dirty) {
				link.preview_pos = link.graph_element->get_child_count();
			}

			VBoxContainer *vbox = memnew(VBoxContainer);
			link.graph_element->add_child(vbox);
			link.graph_element->move_child(vbox, link.preview_pos);

			GraphNode *graph_node = Object::cast_to<GraphNode>(link.graph_element);
			if (graph_node) {
				graph_node->set_slot_draw_stylebox(vbox->get_index(false), false);
			}

			Control *offset = memnew(Control);
			offset->set_custom_minimum_size(Size2(0, 5 * EDSCALE));
			vbox->add_child(offset);

			VisualShaderNodePortPreview *port_preview = memnew(VisualShaderNodePortPreview);
			port_preview->setup(visual_shader, visual_shader->get_shader_type(), p_node_id, p_port_id, p_is_valid);
			port_preview->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
			vbox->add_child(port_preview);
			link.preview_visible = true;
			link.preview_box = vbox;
			link.output_ports[p_port_id].preview_button->set_pressed(true);
		}
	}
}

void VisualShaderGraphPlugin::update_node_deferred(VisualShader::Type p_type, int p_node_id) {
	call_deferred(SNAME("update_node"), p_type, p_node_id);
}

void VisualShaderGraphPlugin::update_node(VisualShader::Type p_type, int p_node_id) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id)) {
		return;
	}
	remove_node(p_type, p_node_id, true);
	add_node(p_type, p_node_id, true);
}

void VisualShaderGraphPlugin::set_input_port_default_value(VisualShader::Type p_type, int p_node_id, int p_port_id, Variant p_value) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id)) {
		return;
	}

	Button *button = links[p_node_id].input_ports[p_port_id].default_input_button;

	switch (p_value.get_type()) {
		case Variant::COLOR: {
			button->set_custom_minimum_size(Size2(30, 0) * EDSCALE);

			Callable ce = callable_mp(editor, &VisualShaderEditor::_draw_color_over_button);
			if (!button->is_connected("draw", ce)) {
				button->connect("draw", ce.bind(button, p_value));
			}
		} break;
		case Variant::BOOL: {
			button->set_text(((bool)p_value) ? "true" : "false");
		} break;
		case Variant::INT:
		case Variant::FLOAT: {
			button->set_text(String::num(p_value, 4));
		} break;
		case Variant::VECTOR2: {
			Vector2 v = p_value;
			button->set_text(String::num(v.x, 3) + "," + String::num(v.y, 3));
		} break;
		case Variant::VECTOR3: {
			Vector3 v = p_value;
			button->set_text(String::num(v.x, 3) + "," + String::num(v.y, 3) + "," + String::num(v.z, 3));
		} break;
		case Variant::QUATERNION: {
			Quaternion v = p_value;
			button->set_text(String::num(v.x, 3) + "," + String::num(v.y, 3) + "," + String::num(v.z, 3) + "," + String::num(v.w, 3));
		} break;
		default: {
		}
	}
}

void VisualShaderGraphPlugin::set_parameter_name(VisualShader::Type p_type, int p_node_id, const String &p_name) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_node_id) && links[p_node_id].parameter_name != nullptr) {
		links[p_node_id].parameter_name->set_text(p_name);
	}
}

void VisualShaderGraphPlugin::update_curve(int p_node_id) {
	if (links.has(p_node_id) && links[p_node_id].curve_editors[0]) {
		Ref<VisualShaderNodeCurveTexture> tex = Object::cast_to<VisualShaderNodeCurveTexture>(links[p_node_id].visual_node);
		ERR_FAIL_COND(!tex.is_valid());

		if (tex->get_texture().is_valid()) {
			links[p_node_id].curve_editors[0]->set_curve(tex->get_texture()->get_curve());
		}
		tex->emit_changed();
	}
}

void VisualShaderGraphPlugin::update_curve_xyz(int p_node_id) {
	if (links.has(p_node_id) && links[p_node_id].curve_editors[0] && links[p_node_id].curve_editors[1] && links[p_node_id].curve_editors[2]) {
		Ref<VisualShaderNodeCurveXYZTexture> tex = Object::cast_to<VisualShaderNodeCurveXYZTexture>(links[p_node_id].visual_node);
		ERR_FAIL_COND(!tex.is_valid());

		if (tex->get_texture().is_valid()) {
			links[p_node_id].curve_editors[0]->set_curve(tex->get_texture()->get_curve_x());
			links[p_node_id].curve_editors[1]->set_curve(tex->get_texture()->get_curve_y());
			links[p_node_id].curve_editors[2]->set_curve(tex->get_texture()->get_curve_z());
		}
		tex->emit_changed();
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

void VisualShaderGraphPlugin::set_expression(VisualShader::Type p_type, int p_node_id, const String &p_expression) {
	if (p_type != visual_shader->get_shader_type() || !links.has(p_node_id) || !links[p_node_id].expression_edit) {
		return;
	}
	links[p_node_id].expression_edit->set_text(p_expression);
}

Ref<Script> VisualShaderGraphPlugin::get_node_script(int p_node_id) const {
	if (!links.has(p_node_id)) {
		return Ref<Script>();
	}

	Ref<VisualShaderNodeCustom> custom = Ref<VisualShaderNodeCustom>(links[p_node_id].visual_node);
	if (custom.is_valid()) {
		return custom->get_script();
	}

	return Ref<Script>();
}

void VisualShaderGraphPlugin::update_node_size(int p_node_id) {
	if (!links.has(p_node_id)) {
		return;
	}
	links[p_node_id].graph_element->reset_size();
}

void VisualShaderGraphPlugin::register_default_input_button(int p_node_id, int p_port_id, Button *p_button) {
	links[p_node_id].input_ports.insert(p_port_id, { p_button });
}

void VisualShaderGraphPlugin::register_expression_edit(int p_node_id, CodeEdit *p_expression_edit) {
	links[p_node_id].expression_edit = p_expression_edit;
}

void VisualShaderGraphPlugin::register_curve_editor(int p_node_id, int p_index, CurveEditor *p_curve_editor) {
	links[p_node_id].curve_editors[p_index] = p_curve_editor;
}

void VisualShaderGraphPlugin::update_parameter_refs() {
	for (KeyValue<int, Link> &E : links) {
		VisualShaderNodeParameterRef *ref = Object::cast_to<VisualShaderNodeParameterRef>(E.value.visual_node);
		if (ref) {
			remove_node(E.value.type, E.key, true);
			add_node(E.value.type, E.key, true);
		}
	}
}

VisualShader::Type VisualShaderGraphPlugin::get_shader_type() const {
	return visual_shader->get_shader_type();
}

void VisualShaderGraphPlugin::set_node_position(VisualShader::Type p_type, int p_id, const Vector2 &p_position) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_id)) {
		links[p_id].graph_element->set_position_offset(p_position);
	}
}

bool VisualShaderGraphPlugin::is_preview_visible(int p_id) const {
	return links[p_id].preview_visible;
}

void VisualShaderGraphPlugin::clear_links() {
	links.clear();
}

void VisualShaderGraphPlugin::register_link(VisualShader::Type p_type, int p_id, VisualShaderNode *p_visual_node, GraphElement *p_graph_element) {
	links.insert(p_id, { p_type, p_visual_node, p_graph_element, p_visual_node->get_output_port_for_preview() != -1, -1, HashMap<int, InputPort>(), HashMap<int, Port>(), nullptr, nullptr, nullptr, { nullptr, nullptr, nullptr } });
}

void VisualShaderGraphPlugin::register_output_port(int p_node_id, int p_port, TextureButton *p_button) {
	links[p_node_id].output_ports.insert(p_port, { p_button });
}

void VisualShaderGraphPlugin::register_parameter_name(int p_node_id, LineEdit *p_parameter_name) {
	links[p_node_id].parameter_name = p_parameter_name;
}

void VisualShaderGraphPlugin::update_theme() {
	vector_expanded_color[0] = editor->get_theme_color(SNAME("axis_x_color"), EditorStringName(Editor)); // red
	vector_expanded_color[1] = editor->get_theme_color(SNAME("axis_y_color"), EditorStringName(Editor)); // green
	vector_expanded_color[2] = editor->get_theme_color(SNAME("axis_z_color"), EditorStringName(Editor)); // blue
	vector_expanded_color[3] = editor->get_theme_color(SNAME("axis_w_color"), EditorStringName(Editor)); // alpha
}

bool VisualShaderGraphPlugin::is_node_has_parameter_instances_relatively(VisualShader::Type p_type, int p_node) const {
	bool result = false;

	Ref<VisualShaderNodeParameter> parameter_node = Object::cast_to<VisualShaderNodeParameter>(visual_shader->get_node_unchecked(p_type, p_node).ptr());
	if (parameter_node.is_valid()) {
		if (parameter_node->get_qualifier() == VisualShaderNodeParameter::QUAL_INSTANCE) {
			return true;
		}
	}

	const LocalVector<int> &prev_connected_nodes = visual_shader->get_prev_connected_nodes(p_type, p_node);

	for (const int &E : prev_connected_nodes) {
		result = is_node_has_parameter_instances_relatively(p_type, E);
		if (result) {
			break;
		}
	}

	return result;
}

void VisualShaderGraphPlugin::add_node(VisualShader::Type p_type, int p_id, bool p_just_update) {
	if (!visual_shader.is_valid() || p_type != visual_shader->get_shader_type()) {
		return;
	}
	GraphEdit *graph = editor->graph;
	if (!graph) {
		return;
	}
	VisualShaderGraphPlugin *graph_plugin = editor->get_graph_plugin();
	if (!graph_plugin) {
		return;
	}
	Shader::Mode mode = visual_shader->get_mode();

	Control *offset;

	static const Color type_color[] = {
		Color(0.38, 0.85, 0.96), // scalar (float)
		Color(0.49, 0.78, 0.94), // scalar (int)
		Color(0.20, 0.88, 0.67), // scalar (uint)
		Color(0.74, 0.57, 0.95), // vector2
		Color(0.84, 0.49, 0.93), // vector3
		Color(1.0, 0.125, 0.95), // vector4
		Color(0.55, 0.65, 0.94), // boolean
		Color(0.96, 0.66, 0.43), // transform
		Color(1.0, 1.0, 0.0), // sampler
	};

	static const String vector_expanded_name[4] = {
		"red",
		"green",
		"blue",
		"alpha"
	};

	// Visual shader specific theme for MSDF font.
	Ref<Theme> vstheme;
	vstheme.instantiate();
	Ref<Font> label_font = EditorNode::get_singleton()->get_editor_theme()->get_font("main_msdf", EditorStringName(EditorFonts));
	Ref<Font> label_bold_font = EditorNode::get_singleton()->get_editor_theme()->get_font("main_bold_msdf", EditorStringName(EditorFonts));
	vstheme->set_font("font", "Label", label_font);
	vstheme->set_font("font", "GraphNodeTitleLabel", label_bold_font);
	vstheme->set_font("font", "LineEdit", label_font);
	vstheme->set_font("font", "Button", label_font);

	Ref<VisualShaderNode> vsnode = visual_shader->get_node(p_type, p_id);

	Ref<VisualShaderNodeResizableBase> resizable_node = vsnode;
	bool is_resizable = resizable_node.is_valid();
	Size2 size = Size2(0, 0);

	Ref<VisualShaderNodeGroupBase> group_node = vsnode;
	bool is_group = group_node.is_valid();

	Ref<VisualShaderNodeComment> comment_node = vsnode;
	bool is_comment = comment_node.is_valid();

	Ref<VisualShaderNodeExpression> expression_node = group_node;
	bool is_expression = expression_node.is_valid();
	String expression = "";

	Ref<VisualShaderNodeCustom> custom_node = vsnode;
	if (custom_node.is_valid()) {
		custom_node->_set_initialized(true);
	}

	GraphNode *node = memnew(GraphNode);
	node->set_title(vsnode->get_caption());

	// All nodes are closable except the output node.
	if (p_id >= 2) {
		vsnode->set_closable(true);
		node->connect("delete_request", callable_mp(editor, &VisualShaderEditor::_delete_node_request).bind(p_type, p_id), CONNECT_DEFERRED);
	}
	graph->add_child(node);
	node->set_theme(vstheme);

	if (p_just_update) {
		Link &link = links[p_id];

		link.graph_element = node;
		link.preview_box = nullptr;
		link.preview_pos = -1;
		link.output_ports.clear();
		link.input_ports.clear();
	} else {
		register_link(p_type, p_id, vsnode.ptr(), node);
	}

	if (is_resizable) {
		size = resizable_node->get_size();

		node->set_resizable(true);
		node->connect("resize_request", callable_mp(editor, &VisualShaderEditor::_node_resized).bind((int)p_type, p_id));
	}
	if (is_expression) {
		expression = expression_node->get_expression();
	}

	if (is_comment) {
		node->set_visible(false);
	}

	node->set_position_offset(visual_shader->get_node_position(p_type, p_id));

	node->set_name(itos(p_id));

	node->connect("dragged", callable_mp(editor, &VisualShaderEditor::_node_dragged).bind(p_id));

	Control *custom_editor = nullptr;
	int port_offset = 1;

	if (is_resizable) {
		editor->call_deferred(SNAME("_set_node_size"), (int)p_type, p_id, size);
	}

	Control *content_offset = memnew(Control);
	content_offset->set_custom_minimum_size(Size2(0, 5 * EDSCALE));
	node->add_child(content_offset);

	if (is_group) {
		port_offset += 1;
	}

	Ref<VisualShaderNodeParticleEmit> emit = vsnode;
	if (emit.is_valid()) {
		node->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
	}

	Ref<VisualShaderNodeParameterRef> parameter_ref = vsnode;
	if (parameter_ref.is_valid()) {
		parameter_ref->set_shader_rid(visual_shader->get_rid());
		parameter_ref->update_parameter_type();
	}

	Ref<VisualShaderNodeParameter> parameter = vsnode;
	HBoxContainer *hb = nullptr;

	if (parameter.is_valid()) {
		LineEdit *parameter_name = memnew(LineEdit);
		register_parameter_name(p_id, parameter_name);
		parameter_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		parameter_name->set_text(parameter->get_parameter_name());
		parameter_name->connect("text_submitted", callable_mp(editor, &VisualShaderEditor::_parameter_line_edit_changed).bind(p_id));
		parameter_name->connect("focus_exited", callable_mp(editor, &VisualShaderEditor::_parameter_line_edit_focus_out).bind(parameter_name, p_id));

		if (vsnode->get_output_port_count() == 1 && vsnode->get_output_port_name(0) == "") {
			hb = memnew(HBoxContainer);
			hb->add_child(parameter_name);
			node->add_child(hb);
		} else {
			node->add_child(parameter_name);
		}
		port_offset++;
	}

	for (int i = 0; i < editor->plugins.size(); i++) {
		vsnode->set_meta("id", p_id);
		vsnode->set_meta("shader_type", (int)p_type);
		custom_editor = editor->plugins.write[i]->create_editor(visual_shader, vsnode);
		vsnode->remove_meta("id");
		vsnode->remove_meta("shader_type");
		if (custom_editor) {
			if (vsnode->is_show_prop_names()) {
				custom_editor->call_deferred(SNAME("_show_prop_names"), true);
			}
			break;
		}
	}

	if (custom_node.is_valid()) {
		bool first = true;
		VBoxContainer *vbox = nullptr;

		for (int i = 0; i < custom_node->dp_props.size(); i++) {
			const VisualShaderNodeCustom::DropDownListProperty &dp = custom_node->dp_props[i];

			if (first) {
				first = false;
				vbox = memnew(VBoxContainer);
				node->add_child(vbox);
				port_offset++;
			}

			HBoxContainer *hbox = memnew(HBoxContainer);
			vbox->add_child(hbox);
			hbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);

			String prop_name = dp.name.strip_edges();
			if (!prop_name.is_empty()) {
				Label *label = memnew(Label);
				label->set_auto_translate(false); // TODO: Implement proper translation switch.
				label->set_text(prop_name + ":");
				hbox->add_child(label);
			}

			OptionButton *op = memnew(OptionButton);
			hbox->add_child(op);
			op->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			op->connect("item_selected", callable_mp(editor, &VisualShaderEditor::_set_custom_node_option).bind(p_id, i), CONNECT_DEFERRED);

			for (const String &s : dp.options) {
				op->add_item(s);
			}
			if (custom_node->dp_selected_cache.has(i)) {
				op->select(custom_node->dp_selected_cache[i]);
			} else {
				op->select(0);
			}
		}
	}

	Ref<VisualShaderNodeCurveTexture> curve = vsnode;
	Ref<VisualShaderNodeCurveXYZTexture> curve_xyz = vsnode;

	bool is_curve = curve.is_valid() || curve_xyz.is_valid();
	if (is_curve) {
		hb = memnew(HBoxContainer);
		node->add_child(hb);
	}

	if (curve.is_valid()) {
		custom_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		if (curve->get_texture().is_valid()) {
			curve->get_texture()->connect_changed(callable_mp(graph_plugin, &VisualShaderGraphPlugin::update_curve).bind(p_id));
		}

		CurveEditor *curve_editor = memnew(CurveEditor);
		node->add_child(curve_editor);
		register_curve_editor(p_id, 0, curve_editor);
		curve_editor->set_custom_minimum_size(Size2(300, 0));
		curve_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		if (curve->get_texture().is_valid()) {
			curve_editor->set_curve(curve->get_texture()->get_curve());
		}
	}

	if (curve_xyz.is_valid()) {
		custom_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		if (curve_xyz->get_texture().is_valid()) {
			curve_xyz->get_texture()->connect_changed(callable_mp(graph_plugin, &VisualShaderGraphPlugin::update_curve_xyz).bind(p_id));
		}

		CurveEditor *curve_editor_x = memnew(CurveEditor);
		node->add_child(curve_editor_x);
		register_curve_editor(p_id, 0, curve_editor_x);
		curve_editor_x->set_custom_minimum_size(Size2(300, 0));
		curve_editor_x->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		if (curve_xyz->get_texture().is_valid()) {
			curve_editor_x->set_curve(curve_xyz->get_texture()->get_curve_x());
		}

		CurveEditor *curve_editor_y = memnew(CurveEditor);
		node->add_child(curve_editor_y);
		register_curve_editor(p_id, 1, curve_editor_y);
		curve_editor_y->set_custom_minimum_size(Size2(300, 0));
		curve_editor_y->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		if (curve_xyz->get_texture().is_valid()) {
			curve_editor_y->set_curve(curve_xyz->get_texture()->get_curve_y());
		}

		CurveEditor *curve_editor_z = memnew(CurveEditor);
		node->add_child(curve_editor_z);
		register_curve_editor(p_id, 2, curve_editor_z);
		curve_editor_z->set_custom_minimum_size(Size2(300, 0));
		curve_editor_z->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		if (curve_xyz->get_texture().is_valid()) {
			curve_editor_z->set_curve(curve_xyz->get_texture()->get_curve_z());
		}
	}

	if (custom_editor) {
		if (is_curve || (hb == nullptr && !vsnode->is_use_prop_slots() && (vsnode->get_output_port_count() == 0 || vsnode->get_output_port_name(0) == "") && (vsnode->get_input_port_count() == 0 || vsnode->get_input_port_name(0) == ""))) {
			// Will be embedded in first port.
		} else {
			port_offset++;
			node->add_child(custom_editor);
			custom_editor = nullptr;
		}
	}

	if (is_group) {
		if (group_node->is_editable()) {
			HBoxContainer *hb2 = memnew(HBoxContainer);

			String input_port_name = "input" + itos(group_node->get_free_input_port_id());
			String output_port_name = "output" + itos(group_node->get_free_output_port_id());

			for (int i = 0; i < MAX(vsnode->get_input_port_count(), vsnode->get_output_port_count()); i++) {
				if (i < vsnode->get_input_port_count()) {
					if (input_port_name == vsnode->get_input_port_name(i)) {
						input_port_name = "_" + input_port_name;
					}
				}
				if (i < vsnode->get_output_port_count()) {
					if (output_port_name == vsnode->get_output_port_name(i)) {
						output_port_name = "_" + output_port_name;
					}
				}
			}

			Button *add_input_btn = memnew(Button);
			add_input_btn->set_text(TTR("Add Input"));
			add_input_btn->connect("pressed", callable_mp(editor, &VisualShaderEditor::_add_input_port).bind(p_id, group_node->get_free_input_port_id(), VisualShaderNode::PORT_TYPE_VECTOR_3D, input_port_name), CONNECT_DEFERRED);
			hb2->add_child(add_input_btn);

			hb2->add_spacer();

			Button *add_output_btn = memnew(Button);
			add_output_btn->set_text(TTR("Add Output"));
			add_output_btn->connect("pressed", callable_mp(editor, &VisualShaderEditor::_add_output_port).bind(p_id, group_node->get_free_output_port_id(), VisualShaderNode::PORT_TYPE_VECTOR_3D, output_port_name), CONNECT_DEFERRED);
			hb2->add_child(add_output_btn);

			node->add_child(hb2);
		}
	}

	int output_port_count = 0;
	for (int i = 0; i < vsnode->get_output_port_count(); i++) {
		if (vsnode->_is_output_port_expanded(i)) {
			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
					output_port_count += 2;
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
					output_port_count += 3;
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
					output_port_count += 4;
				} break;
				default:
					break;
			}
		}
		output_port_count++;
	}
	int max_ports = MAX(vsnode->get_input_port_count(), output_port_count);
	VisualShaderNode::PortType expanded_type = VisualShaderNode::PORT_TYPE_SCALAR;
	int expanded_port_counter = 0;

	for (int i = 0, j = 0; i < max_ports; i++, j++) {
		switch (expanded_type) {
			case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
				if (expanded_port_counter >= 2) {
					expanded_type = VisualShaderNode::PORT_TYPE_SCALAR;
					expanded_port_counter = 0;
					i -= 2;
				}
			} break;
			case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
				if (expanded_port_counter >= 3) {
					expanded_type = VisualShaderNode::PORT_TYPE_SCALAR;
					expanded_port_counter = 0;
					i -= 3;
				}
			} break;
			case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
				if (expanded_port_counter >= 4) {
					expanded_type = VisualShaderNode::PORT_TYPE_SCALAR;
					expanded_port_counter = 0;
					i -= 4;
				}
			} break;
			default:
				break;
		}

		if (vsnode->is_port_separator(i)) {
			node->add_child(memnew(HSeparator));
			port_offset++;
		}

		bool valid_left = j < vsnode->get_input_port_count();
		VisualShaderNode::PortType port_left = VisualShaderNode::PORT_TYPE_SCALAR;
		bool port_left_used = false;
		String name_left;
		if (valid_left) {
			name_left = vsnode->get_input_port_name(i);
			port_left = vsnode->get_input_port_type(i);
			for (const VisualShader::Connection &E : connections) {
				if (E.to_node == p_id && E.to_port == j) {
					port_left_used = true;
					break;
				}
			}
		}

		bool valid_right = true;
		VisualShaderNode::PortType port_right = VisualShaderNode::PORT_TYPE_SCALAR;
		String name_right;

		if (expanded_type == VisualShaderNode::PORT_TYPE_SCALAR) {
			valid_right = i < vsnode->get_output_port_count();
			if (valid_right) {
				name_right = vsnode->get_output_port_name(i);
				port_right = vsnode->get_output_port_type(i);
			}
		} else {
			name_right = vector_expanded_name[expanded_port_counter++];
		}

		bool is_first_hbox = false;
		if (i == 0 && hb != nullptr) {
			is_first_hbox = true;
		} else {
			hb = memnew(HBoxContainer);
		}
		hb->add_theme_constant_override("separation", 7 * EDSCALE);

		Variant default_value;

		if (valid_left && !port_left_used) {
			default_value = vsnode->get_input_port_default_value(i);
		}

		Button *button = memnew(Button);
		hb->add_child(button);
		register_default_input_button(p_id, i, button);
		button->connect("pressed", callable_mp(editor, &VisualShaderEditor::_edit_port_default_input).bind(button, p_id, i));
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
					type_box->add_item(TTR("UInt"));
					type_box->add_item(TTR("Vector2"));
					type_box->add_item(TTR("Vector3"));
					type_box->add_item(TTR("Vector4"));
					type_box->add_item(TTR("Boolean"));
					type_box->add_item(TTR("Transform"));
					type_box->add_item(TTR("Sampler"));
					type_box->select(group_node->get_input_port_type(i));
					type_box->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
					type_box->connect("item_selected", callable_mp(editor, &VisualShaderEditor::_change_input_port_type).bind(p_id, i), CONNECT_DEFERRED);

					LineEdit *name_box = memnew(LineEdit);
					hb->add_child(name_box);
					name_box->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
					name_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
					name_box->set_text(name_left);
					name_box->connect("text_submitted", callable_mp(editor, &VisualShaderEditor::_change_input_port_name).bind(name_box, p_id, i), CONNECT_DEFERRED);
					name_box->connect("focus_exited", callable_mp(editor, &VisualShaderEditor::_port_name_focus_out).bind(name_box, p_id, i, false), CONNECT_DEFERRED);

					Button *remove_btn = memnew(Button);
					remove_btn->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Remove"), EditorStringName(EditorIcons)));
					remove_btn->set_tooltip_text(TTR("Remove") + " " + name_left);
					remove_btn->connect("pressed", callable_mp(editor, &VisualShaderEditor::_remove_input_port).bind(p_id, i), CONNECT_DEFERRED);
					hb->add_child(remove_btn);
				} else {
					Label *label = memnew(Label);
					label->set_auto_translate(false); // TODO: Implement proper translation switch.
					label->set_text(name_left);
					label->add_theme_style_override("normal", editor->get_theme_stylebox(SNAME("label_style"), SNAME("VShaderEditor"))); //more compact
					hb->add_child(label);

					if (vsnode->is_input_port_default(i, mode) && !port_left_used) {
						Label *hint_label = memnew(Label);
						hint_label->set_text(TTR("[default]"));
						hint_label->add_theme_color_override("font_color", editor->get_theme_color(SNAME("font_readonly_color"), SNAME("TextEdit")));
						hint_label->add_theme_style_override("normal", editor->get_theme_stylebox(SNAME("label_style"), SNAME("VShaderEditor")));
						hb->add_child(hint_label);
					}
				}
			}

			if (!is_group && !is_first_hbox) {
				hb->add_spacer();
			}

			if (valid_right) {
				if (is_group) {
					Button *remove_btn = memnew(Button);
					remove_btn->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Remove"), EditorStringName(EditorIcons)));
					remove_btn->set_tooltip_text(TTR("Remove") + " " + name_left);
					remove_btn->connect("pressed", callable_mp(editor, &VisualShaderEditor::_remove_output_port).bind(p_id, i), CONNECT_DEFERRED);
					hb->add_child(remove_btn);

					LineEdit *name_box = memnew(LineEdit);
					hb->add_child(name_box);
					name_box->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
					name_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
					name_box->set_text(name_right);
					name_box->connect("text_submitted", callable_mp(editor, &VisualShaderEditor::_change_output_port_name).bind(name_box, p_id, i), CONNECT_DEFERRED);
					name_box->connect("focus_exited", callable_mp(editor, &VisualShaderEditor::_port_name_focus_out).bind(name_box, p_id, i, true), CONNECT_DEFERRED);

					OptionButton *type_box = memnew(OptionButton);
					hb->add_child(type_box);
					type_box->add_item(TTR("Float"));
					type_box->add_item(TTR("Int"));
					type_box->add_item(TTR("UInt"));
					type_box->add_item(TTR("Vector2"));
					type_box->add_item(TTR("Vector3"));
					type_box->add_item(TTR("Vector4"));
					type_box->add_item(TTR("Boolean"));
					type_box->add_item(TTR("Transform"));
					type_box->select(group_node->get_output_port_type(i));
					type_box->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
					type_box->connect("item_selected", callable_mp(editor, &VisualShaderEditor::_change_output_port_type).bind(p_id, i), CONNECT_DEFERRED);
				} else {
					Label *label = memnew(Label);
					label->set_auto_translate(false); // TODO: Implement proper translation switch.
					label->set_text(name_right);
					label->add_theme_style_override("normal", editor->get_theme_stylebox(SNAME("label_style"), SNAME("VShaderEditor"))); //more compact
					hb->add_child(label);
				}
			}
		}

		if (valid_right) {
			if (expanded_port_counter == 0 && vsnode->is_output_port_expandable(i)) {
				TextureButton *expand = memnew(TextureButton);
				expand->set_toggle_mode(true);
				expand->set_texture_normal(editor->get_editor_theme_icon(SNAME("GuiTreeArrowRight")));
				expand->set_texture_pressed(editor->get_editor_theme_icon(SNAME("GuiTreeArrowDown")));
				expand->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
				expand->set_pressed(vsnode->_is_output_port_expanded(i));
				expand->connect("pressed", callable_mp(editor, &VisualShaderEditor::_expand_output_port).bind(p_id, i, !vsnode->_is_output_port_expanded(i)), CONNECT_DEFERRED);
				hb->add_child(expand);
			}
			if (vsnode->has_output_port_preview(i) && port_right != VisualShaderNode::PORT_TYPE_TRANSFORM && port_right != VisualShaderNode::PORT_TYPE_SAMPLER) {
				TextureButton *preview = memnew(TextureButton);
				preview->set_toggle_mode(true);
				preview->set_texture_normal(editor->get_editor_theme_icon(SNAME("GuiVisibilityHidden")));
				preview->set_texture_pressed(editor->get_editor_theme_icon(SNAME("GuiVisibilityVisible")));
				preview->set_v_size_flags(Control::SIZE_SHRINK_CENTER);

				register_output_port(p_id, j, preview);

				preview->connect("pressed", callable_mp(editor, &VisualShaderEditor::_preview_select_port).bind(p_id, j), CONNECT_DEFERRED);
				hb->add_child(preview);
			}
		}

		if (is_group) {
			offset = memnew(Control);
			offset->set_custom_minimum_size(Size2(0, 5 * EDSCALE));
			node->add_child(offset);
			port_offset++;
		}

		if (!is_first_hbox) {
			node->add_child(hb);
			if (curve_xyz.is_valid()) {
				node->move_child(hb, 1 + expanded_port_counter);
			}
		}

		if (expanded_type != VisualShaderNode::PORT_TYPE_SCALAR) {
			continue;
		}

		int idx = 1;
		if (!is_first_hbox) {
			idx = i + port_offset;
		}
		if (!is_comment) {
			GraphNode *graph_node = Object::cast_to<GraphNode>(node);

			graph_node->set_slot(idx, valid_left, port_left, type_color[port_left], valid_right, port_right, type_color[port_right]);

			if (vsnode->_is_output_port_expanded(i)) {
				switch (vsnode->get_output_port_type(i)) {
					case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
						port_offset++;
						valid_left = (i + 1) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 1);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[0]);
						port_offset++;

						valid_left = (i + 2) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 2);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[1]);

						expanded_type = VisualShaderNode::PORT_TYPE_VECTOR_2D;
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
						port_offset++;
						valid_left = (i + 1) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 1);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[0]);
						port_offset++;

						valid_left = (i + 2) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 2);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[1]);
						port_offset++;

						valid_left = (i + 3) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 3);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[2]);

						expanded_type = VisualShaderNode::PORT_TYPE_VECTOR_3D;
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
						port_offset++;
						valid_left = (i + 1) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 1);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[0]);
						port_offset++;

						valid_left = (i + 2) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 2);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[1]);
						port_offset++;

						valid_left = (i + 3) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 3);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[2]);
						port_offset++;

						valid_left = (i + 4) < vsnode->get_input_port_count();
						port_left = VisualShaderNode::PORT_TYPE_SCALAR;
						if (valid_left) {
							port_left = vsnode->get_input_port_type(i + 4);
						}
						graph_node->set_slot(i + port_offset, valid_left, port_left, type_color[port_left], true, VisualShaderNode::PORT_TYPE_SCALAR, vector_expanded_color[3]);

						expanded_type = VisualShaderNode::PORT_TYPE_VECTOR_4D;
					} break;
					default:
						break;
				}
			}
		}
	}

	bool has_relative_parameter_instances = false;
	if (vsnode->get_output_port_for_preview() >= 0) {
		has_relative_parameter_instances = is_node_has_parameter_instances_relatively(p_type, p_id);
		show_port_preview(p_type, p_id, vsnode->get_output_port_for_preview(), !has_relative_parameter_instances);
	} else {
		offset = memnew(Control);
		offset->set_custom_minimum_size(Size2(0, 4 * EDSCALE));
		node->add_child(offset);
	}

	String error = vsnode->get_warning(mode, p_type);
	if (has_relative_parameter_instances) {
		error += "\n" + TTR("The 2D preview cannot correctly show the result retrieved from instance parameter.");
	}
	if (!error.is_empty()) {
		Label *error_label = memnew(Label);
		error_label->add_theme_color_override("font_color", editor->get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		error_label->set_text(error);
		error_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
		node->add_child(error_label);
	}

	if (is_expression) {
		CodeEdit *expression_box = memnew(CodeEdit);
		Ref<CodeHighlighter> expression_syntax_highlighter;
		expression_syntax_highlighter.instantiate();
		expression_node->set_ctrl_pressed(expression_box, 0);
		node->add_child(expression_box);
		register_expression_edit(p_id, expression_box);

		Color background_color = EDITOR_GET("text_editor/theme/highlighting/background_color");
		Color text_color = EDITOR_GET("text_editor/theme/highlighting/text_color");
		Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
		Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");
		Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
		Color symbol_color = EDITOR_GET("text_editor/theme/highlighting/symbol_color");
		Color function_color = EDITOR_GET("text_editor/theme/highlighting/function_color");
		Color number_color = EDITOR_GET("text_editor/theme/highlighting/number_color");
		Color members_color = EDITOR_GET("text_editor/theme/highlighting/member_variable_color");

		expression_box->set_syntax_highlighter(expression_syntax_highlighter);
		expression_box->add_theme_color_override("background_color", background_color);

		for (const String &E : editor->keyword_list) {
			if (ShaderLanguage::is_control_flow_keyword(E)) {
				expression_syntax_highlighter->add_keyword_color(E, control_flow_keyword_color);
			} else {
				expression_syntax_highlighter->add_keyword_color(E, keyword_color);
			}
		}

		expression_box->begin_bulk_theme_override();
		expression_box->add_theme_font_override("font", editor->get_theme_font(SNAME("expression"), EditorStringName(EditorFonts)));
		expression_box->add_theme_font_size_override("font_size", editor->get_theme_font_size(SNAME("expression_size"), EditorStringName(EditorFonts)));
		expression_box->add_theme_color_override("font_color", text_color);
		expression_box->end_bulk_theme_override();

		expression_syntax_highlighter->set_number_color(number_color);
		expression_syntax_highlighter->set_symbol_color(symbol_color);
		expression_syntax_highlighter->set_function_color(function_color);
		expression_syntax_highlighter->set_member_variable_color(members_color);
		expression_syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
		expression_syntax_highlighter->add_color_region("//", "", comment_color, true);

		expression_box->clear_comment_delimiters();
		expression_box->add_comment_delimiter("/*", "*/", false);
		expression_box->add_comment_delimiter("//", "", true);

		if (!expression_box->has_auto_brace_completion_open_key("/*")) {
			expression_box->add_auto_brace_completion_pair("/*", "*/");
		}

		expression_box->set_text(expression);
		expression_box->set_context_menu_enabled(false);
		expression_box->set_draw_line_numbers(true);

		expression_box->connect("focus_exited", callable_mp(editor, &VisualShaderEditor::_expression_focus_out).bind(expression_box, p_id));
	}
}

void VisualShaderGraphPlugin::remove_node(VisualShader::Type p_type, int p_id, bool p_just_update) {
	if (visual_shader->get_shader_type() == p_type && links.has(p_id)) {
		Node *graph_edit_node = links[p_id].graph_element->get_parent();
		graph_edit_node->remove_child(links[p_id].graph_element);
		memdelete(links[p_id].graph_element);
		if (!p_just_update) {
			links.erase(p_id);
		}
	}
}

void VisualShaderGraphPlugin::connect_nodes(VisualShader::Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	GraphEdit *graph = editor->graph;
	if (!graph) {
		return;
	}

	if (visual_shader.is_valid() && visual_shader->get_shader_type() == p_type) {
		graph->connect_node(itos(p_from_node), p_from_port, itos(p_to_node), p_to_port);

		connections.push_back({ p_from_node, p_from_port, p_to_node, p_to_port });
		if (links[p_to_node].input_ports.has(p_to_port) && links[p_to_node].input_ports[p_to_port].default_input_button != nullptr) {
			links[p_to_node].input_ports[p_to_port].default_input_button->hide();
		}
	}
}

void VisualShaderGraphPlugin::disconnect_nodes(VisualShader::Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	GraphEdit *graph = editor->graph;
	if (!graph) {
		return;
	}

	if (visual_shader.is_valid() && visual_shader->get_shader_type() == p_type) {
		graph->disconnect_node(itos(p_from_node), p_from_port, itos(p_to_node), p_to_port);

		for (const List<VisualShader::Connection>::Element *E = connections.front(); E; E = E->next()) {
			if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
				connections.erase(E);
				break;
			}
		}
		if (links[p_to_node].input_ports.has(p_to_port) && links[p_to_node].input_ports[p_to_port].default_input_button != nullptr && links[p_to_node].visual_node->get_input_port_default_value(p_to_port).get_type() != Variant::NIL) {
			links[p_to_node].input_ports[p_to_port].default_input_button->show();
			set_input_port_default_value(p_type, p_to_node, p_to_port, links[p_to_node].visual_node->get_input_port_default_value(p_to_port));
		}
	}
}

VisualShaderGraphPlugin::~VisualShaderGraphPlugin() {
}

/////////////////

void VisualShaderEditedProperty::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_edited_property", "value"), &VisualShaderEditedProperty::set_edited_property);
	ClassDB::bind_method(D_METHOD("get_edited_property"), &VisualShaderEditedProperty::get_edited_property);

	ADD_PROPERTY(PropertyInfo(Variant::NIL, "edited_property", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "set_edited_property", "get_edited_property");
}

void VisualShaderEditedProperty::set_edited_property(Variant p_variant) {
	edited_property = p_variant;
}

Variant VisualShaderEditedProperty::get_edited_property() const {
	return edited_property;
}

/////////////////

Vector2 VisualShaderEditor::selection_center;
List<VisualShaderEditor::CopyItem> VisualShaderEditor::copy_items_buffer;
List<VisualShader::Connection> VisualShaderEditor::copy_connections_buffer;

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

		visual_shader->connect_changed(callable_mp(this, &VisualShaderEditor::_update_preview));
		visual_shader->set_graph_offset(graph->get_scroll_offset() / EDSCALE);
		_set_mode(visual_shader->get_mode());

		_update_nodes();
	} else {
		if (visual_shader.is_valid()) {
			visual_shader->disconnect_changed(callable_mp(this, &VisualShaderEditor::_update_preview));
		}
		visual_shader.unref();
	}

	if (visual_shader.is_null()) {
		hide();
	} else {
		if (changed) { // to avoid tree collapse
			_update_varying_tree();
			_update_options_menu();
			_update_preview();
			_update_graph();
		}
	}
}

void VisualShaderEditor::add_plugin(const Ref<VisualShaderNodePlugin> &p_plugin) {
	if (plugins.has(p_plugin)) {
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
			add_options.remove_at(i);
			i--;
		}
	}
}

void VisualShaderEditor::add_custom_type(const String &p_name, const String &p_type, const Ref<Script> &p_script, const String &p_description, int p_return_icon_type, const String &p_category, bool p_highend) {
	ERR_FAIL_COND(!p_name.is_valid_identifier());
	ERR_FAIL_COND(p_type.is_empty() && !p_script.is_valid());

	for (int i = 0; i < add_options.size(); i++) {
		const AddOption &op = add_options[i];

		if (op.is_custom) {
			if (!p_type.is_empty()) {
				if (op.type == p_type) {
					return;
				}
			} else if (op.script == p_script) {
				return;
			}
		}
	}

	AddOption ao;
	ao.name = p_name;
	ao.type = p_type;
	ao.script = p_script;
	ao.return_type = p_return_icon_type;
	ao.description = p_description;
	ao.category = p_category;
	ao.highend = p_highend;
	ao.is_custom = true;
	ao.is_native = !p_type.is_empty();

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

Dictionary VisualShaderEditor::get_custom_node_data(Ref<VisualShaderNodeCustom> &p_custom_node) {
	Dictionary dict;
	dict["script"] = p_custom_node->get_script();
	dict["name"] = p_custom_node->_get_name();
	dict["description"] = p_custom_node->_get_description();
	dict["return_icon_type"] = p_custom_node->_get_return_icon_type();
	dict["highend"] = p_custom_node->_is_highend();

	String category = p_custom_node->_get_category();
	category = category.rstrip("/");
	category = category.lstrip("/");
	category = "Addons/" + category;
	if (p_custom_node->has_method("_get_subcategory")) {
		String subcategory = (String)p_custom_node->call("_get_subcategory");
		if (!subcategory.is_empty()) {
			category += "/" + subcategory;
		}
	}
	dict["category"] = category;

	return dict;
}

void VisualShaderEditor::_get_current_mode_limits(int &r_begin_type, int &r_end_type) const {
	switch (visual_shader->get_mode()) {
		case Shader::MODE_CANVAS_ITEM:
		case Shader::MODE_SPATIAL: {
			r_begin_type = 0;
			r_end_type = 3;
		} break;
		case Shader::MODE_PARTICLES: {
			r_begin_type = 3;
			r_end_type = 5 + r_begin_type;
		} break;
		case Shader::MODE_SKY: {
			r_begin_type = 8;
			r_end_type = 1 + r_begin_type;
		} break;
		case Shader::MODE_FOG: {
			r_begin_type = 9;
			r_end_type = 1 + r_begin_type;
		} break;
		default: {
		} break;
	}
}

void VisualShaderEditor::_script_created(const Ref<Script> &p_script) {
	if (p_script.is_null() || p_script->get_instance_base_type() != "VisualShaderNodeCustom") {
		return;
	}
	Ref<VisualShaderNodeCustom> ref;
	ref.instantiate();
	ref->set_script(p_script);

	Dictionary dict = get_custom_node_data(ref);
	add_custom_type(dict["name"], String(), dict["script"], dict["description"], dict["return_icon_type"], dict["category"], dict["highend"]);

	_update_options_menu();
}

void VisualShaderEditor::_update_custom_script(const Ref<Script> &p_script) {
	if (p_script.is_null() || p_script->get_instance_base_type() != "VisualShaderNodeCustom") {
		return;
	}

	Ref<VisualShaderNodeCustom> ref;
	ref.instantiate();
	ref->set_script(p_script);
	if (!ref->is_available(visual_shader->get_mode(), visual_shader->get_shader_type())) {
		for (int i = 0; i < add_options.size(); i++) {
			if (add_options[i].is_custom && add_options[i].script == p_script) {
				add_options.remove_at(i);
				_update_options_menu();
				// TODO: Make indication for the existed custom nodes with that script on graph to be disabled.
				break;
			}
		}
		return;
	}
	Dictionary dict = get_custom_node_data(ref);

	bool found_type = false;
	bool need_rebuild = false;

	for (int i = custom_node_option_idx; i < add_options.size(); i++) {
		if (add_options[i].script == p_script) {
			found_type = true;

			add_options.write[i].name = dict["name"];
			add_options.write[i].return_type = dict["return_icon_type"];
			add_options.write[i].description = dict["description"];
			add_options.write[i].category = dict["category"];
			add_options.write[i].highend = dict["highend"];

			int begin_type = 0;
			int end_type = 0;
			_get_current_mode_limits(begin_type, end_type);

			for (int t = begin_type; t < end_type; t++) {
				VisualShader::Type type = (VisualShader::Type)t;
				Vector<int> nodes = visual_shader->get_node_list(type);

				List<VisualShader::Connection> node_connections;
				visual_shader->get_node_connections(type, &node_connections);

				List<VisualShader::Connection> custom_node_input_connections;
				List<VisualShader::Connection> custom_node_output_connections;

				for (const VisualShader::Connection &E : node_connections) {
					int from = E.from_node;
					int from_port = E.from_port;
					int to = E.to_node;
					int to_port = E.to_port;

					if (graph_plugin->get_node_script(from) == p_script) {
						custom_node_output_connections.push_back({ from, from_port, to, to_port });
					} else if (graph_plugin->get_node_script(to) == p_script) {
						custom_node_input_connections.push_back({ from, from_port, to, to_port });
					}
				}

				for (int node_id : nodes) {
					Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, node_id);
					if (vsnode.is_null()) {
						continue;
					}
					Ref<VisualShaderNodeCustom> custom_node = vsnode;
					if (custom_node.is_null() || custom_node->get_script() != p_script) {
						continue;
					}
					need_rebuild = true;

					// Removes invalid connections.
					{
						int prev_input_port_count = custom_node->get_input_port_count();
						int prev_output_port_count = custom_node->get_output_port_count();

						custom_node->update_ports();

						int input_port_count = custom_node->get_input_port_count();
						int output_port_count = custom_node->get_output_port_count();

						if (output_port_count != prev_output_port_count) {
							for (const VisualShader::Connection &E : custom_node_output_connections) {
								int from = E.from_node;
								int from_idx = E.from_port;
								int to = E.to_node;
								int to_idx = E.to_port;

								if (from_idx >= output_port_count) {
									visual_shader->disconnect_nodes(type, from, from_idx, to, to_idx);
									graph_plugin->disconnect_nodes(type, from, from_idx, to, to_idx);
								}
							}
						}
						if (input_port_count != prev_input_port_count) {
							for (const VisualShader::Connection &E : custom_node_input_connections) {
								int from = E.from_node;
								int from_idx = E.from_port;
								int to = E.to_node;
								int to_idx = E.to_port;

								if (to_idx >= input_port_count) {
									visual_shader->disconnect_nodes(type, from, from_idx, to, to_idx);
									graph_plugin->disconnect_nodes(type, from, from_idx, to, to_idx);
								}
							}
						}
					}

					graph_plugin->update_node(type, node_id);
				}
			}
			break;
		}
	}

	if (!found_type) {
		add_custom_type(dict["name"], String(), dict["script"], dict["description"], dict["return_icon_type"], dict["category"], dict["highend"]);
	}

	// To prevent updating options multiple times when multiple scripts are saved.
	if (!_block_update_options_menu) {
		_block_update_options_menu = true;

		call_deferred(SNAME("_update_options_menu_deferred"));
	}

	// To prevent rebuilding the shader multiple times when multiple scripts are saved.
	if (need_rebuild && !_block_rebuild_shader) {
		_block_rebuild_shader = true;

		call_deferred(SNAME("_rebuild_shader_deferred"));
	}
}

void VisualShaderEditor::_resource_saved(const Ref<Resource> &p_resource) {
	_update_custom_script(Ref<Script>(p_resource.ptr()));
}

void VisualShaderEditor::_resources_removed() {
	bool has_any_instance = false;

	for (const Ref<Script> &scr : custom_scripts_to_delete) {
		for (int i = custom_node_option_idx; i < add_options.size(); i++) {
			if (add_options[i].script == scr) {
				add_options.remove_at(i);

				// Removes all node instances using that script from the graph.
				{
					int begin_type = 0;
					int end_type = 0;
					_get_current_mode_limits(begin_type, end_type);

					for (int t = begin_type; t < end_type; t++) {
						VisualShader::Type type = (VisualShader::Type)t;

						List<VisualShader::Connection> node_connections;
						visual_shader->get_node_connections(type, &node_connections);

						for (const VisualShader::Connection &E : node_connections) {
							int from = E.from_node;
							int from_port = E.from_port;
							int to = E.to_node;
							int to_port = E.to_port;

							if (graph_plugin->get_node_script(from) == scr || graph_plugin->get_node_script(to) == scr) {
								visual_shader->disconnect_nodes(type, from, from_port, to, to_port);
								graph_plugin->disconnect_nodes(type, from, from_port, to, to_port);
							}
						}

						Vector<int> nodes = visual_shader->get_node_list(type);
						for (int node_id : nodes) {
							Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, node_id);
							if (vsnode.is_null()) {
								continue;
							}
							Ref<VisualShaderNodeCustom> custom_node = vsnode;
							if (custom_node.is_null() || custom_node->get_script() != scr) {
								continue;
							}
							visual_shader->remove_node(type, node_id);
							graph_plugin->remove_node(type, node_id, false);

							has_any_instance = true;
						}
					}
				}

				break;
			}
		}
	}
	if (has_any_instance) {
		EditorUndoRedoManager::get_singleton()->clear_history(); // Need to clear undo history, otherwise it may lead to hard-detected errors and crashes (since the script was removed).
		ResourceSaver::save(visual_shader, visual_shader->get_path());
	}
	_update_options_menu();

	custom_scripts_to_delete.clear();
	pending_custom_scripts_to_delete = false;
}

void VisualShaderEditor::_resource_removed(const Ref<Resource> &p_resource) {
	Ref<Script> scr = Ref<Script>(p_resource.ptr());
	if (scr.is_null() || scr->get_instance_base_type() != "VisualShaderNodeCustom") {
		return;
	}

	custom_scripts_to_delete.push_back(scr);

	if (!pending_custom_scripts_to_delete) {
		pending_custom_scripts_to_delete = true;

		call_deferred("_resources_removed");
	}
}

void VisualShaderEditor::_update_options_menu_deferred() {
	_update_options_menu();

	_block_update_options_menu = false;
}

void VisualShaderEditor::_rebuild_shader_deferred() {
	if (visual_shader.is_valid()) {
		visual_shader->rebuild();
	}

	_block_rebuild_shader = false;
}

bool VisualShaderEditor::_is_available(int p_mode) {
	int current_mode = edit_type->get_selected();

	if (p_mode != -1) {
		switch (current_mode) {
			case 0: // Vertex / Emit
				current_mode = 1;
				break;
			case 1: // Fragment / Process
				current_mode = 2;
				break;
			case 2: // Light / Collide
				current_mode = 4;
				break;
			default:
				break;
		}
	}

	return (p_mode == -1 || (p_mode & current_mode) != 0);
}

void VisualShaderEditor::_update_nodes() {
	clear_custom_types();
	Dictionary added;

	// Add GDScript classes.
	{
		List<StringName> class_list;
		ScriptServer::get_global_class_list(&class_list);

		for (int i = 0; i < class_list.size(); i++) {
			if (ScriptServer::get_global_class_native_base(class_list[i]) == "VisualShaderNodeCustom") {
				String script_path = ScriptServer::get_global_class_path(class_list[i]);
				Ref<Resource> res = ResourceLoader::load(script_path);
				ERR_CONTINUE(res.is_null());
				ERR_CONTINUE(!res->is_class("Script"));
				Ref<Script> scr = Ref<Script>(res);

				Ref<VisualShaderNodeCustom> ref;
				ref.instantiate();
				ref->set_script(scr);
				if (!ref->is_available(visual_shader->get_mode(), visual_shader->get_shader_type())) {
					continue;
				}
				Dictionary dict = get_custom_node_data(ref);
				dict["type"] = String();

				String key;
				key = String(dict["category"]) + "/" + String(dict["name"]);

				added[key] = dict;
			}
		}
	}

	// Add GDExtension classes.
	{
		List<StringName> class_list;
		ClassDB::get_class_list(&class_list);

		for (int i = 0; i < class_list.size(); i++) {
			if (ClassDB::get_parent_class(class_list[i]) == "VisualShaderNodeCustom") {
				Object *instance = ClassDB::instantiate(class_list[i]);
				Ref<VisualShaderNodeCustom> ref = Object::cast_to<VisualShaderNodeCustom>(instance);
				ERR_CONTINUE(ref.is_null());
				if (!ref->is_available(visual_shader->get_mode(), visual_shader->get_shader_type())) {
					continue;
				}
				Dictionary dict = get_custom_node_data(ref);
				dict["type"] = class_list[i];
				dict["script"] = Ref<Script>();

				String key;
				key = String(dict["category"]) + "/" + String(dict["name"]);

				added[key] = dict;
			}
		}
	}

	// Disables not-supported copied items.
	{
		for (CopyItem &item : copy_items_buffer) {
			Ref<VisualShaderNodeCustom> custom = Object::cast_to<VisualShaderNodeCustom>(item.node.ptr());

			if (custom.is_valid()) {
				if (!custom->is_available(visual_shader->get_mode(), visual_shader->get_shader_type())) {
					item.disabled = true;
				} else {
					item.disabled = false;
				}
			} else {
				for (int i = 0; i < add_options.size(); i++) {
					if (add_options[i].type == item.node->get_class_name()) {
						if ((add_options[i].func != visual_shader->get_mode() && add_options[i].func != -1) || !_is_available(add_options[i].mode)) {
							item.disabled = true;
						} else {
							item.disabled = false;
						}
						break;
					}
				}
			}
		}
	}

	Array keys = added.keys();
	keys.sort();

	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys.get(i);

		const Dictionary &value = (Dictionary)added[key];

		add_custom_type(value["name"], value["type"], value["script"], value["description"], value["return_icon_type"], value["category"], value["highend"]);
	}

	_update_options_menu();
}

String VisualShaderEditor::_get_description(int p_idx) {
	return add_options[p_idx].description;
}

void VisualShaderEditor::_update_options_menu() {
	node_desc->set_text("");
	members_dialog->get_ok_button()->set_disabled(true);

	members->clear();
	TreeItem *root = members->create_item();

	String filter = node_filter->get_text().strip_edges();
	bool use_filter = !filter.is_empty();

	bool is_first_item = true;

	Color unsupported_color = get_theme_color(SNAME("error_color"), EditorStringName(Editor));
	Color supported_color = get_theme_color(SNAME("warning_color"), EditorStringName(Editor));

	static bool low_driver = GLOBAL_GET("rendering/renderer/rendering_method") == "gl_compatibility";

	HashMap<String, TreeItem *> folders;

	int current_func = -1;

	if (!visual_shader.is_null()) {
		current_func = visual_shader->get_mode();
	}

	Vector<AddOption> custom_options;
	Vector<AddOption> embedded_options;

	static Vector<String> type_filter_exceptions;
	if (type_filter_exceptions.is_empty()) {
		type_filter_exceptions.append("VisualShaderNodeExpression");
	}

	for (int i = 0; i < add_options.size(); i++) {
		if (!use_filter || add_options[i].name.findn(filter) != -1) {
			// port type filtering
			if (members_output_port_type != VisualShaderNode::PORT_TYPE_MAX || members_input_port_type != VisualShaderNode::PORT_TYPE_MAX) {
				Ref<VisualShaderNode> vsn;
				int check_result = 0;

				if (!add_options[i].is_custom) {
					vsn = Ref<VisualShaderNode>(Object::cast_to<VisualShaderNode>(ClassDB::instantiate(add_options[i].type)));
					if (!vsn.is_valid()) {
						continue;
					}

					if (type_filter_exceptions.has(add_options[i].type)) {
						check_result = 1;
					}

					Ref<VisualShaderNodeInput> input = Object::cast_to<VisualShaderNodeInput>(vsn.ptr());
					if (input.is_valid()) {
						input->set_shader_mode(visual_shader->get_mode());
						input->set_shader_type(visual_shader->get_shader_type());
						if (!add_options[i].ops.is_empty() && add_options[i].ops[0].get_type() == Variant::STRING) {
							input->set_input_name((String)add_options[i].ops[0]);
						}
					}

					Ref<VisualShaderNodeExpression> expression = Object::cast_to<VisualShaderNodeExpression>(vsn.ptr());
					if (expression.is_valid()) {
						if (members_input_port_type == VisualShaderNode::PORT_TYPE_SAMPLER) {
							check_result = -1; // expressions creates a port with required type automatically (except for sampler output)
						}
					}

					Ref<VisualShaderNodeParameterRef> parameter_ref = Object::cast_to<VisualShaderNodeParameterRef>(vsn.ptr());
					if (parameter_ref.is_valid()) {
						check_result = -1;

						if (members_input_port_type != VisualShaderNode::PORT_TYPE_MAX) {
							for (int j = 0; j < parameter_ref->get_parameters_count(); j++) {
								if (visual_shader->is_port_types_compatible(parameter_ref->get_port_type_by_index(j), members_input_port_type)) {
									check_result = 1;
									break;
								}
							}
						}
					}
				} else {
					check_result = 1;
				}

				if (members_output_port_type != VisualShaderNode::PORT_TYPE_MAX) {
					if (check_result == 0) {
						for (int j = 0; j < vsn->get_input_port_count(); j++) {
							if (visual_shader->is_port_types_compatible(vsn->get_input_port_type(j), members_output_port_type)) {
								check_result = 1;
								break;
							}
						}
					}

					if (check_result != 1) {
						continue;
					}
				}
				if (members_input_port_type != VisualShaderNode::PORT_TYPE_MAX) {
					if (check_result == 0) {
						for (int j = 0; j < vsn->get_output_port_count(); j++) {
							if (visual_shader->is_port_types_compatible(vsn->get_output_port_type(j), members_input_port_type)) {
								check_result = 1;
								break;
							}
						}
					}

					if (check_result != 1) {
						continue;
					}
				}
			}
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
				item->set_icon(0, get_editor_theme_icon(SNAME("float")));
				break;
			case VisualShaderNode::PORT_TYPE_SCALAR_INT:
				item->set_icon(0, get_editor_theme_icon(SNAME("int")));
				break;
			case VisualShaderNode::PORT_TYPE_SCALAR_UINT:
				item->set_icon(0, get_editor_theme_icon(SNAME("uint")));
				break;
			case VisualShaderNode::PORT_TYPE_VECTOR_2D:
				item->set_icon(0, get_editor_theme_icon(SNAME("Vector2")));
				break;
			case VisualShaderNode::PORT_TYPE_VECTOR_3D:
				item->set_icon(0, get_editor_theme_icon(SNAME("Vector3")));
				break;
			case VisualShaderNode::PORT_TYPE_VECTOR_4D:
				item->set_icon(0, get_editor_theme_icon(SNAME("Vector4")));
				break;
			case VisualShaderNode::PORT_TYPE_BOOLEAN:
				item->set_icon(0, get_editor_theme_icon(SNAME("bool")));
				break;
			case VisualShaderNode::PORT_TYPE_TRANSFORM:
				item->set_icon(0, get_editor_theme_icon(SNAME("Transform3D")));
				break;
			case VisualShaderNode::PORT_TYPE_SAMPLER:
				item->set_icon(0, get_editor_theme_icon(SNAME("ImageTexture")));
				break;
			default:
				break;
		}
		item->set_meta("id", options[i].temp_idx);
	}
}

void VisualShaderEditor::_set_mode(int p_which) {
	if (p_which == VisualShader::MODE_SKY) {
		edit_type_standard->set_visible(false);
		edit_type_particles->set_visible(false);
		edit_type_sky->set_visible(true);
		edit_type_fog->set_visible(false);
		edit_type = edit_type_sky;
		custom_mode_box->set_visible(false);
		varying_button->hide();
		mode = MODE_FLAGS_SKY;
	} else if (p_which == VisualShader::MODE_FOG) {
		edit_type_standard->set_visible(false);
		edit_type_particles->set_visible(false);
		edit_type_sky->set_visible(false);
		edit_type_fog->set_visible(true);
		edit_type = edit_type_fog;
		custom_mode_box->set_visible(false);
		varying_button->hide();
		mode = MODE_FLAGS_FOG;
	} else if (p_which == VisualShader::MODE_PARTICLES) {
		edit_type_standard->set_visible(false);
		edit_type_particles->set_visible(true);
		edit_type_sky->set_visible(false);
		edit_type_fog->set_visible(false);
		edit_type = edit_type_particles;
		if ((edit_type->get_selected() + 3) > VisualShader::TYPE_PROCESS) {
			custom_mode_box->set_visible(false);
		} else {
			custom_mode_box->set_visible(true);
		}
		varying_button->hide();
		mode = MODE_FLAGS_PARTICLES;
	} else {
		edit_type_particles->set_visible(false);
		edit_type_standard->set_visible(true);
		edit_type_sky->set_visible(false);
		edit_type_fog->set_visible(false);
		edit_type = edit_type_standard;
		custom_mode_box->set_visible(false);
		varying_button->show();
		mode = MODE_FLAGS_SPATIAL_CANVASITEM;
	}
	visual_shader->set_shader_type(get_current_shader_type());
}

Size2 VisualShaderEditor::get_minimum_size() const {
	return Size2(10, 200);
}

void VisualShaderEditor::_draw_color_over_button(Object *p_obj, Color p_color) {
	Button *button = Object::cast_to<Button>(p_obj);
	if (!button) {
		return;
	}

	Ref<StyleBox> normal = get_theme_stylebox(SNAME("normal"), SNAME("Button"));
	button->draw_rect(Rect2(normal->get_offset(), button->get_size() - normal->get_minimum_size()), p_color);
}

void VisualShaderEditor::_update_parameters(bool p_update_refs) {
	VisualShaderNodeParameterRef::clear_parameters(visual_shader->get_rid());

	for (int t = 0; t < VisualShader::TYPE_MAX; t++) {
		Vector<int> tnodes = visual_shader->get_node_list((VisualShader::Type)t);
		for (int i = 0; i < tnodes.size(); i++) {
			Ref<VisualShaderNode> vsnode = visual_shader->get_node((VisualShader::Type)t, tnodes[i]);
			Ref<VisualShaderNodeParameter> parameter = vsnode;

			if (parameter.is_valid()) {
				Ref<VisualShaderNodeFloatParameter> float_parameter = vsnode;
				Ref<VisualShaderNodeIntParameter> int_parameter = vsnode;
				Ref<VisualShaderNodeUIntParameter> uint_parameter = vsnode;
				Ref<VisualShaderNodeVec2Parameter> vec2_parameter = vsnode;
				Ref<VisualShaderNodeVec3Parameter> vec3_parameter = vsnode;
				Ref<VisualShaderNodeVec4Parameter> vec4_parameter = vsnode;
				Ref<VisualShaderNodeColorParameter> color_parameter = vsnode;
				Ref<VisualShaderNodeBooleanParameter> boolean_parameter = vsnode;
				Ref<VisualShaderNodeTransformParameter> transform_parameter = vsnode;

				VisualShaderNodeParameterRef::ParameterType parameter_type;
				if (float_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_FLOAT;
				} else if (int_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_INT;
				} else if (uint_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_UINT;
				} else if (boolean_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_BOOLEAN;
				} else if (vec2_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_VECTOR2;
				} else if (vec3_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_VECTOR3;
				} else if (vec4_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_VECTOR4;
				} else if (transform_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_TRANSFORM;
				} else if (color_parameter.is_valid()) {
					parameter_type = VisualShaderNodeParameterRef::PARAMETER_TYPE_COLOR;
				} else {
					parameter_type = VisualShaderNodeParameterRef::UNIFORM_TYPE_SAMPLER;
				}
				VisualShaderNodeParameterRef::add_parameter(visual_shader->get_rid(), parameter->get_parameter_name(), parameter_type);
			}
		}
	}
	if (p_update_refs) {
		graph_plugin->update_parameter_refs();
	}
}

void VisualShaderEditor::_update_parameter_refs(HashSet<String> &p_deleted_names) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	for (int i = 0; i < VisualShader::TYPE_MAX; i++) {
		VisualShader::Type type = VisualShader::Type(i);

		Vector<int> nodes = visual_shader->get_node_list(type);
		for (int j = 0; j < nodes.size(); j++) {
			if (j > 0) {
				Ref<VisualShaderNodeParameterRef> ref = visual_shader->get_node(type, nodes[j]);
				if (ref.is_valid()) {
					if (p_deleted_names.has(ref->get_parameter_name())) {
						undo_redo->add_do_method(ref.ptr(), "set_parameter_name", "[None]");
						undo_redo->add_undo_method(ref.ptr(), "set_parameter_name", ref->get_parameter_name());
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

	graph->set_scroll_offset(visual_shader->get_graph_offset() * EDSCALE);

	VisualShader::Type type = get_current_shader_type();

	graph->clear_connections();
	// Remove all nodes.
	for (int i = 0; i < graph->get_child_count(); i++) {
		if (Object::cast_to<GraphElement>(graph->get_child(i))) {
			Node *node = graph->get_child(i);
			graph->remove_child(node);
			memdelete(node);
			i--;
		}
	}

	List<VisualShader::Connection> node_connections;
	visual_shader->get_node_connections(type, &node_connections);
	graph_plugin->set_connections(node_connections);

	Vector<int> nodes = visual_shader->get_node_list(type);

	_update_parameters(false);
	_update_varyings();

	graph_plugin->clear_links();
	graph_plugin->update_theme();

	for (int n_i = 0; n_i < nodes.size(); n_i++) {
		graph_plugin->add_node(type, nodes[n_i], false);
	}

	for (const VisualShader::Connection &E : node_connections) {
		int from = E.from_node;
		int from_idx = E.from_port;
		int to = E.to_node;
		int to_idx = E.to_port;

		graph->connect_node(itos(from), from_idx, itos(to), to_idx);
	}

	float graph_minimap_opacity = EDITOR_GET("editors/visual_editors/minimap_opacity");
	graph->set_minimap_opacity(graph_minimap_opacity);
	float graph_lines_curvature = EDITOR_GET("editors/visual_editors/lines_curvature");
	graph->set_connection_lines_curvature(graph_lines_curvature);
}

VisualShader::Type VisualShaderEditor::get_current_shader_type() const {
	VisualShader::Type type;
	if (mode & MODE_FLAGS_PARTICLES) {
		type = VisualShader::Type(edit_type->get_selected() + 3 + (custom_mode_enabled ? 3 : 0));
	} else if (mode & MODE_FLAGS_SKY) {
		type = VisualShader::Type(edit_type->get_selected() + 8);
	} else if (mode & MODE_FLAGS_FOG) {
		type = VisualShader::Type(edit_type->get_selected() + 9);
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

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
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

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
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

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
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

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Change Output Port Type"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_type", p_port, p_type);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_type", p_port, node->get_output_port_type(p_port));
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_input_port_name(const String &p_text, Object *p_line_edit, int p_node_id, int p_port_id) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	String prev_name = node->get_input_port_name(p_port_id);
	if (prev_name == p_text) {
		return;
	}

	LineEdit *line_edit = Object::cast_to<LineEdit>(p_line_edit);
	ERR_FAIL_NULL(line_edit);

	String validated_name = visual_shader->validate_port_name(p_text, node.ptr(), p_port_id, false);
	if (validated_name.is_empty() || prev_name == validated_name) {
		line_edit->set_text(node->get_input_port_name(p_port_id));
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Change Input Port Name"));
	undo_redo->add_do_method(node.ptr(), "set_input_port_name", p_port_id, validated_name);
	undo_redo->add_undo_method(node.ptr(), "set_input_port_name", p_port_id, node->get_input_port_name(p_port_id));
	undo_redo->commit_action();
}

void VisualShaderEditor::_change_output_port_name(const String &p_text, Object *p_line_edit, int p_node_id, int p_port_id) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	String prev_name = node->get_output_port_name(p_port_id);
	if (prev_name == p_text) {
		return;
	}

	LineEdit *line_edit = Object::cast_to<LineEdit>(p_line_edit);
	ERR_FAIL_NULL(line_edit);

	String validated_name = visual_shader->validate_port_name(p_text, node.ptr(), p_port_id, true);
	if (validated_name.is_empty() || prev_name == validated_name) {
		line_edit->set_text(node->get_output_port_name(p_port_id));
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Change Output Port Name"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_name", p_port_id, validated_name);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_name", p_port_id, prev_name);
	undo_redo->commit_action();
}

void VisualShaderEditor::_expand_output_port(int p_node, int p_port, bool p_expand) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNode> node = visual_shader->get_node(type, p_node);
	ERR_FAIL_COND(!node.is_valid());

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	if (p_expand) {
		undo_redo->create_action(TTR("Expand Output Port"));
	} else {
		undo_redo->create_action(TTR("Shrink Output Port"));
	}

	undo_redo->add_do_method(node.ptr(), "_set_output_port_expanded", p_port, p_expand);
	undo_redo->add_undo_method(node.ptr(), "_set_output_port_expanded", p_port, !p_expand);

	int type_size = 0;
	switch (node->get_output_port_type(p_port)) {
		case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
			type_size = 2;
		} break;
		case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
			type_size = 3;
		} break;
		case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
			type_size = 4;
		} break;
		default:
			break;
	}

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	for (const VisualShader::Connection &E : conns) {
		int cn_from_node = E.from_node;
		int cn_from_port = E.from_port;
		int cn_to_node = E.to_node;
		int cn_to_port = E.to_port;

		if (cn_from_node == p_node) {
			if (p_expand) {
				if (cn_from_port > p_port) { // reconnect ports after expanded ports
					undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

					undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

					undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port + type_size, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port + type_size, cn_to_node, cn_to_port);

					undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port + type_size, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port + type_size, cn_to_node, cn_to_port);
				}
			} else {
				if (cn_from_port > p_port + type_size) { // reconnect ports after expanded ports
					undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

					undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

					undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, cn_from_node, cn_from_port - type_size, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port - type_size, cn_to_node, cn_to_port);

					undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port - type_size, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port - type_size, cn_to_node, cn_to_port);
				} else if (cn_from_port > p_port) { // disconnect component ports
					undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

					undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
					undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				}
			}
		}
	}

	int preview_port = node->get_output_port_for_preview();
	if (p_expand) {
		if (preview_port > p_port) {
			undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", preview_port + type_size);
			undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", preview_port);
		}
	} else {
		if (preview_port > p_port + type_size) {
			undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", preview_port - type_size);
			undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", preview_port);
		}
	}

	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_remove_input_port(int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeGroupBase> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove Input Port"));

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);
	for (const VisualShader::Connection &E : conns) {
		int cn_from_node = E.from_node;
		int cn_from_port = E.from_port;
		int cn_to_node = E.to_node;
		int cn_to_port = E.to_port;

		if (cn_to_node == p_node) {
			if (cn_to_port == p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
			} else if (cn_to_port > p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port - 1);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port - 1);

				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port - 1);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port - 1);
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

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove Output Port"));

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);
	for (const VisualShader::Connection &E : conns) {
		int cn_from_node = E.from_node;
		int cn_from_port = E.from_port;
		int cn_to_node = E.to_node;
		int cn_to_port = E.to_port;

		if (cn_from_node == p_node) {
			if (cn_from_port == p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
			} else if (cn_from_port > p_port) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);

				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes_forced", type, cn_from_node, cn_from_port - 1, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port - 1, cn_to_node, cn_to_port);

				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port - 1, cn_to_node, cn_to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port - 1, cn_to_node, cn_to_port);
			}
		}
	}

	int preview_port = node->get_output_port_for_preview();
	if (preview_port != -1) {
		if (preview_port == p_port) {
			undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", -1);
			undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", preview_port);
		} else if (preview_port > p_port) {
			undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", preview_port - 1);
			undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", preview_port);
		}
	}

	undo_redo->add_do_method(node.ptr(), "remove_output_port", p_port);
	undo_redo->add_undo_method(node.ptr(), "add_output_port", p_port, (int)node->get_output_port_type(p_port), node->get_output_port_name(p_port));

	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);

	undo_redo->commit_action();
}

void VisualShaderEditor::_expression_focus_out(Object *p_code_edit, int p_node) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeExpression> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	CodeEdit *expression_box = Object::cast_to<CodeEdit>(p_code_edit);

	if (node->get_expression() == expression_box->get_text()) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
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
			text_box = expression_node->is_ctrl_pressed(0);
			if (text_box) {
				text_box->set_custom_minimum_size(Size2(0, 0));
			}
		}

		GraphElement *graph_element = nullptr;
		Node *node2 = graph->get_node(itos(p_node));
		graph_element = Object::cast_to<GraphElement>(node2);
		if (!graph_element) {
			return;
		}

		graph_element->set_custom_minimum_size(size);
		graph_element->reset_size();

		if (!expression_node.is_null() && text_box) {
			Size2 box_size = size;
			if (box_size.x < 150 * EDSCALE || box_size.y < 0) {
				box_size.x = graph_element->get_size().x;
			}
			box_size.x -= text_box->get_offset(SIDE_LEFT);
			box_size.x -= 28 * EDSCALE;
			box_size.y -= text_box->get_offset(SIDE_TOP);
			box_size.y -= 28 * EDSCALE;
			text_box->set_custom_minimum_size(box_size);
			text_box->reset_size();
		}
	}
}

void VisualShaderEditor::_node_resized(const Vector2 &p_new_size, int p_type, int p_node) {
	Ref<VisualShaderNodeResizableBase> node = visual_shader->get_node(VisualShader::Type(p_type), p_node);
	if (node.is_null()) {
		return;
	}

	Vector2 new_size = p_new_size;
	if (graph->is_snapping_enabled() ^ Input::get_singleton()->is_key_pressed(Key::CTRL)) {
		new_size = new_size.snapped(Vector2(graph->get_snapping_distance(), graph->get_snapping_distance()));
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Resize VisualShader Node"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(this, "_set_node_size", p_type, p_node, new_size);
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
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(p_port == -1 ? TTR("Hide Port Preview") : TTR("Show Port Preview"));
	undo_redo->add_do_method(node.ptr(), "set_output_port_for_preview", p_port);
	undo_redo->add_undo_method(node.ptr(), "set_output_port_for_preview", prev_port);
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", (int)type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", (int)type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_comment_title_popup_show(const Point2 &p_position, int p_node_id) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeComment> node = visual_shader->get_node(type, p_node_id);
	if (node.is_null()) {
		return;
	}
	comment_title_change_edit->set_text(node->get_title());
	comment_title_change_popup->set_meta("id", p_node_id);
	comment_title_change_popup->popup();
	comment_title_change_popup->set_position(p_position);
}

void VisualShaderEditor::_comment_title_text_changed(const String &p_new_text) {
	comment_title_change_edit->reset_size();
	comment_title_change_popup->reset_size();
}

void VisualShaderEditor::_comment_title_text_submitted(const String &p_new_text) {
	comment_title_change_popup->hide();
}

void VisualShaderEditor::_comment_title_popup_focus_out() {
	comment_title_change_popup->hide();
}

void VisualShaderEditor::_comment_title_popup_hide() {
	ERR_FAIL_COND(!comment_title_change_popup->has_meta("id"));
	int node_id = (int)comment_title_change_popup->get_meta("id");

	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeComment> node = visual_shader->get_node(type, node_id);

	ERR_FAIL_COND(node.is_null());

	if (node->get_title() == comment_title_change_edit->get_text()) {
		return; // nothing changed - ignored
	}
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Comment Title"));
	undo_redo->add_do_method(node.ptr(), "set_title", comment_title_change_edit->get_text());
	undo_redo->add_undo_method(node.ptr(), "set_title", node->get_title());
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", (int)type, node_id);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", (int)type, node_id);
	undo_redo->commit_action();
}

void VisualShaderEditor::_comment_desc_popup_show(const Point2 &p_position, int p_node_id) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeComment> node = visual_shader->get_node(type, p_node_id);
	if (node.is_null()) {
		return;
	}
	comment_desc_change_edit->set_text(node->get_description());
	comment_desc_change_popup->set_meta("id", p_node_id);
	comment_desc_change_popup->reset_size();
	comment_desc_change_popup->popup();
	comment_desc_change_popup->set_position(p_position);
}

void VisualShaderEditor::_comment_desc_text_changed() {
	comment_desc_change_edit->reset_size();
	comment_desc_change_popup->reset_size();
}

void VisualShaderEditor::_comment_desc_confirm() {
	comment_desc_change_popup->hide();
}

void VisualShaderEditor::_comment_desc_popup_hide() {
	ERR_FAIL_COND(!comment_desc_change_popup->has_meta("id"));
	int node_id = (int)comment_desc_change_popup->get_meta("id");

	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeComment> node = visual_shader->get_node(type, node_id);

	ERR_FAIL_COND(node.is_null());

	if (node->get_description() == comment_desc_change_edit->get_text()) {
		return; // nothing changed - ignored
	}
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Comment Description"));
	undo_redo->add_do_method(node.ptr(), "set_description", comment_desc_change_edit->get_text());
	undo_redo->add_undo_method(node.ptr(), "set_description", node->get_title());
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", (int)type, node_id);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", (int)type, node_id);
	undo_redo->commit_action();
}

void VisualShaderEditor::_parameter_line_edit_changed(const String &p_text, int p_node_id) {
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNodeParameter> node = visual_shader->get_node(type, p_node_id);
	ERR_FAIL_COND(!node.is_valid());

	String validated_name = visual_shader->validate_parameter_name(p_text, node);

	if (validated_name == node->get_parameter_name()) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Parameter Name"));
	undo_redo->add_do_method(node.ptr(), "set_parameter_name", validated_name);
	undo_redo->add_undo_method(node.ptr(), "set_parameter_name", node->get_parameter_name());
	undo_redo->add_do_method(graph_plugin.ptr(), "set_parameter_name", type, p_node_id, validated_name);
	undo_redo->add_undo_method(graph_plugin.ptr(), "set_parameter_name", type, p_node_id, node->get_parameter_name());
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node_deferred", type, p_node_id);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node_deferred", type, p_node_id);

	undo_redo->add_do_method(this, "_update_parameters", true);
	undo_redo->add_undo_method(this, "_update_parameters", true);

	HashSet<String> changed_names;
	changed_names.insert(node->get_parameter_name());
	_update_parameter_refs(changed_names);

	undo_redo->commit_action();
}

void VisualShaderEditor::_parameter_line_edit_focus_out(Object *line_edit, int p_node_id) {
	_parameter_line_edit_changed(Object::cast_to<LineEdit>(line_edit)->get_text(), p_node_id);
}

void VisualShaderEditor::_port_name_focus_out(Object *line_edit, int p_node_id, int p_port_id, bool p_output) {
	if (!p_output) {
		_change_input_port_name(Object::cast_to<LineEdit>(line_edit)->get_text(), line_edit, p_node_id, p_port_id);
	} else {
		_change_output_port_name(Object::cast_to<LineEdit>(line_edit)->get_text(), line_edit, p_node_id, p_port_id);
	}
}

void VisualShaderEditor::_port_edited(const StringName &p_property, const Variant &p_value, const String &p_field, bool p_changing) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNode> vsn = visual_shader->get_node(type, editing_node);
	ERR_FAIL_COND(!vsn.is_valid());

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Input Default Port"));

	Ref<VisualShaderNodeCustom> custom = Object::cast_to<VisualShaderNodeCustom>(vsn.ptr());
	if (custom.is_valid()) {
		undo_redo->add_do_method(custom.ptr(), "_set_input_port_default_value", editing_port, p_value);
		undo_redo->add_undo_method(custom.ptr(), "_set_input_port_default_value", editing_port, vsn->get_input_port_default_value(editing_port));
	} else {
		undo_redo->add_do_method(vsn.ptr(), "set_input_port_default_value", editing_port, p_value);
		undo_redo->add_undo_method(vsn.ptr(), "set_input_port_default_value", editing_port, vsn->get_input_port_default_value(editing_port));
	}
	undo_redo->add_do_method(graph_plugin.ptr(), "set_input_port_default_value", type, editing_node, editing_port, p_value);
	undo_redo->add_undo_method(graph_plugin.ptr(), "set_input_port_default_value", type, editing_node, editing_port, vsn->get_input_port_default_value(editing_port));
	undo_redo->commit_action();
}

void VisualShaderEditor::_edit_port_default_input(Object *p_button, int p_node, int p_port) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNode> vs_node = visual_shader->get_node(type, p_node);
	Variant value = vs_node->get_input_port_default_value(p_port);

	edited_property_holder->set_edited_property(value);

	if (property_editor) {
		property_editor->disconnect("property_changed", callable_mp(this, &VisualShaderEditor::_port_edited));
		property_editor_popup->remove_child(property_editor);
	}

	// TODO: Define these properties with actual PropertyInfo and feed it to the property editor widget.
	property_editor = EditorInspector::instantiate_property_editor(edited_property_holder.ptr(), value.get_type(), "edited_property", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE);
	if (property_editor) {
		property_editor->set_object_and_property(edited_property_holder.ptr(), "edited_property");
		property_editor->update_property();
		property_editor->set_name_split_ratio(0);
		property_editor_popup->add_child(property_editor);

		property_editor->connect("property_changed", callable_mp(this, &VisualShaderEditor::_port_edited));

		Button *button = Object::cast_to<Button>(p_button);
		if (button) {
			property_editor_popup->set_position(button->get_screen_position() + Vector2(0, button->get_size().height) * graph->get_zoom());
		}
		property_editor_popup->reset_size();
		if (button) {
			property_editor_popup->popup();
		} else {
			property_editor_popup->popup_centered_ratio();
		}
	}

	editing_node = p_node;
	editing_port = p_port;
}

void VisualShaderEditor::_set_custom_node_option(int p_index, int p_node, int p_op) {
	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeCustom> node = visual_shader->get_node(type, p_node);
	if (node.is_null()) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Custom Node Option"));
	undo_redo->add_do_method(node.ptr(), "_set_option_index", p_op, p_index);
	undo_redo->add_undo_method(node.ptr(), "_set_option_index", p_op, node->get_option_index(p_op));
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, p_node);
	undo_redo->commit_action();
}

void VisualShaderEditor::_setup_node(VisualShaderNode *p_node, const Vector<Variant> &p_ops) {
	// INPUT
	{
		VisualShaderNodeInput *input = Object::cast_to<VisualShaderNodeInput>(p_node);

		if (input) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::STRING);
			input->set_input_name((String)p_ops[0]);
			return;
		}
	}

	// FLOAT_CONST
	{
		VisualShaderNodeFloatConstant *float_const = Object::cast_to<VisualShaderNodeFloatConstant>(p_node);

		if (float_const) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::FLOAT);
			float_const->set_constant((float)p_ops[0]);
			return;
		}
	}

	// FLOAT_OP
	{
		VisualShaderNodeFloatOp *floatOp = Object::cast_to<VisualShaderNodeFloatOp>(p_node);

		if (floatOp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			floatOp->set_operator((VisualShaderNodeFloatOp::Operator)(int)p_ops[0]);
			return;
		}
	}

	// FLOAT_FUNC
	{
		VisualShaderNodeFloatFunc *floatFunc = Object::cast_to<VisualShaderNodeFloatFunc>(p_node);

		if (floatFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			floatFunc->set_function((VisualShaderNodeFloatFunc::Function)(int)p_ops[0]);
			return;
		}
	}

	// VECTOR_OP
	{
		VisualShaderNodeVectorOp *vecOp = Object::cast_to<VisualShaderNodeVectorOp>(p_node);

		if (vecOp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			ERR_FAIL_COND(p_ops[1].get_type() != Variant::INT);
			vecOp->set_operator((VisualShaderNodeVectorOp::Operator)(int)p_ops[0]);
			vecOp->set_op_type((VisualShaderNodeVectorOp::OpType)(int)p_ops[1]);
			return;
		}
	}

	// VECTOR_FUNC
	{
		VisualShaderNodeVectorFunc *vecFunc = Object::cast_to<VisualShaderNodeVectorFunc>(p_node);

		if (vecFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			ERR_FAIL_COND(p_ops[1].get_type() != Variant::INT);
			vecFunc->set_function((VisualShaderNodeVectorFunc::Function)(int)p_ops[0]);
			vecFunc->set_op_type((VisualShaderNodeVectorFunc::OpType)(int)p_ops[1]);
			return;
		}
	}

	// COLOR_OP
	{
		VisualShaderNodeColorOp *colorOp = Object::cast_to<VisualShaderNodeColorOp>(p_node);

		if (colorOp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			colorOp->set_operator((VisualShaderNodeColorOp::Operator)(int)p_ops[0]);
			return;
		}
	}

	// COLOR_FUNC
	{
		VisualShaderNodeColorFunc *colorFunc = Object::cast_to<VisualShaderNodeColorFunc>(p_node);

		if (colorFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			colorFunc->set_function((VisualShaderNodeColorFunc::Function)(int)p_ops[0]);
			return;
		}
	}

	// INT_OP
	{
		VisualShaderNodeIntOp *intOp = Object::cast_to<VisualShaderNodeIntOp>(p_node);

		if (intOp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			intOp->set_operator((VisualShaderNodeIntOp::Operator)(int)p_ops[0]);
			return;
		}
	}

	// INT_FUNC
	{
		VisualShaderNodeIntFunc *intFunc = Object::cast_to<VisualShaderNodeIntFunc>(p_node);

		if (intFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			intFunc->set_function((VisualShaderNodeIntFunc::Function)(int)p_ops[0]);
			return;
		}
	}

	// UINT_OP
	{
		VisualShaderNodeUIntOp *uintOp = Object::cast_to<VisualShaderNodeUIntOp>(p_node);

		if (uintOp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			uintOp->set_operator((VisualShaderNodeUIntOp::Operator)(int)p_ops[0]);
			return;
		}
	}

	// UINT_FUNC
	{
		VisualShaderNodeUIntFunc *uintFunc = Object::cast_to<VisualShaderNodeUIntFunc>(p_node);

		if (uintFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			uintFunc->set_function((VisualShaderNodeUIntFunc::Function)(int)p_ops[0]);
			return;
		}
	}

	// TRANSFORM_OP
	{
		VisualShaderNodeTransformOp *matOp = Object::cast_to<VisualShaderNodeTransformOp>(p_node);

		if (matOp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			matOp->set_operator((VisualShaderNodeTransformOp::Operator)(int)p_ops[0]);
			return;
		}
	}

	// TRANSFORM_FUNC
	{
		VisualShaderNodeTransformFunc *matFunc = Object::cast_to<VisualShaderNodeTransformFunc>(p_node);

		if (matFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			matFunc->set_function((VisualShaderNodeTransformFunc::Function)(int)p_ops[0]);
			return;
		}
	}

	// VECTOR_COMPOSE
	{
		VisualShaderNodeVectorCompose *vecCompose = Object::cast_to<VisualShaderNodeVectorCompose>(p_node);

		if (vecCompose) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			vecCompose->set_op_type((VisualShaderNodeVectorCompose::OpType)(int)p_ops[0]);
			return;
		}
	}

	// VECTOR_DECOMPOSE
	{
		VisualShaderNodeVectorDecompose *vecDecompose = Object::cast_to<VisualShaderNodeVectorDecompose>(p_node);

		if (vecDecompose) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			vecDecompose->set_op_type((VisualShaderNodeVectorDecompose::OpType)(int)p_ops[0]);
			return;
		}
	}

	// UV_FUNC
	{
		VisualShaderNodeUVFunc *uvFunc = Object::cast_to<VisualShaderNodeUVFunc>(p_node);

		if (uvFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			uvFunc->set_function((VisualShaderNodeUVFunc::Function)(int)p_ops[0]);
			return;
		}
	}

	// IS
	{
		VisualShaderNodeIs *is = Object::cast_to<VisualShaderNodeIs>(p_node);

		if (is) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			is->set_function((VisualShaderNodeIs::Function)(int)p_ops[0]);
			return;
		}
	}

	// COMPARE
	{
		VisualShaderNodeCompare *cmp = Object::cast_to<VisualShaderNodeCompare>(p_node);

		if (cmp) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			cmp->set_function((VisualShaderNodeCompare::Function)(int)p_ops[0]);
			return;
		}
	}

	// DISTANCE
	{
		VisualShaderNodeVectorDistance *dist = Object::cast_to<VisualShaderNodeVectorDistance>(p_node);

		if (dist) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			dist->set_op_type((VisualShaderNodeVectorDistance::OpType)(int)p_ops[0]);
			return;
		}
	}

	// DERIVATIVE
	{
		VisualShaderNodeDerivativeFunc *derFunc = Object::cast_to<VisualShaderNodeDerivativeFunc>(p_node);

		if (derFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			ERR_FAIL_COND(p_ops[1].get_type() != Variant::INT);
			derFunc->set_function((VisualShaderNodeDerivativeFunc::Function)(int)p_ops[0]);
			derFunc->set_op_type((VisualShaderNodeDerivativeFunc::OpType)(int)p_ops[1]);
			return;
		}
	}

	// MIX
	{
		VisualShaderNodeMix *mix = Object::cast_to<VisualShaderNodeMix>(p_node);

		if (mix) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			mix->set_op_type((VisualShaderNodeMix::OpType)(int)p_ops[0]);
			return;
		}
	}

	// CLAMP
	{
		VisualShaderNodeClamp *clampFunc = Object::cast_to<VisualShaderNodeClamp>(p_node);

		if (clampFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			clampFunc->set_op_type((VisualShaderNodeClamp::OpType)(int)p_ops[0]);
			return;
		}
	}

	// SWITCH
	{
		VisualShaderNodeSwitch *switchFunc = Object::cast_to<VisualShaderNodeSwitch>(p_node);

		if (switchFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			switchFunc->set_op_type((VisualShaderNodeSwitch::OpType)(int)p_ops[0]);
			return;
		}
	}

	// FACEFORWARD
	{
		VisualShaderNodeFaceForward *faceForward = Object::cast_to<VisualShaderNodeFaceForward>(p_node);
		if (faceForward) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			faceForward->set_op_type((VisualShaderNodeFaceForward::OpType)(int)p_ops[0]);
			return;
		}
	}

	// LENGTH
	{
		VisualShaderNodeVectorLen *length = Object::cast_to<VisualShaderNodeVectorLen>(p_node);
		if (length) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			length->set_op_type((VisualShaderNodeVectorLen::OpType)(int)p_ops[0]);
			return;
		}
	}

	// SMOOTHSTEP
	{
		VisualShaderNodeSmoothStep *smoothStepFunc = Object::cast_to<VisualShaderNodeSmoothStep>(p_node);

		if (smoothStepFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			smoothStepFunc->set_op_type((VisualShaderNodeSmoothStep::OpType)(int)p_ops[0]);
			return;
		}
	}

	// STEP
	{
		VisualShaderNodeStep *stepFunc = Object::cast_to<VisualShaderNodeStep>(p_node);

		if (stepFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			stepFunc->set_op_type((VisualShaderNodeStep::OpType)(int)p_ops[0]);
			return;
		}
	}

	// MULTIPLY_ADD
	{
		VisualShaderNodeMultiplyAdd *fmaFunc = Object::cast_to<VisualShaderNodeMultiplyAdd>(p_node);

		if (fmaFunc) {
			ERR_FAIL_COND(p_ops[0].get_type() != Variant::INT);
			fmaFunc->set_op_type((VisualShaderNodeMultiplyAdd::OpType)(int)p_ops[0]);
		}
	}
}

void VisualShaderEditor::_add_node(int p_idx, const Vector<Variant> &p_ops, String p_resource_path, int p_node_idx) {
	ERR_FAIL_INDEX(p_idx, add_options.size());

	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNode> vsnode;

	bool is_custom = add_options[p_idx].is_custom;

	if (!is_custom && !add_options[p_idx].type.is_empty()) {
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instantiate(add_options[p_idx].type));
		ERR_FAIL_NULL(vsn);
		if (!p_ops.is_empty()) {
			_setup_node(vsn, p_ops);
		}
		VisualShaderNodeParameterRef *parameter_ref = Object::cast_to<VisualShaderNodeParameterRef>(vsn);
		if (parameter_ref && to_node != -1 && to_slot != -1) {
			VisualShaderNode::PortType input_port_type = visual_shader->get_node(type, to_node)->get_input_port_type(to_slot);
			bool success = false;

			for (int i = 0; i < parameter_ref->get_parameters_count(); i++) {
				if (parameter_ref->get_port_type_by_index(i) == input_port_type) {
					parameter_ref->set_parameter_name(parameter_ref->get_parameter_name_by_index(i));
					success = true;
					break;
				}
			}
			if (!success) {
				for (int i = 0; i < parameter_ref->get_parameters_count(); i++) {
					if (visual_shader->is_port_types_compatible(parameter_ref->get_port_type_by_index(i), input_port_type)) {
						parameter_ref->set_parameter_name(parameter_ref->get_parameter_name_by_index(i));
						break;
					}
				}
			}
		}

		vsnode = Ref<VisualShaderNode>(vsn);
	} else {
		StringName base_type;
		bool is_native = add_options[p_idx].is_native;

		if (is_native) {
			base_type = add_options[p_idx].type;
		} else {
			ERR_FAIL_COND(add_options[p_idx].script.is_null());
			base_type = add_options[p_idx].script->get_instance_base_type();
		}
		VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instantiate(base_type));
		ERR_FAIL_NULL(vsn);
		vsnode = Ref<VisualShaderNode>(vsn);
		if (!is_native) {
			vsnode->set_script(add_options[p_idx].script);
		}
		VisualShaderNodeCustom *custom_node = Object::cast_to<VisualShaderNodeCustom>(vsn);
		ERR_FAIL_NULL(custom_node);
		custom_node->update_property_default_values();
		custom_node->update_input_port_default_values();
		custom_node->update_properties();
	}

	bool is_texture2d = (Object::cast_to<VisualShaderNodeTexture>(vsnode.ptr()) != nullptr);
	bool is_texture3d = (Object::cast_to<VisualShaderNodeTexture3D>(vsnode.ptr()) != nullptr);
	bool is_texture2d_array = (Object::cast_to<VisualShaderNodeTexture2DArray>(vsnode.ptr()) != nullptr);
	bool is_cubemap = (Object::cast_to<VisualShaderNodeCubemap>(vsnode.ptr()) != nullptr);
	bool is_curve = (Object::cast_to<VisualShaderNodeCurveTexture>(vsnode.ptr()) != nullptr);
	bool is_curve_xyz = (Object::cast_to<VisualShaderNodeCurveXYZTexture>(vsnode.ptr()) != nullptr);
	bool is_parameter = (Object::cast_to<VisualShaderNodeParameter>(vsnode.ptr()) != nullptr);

	Point2 position = graph->get_scroll_offset();

	if (saved_node_pos_dirty) {
		position += saved_node_pos;
	} else {
		position += graph->get_size() * 0.5;
		position /= EDSCALE;
	}
	position /= graph->get_zoom();
	saved_node_pos_dirty = false;

	int id_to_use = visual_shader->get_valid_node_id(type);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	if (p_resource_path.is_empty()) {
		undo_redo->create_action(TTR("Add Node to Visual Shader"));
	} else {
		id_to_use += p_node_idx;
	}
	undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, vsnode, position, id_to_use);
	undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_to_use);
	undo_redo->add_do_method(graph_plugin.ptr(), "add_node", type, id_to_use, false);
	undo_redo->add_undo_method(graph_plugin.ptr(), "remove_node", type, id_to_use, false);

	VisualShaderNodeExpression *expr = Object::cast_to<VisualShaderNodeExpression>(vsnode.ptr());
	if (expr) {
		expr->set_size(Size2(250 * EDSCALE, 150 * EDSCALE));
	}

	bool created_expression_port = false;

	if (to_node != -1 && to_slot != -1) {
		VisualShaderNode::PortType input_port_type = visual_shader->get_node(type, to_node)->get_input_port_type(to_slot);

		if (expr && expr->is_editable() && input_port_type != VisualShaderNode::PORT_TYPE_SAMPLER) {
			expr->add_output_port(0, input_port_type, "output0");
			String initial_expression_code;

			switch (input_port_type) {
				case VisualShaderNode::PORT_TYPE_SCALAR:
					initial_expression_code = "output0 = 1.0;";
					break;
				case VisualShaderNode::PORT_TYPE_SCALAR_INT:
					initial_expression_code = "output0 = 1;";
					break;
				case VisualShaderNode::PORT_TYPE_SCALAR_UINT:
					initial_expression_code = "output0 = 1u;";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_2D:
					initial_expression_code = "output0 = vec2(1.0, 1.0);";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D:
					initial_expression_code = "output0 = vec3(1.0, 1.0, 1.0);";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D:
					initial_expression_code = "output0 = vec4(1.0, 1.0, 1.0, 1.0);";
					break;
				case VisualShaderNode::PORT_TYPE_BOOLEAN:
					initial_expression_code = "output0 = true;";
					break;
				case VisualShaderNode::PORT_TYPE_TRANSFORM:
					initial_expression_code = "output0 = mat4(1.0);";
					break;
				default:
					break;
			}

			expr->set_expression(initial_expression_code);
			expr->set_size(Size2(500 * EDSCALE, 200 * EDSCALE));
			created_expression_port = true;
		}
		if (vsnode->get_output_port_count() > 0 || created_expression_port) {
			int _from_node = id_to_use;

			if (created_expression_port) {
				int _from_slot = 0;
				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, _from_node, _from_slot, to_node, to_slot);
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, _from_node, _from_slot, to_node, to_slot);
				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, _from_node, _from_slot, to_node, to_slot);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, _from_node, _from_slot, to_node, to_slot);
			} else {
				// Need to setting up Input node properly before committing since `is_port_types_compatible` (calling below) is using `mode` and `shader_type`.
				VisualShaderNodeInput *input = Object::cast_to<VisualShaderNodeInput>(vsnode.ptr());
				if (input) {
					input->set_shader_mode(visual_shader->get_mode());
					input->set_shader_type(visual_shader->get_shader_type());
				}

				// Attempting to connect to the first correct port.
				for (int i = 0; i < vsnode->get_output_port_count(); i++) {
					if (visual_shader->is_port_types_compatible(vsnode->get_output_port_type(i), input_port_type)) {
						undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, _from_node, i, to_node, to_slot);
						undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, _from_node, i, to_node, to_slot);
						undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, _from_node, i, to_node, to_slot);
						undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, _from_node, i, to_node, to_slot);
						break;
					}
				}
			}
		}
	} else if (from_node != -1 && from_slot != -1) {
		VisualShaderNode::PortType output_port_type = visual_shader->get_node(type, from_node)->get_output_port_type(from_slot);

		if (expr && expr->is_editable()) {
			expr->add_input_port(0, output_port_type, "input0");
			created_expression_port = true;
		}

		if (vsnode->get_input_port_count() > 0 || created_expression_port) {
			int _to_node = id_to_use;

			if (created_expression_port) {
				int _to_slot = 0;
				undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
			} else {
				int _to_slot = -1;

				// Attempting to connect to the default input port or to the first correct port (if it's not found).
				for (int i = 0; i < vsnode->get_input_port_count(); i++) {
					if (visual_shader->is_port_types_compatible(output_port_type, vsnode->get_input_port_type(i))) {
						if (i == vsnode->get_default_input_port(output_port_type)) {
							_to_slot = i;
							break;
						} else if (_to_slot == -1) {
							_to_slot = i;
						}
					}
				}

				if (_to_slot >= 0) {
					undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
					undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
					undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from_node, from_slot, _to_node, _to_slot);
					undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from_node, from_slot, _to_node, _to_slot);
				}
			}

			if (output_port_type == VisualShaderNode::PORT_TYPE_SAMPLER) {
				if (is_texture2d) {
					undo_redo->add_do_method(vsnode.ptr(), "set_source", VisualShaderNodeTexture::SOURCE_PORT);
				}
				if (is_texture3d || is_texture2d_array) {
					undo_redo->add_do_method(vsnode.ptr(), "set_source", VisualShaderNodeSample3D::SOURCE_PORT);
				}
				if (is_cubemap) {
					undo_redo->add_do_method(vsnode.ptr(), "set_source", VisualShaderNodeCubemap::SOURCE_PORT);
				}
			}
		}
	}
	_member_cancel();

	if (is_parameter) {
		undo_redo->add_do_method(this, "_update_parameters", true);
		undo_redo->add_undo_method(this, "_update_parameters", true);
	}

	if (is_curve) {
		graph_plugin->call_deferred(SNAME("update_curve"), id_to_use);
	}

	if (is_curve_xyz) {
		graph_plugin->call_deferred(SNAME("update_curve_xyz"), id_to_use);
	}

	if (p_resource_path.is_empty()) {
		undo_redo->commit_action();
	} else {
		//post-initialization

		if (is_texture2d || is_texture3d || is_curve || is_curve_xyz) {
			undo_redo->add_do_method(vsnode.ptr(), "set_texture", ResourceLoader::load(p_resource_path));
			return;
		}

		if (is_cubemap) {
			undo_redo->add_do_method(vsnode.ptr(), "set_cube_map", ResourceLoader::load(p_resource_path));
			return;
		}

		if (is_texture2d_array) {
			undo_redo->add_do_method(vsnode.ptr(), "set_texture_array", ResourceLoader::load(p_resource_path));
		}
	}
}

void VisualShaderEditor::_add_varying(const String &p_name, VisualShader::VaryingMode p_mode, VisualShader::VaryingType p_type) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTR("Add Varying to Visual Shader: %s"), p_name));

	undo_redo->add_do_method(visual_shader.ptr(), "add_varying", p_name, p_mode, p_type);
	undo_redo->add_undo_method(visual_shader.ptr(), "remove_varying", p_name);

	undo_redo->add_do_method(this, "_update_varyings");
	undo_redo->add_undo_method(this, "_update_varyings");

	for (int i = 0; i <= VisualShader::TYPE_LIGHT; i++) {
		if (p_mode == VisualShader::VARYING_MODE_FRAG_TO_LIGHT && i == 0) {
			continue;
		}

		VisualShader::Type type = VisualShader::Type(i);
		Vector<int> nodes = visual_shader->get_node_list(type);

		for (int j = 0; j < nodes.size(); j++) {
			int node_id = nodes[j];
			Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, node_id);
			Ref<VisualShaderNodeVarying> var = vsnode;

			if (var.is_valid()) {
				undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, node_id);
				undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, node_id);
			}
		}
	}

	undo_redo->add_do_method(this, "_update_varying_tree");
	undo_redo->add_undo_method(this, "_update_varying_tree");
	undo_redo->commit_action();
}

void VisualShaderEditor::_remove_varying(const String &p_name) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTR("Remove Varying from Visual Shader: %s"), p_name));

	VisualShader::VaryingMode var_mode = visual_shader->get_varying_mode(p_name);

	undo_redo->add_do_method(visual_shader.ptr(), "remove_varying", p_name);
	undo_redo->add_undo_method(visual_shader.ptr(), "add_varying", p_name, var_mode, visual_shader->get_varying_type(p_name));

	undo_redo->add_do_method(this, "_update_varyings");
	undo_redo->add_undo_method(this, "_update_varyings");

	for (int i = 0; i <= VisualShader::TYPE_LIGHT; i++) {
		if (var_mode == VisualShader::VARYING_MODE_FRAG_TO_LIGHT && i == 0) {
			continue;
		}

		VisualShader::Type type = VisualShader::Type(i);
		Vector<int> nodes = visual_shader->get_node_list(type);

		for (int j = 0; j < nodes.size(); j++) {
			int node_id = nodes[j];
			Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, node_id);
			Ref<VisualShaderNodeVarying> var = vsnode;

			if (var.is_valid()) {
				String var_name = var->get_varying_name();

				if (var_name == p_name) {
					undo_redo->add_do_method(var.ptr(), "set_varying_name", "[None]");
					undo_redo->add_undo_method(var.ptr(), "set_varying_name", var_name);
					undo_redo->add_do_method(var.ptr(), "set_varying_type", VisualShader::VARYING_TYPE_FLOAT);
					undo_redo->add_undo_method(var.ptr(), "set_varying_type", var->get_varying_type());
				}
				undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type, node_id);
				undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type, node_id);
			}
		}

		List<VisualShader::Connection> node_connections;
		visual_shader->get_node_connections(type, &node_connections);

		for (VisualShader::Connection &E : node_connections) {
			Ref<VisualShaderNodeVaryingGetter> var_getter = Object::cast_to<VisualShaderNodeVaryingGetter>(visual_shader->get_node(type, E.from_node).ptr());
			if (var_getter.is_valid() && E.from_port > 0) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
			}
			Ref<VisualShaderNodeVaryingSetter> var_setter = Object::cast_to<VisualShaderNodeVaryingSetter>(visual_shader->get_node(type, E.to_node).ptr());
			if (var_setter.is_valid() && E.to_port > 0) {
				undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
				undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
			}
		}
	}

	undo_redo->add_do_method(this, "_update_varying_tree");
	undo_redo->add_undo_method(this, "_update_varying_tree");
	undo_redo->commit_action();
}

void VisualShaderEditor::_update_varyings() {
	VisualShaderNodeVarying::clear_varyings();

	for (int i = 0; i < visual_shader->get_varyings_count(); i++) {
		const VisualShader::Varying *var = visual_shader->get_varying_by_index(i);

		if (var != nullptr) {
			VisualShaderNodeVarying::add_varying(var->name, var->mode, var->type);
		}
	}
}

void VisualShaderEditor::_node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_node) {
	VisualShader::Type type = get_current_shader_type();
	drag_buffer.push_back({ type, p_node, p_from, p_to });
	if (!drag_dirty) {
		call_deferred(SNAME("_nodes_dragged"));
	}
	drag_dirty = true;
}

void VisualShaderEditor::_nodes_dragged() {
	drag_dirty = false;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Node(s) Moved"));

	for (const DragOp &E : drag_buffer) {
		undo_redo->add_do_method(visual_shader.ptr(), "set_node_position", E.type, E.node, E.to);
		undo_redo->add_undo_method(visual_shader.ptr(), "set_node_position", E.type, E.node, E.from);
		undo_redo->add_do_method(graph_plugin.ptr(), "set_node_position", E.type, E.node, E.to);
		undo_redo->add_undo_method(graph_plugin.ptr(), "set_node_position", E.type, E.node, E.from);
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

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Nodes Connected"));

	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	for (const VisualShader::Connection &E : conns) {
		if (E.to_node == to && E.to_port == p_to_index) {
			undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
			undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
			undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
			undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
		}
	}

	undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", (int)type, to);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", (int)type, to);
	undo_redo->commit_action();
}

void VisualShaderEditor::_disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {
	graph->disconnect_node(p_from, p_from_index, p_to, p_to_index);

	VisualShader::Type type = get_current_shader_type();

	int from = p_from.to_int();
	int to = p_to.to_int();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Nodes Disconnected"));
	undo_redo->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, from, p_from_index, to, p_to_index);
	undo_redo->add_do_method(graph_plugin.ptr(), "update_node", (int)type, to);
	undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", (int)type, to);
	undo_redo->commit_action();
}

void VisualShaderEditor::_connection_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_position) {
	from_node = p_from.to_int();
	from_slot = p_from_slot;
	VisualShaderNode::PortType input_port_type = VisualShaderNode::PORT_TYPE_MAX;
	VisualShaderNode::PortType output_port_type = VisualShaderNode::PORT_TYPE_MAX;
	Ref<VisualShaderNode> node = visual_shader->get_node(get_current_shader_type(), from_node);
	if (node.is_valid()) {
		output_port_type = node->get_output_port_type(from_slot);
	}
	_show_members_dialog(true, input_port_type, output_port_type);
}

void VisualShaderEditor::_connection_from_empty(const String &p_to, int p_to_slot, const Vector2 &p_release_position) {
	to_node = p_to.to_int();
	to_slot = p_to_slot;
	VisualShaderNode::PortType input_port_type = VisualShaderNode::PORT_TYPE_MAX;
	VisualShaderNode::PortType output_port_type = VisualShaderNode::PORT_TYPE_MAX;
	Ref<VisualShaderNode> node = visual_shader->get_node(get_current_shader_type(), to_node);
	if (node.is_valid()) {
		input_port_type = node->get_input_port_type(to_slot);
	}
	_show_members_dialog(true, input_port_type, output_port_type);
}

void VisualShaderEditor::_delete_nodes(int p_type, const List<int> &p_nodes) {
	VisualShader::Type type = VisualShader::Type(p_type);
	List<VisualShader::Connection> conns;
	visual_shader->get_node_connections(type, &conns);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	for (const int &F : p_nodes) {
		for (const VisualShader::Connection &E : conns) {
			if (E.from_node == F || E.to_node == F) {
				undo_redo->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
			}
		}
	}

	HashSet<String> parameter_names;

	for (const int &F : p_nodes) {
		Ref<VisualShaderNode> node = visual_shader->get_node(type, F);

		undo_redo->add_do_method(visual_shader.ptr(), "remove_node", type, F);
		undo_redo->add_undo_method(visual_shader.ptr(), "add_node", type, node, visual_shader->get_node_position(type, F), F);
		undo_redo->add_undo_method(graph_plugin.ptr(), "add_node", type, F, false);

		VisualShaderNodeParameter *parameter = Object::cast_to<VisualShaderNodeParameter>(node.ptr());
		if (parameter) {
			parameter_names.insert(parameter->get_parameter_name());
		}
	}

	List<VisualShader::Connection> used_conns;
	for (const int &F : p_nodes) {
		for (const VisualShader::Connection &E : conns) {
			if (E.from_node == F || E.to_node == F) {
				bool cancel = false;
				for (List<VisualShader::Connection>::Element *R = used_conns.front(); R; R = R->next()) {
					if (R->get().from_node == E.from_node && R->get().from_port == E.from_port && R->get().to_node == E.to_node && R->get().to_port == E.to_port) {
						cancel = true; // to avoid ERR_ALREADY_EXISTS warning
						break;
					}
				}
				if (!cancel) {
					undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
					undo_redo->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
					used_conns.push_back(E);
				}
			}
		}
	}

	// Delete nodes from the graph.
	for (const int &F : p_nodes) {
		undo_redo->add_do_method(graph_plugin.ptr(), "remove_node", type, F, false);
	}

	// Update parameter refs if any parameter has been deleted.
	if (parameter_names.size() > 0) {
		undo_redo->add_do_method(this, "_update_parameters", true);
		undo_redo->add_undo_method(this, "_update_parameters", true);

		_update_parameter_refs(parameter_names);
	}
}

void VisualShaderEditor::_replace_node(VisualShader::Type p_type_id, int p_node_id, const StringName &p_from, const StringName &p_to) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(visual_shader.ptr(), "replace_node", p_type_id, p_node_id, p_to);
	undo_redo->add_undo_method(visual_shader.ptr(), "replace_node", p_type_id, p_node_id, p_from);
}

void VisualShaderEditor::_update_constant(VisualShader::Type p_type_id, int p_node_id, Variant p_var, int p_preview_port) {
	Ref<VisualShaderNode> node = visual_shader->get_node(p_type_id, p_node_id);
	ERR_FAIL_COND(!node.is_valid());
	ERR_FAIL_COND(!node->has_method("set_constant"));
	node->call("set_constant", p_var);
	if (p_preview_port != -1) {
		node->set_output_port_for_preview(p_preview_port);
	}
}

void VisualShaderEditor::_update_parameter(VisualShader::Type p_type_id, int p_node_id, Variant p_var, int p_preview_port) {
	Ref<VisualShaderNodeParameter> parameter = visual_shader->get_node(p_type_id, p_node_id);
	ERR_FAIL_COND(!parameter.is_valid());

	String valid_name = visual_shader->validate_parameter_name(parameter->get_parameter_name(), parameter);
	parameter->set_parameter_name(valid_name);
	graph_plugin->set_parameter_name(p_type_id, p_node_id, valid_name);

	if (parameter->has_method("set_default_value_enabled")) {
		parameter->call("set_default_value_enabled", true);
		parameter->call("set_default_value", p_var);
	}
	if (p_preview_port != -1) {
		parameter->set_output_port_for_preview(p_preview_port);
	}
}

void VisualShaderEditor::_convert_constants_to_parameters(bool p_vice_versa) {
	VisualShader::Type type_id = get_current_shader_type();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	if (!p_vice_versa) {
		undo_redo->create_action(TTR("Convert Constant Node(s) To Parameter(s)"));
	} else {
		undo_redo->create_action(TTR("Convert Parameter Node(s) To Constant(s)"));
	}

	const HashSet<int> &current_set = p_vice_versa ? selected_parameters : selected_constants;
	HashSet<String> deleted_names;

	for (const int &E : current_set) {
		int node_id = E;
		Ref<VisualShaderNode> node = visual_shader->get_node(type_id, node_id);
		bool caught = false;
		Variant var;

		// float
		if (!p_vice_versa) {
			Ref<VisualShaderNodeFloatConstant> float_const = Object::cast_to<VisualShaderNodeFloatConstant>(node.ptr());
			if (float_const.is_valid()) {
				_replace_node(type_id, node_id, "VisualShaderNodeFloatConstant", "VisualShaderNodeFloatParameter");
				var = float_const->get_constant();
				caught = true;
			}
		} else {
			Ref<VisualShaderNodeFloatParameter> float_parameter = Object::cast_to<VisualShaderNodeFloatParameter>(node.ptr());
			if (float_parameter.is_valid()) {
				_replace_node(type_id, node_id, "VisualShaderNodeFloatParameter", "VisualShaderNodeFloatConstant");
				var = float_parameter->get_default_value();
				caught = true;
			}
		}

		// int
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeIntConstant> int_const = Object::cast_to<VisualShaderNodeIntConstant>(node.ptr());
				if (int_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeIntConstant", "VisualShaderNodeIntParameter");
					var = int_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeIntParameter> int_parameter = Object::cast_to<VisualShaderNodeIntParameter>(node.ptr());
				if (int_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeIntParameter", "VisualShaderNodeIntConstant");
					var = int_parameter->get_default_value();
					caught = true;
				}
			}
		}

		// boolean
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeBooleanConstant> boolean_const = Object::cast_to<VisualShaderNodeBooleanConstant>(node.ptr());
				if (boolean_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeBooleanConstant", "VisualShaderNodeBooleanParameter");
					var = boolean_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeBooleanParameter> boolean_parameter = Object::cast_to<VisualShaderNodeBooleanParameter>(node.ptr());
				if (boolean_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeBooleanParameter", "VisualShaderNodeBooleanConstant");
					var = boolean_parameter->get_default_value();
					caught = true;
				}
			}
		}

		// vec2
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeVec2Constant> vec2_const = Object::cast_to<VisualShaderNodeVec2Constant>(node.ptr());
				if (vec2_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeVec2Constant", "VisualShaderNodeVec2Parameter");
					var = vec2_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeVec2Parameter> vec2_parameter = Object::cast_to<VisualShaderNodeVec2Parameter>(node.ptr());
				if (vec2_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeVec2Parameter", "VisualShaderNodeVec2Constant");
					var = vec2_parameter->get_default_value();
					caught = true;
				}
			}
		}

		// vec3
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeVec3Constant> vec3_const = Object::cast_to<VisualShaderNodeVec3Constant>(node.ptr());
				if (vec3_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeVec3Constant", "VisualShaderNodeVec3Parameter");
					var = vec3_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeVec3Parameter> vec3_parameter = Object::cast_to<VisualShaderNodeVec3Parameter>(node.ptr());
				if (vec3_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeVec3Parameter", "VisualShaderNodeVec3Constant");
					var = vec3_parameter->get_default_value();
					caught = true;
				}
			}
		}

		// vec4
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeVec4Constant> vec4_const = Object::cast_to<VisualShaderNodeVec4Constant>(node.ptr());
				if (vec4_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeVec4Constant", "VisualShaderNodeVec4Parameter");
					var = vec4_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeVec4Parameter> vec4_parameter = Object::cast_to<VisualShaderNodeVec4Parameter>(node.ptr());
				if (vec4_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeVec4Parameter", "VisualShaderNodeVec4Constant");
					var = vec4_parameter->get_default_value();
					caught = true;
				}
			}
		}

		// color
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeColorConstant> color_const = Object::cast_to<VisualShaderNodeColorConstant>(node.ptr());
				if (color_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeColorConstant", "VisualShaderNodeColorParameter");
					var = color_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeColorParameter> color_parameter = Object::cast_to<VisualShaderNodeColorParameter>(node.ptr());
				if (color_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeColorParameter", "VisualShaderNodeColorConstant");
					var = color_parameter->get_default_value();
					caught = true;
				}
			}
		}

		// transform
		if (!caught) {
			if (!p_vice_versa) {
				Ref<VisualShaderNodeTransformConstant> transform_const = Object::cast_to<VisualShaderNodeTransformConstant>(node.ptr());
				if (transform_const.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeTransformConstant", "VisualShaderNodeTransformParameter");
					var = transform_const->get_constant();
					caught = true;
				}
			} else {
				Ref<VisualShaderNodeTransformParameter> transform_parameter = Object::cast_to<VisualShaderNodeTransformParameter>(node.ptr());
				if (transform_parameter.is_valid()) {
					_replace_node(type_id, node_id, "VisualShaderNodeTransformParameter", "VisualShaderNodeTransformConstant");
					var = transform_parameter->get_default_value();
					caught = true;
				}
			}
		}
		ERR_CONTINUE(!caught);
		int preview_port = node->get_output_port_for_preview();

		if (!p_vice_versa) {
			undo_redo->add_do_method(this, "_update_parameter", type_id, node_id, var, preview_port);
			undo_redo->add_undo_method(this, "_update_constant", type_id, node_id, var, preview_port);
		} else {
			undo_redo->add_do_method(this, "_update_constant", type_id, node_id, var, preview_port);
			undo_redo->add_undo_method(this, "_update_parameter", type_id, node_id, var, preview_port);

			Ref<VisualShaderNodeParameter> parameter = Object::cast_to<VisualShaderNodeParameter>(node.ptr());
			ERR_CONTINUE(!parameter.is_valid());

			deleted_names.insert(parameter->get_parameter_name());
		}

		undo_redo->add_do_method(graph_plugin.ptr(), "update_node", type_id, node_id);
		undo_redo->add_undo_method(graph_plugin.ptr(), "update_node", type_id, node_id);
	}

	undo_redo->add_do_method(this, "_update_parameters", true);
	undo_redo->add_undo_method(this, "_update_parameters", true);

	if (deleted_names.size() > 0) {
		_update_parameter_refs(deleted_names);
	}

	undo_redo->commit_action();
}

void VisualShaderEditor::_delete_node_request(int p_type, int p_node) {
	Ref<VisualShaderNode> node = visual_shader->get_node((VisualShader::Type)p_type, p_node);
	if (!node->is_closable()) {
		return;
	}

	List<int> to_erase;
	to_erase.push_back(p_node);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Delete VisualShader Node"));
	_delete_nodes(p_type, to_erase);
	undo_redo->commit_action();
}

void VisualShaderEditor::_delete_nodes_request(const TypedArray<StringName> &p_nodes) {
	List<int> to_erase;

	if (p_nodes.is_empty()) {
		// Called from context menu.
		for (int i = 0; i < graph->get_child_count(); i++) {
			GraphElement *graph_element = Object::cast_to<GraphElement>(graph->get_child(i));
			if (graph_element && graph_element->is_selected()) {
				VisualShader::Type type = get_current_shader_type();
				int id = String(graph_element->get_name()).to_int();
				Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, id);
				if (vsnode->is_closable()) {
					to_erase.push_back(graph_element->get_name().operator String().to_int());
				}
			}
		}
	} else {
		VisualShader::Type type = get_current_shader_type();
		for (int i = 0; i < p_nodes.size(); i++) {
			int id = p_nodes[i].operator String().to_int();
			Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, id);
			if (vsnode->is_closable()) {
				to_erase.push_back(id);
			}
		}
	}

	if (to_erase.is_empty()) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Delete VisualShader Node(s)"));
	_delete_nodes(get_current_shader_type(), to_erase);
	undo_redo->commit_action();
}

void VisualShaderEditor::_node_selected(Object *p_node) {
	VisualShader::Type type = get_current_shader_type();

	GraphElement *graph_element = Object::cast_to<GraphElement>(p_node);
	ERR_FAIL_NULL(graph_element);

	int id = String(graph_element->get_name()).to_int();

	Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, id);
	ERR_FAIL_COND(!vsnode.is_valid());
}

void VisualShaderEditor::_graph_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	VisualShader::Type type = get_current_shader_type();

	Ref<VisualShaderNode> selected_vsnode;
	// Right click actions.
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) {
		selected_constants.clear();
		selected_parameters.clear();
		selected_comment = -1;
		selected_float_constant = -1;

		List<int> selected_closable_graph_elements;
		for (int i = 0; i < graph->get_child_count(); i++) {
			GraphElement *graph_element = Object::cast_to<GraphElement>(graph->get_child(i));
			if (graph_element && graph_element->is_selected()) {
				int id = String(graph_element->get_name()).to_int();
				Ref<VisualShaderNode> vsnode = visual_shader->get_node(type, id);
				if (!vsnode->is_closable()) {
					continue;
				}

				selected_closable_graph_elements.push_back(id);

				Ref<VisualShaderNode> node = visual_shader->get_node(type, id);
				selected_vsnode = node;

				VisualShaderNodeComment *frame_node = Object::cast_to<VisualShaderNodeComment>(node.ptr());
				if (frame_node != nullptr) {
					selected_comment = id;
				}
				VisualShaderNodeConstant *constant_node = Object::cast_to<VisualShaderNodeConstant>(node.ptr());
				if (constant_node != nullptr) {
					selected_constants.insert(id);
				}
				VisualShaderNodeFloatConstant *float_constant_node = Object::cast_to<VisualShaderNodeFloatConstant>(node.ptr());
				if (float_constant_node != nullptr) {
					selected_float_constant = id;
				}
				VisualShaderNodeParameter *parameter_node = Object::cast_to<VisualShaderNodeParameter>(node.ptr());
				if (parameter_node != nullptr && parameter_node->is_convertible_to_constant()) {
					selected_parameters.insert(id);
				}
			}
		}

		if (selected_closable_graph_elements.size() > 1) {
			selected_comment = -1;
			selected_float_constant = -1;
		}

		bool copy_buffer_empty = true;
		for (const CopyItem &item : copy_items_buffer) {
			if (!item.disabled) {
				copy_buffer_empty = false;
				break;
			}
		}

		if (selected_closable_graph_elements.is_empty() && copy_buffer_empty) {
			_show_members_dialog(true);
		} else {
			popup_menu->set_item_disabled(NodeMenuOptions::CUT, selected_closable_graph_elements.is_empty());
			popup_menu->set_item_disabled(NodeMenuOptions::COPY, selected_closable_graph_elements.is_empty());
			popup_menu->set_item_disabled(NodeMenuOptions::PASTE, copy_buffer_empty);
			popup_menu->set_item_disabled(NodeMenuOptions::DELETE, selected_closable_graph_elements.is_empty());
			popup_menu->set_item_disabled(NodeMenuOptions::DUPLICATE, selected_closable_graph_elements.is_empty());
			popup_menu->set_item_disabled(NodeMenuOptions::CLEAR_COPY_BUFFER, copy_buffer_empty);

			int temp = popup_menu->get_item_index(NodeMenuOptions::SEPARATOR2);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}
			temp = popup_menu->get_item_index(NodeMenuOptions::FLOAT_CONSTANTS);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}
			temp = popup_menu->get_item_index(NodeMenuOptions::CONVERT_CONSTANTS_TO_PARAMETERS);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}
			temp = popup_menu->get_item_index(NodeMenuOptions::CONVERT_PARAMETERS_TO_CONSTANTS);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}
			temp = popup_menu->get_item_index(NodeMenuOptions::SEPARATOR3);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}
			temp = popup_menu->get_item_index(NodeMenuOptions::SET_COMMENT_TITLE);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}
			temp = popup_menu->get_item_index(NodeMenuOptions::SET_COMMENT_DESCRIPTION);
			if (temp != -1) {
				popup_menu->remove_item(temp);
			}

			if (selected_constants.size() > 0 || selected_parameters.size() > 0) {
				popup_menu->add_separator("", NodeMenuOptions::SEPARATOR2);

				if (selected_float_constant != -1) {
					popup_menu->add_submenu_item(TTR("Float Constants"), "FloatConstants", int(NodeMenuOptions::FLOAT_CONSTANTS));

					if (!constants_submenu) {
						constants_submenu = memnew(PopupMenu);
						constants_submenu->set_name("FloatConstants");

						for (int i = 0; i < MAX_FLOAT_CONST_DEFS; i++) {
							constants_submenu->add_item(float_constant_defs[i].name, i);
						}
						popup_menu->add_child(constants_submenu);
						constants_submenu->connect("index_pressed", callable_mp(this, &VisualShaderEditor::_float_constant_selected));
					}
				}

				if (selected_constants.size() > 0) {
					popup_menu->add_item(TTR("Convert Constant(s) to Parameter(s)"), NodeMenuOptions::CONVERT_CONSTANTS_TO_PARAMETERS);
				}

				if (selected_parameters.size() > 0) {
					popup_menu->add_item(TTR("Convert Parameter(s) to Constant(s)"), NodeMenuOptions::CONVERT_PARAMETERS_TO_CONSTANTS);
				}
			}

			if (selected_comment != -1) {
				popup_menu->add_separator("", NodeMenuOptions::SEPARATOR3);
				popup_menu->add_item(TTR("Set Comment Title"), NodeMenuOptions::SET_COMMENT_TITLE);
				popup_menu->add_item(TTR("Set Comment Description"), NodeMenuOptions::SET_COMMENT_DESCRIPTION);
			}

			menu_point = graph->get_local_mouse_position();
			Point2 gpos = get_screen_position() + get_local_mouse_position();
			popup_menu->set_position(gpos);
			popup_menu->reset_size();
			popup_menu->popup();
		}
	}
}

void VisualShaderEditor::_show_members_dialog(bool at_mouse_pos, VisualShaderNode::PortType p_input_port_type, VisualShaderNode::PortType p_output_port_type) {
	if (members_input_port_type != p_input_port_type || members_output_port_type != p_output_port_type) {
		members_input_port_type = p_input_port_type;
		members_output_port_type = p_output_port_type;
		_update_options_menu();
	}

	if (at_mouse_pos) {
		saved_node_pos_dirty = true;
		saved_node_pos = graph->get_local_mouse_position();

		Point2 gpos = get_screen_position() + get_local_mouse_position();
		members_dialog->set_position(gpos);
	} else {
		saved_node_pos_dirty = false;
		members_dialog->set_position(graph->get_screen_position() + Point2(5 * EDSCALE, 65 * EDSCALE));
	}

	if (members_dialog->is_visible()) {
		members_dialog->grab_focus();
		return;
	}

	members_dialog->popup();

	// Keep dialog within window bounds.
	Rect2 window_rect = Rect2(get_window()->get_position(), get_window()->get_size());
	Rect2 dialog_rect = Rect2(members_dialog->get_position(), members_dialog->get_size());
	Vector2 difference = (dialog_rect.get_end() - window_rect.get_end()).max(Vector2());
	members_dialog->set_position(members_dialog->get_position() - difference);

	node_filter->call_deferred(SNAME("grab_focus")); // Still not visible.
	node_filter->select_all();
}

void VisualShaderEditor::_varying_menu_id_pressed(int p_idx) {
	switch (VaryingMenuOptions(p_idx)) {
		case VaryingMenuOptions::ADD: {
			_show_add_varying_dialog();
		} break;
		case VaryingMenuOptions::REMOVE: {
			_show_remove_varying_dialog();
		} break;
		default:
			break;
	}
}

void VisualShaderEditor::_show_add_varying_dialog() {
	_varying_name_changed(varying_name->get_text());

	add_varying_dialog->set_position(graph->get_screen_position() + varying_button->get_position() + Point2(5 * EDSCALE, 65 * EDSCALE));
	add_varying_dialog->popup();

	// Keep dialog within window bounds.
	Rect2 window_rect = Rect2(DisplayServer::get_singleton()->window_get_position(), DisplayServer::get_singleton()->window_get_size());
	Rect2 dialog_rect = Rect2(add_varying_dialog->get_position(), add_varying_dialog->get_size());
	Vector2 difference = (dialog_rect.get_end() - window_rect.get_end()).max(Vector2());
	add_varying_dialog->set_position(add_varying_dialog->get_position() - difference);
}

void VisualShaderEditor::_show_remove_varying_dialog() {
	remove_varying_dialog->set_position(graph->get_screen_position() + varying_button->get_position() + Point2(5 * EDSCALE, 65 * EDSCALE));
	remove_varying_dialog->popup();

	// Keep dialog within window bounds.
	Rect2 window_rect = Rect2(DisplayServer::get_singleton()->window_get_position(), DisplayServer::get_singleton()->window_get_size());
	Rect2 dialog_rect = Rect2(remove_varying_dialog->get_position(), remove_varying_dialog->get_size());
	Vector2 difference = (dialog_rect.get_end() - window_rect.get_end()).max(Vector2());
	remove_varying_dialog->set_position(remove_varying_dialog->get_position() - difference);
}

void VisualShaderEditor::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> ie = p_ie;
	if (ie.is_valid() && (ie->get_keycode() == Key::UP || ie->get_keycode() == Key::DOWN || ie->get_keycode() == Key::ENTER || ie->get_keycode() == Key::KP_ENTER)) {
		members->gui_input(ie);
		node_filter->accept_event();
	}
}

void VisualShaderEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			_update_options_menu();
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			graph->get_panner()->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/sub_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EDITOR_GET("editors/panning/simple_panning")));
			graph->set_warped_panning(bool(EDITOR_GET("editors/panning/warped_mouse_panning")));
			graph->set_minimap_opacity(EDITOR_GET("editors/visual_editors/minimap_opacity"));
			graph->set_connection_lines_curvature(EDITOR_GET("editors/visual_editors/lines_curvature"));
			_update_graph();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			node_filter->set_clear_button_enabled(true);

			// collapse tree by default

			TreeItem *category = members->get_root()->get_first_child();
			while (category) {
				category->set_collapsed(true);
				TreeItem *sub_category = category->get_first_child();
				while (sub_category) {
					sub_category->set_collapsed(true);
					sub_category = sub_category->get_next();
				}
				category = category->get_next();
			}

			graph->get_panner()->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/sub_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EDITOR_GET("editors/panning/simple_panning")));
			graph->set_warped_panning(bool(EDITOR_GET("editors/panning/warped_mouse_panning")));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			highend_label->set_modulate(get_theme_color(SNAME("highend_color"), EditorStringName(Editor)));

			node_filter->set_right_icon(Control::get_editor_theme_icon(SNAME("Search")));

			preview_shader->set_icon(Control::get_editor_theme_icon(SNAME("Shader")));

			{
				Color background_color = EDITOR_GET("text_editor/theme/highlighting/background_color");
				Color text_color = EDITOR_GET("text_editor/theme/highlighting/text_color");
				Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
				Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");
				Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
				Color symbol_color = EDITOR_GET("text_editor/theme/highlighting/symbol_color");
				Color function_color = EDITOR_GET("text_editor/theme/highlighting/function_color");
				Color number_color = EDITOR_GET("text_editor/theme/highlighting/number_color");
				Color members_color = EDITOR_GET("text_editor/theme/highlighting/member_variable_color");
				Color error_color = get_theme_color(SNAME("error_color"), EditorStringName(Editor));

				preview_text->add_theme_color_override("background_color", background_color);
				varying_error_label->add_theme_color_override("font_color", error_color);

				for (const String &E : keyword_list) {
					if (ShaderLanguage::is_control_flow_keyword(E)) {
						syntax_highlighter->add_keyword_color(E, control_flow_keyword_color);
					} else {
						syntax_highlighter->add_keyword_color(E, keyword_color);
					}
				}

				preview_text->begin_bulk_theme_override();
				preview_text->add_theme_font_override("font", get_theme_font(SNAME("expression"), EditorStringName(EditorFonts)));
				preview_text->add_theme_font_size_override("font_size", get_theme_font_size(SNAME("expression_size"), EditorStringName(EditorFonts)));
				preview_text->add_theme_color_override("font_color", text_color);
				preview_text->end_bulk_theme_override();

				syntax_highlighter->set_number_color(number_color);
				syntax_highlighter->set_symbol_color(symbol_color);
				syntax_highlighter->set_function_color(function_color);
				syntax_highlighter->set_member_variable_color(members_color);
				syntax_highlighter->clear_color_regions();
				syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
				syntax_highlighter->add_color_region("//", "", comment_color, true);

				preview_text->clear_comment_delimiters();
				preview_text->add_comment_delimiter("/*", "*/", false);
				preview_text->add_comment_delimiter("//", "", true);

				error_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("Panel")));
				error_label->begin_bulk_theme_override();
				error_label->add_theme_font_override("font", get_theme_font(SNAME("status_source"), EditorStringName(EditorFonts)));
				error_label->add_theme_font_size_override("font_size", get_theme_font_size(SNAME("status_source_size"), EditorStringName(EditorFonts)));
				error_label->add_theme_color_override("font_color", error_color);
				error_label->end_bulk_theme_override();
			}

			tools->set_icon(get_editor_theme_icon(SNAME("Tools")));

			if (p_what == NOTIFICATION_THEME_CHANGED && is_visible_in_tree()) {
				_update_graph();
			}
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			Dictionary dd = get_viewport()->gui_get_drag_data();
			if (members->is_visible_in_tree() && dd.has("id")) {
				members->set_drop_mode_flags(Tree::DROP_MODE_ON_ITEM);
			}
		} break;

		case NOTIFICATION_DRAG_END: {
			members->set_drop_mode_flags(0);
		} break;
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

void VisualShaderEditor::_dup_copy_nodes(int p_type, List<CopyItem> &r_items, List<VisualShader::Connection> &r_connections) {
	VisualShader::Type type = (VisualShader::Type)p_type;

	selection_center.x = 0.0f;
	selection_center.y = 0.0f;

	HashSet<int> nodes;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphElement *graph_element = Object::cast_to<GraphElement>(graph->get_child(i));
		if (graph_element) {
			int id = String(graph_element->get_name()).to_int();

			Ref<VisualShaderNode> node = visual_shader->get_node(type, id);
			Ref<VisualShaderNodeOutput> output = node;
			if (output.is_valid()) { // can't duplicate output
				continue;
			}

			if (node.is_valid() && graph_element->is_selected()) {
				Vector2 pos = visual_shader->get_node_position(type, id);
				selection_center += pos;

				CopyItem item;
				item.id = id;
				item.node = visual_shader->get_node(type, id)->duplicate();
				item.position = visual_shader->get_node_position(type, id);

				Ref<VisualShaderNodeResizableBase> resizable_base = node;
				if (resizable_base.is_valid()) {
					item.size = resizable_base->get_size();
				}

				Ref<VisualShaderNodeGroupBase> group = node;
				if (group.is_valid()) {
					item.group_inputs = group->get_inputs();
					item.group_outputs = group->get_outputs();
				}

				Ref<VisualShaderNodeExpression> expression = node;
				if (expression.is_valid()) {
					item.expression = expression->get_expression();
				}

				r_items.push_back(item);

				nodes.insert(id);
			}
		}
	}

	List<VisualShader::Connection> node_connections;
	visual_shader->get_node_connections(type, &node_connections);

	for (const VisualShader::Connection &E : node_connections) {
		if (nodes.has(E.from_node) && nodes.has(E.to_node)) {
			r_connections.push_back(E);
		}
	}

	selection_center /= (float)r_items.size();
}

void VisualShaderEditor::_dup_paste_nodes(int p_type, List<CopyItem> &r_items, const List<VisualShader::Connection> &p_connections, const Vector2 &p_offset, bool p_duplicate) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	if (p_duplicate) {
		undo_redo->create_action(TTR("Duplicate VisualShader Node(s)"));
	} else {
		bool copy_buffer_empty = true;
		for (const CopyItem &item : copy_items_buffer) {
			if (!item.disabled) {
				copy_buffer_empty = false;
				break;
			}
		}
		if (copy_buffer_empty) {
			return;
		}

		undo_redo->create_action(TTR("Paste VisualShader Node(s)"));
	}

	VisualShader::Type type = (VisualShader::Type)p_type;

	int base_id = visual_shader->get_valid_node_id(type);
	int id_from = base_id;
	HashMap<int, int> connection_remap;
	HashSet<int> unsupported_set;
	HashSet<int> added_set;

	for (CopyItem &item : r_items) {
		if (item.disabled) {
			unsupported_set.insert(item.id);
			continue;
		}
		connection_remap[item.id] = id_from;
		Ref<VisualShaderNode> node = item.node->duplicate();

		Ref<VisualShaderNodeResizableBase> resizable_base = Object::cast_to<VisualShaderNodeResizableBase>(node.ptr());
		if (resizable_base.is_valid()) {
			undo_redo->add_do_method(node.ptr(), "set_size", item.size);
		}

		Ref<VisualShaderNodeGroupBase> group = Object::cast_to<VisualShaderNodeGroupBase>(node.ptr());
		if (group.is_valid()) {
			undo_redo->add_do_method(node.ptr(), "set_inputs", item.group_inputs);
			undo_redo->add_do_method(node.ptr(), "set_outputs", item.group_outputs);
		}

		Ref<VisualShaderNodeExpression> expression = Object::cast_to<VisualShaderNodeExpression>(node.ptr());
		if (expression.is_valid()) {
			undo_redo->add_do_method(node.ptr(), "set_expression", item.expression);
		}

		undo_redo->add_do_method(visual_shader.ptr(), "add_node", type, node, item.position + p_offset, id_from);
		undo_redo->add_do_method(graph_plugin.ptr(), "add_node", type, id_from, false);

		added_set.insert(id_from);
		id_from++;
	}

	for (const VisualShader::Connection &E : p_connections) {
		if (unsupported_set.has(E.from_node) || unsupported_set.has(E.to_node)) {
			continue;
		}

		undo_redo->add_do_method(visual_shader.ptr(), "connect_nodes", type, connection_remap[E.from_node], E.from_port, connection_remap[E.to_node], E.to_port);
		undo_redo->add_do_method(graph_plugin.ptr(), "connect_nodes", type, connection_remap[E.from_node], E.from_port, connection_remap[E.to_node], E.to_port);
		undo_redo->add_undo_method(graph_plugin.ptr(), "disconnect_nodes", type, connection_remap[E.from_node], E.from_port, connection_remap[E.to_node], E.to_port);
	}

	id_from = base_id;
	for (const CopyItem &item : r_items) {
		if (item.disabled) {
			continue;
		}
		undo_redo->add_undo_method(visual_shader.ptr(), "remove_node", type, id_from);
		undo_redo->add_undo_method(graph_plugin.ptr(), "remove_node", type, id_from, false);
		id_from++;
	}

	undo_redo->commit_action();

	// reselect nodes by excluding the other ones
	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphElement *graph_element = Object::cast_to<GraphElement>(graph->get_child(i));
		if (graph_element) {
			int id = String(graph_element->get_name()).to_int();
			if (added_set.has(id)) {
				graph_element->set_selected(true);
			} else {
				graph_element->set_selected(false);
			}
		}
	}
}

void VisualShaderEditor::_clear_copy_buffer() {
	copy_items_buffer.clear();
	copy_connections_buffer.clear();
}

void VisualShaderEditor::_duplicate_nodes() {
	int type = get_current_shader_type();

	List<CopyItem> items;
	List<VisualShader::Connection> node_connections;

	_dup_copy_nodes(type, items, node_connections);

	if (items.is_empty()) {
		return;
	}

	_dup_paste_nodes(type, items, node_connections, Vector2(10, 10) * EDSCALE, true);
}

void VisualShaderEditor::_copy_nodes(bool p_cut) {
	_clear_copy_buffer();

	_dup_copy_nodes(get_current_shader_type(), copy_items_buffer, copy_connections_buffer);

	if (p_cut) {
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Cut VisualShader Node(s)"));

		List<int> ids;
		for (const CopyItem &E : copy_items_buffer) {
			ids.push_back(E.id);
		}

		_delete_nodes(get_current_shader_type(), ids);

		undo_redo->commit_action();
	}
}

void VisualShaderEditor::_paste_nodes(bool p_use_custom_position, const Vector2 &p_custom_position) {
	if (copy_items_buffer.is_empty()) {
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

	_dup_paste_nodes(type, copy_items_buffer, copy_connections_buffer, graph->get_scroll_offset() / scale + mpos / scale - selection_center, false);
}

void VisualShaderEditor::_mode_selected(int p_id) {
	int offset = 0;
	if (mode & MODE_FLAGS_PARTICLES) {
		offset = 3;
		if (p_id + offset > VisualShader::TYPE_PROCESS) {
			custom_mode_box->set_visible(false);
			custom_mode_enabled = false;
		} else {
			custom_mode_box->set_visible(true);
			if (custom_mode_box->is_pressed()) {
				custom_mode_enabled = true;
				offset += 3;
			}
		}
	} else if (mode & MODE_FLAGS_SKY) {
		offset = 8;
	} else if (mode & MODE_FLAGS_FOG) {
		offset = 9;
	}

	visual_shader->set_shader_type(VisualShader::Type(p_id + offset));
	_update_nodes();
	_update_graph();

	graph->grab_focus();
}

void VisualShaderEditor::_custom_mode_toggled(bool p_enabled) {
	if (!(mode & MODE_FLAGS_PARTICLES)) {
		return;
	}
	custom_mode_enabled = p_enabled;
	int id = edit_type->get_selected() + 3;
	if (p_enabled) {
		visual_shader->set_shader_type(VisualShader::Type(id + 3));
	} else {
		visual_shader->set_shader_type(VisualShader::Type(id));
	}
	_update_options_menu();
	_update_graph();
}

void VisualShaderEditor::_input_select_item(Ref<VisualShaderNodeInput> p_input, String p_name) {
	String prev_name = p_input->get_input_name();

	if (p_name == prev_name) {
		return;
	}

	VisualShaderNode::PortType next_input_type = p_input->get_input_type_by_name(p_name);
	VisualShaderNode::PortType prev_input_type = p_input->get_input_type_by_name(prev_name);

	bool type_changed = next_input_type != prev_input_type;

	EditorUndoRedoManager *undo_redo_man = EditorUndoRedoManager::get_singleton();
	undo_redo_man->create_action(TTR("Visual Shader Input Type Changed"));

	undo_redo_man->add_do_method(p_input.ptr(), "set_input_name", p_name);
	undo_redo_man->add_undo_method(p_input.ptr(), "set_input_name", prev_name);

	if (type_changed) {
		for (int type_id = 0; type_id < VisualShader::TYPE_MAX; type_id++) {
			VisualShader::Type type = VisualShader::Type(type_id);

			int id = visual_shader->find_node_id(type, p_input);
			if (id != VisualShader::NODE_ID_INVALID) {
				bool is_expanded = p_input->is_output_port_expandable(0) && p_input->_is_output_port_expanded(0);

				int type_size = 0;
				if (is_expanded) {
					switch (next_input_type) {
						case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
							type_size = 2;
						} break;
						case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
							type_size = 3;
						} break;
						case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
							type_size = 4;
						} break;
						default:
							break;
					}
				}

				List<VisualShader::Connection> conns;
				visual_shader->get_node_connections(type, &conns);
				for (const VisualShader::Connection &E : conns) {
					int cn_from_node = E.from_node;
					int cn_from_port = E.from_port;
					int cn_to_node = E.to_node;
					int cn_to_port = E.to_port;

					if (cn_from_node == id) {
						bool is_incompatible_types = !visual_shader->is_port_types_compatible(p_input->get_input_type_by_name(p_name), visual_shader->get_node(type, cn_to_node)->get_input_port_type(cn_to_port));

						if (is_incompatible_types || cn_from_port > type_size) {
							undo_redo_man->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
							undo_redo_man->add_undo_method(visual_shader.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
							undo_redo_man->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
							undo_redo_man->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, cn_from_node, cn_from_port, cn_to_node, cn_to_port);
						}
					}
				}

				undo_redo_man->add_do_method(graph_plugin.ptr(), "update_node", type_id, id);
				undo_redo_man->add_undo_method(graph_plugin.ptr(), "update_node", type_id, id);
			}
		}
	}

	undo_redo_man->commit_action();
}

void VisualShaderEditor::_parameter_ref_select_item(Ref<VisualShaderNodeParameterRef> p_parameter_ref, String p_name) {
	String prev_name = p_parameter_ref->get_parameter_name();

	if (p_name == prev_name) {
		return;
	}

	bool type_changed = p_parameter_ref->get_parameter_type_by_name(p_name) != p_parameter_ref->get_parameter_type_by_name(prev_name);

	EditorUndoRedoManager *undo_redo_man = EditorUndoRedoManager::get_singleton();
	undo_redo_man->create_action(TTR("ParameterRef Name Changed"));

	undo_redo_man->add_do_method(p_parameter_ref.ptr(), "set_parameter_name", p_name);
	undo_redo_man->add_undo_method(p_parameter_ref.ptr(), "set_parameter_name", prev_name);

	// update output port
	for (int type_id = 0; type_id < VisualShader::TYPE_MAX; type_id++) {
		VisualShader::Type type = VisualShader::Type(type_id);
		int id = visual_shader->find_node_id(type, p_parameter_ref);
		if (id != VisualShader::NODE_ID_INVALID) {
			if (type_changed) {
				List<VisualShader::Connection> conns;
				visual_shader->get_node_connections(type, &conns);
				for (const VisualShader::Connection &E : conns) {
					if (E.from_node == id) {
						if (visual_shader->is_port_types_compatible(p_parameter_ref->get_parameter_type_by_name(p_name), visual_shader->get_node(type, E.to_node)->get_input_port_type(E.to_port))) {
							continue;
						}
						undo_redo_man->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
						undo_redo_man->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
						undo_redo_man->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
						undo_redo_man->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
					}
				}
			}
			undo_redo_man->add_do_method(graph_plugin.ptr(), "update_node", type_id, id);
			undo_redo_man->add_undo_method(graph_plugin.ptr(), "update_node", type_id, id);
			break;
		}
	}

	undo_redo_man->commit_action();
}

void VisualShaderEditor::_varying_select_item(Ref<VisualShaderNodeVarying> p_varying, String p_name) {
	String prev_name = p_varying->get_varying_name();

	if (p_name == prev_name) {
		return;
	}

	bool is_getter = Ref<VisualShaderNodeVaryingGetter>(p_varying.ptr()).is_valid();

	EditorUndoRedoManager *undo_redo_man = EditorUndoRedoManager::get_singleton();
	undo_redo_man->create_action(TTR("Varying Name Changed"));

	undo_redo_man->add_do_method(p_varying.ptr(), "set_varying_name", p_name);
	undo_redo_man->add_undo_method(p_varying.ptr(), "set_varying_name", prev_name);

	VisualShader::VaryingType vtype = p_varying->get_varying_type_by_name(p_name);
	VisualShader::VaryingType prev_vtype = p_varying->get_varying_type_by_name(prev_name);

	bool type_changed = vtype != prev_vtype;

	if (type_changed) {
		undo_redo_man->add_do_method(p_varying.ptr(), "set_varying_type", vtype);
		undo_redo_man->add_undo_method(p_varying.ptr(), "set_varying_type", prev_vtype);
	}

	// update ports
	for (int type_id = 0; type_id < VisualShader::TYPE_MAX; type_id++) {
		VisualShader::Type type = VisualShader::Type(type_id);
		int id = visual_shader->find_node_id(type, p_varying);

		if (id != VisualShader::NODE_ID_INVALID) {
			if (type_changed) {
				List<VisualShader::Connection> conns;
				visual_shader->get_node_connections(type, &conns);

				for (const VisualShader::Connection &E : conns) {
					if (is_getter) {
						if (E.from_node == id) {
							if (visual_shader->is_port_types_compatible(p_varying->get_varying_type_by_name(p_name), visual_shader->get_node(type, E.to_node)->get_input_port_type(E.to_port))) {
								continue;
							}
							undo_redo_man->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
							undo_redo_man->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
							undo_redo_man->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
							undo_redo_man->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
						}
					} else {
						if (E.to_node == id) {
							if (visual_shader->is_port_types_compatible(p_varying->get_varying_type_by_name(p_name), visual_shader->get_node(type, E.from_node)->get_output_port_type(E.from_port))) {
								continue;
							}
							undo_redo_man->add_do_method(visual_shader.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
							undo_redo_man->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
							undo_redo_man->add_do_method(graph_plugin.ptr(), "disconnect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
							undo_redo_man->add_undo_method(graph_plugin.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
						}
					}
				}
			}

			undo_redo_man->add_do_method(graph_plugin.ptr(), "update_node", type_id, id);
			undo_redo_man->add_undo_method(graph_plugin.ptr(), "update_node", type_id, id);
			break;
		}
	}

	undo_redo_man->commit_action();
}

void VisualShaderEditor::_float_constant_selected(int p_which) {
	ERR_FAIL_INDEX(p_which, MAX_FLOAT_CONST_DEFS);

	VisualShader::Type type = get_current_shader_type();
	Ref<VisualShaderNodeFloatConstant> node = visual_shader->get_node(type, selected_float_constant);
	ERR_FAIL_COND(!node.is_valid());

	if (Math::is_equal_approx(node->get_constant(), float_constant_defs[p_which].value)) {
		return; // same
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTR("Set Constant: %s"), float_constant_defs[p_which].name));
	undo_redo->add_do_method(node.ptr(), "set_constant", float_constant_defs[p_which].value);
	undo_redo->add_undo_method(node.ptr(), "set_constant", node->get_constant());
	undo_redo->commit_action();
}

void VisualShaderEditor::_member_filter_changed(const String &p_text) {
	_update_options_menu();
}

void VisualShaderEditor::_member_selected() {
	TreeItem *item = members->get_selected();

	if (item != nullptr && item->has_meta("id")) {
		members_dialog->get_ok_button()->set_disabled(false);
		highend_label->set_visible(add_options[item->get_meta("id")].highend);
		node_desc->set_text(_get_description(item->get_meta("id")));
	} else {
		highend_label->set_visible(false);
		members_dialog->get_ok_button()->set_disabled(true);
		node_desc->set_text("");
	}
}

void VisualShaderEditor::_member_unselected() {
}

void VisualShaderEditor::_member_create() {
	TreeItem *item = members->get_selected();
	if (item != nullptr && item->has_meta("id")) {
		int idx = members->get_selected()->get_meta("id");
		_add_node(idx, add_options[idx].ops);
		members_dialog->hide();
	}
}

void VisualShaderEditor::_member_cancel() {
	to_node = -1;
	to_slot = -1;
	from_node = -1;
	from_slot = -1;
}

void VisualShaderEditor::_update_varying_tree() {
	varyings->clear();
	TreeItem *root = varyings->create_item();

	int count = visual_shader->get_varyings_count();

	for (int i = 0; i < count; i++) {
		const VisualShader::Varying *varying = visual_shader->get_varying_by_index(i);

		if (varying) {
			TreeItem *item = varyings->create_item(root);
			item->set_text(0, varying->name);

			if (i == 0) {
				item->select(0);
			}

			switch (varying->type) {
				case VisualShader::VARYING_TYPE_FLOAT:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("float"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_INT:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("int"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_UINT:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("uint"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_VECTOR_2D:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector2"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_VECTOR_3D:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector3"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_VECTOR_4D:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector4"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_BOOLEAN:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("bool"), EditorStringName(EditorIcons)));
					break;
				case VisualShader::VARYING_TYPE_TRANSFORM:
					item->set_icon(0, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Transform3D"), EditorStringName(EditorIcons)));
					break;
				default:
					break;
			}
		}
	}

	varying_button->get_popup()->set_item_disabled(int(VaryingMenuOptions::REMOVE), count == 0);
}

void VisualShaderEditor::_varying_create() {
	_add_varying(varying_name->get_text(), (VisualShader::VaryingMode)varying_mode->get_selected(), (VisualShader::VaryingType)varying_type->get_selected());
	add_varying_dialog->hide();
}

void VisualShaderEditor::_varying_name_changed(const String &p_text) {
	String name = p_text;

	if (!name.is_valid_identifier()) {
		varying_error_label->show();
		varying_error_label->set_text(TTR("Invalid name for varying."));
		add_varying_dialog->get_ok_button()->set_disabled(true);
		return;
	}
	if (visual_shader->has_varying(name)) {
		varying_error_label->show();
		varying_error_label->set_text(TTR("Varying with that name is already exist."));
		add_varying_dialog->get_ok_button()->set_disabled(true);
		return;
	}
	if (varying_error_label->is_visible()) {
		varying_error_label->hide();
		add_varying_dialog->set_size(Size2(add_varying_dialog->get_size().x, 0));
	}
	add_varying_dialog->get_ok_button()->set_disabled(false);
}

void VisualShaderEditor::_varying_deleted() {
	TreeItem *item = varyings->get_selected();

	if (item != nullptr) {
		_remove_varying(item->get_text(0));
		remove_varying_dialog->hide();
	}
}

void VisualShaderEditor::_varying_selected() {
	add_varying_dialog->get_ok_button()->set_disabled(false);
}

void VisualShaderEditor::_varying_unselected() {
	add_varying_dialog->get_ok_button()->set_disabled(true);
}

void VisualShaderEditor::_tools_menu_option(int p_idx) {
	TreeItem *category = members->get_root()->get_first_child();

	switch (p_idx) {
		case EXPAND_ALL:

			while (category) {
				category->set_collapsed(false);
				TreeItem *sub_category = category->get_first_child();
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
				TreeItem *sub_category = category->get_first_child();
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
		case NodeMenuOptions::CUT:
			_copy_nodes(true);
			break;
		case NodeMenuOptions::COPY:
			_copy_nodes(false);
			break;
		case NodeMenuOptions::PASTE:
			_paste_nodes(true, menu_point);
			break;
		case NodeMenuOptions::DELETE:
			_delete_nodes_request(TypedArray<StringName>());
			break;
		case NodeMenuOptions::DUPLICATE:
			_duplicate_nodes();
			break;
		case NodeMenuOptions::CLEAR_COPY_BUFFER:
			_clear_copy_buffer();
			break;
		case NodeMenuOptions::CONVERT_CONSTANTS_TO_PARAMETERS:
			_convert_constants_to_parameters(false);
			break;
		case NodeMenuOptions::CONVERT_PARAMETERS_TO_CONSTANTS:
			_convert_constants_to_parameters(true);
			break;
		case NodeMenuOptions::SET_COMMENT_TITLE:
			_comment_title_popup_show(get_screen_position() + get_local_mouse_position(), selected_comment);
			break;
		case NodeMenuOptions::SET_COMMENT_DESCRIPTION:
			_comment_desc_popup_show(get_screen_position() + get_local_mouse_position(), selected_comment);
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
			_add_node(idx, add_options[idx].ops);
		} else if (d.has("files")) {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Add Node(s) to Visual Shader"));

			if (d["files"].get_type() == Variant::PACKED_STRING_ARRAY) {
				PackedStringArray arr = d["files"];
				for (int i = 0; i < arr.size(); i++) {
					String type = ResourceLoader::get_resource_type(arr[i]);
					if (type == "GDScript") {
						Ref<Script> scr = ResourceLoader::load(arr[i]);
						if (scr->get_instance_base_type() == "VisualShaderNodeCustom") {
							saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
							saved_node_pos_dirty = true;

							int idx = -1;

							for (int j = custom_node_option_idx; j < add_options.size(); j++) {
								if (add_options[j].script.is_valid()) {
									if (add_options[j].script->get_path() == arr[i]) {
										idx = j;
										break;
									}
								}
							}
							if (idx != -1) {
								_add_node(idx, {}, arr[i], i);
							}
						}
					} else if (type == "CurveTexture") {
						saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_node(curve_node_option_idx, {}, arr[i], i);
					} else if (type == "CurveXYZTexture") {
						saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_node(curve_xyz_node_option_idx, {}, arr[i], i);
					} else if (ClassDB::get_parent_class(type) == "Texture2D") {
						saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_node(texture2d_node_option_idx, {}, arr[i], i);
					} else if (type == "Texture2DArray") {
						saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_node(texture2d_array_node_option_idx, {}, arr[i], i);
					} else if (ClassDB::get_parent_class(type) == "Texture3D") {
						saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_node(texture3d_node_option_idx, {}, arr[i], i);
					} else if (type == "Cubemap") {
						saved_node_pos = p_point + Vector2(0, i * 250 * EDSCALE);
						saved_node_pos_dirty = true;
						_add_node(cubemap_node_option_idx, {}, arr[i], i);
					}
				}
			}
			undo_redo->commit_action();
		}
	}
}

void VisualShaderEditor::_show_preview_text() {
	preview_showed = !preview_showed;
	if (preview_showed) {
		if (preview_first) {
			preview_window->set_size(Size2(400 * EDSCALE, 600 * EDSCALE));
			preview_window->popup_centered();
			preview_first = false;
		} else {
			preview_window->popup();
		}
		_preview_size_changed();

		if (pending_update_preview) {
			_update_preview();
			pending_update_preview = false;
		}
	} else {
		preview_window->hide();
	}
}

void VisualShaderEditor::_preview_close_requested() {
	preview_showed = false;
	preview_window->hide();
	preview_shader->set_pressed(false);
}

void VisualShaderEditor::_preview_size_changed() {
	preview_vbox->set_custom_minimum_size(preview_window->get_size());
}

static ShaderLanguage::DataType _visual_shader_editor_get_global_shader_uniform_type(const StringName &p_variable) {
	RS::GlobalShaderParameterType gvt = RS::get_singleton()->global_shader_parameter_get_type(p_variable);
	return (ShaderLanguage::DataType)RS::global_shader_uniform_type_get_shader_datatype(gvt);
}

void VisualShaderEditor::_update_preview() {
	if (!preview_showed) {
		pending_update_preview = true;
		return;
	}

	String code = visual_shader->get_code();

	preview_text->set_text(code);

	ShaderLanguage::ShaderCompileInfo info;
	info.functions = ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(visual_shader->get_mode()));
	info.render_modes = ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(visual_shader->get_mode()));
	info.shader_types = ShaderTypes::get_singleton()->get_types();
	info.global_shader_uniform_type_func = _visual_shader_editor_get_global_shader_uniform_type;

	for (int i = 0; i < preview_text->get_line_count(); i++) {
		preview_text->set_line_background_color(i, Color(0, 0, 0, 0));
	}

	String preprocessed_code;
	{
		String path = visual_shader->get_path();
		String error_pp;
		List<ShaderPreprocessor::FilePosition> err_positions;
		ShaderPreprocessor preprocessor;
		Error err = preprocessor.preprocess(code, path, preprocessed_code, &error_pp, &err_positions);
		if (err != OK) {
			ERR_FAIL_COND(err_positions.is_empty());

			String file = err_positions.front()->get().file;
			int err_line = err_positions.front()->get().line;
			Color error_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
			preview_text->set_line_background_color(err_line - 1, error_line_color);
			error_panel->show();

			error_label->set_text("error(" + file + ":" + itos(err_line) + "): " + error_pp);
			shader_error = true;
			return;
		}
	}

	ShaderLanguage sl;
	Error err = sl.compile(preprocessed_code, info);
	if (err != OK) {
		int err_line;
		String err_text;
		Vector<ShaderLanguage::FilePosition> include_positions = sl.get_include_positions();
		if (include_positions.size() > 1) {
			// Error is in an include.
			err_line = include_positions[0].line;
			err_text = "error(" + itos(err_line) + ") in include " + include_positions[include_positions.size() - 1].file + ":" + itos(include_positions[include_positions.size() - 1].line) + ": " + sl.get_error_text();
		} else {
			err_line = sl.get_error_line();
			err_text = "error(" + itos(err_line) + "): " + sl.get_error_text();
		}

		Color error_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
		preview_text->set_line_background_color(err_line - 1, error_line_color);
		error_panel->show();

		error_label->set_text(err_text);
		shader_error = true;
	} else {
		error_panel->hide();
		shader_error = false;
	}
}

void VisualShaderEditor::_update_next_previews(int p_node_id) {
	VisualShader::Type type = get_current_shader_type();

	LocalVector<int> nodes;
	_get_next_nodes_recursively(type, p_node_id, nodes);

	for (int node_id : nodes) {
		if (graph_plugin->is_preview_visible(node_id)) {
			graph_plugin->update_node_deferred(type, node_id);
		}
	}
}

void VisualShaderEditor::_get_next_nodes_recursively(VisualShader::Type p_type, int p_node_id, LocalVector<int> &r_nodes) const {
	const LocalVector<int> &next_connections = visual_shader->get_next_connected_nodes(p_type, p_node_id);

	for (int node_id : next_connections) {
		r_nodes.push_back(node_id);
		_get_next_nodes_recursively(p_type, node_id, r_nodes);
	}
}

void VisualShaderEditor::_visibility_changed() {
	if (!is_visible()) {
		if (preview_window->is_visible()) {
			preview_shader->set_pressed(false);
			preview_window->hide();
			preview_showed = false;
		}
	}
}

void VisualShaderEditor::_bind_methods() {
	ClassDB::bind_method("_update_nodes", &VisualShaderEditor::_update_nodes);
	ClassDB::bind_method("_update_graph", &VisualShaderEditor::_update_graph);
	ClassDB::bind_method("_add_node", &VisualShaderEditor::_add_node);
	ClassDB::bind_method("_node_changed", &VisualShaderEditor::_node_changed);
	ClassDB::bind_method("_input_select_item", &VisualShaderEditor::_input_select_item);
	ClassDB::bind_method("_parameter_ref_select_item", &VisualShaderEditor::_parameter_ref_select_item);
	ClassDB::bind_method("_varying_select_item", &VisualShaderEditor::_varying_select_item);
	ClassDB::bind_method("_set_node_size", &VisualShaderEditor::_set_node_size);
	ClassDB::bind_method("_clear_copy_buffer", &VisualShaderEditor::_clear_copy_buffer);
	ClassDB::bind_method("_update_parameters", &VisualShaderEditor::_update_parameters);
	ClassDB::bind_method("_update_varyings", &VisualShaderEditor::_update_varyings);
	ClassDB::bind_method("_update_varying_tree", &VisualShaderEditor::_update_varying_tree);
	ClassDB::bind_method("_set_mode", &VisualShaderEditor::_set_mode);
	ClassDB::bind_method("_nodes_dragged", &VisualShaderEditor::_nodes_dragged);
	ClassDB::bind_method("_float_constant_selected", &VisualShaderEditor::_float_constant_selected);
	ClassDB::bind_method("_update_constant", &VisualShaderEditor::_update_constant);
	ClassDB::bind_method("_update_parameter", &VisualShaderEditor::_update_parameter);
	ClassDB::bind_method("_expand_output_port", &VisualShaderEditor::_expand_output_port);
	ClassDB::bind_method("_update_options_menu_deferred", &VisualShaderEditor::_update_options_menu_deferred);
	ClassDB::bind_method("_rebuild_shader_deferred", &VisualShaderEditor::_rebuild_shader_deferred);
	ClassDB::bind_method("_resources_removed", &VisualShaderEditor::_resources_removed);
	ClassDB::bind_method("_update_next_previews", &VisualShaderEditor::_update_next_previews);

	ClassDB::bind_method("_is_available", &VisualShaderEditor::_is_available);
}

VisualShaderEditor::VisualShaderEditor() {
	ShaderLanguage::get_keyword_list(&keyword_list);
	EditorNode::get_singleton()->connect("resource_saved", callable_mp(this, &VisualShaderEditor::_resource_saved));
	FileSystemDock::get_singleton()->get_script_create_dialog()->connect("script_created", callable_mp(this, &VisualShaderEditor::_script_created));
	FileSystemDock::get_singleton()->connect("resource_removed", callable_mp(this, &VisualShaderEditor::_resource_removed));

	graph = memnew(GraphEdit);
	graph->get_menu_hbox()->set_h_size_flags(SIZE_EXPAND_FILL);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->set_h_size_flags(SIZE_EXPAND_FILL);
	graph->set_show_zoom_label(true);
	add_child(graph);
	SET_DRAG_FORWARDING_GCD(graph, VisualShaderEditor);
	float graph_minimap_opacity = EDITOR_GET("editors/visual_editors/minimap_opacity");
	graph->set_minimap_opacity(graph_minimap_opacity);
	float graph_lines_curvature = EDITOR_GET("editors/visual_editors/lines_curvature");
	graph->set_connection_lines_curvature(graph_lines_curvature);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_BOOLEAN);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_TRANSFORM);
	graph->add_valid_right_disconnect_type(VisualShaderNode::PORT_TYPE_SAMPLER);
	//graph->add_valid_left_disconnect_type(0);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->connect("connection_request", callable_mp(this, &VisualShaderEditor::_connection_request), CONNECT_DEFERRED);
	graph->connect("disconnection_request", callable_mp(this, &VisualShaderEditor::_disconnection_request), CONNECT_DEFERRED);
	graph->connect("node_selected", callable_mp(this, &VisualShaderEditor::_node_selected));
	graph->connect("scroll_offset_changed", callable_mp(this, &VisualShaderEditor::_scroll_changed));
	graph->connect("duplicate_nodes_request", callable_mp(this, &VisualShaderEditor::_duplicate_nodes));
	graph->connect("copy_nodes_request", callable_mp(this, &VisualShaderEditor::_copy_nodes).bind(false));
	graph->connect("paste_nodes_request", callable_mp(this, &VisualShaderEditor::_paste_nodes).bind(false, Point2()));
	graph->connect("delete_nodes_request", callable_mp(this, &VisualShaderEditor::_delete_nodes_request));
	graph->connect("gui_input", callable_mp(this, &VisualShaderEditor::_graph_gui_input));
	graph->connect("connection_to_empty", callable_mp(this, &VisualShaderEditor::_connection_to_empty));
	graph->connect("connection_from_empty", callable_mp(this, &VisualShaderEditor::_connection_from_empty));
	graph->connect("visibility_changed", callable_mp(this, &VisualShaderEditor::_visibility_changed));
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_INT, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SCALAR_UINT, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_2D, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_3D, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_VECTOR_4D, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_SCALAR);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_SCALAR_INT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_SCALAR_UINT);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_VECTOR_2D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_VECTOR_3D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_VECTOR_4D);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_BOOLEAN, VisualShaderNode::PORT_TYPE_BOOLEAN);

	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_TRANSFORM, VisualShaderNode::PORT_TYPE_TRANSFORM);
	graph->add_valid_connection_type(VisualShaderNode::PORT_TYPE_SAMPLER, VisualShaderNode::PORT_TYPE_SAMPLER);

	VSeparator *vs = memnew(VSeparator);
	graph->get_menu_hbox()->add_child(vs);
	graph->get_menu_hbox()->move_child(vs, 0);

	custom_mode_box = memnew(CheckBox);
	custom_mode_box->set_text(TTR("Custom"));
	custom_mode_box->set_pressed(false);
	custom_mode_box->set_visible(false);
	custom_mode_box->connect("toggled", callable_mp(this, &VisualShaderEditor::_custom_mode_toggled));

	edit_type_standard = memnew(OptionButton);
	edit_type_standard->add_item(TTR("Vertex"));
	edit_type_standard->add_item(TTR("Fragment"));
	edit_type_standard->add_item(TTR("Light"));
	edit_type_standard->select(1);
	edit_type_standard->connect("item_selected", callable_mp(this, &VisualShaderEditor::_mode_selected));

	edit_type_particles = memnew(OptionButton);
	edit_type_particles->add_item(TTR("Start"));
	edit_type_particles->add_item(TTR("Process"));
	edit_type_particles->add_item(TTR("Collide"));
	edit_type_particles->select(0);
	edit_type_particles->connect("item_selected", callable_mp(this, &VisualShaderEditor::_mode_selected));

	edit_type_sky = memnew(OptionButton);
	edit_type_sky->add_item(TTR("Sky"));
	edit_type_sky->select(0);
	edit_type_sky->connect("item_selected", callable_mp(this, &VisualShaderEditor::_mode_selected));

	edit_type_fog = memnew(OptionButton);
	edit_type_fog->add_item(TTR("Fog"));
	edit_type_fog->select(0);
	edit_type_fog->connect("item_selected", callable_mp(this, &VisualShaderEditor::_mode_selected));

	edit_type = edit_type_standard;

	graph->get_menu_hbox()->add_child(custom_mode_box);
	graph->get_menu_hbox()->move_child(custom_mode_box, 0);
	graph->get_menu_hbox()->add_child(edit_type_standard);
	graph->get_menu_hbox()->move_child(edit_type_standard, 0);
	graph->get_menu_hbox()->add_child(edit_type_particles);
	graph->get_menu_hbox()->move_child(edit_type_particles, 0);
	graph->get_menu_hbox()->add_child(edit_type_sky);
	graph->get_menu_hbox()->move_child(edit_type_sky, 0);
	graph->get_menu_hbox()->add_child(edit_type_fog);
	graph->get_menu_hbox()->move_child(edit_type_fog, 0);

	add_node = memnew(Button);
	add_node->set_flat(true);
	add_node->set_text(TTR("Add Node..."));
	graph->get_menu_hbox()->add_child(add_node);
	graph->get_menu_hbox()->move_child(add_node, 0);
	add_node->connect("pressed", callable_mp(this, &VisualShaderEditor::_show_members_dialog).bind(false, VisualShaderNode::PORT_TYPE_MAX, VisualShaderNode::PORT_TYPE_MAX));

	varying_button = memnew(MenuButton);
	varying_button->set_text(TTR("Manage Varyings"));
	varying_button->set_switch_on_hover(true);
	graph->get_menu_hbox()->add_child(varying_button);

	PopupMenu *varying_menu = varying_button->get_popup();
	varying_menu->add_item(TTR("Add Varying"), int(VaryingMenuOptions::ADD));
	varying_menu->add_item(TTR("Remove Varying"), int(VaryingMenuOptions::REMOVE));
	varying_menu->connect("id_pressed", callable_mp(this, &VisualShaderEditor::_varying_menu_id_pressed));

	preview_shader = memnew(Button);
	preview_shader->set_theme_type_variation("FlatButton");
	preview_shader->set_toggle_mode(true);
	preview_shader->set_tooltip_text(TTR("Show generated shader code."));
	graph->get_menu_hbox()->add_child(preview_shader);
	preview_shader->connect("pressed", callable_mp(this, &VisualShaderEditor::_show_preview_text));

	///////////////////////////////////////
	// PREVIEW WINDOW
	///////////////////////////////////////

	preview_window = memnew(Window);
	preview_window->set_title(TTR("Generated Shader Code"));
	preview_window->set_visible(preview_showed);
	preview_window->set_exclusive(true);
	preview_window->connect("close_requested", callable_mp(this, &VisualShaderEditor::_preview_close_requested));
	preview_window->connect("size_changed", callable_mp(this, &VisualShaderEditor::_preview_size_changed));
	add_child(preview_window);

	preview_vbox = memnew(VBoxContainer);
	preview_window->add_child(preview_vbox);
	preview_vbox->add_theme_constant_override("separation", 0);

	preview_text = memnew(CodeEdit);
	syntax_highlighter.instantiate();
	preview_vbox->add_child(preview_text);
	preview_text->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	preview_text->set_syntax_highlighter(syntax_highlighter);
	preview_text->set_draw_line_numbers(true);
	preview_text->set_editable(false);

	error_panel = memnew(PanelContainer);
	preview_vbox->add_child(error_panel);
	error_panel->set_visible(false);

	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);

	///////////////////////////////////////
	// POPUP MENU
	///////////////////////////////////////

	popup_menu = memnew(PopupMenu);
	add_child(popup_menu);
	popup_menu->add_item(TTR("Add Node"), NodeMenuOptions::ADD);
	popup_menu->add_separator();
	popup_menu->add_item(TTR("Cut"), NodeMenuOptions::CUT);
	popup_menu->add_item(TTR("Copy"), NodeMenuOptions::COPY);
	popup_menu->add_item(TTR("Paste"), NodeMenuOptions::PASTE);
	popup_menu->add_item(TTR("Delete"), NodeMenuOptions::DELETE);
	popup_menu->add_item(TTR("Duplicate"), NodeMenuOptions::DUPLICATE);
	popup_menu->add_item(TTR("Clear Copy Buffer"), NodeMenuOptions::CLEAR_COPY_BUFFER);
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
	tools->set_tooltip_text(TTR("Options"));
	tools->get_popup()->connect("id_pressed", callable_mp(this, &VisualShaderEditor::_tools_menu_option));
	tools->get_popup()->add_item(TTR("Expand All"), EXPAND_ALL);
	tools->get_popup()->add_item(TTR("Collapse All"), COLLAPSE_ALL);

	members = memnew(Tree);
	members_vb->add_child(members);
	SET_DRAG_FORWARDING_GCD(members, VisualShaderEditor);
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
	highend_label->set_tooltip_text(TTR("High-end node"));

	node_desc = memnew(RichTextLabel);
	members_vb->add_child(node_desc);
	node_desc->set_h_size_flags(SIZE_EXPAND_FILL);
	node_desc->set_v_size_flags(SIZE_FILL);
	node_desc->set_custom_minimum_size(Size2(0, 70 * EDSCALE));

	members_dialog = memnew(ConfirmationDialog);
	members_dialog->set_title(TTR("Create Shader Node"));
	members_dialog->set_exclusive(true);
	members_dialog->add_child(members_vb);
	members_dialog->set_ok_button_text(TTR("Create"));
	members_dialog->get_ok_button()->connect("pressed", callable_mp(this, &VisualShaderEditor::_member_create));
	members_dialog->get_ok_button()->set_disabled(true);
	members_dialog->connect("canceled", callable_mp(this, &VisualShaderEditor::_member_cancel));
	add_child(members_dialog);

	// add varyings dialog
	{
		add_varying_dialog = memnew(ConfirmationDialog);
		add_varying_dialog->set_title(TTR("Create Shader Varying"));
		add_varying_dialog->set_exclusive(true);
		add_varying_dialog->set_ok_button_text(TTR("Create"));
		add_varying_dialog->get_ok_button()->connect("pressed", callable_mp(this, &VisualShaderEditor::_varying_create));
		add_varying_dialog->get_ok_button()->set_disabled(true);
		add_child(add_varying_dialog);

		VBoxContainer *vb = memnew(VBoxContainer);
		add_varying_dialog->add_child(vb);

		HBoxContainer *hb = memnew(HBoxContainer);
		vb->add_child(hb);
		hb->set_h_size_flags(SIZE_EXPAND_FILL);

		varying_type = memnew(OptionButton);
		hb->add_child(varying_type);
		varying_type->add_item("Float");
		varying_type->add_item("Int");
		varying_type->add_item("UInt");
		varying_type->add_item("Vector2");
		varying_type->add_item("Vector3");
		varying_type->add_item("Vector4");
		varying_type->add_item("Boolean");
		varying_type->add_item("Transform");

		varying_name = memnew(LineEdit);
		hb->add_child(varying_name);
		varying_name->set_custom_minimum_size(Size2(150 * EDSCALE, 0));
		varying_name->set_h_size_flags(SIZE_EXPAND_FILL);
		varying_name->connect("text_changed", callable_mp(this, &VisualShaderEditor::_varying_name_changed));

		varying_mode = memnew(OptionButton);
		hb->add_child(varying_mode);
		varying_mode->add_item("Vertex -> [Fragment, Light]");
		varying_mode->add_item("Fragment -> Light");

		varying_error_label = memnew(Label);
		vb->add_child(varying_error_label);
		varying_error_label->set_h_size_flags(SIZE_EXPAND_FILL);

		varying_error_label->hide();
	}

	// remove varying dialog
	{
		remove_varying_dialog = memnew(ConfirmationDialog);
		remove_varying_dialog->set_title(TTR("Delete Shader Varying"));
		remove_varying_dialog->set_exclusive(true);
		remove_varying_dialog->set_ok_button_text(TTR("Delete"));
		remove_varying_dialog->get_ok_button()->connect("pressed", callable_mp(this, &VisualShaderEditor::_varying_deleted));
		add_child(remove_varying_dialog);

		VBoxContainer *vb = memnew(VBoxContainer);
		remove_varying_dialog->add_child(vb);

		varyings = memnew(Tree);
		vb->add_child(varyings);
		varyings->set_h_size_flags(SIZE_EXPAND_FILL);
		varyings->set_v_size_flags(SIZE_EXPAND_FILL);
		varyings->set_hide_root(true);
		varyings->set_allow_reselect(true);
		varyings->set_hide_folding(false);
		varyings->set_custom_minimum_size(Size2(180 * EDSCALE, 200 * EDSCALE));
		varyings->connect("item_activated", callable_mp(this, &VisualShaderEditor::_varying_deleted));
		varyings->connect("item_selected", callable_mp(this, &VisualShaderEditor::_varying_selected));
		varyings->connect("nothing_selected", callable_mp(this, &VisualShaderEditor::_varying_unselected));
	}

	alert = memnew(AcceptDialog);
	alert->get_label()->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	alert->get_label()->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	alert->get_label()->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(400, 60) * EDSCALE);
	add_child(alert);

	comment_title_change_popup = memnew(PopupPanel);
	comment_title_change_edit = memnew(LineEdit);
	comment_title_change_edit->set_expand_to_text_length_enabled(true);
	comment_title_change_edit->connect("text_changed", callable_mp(this, &VisualShaderEditor::_comment_title_text_changed));
	comment_title_change_edit->connect("text_submitted", callable_mp(this, &VisualShaderEditor::_comment_title_text_submitted));
	comment_title_change_popup->add_child(comment_title_change_edit);
	comment_title_change_edit->reset_size();
	comment_title_change_popup->reset_size();
	comment_title_change_popup->connect("focus_exited", callable_mp(this, &VisualShaderEditor::_comment_title_popup_focus_out));
	comment_title_change_popup->connect("popup_hide", callable_mp(this, &VisualShaderEditor::_comment_title_popup_hide));
	add_child(comment_title_change_popup);

	comment_desc_change_popup = memnew(PopupPanel);
	VBoxContainer *comment_desc_vbox = memnew(VBoxContainer);
	comment_desc_change_popup->add_child(comment_desc_vbox);
	comment_desc_change_edit = memnew(TextEdit);
	comment_desc_change_edit->connect("text_changed", callable_mp(this, &VisualShaderEditor::_comment_desc_text_changed));
	comment_desc_vbox->add_child(comment_desc_change_edit);
	comment_desc_change_edit->set_custom_minimum_size(Size2(300 * EDSCALE, 150 * EDSCALE));
	comment_desc_change_edit->reset_size();
	comment_desc_change_popup->reset_size();
	comment_desc_change_popup->connect("focus_exited", callable_mp(this, &VisualShaderEditor::_comment_desc_confirm));
	comment_desc_change_popup->connect("popup_hide", callable_mp(this, &VisualShaderEditor::_comment_desc_popup_hide));
	Button *comment_desc_confirm_button = memnew(Button);
	comment_desc_confirm_button->set_text(TTR("OK"));
	comment_desc_vbox->add_child(comment_desc_confirm_button);
	comment_desc_confirm_button->connect("pressed", callable_mp(this, &VisualShaderEditor::_comment_desc_confirm));
	add_child(comment_desc_change_popup);

	///////////////////////////////////////
	// SHADER NODES TREE OPTIONS
	///////////////////////////////////////

	// COLOR

	add_options.push_back(AddOption("ColorFunc", "Color/Common", "VisualShaderNodeColorFunc", TTR("Color function."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ColorOp", "Color/Common", "VisualShaderNodeColorOp", TTR("Color operator."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));

	add_options.push_back(AddOption("Grayscale", "Color/Functions", "VisualShaderNodeColorFunc", TTR("Grayscale function."), { VisualShaderNodeColorFunc::FUNC_GRAYSCALE }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("HSV2RGB", "Color/Functions", "VisualShaderNodeColorFunc", TTR("Converts HSV vector to RGB equivalent."), { VisualShaderNodeColorFunc::FUNC_HSV2RGB, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("RGB2HSV", "Color/Functions", "VisualShaderNodeColorFunc", TTR("Converts RGB vector to HSV equivalent."), { VisualShaderNodeColorFunc::FUNC_RGB2HSV, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Sepia", "Color/Functions", "VisualShaderNodeColorFunc", TTR("Sepia function."), { VisualShaderNodeColorFunc::FUNC_SEPIA }, VisualShaderNode::PORT_TYPE_VECTOR_3D));

	add_options.push_back(AddOption("Burn", "Color/Operators", "VisualShaderNodeColorOp", TTR("Burn operator."), { VisualShaderNodeColorOp::OP_BURN }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Darken", "Color/Operators", "VisualShaderNodeColorOp", TTR("Darken operator."), { VisualShaderNodeColorOp::OP_DARKEN }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Difference", "Color/Operators", "VisualShaderNodeColorOp", TTR("Difference operator."), { VisualShaderNodeColorOp::OP_DIFFERENCE }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Dodge", "Color/Operators", "VisualShaderNodeColorOp", TTR("Dodge operator."), { VisualShaderNodeColorOp::OP_DODGE }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("HardLight", "Color/Operators", "VisualShaderNodeColorOp", TTR("HardLight operator."), { VisualShaderNodeColorOp::OP_HARD_LIGHT }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Lighten", "Color/Operators", "VisualShaderNodeColorOp", TTR("Lighten operator."), { VisualShaderNodeColorOp::OP_LIGHTEN }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Overlay", "Color/Operators", "VisualShaderNodeColorOp", TTR("Overlay operator."), { VisualShaderNodeColorOp::OP_OVERLAY }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Screen", "Color/Operators", "VisualShaderNodeColorOp", TTR("Screen operator."), { VisualShaderNodeColorOp::OP_SCREEN }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("SoftLight", "Color/Operators", "VisualShaderNodeColorOp", TTR("SoftLight operator."), { VisualShaderNodeColorOp::OP_SOFT_LIGHT }, VisualShaderNode::PORT_TYPE_VECTOR_3D));

	add_options.push_back(AddOption("ColorConstant", "Color/Variables", "VisualShaderNodeColorConstant", TTR("Color constant."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ColorParameter", "Color/Variables", "VisualShaderNodeColorParameter", TTR("Color parameter."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));

	// COMMON

	add_options.push_back(AddOption("DerivativeFunc", "Common", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) Derivative function."), {}, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));

	// CONDITIONAL

	const String &compare_func_desc = TTR("Returns the boolean result of the %s comparison between two parameters.");

	add_options.push_back(AddOption("Equal (==)", "Conditional/Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Equal (==)")), { VisualShaderNodeCompare::FUNC_EQUAL }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("GreaterThan (>)", "Conditional/Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Greater Than (>)")), { VisualShaderNodeCompare::FUNC_GREATER_THAN }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("GreaterThanEqual (>=)", "Conditional/Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Greater Than or Equal (>=)")), { VisualShaderNodeCompare::FUNC_GREATER_THAN_EQUAL }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("If", "Conditional/Functions", "VisualShaderNodeIf", TTR("Returns an associated vector if the provided scalars are equal, greater or less."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("IsInf", "Conditional/Functions", "VisualShaderNodeIs", TTR("Returns the boolean result of the comparison between INF and a scalar parameter."), { VisualShaderNodeIs::FUNC_IS_INF }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("IsNaN", "Conditional/Functions", "VisualShaderNodeIs", TTR("Returns the boolean result of the comparison between NaN and a scalar parameter."), { VisualShaderNodeIs::FUNC_IS_NAN }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("LessThan (<)", "Conditional/Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Less Than (<)")), { VisualShaderNodeCompare::FUNC_LESS_THAN }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("LessThanEqual (<=)", "Conditional/Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Less Than or Equal (<=)")), { VisualShaderNodeCompare::FUNC_LESS_THAN_EQUAL }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("NotEqual (!=)", "Conditional/Functions", "VisualShaderNodeCompare", vformat(compare_func_desc, TTR("Not Equal (!=)")), { VisualShaderNodeCompare::FUNC_NOT_EQUAL }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("Switch (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated 3D vector if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Switch2D (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated 2D vector if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("SwitchBool (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated boolean if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_BOOLEAN }, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("SwitchFloat (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated floating-point scalar if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_FLOAT }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SwitchInt (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated integer scalar if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_INT }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("SwitchTransform (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated transform if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_TRANSFORM }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("SwitchUInt (==)", "Conditional/Functions", "VisualShaderNodeSwitch", TTR("Returns an associated unsigned integer scalar if the provided boolean value is true or false."), { VisualShaderNodeSwitch::OP_TYPE_UINT }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));

	add_options.push_back(AddOption("Compare (==)", "Conditional/Common", "VisualShaderNodeCompare", TTR("Returns the boolean result of the comparison between two parameters."), {}, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("Is", "Conditional/Common", "VisualShaderNodeIs", TTR("Returns the boolean result of the comparison between INF (or NaN) and a scalar parameter."), {}, VisualShaderNode::PORT_TYPE_BOOLEAN));

	add_options.push_back(AddOption("BooleanConstant", "Conditional/Variables", "VisualShaderNodeBooleanConstant", TTR("Boolean constant."), {}, VisualShaderNode::PORT_TYPE_BOOLEAN));
	add_options.push_back(AddOption("BooleanParameter", "Conditional/Variables", "VisualShaderNodeBooleanParameter", TTR("Boolean parameter."), {}, VisualShaderNode::PORT_TYPE_BOOLEAN));

	// INPUT

	const String translation_gdsl = "\n\n" + TTR("Translated to '%s' in Godot Shading Language.");
	const String input_param_shader_modes = TTR("'%s' input parameter for all shader modes.") + translation_gdsl;

	// NODE3D-FOR-ALL

	add_options.push_back(AddOption("Exposure", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "exposure", "EXPOSURE"), { "exposure" }, VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("InvProjectionMatrix", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "inv_projection_matrix", "INV_PROJECTION_MATRIX"), { "inv_projection_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("InvViewMatrix", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "inv_view_matrix", "INV_VIEW_MATRIX"), { "inv_view_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ModelMatrix", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "model_matrix", "MODEL_MATRIX"), { "model_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Normal", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "normal", "NORMAL"), { "normal" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("OutputIsSRGB", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "output_is_srgb", "OUTPUT_IS_SRGB"), { "output_is_srgb" }, VisualShaderNode::PORT_TYPE_BOOLEAN, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ProjectionMatrix", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "projection_matrix", "PROJECTION_MATRIX"), { "projection_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Time", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time", "TIME"), { "time" }, VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "uv", "UV"), { "uv" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("UV2", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "uv2", "UV2"), { "uv2" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewMatrix", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "view_matrix", "VIEW_MATRIX"), { "view_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewportSize", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "viewport_size", "VIEWPORT_SIZE"), { "viewport_size" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, -1, Shader::MODE_SPATIAL));

	// CANVASITEM-FOR-ALL

	add_options.push_back(AddOption("Color", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "color", "COLOR"), { "color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("TexturePixelSize", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "texture_pixel_size", "TEXTURE_PIXEL_SIZE"), { "texture_pixel_size" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Time", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time", "TIME"), { "time" }, VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("UV", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "uv", "UV"), { "uv" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, -1, Shader::MODE_CANVAS_ITEM));

	// PARTICLES-FOR-ALL

	add_options.push_back(AddOption("Active", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "active", "ACTIVE"), { "active" }, VisualShaderNode::PORT_TYPE_BOOLEAN, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("AttractorForce", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "attractor_force", "ATTRACTOR_FORCE"), { "attractor_force" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Color", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "color", "COLOR"), { "color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Custom", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "custom", "CUSTOM"), { "custom" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Delta", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "delta", "DELTA"), { "delta" }, VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("EmissionTransform", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "emission_transform", "EMISSION_TRANSFORM"), { "emission_transform" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Index", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "index", "INDEX"), { "index" }, VisualShaderNode::PORT_TYPE_SCALAR_UINT, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("LifeTime", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "lifetime", "LIFETIME"), { "lifetime" }, VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Number", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "number", "NUMBER"), { "number" }, VisualShaderNode::PORT_TYPE_SCALAR_UINT, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("RandomSeed", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "random_seed", "RANDOM_SEED"), { "random_seed" }, VisualShaderNode::PORT_TYPE_SCALAR_UINT, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Restart", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "restart", "RESTART"), { "restart" }, VisualShaderNode::PORT_TYPE_BOOLEAN, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Time", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "time", "TIME"), { "time" }, VisualShaderNode::PORT_TYPE_SCALAR, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Transform", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "transform", "TRANSFORM"), { "transform" }, VisualShaderNode::PORT_TYPE_TRANSFORM, -1, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("Velocity", "Input/All", "VisualShaderNodeInput", vformat(input_param_shader_modes, "velocity", "VELOCITY"), { "velocity" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, -1, Shader::MODE_PARTICLES));

	/////////////////

	add_options.push_back(AddOption("Input", "Input/Common", "VisualShaderNodeInput", TTR("Input parameter.")));

	const String input_param_for_vertex_and_fragment_shader_modes = TTR("'%s' input parameter for vertex and fragment shader modes.") + translation_gdsl;
	const String input_param_for_fragment_and_light_shader_modes = TTR("'%s' input parameter for fragment and light shader modes.") + translation_gdsl;
	const String input_param_for_fragment_shader_mode = TTR("'%s' input parameter for fragment shader mode.") + translation_gdsl;
	const String input_param_for_sky_shader_mode = TTR("'%s' input parameter for sky shader mode.") + translation_gdsl;
	const String input_param_for_fog_shader_mode = TTR("'%s' input parameter for fog shader mode.") + translation_gdsl;
	const String input_param_for_light_shader_mode = TTR("'%s' input parameter for light shader mode.") + translation_gdsl;
	const String input_param_for_vertex_shader_mode = TTR("'%s' input parameter for vertex shader mode.") + translation_gdsl;
	const String input_param_for_start_shader_mode = TTR("'%s' input parameter for start shader mode.") + translation_gdsl;
	const String input_param_for_process_shader_mode = TTR("'%s' input parameter for process shader mode.") + translation_gdsl;
	const String input_param_for_collide_shader_mode = TTR("'%s' input parameter for collide shader mode." + translation_gdsl);
	const String input_param_for_start_and_process_shader_mode = TTR("'%s' input parameter for start and process shader modes.") + translation_gdsl;
	const String input_param_for_process_and_collide_shader_mode = TTR("'%s' input parameter for process and collide shader modes.") + translation_gdsl;
	const String input_param_for_vertex_and_fragment_shader_mode = TTR("'%s' input parameter for vertex and fragment shader modes.") + translation_gdsl;

	// NODE3D INPUTS

	add_options.push_back(AddOption("Binormal", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "binormal", "BINORMAL"), { "binormal" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Color", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "color", "COLOR"), { "color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Custom0", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "custom0", "CUSTOM0"), { "custom0" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Custom1", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "custom1", "CUSTOM1"), { "custom1" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Custom2", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "custom2", "CUSTOM2"), { "custom2" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Custom3", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "custom3", "CUSTOM3"), { "custom3" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("InstanceId", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "instance_id", "INSTANCE_ID"), { "instance_id" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("InstanceCustom", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "instance_custom", "INSTANCE_CUSTOM"), { "instance_custom" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ModelViewMatrix", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "modelview_matrix", "MODELVIEW_MATRIX"), { "modelview_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("PointSize", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "point_size", "POINT_SIZE"), { "point_size" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Tangent", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_mode, "tangent", "TANGENT"), { "tangent" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Vertex", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "vertex", "VERTEX"), { "vertex" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("VertexId", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "vertex_id", "VERTEX_ID"), { "vertex_id" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewIndex", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "view_index", "VIEW_INDEX"), { "view_index" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewMonoLeft", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "view_mono_left", "VIEW_MONO_LEFT"), { "view_mono_left" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewRight", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "view_right", "VIEW_RIGHT"), { "view_right" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("EyeOffset", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "eye_offset", "EYE_OFFSET"), { "eye_offset" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("NodePositionWorld", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "node_position_world", "NODE_POSITION_WORLD"), { "node_position_world" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("CameraPositionWorld", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "camera_position_world", "CAMERA_POSITION_WORLD"), { "camera_position_world" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("CameraDirectionWorld", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "camera_direction_world", "CAMERA_DIRECTION_WORLD"), { "camera_direction_world" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("CameraVisibleLayers", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "camera_visible_layers", "CAMERA_VISIBLE_LAYERS"), { "camera_visible_layers" }, VisualShaderNode::PORT_TYPE_SCALAR_UINT, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("NodePositionView", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "node_position_view", "NODE_POSITION_VIEW"), { "node_position_view" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));

	add_options.push_back(AddOption("Binormal", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "binormal", "BINORMAL"), { "binormal" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Color", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "color", "COLOR"), { "color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FragCoord", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord", "FRAGCOORD"), { "fragcoord" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FrontFacing", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "front_facing", "FRONT_FACING"), { "front_facing" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("PointCoord", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "point_coord", "POINT_COORD"), { "point_coord" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ScreenUV", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_uv", "SCREEN_UV"), { "screen_uv" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Tangent", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "tangent", "TANGENT"), { "tangent" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Vertex", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "vertex", "VERTEX"), { "vertex" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("View", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "view", "VIEW"), { "view" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewIndex", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "view_index", "VIEW_INDEX"), { "view_index" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewMonoLeft", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "view_mono_left", "VIEW_MONO_LEFT"), { "view_mono_left" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ViewRight", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "view_right", "VIEW_RIGHT"), { "view_right" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("EyeOffset", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "eye_offset", "EYE_OFFSET"), { "eye_offset" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("NodePositionWorld", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "node_position_world", "NODE_POSITION_WORLD"), { "node_position_world" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("CameraPositionWorld", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "camera_position_world", "CAMERA_POSITION_WORLD"), { "camera_position_world" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("CameraDirectionWorld", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "camera_direction_world", "CAMERA_DIRECTION_WORLD"), { "camera_direction_world" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("CameraVisibleLayers", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "camera_visible_layers", "CAMERA_VISIBLE_LAYERS"), { "camera_visible_layers" }, VisualShaderNode::PORT_TYPE_SCALAR_UINT, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("NodePositionView", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "node_position_view", "NODE_POSITION_VIEW"), { "node_position_view" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));

	add_options.push_back(AddOption("Albedo", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "albedo", "ALBEDO"), { "albedo" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Attenuation", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "attenuation", "ATTENUATION"), { "attenuation" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Backlight", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "backlight", "BACKLIGHT"), { "backlight" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Diffuse", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "diffuse", "DIFFUSE_LIGHT"), { "diffuse" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("FragCoord", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord", "FRAGCOORD"), { "fragcoord" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Light", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light", "LIGHT"), { "light" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("LightColor", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_color", "LIGHT_COLOR"), { "light_color" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("LightIsDirectional", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_is_directional", "LIGHT_IS_DIRECTIONAL"), { "light_is_directional" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Metallic", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "metallic", "METALLIC"), { "metallic" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Roughness", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "roughness", "ROUGHNESS"), { "roughness" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Specular", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "specular", "SPECULAR_LIGHT"), { "specular" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("View", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "view", "VIEW"), { "view" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));

	// CANVASITEM INPUTS

	add_options.push_back(AddOption("AtLightPass", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "at_light_pass", "AT_LIGHT_PASS"), { "at_light_pass" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("CanvasMatrix", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "canvas_matrix", "CANVAS_MATRIX"), { "canvas_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("InstanceCustom", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "instance_custom", "INSTANCE_CUSTOM"), { "instance_custom" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("InstanceId", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "instance_id", "INSTANCE_ID"), { "instance_id" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ModelMatrix", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "model_matrix", "MODEL_MATRIX"), { "model_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointSize", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "point_size", "POINT_SIZE"), { "point_size" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenMatrix", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "screen_matrix", "SCREEN_MATRIX"), { "screen_matrix" }, VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Vertex", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_mode, "vertex", "VERTEX"), { "vertex" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("VertexId", "Input/Vertex", "VisualShaderNodeInput", vformat(input_param_for_vertex_shader_mode, "vertex_id", "VERTEX_ID"), { "vertex_id" }, VisualShaderNode::PORT_TYPE_SCALAR_INT, TYPE_FLAGS_VERTEX, Shader::MODE_CANVAS_ITEM));

	add_options.push_back(AddOption("AtLightPass", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_modes, "at_light_pass", "AT_LIGHT_PASS"), { "at_light_pass" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("FragCoord", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord", "FRAGCOORD"), { "fragcoord" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("NormalTexture", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "normal_texture", "NORMAL_TEXTURE"), { "normal_texture" }, VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointCoord", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "point_coord", "POINT_COORD"), { "point_coord" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenPixelSize", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "screen_pixel_size", "SCREEN_PIXEL_SIZE"), { "screen_pixel_size" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenUV", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "screen_uv", "SCREEN_UV"), { "screen_uv" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("SpecularShininess", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "specular_shininess", "SPECULAR_SHININESS"), { "specular_shininess" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("SpecularShininessTexture", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_shader_mode, "specular_shininess_texture", "SPECULAR_SHININESS_TEXTURE"), { "specular_shininess_texture" }, VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Texture", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "texture", "TEXTURE"), { "texture" }, VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Vertex", "Input/Fragment", "VisualShaderNodeInput", vformat(input_param_for_vertex_and_fragment_shader_mode, "vertex", "VERTEX"), { "vertex" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));

	add_options.push_back(AddOption("FragCoord", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "fragcoord", "FRAGCOORD"), { "fragcoord" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Light", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light", "LIGHT"), { "light" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightColor", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_color", "LIGHT_COLOR"), { "light_color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightPosition", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_position", "LIGHT_POSITION"), { "light_position" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightDirection", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_direction", "LIGHT_DIRECTION"), { "light_direction" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightIsDirectional", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_is_directional", "LIGHT_IS_DIRECTIONAL"), { "light_is_directional" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightEnergy", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "light_energy", "LIGHT_ENERGY"), { "light_energy" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("LightVertex", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "light_vertex", "LIGHT_VERTEX"), { "light_vertex" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Normal", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "normal", "NORMAL"), { "normal" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("PointCoord", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "point_coord", "POINT_COORD"), { "point_coord" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("ScreenUV", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "screen_uv", "SCREEN_UV"), { "screen_uv" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Shadow", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_light_shader_mode, "shadow", "SHADOW_MODULATE"), { "shadow" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("SpecularShininess", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "specular_shininess", "SPECULAR_SHININESS"), { "specular_shininess" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("Texture", "Input/Light", "VisualShaderNodeInput", vformat(input_param_for_fragment_and_light_shader_modes, "texture", "TEXTURE"), { "texture" }, VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));

	// SKY INPUTS

	add_options.push_back(AddOption("AtCubeMapPass", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "at_cubemap_pass", "AT_CUBEMAP_PASS"), { "at_cubemap_pass" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("AtHalfResPass", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "at_half_res_pass", "AT_HALF_RES_PASS"), { "at_half_res_pass" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("AtQuarterResPass", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "at_quarter_res_pass", "AT_QUARTER_RES_PASS"), { "at_quarter_res_pass" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("EyeDir", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "eyedir", "EYEDIR"), { "eyedir" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("HalfResColor", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "half_res_color", "HALF_RES_COLOR"), { "half_res_color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Color", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light0_color", "LIGHT0_COLOR"), { "light0_color" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Direction", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light0_direction", "LIGHT0_DIRECTION"), { "light0_direction" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Enabled", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light0_enabled", "LIGHT0_ENABLED"), { "light0_enabled" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light0Energy", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light0_energy", "LIGHT0_ENERGY"), { "light0_energy" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Color", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light1_color", "LIGHT1_COLOR"), { "light1_color" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Direction", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light1_direction", "LIGHT1_DIRECTION"), { "light1_direction" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Enabled", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light1_enabled", "LIGHT1_ENABLED"), { "light1_enabled" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light1Energy", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light1_energy", "LIGHT1_ENERGY"), { "light1_energy" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Color", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light2_color", "LIGHT2_COLOR"), { "light2_color" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Direction", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light2_direction", "LIGHT2_DIRECTION"), { "light2_direction" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Enabled", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light2_enabled", "LIGHT2_ENABLED"), { "light2_enabled" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light2Energy", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light2_energy", "LIGHT2_ENERGY"), { "light2_energy" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Color", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light3_color", "LIGHT3_COLOR"), { "light3_color" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Direction", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light3_direction", "LIGHT3_DIRECTION"), { "light3_direction" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Enabled", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light3_enabled", "LIGHT3_ENABLED"), { "light3_enabled" }, VisualShaderNode::PORT_TYPE_BOOLEAN, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Light3Energy", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "light3_energy", "LIGHT3_ENERGY"), { "light3_energy" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Position", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "position", "POSITION"), { "position" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("QuarterResColor", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "quarter_res_color", "QUARTER_RES_COLOR"), { "quarter_res_color" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Radiance", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "radiance", "RADIANCE"), { "radiance" }, VisualShaderNode::PORT_TYPE_SAMPLER, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("ScreenUV", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "screen_uv", "SCREEN_UV"), { "screen_uv" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("FragCoord", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "fragcoord", "FRAGCOORD"), { "fragcoord" }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_SKY, Shader::MODE_SKY));

	add_options.push_back(AddOption("SkyCoords", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "sky_coords", "SKY_COORDS"), { "sky_coords" }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_SKY, Shader::MODE_SKY));
	add_options.push_back(AddOption("Time", "Input/Sky", "VisualShaderNodeInput", vformat(input_param_for_sky_shader_mode, "time", "TIME"), { "time" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_SKY, Shader::MODE_SKY));

	// FOG INPUTS

	add_options.push_back(AddOption("WorldPosition", "Input/Fog", "VisualShaderNodeInput", vformat(input_param_for_fog_shader_mode, "world_position", "WORLD_POSITION"), { "world_position" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FOG, Shader::MODE_FOG));
	add_options.push_back(AddOption("ObjectPosition", "Input/Fog", "VisualShaderNodeInput", vformat(input_param_for_fog_shader_mode, "object_position", "OBJECT_POSITION"), { "object_position" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FOG, Shader::MODE_FOG));
	add_options.push_back(AddOption("UVW", "Input/Fog", "VisualShaderNodeInput", vformat(input_param_for_fog_shader_mode, "uvw", "UVW"), { "uvw" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FOG, Shader::MODE_FOG));
	add_options.push_back(AddOption("Size", "Input/Fog", "VisualShaderNodeInput", vformat(input_param_for_fog_shader_mode, "size", "SIZE"), { "size" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FOG, Shader::MODE_FOG));
	add_options.push_back(AddOption("SDF", "Input/Fog", "VisualShaderNodeInput", vformat(input_param_for_fog_shader_mode, "sdf", "SDF"), { "sdf" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FOG, Shader::MODE_FOG));
	add_options.push_back(AddOption("Time", "Input/Fog", "VisualShaderNodeInput", vformat(input_param_for_fog_shader_mode, "time", "TIME"), { "time" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FOG, Shader::MODE_FOG));

	// PARTICLES INPUTS

	add_options.push_back(AddOption("CollisionDepth", "Input/Collide", "VisualShaderNodeInput", vformat(input_param_for_collide_shader_mode, "collision_depth", "COLLISION_DEPTH"), { "collision_depth" }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_COLLIDE, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("CollisionNormal", "Input/Collide", "VisualShaderNodeInput", vformat(input_param_for_collide_shader_mode, "collision_normal", "COLLISION_NORMAL"), { "collision_normal" }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_COLLIDE, Shader::MODE_PARTICLES));

	// PARTICLES

	add_options.push_back(AddOption("EmitParticle", "Particles", "VisualShaderNodeParticleEmit", "", {}, -1, TYPE_FLAGS_PROCESS | TYPE_FLAGS_PROCESS_CUSTOM | TYPE_FLAGS_COLLIDE, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("ParticleAccelerator", "Particles", "VisualShaderNodeParticleAccelerator", "", {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_PROCESS, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("ParticleRandomness", "Particles", "VisualShaderNodeParticleRandomness", "", {}, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_EMIT | TYPE_FLAGS_PROCESS | TYPE_FLAGS_COLLIDE, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("MultiplyByAxisAngle (*)", "Particles/Transform", "VisualShaderNodeParticleMultiplyByAxisAngle", TTR("A node for help to multiply a position input vector by rotation using specific axis. Intended to work with emitters."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_EMIT | TYPE_FLAGS_PROCESS | TYPE_FLAGS_COLLIDE, Shader::MODE_PARTICLES));

	add_options.push_back(AddOption("BoxEmitter", "Particles/Emitters", "VisualShaderNodeParticleBoxEmitter", "", {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("MeshEmitter", "Particles/Emitters", "VisualShaderNodeParticleMeshEmitter", "", {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("RingEmitter", "Particles/Emitters", "VisualShaderNodeParticleRingEmitter", "", {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));
	add_options.push_back(AddOption("SphereEmitter", "Particles/Emitters", "VisualShaderNodeParticleSphereEmitter", "", {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));

	add_options.push_back(AddOption("ConeVelocity", "Particles/Velocity", "VisualShaderNodeParticleConeVelocity", "", {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_EMIT, Shader::MODE_PARTICLES));

	// SCALAR

	add_options.push_back(AddOption("FloatFunc", "Scalar/Common", "VisualShaderNodeFloatFunc", TTR("Float function."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("FloatOp", "Scalar/Common", "VisualShaderNodeFloatOp", TTR("Float operator."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntFunc", "Scalar/Common", "VisualShaderNodeIntFunc", TTR("Integer function."), {}, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("IntOp", "Scalar/Common", "VisualShaderNodeIntOp", TTR("Integer operator."), {}, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("UIntFunc", "Scalar/Common", "VisualShaderNodeUIntFunc", TTR("Unsigned integer function."), {}, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("UIntOp", "Scalar/Common", "VisualShaderNodeUIntOp", TTR("Unsigned integer operator."), {}, VisualShaderNode::PORT_TYPE_SCALAR_UINT));

	// CONSTANTS

	for (int i = 0; i < MAX_FLOAT_CONST_DEFS; i++) {
		add_options.push_back(AddOption(float_constant_defs[i].name, "Scalar/Constants", "VisualShaderNodeFloatConstant", float_constant_defs[i].desc, { float_constant_defs[i].value }, VisualShaderNode::PORT_TYPE_SCALAR));
	}
	// FUNCTIONS

	add_options.push_back(AddOption("Abs", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the absolute value of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ABS }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Abs", "Scalar/Functions", "VisualShaderNodeIntFunc", TTR("Returns the absolute value of the parameter."), { VisualShaderNodeIntFunc::FUNC_ABS }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("ACos", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the arc-cosine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ACOS }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ACosH", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ACOSH }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ASin", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the arc-sine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ASIN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ASinH", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ASINH }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATan", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the arc-tangent of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ATAN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATan2", "Scalar/Functions", "VisualShaderNodeFloatOp", TTR("Returns the arc-tangent of the parameters."), { VisualShaderNodeFloatOp::OP_ATAN2 }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("ATanH", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), { VisualShaderNodeFloatFunc::FUNC_ATANH }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("BitwiseNOT", "Scalar/Functions", "VisualShaderNodeIntFunc", TTR("Returns the result of bitwise NOT (~a) operation on the integer."), { VisualShaderNodeIntFunc::FUNC_BITWISE_NOT }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("BitwiseNOT", "Scalar/Functions", "VisualShaderNodeUIntFunc", TTR("Returns the result of bitwise NOT (~a) operation on the unsigned integer."), { VisualShaderNodeUIntFunc::FUNC_BITWISE_NOT }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("Ceil", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), { VisualShaderNodeFloatFunc::FUNC_CEIL }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Clamp", "Scalar/Functions", "VisualShaderNodeClamp", TTR("Constrains a value to lie between two further values."), { VisualShaderNodeClamp::OP_TYPE_FLOAT }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Clamp", "Scalar/Functions", "VisualShaderNodeClamp", TTR("Constrains a value to lie between two further values."), { VisualShaderNodeClamp::OP_TYPE_INT }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Clamp", "Scalar/Functions", "VisualShaderNodeClamp", TTR("Constrains a value to lie between two further values."), { VisualShaderNodeClamp::OP_TYPE_UINT }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("Cos", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the cosine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_COS }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("CosH", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the hyperbolic cosine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_COSH }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Degrees", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Converts a quantity in radians to degrees."), { VisualShaderNodeFloatFunc::FUNC_DEGREES }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("DFdX", "Scalar/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Derivative in 'x' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_X, VisualShaderNodeDerivativeFunc::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("DFdY", "Scalar/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Derivative in 'y' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_Y, VisualShaderNodeDerivativeFunc::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("Exp", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Base-e Exponential."), { VisualShaderNodeFloatFunc::FUNC_EXP }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp2", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Base-2 Exponential."), { VisualShaderNodeFloatFunc::FUNC_EXP2 }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Floor", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest integer less than or equal to the parameter."), { VisualShaderNodeFloatFunc::FUNC_FLOOR }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Fract", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Computes the fractional part of the argument."), { VisualShaderNodeFloatFunc::FUNC_FRACT }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("InverseSqrt", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the inverse of the square root of the parameter."), { VisualShaderNodeFloatFunc::FUNC_INVERSE_SQRT }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Natural logarithm."), { VisualShaderNodeFloatFunc::FUNC_LOG }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log2", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Base-2 logarithm."), { VisualShaderNodeFloatFunc::FUNC_LOG2 }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Max", "Scalar/Functions", "VisualShaderNodeFloatOp", TTR("Returns the greater of two values."), { VisualShaderNodeFloatOp::OP_MAX }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Min", "Scalar/Functions", "VisualShaderNodeFloatOp", TTR("Returns the lesser of two values."), { VisualShaderNodeFloatOp::OP_MIN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Mix", "Scalar/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two scalars."), { VisualShaderNodeMix::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("MultiplyAdd (a * b + c)", "Scalar/Functions", "VisualShaderNodeMultiplyAdd", TTR("Performs a fused multiply-add operation (a * b + c) on scalars."), { VisualShaderNodeMultiplyAdd::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Negate (*-1)", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the opposite value of the parameter."), { VisualShaderNodeFloatFunc::FUNC_NEGATE }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Negate (*-1)", "Scalar/Functions", "VisualShaderNodeIntFunc", TTR("Returns the opposite value of the parameter."), { VisualShaderNodeIntFunc::FUNC_NEGATE }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Negate (*-1)", "Scalar/Functions", "VisualShaderNodeUIntFunc", TTR("Returns the opposite value of the parameter."), { VisualShaderNodeUIntFunc::FUNC_NEGATE }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("OneMinus (1-)", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("1.0 - scalar"), { VisualShaderNodeFloatFunc::FUNC_ONEMINUS }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Pow", "Scalar/Functions", "VisualShaderNodeFloatOp", TTR("Returns the value of the first parameter raised to the power of the second."), { VisualShaderNodeFloatOp::OP_POW }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Radians", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Converts a quantity in degrees to radians."), { VisualShaderNodeFloatFunc::FUNC_RADIANS }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Reciprocal", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("1.0 / scalar"), { VisualShaderNodeFloatFunc::FUNC_RECIPROCAL }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Round", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest integer to the parameter."), { VisualShaderNodeFloatFunc::FUNC_ROUND }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("RoundEven", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Finds the nearest even integer to the parameter."), { VisualShaderNodeFloatFunc::FUNC_ROUNDEVEN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Saturate", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Clamps the value between 0.0 and 1.0."), { VisualShaderNodeFloatFunc::FUNC_SATURATE }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sign", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Extracts the sign of the parameter."), { VisualShaderNodeFloatFunc::FUNC_SIGN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sign", "Scalar/Functions", "VisualShaderNodeIntFunc", TTR("Extracts the sign of the parameter."), { VisualShaderNodeIntFunc::FUNC_SIGN }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Sin", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the sine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_SIN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SinH", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the hyperbolic sine of the parameter."), { VisualShaderNodeFloatFunc::FUNC_SINH }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sqrt", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the square root of the parameter."), { VisualShaderNodeFloatFunc::FUNC_SQRT }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("SmoothStep", "Scalar/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), scalar(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if x is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Step", "Scalar/Functions", "VisualShaderNodeStep", TTR("Step function( scalar(edge), scalar(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), { VisualShaderNodeStep::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Sum", "Scalar/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Scalar) Sum of absolute derivative in 'x' and 'y'."), { VisualShaderNodeDerivativeFunc::FUNC_SUM, VisualShaderNodeDerivativeFunc::OP_TYPE_SCALAR }, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("Tan", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the tangent of the parameter."), { VisualShaderNodeFloatFunc::FUNC_TAN }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("TanH", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Returns the hyperbolic tangent of the parameter."), { VisualShaderNodeFloatFunc::FUNC_TANH }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Trunc", "Scalar/Functions", "VisualShaderNodeFloatFunc", TTR("Finds the truncated value of the parameter."), { VisualShaderNodeFloatFunc::FUNC_TRUNC }, VisualShaderNode::PORT_TYPE_SCALAR));

	add_options.push_back(AddOption("Add (+)", "Scalar/Operators", "VisualShaderNodeFloatOp", TTR("Sums two floating-point scalars."), { VisualShaderNodeFloatOp::OP_ADD }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Add (+)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Sums two integer scalars."), { VisualShaderNodeIntOp::OP_ADD }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Add (+)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Sums two unsigned integer scalars."), { VisualShaderNodeUIntOp::OP_ADD }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("BitwiseAND (&)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Returns the result of bitwise AND (a & b) operation for two integers."), { VisualShaderNodeIntOp::OP_BITWISE_AND }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("BitwiseAND (&)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Returns the result of bitwise AND (a & b) operation for two unsigned integers."), { VisualShaderNodeUIntOp::OP_BITWISE_AND }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("BitwiseLeftShift (<<)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Returns the result of bitwise left shift (a << b) operation on the integer."), { VisualShaderNodeIntOp::OP_BITWISE_LEFT_SHIFT }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("BitwiseLeftShift (<<)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Returns the result of bitwise left shift (a << b) operation on the unsigned integer."), { VisualShaderNodeUIntOp::OP_BITWISE_LEFT_SHIFT }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("BitwiseOR (|)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Returns the result of bitwise OR (a | b) operation for two integers."), { VisualShaderNodeIntOp::OP_BITWISE_OR }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("BitwiseOR (|)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Returns the result of bitwise OR (a | b) operation for two unsigned integers."), { VisualShaderNodeUIntOp::OP_BITWISE_OR }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("BitwiseRightShift (>>)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Returns the result of bitwise right shift (a >> b) operation on the integer."), { VisualShaderNodeIntOp::OP_BITWISE_RIGHT_SHIFT }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("BitwiseRightShift (>>)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Returns the result of bitwise right shift (a >> b) operation on the unsigned integer."), { VisualShaderNodeIntOp::OP_BITWISE_RIGHT_SHIFT }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("BitwiseXOR (^)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Returns the result of bitwise XOR (a ^ b) operation on the integer."), { VisualShaderNodeIntOp::OP_BITWISE_XOR }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("BitwiseXOR (^)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Returns the result of bitwise XOR (a ^ b) operation on the unsigned integer."), { VisualShaderNodeUIntOp::OP_BITWISE_XOR }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("Divide (/)", "Scalar/Operators", "VisualShaderNodeFloatOp", TTR("Divides two floating-point scalars."), { VisualShaderNodeFloatOp::OP_DIV }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Divide (/)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Divides two integer scalars."), { VisualShaderNodeIntOp::OP_DIV }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Divide (/)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Divides two unsigned integer scalars."), { VisualShaderNodeUIntOp::OP_DIV }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("Multiply (*)", "Scalar/Operators", "VisualShaderNodeFloatOp", TTR("Multiplies two floating-point scalars."), { VisualShaderNodeFloatOp::OP_MUL }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Multiply (*)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Multiplies two integer scalars."), { VisualShaderNodeIntOp::OP_MUL }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Multiply (*)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Multiplies two unsigned integer scalars."), { VisualShaderNodeUIntOp::OP_MUL }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("Remainder", "Scalar/Operators", "VisualShaderNodeFloatOp", TTR("Returns the remainder of the two floating-point scalars."), { VisualShaderNodeFloatOp::OP_MOD }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Remainder", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Returns the remainder of the two integer scalars."), { VisualShaderNodeIntOp::OP_MOD }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Remainder", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Returns the remainder of the two unsigned integer scalars."), { VisualShaderNodeUIntOp::OP_MOD }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("Subtract (-)", "Scalar/Operators", "VisualShaderNodeFloatOp", TTR("Subtracts two floating-point scalars."), { VisualShaderNodeFloatOp::OP_SUB }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Subtract (-)", "Scalar/Operators", "VisualShaderNodeIntOp", TTR("Subtracts two integer scalars."), { VisualShaderNodeIntOp::OP_SUB }, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("Subtract (-)", "Scalar/Operators", "VisualShaderNodeUIntOp", TTR("Subtracts two unsigned integer scalars."), { VisualShaderNodeUIntOp::OP_SUB }, VisualShaderNode::PORT_TYPE_SCALAR_UINT));

	add_options.push_back(AddOption("FloatConstant", "Scalar/Variables", "VisualShaderNodeFloatConstant", TTR("Scalar floating-point constant."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntConstant", "Scalar/Variables", "VisualShaderNodeIntConstant", TTR("Scalar integer constant."), {}, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("UIntConstant", "Scalar/Variables", "VisualShaderNodeUIntConstant", TTR("Scalar unsigned integer constant."), {}, VisualShaderNode::PORT_TYPE_SCALAR_UINT));
	add_options.push_back(AddOption("FloatParameter", "Scalar/Variables", "VisualShaderNodeFloatParameter", TTR("Scalar floating-point parameter."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("IntParameter", "Scalar/Variables", "VisualShaderNodeIntParameter", TTR("Scalar integer parameter."), {}, VisualShaderNode::PORT_TYPE_SCALAR_INT));
	add_options.push_back(AddOption("UIntParameter", "Scalar/Variables", "VisualShaderNodeUIntParameter", TTR("Scalar unsigned integer parameter."), {}, VisualShaderNode::PORT_TYPE_SCALAR_UINT));

	// SDF
	{
		add_options.push_back(AddOption("ScreenUVToSDF", "SDF", "VisualShaderNodeScreenUVToSDF", TTR("Converts screen UV to a SDF."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
		add_options.push_back(AddOption("SDFRaymarch", "SDF", "VisualShaderNodeSDFRaymarch", TTR("Casts a ray against the screen SDF and returns the distance travelled."), {}, -1, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
		add_options.push_back(AddOption("SDFToScreenUV", "SDF", "VisualShaderNodeSDFToScreenUV", TTR("Converts a SDF to screen UV."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
		add_options.push_back(AddOption("TextureSDF", "SDF", "VisualShaderNodeTextureSDF", TTR("Performs a SDF texture lookup."), {}, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
		add_options.push_back(AddOption("TextureSDFNormal", "SDF", "VisualShaderNodeTextureSDFNormal", TTR("Performs a SDF normal texture lookup."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	}

	// TEXTURES

	add_options.push_back(AddOption("UVFunc", "Textures/Common", "VisualShaderNodeUVFunc", TTR("Function to be applied on texture coordinates."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("UVPolarCoord", "Textures/Common", "VisualShaderNodeUVPolarCoord", TTR("Polar coordinates conversion applied on texture coordinates."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D));

	cubemap_node_option_idx = add_options.size();
	add_options.push_back(AddOption("CubeMap", "Textures/Functions", "VisualShaderNodeCubemap", TTR("Perform the cubic texture lookup."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	curve_node_option_idx = add_options.size();
	add_options.push_back(AddOption("CurveTexture", "Textures/Functions", "VisualShaderNodeCurveTexture", TTR("Perform the curve texture lookup."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	curve_xyz_node_option_idx = add_options.size();
	add_options.push_back(AddOption("CurveXYZTexture", "Textures/Functions", "VisualShaderNodeCurveXYZTexture", TTR("Perform the three components curve texture lookup."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("LinearSceneDepth", "Textures/Functions", "VisualShaderNodeLinearSceneDepth", TTR("Returns the depth value obtained from the depth prepass in a linear space."), {}, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	texture2d_node_option_idx = add_options.size();
	add_options.push_back(AddOption("WorldPositionFromDepth", "Textures/Functions", "VisualShaderNodeWorldPositionFromDepth", TTR("Reconstructs the World Position of the Node from the depth texture."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	texture2d_node_option_idx = add_options.size();
	add_options.push_back(AddOption("ScreenNormalWorldSpace", "Textures/Functions", "VisualShaderNodeScreenNormalWorldSpace", TTR("Unpacks the Screen Normal Texture in World Space"), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	texture2d_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture2D", "Textures/Functions", "VisualShaderNodeTexture", TTR("Perform the 2D texture lookup."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	texture2d_array_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture2DArray", "Textures/Functions", "VisualShaderNodeTexture2DArray", TTR("Perform the 2D-array texture lookup."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	texture3d_node_option_idx = add_options.size();
	add_options.push_back(AddOption("Texture3D", "Textures/Functions", "VisualShaderNodeTexture3D", TTR("Perform the 3D texture lookup."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("UVPanning", "Textures/Functions", "VisualShaderNodeUVFunc", TTR("Apply panning function on texture coordinates."), { VisualShaderNodeUVFunc::FUNC_PANNING }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("UVScaling", "Textures/Functions", "VisualShaderNodeUVFunc", TTR("Apply scaling function on texture coordinates."), { VisualShaderNodeUVFunc::FUNC_SCALING }, VisualShaderNode::PORT_TYPE_VECTOR_2D));

	add_options.push_back(AddOption("CubeMapParameter", "Textures/Variables", "VisualShaderNodeCubemapParameter", TTR("Cubic texture parameter lookup."), {}, VisualShaderNode::PORT_TYPE_SAMPLER));
	add_options.push_back(AddOption("Texture2DParameter", "Textures/Variables", "VisualShaderNodeTexture2DParameter", TTR("2D texture parameter lookup."), {}, VisualShaderNode::PORT_TYPE_SAMPLER));
	add_options.push_back(AddOption("TextureParameterTriplanar", "Textures/Variables", "VisualShaderNodeTextureParameterTriplanar", TTR("2D texture parameter lookup with triplanar."), {}, -1, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Texture2DArrayParameter", "Textures/Variables", "VisualShaderNodeTexture2DArrayParameter", TTR("2D array of textures parameter lookup."), {}, VisualShaderNode::PORT_TYPE_SAMPLER));
	add_options.push_back(AddOption("Texture3DParameter", "Textures/Variables", "VisualShaderNodeTexture3DParameter", TTR("3D texture parameter lookup."), {}, VisualShaderNode::PORT_TYPE_SAMPLER));

	// TRANSFORM

	add_options.push_back(AddOption("TransformFunc", "Transform/Common", "VisualShaderNodeTransformFunc", TTR("Transform function."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformOp", "Transform/Common", "VisualShaderNodeTransformOp", TTR("Transform operator."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM));

	add_options.push_back(AddOption("OuterProduct", "Transform/Composition", "VisualShaderNodeOuterProduct", TTR("Calculate the outer product of a pair of vectors.\n\nOuterProduct treats the first parameter 'c' as a column vector (matrix with one column) and the second parameter 'r' as a row vector (matrix with one row) and does a linear algebraic matrix multiply 'c * r', yielding a matrix whose number of rows is the number of components in 'c' and whose number of columns is the number of components in 'r'."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformCompose", "Transform/Composition", "VisualShaderNodeTransformCompose", TTR("Composes transform from four vectors."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformDecompose", "Transform/Composition", "VisualShaderNodeTransformDecompose", TTR("Decomposes transform to four vectors.")));

	add_options.push_back(AddOption("Determinant", "Transform/Functions", "VisualShaderNodeDeterminant", TTR("Calculates the determinant of a transform."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("GetBillboardMatrix", "Transform/Functions", "VisualShaderNodeBillboard", TTR("Calculates how the object should face the camera to be applied on Model View Matrix output port for 3D objects."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("Inverse", "Transform/Functions", "VisualShaderNodeTransformFunc", TTR("Calculates the inverse of a transform."), { VisualShaderNodeTransformFunc::FUNC_INVERSE }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("Transpose", "Transform/Functions", "VisualShaderNodeTransformFunc", TTR("Calculates the transpose of a transform."), { VisualShaderNodeTransformFunc::FUNC_TRANSPOSE }, VisualShaderNode::PORT_TYPE_TRANSFORM));

	add_options.push_back(AddOption("Add (+)", "Transform/Operators", "VisualShaderNodeTransformOp", TTR("Sums two transforms."), { VisualShaderNodeTransformOp::OP_ADD }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("Divide (/)", "Transform/Operators", "VisualShaderNodeTransformOp", TTR("Divides two transforms."), { VisualShaderNodeTransformOp::OP_A_DIV_B }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("Multiply (*)", "Transform/Operators", "VisualShaderNodeTransformOp", TTR("Multiplies two transforms."), { VisualShaderNodeTransformOp::OP_AxB }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("MultiplyComp (*)", "Transform/Operators", "VisualShaderNodeTransformOp", TTR("Performs per-component multiplication of two transforms."), { VisualShaderNodeTransformOp::OP_AxB_COMP }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("Subtract (-)", "Transform/Operators", "VisualShaderNodeTransformOp", TTR("Subtracts two transforms."), { VisualShaderNodeTransformOp::OP_A_MINUS_B }, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformVectorMult (*)", "Transform/Operators", "VisualShaderNodeTransformVecMult", TTR("Multiplies vector by transform."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));

	add_options.push_back(AddOption("TransformConstant", "Transform/Variables", "VisualShaderNodeTransformConstant", TTR("Transform constant."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM));
	add_options.push_back(AddOption("TransformParameter", "Transform/Variables", "VisualShaderNodeTransformParameter", TTR("Transform parameter."), {}, VisualShaderNode::PORT_TYPE_TRANSFORM));

	// UTILITY

	add_options.push_back(AddOption("DistanceFade", "Utility", "VisualShaderNodeDistanceFade", TTR("The distance fade effect fades out each pixel based on its distance to another object."), {}, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("ProximityFade", "Utility", "VisualShaderNodeProximityFade", TTR("The proximity fade effect fades out each pixel based on its distance to another object."), {}, VisualShaderNode::PORT_TYPE_SCALAR, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("RandomRange", "Utility", "VisualShaderNodeRandomRange", TTR("Returns a random value between the minimum and maximum input values."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Remap", "Utility", "VisualShaderNodeRemap", TTR("Remaps a given input from the input range to the output range."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("RotationByAxis", "Utility", "VisualShaderNodeRotationByAxis", TTR("Rotates an input vector by a given angle."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("RotationByAxis", "Utility", "VisualShaderNodeRotationByAxis", TTR("Rotates an input vector by a given angle."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_VERTEX, Shader::MODE_SPATIAL));

	// VECTOR

	add_options.push_back(AddOption("VectorFunc", "Vector/Common", "VisualShaderNodeVectorFunc", TTR("Vector function."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("VectorOp", "Vector/Common", "VisualShaderNodeVectorOp", TTR("Vector operator."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("VectorCompose", "Vector/Common", "VisualShaderNodeVectorCompose", TTR("Composes vector from scalars.")));
	add_options.push_back(AddOption("VectorDecompose", "Vector/Common", "VisualShaderNodeVectorDecompose", TTR("Decomposes vector to scalars.")));

	add_options.push_back(AddOption("Vector2Compose", "Vector/Composition", "VisualShaderNodeVectorCompose", TTR("Composes 2D vector from two scalars."), { VisualShaderNodeVectorCompose::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Vector2Decompose", "Vector/Composition", "VisualShaderNodeVectorDecompose", TTR("Decomposes 2D vector to two scalars."), { VisualShaderNodeVectorDecompose::OP_TYPE_VECTOR_2D }));
	add_options.push_back(AddOption("Vector3Compose", "Vector/Composition", "VisualShaderNodeVectorCompose", TTR("Composes 3D vector from three scalars."), { VisualShaderNodeVectorCompose::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Vector3Decompose", "Vector/Composition", "VisualShaderNodeVectorDecompose", TTR("Decomposes 3D vector to three scalars."), { VisualShaderNodeVectorDecompose::OP_TYPE_VECTOR_3D }));
	add_options.push_back(AddOption("Vector4Compose", "Vector/Composition", "VisualShaderNodeVectorCompose", TTR("Composes 4D vector from four scalars."), { VisualShaderNodeVectorCompose::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Vector4Decompose", "Vector/Composition", "VisualShaderNodeVectorDecompose", TTR("Decomposes 4D vector to four scalars."), { VisualShaderNodeVectorDecompose::OP_TYPE_VECTOR_4D }));

	add_options.push_back(AddOption("Abs", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the absolute value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ABS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Abs", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the absolute value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ABS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Abs", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the absolute value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ABS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ACos", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ACOS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ACos", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ACOS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ACos", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ACOS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ACosH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ACOSH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ACosH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ACOSH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ACosH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ACOSH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ASin", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ASIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ASin", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ASIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ASin", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ASIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ASinH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ASINH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ASinH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ASINH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ASinH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ASINH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ATan", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ATAN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ATan", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ATAN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ATan", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the arc-tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ATAN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ATan2", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the arc-tangent of the parameters."), { VisualShaderNodeVectorOp::OP_ATAN2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ATan2", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the arc-tangent of the parameters."), { VisualShaderNodeVectorOp::OP_ATAN2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ATan2", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the arc-tangent of the parameters."), { VisualShaderNodeVectorOp::OP_ATAN2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("ATanH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ATANH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("ATanH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ATANH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("ATanH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse hyperbolic tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_ATANH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Ceil", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), { VisualShaderNodeVectorFunc::FUNC_CEIL, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Ceil", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), { VisualShaderNodeVectorFunc::FUNC_CEIL, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Ceil", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer that is greater than or equal to the parameter."), { VisualShaderNodeVectorFunc::FUNC_CEIL, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Clamp", "Vector/Functions", "VisualShaderNodeClamp", TTR("Constrains a value to lie between two further values."), { VisualShaderNodeClamp::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Clamp", "Vector/Functions", "VisualShaderNodeClamp", TTR("Constrains a value to lie between two further values."), { VisualShaderNodeClamp::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Clamp", "Vector/Functions", "VisualShaderNodeClamp", TTR("Constrains a value to lie between two further values."), { VisualShaderNodeClamp::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Cos", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_COS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Cos", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_COS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Cos", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_COS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("CosH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_COSH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("CosH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_COSH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("CosH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic cosine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_COSH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Cross", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Calculates the cross product of two vectors."), { VisualShaderNodeVectorOp::OP_CROSS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Degrees", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in radians to degrees."), { VisualShaderNodeVectorFunc::FUNC_DEGREES, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Degrees", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in radians to degrees."), { VisualShaderNodeVectorFunc::FUNC_DEGREES, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Degrees", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in radians to degrees."), { VisualShaderNodeVectorFunc::FUNC_DEGREES, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("DFdX", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'x' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_X, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("DFdX", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'x' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_X, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("DFdX", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'x' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_X, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("DFdY", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'y' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_Y, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("DFdY", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'y' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_Y, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("DFdY", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Derivative in 'y' using local differencing."), { VisualShaderNodeDerivativeFunc::FUNC_Y, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("Distance2D", "Vector/Functions", "VisualShaderNodeVectorDistance", TTR("Returns the distance between two points."), { VisualShaderNodeVectorDistance::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Distance3D", "Vector/Functions", "VisualShaderNodeVectorDistance", TTR("Returns the distance between two points."), { VisualShaderNodeVectorDistance::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Distance4D", "Vector/Functions", "VisualShaderNodeVectorDistance", TTR("Returns the distance between two points."), { VisualShaderNodeVectorDistance::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Dot", "Vector/Functions", "VisualShaderNodeDotProduct", TTR("Calculates the dot product of two vectors."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Exp", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-e Exponential."), { VisualShaderNodeVectorFunc::FUNC_EXP, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Exp", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-e Exponential."), { VisualShaderNodeVectorFunc::FUNC_EXP, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Exp", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-e Exponential."), { VisualShaderNodeVectorFunc::FUNC_EXP, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Exp2", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 Exponential."), { VisualShaderNodeVectorFunc::FUNC_EXP2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Exp2", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 Exponential."), { VisualShaderNodeVectorFunc::FUNC_EXP2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Exp2", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 Exponential."), { VisualShaderNodeVectorFunc::FUNC_EXP2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("FaceForward", "Vector/Functions", "VisualShaderNodeFaceForward", TTR("Returns the vector that points in the same direction as a reference vector. The function has three vector parameters : N, the vector to orient, I, the incident vector, and Nref, the reference vector. If the dot product of I and Nref is smaller than zero the return value is N. Otherwise -N is returned."), { VisualShaderNodeFaceForward::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("FaceForward", "Vector/Functions", "VisualShaderNodeFaceForward", TTR("Returns the vector that points in the same direction as a reference vector. The function has three vector parameters : N, the vector to orient, I, the incident vector, and Nref, the reference vector. If the dot product of I and Nref is smaller than zero the return value is N. Otherwise -N is returned."), { VisualShaderNodeFaceForward::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("FaceForward", "Vector/Functions", "VisualShaderNodeFaceForward", TTR("Returns the vector that points in the same direction as a reference vector. The function has three vector parameters : N, the vector to orient, I, the incident vector, and Nref, the reference vector. If the dot product of I and Nref is smaller than zero the return value is N. Otherwise -N is returned."), { VisualShaderNodeFaceForward::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Floor", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer less than or equal to the parameter."), { VisualShaderNodeVectorFunc::FUNC_FLOOR, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Floor", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer less than or equal to the parameter."), { VisualShaderNodeVectorFunc::FUNC_FLOOR, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Floor", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer less than or equal to the parameter."), { VisualShaderNodeVectorFunc::FUNC_FLOOR, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Fract", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Computes the fractional part of the argument."), { VisualShaderNodeVectorFunc::FUNC_FRACT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Fract", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Computes the fractional part of the argument."), { VisualShaderNodeVectorFunc::FUNC_FRACT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Fract", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Computes the fractional part of the argument."), { VisualShaderNodeVectorFunc::FUNC_FRACT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Fresnel", "Vector/Functions", "VisualShaderNodeFresnel", TTR("Returns falloff based on the dot product of surface normal and view direction of camera (pass associated inputs to it)."), {}, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("InverseSqrt", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse of the square root of the parameter."), { VisualShaderNodeVectorFunc::FUNC_INVERSE_SQRT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("InverseSqrt", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse of the square root of the parameter."), { VisualShaderNodeVectorFunc::FUNC_INVERSE_SQRT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("InverseSqrt", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the inverse of the square root of the parameter."), { VisualShaderNodeVectorFunc::FUNC_INVERSE_SQRT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Length2D", "Vector/Functions", "VisualShaderNodeVectorLen", TTR("Calculates the length of a vector."), { VisualShaderNodeVectorLen::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Length3D", "Vector/Functions", "VisualShaderNodeVectorLen", TTR("Calculates the length of a vector."), { VisualShaderNodeVectorLen::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Length4D", "Vector/Functions", "VisualShaderNodeVectorLen", TTR("Calculates the length of a vector."), { VisualShaderNodeVectorLen::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_SCALAR));
	add_options.push_back(AddOption("Log", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Natural logarithm."), { VisualShaderNodeVectorFunc::FUNC_LOG, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Log", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Natural logarithm."), { VisualShaderNodeVectorFunc::FUNC_LOG, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Log", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Natural logarithm."), { VisualShaderNodeVectorFunc::FUNC_LOG, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Log2", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 logarithm."), { VisualShaderNodeVectorFunc::FUNC_LOG2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Log2", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 logarithm."), { VisualShaderNodeVectorFunc::FUNC_LOG2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Log2", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Base-2 logarithm."), { VisualShaderNodeVectorFunc::FUNC_LOG2, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Max", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the greater of two values."), { VisualShaderNodeVectorOp::OP_MAX, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Max", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the greater of two values."), { VisualShaderNodeVectorOp::OP_MAX, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Max", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the greater of two values."), { VisualShaderNodeVectorOp::OP_MAX, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Min", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the lesser of two values."), { VisualShaderNodeVectorOp::OP_MIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Min", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the lesser of two values."), { VisualShaderNodeVectorOp::OP_MIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Min", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the lesser of two values."), { VisualShaderNodeVectorOp::OP_MIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Mix", "Vector/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two vectors."), { VisualShaderNodeMix::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Mix", "Vector/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two vectors."), { VisualShaderNodeMix::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Mix", "Vector/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two vectors."), { VisualShaderNodeMix::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("MixS", "Vector/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two vectors using scalar."), { VisualShaderNodeMix::OP_TYPE_VECTOR_2D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("MixS", "Vector/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two vectors using scalar."), { VisualShaderNodeMix::OP_TYPE_VECTOR_3D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("MixS", "Vector/Functions", "VisualShaderNodeMix", TTR("Linear interpolation between two vectors using scalar."), { VisualShaderNodeMix::OP_TYPE_VECTOR_4D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("MultiplyAdd (a * b + c)", "Vector/Functions", "VisualShaderNodeMultiplyAdd", TTR("Performs a fused multiply-add operation (a * b + c) on vectors."), { VisualShaderNodeMultiplyAdd::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("MultiplyAdd (a * b + c)", "Vector/Functions", "VisualShaderNodeMultiplyAdd", TTR("Performs a fused multiply-add operation (a * b + c) on vectors."), { VisualShaderNodeMultiplyAdd::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("MultiplyAdd (a * b + c)", "Vector/Functions", "VisualShaderNodeMultiplyAdd", TTR("Performs a fused multiply-add operation (a * b + c) on vectors."), { VisualShaderNodeMultiplyAdd::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Negate (*-1)", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the opposite value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_NEGATE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Negate (*-1)", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the opposite value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_NEGATE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Negate (*-1)", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the opposite value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_NEGATE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Normalize", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Calculates the normalize product of vector."), { VisualShaderNodeVectorFunc::FUNC_NORMALIZE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Normalize", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Calculates the normalize product of vector."), { VisualShaderNodeVectorFunc::FUNC_NORMALIZE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Normalize", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Calculates the normalize product of vector."), { VisualShaderNodeVectorFunc::FUNC_NORMALIZE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("OneMinus (1-)", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("1.0 - vector"), { VisualShaderNodeVectorFunc::FUNC_ONEMINUS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("OneMinus (1-)", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("1.0 - vector"), { VisualShaderNodeVectorFunc::FUNC_ONEMINUS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("OneMinus (1-)", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("1.0 - vector"), { VisualShaderNodeVectorFunc::FUNC_ONEMINUS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Pow (^)", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the value of the first parameter raised to the power of the second."), { VisualShaderNodeVectorOp::OP_POW, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Pow (^)", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the value of the first parameter raised to the power of the second."), { VisualShaderNodeVectorOp::OP_POW, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Pow (^)", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the value of the first parameter raised to the power of the second."), { VisualShaderNodeVectorOp::OP_POW, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Radians", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in degrees to radians."), { VisualShaderNodeVectorFunc::FUNC_RADIANS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Radians", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in degrees to radians."), { VisualShaderNodeVectorFunc::FUNC_RADIANS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Radians", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Converts a quantity in degrees to radians."), { VisualShaderNodeVectorFunc::FUNC_RADIANS, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Reciprocal", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("1.0 / vector"), { VisualShaderNodeVectorFunc::FUNC_RECIPROCAL, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Reciprocal", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("1.0 / vector"), { VisualShaderNodeVectorFunc::FUNC_RECIPROCAL, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Reciprocal", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("1.0 / vector"), { VisualShaderNodeVectorFunc::FUNC_RECIPROCAL, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Reflect", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the vector that points in the direction of reflection ( a : incident vector, b : normal vector )."), { VisualShaderNodeVectorOp::OP_REFLECT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Reflect", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the vector that points in the direction of reflection ( a : incident vector, b : normal vector )."), { VisualShaderNodeVectorOp::OP_REFLECT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Reflect", "Vector/Functions", "VisualShaderNodeVectorOp", TTR("Returns the vector that points in the direction of reflection ( a : incident vector, b : normal vector )."), { VisualShaderNodeVectorOp::OP_REFLECT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Refract", "Vector/Functions", "VisualShaderNodeVectorRefract", TTR("Returns the vector that points in the direction of refraction."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Refract", "Vector/Functions", "VisualShaderNodeVectorRefract", TTR("Returns the vector that points in the direction of refraction."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Refract", "Vector/Functions", "VisualShaderNodeVectorRefract", TTR("Returns the vector that points in the direction of refraction."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Round", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer to the parameter."), { VisualShaderNodeVectorFunc::FUNC_ROUND, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Round", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer to the parameter."), { VisualShaderNodeVectorFunc::FUNC_ROUND, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Round", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest integer to the parameter."), { VisualShaderNodeVectorFunc::FUNC_ROUND, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("RoundEven", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest even integer to the parameter."), { VisualShaderNodeVectorFunc::FUNC_ROUNDEVEN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("RoundEven", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest even integer to the parameter."), { VisualShaderNodeVectorFunc::FUNC_ROUNDEVEN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("RoundEven", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the nearest even integer to the parameter."), { VisualShaderNodeVectorFunc::FUNC_ROUNDEVEN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Saturate", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Clamps the value between 0.0 and 1.0."), { VisualShaderNodeVectorFunc::FUNC_SATURATE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Saturate", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Clamps the value between 0.0 and 1.0."), { VisualShaderNodeVectorFunc::FUNC_SATURATE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Saturate", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Clamps the value between 0.0 and 1.0."), { VisualShaderNodeVectorFunc::FUNC_SATURATE, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Sign", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Extracts the sign of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SIGN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Sign", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Extracts the sign of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SIGN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Sign", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Extracts the sign of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SIGN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Sin", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Sin", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Sin", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SIN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("SinH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SINH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("SinH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SINH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("SinH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic sine of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SINH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Sqrt", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the square root of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SQRT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Sqrt", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the square root of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SQRT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Sqrt", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the square root of the parameter."), { VisualShaderNodeVectorFunc::FUNC_SQRT, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("SmoothStep", "Vector/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( vector(edge0), vector(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("SmoothStep", "Vector/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( vector(edge0), vector(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("SmoothStep", "Vector/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( vector(edge0), vector(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("SmoothStepS", "Vector/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_VECTOR_2D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("SmoothStepS", "Vector/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_VECTOR_3D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("SmoothStepS", "Vector/Functions", "VisualShaderNodeSmoothStep", TTR("SmoothStep function( scalar(edge0), scalar(edge1), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge0' and 1.0 if 'x' is larger than 'edge1'. Otherwise the return value is interpolated between 0.0 and 1.0 using Hermite polynomials."), { VisualShaderNodeSmoothStep::OP_TYPE_VECTOR_4D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Step", "Vector/Functions", "VisualShaderNodeStep", TTR("Step function( vector(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), { VisualShaderNodeStep::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Step", "Vector/Functions", "VisualShaderNodeStep", TTR("Step function( vector(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), { VisualShaderNodeStep::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("StepS", "Vector/Functions", "VisualShaderNodeStep", TTR("Step function( scalar(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), { VisualShaderNodeStep::OP_TYPE_VECTOR_2D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("StepS", "Vector/Functions", "VisualShaderNodeStep", TTR("Step function( scalar(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), { VisualShaderNodeStep::OP_TYPE_VECTOR_3D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("StepS", "Vector/Functions", "VisualShaderNodeStep", TTR("Step function( scalar(edge), vector(x) ).\n\nReturns 0.0 if 'x' is smaller than 'edge' and otherwise 1.0."), { VisualShaderNodeStep::OP_TYPE_VECTOR_4D_SCALAR }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Sum (+)", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Sum of absolute derivative in 'x' and 'y'."), { VisualShaderNodeDerivativeFunc::FUNC_SUM, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("Sum (+)", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Sum of absolute derivative in 'x' and 'y'."), { VisualShaderNodeDerivativeFunc::FUNC_SUM, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("Sum (+)", "Vector/Functions", "VisualShaderNodeDerivativeFunc", TTR("(Fragment/Light mode only) (Vector) Sum of absolute derivative in 'x' and 'y'."), { VisualShaderNodeDerivativeFunc::FUNC_SUM, VisualShaderNodeDerivativeFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, -1, true));
	add_options.push_back(AddOption("Tan", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TAN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Tan", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TAN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Tan", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TAN, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("TanH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TANH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("TanH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TANH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("TanH", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Returns the hyperbolic tangent of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TANH, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Trunc", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the truncated value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TRUNC, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Trunc", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the truncated value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TRUNC, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Trunc", "Vector/Functions", "VisualShaderNodeVectorFunc", TTR("Finds the truncated value of the parameter."), { VisualShaderNodeVectorFunc::FUNC_TRUNC, VisualShaderNodeVectorFunc::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));

	add_options.push_back(AddOption("Add (+)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Adds 2D vector to 2D vector."), { VisualShaderNodeVectorOp::OP_ADD, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Add (+)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Adds 3D vector to 3D vector."), { VisualShaderNodeVectorOp::OP_ADD, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Add (+)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Adds 4D vector to 4D vector."), { VisualShaderNodeVectorOp::OP_ADD, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Divide (/)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Divides 2D vector by 2D vector."), { VisualShaderNodeVectorOp::OP_DIV, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Divide (/)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Divides 3D vector by 3D vector."), { VisualShaderNodeVectorOp::OP_DIV, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Divide (/)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Divides 4D vector by 4D vector."), { VisualShaderNodeVectorOp::OP_DIV, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Multiply (*)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Multiplies 2D vector by 2D vector."), { VisualShaderNodeVectorOp::OP_MUL, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Multiply (*)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Multiplies 3D vector by 3D vector."), { VisualShaderNodeVectorOp::OP_MUL, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Multiply (*)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Multiplies 4D vector by 4D vector."), { VisualShaderNodeVectorOp::OP_MUL, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Remainder", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Returns the remainder of the two 2D vectors."), { VisualShaderNodeVectorOp::OP_MOD, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Remainder", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Returns the remainder of the two 3D vectors."), { VisualShaderNodeVectorOp::OP_MOD, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Remainder", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Returns the remainder of the two 4D vectors."), { VisualShaderNodeVectorOp::OP_MOD, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Subtract (-)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Subtracts 2D vector from 2D vector."), { VisualShaderNodeVectorOp::OP_SUB, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D }, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Subtract (-)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Subtracts 3D vector from 3D vector."), { VisualShaderNodeVectorOp::OP_SUB, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D }, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Subtract (-)", "Vector/Operators", "VisualShaderNodeVectorOp", TTR("Subtracts 4D vector from 4D vector."), { VisualShaderNodeVectorOp::OP_SUB, VisualShaderNodeVectorOp::OP_TYPE_VECTOR_4D }, VisualShaderNode::PORT_TYPE_VECTOR_4D));

	add_options.push_back(AddOption("Vector2Constant", "Vector/Variables", "VisualShaderNodeVec2Constant", TTR("2D vector constant."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Vector2Parameter", "Vector/Variables", "VisualShaderNodeVec2Parameter", TTR("2D vector parameter."), {}, VisualShaderNode::PORT_TYPE_VECTOR_2D));
	add_options.push_back(AddOption("Vector3Constant", "Vector/Variables", "VisualShaderNodeVec3Constant", TTR("3D vector constant."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Vector3Parameter", "Vector/Variables", "VisualShaderNodeVec3Parameter", TTR("3D vector parameter."), {}, VisualShaderNode::PORT_TYPE_VECTOR_3D));
	add_options.push_back(AddOption("Vector4Constant", "Vector/Variables", "VisualShaderNodeVec4Constant", TTR("4D vector constant."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));
	add_options.push_back(AddOption("Vector4Parameter", "Vector/Variables", "VisualShaderNodeVec4Parameter", TTR("4D vector parameter."), {}, VisualShaderNode::PORT_TYPE_VECTOR_4D));

	// SPECIAL

	add_options.push_back(AddOption("Expression", "Special", "VisualShaderNodeExpression", TTR("Custom Godot Shader Language expression, with custom amount of input and output ports. This is a direct injection of code into the vertex/fragment/light function, do not use it to write the function declarations inside.")));
	add_options.push_back(AddOption("GlobalExpression", "Special", "VisualShaderNodeGlobalExpression", TTR("Custom Godot Shader Language expression, which is placed on top of the resulted shader. You can place various function definitions inside and call it later in the Expressions. You can also declare varyings, parameters and constants.")));
	add_options.push_back(AddOption("ParameterRef", "Special", "VisualShaderNodeParameterRef", TTR("A reference to an existing parameter.")));
	add_options.push_back(AddOption("VaryingGetter", "Special", "VisualShaderNodeVaryingGetter", TTR("Get varying parameter."), {}, -1, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("VaryingSetter", "Special", "VisualShaderNodeVaryingSetter", TTR("Set varying parameter."), {}, -1, TYPE_FLAGS_VERTEX | TYPE_FLAGS_FRAGMENT, Shader::MODE_SPATIAL));
	add_options.push_back(AddOption("VaryingGetter", "Special", "VisualShaderNodeVaryingGetter", TTR("Get varying parameter."), {}, -1, TYPE_FLAGS_FRAGMENT | TYPE_FLAGS_LIGHT, Shader::MODE_CANVAS_ITEM));
	add_options.push_back(AddOption("VaryingSetter", "Special", "VisualShaderNodeVaryingSetter", TTR("Set varying parameter."), {}, -1, TYPE_FLAGS_VERTEX | TYPE_FLAGS_FRAGMENT, Shader::MODE_CANVAS_ITEM));

	custom_node_option_idx = add_options.size();

	/////////////////////////////////////////////////////////////////////

	Ref<VisualShaderNodePluginDefault> default_plugin;
	default_plugin.instantiate();
	default_plugin->set_editor(this);
	add_plugin(default_plugin);

	graph_plugin.instantiate();
	graph_plugin->set_editor(this);

	property_editor_popup = memnew(PopupPanel);
	property_editor_popup->set_min_size(Size2(180, 0) * EDSCALE);
	add_child(property_editor_popup);

	edited_property_holder.instantiate();
}

class VisualShaderNodePluginInputEditor : public OptionButton {
	GDCLASS(VisualShaderNodePluginInputEditor, OptionButton);

	VisualShaderEditor *editor = nullptr;
	Ref<VisualShaderNodeInput> input;

public:
	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_READY: {
				connect("item_selected", callable_mp(this, &VisualShaderNodePluginInputEditor::_item_selected));
			} break;
		}
	}

	void _item_selected(int p_item) {
		editor->call_deferred(SNAME("_input_select_item"), input, get_item_text(p_item));
	}

	void setup(VisualShaderEditor *p_editor, const Ref<VisualShaderNodeInput> &p_input) {
		editor = p_editor;
		input = p_input;
		Ref<Texture2D> type_icon[] = {
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("float"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("int"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("uint"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector2"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector3"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector4"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("bool"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Transform3D"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("ImageTexture"), EditorStringName(EditorIcons)),
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

class VisualShaderNodePluginVaryingEditor : public OptionButton {
	GDCLASS(VisualShaderNodePluginVaryingEditor, OptionButton);

	VisualShaderEditor *editor = nullptr;
	Ref<VisualShaderNodeVarying> varying;

public:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			connect("item_selected", callable_mp(this, &VisualShaderNodePluginVaryingEditor::_item_selected));
		}
	}

	void _item_selected(int p_item) {
		editor->call_deferred(SNAME("_varying_select_item"), varying, get_item_text(p_item));
	}

	void setup(VisualShaderEditor *p_editor, const Ref<VisualShaderNodeVarying> &p_varying, VisualShader::Type p_type) {
		editor = p_editor;
		varying = p_varying;

		Ref<Texture2D> type_icon[] = {
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("float"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("int"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("uint"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector2"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector3"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector4"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("bool"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Transform3D"), EditorStringName(EditorIcons)),
		};

		bool is_getter = Ref<VisualShaderNodeVaryingGetter>(p_varying.ptr()).is_valid();

		add_item("[None]");

		int to_select = -1;
		for (int i = 0, j = 0; i < varying->get_varyings_count(); i++) {
			VisualShader::VaryingMode mode = varying->get_varying_mode_by_index(i);
			if (is_getter) {
				if (mode == VisualShader::VARYING_MODE_FRAG_TO_LIGHT) {
					if (p_type != VisualShader::TYPE_LIGHT) {
						j++;
						continue;
					}
				} else {
					if (p_type != VisualShader::TYPE_FRAGMENT && p_type != VisualShader::TYPE_LIGHT) {
						j++;
						continue;
					}
				}
			} else {
				if (mode == VisualShader::VARYING_MODE_FRAG_TO_LIGHT) {
					if (p_type != VisualShader::TYPE_FRAGMENT) {
						j++;
						continue;
					}
				} else {
					if (p_type != VisualShader::TYPE_VERTEX) {
						j++;
						continue;
					}
				}
			}
			if (varying->get_varying_name() == varying->get_varying_name_by_index(i)) {
				to_select = i - j + 1;
			}
			add_icon_item(type_icon[varying->get_varying_type_by_index(i)], varying->get_varying_name_by_index(i));
		}

		if (to_select >= 0) {
			select(to_select);
		}
	}
};

////////////////

class VisualShaderNodePluginParameterRefEditor : public OptionButton {
	GDCLASS(VisualShaderNodePluginParameterRefEditor, OptionButton);

	VisualShaderEditor *editor = nullptr;
	Ref<VisualShaderNodeParameterRef> parameter_ref;

public:
	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_READY: {
				connect("item_selected", callable_mp(this, &VisualShaderNodePluginParameterRefEditor::_item_selected));
			} break;
		}
	}

	void _item_selected(int p_item) {
		editor->call_deferred(SNAME("_parameter_ref_select_item"), parameter_ref, get_item_text(p_item));
	}

	void setup(VisualShaderEditor *p_editor, const Ref<VisualShaderNodeParameterRef> &p_parameter_ref) {
		editor = p_editor;
		parameter_ref = p_parameter_ref;

		Ref<Texture2D> type_icon[] = {
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("float"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("int"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("uint"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("bool"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector2"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector3"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Vector4"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Transform3D"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Color"), EditorStringName(EditorIcons)),
			EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("ImageTexture"), EditorStringName(EditorIcons)),
		};

		add_item("[None]");
		int to_select = -1;
		for (int i = 0; i < p_parameter_ref->get_parameters_count(); i++) {
			if (p_parameter_ref->get_parameter_name() == p_parameter_ref->get_parameter_name_by_index(i)) {
				to_select = i + 1;
			}
			add_icon_item(type_icon[p_parameter_ref->get_parameter_type_by_index(i)], p_parameter_ref->get_parameter_name_by_index(i));
		}

		if (to_select >= 0) {
			select(to_select);
		}
	}
};

////////////////

class VisualShaderNodePluginDefaultEditor : public VBoxContainer {
	GDCLASS(VisualShaderNodePluginDefaultEditor, VBoxContainer);
	VisualShaderEditor *editor = nullptr;
	Ref<Resource> parent_resource;
	int node_id = 0;
	VisualShader::Type shader_type;

public:
	void _property_changed(const String &p_property, const Variant &p_value, const String &p_field = "", bool p_changing = false) {
		if (p_changing) {
			return;
		}

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

		updating = true;
		undo_redo->create_action(vformat(TTR("Edit Visual Property: %s"), p_property), UndoRedo::MERGE_ENDS);
		undo_redo->add_do_property(node.ptr(), p_property, p_value);
		undo_redo->add_undo_property(node.ptr(), p_property, node->get(p_property));

		if (p_value.get_type() == Variant::OBJECT) {
			Ref<Resource> prev_res = node->get(p_property);
			Ref<Resource> curr_res = p_value;

			if (curr_res.is_null()) {
				undo_redo->add_do_method(this, "_open_inspector", (Ref<Resource>)parent_resource.ptr());
			} else {
				undo_redo->add_do_method(this, "_open_inspector", (Ref<Resource>)curr_res.ptr());
			}
			if (!prev_res.is_null()) {
				undo_redo->add_undo_method(this, "_open_inspector", (Ref<Resource>)prev_res.ptr());
			} else {
				undo_redo->add_undo_method(this, "_open_inspector", (Ref<Resource>)parent_resource.ptr());
			}
		}
		if (p_property != "constant") {
			VisualShaderGraphPlugin *graph_plugin = editor->get_graph_plugin();
			if (graph_plugin) {
				undo_redo->add_do_method(editor, "_update_next_previews", node_id);
				undo_redo->add_undo_method(editor, "_update_next_previews", node_id);
				undo_redo->add_do_method(graph_plugin, "update_node_deferred", shader_type, node_id);
				undo_redo->add_undo_method(graph_plugin, "update_node_deferred", shader_type, node_id);
			}
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

	void _resource_selected(const String &p_path, Ref<Resource> p_resource) {
		_open_inspector(p_resource);
	}

	void _open_inspector(Ref<Resource> p_resource) {
		InspectorDock::get_inspector_singleton()->edit(p_resource.ptr());
	}

	bool updating = false;
	Ref<VisualShaderNode> node;
	Vector<EditorProperty *> properties;
	Vector<Label *> prop_names;

	void _show_prop_names(bool p_show) {
		for (int i = 0; i < prop_names.size(); i++) {
			prop_names[i]->set_visible(p_show);
		}
	}

	void setup(VisualShaderEditor *p_editor, Ref<Resource> p_parent_resource, Vector<EditorProperty *> p_properties, const Vector<StringName> &p_names, const HashMap<StringName, String> &p_overrided_names, Ref<VisualShaderNode> p_node) {
		editor = p_editor;
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
			if (p_overrided_names.has(p_names[i])) {
				prop_name_str = p_overrided_names[p_names[i]] + ":";
			} else {
				prop_name_str = prop_name_str.capitalize() + ":";
			}
			prop_name->set_auto_translate(false); // TODO: Implement proper translation switch.
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
		node->connect_changed(callable_mp(this, &VisualShaderNodePluginDefaultEditor::_node_changed));
	}

	static void _bind_methods() {
		ClassDB::bind_method("_open_inspector", &VisualShaderNodePluginDefaultEditor::_open_inspector); // Used by UndoRedo.
		ClassDB::bind_method("_show_prop_names", &VisualShaderNodePluginDefaultEditor::_show_prop_names); // Used with call_deferred.
	}
};

Control *VisualShaderNodePluginDefault::create_editor(const Ref<Resource> &p_parent_resource, const Ref<VisualShaderNode> &p_node) {
	Ref<VisualShader> p_shader = Ref<VisualShader>(p_parent_resource.ptr());

	if (p_shader.is_valid() && (p_node->is_class("VisualShaderNodeVaryingGetter") || p_node->is_class("VisualShaderNodeVaryingSetter"))) {
		VisualShaderNodePluginVaryingEditor *editor = memnew(VisualShaderNodePluginVaryingEditor);
		editor->setup(vseditor, p_node, p_shader->get_shader_type());
		return editor;
	}

	if (p_node->is_class("VisualShaderNodeParameterRef")) {
		VisualShaderNodePluginParameterRefEditor *editor = memnew(VisualShaderNodePluginParameterRefEditor);
		editor->setup(vseditor, p_node);
		return editor;
	}

	if (p_node->is_class("VisualShaderNodeInput")) {
		VisualShaderNodePluginInputEditor *editor = memnew(VisualShaderNodePluginInputEditor);
		editor->setup(vseditor, p_node);
		return editor;
	}

	Vector<StringName> properties = p_node->get_editable_properties();
	if (properties.size() == 0) {
		return nullptr;
	}

	List<PropertyInfo> props;
	p_node->get_property_list(&props);

	Vector<PropertyInfo> pinfo;

	for (const PropertyInfo &E : props) {
		for (int i = 0; i < properties.size(); i++) {
			if (E.name == String(properties[i])) {
				pinfo.push_back(E);
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
		} else if (Object::cast_to<EditorPropertyTransform3D>(prop) || Object::cast_to<EditorPropertyVector3>(prop)) {
			prop->set_custom_minimum_size(Size2(250 * EDSCALE, 0));
		} else if (Object::cast_to<EditorPropertyQuaternion>(prop)) {
			prop->set_custom_minimum_size(Size2(320 * EDSCALE, 0));
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
	editor->setup(vseditor, p_parent_resource, editors, properties, p_node->get_editable_properties_names(), p_node);
	return editor;
}

void EditorPropertyVisualShaderMode::_option_selected(int p_which) {
	Ref<VisualShader> visual_shader(Object::cast_to<VisualShader>(get_edited_object()));
	if (visual_shader->get_mode() == p_which) {
		return;
	}

	ShaderEditorPlugin *shader_editor = Object::cast_to<ShaderEditorPlugin>(EditorNode::get_editor_data().get_editor_by_name("Shader"));
	if (!shader_editor) {
		return;
	}
	VisualShaderEditor *editor = shader_editor->get_visual_shader_editor(visual_shader);
	if (!editor) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Visual Shader Mode Changed"));
	//do is easy
	undo_redo->add_do_method(visual_shader.ptr(), "set_mode", p_which);
	undo_redo->add_undo_method(visual_shader.ptr(), "set_mode", visual_shader->get_mode());

	undo_redo->add_do_method(editor, "_set_mode", p_which);
	undo_redo->add_undo_method(editor, "_set_mode", visual_shader->get_mode());

	//now undo is hell

	//1. restore connections to output
	for (int i = 0; i < VisualShader::TYPE_MAX; i++) {
		VisualShader::Type type = VisualShader::Type(i);
		List<VisualShader::Connection> conns;
		visual_shader->get_node_connections(type, &conns);
		for (const VisualShader::Connection &E : conns) {
			if (E.to_node == VisualShader::NODE_ID_OUTPUT) {
				undo_redo->add_undo_method(visual_shader.ptr(), "connect_nodes", type, E.from_node, E.from_port, E.to_node, E.to_port);
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

	for (const PropertyInfo &E : props) {
		if (E.name.begins_with("flags/") || E.name.begins_with("modes/")) {
			undo_redo->add_undo_property(visual_shader.ptr(), E.name, visual_shader->get(E.name));
		}
	}

	//4. delete varyings (if needed)
	if (p_which == VisualShader::MODE_PARTICLES || p_which == VisualShader::MODE_SKY || p_which == VisualShader::MODE_FOG) {
		int var_count = visual_shader->get_varyings_count();

		if (var_count > 0) {
			for (int i = 0; i < var_count; i++) {
				const VisualShader::Varying *var = visual_shader->get_varying_by_index(i);
				undo_redo->add_do_method(visual_shader.ptr(), "remove_varying", var->name);
				undo_redo->add_undo_method(visual_shader.ptr(), "add_varying", var->name, var->mode, var->type);
			}

			undo_redo->add_do_method(editor, "_update_varyings");
			undo_redo->add_undo_method(editor, "_update_varyings");
		}
	}

	undo_redo->add_do_method(editor, "_update_nodes");
	undo_redo->add_undo_method(editor, "_update_nodes");

	undo_redo->add_do_method(editor, "_update_graph");
	undo_redo->add_undo_method(editor, "_update_graph");

	undo_redo->commit_action();
}

void EditorPropertyVisualShaderMode::update_property() {
	int which = get_edited_property_value();
	options->select(which);
}

void EditorPropertyVisualShaderMode::setup(const Vector<String> &p_options) {
	for (int i = 0; i < p_options.size(); i++) {
		options->add_item(p_options[i], i);
	}
}

void EditorPropertyVisualShaderMode::set_option_button_clip(bool p_enable) {
	options->set_clip_text(p_enable);
}

void EditorPropertyVisualShaderMode::_bind_methods() {
}

EditorPropertyVisualShaderMode::EditorPropertyVisualShaderMode() {
	options = memnew(OptionButton);
	options->set_clip_text(true);
	add_child(options);
	add_focusable(options);
	options->connect("item_selected", callable_mp(this, &EditorPropertyVisualShaderMode::_option_selected));
}

bool EditorInspectorVisualShaderModePlugin::can_handle(Object *p_object) {
	return true; // Can handle everything.
}

bool EditorInspectorVisualShaderModePlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	if (p_path == "mode" && p_object->is_class("VisualShader") && p_type == Variant::INT) {
		EditorPropertyVisualShaderMode *mode_editor = memnew(EditorPropertyVisualShaderMode);
		Vector<String> options = p_hint_text.split(",");
		mode_editor->setup(options);
		add_property_editor(p_path, mode_editor);

		return true;
	}

	return false;
}

//////////////////////////////////

void VisualShaderNodePortPreview::_shader_changed() {
	if (!is_valid || shader.is_null()) {
		return;
	}

	Vector<VisualShader::DefaultTextureParam> default_textures;
	String shader_code = shader->generate_preview_shader(type, node, port, default_textures);

	Ref<Shader> preview_shader;
	preview_shader.instantiate();
	preview_shader->set_code(shader_code);
	for (int i = 0; i < default_textures.size(); i++) {
		for (int j = 0; j < default_textures[i].params.size(); j++) {
			preview_shader->set_default_texture_parameter(default_textures[i].name, default_textures[i].params[j], j);
		}
	}

	Ref<ShaderMaterial> mat;
	mat.instantiate();
	mat->set_shader(preview_shader);

	//find if a material is also being edited and copy parameters to this one

	for (int i = EditorNode::get_singleton()->get_editor_selection_history()->get_path_size() - 1; i >= 0; i--) {
		Object *object = ObjectDB::get_instance(EditorNode::get_singleton()->get_editor_selection_history()->get_path_object(i));
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
			src_mat->get_shader()->get_shader_uniform_list(&params);
			for (const PropertyInfo &E : params) {
				mat->set(E.name, src_mat->get(E.name));
			}
		}
	}

	set_material(mat);
}

void VisualShaderNodePortPreview::setup(const Ref<VisualShader> &p_shader, VisualShader::Type p_type, int p_node, int p_port, bool p_is_valid) {
	shader = p_shader;
	shader->connect_changed(callable_mp(this, &VisualShaderNodePortPreview::_shader_changed), CONNECT_DEFERRED);
	type = p_type;
	port = p_port;
	node = p_node;
	is_valid = p_is_valid;
	queue_redraw();
	_shader_changed();
}

Size2 VisualShaderNodePortPreview::get_minimum_size() const {
	int port_preview_size = EDITOR_GET("editors/visual_editors/visual_shader/port_preview_size");
	return Size2(port_preview_size, port_preview_size) * EDSCALE;
}

void VisualShaderNodePortPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Vector<Vector2> points = {
				Vector2(),
				Vector2(get_size().width, 0),
				get_size(),
				Vector2(0, get_size().height)
			};

			Vector<Vector2> uvs = {
				Vector2(0, 0),
				Vector2(1, 0),
				Vector2(1, 1),
				Vector2(0, 1)
			};

			if (is_valid) {
				Vector<Color> colors = {
					Color(1, 1, 1, 1),
					Color(1, 1, 1, 1),
					Color(1, 1, 1, 1),
					Color(1, 1, 1, 1)
				};
				draw_primitive(points, colors, uvs);
			} else {
				Vector<Color> colors = {
					Color(0, 0, 0, 1),
					Color(0, 0, 0, 1),
					Color(0, 0, 0, 1),
					Color(0, 0, 0, 1)
				};
				draw_primitive(points, colors, uvs);
			}

		} break;
	}
}

void VisualShaderNodePortPreview::_bind_methods() {
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
	shader.instantiate();

	String code = vshader->get_code();
	shader->set_code(code);

	return shader;
}
