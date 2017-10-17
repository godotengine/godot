/*************************************************************************/
/*  visual_script_editor.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "visual_script_editor.h"

#include "editor/editor_node.h"
#include "editor/editor_resource_preview.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "visual_script_expression.h"
#include "visual_script_flow_control.h"
#include "visual_script_func_nodes.h"
#include "visual_script_nodes.h"

#ifdef TOOLS_ENABLED
class VisualScriptEditorSignalEdit : public Object {

	GDCLASS(VisualScriptEditorSignalEdit, Object)

	StringName sig;

public:
	UndoRedo *undo_redo;
	Ref<VisualScript> script;

protected:
	static void _bind_methods() {
		ClassDB::bind_method("_sig_changed", &VisualScriptEditorSignalEdit::_sig_changed);
		ADD_SIGNAL(MethodInfo("changed"));
	}

	void _sig_changed() {

		_change_notify();
		emit_signal("changed");
	}

	bool _set(const StringName &p_name, const Variant &p_value) {

		if (sig == StringName())
			return false;

		if (p_name == "argument_count") {

			int new_argc = p_value;
			int argc = script->custom_signal_get_argument_count(sig);
			if (argc == new_argc)
				return true;

			undo_redo->create_action(TTR("Change Signal Arguments"));

			if (new_argc < argc) {
				for (int i = new_argc; i < argc; i++) {
					undo_redo->add_do_method(script.ptr(), "custom_signal_remove_argument", sig, new_argc);
					undo_redo->add_undo_method(script.ptr(), "custom_signal_add_argument", sig, script->custom_signal_get_argument_name(sig, i), script->custom_signal_get_argument_type(sig, i), -1);
				}
			} else if (new_argc > argc) {

				for (int i = argc; i < new_argc; i++) {

					undo_redo->add_do_method(script.ptr(), "custom_signal_add_argument", sig, Variant::NIL, "arg" + itos(i + 1), -1);
					undo_redo->add_undo_method(script.ptr(), "custom_signal_remove_argument", sig, argc);
				}
			}

			undo_redo->add_do_method(this, "_sig_changed");
			undo_redo->add_undo_method(this, "_sig_changed");

			undo_redo->commit_action();

			return true;
		}
		if (String(p_name).begins_with("argument/")) {
			int idx = String(p_name).get_slice("/", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(idx, script->custom_signal_get_argument_count(sig), false);
			String what = String(p_name).get_slice("/", 2);
			if (what == "type") {

				int old_type = script->custom_signal_get_argument_type(sig, idx);
				int new_type = p_value;
				undo_redo->create_action(TTR("Change Argument Type"));
				undo_redo->add_do_method(script.ptr(), "custom_signal_set_argument_type", sig, idx, new_type);
				undo_redo->add_undo_method(script.ptr(), "custom_signal_set_argument_type", sig, idx, old_type);
				undo_redo->commit_action();

				return true;
			}

			if (what == "name") {

				String old_name = script->custom_signal_get_argument_name(sig, idx);
				String new_name = p_value;
				undo_redo->create_action(TTR("Change Argument name"));
				undo_redo->add_do_method(script.ptr(), "custom_signal_set_argument_name", sig, idx, new_name);
				undo_redo->add_undo_method(script.ptr(), "custom_signal_set_argument_name", sig, idx, old_name);
				undo_redo->commit_action();
				return true;
			}
		}

		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {

		if (sig == StringName())
			return false;

		if (p_name == "argument_count") {
			r_ret = script->custom_signal_get_argument_count(sig);
			return true;
		}
		if (String(p_name).begins_with("argument/")) {
			int idx = String(p_name).get_slice("/", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(idx, script->custom_signal_get_argument_count(sig), false);
			String what = String(p_name).get_slice("/", 2);
			if (what == "type") {
				r_ret = script->custom_signal_get_argument_type(sig, idx);
				return true;
			}
			if (what == "name") {
				r_ret = script->custom_signal_get_argument_name(sig, idx);
				return true;
			}
		}

		return false;
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {

		if (sig == StringName())
			return;

		p_list->push_back(PropertyInfo(Variant::INT, "argument_count", PROPERTY_HINT_RANGE, "0,256"));
		String argt = "Variant";
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			argt += "," + Variant::get_type_name(Variant::Type(i));
		}

		for (int i = 0; i < script->custom_signal_get_argument_count(sig); i++) {
			p_list->push_back(PropertyInfo(Variant::INT, "argument/" + itos(i + 1) + "/type", PROPERTY_HINT_ENUM, argt));
			p_list->push_back(PropertyInfo(Variant::STRING, "argument/" + itos(i + 1) + "/name"));
		}
	}

public:
	void edit(const StringName &p_sig) {

		sig = p_sig;
		_change_notify();
	}

	VisualScriptEditorSignalEdit() { undo_redo = NULL; }
};

class VisualScriptEditorVariableEdit : public Object {

	GDCLASS(VisualScriptEditorVariableEdit, Object)

	StringName var;

public:
	UndoRedo *undo_redo;
	Ref<VisualScript> script;

protected:
	static void _bind_methods() {
		ClassDB::bind_method("_var_changed", &VisualScriptEditorVariableEdit::_var_changed);
		ClassDB::bind_method("_var_value_changed", &VisualScriptEditorVariableEdit::_var_value_changed);
		ADD_SIGNAL(MethodInfo("changed"));
	}

	void _var_changed() {

		_change_notify();
		emit_signal("changed");
	}
	void _var_value_changed() {

		_change_notify("value"); //so the whole tree is not redrawn, makes editing smoother in general
		emit_signal("changed");
	}

	bool _set(const StringName &p_name, const Variant &p_value) {

		if (var == StringName())
			return false;

		if (String(p_name) == "value") {
			undo_redo->create_action(TTR("Set Variable Default Value"));
			Variant current = script->get_variable_default_value(var);
			undo_redo->add_do_method(script.ptr(), "set_variable_default_value", var, p_value);
			undo_redo->add_undo_method(script.ptr(), "set_variable_default_value", var, current);
			undo_redo->add_do_method(this, "_var_value_changed");
			undo_redo->add_undo_method(this, "_var_value_changed");
			undo_redo->commit_action();
			return true;
		}

		Dictionary d = script->call("get_variable_info", var);

		if (String(p_name) == "type") {

			Dictionary dc = d.copy();
			dc["type"] = p_value;
			undo_redo->create_action(TTR("Set Variable Type"));
			undo_redo->add_do_method(script.ptr(), "set_variable_info", var, dc);
			undo_redo->add_undo_method(script.ptr(), "set_variable_info", var, d);
			undo_redo->add_do_method(this, "_var_changed");
			undo_redo->add_undo_method(this, "_var_changed");
			undo_redo->commit_action();
			return true;
		}

		if (String(p_name) == "hint") {

			Dictionary dc = d.copy();
			dc["hint"] = p_value;
			undo_redo->create_action(TTR("Set Variable Type"));
			undo_redo->add_do_method(script.ptr(), "set_variable_info", var, dc);
			undo_redo->add_undo_method(script.ptr(), "set_variable_info", var, d);
			undo_redo->add_do_method(this, "_var_changed");
			undo_redo->add_undo_method(this, "_var_changed");
			undo_redo->commit_action();
			return true;
		}

		if (String(p_name) == "hint_string") {

			Dictionary dc = d.copy();
			dc["hint_string"] = p_value;
			undo_redo->create_action(TTR("Set Variable Type"));
			undo_redo->add_do_method(script.ptr(), "set_variable_info", var, dc);
			undo_redo->add_undo_method(script.ptr(), "set_variable_info", var, d);
			undo_redo->add_do_method(this, "_var_changed");
			undo_redo->add_undo_method(this, "_var_changed");
			undo_redo->commit_action();
			return true;
		}

		if (String(p_name) == "export") {
			script->set_variable_export(var, p_value);
			EditorNode::get_singleton()->get_property_editor()->update_tree();
			return true;
		}

		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {

		if (var == StringName())
			return false;

		if (String(p_name) == "value") {
			r_ret = script->get_variable_default_value(var);
			return true;
		}

		PropertyInfo pinfo = script->get_variable_info(var);

		if (String(p_name) == "type") {
			r_ret = pinfo.type;
			return true;
		}
		if (String(p_name) == "hint") {
			r_ret = pinfo.hint;
			return true;
		}
		if (String(p_name) == "hint_string") {
			r_ret = pinfo.hint_string;
			return true;
		}

		if (String(p_name) == "export") {
			r_ret = script->get_variable_export(var);
			return true;
		}

		return false;
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {

		if (var == StringName())
			return;

		String argt = "Variant";
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			argt += "," + Variant::get_type_name(Variant::Type(i));
		}
		p_list->push_back(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt));
		p_list->push_back(PropertyInfo(script->get_variable_info(var).type, "value", script->get_variable_info(var).hint, script->get_variable_info(var).hint_string, PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, "hint", PROPERTY_HINT_ENUM, "None,Range,ExpRange,Enum,ExpEasing,Length,SpriteFrame,KeyAccel,BitFlags,AllFlags,File,Dir,GlobalFile,GlobalDir,ResourceType,MultilineText"));
		p_list->push_back(PropertyInfo(Variant::STRING, "hint_string"));
		p_list->push_back(PropertyInfo(Variant::BOOL, "export"));
	}

public:
	void edit(const StringName &p_var) {

		var = p_var;
		_change_notify();
	}

	VisualScriptEditorVariableEdit() { undo_redo = NULL; }
};

static Color _color_from_type(Variant::Type p_type, bool dark_theme = true) {
	Color color;
	if (dark_theme)
		switch (p_type) {
			case Variant::NIL: color = Color::html("#69ecbd"); break;

			case Variant::BOOL: color = Color::html("#8da6f0"); break;
			case Variant::INT: color = Color::html("#7dc6ef"); break;
			case Variant::REAL: color = Color::html("#61daf4"); break;
			case Variant::STRING: color = Color::html("#6ba7ec"); break;

			case Variant::VECTOR2: color = Color::html("#bd91f1"); break;
			case Variant::RECT2: color = Color::html("#f191a5"); break;
			case Variant::VECTOR3: color = Color::html("#d67dee"); break;
			case Variant::TRANSFORM2D: color = Color::html("#c4ec69"); break;
			case Variant::PLANE: color = Color::html("#f77070"); break;
			case Variant::QUAT: color = Color::html("#ec69a3"); break;
			case Variant::RECT3: color = Color::html("#ee7991"); break;
			case Variant::BASIS: color = Color::html("#e3ec69"); break;
			case Variant::TRANSFORM: color = Color::html("#f6a86e"); break;

			case Variant::COLOR: color = Color::html("#9dff70"); break;
			case Variant::NODE_PATH: color = Color::html("#6993ec"); break;
			case Variant::_RID: color = Color::html("#69ec9a"); break;
			case Variant::OBJECT: color = Color::html("#79f3e8"); break;
			case Variant::DICTIONARY: color = Color::html("#77edb1"); break;

			case Variant::ARRAY: color = Color::html("#e0e0e0"); break;
			case Variant::POOL_BYTE_ARRAY: color = Color::html("#aaf4c8"); break;
			case Variant::POOL_INT_ARRAY: color = Color::html("#afdcf5"); break;
			case Variant::POOL_REAL_ARRAY: color = Color::html("#97e7f8"); break;
			case Variant::POOL_STRING_ARRAY: color = Color::html("#9dc4f2"); break;
			case Variant::POOL_VECTOR2_ARRAY: color = Color::html("#d1b3f5"); break;
			case Variant::POOL_VECTOR3_ARRAY: color = Color::html("#df9bf2"); break;
			case Variant::POOL_COLOR_ARRAY: color = Color::html("#e9ff97"); break;

			default:
				color.set_hsv(p_type / float(Variant::VARIANT_MAX), 0.7, 0.7);
		}
	else
		switch (p_type) {
			case Variant::NIL: color = Color::html("#25e3a0"); break;

			case Variant::BOOL: color = Color::html("#6d8eeb"); break;
			case Variant::INT: color = Color::html("#4fb2e9"); break;
			case Variant::REAL: color = Color::html("#27ccf0"); break;
			case Variant::STRING: color = Color::html("#4690e7"); break;

			case Variant::VECTOR2: color = Color::html("#ad76ee"); break;
			case Variant::RECT2: color = Color::html("#ee758e"); break;
			case Variant::VECTOR3: color = Color::html("#dc6aed"); break;
			case Variant::TRANSFORM2D: color = Color::html("#96ce1a"); break;
			case Variant::PLANE: color = Color::html("#f77070"); break;
			case Variant::QUAT: color = Color::html("#ec69a3"); break;
			case Variant::RECT3: color = Color::html("#ee7991"); break;
			case Variant::BASIS: color = Color::html("#b2bb19"); break;
			case Variant::TRANSFORM: color = Color::html("#f49047"); break;

			case Variant::COLOR: color = Color::html("#3cbf00"); break;
			case Variant::NODE_PATH: color = Color::html("#6993ec"); break;
			case Variant::_RID: color = Color::html("#2ce573"); break;
			case Variant::OBJECT: color = Color::html("#12d5c3"); break;
			case Variant::DICTIONARY: color = Color::html("#57e99f"); break;

			case Variant::ARRAY: color = Color::html("#737373"); break;
			case Variant::POOL_BYTE_ARRAY: color = Color::html("#61ea98"); break;
			case Variant::POOL_INT_ARRAY: color = Color::html("#61baeb"); break;
			case Variant::POOL_REAL_ARRAY: color = Color::html("#40d3f2"); break;
			case Variant::POOL_STRING_ARRAY: color = Color::html("#609fea"); break;
			case Variant::POOL_VECTOR2_ARRAY: color = Color::html("#9d5dea"); break;
			case Variant::POOL_VECTOR3_ARRAY: color = Color::html("#ca5aea"); break;
			case Variant::POOL_COLOR_ARRAY: color = Color::html("#92ba00"); break;

			default:
				color.set_hsv(p_type / float(Variant::VARIANT_MAX), 0.3, 0.3);
		}

	return color;
}

void VisualScriptEditor::_update_graph_connections() {

	graph->clear_connections();

	List<VisualScript::SequenceConnection> sequence_conns;
	script->get_sequence_connection_list(edited_func, &sequence_conns);

	for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {

		graph->connect_node(itos(E->get().from_node), E->get().from_output, itos(E->get().to_node), 0);
	}

	List<VisualScript::DataConnection> data_conns;
	script->get_data_connection_list(edited_func, &data_conns);

	for (List<VisualScript::DataConnection>::Element *E = data_conns.front(); E; E = E->next()) {

		VisualScript::DataConnection dc = E->get();

		Ref<VisualScriptNode> from_node = script->get_node(edited_func, E->get().from_node);
		Ref<VisualScriptNode> to_node = script->get_node(edited_func, E->get().to_node);

		if (to_node->has_input_sequence_port()) {
			dc.to_port++;
		}

		dc.from_port += from_node->get_output_sequence_port_count();

		graph->connect_node(itos(E->get().from_node), dc.from_port, itos(E->get().to_node), dc.to_port);
	}
}

void VisualScriptEditor::_update_graph(int p_only_id) {

	if (updating_graph)
		return;

	updating_graph = true;

	//byebye all nodes
	if (p_only_id >= 0) {
		if (graph->has_node(itos(p_only_id))) {
			Node *gid = graph->get_node(itos(p_only_id));
			if (gid)
				memdelete(gid);
		}
	} else {

		for (int i = 0; i < graph->get_child_count(); i++) {

			if (Object::cast_to<GraphNode>(graph->get_child(i))) {
				memdelete(graph->get_child(i));
				i--;
			}
		}
	}

	if (!script->has_function(edited_func)) {
		graph->hide();
		select_func_text->show();
		updating_graph = false;
		return;
	}

	graph->show();
	select_func_text->hide();

	Ref<Texture> type_icons[Variant::VARIANT_MAX] = {
		Control::get_icon("MiniVariant", "EditorIcons"),
		Control::get_icon("MiniBoolean", "EditorIcons"),
		Control::get_icon("MiniInteger", "EditorIcons"),
		Control::get_icon("MiniFloat", "EditorIcons"),
		Control::get_icon("MiniString", "EditorIcons"),
		Control::get_icon("MiniVector2", "EditorIcons"),
		Control::get_icon("MiniRect2", "EditorIcons"),
		Control::get_icon("MiniVector3", "EditorIcons"),
		Control::get_icon("MiniTransform2D", "EditorIcons"),
		Control::get_icon("MiniPlane", "EditorIcons"),
		Control::get_icon("MiniQuat", "EditorIcons"),
		Control::get_icon("MiniAabb", "EditorIcons"),
		Control::get_icon("MiniBasis", "EditorIcons"),
		Control::get_icon("MiniTransform", "EditorIcons"),
		Control::get_icon("MiniColor", "EditorIcons"),
		Control::get_icon("MiniPath", "EditorIcons"),
		Control::get_icon("MiniRid", "EditorIcons"),
		Control::get_icon("MiniObject", "EditorIcons"),
		Control::get_icon("MiniDictionary", "EditorIcons"),
		Control::get_icon("MiniArray", "EditorIcons"),
		Control::get_icon("MiniRawArray", "EditorIcons"),
		Control::get_icon("MiniIntArray", "EditorIcons"),
		Control::get_icon("MiniFloatArray", "EditorIcons"),
		Control::get_icon("MiniStringArray", "EditorIcons"),
		Control::get_icon("MiniVector2Array", "EditorIcons"),
		Control::get_icon("MiniVector3Array", "EditorIcons"),
		Control::get_icon("MiniColorArray", "EditorIcons")
	};

	Ref<Texture> seq_port = Control::get_icon("VisualShaderPort", "EditorIcons");

	List<int> ids;
	script->get_node_list(edited_func, &ids);
	StringName editor_icons = "EditorIcons";

	for (List<int>::Element *E = ids.front(); E; E = E->next()) {

		if (p_only_id >= 0 && p_only_id != E->get())
			continue;

		Ref<VisualScriptNode> node = script->get_node(edited_func, E->get());
		Vector2 pos = script->get_node_position(edited_func, E->get());

		GraphNode *gnode = memnew(GraphNode);
		gnode->set_title(node->get_caption());
		if (error_line == E->get()) {
			gnode->set_overlay(GraphNode::OVERLAY_POSITION);
		} else if (node->is_breakpoint()) {
			gnode->set_overlay(GraphNode::OVERLAY_BREAKPOINT);
		}

		gnode->set_meta("__vnode", node);
		gnode->set_name(itos(E->get()));
		gnode->connect("dragged", this, "_node_moved", varray(E->get()));
		gnode->connect("close_request", this, "_remove_node", varray(E->get()), CONNECT_DEFERRED);

		if (E->get() != script->get_function_node_id(edited_func)) {
			//function can't be erased
			gnode->set_show_close_button(true);
		}

		if (Object::cast_to<VisualScriptExpression>(node.ptr())) {

			LineEdit *line_edit = memnew(LineEdit);
			line_edit->set_text(node->get_text());
			line_edit->set_expand_to_text_length(true);
			line_edit->add_font_override("font", get_font("source", "EditorFonts"));
			gnode->add_child(line_edit);
			line_edit->connect("text_changed", this, "_expression_text_changed", varray(E->get()));
		} else {
			Label *text = memnew(Label);
			text->set_text(node->get_text());
			gnode->add_child(text);
		}

		if (Object::cast_to<VisualScriptComment>(node.ptr())) {
			Ref<VisualScriptComment> vsc = node;
			gnode->set_comment(true);
			gnode->set_resizable(true);
			gnode->set_custom_minimum_size(vsc->get_size() * EDSCALE);
			gnode->connect("resize_request", this, "_comment_node_resized", varray(E->get()));
		}

		if (node_styles.has(node->get_category())) {
			Ref<StyleBoxFlat> sbf = node_styles[node->get_category()];
			if (gnode->is_comment())
				sbf = EditorNode::get_singleton()->get_theme_base()->get_theme()->get_stylebox("comment", "GraphNode");

			Color c = sbf->get_border_color(MARGIN_TOP);
			c.a = 1;
			if (EditorSettings::get_singleton()->get("interface/theme/use_graph_node_headers")) {
				Color mono_color = ((c.r + c.g + c.b) / 3) < 0.7 ? Color(1.0, 1.0, 1.0) : Color(0.0, 0.0, 0.0);
				mono_color.a = 0.85;
				c = mono_color;
			}

			gnode->add_color_override("title_color", c);
			c.a = 0.7;
			gnode->add_color_override("close_color", c);
			gnode->add_style_override("frame", sbf);
		}

		const Color mono_color = get_color("mono_color", "Editor");

		int slot_idx = 0;

		bool single_seq_output = node->get_output_sequence_port_count() == 1 && node->get_output_sequence_port_text(0) == String();
		gnode->set_slot(0, node->has_input_sequence_port(), TYPE_SEQUENCE, mono_color, single_seq_output, TYPE_SEQUENCE, mono_color, seq_port, seq_port);
		gnode->set_offset(pos * EDSCALE);
		slot_idx++;

		int mixed_seq_ports = 0;

		if (!single_seq_output) {

			if (node->has_mixed_input_and_sequence_ports()) {
				mixed_seq_ports = node->get_output_sequence_port_count();
			} else {
				for (int i = 0; i < node->get_output_sequence_port_count(); i++) {

					Label *text2 = memnew(Label);
					text2->set_text(node->get_output_sequence_port_text(i));
					text2->set_align(Label::ALIGN_RIGHT);
					gnode->add_child(text2);
					gnode->set_slot(slot_idx, false, 0, Color(), true, TYPE_SEQUENCE, mono_color, seq_port, seq_port);
					slot_idx++;
				}
			}
		}

		for (int i = 0; i < MAX(node->get_output_value_port_count(), MAX(mixed_seq_ports, node->get_input_value_port_count())); i++) {

			bool left_ok = false;
			Variant::Type left_type = Variant::NIL;
			String left_name;

			if (i < node->get_input_value_port_count()) {
				PropertyInfo pi = node->get_input_value_port_info(i);
				left_ok = true;
				left_type = pi.type;
				left_name = pi.name;
			}

			bool right_ok = false;
			Variant::Type right_type = Variant::NIL;
			String right_name;

			if (i >= mixed_seq_ports && i < node->get_output_value_port_count() + mixed_seq_ports) {
				PropertyInfo pi = node->get_output_value_port_info(i - mixed_seq_ports);
				right_ok = true;
				right_type = pi.type;
				right_name = pi.name;
			}

			HBoxContainer *hbc = memnew(HBoxContainer);

			if (left_ok) {

				Ref<Texture> t;
				if (left_type >= 0 && left_type < Variant::VARIANT_MAX) {
					t = type_icons[left_type];
				}
				if (t.is_valid()) {
					TextureRect *tf = memnew(TextureRect);
					tf->set_texture(t);
					tf->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
					hbc->add_child(tf);
				}

				hbc->add_child(memnew(Label(left_name)));

				if (left_type != Variant::NIL && !script->is_input_value_port_connected(edited_func, E->get(), i)) {

					PropertyInfo pi = node->get_input_value_port_info(i);
					Button *button = memnew(Button);
					Variant value = node->get_default_input_value(i);
					if (value.get_type() != left_type) {
						//different type? for now convert
						//not the same, reconvert
						Variant::CallError ce;
						const Variant *existingp = &value;
						value = Variant::construct(left_type, &existingp, 1, ce, false);
					}

					if (left_type == Variant::COLOR) {
						button->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
						button->connect("draw", this, "_draw_color_over_button", varray(button, value));
					} else if (left_type == Variant::OBJECT && Ref<Resource>(value).is_valid()) {

						Ref<Resource> res = value;
						Array arr;
						arr.push_back(button->get_instance_id());
						arr.push_back(String(value));
						EditorResourcePreview::get_singleton()->queue_edited_resource_preview(res, this, "_button_resource_previewed", arr);

					} else if (pi.type == Variant::INT && pi.hint == PROPERTY_HINT_ENUM) {

						button->set_text(pi.hint_string.get_slice(",", value));
					} else {

						button->set_text(value);
					}
					button->connect("pressed", this, "_default_value_edited", varray(button, E->get(), i));
					hbc->add_child(button);
				}
			} else {
				Control *c = memnew(Control);
				c->set_custom_minimum_size(Size2(10, 0) * EDSCALE);
				hbc->add_child(c);
			}

			hbc->add_spacer();

			if (i < mixed_seq_ports) {

				Label *text2 = memnew(Label);
				text2->set_text(node->get_output_sequence_port_text(i));
				text2->set_align(Label::ALIGN_RIGHT);
				hbc->add_child(text2);
			}

			if (right_ok) {

				hbc->add_child(memnew(Label(right_name)));

				Ref<Texture> t;
				if (right_type >= 0 && right_type < Variant::VARIANT_MAX) {
					t = type_icons[right_type];
				}
				if (t.is_valid()) {
					TextureRect *tf = memnew(TextureRect);
					tf->set_texture(t);
					tf->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
					hbc->add_child(tf);
				}
			}

			gnode->add_child(hbc);

			bool dark_theme = get_constant("dark_theme", "Editor");
			if (i < mixed_seq_ports) {
				gnode->set_slot(slot_idx, left_ok, left_type, _color_from_type(left_type, dark_theme), true, TYPE_SEQUENCE, mono_color, Ref<Texture>(), seq_port);
			} else {
				gnode->set_slot(slot_idx, left_ok, left_type, _color_from_type(left_type, dark_theme), right_ok, right_type, _color_from_type(right_type, dark_theme));
			}

			slot_idx++;
		}

		graph->add_child(gnode);

		if (gnode->is_comment()) {
			graph->move_child(gnode, 0);
		}
	}

	_update_graph_connections();
	graph->call_deferred("set_scroll_ofs", script->get_function_scroll(edited_func) * EDSCALE); //may need to adapt a bit, let it do so
	updating_graph = false;
}

void VisualScriptEditor::_update_members() {

	updating_members = true;

	members->clear();
	TreeItem *root = members->create_item();

	TreeItem *functions = members->create_item(root);
	functions->set_selectable(0, false);
	functions->set_text(0, TTR("Functions:"));
	functions->add_button(0, Control::get_icon("Override", "EditorIcons"), 1);
	functions->add_button(0, Control::get_icon("Add", "EditorIcons"), 0);
	functions->set_custom_color(0, Control::get_color("mono_color", "Editor"));

	List<StringName> func_names;
	script->get_function_list(&func_names);
	for (List<StringName>::Element *E = func_names.front(); E; E = E->next()) {
		TreeItem *ti = members->create_item(functions);
		ti->set_text(0, E->get());
		ti->set_selectable(0, true);
		ti->set_editable(0, true);
		ti->set_metadata(0, E->get());
		if (selected == E->get())
			ti->select(0);
	}

	TreeItem *variables = members->create_item(root);
	variables->set_selectable(0, false);
	variables->set_text(0, TTR("Variables:"));
	variables->add_button(0, Control::get_icon("Add", "EditorIcons"));
	variables->set_custom_color(0, Control::get_color("mono_color", "Editor"));

	Ref<Texture> type_icons[Variant::VARIANT_MAX] = {
		Control::get_icon("MiniVariant", "EditorIcons"),
		Control::get_icon("MiniBoolean", "EditorIcons"),
		Control::get_icon("MiniInteger", "EditorIcons"),
		Control::get_icon("MiniFloat", "EditorIcons"),
		Control::get_icon("MiniString", "EditorIcons"),
		Control::get_icon("MiniVector2", "EditorIcons"),
		Control::get_icon("MiniRect2", "EditorIcons"),
		Control::get_icon("MiniVector3", "EditorIcons"),
		Control::get_icon("MiniMatrix32", "EditorIcons"),
		Control::get_icon("MiniPlane", "EditorIcons"),
		Control::get_icon("MiniQuat", "EditorIcons"),
		Control::get_icon("MiniAabb", "EditorIcons"),
		Control::get_icon("MiniMatrix3", "EditorIcons"),
		Control::get_icon("MiniTransform", "EditorIcons"),
		Control::get_icon("MiniColor", "EditorIcons"),
		Control::get_icon("MiniPath", "EditorIcons"),
		Control::get_icon("MiniRid", "EditorIcons"),
		Control::get_icon("MiniObject", "EditorIcons"),
		Control::get_icon("MiniDictionary", "EditorIcons"),
		Control::get_icon("MiniArray", "EditorIcons"),
		Control::get_icon("MiniRawArray", "EditorIcons"),
		Control::get_icon("MiniIntArray", "EditorIcons"),
		Control::get_icon("MiniFloatArray", "EditorIcons"),
		Control::get_icon("MiniStringArray", "EditorIcons"),
		Control::get_icon("MiniVector2Array", "EditorIcons"),
		Control::get_icon("MiniVector3Array", "EditorIcons"),
		Control::get_icon("MiniColorArray", "EditorIcons")
	};

	List<StringName> var_names;
	script->get_variable_list(&var_names);
	for (List<StringName>::Element *E = var_names.front(); E; E = E->next()) {
		TreeItem *ti = members->create_item(variables);

		ti->set_text(0, E->get());
		Variant var = script->get_variable_default_value(E->get());
		ti->set_suffix(0, "= " + String(var));
		ti->set_icon(0, type_icons[script->get_variable_info(E->get()).type]);

		ti->set_selectable(0, true);
		ti->set_editable(0, true);
		ti->set_metadata(0, E->get());
		if (selected == E->get())
			ti->select(0);
	}

	TreeItem *_signals = members->create_item(root);
	_signals->set_selectable(0, false);
	_signals->set_text(0, TTR("Signals:"));
	_signals->add_button(0, Control::get_icon("Add", "EditorIcons"));
	_signals->set_custom_color(0, Control::get_color("mono_color", "Editor"));

	List<StringName> signal_names;
	script->get_custom_signal_list(&signal_names);
	for (List<StringName>::Element *E = signal_names.front(); E; E = E->next()) {
		TreeItem *ti = members->create_item(_signals);
		ti->set_text(0, E->get());
		ti->set_selectable(0, true);
		ti->set_editable(0, true);
		ti->set_metadata(0, E->get());
		if (selected == E->get())
			ti->select(0);
	}

	String base_type = script->get_instance_base_type();
	String icon_type = base_type;
	if (!Control::has_icon(base_type, "EditorIcons")) {
		icon_type = "Object";
	}

	base_type_select->set_text(base_type);
	base_type_select->set_icon(Control::get_icon(icon_type, "EditorIcons"));

	updating_members = false;
}

void VisualScriptEditor::_member_selected() {

	if (updating_members)
		return;

	TreeItem *ti = members->get_selected();
	ERR_FAIL_COND(!ti);

	selected = ti->get_metadata(0);
	//print_line("selected: "+String(selected));

	if (ti->get_parent() == members->get_root()->get_children()) {

		if (edited_func != selected) {

			revert_on_drag = edited_func;
			edited_func = selected;
			_update_members();
			_update_graph();
		}

		return; //or crash because it will become invalid
	}
}

void VisualScriptEditor::_member_edited() {

	if (updating_members)
		return;

	TreeItem *ti = members->get_edited();
	ERR_FAIL_COND(!ti);

	String name = ti->get_metadata(0);
	String new_name = ti->get_text(0);

	if (name == new_name)
		return;

	if (!new_name.is_valid_identifier()) {

		EditorNode::get_singleton()->show_warning(TTR("Name is not a valid identifier:") + " " + new_name);
		updating_members = true;
		ti->set_text(0, name);
		updating_members = false;
		return;
	}

	if (script->has_function(new_name) || script->has_variable(new_name) || script->has_custom_signal(new_name)) {

		EditorNode::get_singleton()->show_warning(TTR("Name already in use by another func/var/signal:") + " " + new_name);
		updating_members = true;
		ti->set_text(0, name);
		updating_members = false;
		return;
	}

	TreeItem *root = members->get_root();

	if (ti->get_parent() == root->get_children()) {

		if (edited_func == selected) {
			edited_func = new_name;
		}
		selected = new_name;

		int node_id = script->get_function_node_id(name);
		Ref<VisualScriptFunction> func;
		if (script->has_node(name, node_id)) {
			func = script->get_node(name, node_id);
		}
		undo_redo->create_action(TTR("Rename Function"));
		undo_redo->add_do_method(script.ptr(), "rename_function", name, new_name);
		undo_redo->add_undo_method(script.ptr(), "rename_function", new_name, name);
		if (func.is_valid()) {

			undo_redo->add_do_method(func.ptr(), "set_name", new_name);
			undo_redo->add_undo_method(func.ptr(), "set_name", name);
		}
		undo_redo->add_do_method(this, "_update_members");
		undo_redo->add_undo_method(this, "_update_members");
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
		undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
		undo_redo->commit_action();

		//		_update_graph();

		return; //or crash because it will become invalid
	}

	if (ti->get_parent() == root->get_children()->get_next()) {

		selected = new_name;
		undo_redo->create_action(TTR("Rename Variable"));
		undo_redo->add_do_method(script.ptr(), "rename_variable", name, new_name);
		undo_redo->add_undo_method(script.ptr(), "rename_variable", new_name, name);
		undo_redo->add_do_method(this, "_update_members");
		undo_redo->add_undo_method(this, "_update_members");
		undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
		undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
		undo_redo->commit_action();

		return; //or crash because it will become invalid
	}

	if (ti->get_parent() == root->get_children()->get_next()->get_next()) {

		selected = new_name;
		undo_redo->create_action(TTR("Rename Signal"));
		undo_redo->add_do_method(script.ptr(), "rename_custom_signal", name, new_name);
		undo_redo->add_undo_method(script.ptr(), "rename_custom_signal", new_name, name);
		undo_redo->add_do_method(this, "_update_members");
		undo_redo->add_undo_method(this, "_update_members");
		undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
		undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
		undo_redo->commit_action();

		return; //or crash because it will become invalid
	}
}

void VisualScriptEditor::_member_button(Object *p_item, int p_column, int p_button) {

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);

	TreeItem *root = members->get_root();

	if (ti->get_parent() == root) {
		//main buttons
		if (ti == root->get_children()) {
			//add function, this one uses menu

			if (p_button == 1) {

				new_virtual_method_select->select_method_from_base_type(script->get_instance_base_type(), String(), true);

				return;
			} else if (p_button == 0) {

				String name = _validate_name("new_function");
				selected = name;
				edited_func = selected;

				Ref<VisualScriptFunction> func_node;
				func_node.instance();
				func_node->set_name(name);

				undo_redo->create_action(TTR("Add Function"));
				undo_redo->add_do_method(script.ptr(), "add_function", name);
				undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id(), func_node);
				undo_redo->add_undo_method(script.ptr(), "remove_function", name);
				undo_redo->add_do_method(this, "_update_members");
				undo_redo->add_undo_method(this, "_update_members");
				undo_redo->add_do_method(this, "_update_graph");
				undo_redo->add_undo_method(this, "_update_graph");
				undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
				undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
				undo_redo->commit_action();

				_update_graph();
			}

			return; //or crash because it will become invalid
		}

		if (ti == root->get_children()->get_next()) {
			//add variable
			String name = _validate_name("new_variable");
			selected = name;

			undo_redo->create_action(TTR("Add Variable"));
			undo_redo->add_do_method(script.ptr(), "add_variable", name);
			undo_redo->add_undo_method(script.ptr(), "remove_variable", name);
			undo_redo->add_do_method(this, "_update_members");
			undo_redo->add_undo_method(this, "_update_members");
			undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
			undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
			undo_redo->commit_action();
			return; //or crash because it will become invalid
		}

		if (ti == root->get_children()->get_next()->get_next()) {
			//add variable
			String name = _validate_name("new_signal");
			selected = name;

			undo_redo->create_action(TTR("Add Signal"));
			undo_redo->add_do_method(script.ptr(), "add_custom_signal", name);
			undo_redo->add_undo_method(script.ptr(), "remove_custom_signal", name);
			undo_redo->add_do_method(this, "_update_members");
			undo_redo->add_undo_method(this, "_update_members");
			undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
			undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
			undo_redo->commit_action();
			return; //or crash because it will become invalid
		}
	}
}

void VisualScriptEditor::_expression_text_changed(const String &p_text, int p_id) {

	Ref<VisualScriptExpression> vse = script->get_node(edited_func, p_id);
	if (!vse.is_valid())
		return;

	updating_graph = true;

	undo_redo->create_action(TTR("Change Expression"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_property(vse.ptr(), "expression", p_text);
	undo_redo->add_undo_property(vse.ptr(), "expression", vse->get("expression"));
	undo_redo->add_do_method(this, "_update_graph", p_id);
	undo_redo->add_undo_method(this, "_update_graph", p_id);
	undo_redo->commit_action();

	Node *node = graph->get_node(itos(p_id));
	if (Object::cast_to<Control>(node))
		Object::cast_to<Control>(node)->set_size(Vector2(1, 1)); //shrink if text is smaller

	updating_graph = false;
}

void VisualScriptEditor::_available_node_doubleclicked() {

	TreeItem *item = nodes->get_selected();

	if (!item)
		return;

	String which = item->get_metadata(0);
	if (which == String())
		return;

	Vector2 ofs = graph->get_scroll_ofs() + graph->get_size() * 0.5;

	if (graph->is_using_snap()) {
		int snap = graph->get_snap();
		ofs = ofs.snapped(Vector2(snap, snap));
	}

	ofs /= EDSCALE;

	while (true) {
		bool exists = false;
		List<int> existing;
		script->get_node_list(edited_func, &existing);
		for (List<int>::Element *E = existing.front(); E; E = E->next()) {
			Point2 pos = script->get_node_position(edited_func, E->get());
			if (pos.distance_to(ofs) < 15) {
				ofs += Vector2(graph->get_snap(), graph->get_snap());
				exists = true;
				break;
			}
		}

		if (exists)
			continue;
		break;
	}

	Ref<VisualScriptNode> vnode = VisualScriptLanguage::singleton->create_node_from_name(which);
	int new_id = script->get_available_id();

	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, vnode, ofs);
	undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();

	Node *node = graph->get_node(itos(new_id));
	if (node) {
		graph->set_selected(node);
		_node_selected(node);
	}
}

void VisualScriptEditor::_update_available_nodes() {

	nodes->clear();

	TreeItem *root = nodes->create_item();

	Map<String, TreeItem *> path_cache;

	String filter = node_filter->get_text();

	List<String> fnodes;
	VisualScriptLanguage::singleton->get_registered_node_names(&fnodes);

	for (List<String>::Element *E = fnodes.front(); E; E = E->next()) {

		Vector<String> path = E->get().split("/");

		if (filter != String() && path.size() && path[path.size() - 1].findn(filter) == -1)
			continue;

		String sp;
		TreeItem *parent = root;

		for (int i = 0; i < path.size() - 1; i++) {

			if (i > 0)
				sp += ",";
			sp += path[i];
			if (!path_cache.has(sp)) {
				TreeItem *pathn = nodes->create_item(parent);
				pathn->set_selectable(0, false);
				pathn->set_text(0, path[i].capitalize());
				path_cache[sp] = pathn;
				parent = pathn;
				if (filter == String()) {
					pathn->set_collapsed(true); //should remember state
				}
			} else {
				parent = path_cache[sp];
			}
		}

		TreeItem *item = nodes->create_item(parent);
		item->set_text(0, path[path.size() - 1].capitalize());
		item->set_selectable(0, true);
		item->set_metadata(0, E->get());
	}
}

String VisualScriptEditor::_validate_name(const String &p_name) const {

	String valid = p_name;

	int counter = 1;
	while (true) {

		bool exists = script->has_function(valid) || script->has_variable(valid) || script->has_custom_signal(valid);

		if (exists) {
			counter++;
			valid = p_name + "_" + itos(counter);
			continue;
		}

		break;
	}

	return valid;
}

void VisualScriptEditor::_on_nodes_delete() {

	List<int> to_erase;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			if (gn->is_selected() && gn->is_close_button_visible()) {
				to_erase.push_back(gn->get_name().operator String().to_int());
			}
		}
	}

	if (to_erase.empty())
		return;

	undo_redo->create_action(TTR("Remove VisualScript Nodes"));

	for (List<int>::Element *F = to_erase.front(); F; F = F->next()) {

		undo_redo->add_do_method(script.ptr(), "remove_node", edited_func, F->get());
		undo_redo->add_undo_method(script.ptr(), "add_node", edited_func, F->get(), script->get_node(edited_func, F->get()), script->get_node_position(edited_func, F->get()));

		List<VisualScript::SequenceConnection> sequence_conns;
		script->get_sequence_connection_list(edited_func, &sequence_conns);

		for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {

			if (E->get().from_node == F->get() || E->get().to_node == F->get()) {
				undo_redo->add_undo_method(script.ptr(), "sequence_connect", edited_func, E->get().from_node, E->get().from_output, E->get().to_node);
			}
		}

		List<VisualScript::DataConnection> data_conns;
		script->get_data_connection_list(edited_func, &data_conns);

		for (List<VisualScript::DataConnection>::Element *E = data_conns.front(); E; E = E->next()) {

			if (E->get().from_node == F->get() || E->get().to_node == F->get()) {
				undo_redo->add_undo_method(script.ptr(), "data_connect", edited_func, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}
		}
	}
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();
}

void VisualScriptEditor::_on_nodes_duplicate() {

	List<int> to_duplicate;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			if (gn->is_selected() && gn->is_close_button_visible()) {
				to_duplicate.push_back(gn->get_name().operator String().to_int());
			}
		}
	}

	if (to_duplicate.empty())
		return;

	undo_redo->create_action(TTR("Duplicate VisualScript Nodes"));
	int idc = script->get_available_id() + 1;

	Set<int> to_select;

	for (List<int>::Element *F = to_duplicate.front(); F; F = F->next()) {

		Ref<VisualScriptNode> node = script->get_node(edited_func, F->get());

		Ref<VisualScriptNode> dupe = node->duplicate();

		int new_id = idc++;
		to_select.insert(new_id);
		undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, dupe, script->get_node_position(edited_func, F->get()) + Vector2(20, 20));
		undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
	}
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			int id = gn->get_name().operator String().to_int();
			gn->set_selected(to_select.has(id));
		}
	}

	if (to_select.size()) {
		EditorNode::get_singleton()->push_item(script->get_node(edited_func, to_select.front()->get()).ptr());
	}
}

void VisualScriptEditor::_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		revert_on_drag = String(); //so we can still drag functions
	}
}

Variant VisualScriptEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {

	if (p_from == nodes) {

		TreeItem *it = nodes->get_item_at_position(p_point);
		if (!it)
			return Variant();
		String type = it->get_metadata(0);
		if (type == String())
			return Variant();

		Dictionary dd;
		dd["type"] = "visual_script_node_drag";
		dd["node_type"] = type;

		Label *label = memnew(Label);
		label->set_text(it->get_text(0));
		set_drag_preview(label);
		return dd;
	}

	if (p_from == members) {

		TreeItem *it = members->get_item_at_position(p_point);
		if (!it)
			return Variant();

		String type = it->get_metadata(0);

		if (type == String())
			return Variant();

		Dictionary dd;
		TreeItem *root = members->get_root();

		if (it->get_parent() == root->get_children()) {

			dd["type"] = "visual_script_function_drag";
			dd["function"] = type;
			if (revert_on_drag != String()) {
				edited_func = revert_on_drag; //revert so function does not change
				revert_on_drag = String();
				_update_graph();
			}
		} else if (it->get_parent() == root->get_children()->get_next()) {

			dd["type"] = "visual_script_variable_drag";
			dd["variable"] = type;
		} else if (it->get_parent() == root->get_children()->get_next()->get_next()) {

			dd["type"] = "visual_script_signal_drag";
			dd["signal"] = type;

		} else {
			return Variant();
		}

		Label *label = memnew(Label);
		label->set_text(it->get_text(0));
		set_drag_preview(label);
		return dd;
	}
	return Variant();
}

bool VisualScriptEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	if (p_from == graph) {

		Dictionary d = p_data;
		if (d.has("type") &&
				(String(d["type"]) == "visual_script_node_drag" ||
						String(d["type"]) == "visual_script_function_drag" ||
						String(d["type"]) == "visual_script_variable_drag" ||
						String(d["type"]) == "visual_script_signal_drag" ||
						String(d["type"]) == "obj_property" ||
						String(d["type"]) == "resource" ||
						String(d["type"]) == "files" ||
						String(d["type"]) == "nodes")) {

			if (String(d["type"]) == "obj_property") {

#ifdef OSX_ENABLED
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Meta to drop a Getter. Hold Shift to drop a generic signature."));
#else
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Ctrl to drop a Getter. Hold Shift to drop a generic signature."));
#endif
			}

			if (String(d["type"]) == "nodes") {

#ifdef OSX_ENABLED
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Meta to drop a simple reference to the node."));
#else
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Ctrl to drop a simple reference to the node."));
#endif
			}

			if (String(d["type"]) == "visual_script_variable_drag") {

#ifdef OSX_ENABLED
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Meta to drop a Variable Setter."));
#else
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Ctrl to drop a Variable Setter."));
#endif
			}

			return true;
		}
	}

	return false;
}

#ifdef TOOLS_ENABLED

static Node *_find_script_node(Node *p_edited_scene, Node *p_current_node, const Ref<Script> &script) {

	if (p_edited_scene != p_current_node && p_current_node->get_owner() != p_edited_scene)
		return NULL;

	Ref<Script> scr = p_current_node->get_script();

	if (scr.is_valid() && scr == script)
		return p_current_node;

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *n = _find_script_node(p_edited_scene, p_current_node->get_child(i), script);
		if (n)
			return n;
	}

	return NULL;
}

#endif

void VisualScriptEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {

	if (p_from != graph) {
		return;
	}

	Dictionary d = p_data;

	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "visual_script_node_drag") {
		if (!d.has("node_type") || String(d["node_type"]) == "Null") {
			return;
		}

		Vector2 ofs = graph->get_scroll_ofs() + p_point;

		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;

		Ref<VisualScriptNode> vnode = VisualScriptLanguage::singleton->create_node_from_name(d["node_type"]);
		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, vnode, ofs);
		undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();

		Node *node = graph->get_node(itos(new_id));
		if (node) {
			graph->set_selected(node);
			_node_selected(node);
		}
	}

	if (String(d["type"]) == "visual_script_variable_drag") {

#ifdef OSX_ENABLED
		bool use_set = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool use_set = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif
		Vector2 ofs = graph->get_scroll_ofs() + p_point;
		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;

		Ref<VisualScriptNode> vnode;
		if (use_set) {
			Ref<VisualScriptVariableSet> vnodes;
			vnodes.instance();
			vnodes->set_variable(d["variable"]);
			vnode = vnodes;
		} else {

			Ref<VisualScriptVariableGet> vnodeg;
			vnodeg.instance();
			vnodeg->set_variable(d["variable"]);
			vnode = vnodeg;
		}

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, vnode, ofs);
		undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();

		Node *node = graph->get_node(itos(new_id));
		if (node) {
			graph->set_selected(node);
			_node_selected(node);
		}
	}

	if (String(d["type"]) == "visual_script_function_drag") {

		Vector2 ofs = graph->get_scroll_ofs() + p_point;
		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;

		Ref<VisualScriptFunctionCall> vnode;
		vnode.instance();
		vnode->set_call_mode(VisualScriptFunctionCall::CALL_MODE_SELF);

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, vnode, ofs);
		undo_redo->add_do_method(vnode.ptr(), "set_base_type", script->get_instance_base_type());
		undo_redo->add_do_method(vnode.ptr(), "set_function", d["function"]);

		undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();

		Node *node = graph->get_node(itos(new_id));
		if (node) {
			graph->set_selected(node);
			_node_selected(node);
		}
	}

	if (String(d["type"]) == "visual_script_signal_drag") {

		Vector2 ofs = graph->get_scroll_ofs() + p_point;
		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;

		Ref<VisualScriptEmitSignal> vnode;
		vnode.instance();
		vnode->set_signal(d["signal"]);

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, vnode, ofs);
		undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();

		Node *node = graph->get_node(itos(new_id));
		if (node) {
			graph->set_selected(node);
			_node_selected(node);
		}
	}

	if (String(d["type"]) == "resource") {

		Vector2 ofs = graph->get_scroll_ofs() + p_point;
		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;

		Ref<VisualScriptPreload> prnode;
		prnode.instance();
		prnode->set_preload(d["resource"]);

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Preload Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, prnode, ofs);
		undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();

		Node *node = graph->get_node(itos(new_id));
		if (node) {
			graph->set_selected(node);
			_node_selected(node);
		}
	}

	if (String(d["type"]) == "files") {

		Vector2 ofs = graph->get_scroll_ofs() + p_point;
		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;

		Array files = d["files"];

		List<int> new_ids;
		int new_id = script->get_available_id();

		if (files.size()) {
			undo_redo->create_action(TTR("Add Preload Node"));

			for (int i = 0; i < files.size(); i++) {

				Ref<Resource> res = ResourceLoader::load(files[i]);
				if (!res.is_valid())
					continue;

				Ref<VisualScriptPreload> prnode;
				prnode.instance();
				prnode->set_preload(res);

				undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, prnode, ofs);
				undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
				new_ids.push_back(new_id);
				new_id++;
				ofs += Vector2(20, 20) * EDSCALE;
			}

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();
		}

		for (List<int>::Element *E = new_ids.front(); E; E = E->next()) {

			Node *node = graph->get_node(itos(E->get()));
			if (node) {
				graph->set_selected(node);
				_node_selected(node);
			}
		}
	}

	if (String(d["type"]) == "nodes") {

		Node *sn = _find_script_node(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root(), script);

		if (!sn) {
			EditorNode::get_singleton()->show_warning("Can't drop nodes because script '" + get_name() + "' is not used in this scene.");
			return;
		}

#ifdef OSX_ENABLED
		bool use_node = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool use_node = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif

		Array nodes = d["nodes"];

		Vector2 ofs = graph->get_scroll_ofs() + p_point;

		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}
		ofs /= EDSCALE;

		undo_redo->create_action(TTR("Add Node(s) From Tree"));
		int base_id = script->get_available_id();

		if (nodes.size() > 1) {
			use_node = true;
		}

		for (int i = 0; i < nodes.size(); i++) {

			NodePath np = nodes[i];
			Node *node = get_node(np);
			if (!node) {
				continue;
			}

			Ref<VisualScriptNode> n;

			if (use_node) {
				Ref<VisualScriptSceneNode> scene_node;
				scene_node.instance();
				scene_node->set_node_path(sn->get_path_to(node));
				n = scene_node;

			} else {
				Ref<VisualScriptFunctionCall> call;
				call.instance();
				call->set_call_mode(VisualScriptFunctionCall::CALL_MODE_NODE_PATH);
				call->set_base_path(sn->get_path_to(node));
				call->set_base_type(node->get_class());
				n = call;

				method_select->select_method_from_instance(node);
				selecting_method_id = base_id;
			}

			undo_redo->add_do_method(script.ptr(), "add_node", edited_func, base_id, n, ofs);
			undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, base_id);

			base_id++;
			ofs += Vector2(25, 25);
		}
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
	}

	if (String(d["type"]) == "obj_property") {

		Node *sn = _find_script_node(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root(), script);

		if (!sn && !Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
			EditorNode::get_singleton()->show_warning("Can't drop properties because script '" + get_name() + "' is not used in this scene.\nDrop holding 'Shift' to just copy the signature.");
			return;
		}

		Object *obj = d["object"];

		if (!obj)
			return;

		Node *node = Object::cast_to<Node>(obj);
		Vector2 ofs = graph->get_scroll_ofs() + p_point;

		if (graph->is_using_snap()) {
			int snap = graph->get_snap();
			ofs = ofs.snapped(Vector2(snap, snap));
		}

		ofs /= EDSCALE;
#ifdef OSX_ENABLED
		bool use_get = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool use_get = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif

		if (!node || Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {

			if (use_get)
				undo_redo->create_action(TTR("Add Getter Property"));
			else
				undo_redo->create_action(TTR("Add Setter Property"));

			int base_id = script->get_available_id();

			Ref<VisualScriptNode> vnode;

			if (!use_get) {

				Ref<VisualScriptPropertySet> pset;
				pset.instance();
				pset->set_call_mode(VisualScriptPropertySet::CALL_MODE_INSTANCE);
				pset->set_base_type(obj->get_class());
				/*if (use_value) {
						pset->set_use_builtin_value(true);
						pset->set_builtin_value(d["value"]);
					}*/
				vnode = pset;
			} else {

				Ref<VisualScriptPropertyGet> pget;
				pget.instance();
				pget->set_call_mode(VisualScriptPropertyGet::CALL_MODE_INSTANCE);
				pget->set_base_type(obj->get_class());

				vnode = pget;
			}

			undo_redo->add_do_method(script.ptr(), "add_node", edited_func, base_id, vnode, ofs);
			undo_redo->add_do_method(vnode.ptr(), "set_property", d["property"]);
			if (!use_get) {
				undo_redo->add_do_method(vnode.ptr(), "set_default_input_value", 0, d["value"]);
			}

			undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, base_id);

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();

		} else {

			if (use_get)
				undo_redo->create_action(TTR("Add Getter Property"));
			else
				undo_redo->create_action(TTR("Add Setter Property"));

			int base_id = script->get_available_id();

			Ref<VisualScriptNode> vnode;

			if (!use_get) {

				Ref<VisualScriptPropertySet> pset;
				pset.instance();
				if (sn == node) {
					pset->set_call_mode(VisualScriptPropertySet::CALL_MODE_SELF);
				} else {
					pset->set_call_mode(VisualScriptPropertySet::CALL_MODE_NODE_PATH);
					pset->set_base_path(sn->get_path_to(node));
				}

				vnode = pset;
			} else {

				Ref<VisualScriptPropertyGet> pget;
				pget.instance();
				if (sn == node) {
					pget->set_call_mode(VisualScriptPropertyGet::CALL_MODE_SELF);
				} else {
					pget->set_call_mode(VisualScriptPropertyGet::CALL_MODE_NODE_PATH);
					pget->set_base_path(sn->get_path_to(node));
				}
				vnode = pget;
			}
			undo_redo->add_do_method(script.ptr(), "add_node", edited_func, base_id, vnode, ofs);
			undo_redo->add_do_method(vnode.ptr(), "set_property", d["property"]);
			if (!use_get) {
				undo_redo->add_do_method(vnode.ptr(), "set_default_input_value", 0, d["value"]);
			}
			undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, base_id);

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();
		}
	}
}

void VisualScriptEditor::_selected_method(const String &p_method) {

	Ref<VisualScriptFunctionCall> vsfc = script->get_node(edited_func, selecting_method_id);
	if (!vsfc.is_valid())
		return;
	vsfc->set_function(p_method);
}

void VisualScriptEditor::_draw_color_over_button(Object *obj, Color p_color) {

	Button *button = Object::cast_to<Button>(obj);
	if (!button)
		return;

	Ref<StyleBox> normal = get_stylebox("normal", "Button");
	button->draw_rect(Rect2(normal->get_offset(), button->get_size() - normal->get_minimum_size()), p_color);
}

void VisualScriptEditor::_button_resource_previewed(const String &p_path, const Ref<Texture> &p_preview, Variant p_ud) {

	Array ud = p_ud;
	ERR_FAIL_COND(ud.size() != 2);

	ObjectID id = ud[0];
	Object *obj = ObjectDB::get_instance(id);

	if (!obj)
		return;

	Button *b = Object::cast_to<Button>(obj);
	ERR_FAIL_COND(!b);

	if (p_preview.is_null()) {
		b->set_text(ud[1]);
	} else {

		b->set_icon(p_preview);
	}
}

/////////////////////////

void VisualScriptEditor::apply_code() {
}

Ref<Script> VisualScriptEditor::get_edited_script() const {

	return script;
}

Vector<String> VisualScriptEditor::get_functions() {

	return Vector<String>();
}

void VisualScriptEditor::set_edited_script(const Ref<Script> &p_script) {

	script = p_script;
	signal_editor->script = p_script;
	signal_editor->undo_redo = undo_redo;
	variable_editor->script = p_script;
	variable_editor->undo_redo = undo_redo;

	script->connect("node_ports_changed", this, "_node_ports_changed");

	_update_members();
	_update_available_nodes();
}

void VisualScriptEditor::reload_text() {
}

String VisualScriptEditor::get_name() {

	String name;

	if (script->get_path().find("local://") == -1 && script->get_path().find("::") == -1) {
		name = script->get_path().get_file();
		if (is_unsaved()) {
			name += "(*)";
		}
	} else if (script->get_name() != "")
		name = script->get_name();
	else
		name = script->get_class() + "(" + itos(script->get_instance_id()) + ")";

	return name;
}

Ref<Texture> VisualScriptEditor::get_icon() {

	return Control::get_icon("VisualScript", "EditorIcons");
}

bool VisualScriptEditor::is_unsaved() {
#ifdef TOOLS_ENABLED

	return script->is_edited() || script->are_subnodes_edited();
#else
	return false;
#endif
}

Variant VisualScriptEditor::get_edit_state() {

	Dictionary d;
	d["function"] = edited_func;
	d["scroll"] = graph->get_scroll_ofs();
	d["zoom"] = graph->get_zoom();
	d["using_snap"] = graph->is_using_snap();
	d["snap"] = graph->get_snap();
	return d;
}

void VisualScriptEditor::set_edit_state(const Variant &p_state) {

	Dictionary d = p_state;
	if (d.has("function")) {
		edited_func = p_state;
		selected = edited_func;
	}

	_update_graph();
	_update_members();

	if (d.has("scroll")) {
		graph->set_scroll_ofs(d["scroll"]);
	}
	if (d.has("zoom")) {
		graph->set_zoom(d["zoom"]);
	}
	if (d.has("snap")) {
		graph->set_snap(d["snap"]);
	}
	if (d.has("snap_enabled")) {
		graph->set_use_snap(d["snap_enabled"]);
	}
}

void VisualScriptEditor::_center_on_node(int p_id) {

	Node *n = graph->get_node(itos(p_id));
	GraphNode *gn = Object::cast_to<GraphNode>(n);
	if (gn) {
		gn->set_selected(true);
		Vector2 new_scroll = gn->get_offset() - graph->get_size() * 0.5 + gn->get_size() * 0.5;
		graph->set_scroll_ofs(new_scroll);
		script->set_function_scroll(edited_func, new_scroll / EDSCALE);
		script->set_edited(true); //so it's saved
	}
}

void VisualScriptEditor::goto_line(int p_line, bool p_with_error) {

	p_line += 1; //add one because script lines begin from 0.

	if (p_with_error)
		error_line = p_line;

	List<StringName> functions;
	script->get_function_list(&functions);
	for (List<StringName>::Element *E = functions.front(); E; E = E->next()) {

		if (script->has_node(E->get(), p_line)) {

			edited_func = E->get();
			selected = edited_func;
			_update_graph();
			_update_members();

			call_deferred("call_deferred", "_center_on_node", p_line); //editor might be just created and size might not exist yet

			return;
		}
	}
}

void VisualScriptEditor::trim_trailing_whitespace() {
}

void VisualScriptEditor::convert_indent_to_spaces() {
}

void VisualScriptEditor::convert_indent_to_tabs() {
}

void VisualScriptEditor::ensure_focus() {

	graph->grab_focus();
}

void VisualScriptEditor::tag_saved_version() {
}

void VisualScriptEditor::reload(bool p_soft) {
}

void VisualScriptEditor::get_breakpoints(List<int> *p_breakpoints) {

	List<StringName> functions;
	script->get_function_list(&functions);
	for (List<StringName>::Element *E = functions.front(); E; E = E->next()) {

		List<int> nodes;
		script->get_node_list(E->get(), &nodes);
		for (List<int>::Element *F = nodes.front(); F; F = F->next()) {

			Ref<VisualScriptNode> vsn = script->get_node(E->get(), F->get());
			if (vsn->is_breakpoint()) {
				p_breakpoints->push_back(F->get() - 1); //subtract 1 because breakpoints in text start from zero
			}
		}
	}
}

void VisualScriptEditor::add_callback(const String &p_function, PoolStringArray p_args) {

	if (script->has_function(p_function)) {
		edited_func = p_function;
		selected = edited_func;
		_update_members();
		_update_graph();
		return;
	}

	Ref<VisualScriptFunction> func;
	func.instance();
	for (int i = 0; i < p_args.size(); i++) {

		String name = p_args[i];
		Variant::Type type = Variant::NIL;

		if (name.find(":") != -1) {
			String tt = name.get_slice(":", 1);
			name = name.get_slice(":", 0);
			for (int j = 0; j < Variant::VARIANT_MAX; j++) {

				String tname = Variant::get_type_name(Variant::Type(j));
				if (tname == tt) {
					type = Variant::Type(j);
					break;
				}
			}
		}

		func->add_argument(type, name);
	}

	func->set_name(p_function);
	script->add_function(p_function);
	script->add_node(p_function, script->get_available_id(), func);

	edited_func = p_function;
	selected = edited_func;
	_update_members();
	_update_graph();
	graph->call_deferred("set_scroll_ofs", script->get_function_scroll(edited_func)); //for first time it might need to be later

	//undo_redo->clear_history();
}

bool VisualScriptEditor::show_members_overview() {
	return false;
}

void VisualScriptEditor::update_settings() {

	_update_graph();
}

void VisualScriptEditor::set_debugger_active(bool p_active) {
	if (!p_active) {
		error_line = -1;
		_update_graph(); //clear line break
	}
}

void VisualScriptEditor::set_tooltip_request_func(String p_method, Object *p_obj) {
}

Control *VisualScriptEditor::get_edit_menu() {

	return edit_menu;
}

void VisualScriptEditor::_change_base_type() {

	select_base_type->popup_create(true);
}

void VisualScriptEditor::clear_edit_menu() {
	memdelete(edit_menu);
	memdelete(left_vsplit);
}

void VisualScriptEditor::_change_base_type_callback() {

	String bt = select_base_type->get_selected_type();

	ERR_FAIL_COND(bt == String());
	undo_redo->create_action(TTR("Change Base Type"));
	undo_redo->add_do_method(script.ptr(), "set_instance_base_type", bt);
	undo_redo->add_undo_method(script.ptr(), "set_instance_base_type", script->get_instance_base_type());
	undo_redo->add_do_method(this, "_update_members");
	undo_redo->add_undo_method(this, "_update_members");
	undo_redo->commit_action();
}

void VisualScriptEditor::_node_selected(Node *p_node) {

	Ref<VisualScriptNode> vnode = p_node->get_meta("__vnode");
	if (vnode.is_null())
		return;

	EditorNode::get_singleton()->push_item(vnode.ptr()); //edit node in inspector
}

static bool _get_out_slot(const Ref<VisualScriptNode> &p_node, int p_slot, int &r_real_slot, bool &r_sequence) {

	if (p_slot < p_node->get_output_sequence_port_count()) {
		r_sequence = true;
		r_real_slot = p_slot;

		return true;
	}

	r_real_slot = p_slot - p_node->get_output_sequence_port_count();
	r_sequence = false;

	return (r_real_slot < p_node->get_output_value_port_count());
}

static bool _get_in_slot(const Ref<VisualScriptNode> &p_node, int p_slot, int &r_real_slot, bool &r_sequence) {

	if (p_slot == 0 && p_node->has_input_sequence_port()) {
		r_sequence = true;
		r_real_slot = 0;
		return true;
	}

	r_real_slot = p_slot - (p_node->has_input_sequence_port() ? 1 : 0);
	r_sequence = false;

	return r_real_slot < p_node->get_input_value_port_count();
}

void VisualScriptEditor::_begin_node_move() {

	undo_redo->create_action(TTR("Move Node(s)"));
}

void VisualScriptEditor::_end_node_move() {

	undo_redo->commit_action();
}

void VisualScriptEditor::_move_node(String func, int p_id, const Vector2 &p_to) {

	if (func == String(edited_func)) {
		Node *node = graph->get_node(itos(p_id));
		if (Object::cast_to<GraphNode>(node))
			Object::cast_to<GraphNode>(node)->set_offset(p_to);
	}
	script->set_node_position(edited_func, p_id, p_to / EDSCALE);
}

void VisualScriptEditor::_node_moved(Vector2 p_from, Vector2 p_to, int p_id) {

	undo_redo->add_do_method(this, "_move_node", String(edited_func), p_id, p_to);
	undo_redo->add_undo_method(this, "_move_node", String(edited_func), p_id, p_from);
}

void VisualScriptEditor::_remove_node(int p_id) {

	undo_redo->create_action(TTR("Remove VisualScript Node"));

	undo_redo->add_do_method(script.ptr(), "remove_node", edited_func, p_id);
	undo_redo->add_undo_method(script.ptr(), "add_node", edited_func, p_id, script->get_node(edited_func, p_id), script->get_node_position(edited_func, p_id));

	List<VisualScript::SequenceConnection> sequence_conns;
	script->get_sequence_connection_list(edited_func, &sequence_conns);

	for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {

		if (E->get().from_node == p_id || E->get().to_node == p_id) {
			undo_redo->add_undo_method(script.ptr(), "sequence_connect", edited_func, E->get().from_node, E->get().from_output, E->get().to_node);
		}
	}

	List<VisualScript::DataConnection> data_conns;
	script->get_data_connection_list(edited_func, &data_conns);

	for (List<VisualScript::DataConnection>::Element *E = data_conns.front(); E; E = E->next()) {

		if (E->get().from_node == p_id || E->get().to_node == p_id) {
			undo_redo->add_undo_method(script.ptr(), "data_connect", edited_func, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();
}

void VisualScriptEditor::_node_ports_changed(const String &p_func, int p_id) {

	if (p_func != String(edited_func))
		return;

	_update_graph(p_id);
}

void VisualScriptEditor::_graph_connected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot) {

	Ref<VisualScriptNode> from_node = script->get_node(edited_func, p_from.to_int());
	ERR_FAIL_COND(!from_node.is_valid());

	bool from_seq;
	int from_port;

	if (!_get_out_slot(from_node, p_from_slot, from_port, from_seq))
		return; //can't  connect this, it' s invalid

	Ref<VisualScriptNode> to_node = script->get_node(edited_func, p_to.to_int());
	ERR_FAIL_COND(!to_node.is_valid());

	bool to_seq;
	int to_port;

	if (!_get_in_slot(to_node, p_to_slot, to_port, to_seq))
		return; //can't  connect this, it' s invalid

	ERR_FAIL_COND(from_seq != to_seq);

	undo_redo->create_action(TTR("Connect Nodes"));

	if (from_seq) {
		undo_redo->add_do_method(script.ptr(), "sequence_connect", edited_func, p_from.to_int(), from_port, p_to.to_int());
		undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", edited_func, p_from.to_int(), from_port, p_to.to_int());
	} else {

		// disconect current, and connect the new one
		if (script->is_input_value_port_connected(edited_func, p_to.to_int(), to_port)) {
			int conn_from;
			int conn_port;
			script->get_input_value_port_connection_source(edited_func, p_to.to_int(), to_port, &conn_from, &conn_port);
			undo_redo->add_do_method(script.ptr(), "data_disconnect", edited_func, conn_from, conn_port, p_to.to_int(), to_port);
			undo_redo->add_undo_method(script.ptr(), "data_connect", edited_func, conn_from, conn_port, p_to.to_int(), to_port);
		}

		undo_redo->add_do_method(script.ptr(), "data_connect", edited_func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		undo_redo->add_undo_method(script.ptr(), "data_disconnect", edited_func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		//update nodes in sgraph
		undo_redo->add_do_method(this, "_update_graph", p_from.to_int());
		undo_redo->add_do_method(this, "_update_graph", p_to.to_int());
		undo_redo->add_undo_method(this, "_update_graph", p_from.to_int());
		undo_redo->add_undo_method(this, "_update_graph", p_to.to_int());
	}

	undo_redo->add_do_method(this, "_update_graph_connections");
	undo_redo->add_undo_method(this, "_update_graph_connections");

	undo_redo->commit_action();
}

void VisualScriptEditor::_graph_disconnected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot) {

	Ref<VisualScriptNode> from_node = script->get_node(edited_func, p_from.to_int());
	ERR_FAIL_COND(!from_node.is_valid());

	bool from_seq;
	int from_port;

	if (!_get_out_slot(from_node, p_from_slot, from_port, from_seq))
		return; //can't  connect this, it' s invalid

	Ref<VisualScriptNode> to_node = script->get_node(edited_func, p_to.to_int());
	ERR_FAIL_COND(!to_node.is_valid());

	bool to_seq;
	int to_port;

	if (!_get_in_slot(to_node, p_to_slot, to_port, to_seq))
		return; //can't  connect this, it' s invalid

	ERR_FAIL_COND(from_seq != to_seq);

	undo_redo->create_action(TTR("Connect Nodes"));

	if (from_seq) {
		undo_redo->add_do_method(script.ptr(), "sequence_disconnect", edited_func, p_from.to_int(), from_port, p_to.to_int());
		undo_redo->add_undo_method(script.ptr(), "sequence_connect", edited_func, p_from.to_int(), from_port, p_to.to_int());
	} else {
		undo_redo->add_do_method(script.ptr(), "data_disconnect", edited_func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		undo_redo->add_undo_method(script.ptr(), "data_connect", edited_func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		//update nodes in sgraph
		undo_redo->add_do_method(this, "_update_graph", p_from.to_int());
		undo_redo->add_do_method(this, "_update_graph", p_to.to_int());
		undo_redo->add_undo_method(this, "_update_graph", p_from.to_int());
		undo_redo->add_undo_method(this, "_update_graph", p_to.to_int());
	}
	undo_redo->add_do_method(this, "_update_graph_connections");
	undo_redo->add_undo_method(this, "_update_graph_connections");

	undo_redo->commit_action();
}

void VisualScriptEditor::_graph_connect_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_pos) {

	Node *node = graph->get_node(p_from);
	GraphNode *gn = Object::cast_to<GraphNode>(node);
	if (!gn)
		return;

	Ref<VisualScriptNode> vsn = script->get_node(edited_func, p_from.to_int());
	if (!vsn.is_valid())
		return;

	if (p_from_slot < vsn->get_output_sequence_port_count()) {

		port_action_popup->clear();
		port_action_popup->add_item(TTR("Condition"), CREATE_COND);
		port_action_popup->add_item(TTR("Sequence"), CREATE_SEQUENCE);
		port_action_popup->add_item(TTR("Switch"), CREATE_SWITCH);
		port_action_popup->add_item(TTR("Iterator"), CREATE_ITERATOR);
		port_action_popup->add_item(TTR("While"), CREATE_WHILE);
		port_action_popup->add_item(TTR("Return"), CREATE_RETURN);

		port_action_node = p_from.to_int();
		port_action_output = p_from_slot;

	} else {
		port_action_popup->clear();
		port_action_popup->add_item(TTR("Call"), CREATE_CALL);
		port_action_popup->add_item(TTR("Get"), CREATE_GET);
		port_action_popup->add_item(TTR("Set"), CREATE_SET);

		port_action_output = p_from_slot - vsn->get_output_sequence_port_count();
		port_action_node = p_from.to_int();
	}

	port_action_pos = p_release_pos;
	port_action_popup->set_size(Size2(1, 1));
	port_action_popup->set_position(graph->get_global_position() + p_release_pos);
	port_action_popup->popup();
}

VisualScriptNode::TypeGuess VisualScriptEditor::_guess_output_type(int p_port_action_node, int p_port_action_output, Set<int> &visited_nodes) {

	VisualScriptNode::TypeGuess tg;
	tg.type = Variant::NIL;

	if (visited_nodes.has(p_port_action_node))
		return tg; //no loop

	visited_nodes.insert(p_port_action_node);

	Ref<VisualScriptNode> node = script->get_node(edited_func, p_port_action_node);

	if (!node.is_valid()) {

		return tg;
	}

	Vector<VisualScriptNode::TypeGuess> in_guesses;

	for (int i = 0; i < node->get_input_value_port_count(); i++) {
		PropertyInfo pi = node->get_input_value_port_info(i);
		VisualScriptNode::TypeGuess g;
		g.type = pi.type;

		if (g.type == Variant::NIL || g.type == Variant::OBJECT) {
			//any or object input, must further guess what this is
			int from_node;
			int from_port;

			if (script->get_input_value_port_connection_source(edited_func, p_port_action_node, i, &from_node, &from_port)) {

				g = _guess_output_type(from_node, from_port, visited_nodes);
			} else {
				Variant defval = node->get_default_input_value(i);
				if (defval.get_type() == Variant::OBJECT) {

					Object *obj = defval;

					if (obj) {

						g.type = Variant::OBJECT;
						g.gdclass = obj->get_class();
						g.script = obj->get_script();
					}
				}
			}
		}

		in_guesses.push_back(g);
	}

	return node->guess_output_type(in_guesses.ptr(), p_port_action_output);
}

void VisualScriptEditor::_port_action_menu(int p_option) {

	Vector2 ofs = graph->get_scroll_ofs() + port_action_pos;
	if (graph->is_using_snap()) {
		int snap = graph->get_snap();
		ofs = ofs.snapped(Vector2(snap, snap));
	}
	ofs /= EDSCALE;

	bool seq_connect = false;

	Ref<VisualScriptNode> vnode;
	Set<int> vn;

	switch (p_option) {

		case CREATE_CALL: {

			Ref<VisualScriptFunctionCall> n;
			n.instance();
			vnode = n;

			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);

			if (tg.type == Variant::OBJECT) {
				n->set_call_mode(VisualScriptFunctionCall::CALL_MODE_INSTANCE);

				if (tg.gdclass != StringName()) {
					n->set_base_type(tg.gdclass);
				} else {
					n->set_base_type("Object");
				}

				if (tg.script.is_valid()) {
					n->set_base_script(tg.script->get_path());
					new_connect_node_select->select_method_from_script(tg.script);
				} else {
					new_connect_node_select->select_method_from_base_type(n->get_base_type());
				}

			} else {
				n->set_call_mode(VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE);
				n->set_basic_type(tg.type);
				new_connect_node_select->select_method_from_basic_type(tg.type);
			}

		} break;
		case CREATE_SET: {

			Ref<VisualScriptPropertySet> n;
			n.instance();
			vnode = n;

			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);

			if (tg.type == Variant::OBJECT) {
				n->set_call_mode(VisualScriptPropertySet::CALL_MODE_INSTANCE);

				if (tg.gdclass != StringName()) {
					n->set_base_type(tg.gdclass);
				} else {
					n->set_base_type("Object");
				}

				if (tg.script.is_valid()) {
					n->set_base_script(tg.script->get_path());
					new_connect_node_select->select_property_from_script(tg.script);
				} else {
					new_connect_node_select->select_property_from_base_type(n->get_base_type());
				}

			} else {
				n->set_call_mode(VisualScriptPropertySet::CALL_MODE_BASIC_TYPE);
				n->set_basic_type(tg.type);
				new_connect_node_select->select_property_from_basic_type(tg.type);
			}
		} break;
		case CREATE_GET: {

			Ref<VisualScriptPropertyGet> n;
			n.instance();
			vnode = n;

			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);

			if (tg.type == Variant::OBJECT) {
				n->set_call_mode(VisualScriptPropertyGet::CALL_MODE_INSTANCE);

				if (tg.gdclass != StringName()) {
					n->set_base_type(tg.gdclass);
				} else {
					n->set_base_type("Object");
				}

				if (tg.script.is_valid()) {
					n->set_base_script(tg.script->get_path());
					new_connect_node_select->select_property_from_script(tg.script);
				} else {
					new_connect_node_select->select_property_from_base_type(n->get_base_type());
				}

			} else {
				n->set_call_mode(VisualScriptPropertyGet::CALL_MODE_BASIC_TYPE);
				n->set_basic_type(tg.type);
				new_connect_node_select->select_property_from_basic_type(tg.type);
			}

		} break;
		case CREATE_COND: {

			Ref<VisualScriptCondition> n;
			n.instance();
			vnode = n;
			seq_connect = true;

		} break;
		case CREATE_SEQUENCE: {

			Ref<VisualScriptSequence> n;
			n.instance();
			vnode = n;
			seq_connect = true;

		} break;
		case CREATE_SWITCH: {

			Ref<VisualScriptSwitch> n;
			n.instance();
			vnode = n;
			seq_connect = true;

		} break;
		case CREATE_ITERATOR: {

			Ref<VisualScriptIterator> n;
			n.instance();
			vnode = n;
			seq_connect = true;

		} break;
		case CREATE_WHILE: {

			Ref<VisualScriptWhile> n;
			n.instance();
			vnode = n;
			seq_connect = true;

		} break;
		case CREATE_RETURN: {

			Ref<VisualScriptReturn> n;
			n.instance();
			vnode = n;
			seq_connect = true;

		} break;
	}

	int new_id = script->get_available_id();
	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, vnode, ofs);
	if (seq_connect) {
		undo_redo->add_do_method(script.ptr(), "sequence_connect", edited_func, port_action_node, port_action_output, new_id);
	}
	undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
	undo_redo->add_do_method(this, "_update_graph", new_id);
	undo_redo->add_undo_method(this, "_update_graph", new_id);
	undo_redo->commit_action();

	port_action_new_node = new_id;
}

void VisualScriptEditor::_selected_connect_node_method_or_setget(const String &p_text) {

	Ref<VisualScriptNode> vsn = script->get_node(edited_func, port_action_new_node);

	if (Object::cast_to<VisualScriptFunctionCall>(vsn.ptr())) {

		Ref<VisualScriptFunctionCall> vsfc = vsn;
		vsfc->set_function(p_text);
		script->data_connect(edited_func, port_action_node, port_action_output, port_action_new_node, 0);
	}

	if (Object::cast_to<VisualScriptPropertySet>(vsn.ptr())) {

		Ref<VisualScriptPropertySet> vsp = vsn;
		vsp->set_property(p_text);
		script->data_connect(edited_func, port_action_node, port_action_output, port_action_new_node, 0);
	}

	if (Object::cast_to<VisualScriptPropertyGet>(vsn.ptr())) {

		Ref<VisualScriptPropertyGet> vsp = vsn;
		vsp->set_property(p_text);
		script->data_connect(edited_func, port_action_node, port_action_output, port_action_new_node, 0);
	}

	_update_graph(port_action_new_node);
	_update_graph_connections();
}

void VisualScriptEditor::_selected_new_virtual_method(const String &p_text) {

	String name = p_text;
	if (script->has_function(name)) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Script already has function '%s'"), name));
		return;
	}

	MethodInfo minfo;
	{
		List<MethodInfo> methods;
		bool found = false;
		ClassDB::get_virtual_methods(script->get_instance_base_type(), &methods);
		for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
			if (E->get().name == name) {
				minfo = E->get();
				found = true;
			}
		}

		ERR_FAIL_COND(!found);
	}

	selected = name;
	edited_func = selected;
	Ref<VisualScriptFunction> func_node;
	func_node.instance();
	func_node->set_name(name);

	undo_redo->create_action(TTR("Add Function"));
	undo_redo->add_do_method(script.ptr(), "add_function", name);

	for (int i = 0; i < minfo.arguments.size(); i++) {
		func_node->add_argument(minfo.arguments[i].type, minfo.arguments[i].name);
	}

	undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id(), func_node);
	if (minfo.return_val.type != Variant::NIL || minfo.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
		Ref<VisualScriptReturn> ret_node;
		ret_node.instance();
		ret_node->set_return_type(minfo.return_val.type);
		ret_node->set_enable_return_value(true);
		ret_node->set_name(name);
		undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id() + 1, ret_node, Vector2(500, 0));
	}

	undo_redo->add_undo_method(script.ptr(), "remove_function", name);
	undo_redo->add_do_method(this, "_update_members");
	undo_redo->add_undo_method(this, "_update_members");
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();

	_update_graph();
}

void VisualScriptEditor::_cancel_connect_node_method_or_setget() {

	script->remove_node(edited_func, port_action_new_node);
	_update_graph();
}

void VisualScriptEditor::_default_value_changed() {

	Ref<VisualScriptNode> vsn = script->get_node(edited_func, editing_id);
	if (vsn.is_null())
		return;

	undo_redo->create_action(TTR("Change Input Value"));
	undo_redo->add_do_method(vsn.ptr(), "set_default_input_value", editing_input, default_value_edit->get_variant());
	undo_redo->add_undo_method(vsn.ptr(), "set_default_input_value", editing_input, vsn->get_default_input_value(editing_input));

	undo_redo->add_do_method(this, "_update_graph", editing_id);
	undo_redo->add_undo_method(this, "_update_graph", editing_id);
	undo_redo->commit_action();
}

void VisualScriptEditor::_default_value_edited(Node *p_button, int p_id, int p_input_port) {

	Ref<VisualScriptNode> vsn = script->get_node(edited_func, p_id);
	if (vsn.is_null())
		return;

	PropertyInfo pinfo = vsn->get_input_value_port_info(p_input_port);
	Variant existing = vsn->get_default_input_value(p_input_port);
	if (pinfo.type != Variant::NIL && existing.get_type() != pinfo.type) {

		Variant::CallError ce;
		const Variant *existingp = &existing;
		existing = Variant::construct(pinfo.type, &existingp, 1, ce, false);
	}

	default_value_edit->set_position(Object::cast_to<Control>(p_button)->get_global_position() + Vector2(0, Object::cast_to<Control>(p_button)->get_size().y));
	default_value_edit->set_size(Size2(1, 1));
	if (default_value_edit->edit(NULL, pinfo.name, pinfo.type, existing, pinfo.hint, pinfo.hint_string)) {
		if (pinfo.hint == PROPERTY_HINT_MULTILINE_TEXT)
			default_value_edit->popup_centered_ratio();
		else
			default_value_edit->popup();
	}

	editing_id = p_id;
	editing_input = p_input_port;
}

void VisualScriptEditor::_show_hint(const String &p_hint) {

	hint_text->set_text(p_hint);
	hint_text->show();
	hint_text_timer->start();
}

void VisualScriptEditor::_hide_timer() {

	hint_text->hide();
}

void VisualScriptEditor::_node_filter_changed(const String &p_text) {

	_update_available_nodes();
}

void VisualScriptEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {
		node_filter->add_icon_override("right_icon", Control::get_icon("Search", "EditorIcons"));
		variable_editor->connect("changed", this, "_update_members");
		signal_editor->connect("changed", this, "_update_members");

		Ref<Theme> tm = EditorNode::get_singleton()->get_theme_base()->get_theme();

		bool dark_theme = tm->get_constant("dark_theme", "Editor");

		List<Pair<String, Color> > colors;
		if (dark_theme) {
			colors.push_back(Pair<String, Color>("flow_control", Color::html("#f4f4f4")));
			colors.push_back(Pair<String, Color>("functions", Color::html("#f58581")));
			colors.push_back(Pair<String, Color>("data", Color::html("#80f6cf")));
			colors.push_back(Pair<String, Color>("operators", Color::html("#ab97df")));
			colors.push_back(Pair<String, Color>("custom", Color::html("#80bbf6")));
			colors.push_back(Pair<String, Color>("constants", Color::html("#f680b0")));
		} else {
			colors.push_back(Pair<String, Color>("flow_control", Color::html("#424242")));
			colors.push_back(Pair<String, Color>("functions", Color::html("#f26661")));
			colors.push_back(Pair<String, Color>("data", Color::html("#13bb83")));
			colors.push_back(Pair<String, Color>("operators", Color::html("#8265d0")));
			colors.push_back(Pair<String, Color>("custom", Color::html("#4ea0f2")));
			colors.push_back(Pair<String, Color>("constants", Color::html("#f02f7d")));
		}

		for (List<Pair<String, Color> >::Element *E = colors.front(); E; E = E->next()) {
			Ref<StyleBoxFlat> sb = tm->get_stylebox("frame", "GraphNode");
			if (!sb.is_null()) {
				Ref<StyleBoxFlat> frame_style = sb->duplicate();
				Color c = sb->get_border_color(MARGIN_TOP);
				Color cn = E->get().second;
				cn.a = c.a;
				frame_style->set_border_color_all(cn);
				node_styles[E->get().first] = frame_style;
			}
		}
	}
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		left_vsplit->set_visible(is_visible_in_tree());
	}
}

void VisualScriptEditor::_graph_ofs_changed(const Vector2 &p_ofs) {

	if (updating_graph || !script.is_valid())
		return;

	updating_graph = true;

	if (script->has_function(edited_func)) {
		script->set_function_scroll(edited_func, graph->get_scroll_ofs() / EDSCALE);
		script->set_edited(true);
	}
	updating_graph = false;
}

void VisualScriptEditor::_comment_node_resized(const Vector2 &p_new_size, int p_node) {

	if (updating_graph)
		return;

	Ref<VisualScriptComment> vsc = script->get_node(edited_func, p_node);
	if (vsc.is_null())
		return;

	Node *node = graph->get_node(itos(p_node));
	GraphNode *gn = Object::cast_to<GraphNode>(node);
	if (!gn)
		return;

	updating_graph = true;

	graph->set_block_minimum_size_adjust(true); //faster resize

	undo_redo->create_action("Resize Comment", UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(vsc.ptr(), "set_size", p_new_size / EDSCALE);
	undo_redo->add_undo_method(vsc.ptr(), "set_size", vsc->get_size());
	undo_redo->commit_action();

	gn->set_custom_minimum_size(p_new_size); //for this time since graph update is blocked
	gn->set_size(Size2(1, 1));
	graph->set_block_minimum_size_adjust(false);
	updating_graph = false;
}

void VisualScriptEditor::_menu_option(int p_what) {

	switch (p_what) {
		case EDIT_DELETE_NODES: {
			_on_nodes_delete();
		} break;
		case EDIT_TOGGLE_BREAKPOINT: {

			List<String> reselect;
			for (int i = 0; i < graph->get_child_count(); i++) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
				if (gn) {
					if (gn->is_selected()) {
						int id = String(gn->get_name()).to_int();
						Ref<VisualScriptNode> vsn = script->get_node(edited_func, id);
						if (vsn.is_valid()) {
							vsn->set_breakpoint(!vsn->is_breakpoint());
							reselect.push_back(gn->get_name());
						}
					}
				}
			}

			_update_graph();

			for (List<String>::Element *E = reselect.front(); E; E = E->next()) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_node(E->get()));
				gn->set_selected(true);
			}

		} break;
		case EDIT_FIND_NODE_TYPE: {
			//popup disappearing grabs focus to owner, so use call deferred
			node_filter->call_deferred("grab_focus");
			node_filter->call_deferred("select_all");
		} break;
		case EDIT_COPY_NODES:
		case EDIT_CUT_NODES: {

			if (!script->has_function(edited_func))
				break;

			clipboard->nodes.clear();
			clipboard->data_connections.clear();
			clipboard->sequence_connections.clear();

			for (int i = 0; i < graph->get_child_count(); i++) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
				if (gn) {
					if (gn->is_selected()) {

						int id = String(gn->get_name()).to_int();
						Ref<VisualScriptNode> node = script->get_node(edited_func, id);
						if (Object::cast_to<VisualScriptFunction>(*node)) {
							EditorNode::get_singleton()->show_warning(TTR("Can't copy the function node."));
							return;
						}
						if (node.is_valid()) {
							clipboard->nodes[id] = node->duplicate();
							clipboard->nodes_positions[id] = script->get_node_position(edited_func, id);
						}
					}
				}
			}

			if (clipboard->nodes.empty())
				break;

			List<VisualScript::SequenceConnection> sequence_connections;

			script->get_sequence_connection_list(edited_func, &sequence_connections);

			for (List<VisualScript::SequenceConnection>::Element *E = sequence_connections.front(); E; E = E->next()) {

				if (clipboard->nodes.has(E->get().from_node) && clipboard->nodes.has(E->get().to_node)) {

					clipboard->sequence_connections.insert(E->get());
				}
			}

			List<VisualScript::DataConnection> data_connections;

			script->get_data_connection_list(edited_func, &data_connections);

			for (List<VisualScript::DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {

				if (clipboard->nodes.has(E->get().from_node) && clipboard->nodes.has(E->get().to_node)) {

					clipboard->data_connections.insert(E->get());
				}
			}

			if (p_what == EDIT_CUT_NODES) {
				_on_nodes_delete(); // oh yeah, also delete on cut
			}

		} break;
		case EDIT_PASTE_NODES: {
			if (!script->has_function(edited_func))
				break;

			if (clipboard->nodes.empty()) {
				EditorNode::get_singleton()->show_warning(TTR("Clipboard is empty!"));
				break;
			}

			Map<int, int> remap;

			undo_redo->create_action(TTR("Paste VisualScript Nodes"));
			int idc = script->get_available_id() + 1;

			Set<int> to_select;

			Set<Vector2> existing_positions;

			{
				List<int> nodes;
				script->get_node_list(edited_func, &nodes);
				for (List<int>::Element *E = nodes.front(); E; E = E->next()) {
					Vector2 pos = script->get_node_position(edited_func, E->get()).snapped(Vector2(2, 2));
					existing_positions.insert(pos);
				}
			}

			for (Map<int, Ref<VisualScriptNode> >::Element *E = clipboard->nodes.front(); E; E = E->next()) {

				Ref<VisualScriptNode> node = E->get()->duplicate();

				int new_id = idc++;
				to_select.insert(new_id);

				remap[E->key()] = new_id;

				Vector2 paste_pos = clipboard->nodes_positions[E->key()];

				while (existing_positions.has(paste_pos.snapped(Vector2(2, 2)))) {
					paste_pos += Vector2(20, 20) * EDSCALE;
				}

				undo_redo->add_do_method(script.ptr(), "add_node", edited_func, new_id, node, paste_pos);
				undo_redo->add_undo_method(script.ptr(), "remove_node", edited_func, new_id);
			}

			for (Set<VisualScript::SequenceConnection>::Element *E = clipboard->sequence_connections.front(); E; E = E->next()) {

				undo_redo->add_do_method(script.ptr(), "sequence_connect", edited_func, remap[E->get().from_node], E->get().from_output, remap[E->get().to_node]);
				undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", edited_func, remap[E->get().from_node], E->get().from_output, remap[E->get().to_node]);
			}

			for (Set<VisualScript::DataConnection>::Element *E = clipboard->data_connections.front(); E; E = E->next()) {

				undo_redo->add_do_method(script.ptr(), "data_connect", edited_func, remap[E->get().from_node], E->get().from_port, remap[E->get().to_node], E->get().to_port);
				undo_redo->add_undo_method(script.ptr(), "data_disconnect", edited_func, remap[E->get().from_node], E->get().from_port, remap[E->get().to_node], E->get().to_port);
			}

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");

			undo_redo->commit_action();

			for (int i = 0; i < graph->get_child_count(); i++) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
				if (gn) {
					int id = gn->get_name().operator String().to_int();
					gn->set_selected(to_select.has(id));
				}
			}
		} break;
	}
}

void VisualScriptEditor::_member_rmb_selected(const Vector2 &p_pos) {

	TreeItem *ti = members->get_selected();
	ERR_FAIL_COND(!ti);

	member_popup->clear();
	member_popup->set_position(members->get_global_position() + p_pos);
	member_popup->set_size(Vector2());

	TreeItem *root = members->get_root();

	Ref<Texture> del_icon = Control::get_icon("Remove", "EditorIcons");

	Ref<Texture> edit_icon = Control::get_icon("Edit", "EditorIcons");

	if (ti->get_parent() == root->get_children()) {

		member_type = MEMBER_FUNCTION;
		member_name = ti->get_text(0);
		member_popup->add_icon_item(del_icon, TTR("Remove Function"), MEMBER_REMOVE);
		member_popup->popup();
		return;
	}

	if (ti->get_parent() == root->get_children()->get_next()) {

		member_type = MEMBER_VARIABLE;
		member_name = ti->get_text(0);
		member_popup->add_icon_item(edit_icon, TTR("Edit Variable"), MEMBER_EDIT);
		member_popup->add_separator();
		member_popup->add_icon_item(del_icon, TTR("Remove Variable"), MEMBER_REMOVE);
		member_popup->popup();
		return;
	}

	if (ti->get_parent() == root->get_children()->get_next()->get_next()) {

		member_type = MEMBER_SIGNAL;
		member_name = ti->get_text(0);
		member_popup->add_icon_item(edit_icon, TTR("Edit Signal"), MEMBER_EDIT);
		member_popup->add_separator();
		member_popup->add_icon_item(del_icon, TTR("Remove Signal"), MEMBER_REMOVE);
		member_popup->popup();
		return;
	}
}

void VisualScriptEditor::_member_option(int p_option) {

	switch (member_type) {
		case MEMBER_FUNCTION: {

			if (p_option == MEMBER_REMOVE) {
				//delete the function
				String name = member_name;

				undo_redo->create_action(TTR("Remove Function"));
				undo_redo->add_do_method(script.ptr(), "remove_function", name);
				undo_redo->add_undo_method(script.ptr(), "add_function", name);
				List<int> nodes;
				script->get_node_list(name, &nodes);
				for (List<int>::Element *E = nodes.front(); E; E = E->next()) {
					undo_redo->add_undo_method(script.ptr(), "add_node", name, E->get(), script->get_node(name, E->get()), script->get_node_position(name, E->get()));
				}

				List<VisualScript::SequenceConnection> seq_connections;

				script->get_sequence_connection_list(name, &seq_connections);

				for (List<VisualScript::SequenceConnection>::Element *E = seq_connections.front(); E; E = E->next()) {
					undo_redo->add_undo_method(script.ptr(), "sequence_connect", name, E->get().from_node, E->get().from_output, E->get().to_node);
				}

				List<VisualScript::DataConnection> data_connections;

				script->get_data_connection_list(name, &data_connections);

				for (List<VisualScript::DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
					undo_redo->add_undo_method(script.ptr(), "data_connect", name, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
				}

				/*
				for(int i=0;i<script->function_get_argument_count(name);i++) {
					undo_redo->add_undo_method(script.ptr(),"function_add_argument",name,script->function_get_argument_name(name,i),script->function_get_argument_type(name,i));
				}
				*/
				undo_redo->add_do_method(this, "_update_members");
				undo_redo->add_undo_method(this, "_update_members");
				undo_redo->add_do_method(this, "_update_graph");
				undo_redo->add_undo_method(this, "_update_graph");
				undo_redo->commit_action();
			}
		} break;
		case MEMBER_VARIABLE: {

			String name = member_name;

			if (p_option == MEMBER_REMOVE) {
				undo_redo->create_action(TTR("Remove Variable"));
				undo_redo->add_do_method(script.ptr(), "remove_variable", name);
				undo_redo->add_undo_method(script.ptr(), "add_variable", name, script->get_variable_default_value(name));
				undo_redo->add_undo_method(script.ptr(), "set_variable_info", name, script->call("get_variable_info", name)); //return as dict
				undo_redo->add_do_method(this, "_update_members");
				undo_redo->add_undo_method(this, "_update_members");
				undo_redo->commit_action();
			} else if (p_option == MEMBER_EDIT) {
				variable_editor->edit(name);
				edit_variable_dialog->set_title(TTR("Editing Variable:") + " " + name);
				edit_variable_dialog->popup_centered_minsize(Size2(400, 200) * EDSCALE);
			}
		} break;
		case MEMBER_SIGNAL: {
			String name = member_name;

			if (p_option == MEMBER_REMOVE) {
				undo_redo->create_action(TTR("Remove Signal"));
				undo_redo->add_do_method(script.ptr(), "remove_custom_signal", name);
				undo_redo->add_undo_method(script.ptr(), "add_custom_signal", name);

				for (int i = 0; i < script->custom_signal_get_argument_count(name); i++) {
					undo_redo->add_undo_method(script.ptr(), "custom_signal_add_argument", name, script->custom_signal_get_argument_name(name, i), script->custom_signal_get_argument_type(name, i));
				}

				undo_redo->add_do_method(this, "_update_members");
				undo_redo->add_undo_method(this, "_update_members");
				undo_redo->commit_action();
			} else if (p_option == MEMBER_EDIT) {

				signal_editor->edit(name);
				edit_signal_dialog->set_title(TTR("Editing Signal:") + " " + name);
				edit_signal_dialog->popup_centered_minsize(Size2(400, 300) * EDSCALE);
			}
		} break;
	}
}

void VisualScriptEditor::_bind_methods() {

	ClassDB::bind_method("_member_button", &VisualScriptEditor::_member_button);
	ClassDB::bind_method("_member_edited", &VisualScriptEditor::_member_edited);
	ClassDB::bind_method("_member_selected", &VisualScriptEditor::_member_selected);
	ClassDB::bind_method("_update_members", &VisualScriptEditor::_update_members);
	ClassDB::bind_method("_change_base_type", &VisualScriptEditor::_change_base_type);
	ClassDB::bind_method("_change_base_type_callback", &VisualScriptEditor::_change_base_type_callback);
	ClassDB::bind_method("_node_selected", &VisualScriptEditor::_node_selected);
	ClassDB::bind_method("_node_moved", &VisualScriptEditor::_node_moved);
	ClassDB::bind_method("_move_node", &VisualScriptEditor::_move_node);
	ClassDB::bind_method("_begin_node_move", &VisualScriptEditor::_begin_node_move);
	ClassDB::bind_method("_end_node_move", &VisualScriptEditor::_end_node_move);
	ClassDB::bind_method("_remove_node", &VisualScriptEditor::_remove_node);
	ClassDB::bind_method("_update_graph", &VisualScriptEditor::_update_graph, DEFVAL(-1));
	ClassDB::bind_method("_node_ports_changed", &VisualScriptEditor::_node_ports_changed);
	ClassDB::bind_method("_available_node_doubleclicked", &VisualScriptEditor::_available_node_doubleclicked);
	ClassDB::bind_method("_default_value_edited", &VisualScriptEditor::_default_value_edited);
	ClassDB::bind_method("_default_value_changed", &VisualScriptEditor::_default_value_changed);
	ClassDB::bind_method("_menu_option", &VisualScriptEditor::_menu_option);
	ClassDB::bind_method("_graph_ofs_changed", &VisualScriptEditor::_graph_ofs_changed);
	ClassDB::bind_method("_center_on_node", &VisualScriptEditor::_center_on_node);
	ClassDB::bind_method("_comment_node_resized", &VisualScriptEditor::_comment_node_resized);
	ClassDB::bind_method("_button_resource_previewed", &VisualScriptEditor::_button_resource_previewed);
	ClassDB::bind_method("_port_action_menu", &VisualScriptEditor::_port_action_menu);
	ClassDB::bind_method("_selected_connect_node_method_or_setget", &VisualScriptEditor::_selected_connect_node_method_or_setget);
	ClassDB::bind_method("_selected_new_virtual_method", &VisualScriptEditor::_selected_new_virtual_method);

	ClassDB::bind_method("_cancel_connect_node_method_or_setget", &VisualScriptEditor::_cancel_connect_node_method_or_setget);
	ClassDB::bind_method("_expression_text_changed", &VisualScriptEditor::_expression_text_changed);

	ClassDB::bind_method("get_drag_data_fw", &VisualScriptEditor::get_drag_data_fw);
	ClassDB::bind_method("can_drop_data_fw", &VisualScriptEditor::can_drop_data_fw);
	ClassDB::bind_method("drop_data_fw", &VisualScriptEditor::drop_data_fw);

	ClassDB::bind_method("_input", &VisualScriptEditor::_input);
	ClassDB::bind_method("_on_nodes_delete", &VisualScriptEditor::_on_nodes_delete);
	ClassDB::bind_method("_on_nodes_duplicate", &VisualScriptEditor::_on_nodes_duplicate);

	ClassDB::bind_method("_hide_timer", &VisualScriptEditor::_hide_timer);

	ClassDB::bind_method("_graph_connected", &VisualScriptEditor::_graph_connected);
	ClassDB::bind_method("_graph_disconnected", &VisualScriptEditor::_graph_disconnected);
	ClassDB::bind_method("_graph_connect_to_empty", &VisualScriptEditor::_graph_connect_to_empty);

	ClassDB::bind_method("_update_graph_connections", &VisualScriptEditor::_update_graph_connections);
	ClassDB::bind_method("_node_filter_changed", &VisualScriptEditor::_node_filter_changed);

	ClassDB::bind_method("_selected_method", &VisualScriptEditor::_selected_method);
	ClassDB::bind_method("_draw_color_over_button", &VisualScriptEditor::_draw_color_over_button);

	ClassDB::bind_method("_member_rmb_selected", &VisualScriptEditor::_member_rmb_selected);

	ClassDB::bind_method("_member_option", &VisualScriptEditor::_member_option);
}

VisualScriptEditor::VisualScriptEditor() {

	if (!clipboard) {
		clipboard = memnew(Clipboard);
	}
	updating_graph = false;

	edit_menu = memnew(MenuButton);
	edit_menu->set_text(TTR("Edit"));
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/delete_selected"), EDIT_DELETE_NODES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/toggle_breakpoint"), EDIT_TOGGLE_BREAKPOINT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/find_node_type"), EDIT_FIND_NODE_TYPE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/copy_nodes"), EDIT_COPY_NODES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/cut_nodes"), EDIT_CUT_NODES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/paste_nodes"), EDIT_PASTE_NODES);

	edit_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	left_vsplit = memnew(VSplitContainer);
	ScriptEditor::get_singleton()->get_left_list_split()->call_deferred("add_child", left_vsplit); //add but wait until done settig up this
	left_vsplit->set_v_size_flags(SIZE_EXPAND_FILL);
	left_vsplit->set_stretch_ratio(2);
	left_vsplit->hide();

	VBoxContainer *left_vb = memnew(VBoxContainer);
	left_vsplit->add_child(left_vb);
	left_vb->set_v_size_flags(SIZE_EXPAND_FILL);
	//left_vb->set_custom_minimum_size(Size2(230, 1) * EDSCALE);

	base_type_select = memnew(Button);
	left_vb->add_margin_child(TTR("Base Type:"), base_type_select);
	base_type_select->connect("pressed", this, "_change_base_type");

	members = memnew(Tree);
	left_vb->add_margin_child(TTR("Members:"), members, true);
	members->set_hide_root(true);
	members->connect("button_pressed", this, "_member_button");
	members->connect("item_edited", this, "_member_edited");
	members->connect("cell_selected", this, "_member_selected", varray(), CONNECT_DEFERRED);
	members->set_allow_reselect(true);
	members->set_hide_folding(true);
	members->set_drag_forwarding(this);

	VBoxContainer *left_vb2 = memnew(VBoxContainer);
	left_vsplit->add_child(left_vb2);
	left_vb2->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *vbc_nodes = memnew(VBoxContainer);
	HBoxContainer *hbc_nodes = memnew(HBoxContainer);
	node_filter = memnew(LineEdit);
	node_filter->connect("text_changed", this, "_node_filter_changed");
	hbc_nodes->add_child(node_filter);
	node_filter->set_h_size_flags(SIZE_EXPAND_FILL);
	vbc_nodes->add_child(hbc_nodes);

	nodes = memnew(Tree);
	vbc_nodes->add_child(nodes);
	nodes->set_v_size_flags(SIZE_EXPAND_FILL);

	left_vb2->add_margin_child(TTR("Available Nodes:"), vbc_nodes, true);

	nodes->set_hide_root(true);
	nodes->connect("item_activated", this, "_available_node_doubleclicked");
	nodes->set_drag_forwarding(this);

	graph = memnew(GraphEdit);
	add_child(graph);
	graph->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	graph->connect("node_selected", this, "_node_selected");
	graph->connect("_begin_node_move", this, "_begin_node_move");
	graph->connect("_end_node_move", this, "_end_node_move");
	graph->connect("delete_nodes_request", this, "_on_nodes_delete");
	graph->connect("duplicate_nodes_request", this, "_on_nodes_duplicate");
	graph->set_drag_forwarding(this);
	graph->hide();
	graph->connect("scroll_offset_changed", this, "_graph_ofs_changed");

	select_func_text = memnew(Label);
	select_func_text->set_text(TTR("Select or create a function to edit graph"));
	select_func_text->set_align(Label::ALIGN_CENTER);
	select_func_text->set_valign(Label::VALIGN_CENTER);
	select_func_text->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(select_func_text);

	hint_text = memnew(Label);
	hint_text->set_anchor_and_margin(MARGIN_TOP, ANCHOR_END, -100);
	hint_text->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);
	hint_text->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	hint_text->set_align(Label::ALIGN_CENTER);
	hint_text->set_valign(Label::VALIGN_CENTER);
	graph->add_child(hint_text);

	hint_text_timer = memnew(Timer);
	hint_text_timer->set_wait_time(4);
	hint_text_timer->connect("timeout", this, "_hide_timer");
	add_child(hint_text_timer);

	//allowed casts (connections)
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		graph->add_valid_connection_type(Variant::NIL, i);
		graph->add_valid_connection_type(i, Variant::NIL);
		for (int j = 0; j < Variant::VARIANT_MAX; j++) {
			if (Variant::can_convert(Variant::Type(i), Variant::Type(j))) {
				graph->add_valid_connection_type(i, j);
			}
		}

		graph->add_valid_right_disconnect_type(i);
	}

	graph->add_valid_left_disconnect_type(TYPE_SEQUENCE);

	graph->connect("connection_request", this, "_graph_connected");
	graph->connect("disconnection_request", this, "_graph_disconnected");
	graph->connect("connection_to_empty", this, "_graph_connect_to_empty");

	edit_signal_dialog = memnew(AcceptDialog);
	edit_signal_dialog->get_ok()->set_text(TTR("Close"));
	add_child(edit_signal_dialog);
	edit_signal_dialog->set_title(TTR("Edit Signal Arguments:"));

	signal_editor = memnew(VisualScriptEditorSignalEdit);
	edit_signal_edit = memnew(PropertyEditor);
	edit_signal_edit->hide_top_label();
	edit_signal_dialog->add_child(edit_signal_edit);

	edit_signal_edit->edit(signal_editor);

	edit_variable_dialog = memnew(AcceptDialog);
	edit_variable_dialog->get_ok()->set_text(TTR("Close"));
	add_child(edit_variable_dialog);
	edit_variable_dialog->set_title(TTR("Edit Variable:"));

	variable_editor = memnew(VisualScriptEditorVariableEdit);
	edit_variable_edit = memnew(PropertyEditor);
	edit_variable_edit->hide_top_label();
	edit_variable_dialog->add_child(edit_variable_edit);

	edit_variable_edit->edit(variable_editor);

	select_base_type = memnew(CreateDialog);
	select_base_type->set_base_type("Object"); //anything goes
	select_base_type->connect("create", this, "_change_base_type_callback");
	select_base_type->get_ok()->set_text(TTR("Change"));
	add_child(select_base_type);

	undo_redo = EditorNode::get_singleton()->get_undo_redo();

	updating_members = false;

	set_process_input(true); //for revert on drag
	set_process_unhandled_input(true); //for revert on drag

	default_value_edit = memnew(CustomPropertyEditor);
	add_child(default_value_edit);
	default_value_edit->connect("variant_changed", this, "_default_value_changed");

	method_select = memnew(PropertySelector);
	add_child(method_select);
	method_select->connect("selected", this, "_selected_method");
	error_line = -1;

	new_connect_node_select = memnew(PropertySelector);
	add_child(new_connect_node_select);
	new_connect_node_select->connect("selected", this, "_selected_connect_node_method_or_setget");
	new_connect_node_select->get_cancel()->connect("pressed", this, "_cancel_connect_node_method_or_setget");

	new_virtual_method_select = memnew(PropertySelector);
	add_child(new_virtual_method_select);
	new_virtual_method_select->connect("selected", this, "_selected_new_virtual_method");
	new_virtual_method_select->get_cancel()->connect("pressed", this, "_selected_new_virtual_method");

	port_action_popup = memnew(PopupMenu);
	add_child(port_action_popup);
	port_action_popup->connect("id_pressed", this, "_port_action_menu");

	member_popup = memnew(PopupMenu);
	add_child(member_popup);
	members->connect("item_rmb_selected", this, "_member_rmb_selected");
	members->set_allow_rmb_select(true);
	member_popup->connect("id_pressed", this, "_member_option");
}

VisualScriptEditor::~VisualScriptEditor() {

	undo_redo->clear_history(); //avoid crashes
	memdelete(signal_editor);
	memdelete(variable_editor);
}

static ScriptEditorBase *create_editor(const Ref<Script> &p_script) {

	if (Object::cast_to<VisualScript>(*p_script)) {
		return memnew(VisualScriptEditor);
	}

	return NULL;
}

VisualScriptEditor::Clipboard *VisualScriptEditor::clipboard = NULL;

void VisualScriptEditor::free_clipboard() {
	if (clipboard)
		memdelete(clipboard);
}

static void register_editor_callback() {

	ScriptEditor::register_create_script_editor_function(create_editor);

	ED_SHORTCUT("visual_script_editor/delete_selected", TTR("Delete Selected"));
	ED_SHORTCUT("visual_script_editor/toggle_breakpoint", TTR("Toggle Breakpoint"), KEY_F9);
	ED_SHORTCUT("visual_script_editor/find_node_type", TTR("Find Node Type"), KEY_MASK_CMD + KEY_F);
	ED_SHORTCUT("visual_script_editor/copy_nodes", TTR("Copy Nodes"), KEY_MASK_CMD + KEY_C);
	ED_SHORTCUT("visual_script_editor/cut_nodes", TTR("Cut Nodes"), KEY_MASK_CMD + KEY_X);
	ED_SHORTCUT("visual_script_editor/paste_nodes", TTR("Paste Nodes"), KEY_MASK_CMD + KEY_V);
}

void VisualScriptEditor::register_editor() {

	//too early to register stuff here, request a callback
	EditorNode::add_plugin_init_callback(register_editor_callback);
}

#endif
