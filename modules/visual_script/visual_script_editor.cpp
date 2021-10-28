/*************************************************************************/
/*  visual_script_editor.cpp                                             */
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

#include "visual_script_editor.h"

#include "core/object.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/script_language.h"
#include "core/variant.h"
#include "editor/editor_node.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_scale.h"
#include "scene/main/viewport.h"
#include "visual_script_expression.h"
#include "visual_script_flow_control.h"
#include "visual_script_func_nodes.h"
#include "visual_script_nodes.h"

#ifdef TOOLS_ENABLED
class VisualScriptEditorSignalEdit : public Object {
	GDCLASS(VisualScriptEditorSignalEdit, Object);

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
		if (sig == StringName()) {
			return false;
		}

		if (p_name == "argument_count") {
			int new_argc = p_value;
			int argc = script->custom_signal_get_argument_count(sig);
			if (argc == new_argc) {
				return true;
			}

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
		if (sig == StringName()) {
			return false;
		}

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
		if (sig == StringName()) {
			return;
		}

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

	VisualScriptEditorSignalEdit() { undo_redo = nullptr; }
};

class VisualScriptEditorVariableEdit : public Object {
	GDCLASS(VisualScriptEditorVariableEdit, Object);

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
		if (var == StringName()) {
			return false;
		}

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
			Dictionary dc = d.duplicate();
			dc["type"] = p_value;
			undo_redo->create_action(TTR("Set Variable Type"));
			undo_redo->add_do_method(script.ptr(), "set_variable_info", var, dc);
			undo_redo->add_undo_method(script.ptr(), "set_variable_info", var, d);

			// Setting the default value.
			Variant::Type type = (Variant::Type)(int)p_value;
			if (type != Variant::NIL) {
				Variant default_value;
				Variant::CallError ce;
				default_value = Variant::construct(type, nullptr, 0, ce);
				if (ce.error == Variant::CallError::CALL_OK) {
					undo_redo->add_do_method(script.ptr(), "set_variable_default_value", var, default_value);
					undo_redo->add_undo_method(script.ptr(), "set_variable_default_value", var, dc["value"]);
				}
			}

			undo_redo->add_do_method(this, "_var_changed");
			undo_redo->add_undo_method(this, "_var_changed");
			undo_redo->commit_action();
			return true;
		}

		if (String(p_name) == "hint") {
			Dictionary dc = d.duplicate();
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
			Dictionary dc = d.duplicate();
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
			EditorNode::get_singleton()->get_inspector()->update_tree();
			return true;
		}

		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (var == StringName()) {
			return false;
		}

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
		if (var == StringName()) {
			return;
		}

		String argt = "Variant";
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			argt += "," + Variant::get_type_name(Variant::Type(i));
		}
		p_list->push_back(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt));
		p_list->push_back(PropertyInfo(script->get_variable_info(var).type, "value", script->get_variable_info(var).hint, script->get_variable_info(var).hint_string, PROPERTY_USAGE_DEFAULT));
		// Update this when PropertyHint changes
		p_list->push_back(PropertyInfo(Variant::INT, "hint", PROPERTY_HINT_ENUM, "None,Range,ExpRange,Enum,ExpEasing,Length,SpriteFrame,KeyAccel,Flags,Layers2dRender,Layers2dPhysics,Layer3dRender,Layer3dPhysics,File,Dir,GlobalFile,GlobalDir,ResourceType,MultilineText,PlaceholderText,ColorNoAlpha,ImageCompressLossy,ImageCompressLossLess,ObjectId,TypeString,NodePathToEditedNode,MethodOfVariantType,MethodOfBaseType,MethodOfInstance,MethodOfScript,PropertyOfVariantType,PropertyOfBaseType,PropertyOfInstance,PropertyOfScript,ObjectTooBig,NodePathValidTypes,SaveFile"));
		p_list->push_back(PropertyInfo(Variant::STRING, "hint_string"));
		p_list->push_back(PropertyInfo(Variant::BOOL, "export"));
	}

public:
	void edit(const StringName &p_var) {
		var = p_var;
		_change_notify();
	}

	VisualScriptEditorVariableEdit() { undo_redo = nullptr; }
};

static Color _color_from_type(Variant::Type p_type, bool dark_theme = true) {
	Color color;
	if (dark_theme) {
		switch (p_type) {
			case Variant::NIL:
				color = Color(0.41, 0.93, 0.74);
				break;

			case Variant::BOOL:
				color = Color(0.55, 0.65, 0.94);
				break;
			case Variant::INT:
				color = Color(0.49, 0.78, 0.94);
				break;
			case Variant::REAL:
				color = Color(0.38, 0.85, 0.96);
				break;
			case Variant::STRING:
				color = Color(0.42, 0.65, 0.93);
				break;

			case Variant::VECTOR2:
				color = Color(0.74, 0.57, 0.95);
				break;
			case Variant::RECT2:
				color = Color(0.95, 0.57, 0.65);
				break;
			case Variant::VECTOR3:
				color = Color(0.84, 0.49, 0.93);
				break;
			case Variant::TRANSFORM2D:
				color = Color(0.77, 0.93, 0.41);
				break;
			case Variant::PLANE:
				color = Color(0.97, 0.44, 0.44);
				break;
			case Variant::QUAT:
				color = Color(0.93, 0.41, 0.64);
				break;
			case Variant::AABB:
				color = Color(0.93, 0.47, 0.57);
				break;
			case Variant::BASIS:
				color = Color(0.89, 0.93, 0.41);
				break;
			case Variant::TRANSFORM:
				color = Color(0.96, 0.66, 0.43);
				break;

			case Variant::COLOR:
				color = Color(0.62, 1.0, 0.44);
				break;
			case Variant::NODE_PATH:
				color = Color(0.41, 0.58, 0.93);
				break;
			case Variant::_RID:
				color = Color(0.41, 0.93, 0.6);
				break;
			case Variant::OBJECT:
				color = Color(0.47, 0.95, 0.91);
				break;
			case Variant::DICTIONARY:
				color = Color(0.47, 0.93, 0.69);
				break;

			case Variant::ARRAY:
				color = Color(0.88, 0.88, 0.88);
				break;
			case Variant::POOL_BYTE_ARRAY:
				color = Color(0.67, 0.96, 0.78);
				break;
			case Variant::POOL_INT_ARRAY:
				color = Color(0.69, 0.86, 0.96);
				break;
			case Variant::POOL_REAL_ARRAY:
				color = Color(0.59, 0.91, 0.97);
				break;
			case Variant::POOL_STRING_ARRAY:
				color = Color(0.62, 0.77, 0.95);
				break;
			case Variant::POOL_VECTOR2_ARRAY:
				color = Color(0.82, 0.7, 0.96);
				break;
			case Variant::POOL_VECTOR3_ARRAY:
				color = Color(0.87, 0.61, 0.95);
				break;
			case Variant::POOL_COLOR_ARRAY:
				color = Color(0.91, 1.0, 0.59);
				break;

			default:
				color.set_hsv(p_type / float(Variant::VARIANT_MAX), 0.7, 0.7);
		}
	} else {
		switch (p_type) {
			case Variant::NIL:
				color = Color(0.15, 0.89, 0.63);
				break;

			case Variant::BOOL:
				color = Color(0.43, 0.56, 0.92);
				break;
			case Variant::INT:
				color = Color(0.31, 0.7, 0.91);
				break;
			case Variant::REAL:
				color = Color(0.15, 0.8, 0.94);
				break;
			case Variant::STRING:
				color = Color(0.27, 0.56, 0.91);
				break;

			case Variant::VECTOR2:
				color = Color(0.68, 0.46, 0.93);
				break;
			case Variant::RECT2:
				color = Color(0.93, 0.46, 0.56);
				break;
			case Variant::VECTOR3:
				color = Color(0.86, 0.42, 0.93);
				break;
			case Variant::TRANSFORM2D:
				color = Color(0.59, 0.81, 0.1);
				break;
			case Variant::PLANE:
				color = Color(0.97, 0.44, 0.44);
				break;
			case Variant::QUAT:
				color = Color(0.93, 0.41, 0.64);
				break;
			case Variant::AABB:
				color = Color(0.93, 0.47, 0.57);
				break;
			case Variant::BASIS:
				color = Color(0.7, 0.73, 0.1);
				break;
			case Variant::TRANSFORM:
				color = Color(0.96, 0.56, 0.28);
				break;

			case Variant::COLOR:
				color = Color(0.24, 0.75, 0.0);
				break;
			case Variant::NODE_PATH:
				color = Color(0.41, 0.58, 0.93);
				break;
			case Variant::_RID:
				color = Color(0.17, 0.9, 0.45);
				break;
			case Variant::OBJECT:
				color = Color(0.07, 0.84, 0.76);
				break;
			case Variant::DICTIONARY:
				color = Color(0.34, 0.91, 0.62);
				break;

			case Variant::ARRAY:
				color = Color(0.45, 0.45, 0.45);
				break;
			case Variant::POOL_BYTE_ARRAY:
				color = Color(0.38, 0.92, 0.6);
				break;
			case Variant::POOL_INT_ARRAY:
				color = Color(0.38, 0.73, 0.92);
				break;
			case Variant::POOL_REAL_ARRAY:
				color = Color(0.25, 0.83, 0.95);
				break;
			case Variant::POOL_STRING_ARRAY:
				color = Color(0.38, 0.62, 0.92);
				break;
			case Variant::POOL_VECTOR2_ARRAY:
				color = Color(0.62, 0.36, 0.92);
				break;
			case Variant::POOL_VECTOR3_ARRAY:
				color = Color(0.79, 0.35, 0.92);
				break;
			case Variant::POOL_COLOR_ARRAY:
				color = Color(0.57, 0.73, 0.0);
				break;

			default:
				color.set_hsv(p_type / float(Variant::VARIANT_MAX), 0.3, 0.3);
		}
	}

	return color;
}

void VisualScriptEditor::_update_graph_connections() {
	graph->clear_connections();

	List<StringName> funcs;
	script->get_function_list(&funcs);

	if (funcs.size() <= 0) {
		updating_graph = false;
		return;
	}

	for (List<StringName>::Element *F = funcs.front(); F; F = F->next()) {
		List<VisualScript::SequenceConnection> sequence_conns;
		script->get_sequence_connection_list(F->get(), &sequence_conns);

		for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {
			graph->connect_node(itos(E->get().from_node), E->get().from_output, itos(E->get().to_node), 0);
		}

		List<VisualScript::DataConnection> data_conns;
		script->get_data_connection_list(F->get(), &data_conns);

		for (List<VisualScript::DataConnection>::Element *E = data_conns.front(); E; E = E->next()) {
			VisualScript::DataConnection dc = E->get();

			Ref<VisualScriptNode> from_node = script->get_node(F->get(), E->get().from_node);
			Ref<VisualScriptNode> to_node = script->get_node(F->get(), E->get().to_node);

			if (to_node->has_input_sequence_port()) {
				dc.to_port++;
			}

			dc.from_port += from_node->get_output_sequence_port_count();

			graph->connect_node(itos(E->get().from_node), dc.from_port, itos(E->get().to_node), dc.to_port);
		}
	}
}

void VisualScriptEditor::_update_graph(int p_only_id) {
	if (updating_graph) {
		return;
	}

	updating_graph = true;

	//byebye all nodes
	if (p_only_id >= 0) {
		if (graph->has_node(itos(p_only_id))) {
			Node *gid = graph->get_node(itos(p_only_id));
			if (gid) {
				memdelete(gid);
			}
		}
	} else {
		for (int i = 0; i < graph->get_child_count(); i++) {
			if (Object::cast_to<GraphNode>(graph->get_child(i))) {
				memdelete(graph->get_child(i));
				i--;
			}
		}
	}

	List<StringName> funcs;
	script->get_function_list(&funcs);

	if (funcs.size() <= 0) {
		graph->hide();
		select_func_text->show();
		updating_graph = false;
		return;
	}

	graph->show();
	select_func_text->hide();

	Ref<Texture> type_icons[Variant::VARIANT_MAX] = {
		Control::get_icon("Variant", "EditorIcons"),
		Control::get_icon("bool", "EditorIcons"),
		Control::get_icon("int", "EditorIcons"),
		Control::get_icon("float", "EditorIcons"),
		Control::get_icon("String", "EditorIcons"),
		Control::get_icon("Vector2", "EditorIcons"),
		Control::get_icon("Rect2", "EditorIcons"),
		Control::get_icon("Vector3", "EditorIcons"),
		Control::get_icon("Transform2D", "EditorIcons"),
		Control::get_icon("Plane", "EditorIcons"),
		Control::get_icon("Quat", "EditorIcons"),
		Control::get_icon("AABB", "EditorIcons"),
		Control::get_icon("Basis", "EditorIcons"),
		Control::get_icon("Transform", "EditorIcons"),
		Control::get_icon("Color", "EditorIcons"),
		Control::get_icon("NodePath", "EditorIcons"),
		Control::get_icon("RID", "EditorIcons"),
		Control::get_icon("MiniObject", "EditorIcons"),
		Control::get_icon("Dictionary", "EditorIcons"),
		Control::get_icon("Array", "EditorIcons"),
		Control::get_icon("PoolByteArray", "EditorIcons"),
		Control::get_icon("PoolIntArray", "EditorIcons"),
		Control::get_icon("PoolRealArray", "EditorIcons"),
		Control::get_icon("PoolStringArray", "EditorIcons"),
		Control::get_icon("PoolVector2Array", "EditorIcons"),
		Control::get_icon("PoolVector3Array", "EditorIcons"),
		Control::get_icon("PoolColorArray", "EditorIcons")
	};

	Ref<Texture> seq_port = Control::get_icon("VisualShaderPort", "EditorIcons");

	for (List<StringName>::Element *F = funcs.front(); F; F = F->next()) { // loop through all the functions

		List<int> ids;
		script->get_node_list(F->get(), &ids);
		StringName editor_icons = "EditorIcons";

		for (List<int>::Element *E = ids.front(); E; E = E->next()) {
			if (p_only_id >= 0 && p_only_id != E->get()) {
				continue;
			}

			Ref<VisualScriptNode> node = script->get_node(F->get(), E->get());
			Vector2 pos = script->get_node_position(F->get(), E->get());

			GraphNode *gnode = memnew(GraphNode);
			gnode->set_title(node->get_caption());
			gnode->set_offset(pos * EDSCALE);
			if (error_line == E->get()) {
				gnode->set_overlay(GraphNode::OVERLAY_POSITION);
			} else if (node->is_breakpoint()) {
				gnode->set_overlay(GraphNode::OVERLAY_BREAKPOINT);
			}

			gnode->set_meta("__vnode", node);
			gnode->set_name(itos(E->get()));
			gnode->connect("dragged", this, "_node_moved", varray(E->get()));
			gnode->connect("close_request", this, "_remove_node", varray(E->get()), CONNECT_DEFERRED);

			if (E->get() != script->get_function_node_id(F->get())) {
				//function can't be erased
				gnode->set_show_close_button(true);
			}

			bool has_gnode_text = false;

			Ref<VisualScriptLists> nd_list = node;
			bool is_vslist = nd_list.is_valid();
			if (is_vslist) {
				HBoxContainer *hbnc = memnew(HBoxContainer);
				if (nd_list->is_input_port_editable()) {
					has_gnode_text = true;
					Button *btn = memnew(Button);
					btn->set_text(TTR("Add Input Port"));
					hbnc->add_child(btn);
					btn->connect("pressed", this, "_add_input_port", varray(E->get()), CONNECT_DEFERRED);
				}
				if (nd_list->is_output_port_editable()) {
					if (nd_list->is_input_port_editable()) {
						hbnc->add_spacer();
					}
					has_gnode_text = true;
					Button *btn = memnew(Button);
					btn->set_text(TTR("Add Output Port"));
					hbnc->add_child(btn);
					btn->connect("pressed", this, "_add_output_port", varray(E->get()), CONNECT_DEFERRED);
				}
				gnode->add_child(hbnc);
			} else if (Object::cast_to<VisualScriptExpression>(node.ptr())) {
				has_gnode_text = true;
				LineEdit *line_edit = memnew(LineEdit);
				line_edit->set_text(node->get_text());
				line_edit->set_expand_to_text_length(true);
				line_edit->add_font_override("font", get_font("source", "EditorFonts"));
				gnode->add_child(line_edit);
				line_edit->connect("text_changed", this, "_expression_text_changed", varray(E->get()));
			} else {
				String text = node->get_text();
				if (!text.empty()) {
					has_gnode_text = true;
					Label *label = memnew(Label);
					label->set_text(text);
					gnode->add_child(label);
				}
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
				if (gnode->is_comment()) {
					sbf = EditorNode::get_singleton()->get_theme_base()->get_theme()->get_stylebox("comment", "GraphNode");
				}

				Color c = sbf->get_border_color();
				c.a = 1;
				if (EditorSettings::get_singleton()->get("interface/theme/use_graph_node_headers")) {
					Color mono_color = ((c.r + c.g + c.b) / 3) < 0.7 ? Color(1.0, 1.0, 1.0) : Color(0.0, 0.0, 0.0);
					mono_color.a = 0.85;
					c = mono_color;
				}
				gnode->add_color_override("title_color", c);
				c.a = 0.7;
				gnode->add_color_override("close_color", c);
				gnode->add_color_override("resizer_color", c);
				gnode->add_style_override("frame", sbf);
			}

			const Color mono_color = get_color("mono_color", "Editor");

			int slot_idx = 0;

			bool single_seq_output = node->get_output_sequence_port_count() == 1 && node->get_output_sequence_port_text(0) == String();
			if ((node->has_input_sequence_port() || single_seq_output) || has_gnode_text) {
				// IF has_gnode_text is true BUT we have no sequence ports to draw (in here),
				// we still draw the disabled default ones to shift up the slots by one,
				// so the slots DON'T start with the content text.

				// IF has_gnode_text is false, but we DO want to draw default sequence ports,
				// we draw a dummy text to take up the position of the sequence nodes, so all the other ports are still aligned correctly.
				if (!has_gnode_text) {
					Label *dummy = memnew(Label);
					dummy->set_text(" ");
					gnode->add_child(dummy);
				}
				gnode->set_slot(0, node->has_input_sequence_port(), TYPE_SEQUENCE, mono_color, single_seq_output, TYPE_SEQUENCE, mono_color, seq_port, seq_port);
				slot_idx++;
			}

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
				VBoxContainer *vbc = memnew(VBoxContainer);
				HBoxContainer *hbc = memnew(HBoxContainer);
				HBoxContainer *hbc2 = memnew(HBoxContainer);
				vbc->add_child(hbc);
				vbc->add_child(hbc2);
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

					if (is_vslist) {
						if (nd_list->is_input_port_name_editable()) {
							LineEdit *name_box = memnew(LineEdit);
							hbc->add_child(name_box);
							name_box->set_custom_minimum_size(Size2(60 * EDSCALE, 0));
							name_box->set_text(left_name);
							name_box->set_expand_to_text_length(true);
							name_box->connect("resized", this, "_update_node_size", varray(E->get()));
							name_box->connect("focus_exited", this, "_port_name_focus_out", varray(name_box, E->get(), i, true));
						} else {
							hbc->add_child(memnew(Label(left_name)));
						}

						if (nd_list->is_input_port_type_editable()) {
							OptionButton *opbtn = memnew(OptionButton);
							for (int j = Variant::NIL; j < Variant::VARIANT_MAX; j++) {
								opbtn->add_item(Variant::get_type_name(Variant::Type(j)));
							}
							opbtn->select(left_type);
							opbtn->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
							hbc->add_child(opbtn);
							opbtn->connect("item_selected", this, "_change_port_type", varray(E->get(), i, true), CONNECT_DEFERRED);
						}

						Button *rmbtn = memnew(Button);
						rmbtn->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Remove", "EditorIcons"));
						hbc->add_child(rmbtn);
						rmbtn->connect("pressed", this, "_remove_input_port", varray(E->get(), i), CONNECT_DEFERRED);
					} else {
						hbc->add_child(memnew(Label(left_name)));
					}

					if (left_type != Variant::NIL && !script->is_input_value_port_connected(F->get(), E->get(), i)) {
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
						hbc2->add_child(button);
					}
				} else {
					Control *c = memnew(Control);
					c->set_custom_minimum_size(Size2(10, 0) * EDSCALE);
					hbc->add_child(c);
				}

				hbc->add_spacer();
				hbc2->add_spacer();

				if (i < mixed_seq_ports) {
					Label *text2 = memnew(Label);
					text2->set_text(node->get_output_sequence_port_text(i));
					text2->set_align(Label::ALIGN_RIGHT);
					hbc->add_child(text2);
				}

				if (right_ok) {
					if (is_vslist) {
						Button *rmbtn = memnew(Button);
						rmbtn->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Remove", "EditorIcons"));
						hbc->add_child(rmbtn);
						rmbtn->connect("pressed", this, "_remove_output_port", varray(E->get(), i), CONNECT_DEFERRED);

						if (nd_list->is_output_port_type_editable()) {
							OptionButton *opbtn = memnew(OptionButton);
							for (int j = Variant::NIL; j < Variant::VARIANT_MAX; j++) {
								opbtn->add_item(Variant::get_type_name(Variant::Type(j)));
							}
							opbtn->select(right_type);
							opbtn->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
							hbc->add_child(opbtn);
							opbtn->connect("item_selected", this, "_change_port_type", varray(E->get(), i, false), CONNECT_DEFERRED);
						}

						if (nd_list->is_output_port_name_editable()) {
							LineEdit *name_box = memnew(LineEdit);
							hbc->add_child(name_box);
							name_box->set_custom_minimum_size(Size2(60 * EDSCALE, 0));
							name_box->set_text(right_name);
							name_box->set_expand_to_text_length(true);
							name_box->connect("resized", this, "_update_node_size", varray(E->get()));
							name_box->connect("focus_exited", this, "_port_name_focus_out", varray(name_box, E->get(), i, false));
						} else {
							hbc->add_child(memnew(Label(right_name)));
						}
					} else {
						hbc->add_child(memnew(Label(right_name)));
					}

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

				gnode->add_child(vbc);

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
	}
	_update_graph_connections();

	float graph_minimap_opacity = EditorSettings::get_singleton()->get("editors/visual_editors/minimap_opacity");
	graph->set_minimap_opacity(graph_minimap_opacity);

	// use default_func instead of default_func for now I think that should be good stop gap solution to ensure not breaking anything
	graph->call_deferred("set_scroll_ofs", script->get_function_scroll(default_func) * EDSCALE);
	updating_graph = false;
}

void VisualScriptEditor::_change_port_type(int p_select, int p_id, int p_port, bool is_input) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptLists> vsn = script->get_node(func, p_id);
	if (!vsn.is_valid()) {
		return;
	}

	undo_redo->create_action(TTR("Change Port Type"));
	if (is_input) {
		undo_redo->add_do_method(vsn.ptr(), "set_input_data_port_type", p_port, Variant::Type(p_select));
		undo_redo->add_undo_method(vsn.ptr(), "set_input_data_port_type", p_port, vsn->get_input_value_port_info(p_port).type);
	} else {
		undo_redo->add_do_method(vsn.ptr(), "set_output_data_port_type", p_port, Variant::Type(p_select));
		undo_redo->add_undo_method(vsn.ptr(), "set_output_data_port_type", p_port, vsn->get_output_value_port_info(p_port).type);
	}
	undo_redo->commit_action();
}

void VisualScriptEditor::_update_node_size(int p_id) {
	Node *node = graph->get_node(itos(p_id));
	if (Object::cast_to<Control>(node)) {
		Object::cast_to<Control>(node)->set_size(Vector2(1, 1)); //shrink if text is smaller
	}
}
void VisualScriptEditor::_port_name_focus_out(const Node *p_name_box, int p_id, int p_port, bool is_input) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptLists> vsn = script->get_node(func, p_id);
	if (!vsn.is_valid()) {
		return;
	}

	String text;

	if (Object::cast_to<LineEdit>(p_name_box)) {
		text = Object::cast_to<LineEdit>(p_name_box)->get_text();
	} else {
		return;
	}

	undo_redo->create_action(TTR("Change Port Name"));
	if (is_input) {
		undo_redo->add_do_method(vsn.ptr(), "set_input_data_port_name", p_port, text);
		undo_redo->add_undo_method(vsn.ptr(), "set_input_data_port_name", p_port, vsn->get_input_value_port_info(p_port).name);
	} else {
		undo_redo->add_do_method(vsn.ptr(), "set_output_data_port_name", p_port, text);
		undo_redo->add_undo_method(vsn.ptr(), "set_output_data_port_name", p_port, vsn->get_output_value_port_info(p_port).name);
	}
	undo_redo->commit_action();
}

void VisualScriptEditor::_update_members() {
	ERR_FAIL_COND(!script.is_valid());

	updating_members = true;

	members->clear();
	TreeItem *root = members->create_item();

	TreeItem *functions = members->create_item(root);
	functions->set_selectable(0, false);
	functions->set_text(0, TTR("Functions:"));
	functions->add_button(0, Control::get_icon("Override", "EditorIcons"), 1, false, TTR("Override an existing built-in function."));
	functions->add_button(0, Control::get_icon("Add", "EditorIcons"), 0, false, TTR("Create a new function."));
	functions->set_custom_color(0, Control::get_color("mono_color", "Editor"));

	List<StringName> func_names;
	script->get_function_list(&func_names);
	for (List<StringName>::Element *E = func_names.front(); E; E = E->next()) {
		if (E->get() == default_func) {
			continue;
		}

		TreeItem *ti = members->create_item(functions);
		ti->set_text(0, E->get());
		ti->set_selectable(0, true);
		ti->set_metadata(0, E->get());
		ti->add_button(0, Control::get_icon("Edit", "EditorIcons"), 0);
		if (selected == E->get()) {
			ti->select(0);
		}
	}

	TreeItem *variables = members->create_item(root);
	variables->set_selectable(0, false);
	variables->set_text(0, TTR("Variables:"));
	variables->add_button(0, Control::get_icon("Add", "EditorIcons"), -1, false, TTR("Create a new variable."));
	variables->set_custom_color(0, Control::get_color("mono_color", "Editor"));

	Ref<Texture> type_icons[Variant::VARIANT_MAX] = {
		Control::get_icon("Variant", "EditorIcons"),
		Control::get_icon("bool", "EditorIcons"),
		Control::get_icon("int", "EditorIcons"),
		Control::get_icon("float", "EditorIcons"),
		Control::get_icon("String", "EditorIcons"),
		Control::get_icon("Vector2", "EditorIcons"),
		Control::get_icon("Rect2", "EditorIcons"),
		Control::get_icon("Vector3", "EditorIcons"),
		Control::get_icon("Transform2D", "EditorIcons"),
		Control::get_icon("Plane", "EditorIcons"),
		Control::get_icon("Quat", "EditorIcons"),
		Control::get_icon("AABB", "EditorIcons"),
		Control::get_icon("Basis", "EditorIcons"),
		Control::get_icon("Transform", "EditorIcons"),
		Control::get_icon("Color", "EditorIcons"),
		Control::get_icon("NodePath", "EditorIcons"),
		Control::get_icon("RID", "EditorIcons"),
		Control::get_icon("MiniObject", "EditorIcons"),
		Control::get_icon("Dictionary", "EditorIcons"),
		Control::get_icon("Array", "EditorIcons"),
		Control::get_icon("PoolByteArray", "EditorIcons"),
		Control::get_icon("PoolIntArray", "EditorIcons"),
		Control::get_icon("PoolRealArray", "EditorIcons"),
		Control::get_icon("PoolStringArray", "EditorIcons"),
		Control::get_icon("PoolVector2Array", "EditorIcons"),
		Control::get_icon("PoolVector3Array", "EditorIcons"),
		Control::get_icon("PoolColorArray", "EditorIcons")
	};

	List<StringName> var_names;
	script->get_variable_list(&var_names);
	for (List<StringName>::Element *E = var_names.front(); E; E = E->next()) {
		TreeItem *ti = members->create_item(variables);

		ti->set_text(0, E->get());

		ti->set_suffix(0, "= " + _sanitized_variant_text(E->get()));
		ti->set_icon(0, type_icons[script->get_variable_info(E->get()).type]);

		ti->set_selectable(0, true);
		ti->set_editable(0, true);
		ti->set_metadata(0, E->get());
		if (selected == E->get()) {
			ti->select(0);
		}
	}

	TreeItem *_signals = members->create_item(root);
	_signals->set_selectable(0, false);
	_signals->set_text(0, TTR("Signals:"));
	_signals->add_button(0, Control::get_icon("Add", "EditorIcons"), -1, false, TTR("Create a new signal."));
	_signals->set_custom_color(0, Control::get_color("mono_color", "Editor"));

	List<StringName> signal_names;
	script->get_custom_signal_list(&signal_names);
	for (List<StringName>::Element *E = signal_names.front(); E; E = E->next()) {
		TreeItem *ti = members->create_item(_signals);
		ti->set_text(0, E->get());
		ti->set_selectable(0, true);
		ti->set_editable(0, true);
		ti->set_metadata(0, E->get());
		if (selected == E->get()) {
			ti->select(0);
		}
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

String VisualScriptEditor::_sanitized_variant_text(const StringName &property_name) {
	Variant var = script->get_variable_default_value(property_name);

	if (script->get_variable_info(property_name).type != Variant::NIL) {
		Variant::CallError ce;
		const Variant *converted = &var;
		var = Variant::construct(script->get_variable_info(property_name).type, &converted, 1, ce, false);
	}

	return String(var);
}

void VisualScriptEditor::_member_selected() {
	if (updating_members) {
		return;
	}

	TreeItem *ti = members->get_selected();
	ERR_FAIL_COND(!ti);

	selected = ti->get_metadata(0);

	if (ti->get_parent() == members->get_root()->get_children()) {
#ifdef OSX_ENABLED
		bool held_ctrl = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool held_ctrl = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif
		if (held_ctrl) {
			ERR_FAIL_COND(!script->has_function(selected));
			_center_on_node(selected, script->get_function_node_id(selected));
		}
	}
}

void VisualScriptEditor::_member_edited() {
	if (updating_members) {
		return;
	}

	TreeItem *ti = members->get_edited();
	ERR_FAIL_COND(!ti);

	String name = ti->get_metadata(0);
	String new_name = ti->get_text(0);

	if (name == new_name) {
		return;
	}

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

		// also fix all function calls
		List<StringName> flst;
		script->get_function_list(&flst);
		for (List<StringName>::Element *E = flst.front(); E; E = E->next()) {
			List<int> lst;
			script->get_node_list(E->get(), &lst);
			for (List<int>::Element *F = lst.front(); F; F = F->next()) {
				Ref<VisualScriptFunctionCall> fncall = script->get_node(E->get(), F->get());
				if (!fncall.is_valid()) {
					continue;
				}
				if (fncall->get_function() == name) {
					undo_redo->add_do_method(fncall.ptr(), "set_function", new_name);
					undo_redo->add_undo_method(fncall.ptr(), "set_function", name);
				}
			}
		}

		undo_redo->add_do_method(this, "_update_members");
		undo_redo->add_undo_method(this, "_update_members");
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
		undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
		undo_redo->commit_action();

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

void VisualScriptEditor::_create_function_dialog() {
	function_create_dialog->popup_centered();
	func_name_box->set_text("");
	func_name_box->grab_focus();
	for (int i = 0; i < func_input_vbox->get_child_count(); i++) {
		Node *nd = func_input_vbox->get_child(i);
		nd->queue_delete();
	}
}

void VisualScriptEditor::_create_function() {
	String name = _validate_name((func_name_box->get_text() == "") ? "new_func" : func_name_box->get_text());
	selected = name;
	Vector2 pos = _get_available_pos();

	Ref<VisualScriptFunction> func_node;
	func_node.instance();
	func_node->set_name(name);

	for (int i = 0; i < func_input_vbox->get_child_count(); i++) {
		OptionButton *opbtn = Object::cast_to<OptionButton>(func_input_vbox->get_child(i)->get_child(3));
		LineEdit *lne = Object::cast_to<LineEdit>(func_input_vbox->get_child(i)->get_child(1));
		if (!opbtn || !lne) {
			continue;
		}
		Variant::Type arg_type = Variant::Type(opbtn->get_selected());
		String arg_name = lne->get_text();
		func_node->add_argument(arg_type, arg_name);
	}

	undo_redo->create_action(TTR("Add Function"));
	undo_redo->add_do_method(script.ptr(), "add_function", name);
	undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id(), func_node, pos);
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

void VisualScriptEditor::_add_node_dialog() {
	_generic_search(script->get_instance_base_type(), graph->get_global_position() + Vector2(55, 80), true);
}

void VisualScriptEditor::_add_func_input() {
	HBoxContainer *hbox = memnew(HBoxContainer);
	hbox->set_h_size_flags(SIZE_EXPAND_FILL);

	Label *name_label = memnew(Label);
	name_label->set_text(TTR("Name:"));
	hbox->add_child(name_label);

	LineEdit *name_box = memnew(LineEdit);
	name_box->set_h_size_flags(SIZE_EXPAND_FILL);
	name_box->set_text("input");
	name_box->connect("focus_entered", this, "_deselect_input_names");
	hbox->add_child(name_box);

	Label *type_label = memnew(Label);
	type_label->set_text(TTR("Type:"));
	hbox->add_child(type_label);

	OptionButton *type_box = memnew(OptionButton);
	type_box->set_custom_minimum_size(Size2(120 * EDSCALE, 0));
	for (int i = Variant::NIL; i < Variant::VARIANT_MAX; i++) {
		type_box->add_item(Variant::get_type_name(Variant::Type(i)));
	}
	type_box->select(1);
	hbox->add_child(type_box);

	Button *delete_button = memnew(Button);
	delete_button->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Remove", "EditorIcons"));
	delete_button->set_tooltip(vformat(TTR("Delete input port")));
	hbox->add_child(delete_button);

	for (int i = 0; i < func_input_vbox->get_child_count(); i++) {
		LineEdit *line_edit = (LineEdit *)func_input_vbox->get_child(i)->get_child(1);
		line_edit->deselect();
	}

	func_input_vbox->add_child(hbox);
	hbox->set_meta("id", hbox->get_position_in_parent());

	delete_button->connect("pressed", this, "_remove_func_input", varray(hbox));

	name_box->select_all();
	name_box->grab_focus();
}

void VisualScriptEditor::_remove_func_input(Node *p_node) {
	func_input_vbox->remove_child(p_node);
	p_node->queue_delete();
}

void VisualScriptEditor::_deselect_input_names() {
	int cn = func_input_vbox->get_child_count();
	for (int i = 0; i < cn; i++) {
		LineEdit *lne = Object::cast_to<LineEdit>(func_input_vbox->get_child(i)->get_child(1));
		if (lne) {
			lne->deselect();
		}
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
				Vector2 pos = _get_available_pos();

				Ref<VisualScriptFunction> func_node;
				func_node.instance();
				func_node->set_name(name);

				undo_redo->create_action(TTR("Add Function"));
				undo_redo->add_do_method(script.ptr(), "add_function", name);
				undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id(), func_node, pos);
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
	} else if (ti->get_parent() == root->get_children()) {
		selected = ti->get_text(0);
		function_name_edit->set_position(Input::get_singleton()->get_mouse_position() - Vector2(60, -10));
		function_name_edit->popup();
		function_name_box->set_text(selected);
		function_name_box->select_all();
	}
}

void VisualScriptEditor::_add_input_port(int p_id) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptLists> vsn = script->get_node(func, p_id);
	if (!vsn.is_valid()) {
		return;
	}

	updating_graph = true;

	undo_redo->create_action(TTR("Add Input Port"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(vsn.ptr(), "add_input_data_port", Variant::NIL, "arg", -1);
	undo_redo->add_do_method(this, "_update_graph", p_id);

	undo_redo->add_undo_method(vsn.ptr(), "remove_input_data_port", vsn->get_input_value_port_count());
	undo_redo->add_undo_method(this, "_update_graph", p_id);

	updating_graph = false;

	undo_redo->commit_action();
}

void VisualScriptEditor::_add_output_port(int p_id) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptLists> vsn = script->get_node(func, p_id);
	if (!vsn.is_valid()) {
		return;
	}

	updating_graph = true;

	undo_redo->create_action(TTR("Add Output Port"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(vsn.ptr(), "add_output_data_port", Variant::NIL, "arg", -1);
	undo_redo->add_do_method(this, "_update_graph", p_id);

	undo_redo->add_undo_method(vsn.ptr(), "remove_output_data_port", vsn->get_output_value_port_count());
	undo_redo->add_undo_method(this, "_update_graph", p_id);

	updating_graph = false;

	undo_redo->commit_action();
}

void VisualScriptEditor::_remove_input_port(int p_id, int p_port) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptLists> vsn = script->get_node(func, p_id);
	if (!vsn.is_valid()) {
		return;
	}

	updating_graph = true;

	undo_redo->create_action(TTR("Remove Input Port"), UndoRedo::MERGE_ENDS);

	int conn_from = -1, conn_port = -1;
	script->get_input_value_port_connection_source(func, p_id, p_port, &conn_from, &conn_port);

	if (conn_from != -1) {
		undo_redo->add_do_method(script.ptr(), "data_disconnect", func, conn_from, conn_port, p_id, p_port);
	}

	undo_redo->add_do_method(vsn.ptr(), "remove_input_data_port", p_port);
	undo_redo->add_do_method(this, "_update_graph", p_id);

	if (conn_from != -1) {
		undo_redo->add_undo_method(script.ptr(), "data_connect", func, conn_from, conn_port, p_id, p_port);
	}

	undo_redo->add_undo_method(vsn.ptr(), "add_input_data_port", vsn->get_input_value_port_info(p_port).type, vsn->get_input_value_port_info(p_port).name, p_port);
	undo_redo->add_undo_method(this, "_update_graph", p_id);

	updating_graph = false;

	undo_redo->commit_action();
}

void VisualScriptEditor::_remove_output_port(int p_id, int p_port) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptLists> vsn = script->get_node(func, p_id);
	if (!vsn.is_valid()) {
		return;
	}

	updating_graph = true;

	undo_redo->create_action(TTR("Remove Output Port"), UndoRedo::MERGE_ENDS);

	List<VisualScript::DataConnection> data_connections;
	script->get_data_connection_list(func, &data_connections);

	HashMap<int, Set<int>> conn_map;
	for (const List<VisualScript::DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
		if (E->get().from_node == p_id && E->get().from_port == p_port) {
			// push into the connections map
			if (!conn_map.has(E->get().to_node)) {
				conn_map.set(E->get().to_node, Set<int>());
			}
			conn_map[E->get().to_node].insert(E->get().to_port);
		}
	}

	undo_redo->add_do_method(vsn.ptr(), "remove_output_data_port", p_port);
	undo_redo->add_do_method(this, "_update_graph", p_id);

	List<int> keys;
	conn_map.get_key_list(&keys);
	for (const List<int>::Element *E = keys.front(); E; E = E->next()) {
		for (const Set<int>::Element *F = conn_map[E->get()].front(); F; F = F->next()) {
			undo_redo->add_undo_method(script.ptr(), "data_connect", func, p_id, p_port, E->get(), F->get());
		}
	}

	undo_redo->add_undo_method(vsn.ptr(), "add_output_data_port", vsn->get_output_value_port_info(p_port).type, vsn->get_output_value_port_info(p_port).name, p_port);
	undo_redo->add_undo_method(this, "_update_graph", p_id);

	updating_graph = false;

	undo_redo->commit_action();
}

void VisualScriptEditor::_expression_text_changed(const String &p_text, int p_id) {
	StringName func = _get_function_of_node(p_id);

	Ref<VisualScriptExpression> vse = script->get_node(func, p_id);
	if (!vse.is_valid()) {
		return;
	}

	updating_graph = true;

	undo_redo->create_action(TTR("Change Expression"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_property(vse.ptr(), "expression", p_text);
	undo_redo->add_undo_property(vse.ptr(), "expression", vse->get("expression"));
	undo_redo->add_do_method(this, "_update_graph", p_id);
	undo_redo->add_undo_method(this, "_update_graph", p_id);
	undo_redo->commit_action();

	Node *node = graph->get_node(itos(p_id));
	if (Object::cast_to<Control>(node)) {
		Object::cast_to<Control>(node)->set_size(Vector2(1, 1)); //shrink if text is smaller
	}

	updating_graph = false;
}

Vector2 VisualScriptEditor::_get_pos_in_graph(Vector2 p_point) const {
	Vector2 pos = (graph->get_scroll_ofs() + p_point) / (graph->get_zoom() * EDSCALE);
	if (graph->is_using_snap()) {
		int snap = graph->get_snap();
		pos = pos.snapped(Vector2(snap, snap));
	}
	return pos;
}

Vector2 VisualScriptEditor::_get_available_pos(bool p_centered, Vector2 p_pos) const {
	if (p_centered) {
		p_pos = _get_pos_in_graph(graph->get_size() * 0.5);
	}

	while (true) {
		bool exists = false;
		List<StringName> all_fn;
		script->get_function_list(&all_fn);
		for (List<StringName>::Element *F = all_fn.front(); F; F = F->next()) {
			StringName curr_fn = F->get();
			List<int> existing;
			script->get_node_list(curr_fn, &existing);
			for (List<int>::Element *E = existing.front(); E; E = E->next()) {
				Point2 pos = script->get_node_position(curr_fn, E->get());
				if (pos.distance_to(p_pos) < 50) {
					p_pos += Vector2(graph->get_snap(), graph->get_snap());
					exists = true;
					break;
				}
			}
		}
		if (exists) {
			continue;
		}
		break;
	}

	return p_pos;
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
	// delete all the selected nodes

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

	undo_redo->create_action(TTR("Remove VisualScript Nodes"));

	for (List<int>::Element *F = to_erase.front(); F; F = F->next()) {
		int cr_node = F->get();

		StringName func = _get_function_of_node(cr_node);

		undo_redo->add_do_method(script.ptr(), "remove_node", func, cr_node);
		undo_redo->add_undo_method(script.ptr(), "add_node", func, cr_node, script->get_node(func, cr_node), script->get_node_position(func, cr_node));

		List<VisualScript::SequenceConnection> sequence_conns;
		script->get_sequence_connection_list(func, &sequence_conns);

		for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {
			if (E->get().from_node == cr_node || E->get().to_node == cr_node) {
				undo_redo->add_undo_method(script.ptr(), "sequence_connect", func, E->get().from_node, E->get().from_output, E->get().to_node);
			}
		}

		List<VisualScript::DataConnection> data_conns;
		script->get_data_connection_list(func, &data_conns);

		for (List<VisualScript::DataConnection>::Element *E = data_conns.front(); E; E = E->next()) {
			if (E->get().from_node == F->get() || E->get().to_node == F->get()) {
				undo_redo->add_undo_method(script.ptr(), "data_connect", func, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}
		}
	}
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();
}

void VisualScriptEditor::_on_nodes_duplicate() {
	Set<int> to_duplicate;
	List<StringName> funcs;

	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			if (gn->is_selected() && gn->is_close_button_visible()) {
				int id = gn->get_name().operator String().to_int();
				to_duplicate.insert(id);
				funcs.push_back(_get_function_of_node(id));
			}
		}
	}

	if (to_duplicate.empty()) {
		return;
	}

	undo_redo->create_action(TTR("Duplicate VisualScript Nodes"));
	int idc = script->get_available_id() + 1;

	Set<int> to_select;
	HashMap<int, int> remap;

	for (Set<int>::Element *F = to_duplicate.front(); F; F = F->next()) {
		// duplicate from the specific function but place it into the default func as it would lack the connections
		StringName func = _get_function_of_node(F->get());
		Ref<VisualScriptNode> node = script->get_node(func, F->get());

		Ref<VisualScriptNode> dupe = node->duplicate(true);

		int new_id = idc++;
		remap.set(F->get(), new_id);

		to_select.insert(new_id);
		undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, dupe, script->get_node_position(func, F->get()) + Vector2(20, 20));
		undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
	}

	for (List<StringName>::Element *F = funcs.front(); F; F = F->next()) {
		List<VisualScript::SequenceConnection> seqs;
		script->get_sequence_connection_list(F->get(), &seqs);
		for (List<VisualScript::SequenceConnection>::Element *E = seqs.front(); E; E = E->next()) {
			if (to_duplicate.has(E->get().from_node) && to_duplicate.has(E->get().to_node)) {
				undo_redo->add_do_method(script.ptr(), "sequence_connect", default_func, remap[E->get().from_node], E->get().from_output, remap[E->get().to_node]);
			}
		}

		List<VisualScript::DataConnection> data;
		script->get_data_connection_list(F->get(), &data);
		for (List<VisualScript::DataConnection>::Element *E = data.front(); E; E = E->next()) {
			if (to_duplicate.has(E->get().from_node) && to_duplicate.has(E->get().to_node)) {
				undo_redo->add_do_method(script.ptr(), "data_connect", default_func, remap[E->get().from_node], E->get().from_port, remap[E->get().to_node], E->get().to_port);
			}
		}
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
		EditorNode::get_singleton()->push_item(script->get_node(default_func, to_select.front()->get()).ptr());
	}
}

void VisualScriptEditor::_generic_search(String p_base_type, Vector2 pos, bool node_centered) {
	if (node_centered) {
		port_action_pos = graph->get_size() / 2.0f;
	} else {
		port_action_pos = graph->get_viewport()->get_mouse_position() - graph->get_global_position();
	}

	new_connect_node_select->select_from_visual_script(p_base_type, false, false); // neither connecting nor reset text

	// ensure that the dialog fits inside the graph
	Size2 bounds = graph->get_global_position() + graph->get_size() - new_connect_node_select->get_size();
	pos.x = pos.x > bounds.x ? bounds.x : pos.x;
	pos.y = pos.y > bounds.y ? bounds.y : pos.y;

	if (pos != Vector2()) {
		new_connect_node_select->set_position(pos);
	}
}

void VisualScriptEditor::_input(const Ref<InputEvent> &p_event) {
	// GUI input for VS Editor Plugin
	Ref<InputEventMouseButton> key = p_event;

	if (key.is_valid() && !key->is_pressed()) {
		mouse_up_position = Input::get_singleton()->get_mouse_position();
	}
}

void VisualScriptEditor::_graph_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> key = p_event;

	if (key.is_valid() && key->is_pressed() && key->get_button_mask() == BUTTON_RIGHT) {
		saved_position = graph->get_local_mouse_position();

		Point2 gpos = Input::get_singleton()->get_mouse_position();
		_generic_search(script->get_instance_base_type(), gpos);
	}
}

void VisualScriptEditor::_members_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key = p_event;
	if (key.is_valid() && key->is_pressed() && !key->is_echo()) {
		if (members->has_focus()) {
			TreeItem *ti = members->get_selected();
			if (ti) {
				TreeItem *root = members->get_root();
				if (ti->get_parent() == root->get_children()) {
					member_type = MEMBER_FUNCTION;
				}
				if (ti->get_parent() == root->get_children()->get_next()) {
					member_type = MEMBER_VARIABLE;
				}
				if (ti->get_parent() == root->get_children()->get_next()->get_next()) {
					member_type = MEMBER_SIGNAL;
				}
				member_name = ti->get_text(0);
			}
			if (ED_IS_SHORTCUT("visual_script_editor/delete_selected", p_event)) {
				_member_option(MEMBER_REMOVE);
			}
			if (ED_IS_SHORTCUT("visual_script_editor/edit_member", p_event)) {
				_member_option(MEMBER_EDIT);
			}
		}
	}

	Ref<InputEventMouseButton> btn = p_event;
	if (btn.is_valid() && btn->is_doubleclick()) {
		TreeItem *ti = members->get_selected();
		if (ti && ti->get_parent() == members->get_root()->get_children()) { // to check if it's a function
			_center_on_node(ti->get_metadata(0), script->get_function_node_id(ti->get_metadata(0)));
		}
	}
}

void VisualScriptEditor::_rename_function(const String &name, const String &new_name) {
	if (!new_name.is_valid_identifier()) {
		EditorNode::get_singleton()->show_warning(TTR("Name is not a valid identifier:") + " " + new_name);
		return;
	}

	if (script->has_function(new_name) || script->has_variable(new_name) || script->has_custom_signal(new_name)) {
		EditorNode::get_singleton()->show_warning(TTR("Name already in use by another func/var/signal:") + " " + new_name);
		return;
	}

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

	// also fix all function calls
	List<StringName> flst;
	script->get_function_list(&flst);
	for (List<StringName>::Element *E = flst.front(); E; E = E->next()) {
		List<int> lst;
		script->get_node_list(E->get(), &lst);
		for (List<int>::Element *F = lst.front(); F; F = F->next()) {
			Ref<VisualScriptFunctionCall> fncall = script->get_node(E->get(), F->get());
			if (!fncall.is_valid()) {
				continue;
			}
			if (fncall->get_function() == name) {
				undo_redo->add_do_method(fncall.ptr(), "set_function", new_name);
				undo_redo->add_undo_method(fncall.ptr(), "set_function", name);
			}
		}
	}

	undo_redo->add_do_method(this, "_update_members");
	undo_redo->add_undo_method(this, "_update_members");
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
	undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");
	undo_redo->commit_action();
}

void VisualScriptEditor::_fn_name_box_input(const Ref<InputEvent> &p_event) {
	if (!function_name_edit->is_visible()) {
		return;
	}

	Ref<InputEventKey> key = p_event;
	if (key.is_valid() && key->is_pressed() && key->get_scancode() == KEY_ENTER) {
		function_name_edit->hide();
		_rename_function(selected, function_name_box->get_text());
		function_name_box->clear();
	}
}

Variant VisualScriptEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (p_from == members) {
		TreeItem *it = members->get_item_at_position(p_point);
		if (!it) {
			return Variant();
		}

		String type = it->get_metadata(0);

		if (type == String()) {
			return Variant();
		}

		Dictionary dd;
		TreeItem *root = members->get_root();

		if (it->get_parent() == root->get_children()) {
			dd["type"] = "visual_script_function_drag";
			dd["function"] = type;
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
				const_cast<VisualScriptEditor *>(this)->_show_hint(vformat(TTR("Hold %s to drop a Getter. Hold Shift to drop a generic signature."), find_keycode_name(KEY_META)));
#else
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Ctrl to drop a Getter. Hold Shift to drop a generic signature."));
#endif
			}

			if (String(d["type"]) == "nodes") {
#ifdef OSX_ENABLED
				const_cast<VisualScriptEditor *>(this)->_show_hint(vformat(TTR("Hold %s to drop a simple reference to the node."), find_keycode_name(KEY_META)));
#else
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Ctrl to drop a simple reference to the node."));
#endif
			}

			if (String(d["type"]) == "visual_script_variable_drag") {
#ifdef OSX_ENABLED
				const_cast<VisualScriptEditor *>(this)->_show_hint(vformat(TTR("Hold %s to drop a Variable Setter."), find_keycode_name(KEY_META)));
#else
				const_cast<VisualScriptEditor *>(this)->_show_hint(TTR("Hold Ctrl to drop a Variable Setter."));
#endif
			}

			return true;
		}
	}

	return false;
}

static Node *_find_script_node(Node *p_edited_scene, Node *p_current_node, const Ref<Script> &script) {
	if (p_edited_scene != p_current_node && p_current_node->get_owner() != p_edited_scene) {
		return nullptr;
	}

	Ref<Script> scr = p_current_node->get_script();

	if (scr.is_valid() && scr == script) {
		return p_current_node;
	}

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *n = _find_script_node(p_edited_scene, p_current_node->get_child(i), script);
		if (n) {
			return n;
		}
	}

	return nullptr;
}

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

		Vector2 pos = _get_pos_in_graph(p_point);

		int new_id = _create_new_node_from_name(d["node_type"], pos, default_func);

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
		Vector2 pos = _get_pos_in_graph(p_point);

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
		undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, vnode, pos);
		undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
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
		Vector2 pos = _get_pos_in_graph(p_point);

		Ref<VisualScriptFunctionCall> vnode;
		vnode.instance();
		vnode->set_call_mode(VisualScriptFunctionCall::CALL_MODE_SELF);

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, vnode, pos);
		undo_redo->add_do_method(vnode.ptr(), "set_base_type", script->get_instance_base_type());
		undo_redo->add_do_method(vnode.ptr(), "set_function", d["function"]);

		undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
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
		Vector2 pos = _get_pos_in_graph(p_point);

		Ref<VisualScriptEmitSignal> vnode;
		vnode.instance();
		vnode->set_signal(d["signal"]);

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, vnode, pos);
		undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
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
		Vector2 pos = _get_pos_in_graph(p_point);

		Ref<VisualScriptPreload> prnode;
		prnode.instance();
		prnode->set_preload(d["resource"]);

		int new_id = script->get_available_id();

		undo_redo->create_action(TTR("Add Preload Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, prnode, pos);
		undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
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
#ifdef OSX_ENABLED
		bool use_preload = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool use_preload = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif
		Vector2 pos = _get_pos_in_graph(p_point);

		Array files = d["files"];

		List<int> new_ids;
		int new_id = script->get_available_id();

		if (files.size()) {
			undo_redo->create_action(TTR("Add Node(s)"));

			for (int i = 0; i < files.size(); i++) {
				Ref<Resource> res = ResourceLoader::load(files[i]);
				if (!res.is_valid()) {
					continue;
				}
				Ref<Script> drop_script = ResourceLoader::load(files[i]);
				if (drop_script.is_valid() && drop_script->is_tool() && drop_script->get_instance_base_type() == "VisualScriptCustomNode" && !use_preload) {
					Ref<VisualScriptCustomNode> vscn;
					vscn.instance();
					vscn->set_script(drop_script.get_ref_ptr());

					undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, vscn, pos);
					undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
				} else {
					Ref<VisualScriptPreload> prnode;
					prnode.instance();
					prnode->set_preload(res);

					undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, prnode, pos);
					undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
				}
				new_ids.push_back(new_id);
				new_id++;
				pos += Vector2(20, 20);
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
			EditorNode::get_singleton()->show_warning(vformat(TTR("Can't drop nodes because script '%s' is not used in this scene."), get_name()));
			return;
		}

#ifdef OSX_ENABLED
		bool use_node = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool use_node = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif

		Array nodes = d["nodes"];

		Vector2 pos = _get_pos_in_graph(p_point);

		undo_redo->create_action(TTR("Add Node(s) From Tree"));
		int base_id = script->get_available_id();

		if (use_node || nodes.size() > 1) {
			for (int i = 0; i < nodes.size(); i++) {
				NodePath np = nodes[i];
				Node *node = get_node(np);
				if (!node) {
					continue;
				}

				Ref<VisualScriptNode> n;
				Ref<VisualScriptSceneNode> scene_node;
				scene_node.instance();
				scene_node->set_node_path(sn->get_path_to(node));
				n = scene_node;

				undo_redo->add_do_method(script.ptr(), "add_node", default_func, base_id, n, pos);
				undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, base_id);

				base_id++;
				pos += Vector2(25, 25);
			}
		} else {
			NodePath np = nodes[0];
			Node *node = get_node(np);
			drop_position = pos;
			drop_node = node;
			drop_path = sn->get_path_to(node);
			new_connect_node_select->select_from_instance(node, "", false, node->get_class());
		}
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
	}

	if (String(d["type"]) == "obj_property") {
		Node *sn = _find_script_node(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root(), script);

		if (!sn && !Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
			EditorNode::get_singleton()->show_warning(vformat(TTR("Can't drop properties because script '%s' is not used in this scene.\nDrop holding 'Shift' to just copy the signature."), get_name()));
			return;
		}

		Object *obj = d["object"];

		if (!obj) {
			return;
		}

		Node *node = Object::cast_to<Node>(obj);
		Vector2 pos = _get_pos_in_graph(p_point);

#ifdef OSX_ENABLED
		bool use_get = Input::get_singleton()->is_key_pressed(KEY_META);
#else
		bool use_get = Input::get_singleton()->is_key_pressed(KEY_CONTROL);
#endif

		if (!node || Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
			if (use_get) {
				undo_redo->create_action(TTR("Add Getter Property"));
			} else {
				undo_redo->create_action(TTR("Add Setter Property"));
			}

			int base_id = script->get_available_id();

			Ref<VisualScriptNode> vnode;

			if (!use_get) {
				Ref<VisualScriptPropertySet> pset;
				pset.instance();
				pset->set_call_mode(VisualScriptPropertySet::CALL_MODE_INSTANCE);
				pset->set_base_type(obj->get_class());
				vnode = pset;
			} else {
				Ref<VisualScriptPropertyGet> pget;
				pget.instance();
				pget->set_call_mode(VisualScriptPropertyGet::CALL_MODE_INSTANCE);
				pget->set_base_type(obj->get_class());

				vnode = pget;
			}

			undo_redo->add_do_method(script.ptr(), "add_node", default_func, base_id, vnode, pos);
			undo_redo->add_do_method(vnode.ptr(), "set_property", d["property"]);
			if (!use_get) {
				undo_redo->add_do_method(vnode.ptr(), "set_default_input_value", 0, d["value"]);
			}

			undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, base_id);

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();

		} else {
			if (use_get) {
				undo_redo->create_action(TTR("Add Getter Property"));
			} else {
				undo_redo->create_action(TTR("Add Setter Property"));
			}

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
			undo_redo->add_do_method(script.ptr(), "add_node", default_func, base_id, vnode, pos);
			undo_redo->add_do_method(vnode.ptr(), "set_property", d["property"]);
			if (!use_get) {
				undo_redo->add_do_method(vnode.ptr(), "set_default_input_value", 0, d["value"]);
			}
			undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, base_id);

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();
		}
	}
}

void VisualScriptEditor::_selected_method(const String &p_method, const String &p_type, const bool p_connecting) {
	Ref<VisualScriptFunctionCall> vsfc = script->get_node(default_func, selecting_method_id);
	if (!vsfc.is_valid()) {
		return;
	}
	vsfc->set_function(p_method);
}

void VisualScriptEditor::_draw_color_over_button(Object *obj, Color p_color) {
	Button *button = Object::cast_to<Button>(obj);
	if (!button) {
		return;
	}

	Ref<StyleBox> normal = get_stylebox("normal", "Button");
	button->draw_rect(Rect2(normal->get_offset(), button->get_size() - normal->get_minimum_size()), p_color);
}

void VisualScriptEditor::_button_resource_previewed(const String &p_path, const Ref<Texture> &p_preview, const Ref<Texture> &p_small_preview, Variant p_ud) {
	Array ud = p_ud;
	ERR_FAIL_COND(ud.size() != 2);

	ObjectID id = ud[0];
	Object *obj = ObjectDB::get_instance(id);

	if (!obj) {
		return;
	}

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

RES VisualScriptEditor::get_edited_resource() const {
	return script;
}

void VisualScriptEditor::set_edited_resource(const RES &p_res) {
	ERR_FAIL_COND(script.is_valid());
	ERR_FAIL_COND(p_res.is_null());
	script = p_res;
	signal_editor->script = script;
	signal_editor->undo_redo = undo_redo;
	variable_editor->script = script;
	variable_editor->undo_redo = undo_redo;

	script->connect("node_ports_changed", this, "_node_ports_changed");

	default_func = script->get_default_func();

	if (!script->has_function(default_func)) // this is the supposed default function
	{
		script->add_function(default_func);
		script->set_edited(true); //so that if a function was added it's saved
	}

	_update_graph();
	call_deferred("_update_members");
}

void VisualScriptEditor::enable_editor() {
}

Vector<String> VisualScriptEditor::get_functions() {
	return Vector<String>();
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
	} else if (script->get_name() != "") {
		name = script->get_name();
	} else {
		name = script->get_class() + "(" + itos(script->get_instance_id()) + ")";
	}

	return name;
}

Ref<Texture> VisualScriptEditor::get_icon() {
	return Control::get_icon("VisualScript", "EditorIcons");
}

bool VisualScriptEditor::is_unsaved() {
	return script->is_edited() || script->are_subnodes_edited();
}

Variant VisualScriptEditor::get_edit_state() {
	Dictionary d;
	d["function"] = default_func;
	d["scroll"] = graph->get_scroll_ofs();
	d["zoom"] = graph->get_zoom();
	d["using_snap"] = graph->is_using_snap();
	d["snap"] = graph->get_snap();
	return d;
}

void VisualScriptEditor::set_edit_state(const Variant &p_state) {
	Dictionary d = p_state;
	if (d.has("function")) {
		selected = default_func;
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

void VisualScriptEditor::_center_on_node(const StringName &p_func, int p_id) {
	Node *n = graph->get_node(itos(p_id));
	GraphNode *gn = Object::cast_to<GraphNode>(n);

	// clear selection
	for (int i = 0; i < graph->get_child_count(); i++) {
		GraphNode *gnd = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gnd) {
			gnd->set_selected(false);
		}
	}

	if (gn) {
		gn->set_selected(true);
		Vector2 new_scroll = gn->get_offset() - graph->get_size() * 0.5 + gn->get_size() * 0.5;
		graph->set_scroll_ofs(new_scroll);
		script->set_function_scroll(p_func, new_scroll / EDSCALE);
		script->set_edited(true);
	}
}

void VisualScriptEditor::goto_line(int p_line, bool p_with_error) {
	p_line += 1; //add one because script lines begin from 0.

	if (p_with_error) {
		error_line = p_line;
	}

	List<StringName> functions;
	script->get_function_list(&functions);
	for (List<StringName>::Element *E = functions.front(); E; E = E->next()) {
		if (script->has_node(E->get(), p_line)) {
			_update_graph();
			_update_members();

			call_deferred("call_deferred", "_center_on_node", E->get(), p_line); //editor might be just created and size might not exist yet
			return;
		}
	}
}

void VisualScriptEditor::set_executing_line(int p_line) {
	// todo: add a way to show which node is executing right now.
}

void VisualScriptEditor::clear_executing_line() {
	// todo: add a way to show which node is executing right now.
}

void VisualScriptEditor::trim_trailing_whitespace() {
}

void VisualScriptEditor::insert_final_newline() {
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
	_update_graph();
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
		_update_members();
		_update_graph();
		_center_on_node(p_function, script->get_function_node_id(p_function));
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

	_update_members();
	_update_graph();

	_center_on_node(p_function, script->get_function_node_id(p_function));
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
	select_base_type->popup_create(true, true);
}

void VisualScriptEditor::_toggle_tool_script() {
	script->set_tool_enabled(!script->is_tool());
}

void VisualScriptEditor::clear_edit_menu() {
	memdelete(edit_menu);
	memdelete(members_section);
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
	if (vnode.is_null()) {
		return;
	}

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

void VisualScriptEditor::_move_node(const StringName &p_func, int p_id, const Vector2 &p_to) {
	if (!script->has_function(p_func)) {
		return;
	}

	Node *node = graph->get_node(itos(p_id));

	if (Object::cast_to<GraphNode>(node)) {
		Object::cast_to<GraphNode>(node)->set_offset(p_to);
	}

	script->set_node_position(p_func, p_id, p_to / EDSCALE);
}

StringName VisualScriptEditor::_get_function_of_node(int p_id) const {
	List<StringName> funcs;
	script->get_function_list(&funcs);
	for (List<StringName>::Element *E = funcs.front(); E; E = E->next()) {
		if (script->has_node(E->get(), p_id)) {
			return E->get();
		}
	}

	return ""; // this is passed to avoid crash and is tested against later
}

void VisualScriptEditor::_node_moved(Vector2 p_from, Vector2 p_to, int p_id) {
	StringName func = _get_function_of_node(p_id);

	undo_redo->add_do_method(this, "_move_node", func, p_id, p_to);
	undo_redo->add_undo_method(this, "_move_node", func, p_id, p_from);
}

void VisualScriptEditor::_remove_node(int p_id) {
	undo_redo->create_action(TTR("Remove VisualScript Node"));

	StringName func = _get_function_of_node(p_id);

	undo_redo->add_do_method(script.ptr(), "remove_node", func, p_id);
	undo_redo->add_undo_method(script.ptr(), "add_node", func, p_id, script->get_node(func, p_id), script->get_node_position(func, p_id));

	List<VisualScript::SequenceConnection> sequence_conns;
	script->get_sequence_connection_list(func, &sequence_conns);

	for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {
		if (E->get().from_node == p_id || E->get().to_node == p_id) {
			undo_redo->add_undo_method(script.ptr(), "sequence_connect", func, E->get().from_node, E->get().from_output, E->get().to_node);
		}
	}

	List<VisualScript::DataConnection> data_conns;
	script->get_data_connection_list(func, &data_conns);

	for (List<VisualScript::DataConnection>::Element *E = data_conns.front(); E; E = E->next()) {
		if (E->get().from_node == p_id || E->get().to_node == p_id) {
			undo_redo->add_undo_method(script.ptr(), "data_connect", func, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();
}

void VisualScriptEditor::_node_ports_changed(const String &p_func, int p_id) {
	_update_graph(p_id);
}

bool VisualScriptEditor::node_has_sequence_connections(const StringName &p_func, int p_id) {
	List<VisualScript::SequenceConnection> sequence_conns;
	script->get_sequence_connection_list(p_func, &sequence_conns);

	for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {
		int from = E->get().from_node;
		int to = E->get().to_node;

		if (to == p_id || from == p_id) {
			return true;
		}
	}

	return false;
}

void VisualScriptEditor::_graph_connected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot) {
	StringName from_func = _get_function_of_node(p_from.to_int());

	Ref<VisualScriptNode> from_node = script->get_node(from_func, p_from.to_int());
	ERR_FAIL_COND(!from_node.is_valid());

	bool from_seq;
	int from_port;

	if (!_get_out_slot(from_node, p_from_slot, from_port, from_seq)) {
		return; //can't connect this, it's invalid
	}

	StringName to_func = _get_function_of_node(p_to.to_int());

	Ref<VisualScriptNode> to_node = script->get_node(to_func, p_to.to_int());
	ERR_FAIL_COND(!to_node.is_valid());

	bool to_seq;
	int to_port;

	if (!_get_in_slot(to_node, p_to_slot, to_port, to_seq)) {
		return; //can't connect this, it's invalid
	}

	ERR_FAIL_COND(from_seq != to_seq);

	// Checking to prevent warnings.
	if (from_seq) {
		if (script->has_sequence_connection(from_func, p_from.to_int(), from_port, p_to.to_int())) {
			return;
		}
	} else if (script->has_data_connection(from_func, p_from.to_int(), from_port, p_to.to_int(), to_port)) {
		return;
	}

	// Preventing connection to itself.
	if (p_from.to_int() == p_to.to_int()) {
		return;
	}

	// Do all the checks here
	StringName func; // this the func where we store the one the nodes at the end of the resolution on having multiple nodes

	undo_redo->create_action(TTR("Connect Nodes"));

	if (from_func == to_func) {
		func = to_func;
	} else if (from_seq) {
		// this is a sequence connection
		_move_nodes_with_rescan(to_func, from_func, p_to.to_int()); // this function moves the nodes from func1 to func2
		func = from_func;
	} else {
		if (node_has_sequence_connections(to_func, p_to.to_int())) {
			if (node_has_sequence_connections(from_func, p_from.to_int())) {
				ERR_PRINT("Trying to connect between different sequence node trees");
				return;
			} else {
				_move_nodes_with_rescan(from_func, to_func, p_from.to_int());
				func = to_func;
			}
		} else if (node_has_sequence_connections(from_func, p_from.to_int())) {
			if (from_func == default_func) {
				_move_nodes_with_rescan(from_func, to_func, p_from.to_int());
				func = to_func;
			} else {
				_move_nodes_with_rescan(to_func, from_func, p_to.to_int());
				func = from_func;
			}
		} else {
			if (to_func == default_func) {
				_move_nodes_with_rescan(to_func, from_func, p_to.to_int());
				func = from_func;
			} else {
				_move_nodes_with_rescan(from_func, to_func, p_from.to_int());
				func = to_func;
			}
		}
	}

	if (from_seq) {
		undo_redo->add_do_method(script.ptr(), "sequence_connect", func, p_from.to_int(), from_port, p_to.to_int());
		// this undo error on undo after move can't be removed without painful gymnastics
		undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", func, p_from.to_int(), from_port, p_to.to_int());
	} else {
		bool converted = false;
		int conv_node = -1;

		Ref<VisualScriptOperator> oper = to_node;
		if (oper.is_valid() && oper->get_typed() == Variant::NIL) {
			// it's an operator Node and if the type is already nil
			if (from_node->get_output_value_port_info(from_port).type != Variant::NIL) {
				oper->set_typed(from_node->get_output_value_port_info(from_port).type);
			}
		}

		Ref<VisualScriptOperator> operf = from_node;
		if (operf.is_valid() && operf->get_typed() == Variant::NIL) {
			// it's an operator Node and if the type is already nil
			if (to_node->get_input_value_port_info(to_port).type != Variant::NIL) {
				operf->set_typed(to_node->get_input_value_port_info(to_port).type);
			}
		}

		Variant::Type to_type = to_node->get_input_value_port_info(to_port).type;
		Variant::Type from_type = from_node->get_output_value_port_info(from_port).type;

		if (to_type != Variant::NIL && from_type != Variant::NIL && to_type != from_type) {
			// add a constructor node between the ports
			bool exceptions = false; // true if there are any exceptions
			exceptions = exceptions || (to_type == Variant::INT && from_type == Variant::REAL);
			exceptions = exceptions || (to_type == Variant::REAL && from_type == Variant::INT);
			if (Variant::can_convert(from_type, to_type) && !exceptions) {
				MethodInfo mi;
				mi.name = Variant::get_type_name(to_type);
				PropertyInfo pi;
				pi.name = "from";
				pi.type = from_type;
				mi.arguments.push_back(pi);
				mi.return_val.type = to_type;
				// we know that this is allowed so create a new constructor node
				Ref<VisualScriptConstructor> constructor;
				constructor.instance();
				constructor->set_constructor_type(to_type);
				constructor->set_constructor(mi);
				// add the new constructor node

				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_node(p_from));
				GraphNode *gn2 = Object::cast_to<GraphNode>(graph->get_node(p_to));
				if (gn && gn2) {
					Vector2 from_node_size = gn->get_rect().get_size();
					Vector2 to_node_size = gn2->get_rect().get_size();
					Vector2 to_node_pos = script->get_node_position(func, p_to.to_int());
					Vector2 from_node_pos = script->get_node_position(func, p_from.to_int());
					Vector2 new_to_node_pos = from_node_pos;
					Vector2 constructor_pos;
					if ((to_node_pos.x - from_node_pos.x) < 0) {
						// to is behind from node
						if (to_node_pos.x > (from_node_pos.x - to_node_size.x - 240)) {
							new_to_node_pos.x = from_node_pos.x - to_node_size.x - 240; // approx size of constructor node + padding
						} else {
							new_to_node_pos.x = to_node_pos.x;
						}
						new_to_node_pos.y = to_node_pos.y;
						constructor_pos.x = from_node_pos.x - 210;
						constructor_pos.y = to_node_pos.y;
					} else {
						// to is ahead of from node
						if (to_node_pos.x < (from_node_size.x + from_node_pos.x + 240)) {
							new_to_node_pos.x = from_node_size.x + from_node_pos.x + 240; // approx size of constructor node + padding
						} else {
							new_to_node_pos.x = to_node_pos.x;
						}
						new_to_node_pos.y = to_node_pos.y;
						constructor_pos.x = from_node_size.x + from_node_pos.x + 10;
						constructor_pos.y = to_node_pos.y;
					}
					undo_redo->add_do_method(this, "_move_node", func, p_to.to_int(), new_to_node_pos);
					undo_redo->add_undo_method(this, "_move_node", func, p_to.to_int(), to_node_pos);
					conv_node = script->get_available_id();
					undo_redo->add_do_method(script.ptr(), "add_node", func, conv_node, constructor, _get_available_pos(false, constructor_pos));
					undo_redo->add_undo_method(script.ptr(), "remove_node", func, conv_node);
					converted = true;
				}
			}
		}

		// disconnect current, and connect the new one
		if (script->is_input_value_port_connected(func, p_to.to_int(), to_port)) {
			if (can_swap && data_disconnect_node == p_to.to_int()) {
				int conn_from;
				int conn_port;
				script->get_input_value_port_connection_source(func, p_to.to_int(), to_port, &conn_from, &conn_port);
				undo_redo->add_do_method(script.ptr(), "data_disconnect", func, conn_from, conn_port, p_to.to_int(), to_port);
				undo_redo->add_do_method(script.ptr(), "data_connect", func, conn_from, conn_port, data_disconnect_node, data_disconnect_port);
				undo_redo->add_undo_method(script.ptr(), "data_disconnect", func, conn_from, conn_port, data_disconnect_node, data_disconnect_port);
				undo_redo->add_undo_method(script.ptr(), "data_connect", func, conn_from, conn_port, p_to.to_int(), to_port);
				can_swap = false; // swapped
			} else {
				int conn_from;
				int conn_port;
				script->get_input_value_port_connection_source(func, p_to.to_int(), to_port, &conn_from, &conn_port);
				undo_redo->add_do_method(script.ptr(), "data_disconnect", func, conn_from, conn_port, p_to.to_int(), to_port);
				undo_redo->add_undo_method(script.ptr(), "data_connect", func, conn_from, conn_port, p_to.to_int(), to_port);
			}
		}
		if (!converted) {
			undo_redo->add_do_method(script.ptr(), "data_connect", func, p_from.to_int(), from_port, p_to.to_int(), to_port);
			undo_redo->add_undo_method(script.ptr(), "data_disconnect", func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		} else {
			// this is noice
			undo_redo->add_do_method(script.ptr(), "data_connect", func, p_from.to_int(), from_port, conv_node, 0);
			undo_redo->add_do_method(script.ptr(), "data_connect", func, conv_node, 0, p_to.to_int(), to_port);
			// I don't think this is needed but gonna leave it here for now... until I need to finalise it all
			undo_redo->add_undo_method(script.ptr(), "data_disconnect", func, p_from.to_int(), from_port, conv_node, 0);
			undo_redo->add_undo_method(script.ptr(), "data_disconnect", func, conv_node, 0, p_to.to_int(), to_port);
		}
		//update nodes in graph
		if (!converted) {
			undo_redo->add_do_method(this, "_update_graph", p_from.to_int());
			undo_redo->add_do_method(this, "_update_graph", p_to.to_int());
			undo_redo->add_undo_method(this, "_update_graph", p_from.to_int());
			undo_redo->add_undo_method(this, "_update_graph", p_to.to_int());
		} else {
			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
		}
	}

	undo_redo->add_do_method(this, "_update_graph_connections");
	undo_redo->add_undo_method(this, "_update_graph_connections");

	undo_redo->commit_action();
}

void VisualScriptEditor::_graph_disconnected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot) {
	StringName func = _get_function_of_node(p_from.to_int());
	ERR_FAIL_COND(func != _get_function_of_node(p_to.to_int()));

	Ref<VisualScriptNode> from_node = script->get_node(func, p_from.to_int());
	ERR_FAIL_COND(!from_node.is_valid());

	bool from_seq;
	int from_port;

	if (!_get_out_slot(from_node, p_from_slot, from_port, from_seq)) {
		return; //can't connect this, it's invalid
	}

	Ref<VisualScriptNode> to_node = script->get_node(func, p_to.to_int());
	ERR_FAIL_COND(!to_node.is_valid());

	bool to_seq;
	int to_port;

	if (!_get_in_slot(to_node, p_to_slot, to_port, to_seq)) {
		return; //can't connect this, it's invalid
	}

	ERR_FAIL_COND(from_seq != to_seq);

	undo_redo->create_action(TTR("Disconnect Nodes"));

	if (from_seq) {
		undo_redo->add_do_method(script.ptr(), "sequence_disconnect", func, p_from.to_int(), from_port, p_to.to_int());
		undo_redo->add_undo_method(script.ptr(), "sequence_connect", func, p_from.to_int(), from_port, p_to.to_int());
	} else {
		can_swap = true;
		data_disconnect_node = p_to.to_int();
		data_disconnect_port = to_port;

		undo_redo->add_do_method(script.ptr(), "data_disconnect", func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		undo_redo->add_undo_method(script.ptr(), "data_connect", func, p_from.to_int(), from_port, p_to.to_int(), to_port);
		//update relevant nodes in the graph
		undo_redo->add_do_method(this, "_update_graph", p_from.to_int());
		undo_redo->add_do_method(this, "_update_graph", p_to.to_int());
		undo_redo->add_undo_method(this, "_update_graph", p_from.to_int());
		undo_redo->add_undo_method(this, "_update_graph", p_to.to_int());
	}
	undo_redo->add_do_method(this, "_update_graph_connections");
	undo_redo->add_undo_method(this, "_update_graph_connections");

	undo_redo->commit_action();
}

void VisualScriptEditor::_move_nodes_with_rescan(const StringName &p_func_from, const StringName &p_func_to, int p_id) {
	Set<int> nodes_to_move;
	HashMap<int, Map<int, int>> seqconns_to_move; // from => List(outp, to)
	HashMap<int, Map<int, Pair<int, int>>> dataconns_to_move; // to => List(inp_p => from, outp)

	nodes_to_move.insert(p_id);
	Set<int> sequence_connections;
	{
		List<VisualScript::SequenceConnection> sequence_conns;
		script->get_sequence_connection_list(p_func_from, &sequence_conns);

		HashMap<int, Map<int, int>> seqcons; // from => List(out_p => to)

		for (List<VisualScript::SequenceConnection>::Element *E = sequence_conns.front(); E; E = E->next()) {
			int from = E->get().from_node;
			int to = E->get().to_node;
			int out_p = E->get().from_output;
			if (!seqcons.has(from)) {
				seqcons.set(from, Map<int, int>());
			}
			seqcons[from].insert(out_p, to);
			sequence_connections.insert(to);
			sequence_connections.insert(from);
		}

		int conn = p_id;
		List<int> stack;
		HashMap<int, Set<int>> seen; // from, outp
		while (seqcons.has(conn)) {
			for (auto E = seqcons[conn].front(); E; E = E->next()) {
				if (seen.has(conn) && seen[conn].has(E->key())) {
					if (!E->next()) {
						if (stack.size() > 0) {
							conn = stack.back()->get();
							stack.pop_back();
							break;
						}
						conn = -101;
						break;
					}
					continue;
				}
				if (!seen.has(conn)) {
					seen.set(conn, Set<int>());
				}
				seen[conn].insert(E->key());
				stack.push_back(conn);
				if (!seqconns_to_move.has(conn)) {
					seqconns_to_move.set(conn, Map<int, int>());
				}
				seqconns_to_move[conn].insert(E->key(), E->get());
				conn = E->get();
				nodes_to_move.insert(conn);
				break;
			}
			if (!seqcons.has(conn) && stack.size() > 0) {
				conn = stack.back()->get();
				stack.pop_back();
			}
		}
	}

	{
		List<VisualScript::DataConnection> data_connections;
		script->get_data_connection_list(p_func_from, &data_connections);
		int func_from_node_id = script->get_function_node_id(p_func_from);

		HashMap<int, Map<int, Pair<int, int>>> connections;

		for (List<VisualScript::DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
			int from = E->get().from_node;
			int to = E->get().to_node;
			int out_p = E->get().from_port;
			int in_p = E->get().to_port;

			// skip if the from_node is a function node
			if (from == func_from_node_id) {
				continue;
			}

			if (!connections.has(to)) {
				connections.set(to, Map<int, Pair<int, int>>());
			}
			connections[to].insert(in_p, Pair<int, int>(from, out_p));
		}

		// go through the HashMap and do all sorts of crazy ass stuff now...
		Set<int> nodes_to_be_added;
		for (Set<int>::Element *F = nodes_to_move.front(); F; F = F->next()) {
			HashMap<int, Set<int>> seen;
			List<int> stack;
			int id = F->get();
			while (connections.has(id)) {
				for (auto E = connections[id].front(); E; E = E->next()) {
					if (seen.has(id) && seen[id].has(E->key())) {
						if (!E->next()) {
							if (stack.size() > 0) {
								id = stack.back()->get();
								stack.pop_back();
								break;
							}
							id = -11; // I assume ids can't be negative should confirm it...
							break;
						}
						continue;
					}

					if (sequence_connections.has(E->get().first)) {
						if (!nodes_to_move.has(E->get().first)) {
							if (stack.size() > 0) {
								id = stack.back()->get();
								stack.pop_back();
								break;
							}
							id = -11; // I assume ids can't be negative should confirm it...
							break;
						}
					}

					if (!seen.has(id)) {
						seen.set(id, Set<int>());
					}
					seen[id].insert(E->key());
					stack.push_back(id);
					if (!dataconns_to_move.has(id)) {
						dataconns_to_move.set(id, Map<int, Pair<int, int>>());
					}
					dataconns_to_move[id].insert(E->key(), Pair<int, int>(E->get().first, E->get().second));
					id = E->get().first;
					nodes_to_be_added.insert(id);
					break;
				}
				if (!connections.has(id) && stack.size() > 0) {
					id = stack.back()->get();
					stack.pop_back();
				}
			}
		}
		for (Set<int>::Element *E = nodes_to_be_added.front(); E; E = E->next()) {
			nodes_to_move.insert(E->get());
		}
	}

	// * this is primarily for the sake of the having proper undo
	List<VisualScript::SequenceConnection> seqext;
	List<VisualScript::DataConnection> dataext;

	List<VisualScript::SequenceConnection> seq_connections;
	script->get_sequence_connection_list(p_func_from, &seq_connections);

	for (List<VisualScript::SequenceConnection>::Element *E = seq_connections.front(); E; E = E->next()) {
		if (!nodes_to_move.has(E->get().from_node) && nodes_to_move.has(E->get().to_node)) {
			seqext.push_back(E->get());
		} else if (nodes_to_move.has(E->get().from_node) && !nodes_to_move.has(E->get().to_node)) {
			seqext.push_back(E->get());
		}
	}

	List<VisualScript::DataConnection> data_connections;
	script->get_data_connection_list(p_func_from, &data_connections);

	for (List<VisualScript::DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
		if (!nodes_to_move.has(E->get().from_node) && nodes_to_move.has(E->get().to_node)) {
			dataext.push_back(E->get());
		} else if (nodes_to_move.has(E->get().from_node) && !nodes_to_move.has(E->get().to_node)) {
			dataext.push_back(E->get());
		}
	}

	// undo_redo->create_action("Rescan Functions");

	for (Set<int>::Element *E = nodes_to_move.front(); E; E = E->next()) {
		int id = E->get();

		undo_redo->add_do_method(script.ptr(), "remove_node", p_func_from, id);
		undo_redo->add_do_method(script.ptr(), "add_node", p_func_to, id, script->get_node(p_func_from, id), script->get_node_position(p_func_from, id));

		undo_redo->add_undo_method(script.ptr(), "remove_node", p_func_to, id);
		undo_redo->add_undo_method(script.ptr(), "add_node", p_func_from, id, script->get_node(p_func_from, id), script->get_node_position(p_func_from, id));
	}

	List<int> skeys;
	seqconns_to_move.get_key_list(&skeys);
	for (List<int>::Element *E = skeys.front(); E; E = E->next()) {
		int from_node = E->get();
		for (Map<int, int>::Element *F = seqconns_to_move[from_node].front(); F; F = F->next()) {
			int from_port = F->key();
			int to_node = F->get();
			undo_redo->add_do_method(script.ptr(), "sequence_connect", p_func_to, from_node, from_port, to_node);
			undo_redo->add_undo_method(script.ptr(), "sequence_connect", p_func_from, from_node, from_port, to_node);
		}
	}

	List<int> keys;
	dataconns_to_move.get_key_list(&keys);
	for (List<int>::Element *E = keys.front(); E; E = E->next()) {
		int to_node = E->get(); // to_node
		for (Map<int, Pair<int, int>>::Element *F = dataconns_to_move[E->get()].front(); F; F = F->next()) {
			int inp_p = F->key();
			Pair<int, int> fro = F->get();

			undo_redo->add_do_method(script.ptr(), "data_connect", p_func_to, fro.first, fro.second, to_node, inp_p);
			undo_redo->add_undo_method(script.ptr(), "data_connect", p_func_from, fro.first, fro.second, to_node, inp_p);
		}
	}

	// this to have proper undo operations
	for (List<VisualScript::SequenceConnection>::Element *E = seqext.front(); E; E = E->next()) {
		undo_redo->add_undo_method(script.ptr(), "sequence_connect", p_func_from, E->get().from_node, E->get().from_output, E->get().to_node);
	}
	for (List<VisualScript::DataConnection>::Element *E = dataext.front(); E; E = E->next()) {
		undo_redo->add_undo_method(script.ptr(), "data_connect", p_func_from, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
	}
	// this doesn't need do methods as they are handled by the subsequent do calls implicitly

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	// undo_redo->commit_action();
}

void VisualScriptEditor::_graph_connect_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_pos) {
	Node *node = graph->get_node(p_from);
	GraphNode *gn = Object::cast_to<GraphNode>(node);
	if (!gn) {
		return;
	}

	StringName func = _get_function_of_node(p_from.to_int());

	Ref<VisualScriptNode> vsn = script->get_node(func, p_from.to_int());
	if (!vsn.is_valid()) {
		return;
	}

	if (vsn->get_output_value_port_count() || vsn->get_output_sequence_port_count()) {
		port_action_pos = p_release_pos;
	}

	if (p_from_slot < vsn->get_output_sequence_port_count()) {
		port_action_node = p_from.to_int();
		port_action_output = p_from_slot;
		_port_action_menu(CREATE_ACTION, func);
	} else {
		port_action_output = p_from_slot - vsn->get_output_sequence_port_count();
		port_action_node = p_from.to_int();
		_port_action_menu(CREATE_CALL_SET_GET, func);
	}
}

VisualScriptNode::TypeGuess VisualScriptEditor::_guess_output_type(int p_port_action_node, int p_port_action_output, Set<int> &visited_nodes) {
	VisualScriptNode::TypeGuess tg;
	tg.type = Variant::NIL;

	if (visited_nodes.has(p_port_action_node)) {
		return tg; //no loop
	}

	visited_nodes.insert(p_port_action_node);

	StringName func = _get_function_of_node(p_port_action_node);

	Ref<VisualScriptNode> node = script->get_node(func, p_port_action_node);

	if (!node.is_valid() || node->get_output_value_port_count() <= p_port_action_output) {
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

			if (script->get_input_value_port_connection_source(func, p_port_action_node, i, &from_node, &from_port)) {
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

	return node->guess_output_type(in_guesses.ptrw(), p_port_action_output);
}

void VisualScriptEditor::_port_action_menu(int p_option, const StringName &func) {
	Set<int> vn;

	switch (p_option) {
		case CREATE_CALL_SET_GET: {
			Ref<VisualScriptFunctionCall> n;
			n.instance();

			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);

			if (tg.gdclass != StringName()) {
				n->set_base_type(tg.gdclass);
			} else {
				n->set_base_type("Object");
			}
			String type_string;
			if (script->get_node(func, port_action_node)->get_output_value_port_count() > 0) {
				type_string = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint_string;
			}
			if (tg.type == Variant::OBJECT) {
				if (tg.script.is_valid()) {
					new_connect_node_select->select_from_script(tg.script, "");
				} else if (type_string != String()) {
					new_connect_node_select->select_from_base_type(type_string);
				} else {
					new_connect_node_select->select_from_base_type(n->get_base_type());
				}
			} else if (tg.type == Variant::NIL) {
				new_connect_node_select->select_from_base_type("");
			} else {
				new_connect_node_select->select_from_basic_type(tg.type);
			}
			// ensure that the dialog fits inside the graph
			Vector2 pos = mouse_up_position;
			Size2 bounds = graph->get_global_position() + graph->get_size() - new_connect_node_select->get_size();
			pos.x = pos.x > bounds.x ? bounds.x : pos.x;
			pos.y = pos.y > bounds.y ? bounds.y : pos.y;
			new_connect_node_select->set_position(pos);
		} break;
		case CREATE_ACTION: {
			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);
			PropertyInfo property_info;
			if (script->get_node(func, port_action_node)->get_output_value_port_count() > 0) {
				property_info = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output);
			}
			if (tg.type == Variant::OBJECT) {
				if (property_info.type == Variant::OBJECT && property_info.hint_string != String()) {
					new_connect_node_select->select_from_action(property_info.hint_string);
				} else {
					new_connect_node_select->select_from_action("");
				}
			} else if (tg.type == Variant::NIL) {
				new_connect_node_select->select_from_action("");
			} else {
				new_connect_node_select->select_from_action(Variant::get_type_name(tg.type));
			}
			// ensure that the dialog fits inside the graph
			Vector2 pos = mouse_up_position;
			Size2 bounds = graph->get_global_position() + graph->get_size() - new_connect_node_select->get_size();
			pos.x = pos.x > bounds.x ? bounds.x : pos.x;
			pos.y = pos.y > bounds.y ? bounds.y : pos.y;
			new_connect_node_select->set_position(pos);
		} break;
	}
}

void VisualScriptEditor::connect_data(Ref<VisualScriptNode> vnode_old, Ref<VisualScriptNode> vnode, int new_id) {
	undo_redo->create_action(TTR("Connect Node Data"));
	VisualScriptReturn *vnode_return = Object::cast_to<VisualScriptReturn>(vnode.ptr());
	if (vnode_return != nullptr && vnode_old->get_output_value_port_count() > 0) {
		vnode_return->set_enable_return_value(true);
	}
	if (vnode_old->get_output_value_port_count() <= 0) {
		undo_redo->commit_action();
		return;
	}
	if (vnode->get_input_value_port_count() <= 0) {
		undo_redo->commit_action();
		return;
	}
	int port = port_action_output;
	int value_count = vnode_old->get_output_value_port_count();
	if (port >= value_count) {
		port = 0;
	}
	StringName func = _get_function_of_node(port_action_node);
	undo_redo->add_do_method(script.ptr(), "data_connect", func, port_action_node, port, new_id, 0);
	undo_redo->add_undo_method(script.ptr(), "data_disconnect", func, port_action_node, port, new_id, 0);
	undo_redo->commit_action();
}

void VisualScriptEditor::_selected_connect_node(const String &p_text, const String &p_category, const bool p_connecting) {
	Vector2 pos = _get_pos_in_graph(port_action_pos);

	Set<int> vn;

	if (drop_position != Vector2()) {
		pos = drop_position;
	}
	drop_position = Vector2();

	bool port_node_exists = true;

	StringName func = _get_function_of_node(port_action_node);
	if (func == StringName()) {
		func = default_func;
		port_node_exists = false;
	}

	if (p_category == "visualscript") {
		Ref<VisualScriptNode> vnode_new = VisualScriptLanguage::singleton->create_node_from_name(p_text);
		Ref<VisualScriptNode> vnode_old;
		if (port_node_exists) {
			vnode_old = script->get_node(func, port_action_node);
		}
		int new_id = script->get_available_id();

		if (Object::cast_to<VisualScriptOperator>(vnode_new.ptr()) && vnode_old.is_valid()) {
			Variant::Type type = vnode_old->get_output_value_port_info(port_action_output).type;
			Object::cast_to<VisualScriptOperator>(vnode_new.ptr())->set_typed(type);
		}

		if (Object::cast_to<VisualScriptTypeCast>(vnode_new.ptr()) && vnode_old.is_valid()) {
			Variant::Type type = vnode_old->get_output_value_port_info(port_action_output).type;
			String hint_name = vnode_old->get_output_value_port_info(port_action_output).hint_string;

			if (type == Variant::OBJECT) {
				Object::cast_to<VisualScriptTypeCast>(vnode_new.ptr())->set_base_type(hint_name);
			} else if (type == Variant::NIL) {
				Object::cast_to<VisualScriptTypeCast>(vnode_new.ptr())->set_base_type("");
			} else {
				Object::cast_to<VisualScriptTypeCast>(vnode_new.ptr())->set_base_type(Variant::get_type_name(type));
			}
		}

		undo_redo->create_action(TTR("Add Node"));
		undo_redo->add_do_method(script.ptr(), "add_node", func, new_id, vnode_new, pos);
		if (vnode_old.is_valid() && p_connecting) {
			connect_seq(vnode_old, vnode_new, new_id);
			connect_data(vnode_old, vnode_new, new_id);
		}

		undo_redo->add_undo_method(script.ptr(), "remove_node", func, new_id);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		return;
	}

	Ref<VisualScriptNode> vnode;
	Ref<VisualScriptPropertySet> script_prop_set;

	if (p_category == String("method")) {
		Ref<VisualScriptFunctionCall> n;
		n.instance();

		if (!drop_path.is_empty()) {
			if (drop_path == String(".")) {
				n->set_call_mode(VisualScriptFunctionCall::CALL_MODE_SELF);
			} else {
				n->set_call_mode(VisualScriptFunctionCall::CALL_MODE_NODE_PATH);
				n->set_base_path(drop_path);
			}
		}
		if (drop_node) {
			n->set_base_type(drop_node->get_class());
			if (drop_node->get_script_instance()) {
				n->set_base_script(drop_node->get_script_instance()->get_script()->get_path());
			}
		}
		vnode = n;
	} else if (p_category == String("set")) {
		Ref<VisualScriptPropertySet> n;
		n.instance();
		if (!drop_path.is_empty()) {
			if (drop_path == String(".")) {
				n->set_call_mode(VisualScriptPropertySet::CALL_MODE_SELF);
			} else {
				n->set_call_mode(VisualScriptPropertySet::CALL_MODE_NODE_PATH);
				n->set_base_path(drop_path);
			}
		}
		if (drop_node) {
			n->set_base_type(drop_node->get_class());
			if (drop_node->get_script_instance()) {
				n->set_base_script(drop_node->get_script_instance()->get_script()->get_path());
			}
		}
		vnode = n;
		script_prop_set = n;
	} else if (p_category == String("get")) {
		Ref<VisualScriptPropertyGet> n;
		n.instance();
		n->set_property(p_text);
		if (!drop_path.is_empty()) {
			if (drop_path == String(".")) {
				n->set_call_mode(VisualScriptPropertyGet::CALL_MODE_SELF);
			} else {
				n->set_call_mode(VisualScriptPropertyGet::CALL_MODE_NODE_PATH);
				n->set_base_path(drop_path);
			}
		}
		if (drop_node) {
			n->set_base_type(drop_node->get_class());
			if (drop_node->get_script_instance()) {
				n->set_base_script(drop_node->get_script_instance()->get_script()->get_path());
			}
		}
		vnode = n;
	}
	drop_path = String();
	drop_node = nullptr;

	if (p_category == String("action")) {
		if (p_text == "VisualScriptCondition") {
			Ref<VisualScriptCondition> n;
			n.instance();
			vnode = n;
		}
		if (p_text == "VisualScriptSwitch") {
			Ref<VisualScriptSwitch> n;
			n.instance();
			vnode = n;
		} else if (p_text == "VisualScriptSequence") {
			Ref<VisualScriptSequence> n;
			n.instance();
			vnode = n;
		} else if (p_text == "VisualScriptIterator") {
			Ref<VisualScriptIterator> n;
			n.instance();
			vnode = n;
		} else if (p_text == "VisualScriptWhile") {
			Ref<VisualScriptWhile> n;
			n.instance();
			vnode = n;
		} else if (p_text == "VisualScriptReturn") {
			Ref<VisualScriptReturn> n;
			n.instance();
			vnode = n;
		}
	}

	int new_id = script->get_available_id();
	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(script.ptr(), "add_node", func, new_id, vnode, pos);
	undo_redo->add_undo_method(script.ptr(), "remove_node", func, new_id);
	undo_redo->add_do_method(this, "_update_graph", new_id);
	undo_redo->add_undo_method(this, "_update_graph", new_id);
	undo_redo->commit_action();

	if (script_prop_set.is_valid()) {
		script_prop_set->set_property(p_text);
	}

	port_action_new_node = new_id;

	Ref<VisualScriptNode> vsn = script->get_node(func, port_action_new_node);

	if (Object::cast_to<VisualScriptFunctionCall>(vsn.ptr())) {
		Ref<VisualScriptFunctionCall> vsfc = vsn;
		vsfc->set_function(p_text);

		if (port_node_exists && p_connecting) {
			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);

			if (tg.type == Variant::OBJECT) {
				vsfc->set_call_mode(VisualScriptFunctionCall::CALL_MODE_INSTANCE);
				vsfc->set_base_type(String(""));
				if (tg.gdclass != StringName()) {
					vsfc->set_base_type(tg.gdclass);

				} else if (script->get_node(func, port_action_node).is_valid()) {
					PropertyHint hint = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint;
					String base_type = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint_string;

					if (base_type != String() && hint == PROPERTY_HINT_TYPE_STRING) {
						vsfc->set_base_type(base_type);
					}
					if (p_text == "call" || p_text == "call_deferred") {
						vsfc->set_function(String(""));
					}
				}
				if (tg.script.is_valid()) {
					vsfc->set_base_script(tg.script->get_path());
				}
			} else if (tg.type == Variant::NIL) {
				vsfc->set_call_mode(VisualScriptFunctionCall::CALL_MODE_INSTANCE);
				vsfc->set_base_type(String(""));
			} else {
				vsfc->set_call_mode(VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE);
				vsfc->set_basic_type(tg.type);
			}
		}
	}

	if (port_node_exists && p_connecting) {
		if (Object::cast_to<VisualScriptPropertySet>(vsn.ptr())) {
			Ref<VisualScriptPropertySet> vsp = vsn;

			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);
			if (tg.type == Variant::OBJECT) {
				vsp->set_call_mode(VisualScriptPropertySet::CALL_MODE_INSTANCE);
				vsp->set_base_type(String(""));
				if (tg.gdclass != StringName()) {
					vsp->set_base_type(tg.gdclass);

				} else if (script->get_node(func, port_action_node).is_valid()) {
					PropertyHint hint = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint;
					String base_type = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint_string;

					if (base_type != String() && hint == PROPERTY_HINT_TYPE_STRING) {
						vsp->set_base_type(base_type);
					}
				}
				if (tg.script.is_valid()) {
					vsp->set_base_script(tg.script->get_path());
				}
			} else if (tg.type == Variant::NIL) {
				vsp->set_call_mode(VisualScriptPropertySet::CALL_MODE_INSTANCE);
				vsp->set_base_type(String(""));
			} else {
				vsp->set_call_mode(VisualScriptPropertySet::CALL_MODE_BASIC_TYPE);
				vsp->set_basic_type(tg.type);
			}
		}

		if (Object::cast_to<VisualScriptPropertyGet>(vsn.ptr())) {
			Ref<VisualScriptPropertyGet> vsp = vsn;

			VisualScriptNode::TypeGuess tg = _guess_output_type(port_action_node, port_action_output, vn);
			if (tg.type == Variant::OBJECT) {
				vsp->set_call_mode(VisualScriptPropertyGet::CALL_MODE_INSTANCE);
				vsp->set_base_type(String(""));
				if (tg.gdclass != StringName()) {
					vsp->set_base_type(tg.gdclass);

				} else if (script->get_node(func, port_action_node).is_valid()) {
					PropertyHint hint = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint;
					String base_type = script->get_node(func, port_action_node)->get_output_value_port_info(port_action_output).hint_string;
					if (base_type != String() && hint == PROPERTY_HINT_TYPE_STRING) {
						vsp->set_base_type(base_type);
					}
				}
				if (tg.script.is_valid()) {
					vsp->set_base_script(tg.script->get_path());
				}
			} else if (tg.type == Variant::NIL) {
				vsp->set_call_mode(VisualScriptPropertyGet::CALL_MODE_INSTANCE);
				vsp->set_base_type(String(""));
			} else {
				vsp->set_call_mode(VisualScriptPropertyGet::CALL_MODE_BASIC_TYPE);
				vsp->set_basic_type(tg.type);
			}
		}
	}
	if (port_node_exists) {
		Ref<VisualScriptNode> vnode_old = script->get_node(func, port_action_node);
		if (vnode_old.is_valid() && p_connecting) {
			connect_seq(vnode_old, vnode, port_action_new_node);
			connect_data(vnode_old, vnode, port_action_new_node);
		}
	}
	_update_graph(port_action_new_node);
	if (port_node_exists) {
		_update_graph_connections();
	}
}

void VisualScriptEditor::connect_seq(Ref<VisualScriptNode> vnode_old, Ref<VisualScriptNode> vnode_new, int new_id) {
	VisualScriptOperator *vnode_operator = Object::cast_to<VisualScriptOperator>(vnode_new.ptr());
	if (vnode_operator != nullptr && !vnode_operator->has_input_sequence_port()) {
		return;
	}
	VisualScriptConstructor *vnode_constructor = Object::cast_to<VisualScriptConstructor>(vnode_new.ptr());
	if (vnode_constructor != nullptr) {
		return;
	}
	if (vnode_old->get_output_sequence_port_count() <= 0) {
		return;
	}
	if (!vnode_new->has_input_sequence_port()) {
		return;
	}

	StringName func = _get_function_of_node(port_action_node);

	undo_redo->create_action(TTR("Connect Node Sequence"));
	int pass_port = -vnode_old->get_output_sequence_port_count() + 1;
	int return_port = port_action_output - 1;
	if (vnode_old->get_output_value_port_info(port_action_output).name == String("pass") &&
			!script->get_output_sequence_ports_connected(func, port_action_node).has(pass_port)) {
		undo_redo->add_do_method(script.ptr(), "sequence_connect", func, port_action_node, pass_port, new_id);
		undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", func, port_action_node, pass_port, new_id);
	} else if (vnode_old->get_output_value_port_info(port_action_output).name == String("return") &&
			!script->get_output_sequence_ports_connected(func, port_action_node).has(return_port)) {
		undo_redo->add_do_method(script.ptr(), "sequence_connect", func, port_action_node, return_port, new_id);
		undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", func, port_action_node, return_port, new_id);
	} else {
		for (int port = 0; port < vnode_old->get_output_sequence_port_count(); port++) {
			int count = vnode_old->get_output_sequence_port_count();
			if (port_action_output < count && !script->get_output_sequence_ports_connected(func, port_action_node).has(port_action_output)) {
				undo_redo->add_do_method(script.ptr(), "sequence_connect", func, port_action_node, port_action_output, new_id);
				undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", func, port_action_node, port_action_output, new_id);
				break;
			} else if (!script->get_output_sequence_ports_connected(func, port_action_node).has(port)) {
				undo_redo->add_do_method(script.ptr(), "sequence_connect", func, port_action_node, port, new_id);
				undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", func, port_action_node, port, new_id);
				break;
			}
		}
	}

	undo_redo->commit_action();
}

void VisualScriptEditor::_selected_new_virtual_method(const String &p_text, const String &p_category, const bool p_connecting) {
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
	Ref<VisualScriptFunction> func_node;
	func_node.instance();
	func_node->set_name(name);

	undo_redo->create_action(TTR("Add Function"));
	undo_redo->add_do_method(script.ptr(), "add_function", name);

	for (int i = 0; i < minfo.arguments.size(); i++) {
		func_node->add_argument(minfo.arguments[i].type, minfo.arguments[i].name, -1, minfo.arguments[i].hint, minfo.arguments[i].hint_string);
	}

	Vector2 pos = _get_available_pos();

	undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id(), func_node, pos);
	if (minfo.return_val.type != Variant::NIL || minfo.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
		Ref<VisualScriptReturn> ret_node;
		ret_node.instance();
		ret_node->set_return_type(minfo.return_val.type);
		ret_node->set_enable_return_value(true);
		ret_node->set_name(name);
		undo_redo->add_do_method(script.ptr(), "add_node", name, script->get_available_id() + 1, ret_node, _get_available_pos(false, pos + Vector2(500, 0)));
	}

	undo_redo->add_undo_method(script.ptr(), "remove_function", name);
	undo_redo->add_do_method(this, "_update_members");
	undo_redo->add_undo_method(this, "_update_members");
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");

	undo_redo->commit_action();

	_update_graph();
}

void VisualScriptEditor::_cancel_connect_node() {
	// ensure the cancel is done
	port_action_new_node = -1;
}

int VisualScriptEditor::_create_new_node_from_name(const String &p_text, const Vector2 &p_point, const StringName &p_func) {
	StringName func = default_func;
	if (p_func != StringName()) {
		func = p_func;
	}

	Ref<VisualScriptNode> vnode = VisualScriptLanguage::singleton->create_node_from_name(p_text);
	int new_id = script->get_available_id();
	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(script.ptr(), "add_node", func, new_id, vnode, p_point);
	undo_redo->add_undo_method(script.ptr(), "remove_node", func, new_id);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	return new_id;
}

void VisualScriptEditor::_default_value_changed() {
	Ref<VisualScriptNode> vsn = script->get_node(_get_function_of_node(editing_id), editing_id);
	if (vsn.is_null()) {
		return;
	}

	undo_redo->create_action(TTR("Change Input Value"));
	undo_redo->add_do_method(vsn.ptr(), "set_default_input_value", editing_input, default_value_edit->get_variant());
	undo_redo->add_undo_method(vsn.ptr(), "set_default_input_value", editing_input, vsn->get_default_input_value(editing_input));

	undo_redo->add_do_method(this, "_update_graph", editing_id);
	undo_redo->add_undo_method(this, "_update_graph", editing_id);
	undo_redo->commit_action();
}

void VisualScriptEditor::_default_value_edited(Node *p_button, int p_id, int p_input_port) {
	Ref<VisualScriptNode> vsn = script->get_node(_get_function_of_node(p_id), p_id);
	if (vsn.is_null()) {
		return;
	}

	PropertyInfo pinfo = vsn->get_input_value_port_info(p_input_port);
	Variant existing = vsn->get_default_input_value(p_input_port);
	if (pinfo.type != Variant::NIL && existing.get_type() != pinfo.type) {
		Variant::CallError ce;
		const Variant *existingp = &existing;
		existing = Variant::construct(pinfo.type, &existingp, 1, ce, false);
	}

	default_value_edit->set_position(Object::cast_to<Control>(p_button)->get_global_position() + Vector2(0, Object::cast_to<Control>(p_button)->get_size().y));
	default_value_edit->set_size(Size2(1, 1));

	if (pinfo.type == Variant::NODE_PATH) {
		Node *edited_scene = get_tree()->get_edited_scene_root();
		if (edited_scene) { // Fixing an old crash bug ( Visual Script Crashes on editing NodePath with an empty scene open)
			Node *script_node = _find_script_node(edited_scene, edited_scene, script);

			if (script_node) {
				//pick a node relative to the script, IF the script exists
				pinfo.hint = PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE;
				pinfo.hint_string = script_node->get_path();
			} else {
				//pick a path relative to edited scene
				pinfo.hint = PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE;
				pinfo.hint_string = get_tree()->get_edited_scene_root()->get_path();
			}
		}
	}

	if (default_value_edit->edit(nullptr, pinfo.name, pinfo.type, existing, pinfo.hint, pinfo.hint_string)) {
		if (pinfo.hint == PROPERTY_HINT_MULTILINE_TEXT) {
			default_value_edit->popup_centered_ratio();
		} else {
			default_value_edit->popup();
		}
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

void VisualScriptEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			variable_editor->connect("changed", this, "_update_members");
			signal_editor->connect("changed", this, "_update_members");
			FALLTHROUGH;
		}
		case NOTIFICATION_THEME_CHANGED: {
			if (p_what != NOTIFICATION_READY && !is_visible_in_tree()) {
				return;
			}

			edit_variable_edit->add_style_override("bg", get_stylebox("bg", "Tree"));
			edit_signal_edit->add_style_override("bg", get_stylebox("bg", "Tree"));
			func_input_scroll->add_style_override("bg", get_stylebox("bg", "Tree"));

			Ref<Theme> tm = EditorNode::get_singleton()->get_theme_base()->get_theme();

			bool dark_theme = tm->get_constant("dark_theme", "Editor");

			List<Pair<String, Color>> colors;
			if (dark_theme) {
				colors.push_back(Pair<String, Color>("flow_control", Color(0.96, 0.96, 0.96)));
				colors.push_back(Pair<String, Color>("functions", Color(0.96, 0.52, 0.51)));
				colors.push_back(Pair<String, Color>("data", Color(0.5, 0.96, 0.81)));
				colors.push_back(Pair<String, Color>("operators", Color(0.67, 0.59, 0.87)));
				colors.push_back(Pair<String, Color>("custom", Color(0.5, 0.73, 0.96)));
				colors.push_back(Pair<String, Color>("constants", Color(0.96, 0.5, 0.69)));
			} else {
				colors.push_back(Pair<String, Color>("flow_control", Color(0.26, 0.26, 0.26)));
				colors.push_back(Pair<String, Color>("functions", Color(0.95, 0.4, 0.38)));
				colors.push_back(Pair<String, Color>("data", Color(0.07, 0.73, 0.51)));
				colors.push_back(Pair<String, Color>("operators", Color(0.51, 0.4, 0.82)));
				colors.push_back(Pair<String, Color>("custom", Color(0.31, 0.63, 0.95)));
				colors.push_back(Pair<String, Color>("constants", Color(0.94, 0.18, 0.49)));
			}

			for (List<Pair<String, Color>>::Element *E = colors.front(); E; E = E->next()) {
				Ref<StyleBoxFlat> sb = tm->get_stylebox("frame", "GraphNode");
				if (!sb.is_null()) {
					Ref<StyleBoxFlat> frame_style = sb->duplicate();
					Color c = sb->get_border_color();
					Color cn = E->get().second;
					cn.a = c.a;
					frame_style->set_border_color(cn);
					node_styles[E->get().first] = frame_style;
				}
			}

			if (is_visible_in_tree() && script.is_valid()) {
				_update_members();
				_update_graph();
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			members_section->set_visible(is_visible_in_tree());
		} break;
	}
}

void VisualScriptEditor::_graph_ofs_changed(const Vector2 &p_ofs) {
	if (updating_graph || !script.is_valid()) {
		return;
	}

	updating_graph = true;

	// Just use the default func for all the properties that need to be handled for drawing rather than adding to the Visual Script Class
	if (script->has_function(default_func)) {
		script->set_function_scroll(default_func, graph->get_scroll_ofs() / EDSCALE);
		script->set_edited(true);
	}
	updating_graph = false;
}

void VisualScriptEditor::_comment_node_resized(const Vector2 &p_new_size, int p_node) {
	if (updating_graph) {
		return;
	}

	StringName func = _get_function_of_node(p_node);

	Ref<VisualScriptComment> vsc = script->get_node(func, p_node);
	if (vsc.is_null()) {
		return;
	}

	Node *node = graph->get_node(itos(p_node));
	GraphNode *gn = Object::cast_to<GraphNode>(node);
	if (!gn) {
		return;
	}

	Vector2 new_size = p_new_size;
	if (graph->is_using_snap()) {
		Vector2 snap = Vector2(graph->get_snap(), graph->get_snap());
		Vector2 min_size = (gn->get_minimum_size() + (snap * 0.5)).snapped(snap);
		new_size = new_size.snapped(snap);
		new_size.x = MAX(new_size.x, min_size.x);
		new_size.y = MAX(new_size.y, min_size.y);
	}

	updating_graph = true;

	graph->set_block_minimum_size_adjust(true); //faster resize

	undo_redo->create_action(TTR("Resize Comment"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(vsc.ptr(), "set_size", new_size / EDSCALE);
	undo_redo->add_undo_method(vsc.ptr(), "set_size", vsc->get_size());
	undo_redo->commit_action();

	gn->set_custom_minimum_size(new_size);
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
						StringName func = _get_function_of_node(id);
						Ref<VisualScriptNode> vsn = script->get_node(func, id);
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
			_generic_search(script->get_instance_base_type());
		} break;
		case EDIT_COPY_NODES:
		case EDIT_CUT_NODES: {
			if (!script->has_function(default_func)) {
				break;
			}

			clipboard->nodes.clear();
			clipboard->data_connections.clear();
			clipboard->sequence_connections.clear();

			Set<String> funcs;
			for (int i = 0; i < graph->get_child_count(); i++) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
				if (gn) {
					if (gn->is_selected()) {
						int id = String(gn->get_name()).to_int();
						StringName func = _get_function_of_node(id);
						Ref<VisualScriptNode> node = script->get_node(func, id);
						if (Object::cast_to<VisualScriptFunction>(*node)) {
							EditorNode::get_singleton()->show_warning(TTR("Can't copy the function node."));
							return;
						}
						if (node.is_valid()) {
							clipboard->nodes[id] = node->duplicate(true);
							clipboard->nodes_positions[id] = script->get_node_position(func, id);
							funcs.insert(String(func));
						}
					}
				}
			}

			if (clipboard->nodes.empty()) {
				break;
			}

			for (Set<String>::Element *F = funcs.front(); F; F = F->next()) {
				List<VisualScript::SequenceConnection> sequence_connections;

				script->get_sequence_connection_list(F->get(), &sequence_connections);

				for (List<VisualScript::SequenceConnection>::Element *E = sequence_connections.front(); E; E = E->next()) {
					if (clipboard->nodes.has(E->get().from_node) && clipboard->nodes.has(E->get().to_node)) {
						clipboard->sequence_connections.insert(E->get());
					}
				}

				List<VisualScript::DataConnection> data_connections;

				script->get_data_connection_list(F->get(), &data_connections);

				for (List<VisualScript::DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
					if (clipboard->nodes.has(E->get().from_node) && clipboard->nodes.has(E->get().to_node)) {
						clipboard->data_connections.insert(E->get());
					}
				}
			}
			if (p_what == EDIT_CUT_NODES) {
				_on_nodes_delete(); // oh yeah, also delete on cut
			}

		} break;
		case EDIT_PASTE_NODES: {
			if (!script->has_function(default_func)) {
				break;
			}

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
				List<StringName> functions;
				script->get_function_list(&functions);
				for (List<StringName>::Element *F = functions.front(); F; F = F->next()) {
					List<int> nodes;
					script->get_node_list(F->get(), &nodes);
					for (List<int>::Element *E = nodes.front(); E; E = E->next()) {
						Vector2 pos = script->get_node_position(F->get(), E->get()).snapped(Vector2(2, 2));
						existing_positions.insert(pos);
					}
				}
			}

			for (Map<int, Ref<VisualScriptNode>>::Element *E = clipboard->nodes.front(); E; E = E->next()) {
				Ref<VisualScriptNode> node = E->get()->duplicate();

				int new_id = idc++;
				to_select.insert(new_id);

				remap[E->key()] = new_id;

				Vector2 paste_pos = clipboard->nodes_positions[E->key()];

				while (existing_positions.has(paste_pos.snapped(Vector2(2, 2)))) {
					paste_pos += Vector2(20, 20) * EDSCALE;
				}

				undo_redo->add_do_method(script.ptr(), "add_node", default_func, new_id, node, paste_pos);
				undo_redo->add_undo_method(script.ptr(), "remove_node", default_func, new_id);
			}

			for (Set<VisualScript::SequenceConnection>::Element *E = clipboard->sequence_connections.front(); E; E = E->next()) {
				undo_redo->add_do_method(script.ptr(), "sequence_connect", default_func, remap[E->get().from_node], E->get().from_output, remap[E->get().to_node]);
				undo_redo->add_undo_method(script.ptr(), "sequence_disconnect", default_func, remap[E->get().from_node], E->get().from_output, remap[E->get().to_node]);
			}

			for (Set<VisualScript::DataConnection>::Element *E = clipboard->data_connections.front(); E; E = E->next()) {
				undo_redo->add_do_method(script.ptr(), "data_connect", default_func, remap[E->get().from_node], E->get().from_port, remap[E->get().to_node], E->get().to_port);
				undo_redo->add_undo_method(script.ptr(), "data_disconnect", default_func, remap[E->get().from_node], E->get().from_port, remap[E->get().to_node], E->get().to_port);
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
		case EDIT_CREATE_FUNCTION: {
			StringName function = "";
			Map<int, Ref<VisualScriptNode>> nodes;
			Set<int> selections;
			for (int i = 0; i < graph->get_child_count(); i++) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
				if (gn) {
					if (gn->is_selected()) {
						int id = String(gn->get_name()).to_int();
						StringName func = _get_function_of_node(id);
						Ref<VisualScriptNode> node = script->get_node(func, id);
						if (Object::cast_to<VisualScriptFunction>(*node)) {
							EditorNode::get_singleton()->show_warning(TTR("Can't create function with a function node."));
							return;
						}
						if (node.is_valid()) {
							if (func != function && function != StringName("")) {
								EditorNode::get_singleton()->show_warning(TTR("Can't create function of nodes from nodes of multiple functions."));
								return;
							}
							nodes.insert(id, node);
							selections.insert(id);
							function = func;
						}
					}
				}
			}

			if (nodes.size() == 0) {
				return; // nothing to be done if there are no valid nodes selected
			}

			Set<VisualScript::SequenceConnection> seqmove;
			Set<VisualScript::DataConnection> datamove;

			Set<VisualScript::SequenceConnection> seqext;
			Set<VisualScript::DataConnection> dataext;

			int start_node = -1;
			Set<int> end_nodes;
			if (nodes.size() == 1) {
				Ref<VisualScriptNode> nd = script->get_node(function, nodes.front()->key());
				if (nd.is_valid() && nd->has_input_sequence_port()) {
					start_node = nodes.front()->key();
				} else {
					EditorNode::get_singleton()->show_warning(TTR("Select at least one node with sequence port."));
					return;
				}
			} else {
				List<VisualScript::SequenceConnection> seqs;
				script->get_sequence_connection_list(function, &seqs);

				if (seqs.size() == 0) {
					// in case there are no sequence connections
					// select the top most node cause that's probably how
					// the user wants to connect the nodes
					int top_nd = -1;
					Vector2 top;
					for (Map<int, Ref<VisualScriptNode>>::Element *E = nodes.front(); E; E = E->next()) {
						Ref<VisualScriptNode> nd = script->get_node(function, E->key());
						if (nd.is_valid() && nd->has_input_sequence_port()) {
							if (top_nd < 0) {
								top_nd = E->key();
								top = script->get_node_position(function, top_nd);
							}
							Vector2 pos = script->get_node_position(function, E->key());
							if (top.y > pos.y) {
								top_nd = E->key();
								top = pos;
							}
						}
					}
					Ref<VisualScriptNode> nd = script->get_node(function, top_nd);
					if (nd.is_valid() && nd->has_input_sequence_port()) {
						start_node = top_nd;
					} else {
						EditorNode::get_singleton()->show_warning(TTR("Select at least one node with sequence port."));
						return;
					}
				} else {
					// pick the node with input sequence
					Set<int> nodes_from;
					Set<int> nodes_to;
					for (List<VisualScript::SequenceConnection>::Element *E = seqs.front(); E; E = E->next()) {
						if (nodes.has(E->get().from_node) && nodes.has(E->get().to_node)) {
							seqmove.insert(E->get());
							nodes_from.insert(E->get().from_node);
						} else if (nodes.has(E->get().from_node) && !nodes.has(E->get().to_node)) {
							seqext.insert(E->get());
						} else if (!nodes.has(E->get().from_node) && nodes.has(E->get().to_node)) {
							if (start_node == -1) {
								seqext.insert(E->get());
								start_node = E->get().to_node;
							} else {
								EditorNode::get_singleton()->show_warning(TTR("Try to only have one sequence input in selection."));
								return;
							}
						}
						nodes_to.insert(E->get().to_node);
					}

					// to use to add return nodes
					_get_ends(start_node, seqs, selections, end_nodes);

					if (start_node == -1) {
						// if we still don't have a start node then
						// run through the nodes and select the first tree node
						// ie node without any input sequence but output sequence
						for (Set<int>::Element *E = nodes_from.front(); E; E = E->next()) {
							if (!nodes_to.has(E->get())) {
								start_node = E->get();
							}
						}
					}
				}
			}

			if (start_node == -1) {
				return; // this should not happen, but just in case something goes wrong
			}

			List<Variant::Type> inputs; // input types
			List<Pair<int, int>> input_connections;
			{
				List<VisualScript::DataConnection> dats;
				script->get_data_connection_list(function, &dats);
				for (List<VisualScript::DataConnection>::Element *E = dats.front(); E; E = E->next()) {
					if (nodes.has(E->get().from_node) && nodes.has(E->get().to_node)) {
						datamove.insert(E->get());
					} else if (!nodes.has(E->get().from_node) && nodes.has(E->get().to_node)) {
						// add all these as inputs for the Function
						Ref<VisualScriptNode> node = script->get_node(function, E->get().to_node);
						if (node.is_valid()) {
							dataext.insert(E->get());
							PropertyInfo pi = node->get_input_value_port_info(E->get().to_port);
							inputs.push_back(pi.type);
							input_connections.push_back(Pair<int, int>(E->get().to_node, E->get().to_port));
						}
					} else if (nodes.has(E->get().from_node) && !nodes.has(E->get().to_node)) {
						dataext.insert(E->get());
					}
				}
			}

			String new_fn = _validate_name("new_function");

			Vector2 pos = _get_available_pos(false, script->get_node_position(function, start_node) - Vector2(80, 150));

			Ref<VisualScriptFunction> func_node;
			func_node.instance();
			func_node->set_name(new_fn);

			undo_redo->create_action(TTR("Create Function"));

			undo_redo->add_do_method(script.ptr(), "add_function", new_fn);
			int fn_id = script->get_available_id();
			undo_redo->add_do_method(script.ptr(), "add_node", new_fn, fn_id, func_node, pos);
			undo_redo->add_undo_method(script.ptr(), "remove_function", new_fn);
			undo_redo->add_do_method(this, "_update_members");
			undo_redo->add_undo_method(this, "_update_members");
			undo_redo->add_do_method(this, "emit_signal", "edited_script_changed");
			undo_redo->add_undo_method(this, "emit_signal", "edited_script_changed");

			// Move the nodes

			for (Map<int, Ref<VisualScriptNode>>::Element *E = nodes.front(); E; E = E->next()) {
				undo_redo->add_do_method(script.ptr(), "remove_node", function, E->key());
				undo_redo->add_do_method(script.ptr(), "add_node", new_fn, E->key(), E->get(), script->get_node_position(function, E->key()));

				// undo_redo->add_undo_method(script.ptr(), "remove_node", new_fn, E->key()); not needed cause we already remove the function :P
				undo_redo->add_undo_method(script.ptr(), "add_node", function, E->key(), E->get(), script->get_node_position(function, E->key()));
			}

			for (Set<VisualScript::SequenceConnection>::Element *E = seqmove.front(); E; E = E->next()) {
				undo_redo->add_do_method(script.ptr(), "sequence_connect", new_fn, E->get().from_node, E->get().from_output, E->get().to_node);
				undo_redo->add_undo_method(script.ptr(), "sequence_connect", function, E->get().from_node, E->get().from_output, E->get().to_node);
			}

			for (Set<VisualScript::DataConnection>::Element *E = datamove.front(); E; E = E->next()) {
				undo_redo->add_do_method(script.ptr(), "data_connect", new_fn, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
				undo_redo->add_undo_method(script.ptr(), "data_connect", function, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}

			// Add undo for external connections as well so that it's easier to revert back and forth
			// these didn't require do methods as it's already handled internally by other do calls
			for (Set<VisualScript::SequenceConnection>::Element *E = seqext.front(); E; E = E->next()) {
				undo_redo->add_undo_method(script.ptr(), "sequence_connect", function, E->get().from_node, E->get().from_output, E->get().to_node);
			}
			for (Set<VisualScript::DataConnection>::Element *E = dataext.front(); E; E = E->next()) {
				undo_redo->add_undo_method(script.ptr(), "data_connect", function, E->get().from_node, E->get().from_port, E->get().to_node, E->get().to_port);
			}

			// I don't really think we need support for non sequenced functions at this moment
			undo_redo->add_do_method(script.ptr(), "sequence_connect", new_fn, fn_id, 0, start_node);

			// end nodes are mapped to the return nodes with data connections if possible
			int m = 1;
			for (Set<int>::Element *G = end_nodes.front(); G; G = G->next()) {
				Ref<VisualScriptReturn> ret_node;
				ret_node.instance();

				int ret_id = fn_id + (m++);
				selections.insert(ret_id);
				Vector2 posi = _get_available_pos(false, script->get_node_position(function, G->get()) + Vector2(80, -100));
				undo_redo->add_do_method(script.ptr(), "add_node", new_fn, ret_id, ret_node, posi);
				undo_redo->add_undo_method(script.ptr(), "remove_node", new_fn, ret_id);

				undo_redo->add_do_method(script.ptr(), "sequence_connect", new_fn, G->get(), 0, ret_id);
				// add data outputs from each of the end_nodes
				Ref<VisualScriptNode> vsn = script->get_node(function, G->get());
				if (vsn.is_valid() && vsn->get_output_value_port_count() > 0) {
					ret_node->set_enable_return_value(true);
					// use the zeroth data port cause that's the likely one that is planned to be used
					ret_node->set_return_type(vsn->get_output_value_port_info(0).type);
					undo_redo->add_do_method(script.ptr(), "data_connect", new_fn, G->get(), 0, ret_id, 0);
				}
			}

			// * might make the system more intelligent by checking port from info.
			int i = 0;
			List<Pair<int, int>>::Element *F = input_connections.front();
			for (List<Variant::Type>::Element *E = inputs.front(); E && F; E = E->next(), F = F->next()) {
				func_node->add_argument(E->get(), "arg_" + String::num_int64(i), i);
				undo_redo->add_do_method(script.ptr(), "data_connect", new_fn, fn_id, i, F->get().first, F->get().second);
				i++; // increment i
			}

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");

			undo_redo->commit_action();

			// make sure all Nodes get marked for selection so that they can be moved together
			selections.insert(fn_id);
			for (int k = 0; k < graph->get_child_count(); k++) {
				GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(k));
				if (gn) {
					int id = gn->get_name().operator String().to_int();
					gn->set_selected(selections.has(id));
				}
			}

			// Ensure Preview Selection is of newly created function node
			if (selections.size()) {
				EditorNode::get_singleton()->push_item(func_node.ptr());
			}

		} break;
		case REFRESH_GRAPH: {
			_update_graph();
		} break;
	}
}

// this is likely going to be very slow and I am not sure if I should keep it
// but I hope that it will not be a problem considering that we won't be creating functions so frequently
// and cyclic connections would be a problem but hopefully we won't let them get to this point
void VisualScriptEditor::_get_ends(int p_node, const List<VisualScript::SequenceConnection> &p_seqs, const Set<int> &p_selected, Set<int> &r_end_nodes) {
	for (const List<VisualScript::SequenceConnection>::Element *E = p_seqs.front(); E; E = E->next()) {
		int from = E->get().from_node;
		int to = E->get().to_node;

		if (from == p_node && p_selected.has(to)) {
			// this is an interior connection move forward to the to node
			_get_ends(to, p_seqs, p_selected, r_end_nodes);
		} else if (from == p_node && !p_selected.has(to)) {
			r_end_nodes.insert(from);
		}
	}
}

void VisualScriptEditor::_member_rmb_selected(const Vector2 &p_pos) {
	TreeItem *ti = members->get_selected();
	ERR_FAIL_COND(!ti);

	member_popup->clear();
	member_popup->set_position(members->get_global_position() + p_pos);
	member_popup->set_size(Vector2());

	function_name_edit->set_position(members->get_global_position() + p_pos);
	function_name_edit->set_size(Vector2());

	TreeItem *root = members->get_root();

	Ref<Texture> del_icon = Control::get_icon("Remove", "EditorIcons");

	Ref<Texture> edit_icon = Control::get_icon("Edit", "EditorIcons");

	if (ti->get_parent() == root->get_children()) {
		member_type = MEMBER_FUNCTION;
		member_name = ti->get_text(0);
		member_popup->add_icon_shortcut(edit_icon, ED_GET_SHORTCUT("visual_script_editor/edit_member"), MEMBER_EDIT);
		member_popup->add_separator();
		member_popup->add_icon_shortcut(del_icon, ED_GET_SHORTCUT("visual_script_editor/delete_selected"), MEMBER_REMOVE);
		member_popup->popup();
		return;
	}

	if (ti->get_parent() == root->get_children()->get_next()) {
		member_type = MEMBER_VARIABLE;
		member_name = ti->get_text(0);
		member_popup->add_icon_shortcut(edit_icon, ED_GET_SHORTCUT("visual_script_editor/edit_member"), MEMBER_EDIT);
		member_popup->add_separator();
		member_popup->add_icon_shortcut(del_icon, ED_GET_SHORTCUT("visual_script_editor/delete_selected"), MEMBER_REMOVE);
		member_popup->popup();
		return;
	}

	if (ti->get_parent() == root->get_children()->get_next()->get_next()) {
		member_type = MEMBER_SIGNAL;
		member_name = ti->get_text(0);
		member_popup->add_icon_shortcut(edit_icon, ED_GET_SHORTCUT("visual_script_editor/edit_member"), MEMBER_EDIT);
		member_popup->add_separator();
		member_popup->add_icon_shortcut(del_icon, ED_GET_SHORTCUT("visual_script_editor/delete_selected"), MEMBER_REMOVE);
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

				undo_redo->add_do_method(this, "_update_members");
				undo_redo->add_undo_method(this, "_update_members");
				undo_redo->add_do_method(this, "_update_graph");
				undo_redo->add_undo_method(this, "_update_graph");
				undo_redo->commit_action();
			} else if (p_option == MEMBER_EDIT) {
				selected = members->get_selected()->get_text(0);
				function_name_edit->popup();
				function_name_box->set_text(selected);
				function_name_box->select_all();
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

void VisualScriptEditor::add_syntax_highlighter(SyntaxHighlighter *p_highlighter) {
}

void VisualScriptEditor::set_syntax_highlighter(SyntaxHighlighter *p_highlighter) {
}

void VisualScriptEditor::_bind_methods() {
	ClassDB::bind_method("_member_button", &VisualScriptEditor::_member_button);
	ClassDB::bind_method("_member_edited", &VisualScriptEditor::_member_edited);
	ClassDB::bind_method("_member_selected", &VisualScriptEditor::_member_selected);
	ClassDB::bind_method("_update_members", &VisualScriptEditor::_update_members);
	ClassDB::bind_method("_members_gui_input", &VisualScriptEditor::_members_gui_input);
	ClassDB::bind_method("_member_rmb_selected", &VisualScriptEditor::_member_rmb_selected);
	ClassDB::bind_method("_member_option", &VisualScriptEditor::_member_option);
	ClassDB::bind_method("_fn_name_box_input", &VisualScriptEditor::_fn_name_box_input);

	ClassDB::bind_method("_change_base_type", &VisualScriptEditor::_change_base_type);
	ClassDB::bind_method("_change_base_type_callback", &VisualScriptEditor::_change_base_type_callback);
	ClassDB::bind_method("_toggle_tool_script", &VisualScriptEditor::_toggle_tool_script);
	ClassDB::bind_method("_node_selected", &VisualScriptEditor::_node_selected);
	ClassDB::bind_method("_node_moved", &VisualScriptEditor::_node_moved);
	ClassDB::bind_method("_move_node", &VisualScriptEditor::_move_node);
	ClassDB::bind_method("_begin_node_move", &VisualScriptEditor::_begin_node_move);
	ClassDB::bind_method("_end_node_move", &VisualScriptEditor::_end_node_move);
	ClassDB::bind_method("_remove_node", &VisualScriptEditor::_remove_node);
	ClassDB::bind_method("_update_graph", &VisualScriptEditor::_update_graph, DEFVAL(-1));
	ClassDB::bind_method("_node_ports_changed", &VisualScriptEditor::_node_ports_changed);

	ClassDB::bind_method("_create_function_dialog", &VisualScriptEditor::_create_function_dialog);
	ClassDB::bind_method("_create_function", &VisualScriptEditor::_create_function);
	ClassDB::bind_method("_add_node_dialog", &VisualScriptEditor::_add_node_dialog);
	ClassDB::bind_method("_add_func_input", &VisualScriptEditor::_add_func_input);
	ClassDB::bind_method("_remove_func_input", &VisualScriptEditor::_remove_func_input);
	ClassDB::bind_method("_deselect_input_names", &VisualScriptEditor::_deselect_input_names);

	ClassDB::bind_method("_default_value_edited", &VisualScriptEditor::_default_value_edited);
	ClassDB::bind_method("_default_value_changed", &VisualScriptEditor::_default_value_changed);
	ClassDB::bind_method("_menu_option", &VisualScriptEditor::_menu_option);
	ClassDB::bind_method("_graph_ofs_changed", &VisualScriptEditor::_graph_ofs_changed);
	ClassDB::bind_method("_center_on_node", &VisualScriptEditor::_center_on_node);
	ClassDB::bind_method("_comment_node_resized", &VisualScriptEditor::_comment_node_resized);
	ClassDB::bind_method("_button_resource_previewed", &VisualScriptEditor::_button_resource_previewed);
	ClassDB::bind_method("_port_action_menu", &VisualScriptEditor::_port_action_menu);
	ClassDB::bind_method("_selected_connect_node", &VisualScriptEditor::_selected_connect_node);
	ClassDB::bind_method("_selected_new_virtual_method", &VisualScriptEditor::_selected_new_virtual_method);

	ClassDB::bind_method("_cancel_connect_node", &VisualScriptEditor::_cancel_connect_node);
	ClassDB::bind_method("_create_new_node_from_name", &VisualScriptEditor::_create_new_node_from_name);
	ClassDB::bind_method("_expression_text_changed", &VisualScriptEditor::_expression_text_changed);
	ClassDB::bind_method("_add_input_port", &VisualScriptEditor::_add_input_port);
	ClassDB::bind_method("_add_output_port", &VisualScriptEditor::_add_output_port);
	ClassDB::bind_method("_remove_input_port", &VisualScriptEditor::_remove_input_port);
	ClassDB::bind_method("_remove_output_port", &VisualScriptEditor::_remove_output_port);
	ClassDB::bind_method("_change_port_type", &VisualScriptEditor::_change_port_type);
	ClassDB::bind_method("_update_node_size", &VisualScriptEditor::_update_node_size);
	ClassDB::bind_method("_port_name_focus_out", &VisualScriptEditor::_port_name_focus_out);

	ClassDB::bind_method("get_drag_data_fw", &VisualScriptEditor::get_drag_data_fw);
	ClassDB::bind_method("can_drop_data_fw", &VisualScriptEditor::can_drop_data_fw);
	ClassDB::bind_method("drop_data_fw", &VisualScriptEditor::drop_data_fw);

	ClassDB::bind_method("_input", &VisualScriptEditor::_input);
	ClassDB::bind_method("_graph_gui_input", &VisualScriptEditor::_graph_gui_input);

	ClassDB::bind_method("_on_nodes_delete", &VisualScriptEditor::_on_nodes_delete);
	ClassDB::bind_method("_on_nodes_duplicate", &VisualScriptEditor::_on_nodes_duplicate);

	ClassDB::bind_method("_hide_timer", &VisualScriptEditor::_hide_timer);

	ClassDB::bind_method("_graph_connected", &VisualScriptEditor::_graph_connected);
	ClassDB::bind_method("_graph_disconnected", &VisualScriptEditor::_graph_disconnected);
	ClassDB::bind_method("_graph_connect_to_empty", &VisualScriptEditor::_graph_connect_to_empty);

	ClassDB::bind_method("_update_graph_connections", &VisualScriptEditor::_update_graph_connections);

	ClassDB::bind_method("_selected_method", &VisualScriptEditor::_selected_method);
	ClassDB::bind_method("_draw_color_over_button", &VisualScriptEditor::_draw_color_over_button);

	ClassDB::bind_method("_generic_search", &VisualScriptEditor::_generic_search);
}

VisualScriptEditor::VisualScriptEditor() {
	if (!clipboard) {
		clipboard = memnew(Clipboard);
	}
	updating_graph = false;
	saved_pos_dirty = false;
	saved_position = Vector2(0, 0);

	edit_menu = memnew(MenuButton);
	edit_menu->set_text(TTR("Edit"));
	edit_menu->set_switch_on_hover(true);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/delete_selected"), EDIT_DELETE_NODES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/toggle_breakpoint"), EDIT_TOGGLE_BREAKPOINT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/find_node_type"), EDIT_FIND_NODE_TYPE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/copy_nodes"), EDIT_COPY_NODES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/cut_nodes"), EDIT_CUT_NODES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/paste_nodes"), EDIT_PASTE_NODES);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/create_function"), EDIT_CREATE_FUNCTION);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("visual_script_editor/refresh_nodes"), REFRESH_GRAPH);
	edit_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	members_section = memnew(VBoxContainer);
	// Add but wait until done setting up this.
	ScriptEditor::get_singleton()->get_left_list_split()->call_deferred("add_child", members_section);
	members_section->set_v_size_flags(SIZE_EXPAND_FILL);

	CheckButton *tool_script_check = memnew(CheckButton);
	tool_script_check->set_text(TTR("Make Tool:"));
	members_section->add_child(tool_script_check);
	tool_script_check->connect("pressed", this, "_toggle_tool_script");

	///       Members        ///

	members = memnew(Tree);
	members_section->add_margin_child(TTR("Members:"), members, true);
	members->set_custom_minimum_size(Size2(0, 50 * EDSCALE));
	members->set_hide_root(true);
	members->connect("button_pressed", this, "_member_button");
	members->connect("item_edited", this, "_member_edited");
	members->connect("cell_selected", this, "_member_selected", varray(), CONNECT_DEFERRED);
	members->connect("gui_input", this, "_members_gui_input");
	members->connect("item_rmb_selected", this, "_member_rmb_selected");
	members->set_allow_rmb_select(true);
	members->set_allow_reselect(true);
	members->set_hide_folding(true);
	members->set_drag_forwarding(this);

	member_popup = memnew(PopupMenu);
	add_child(member_popup);
	member_popup->connect("id_pressed", this, "_member_option");

	function_name_edit = memnew(PopupDialog);
	function_name_box = memnew(LineEdit);
	function_name_edit->add_child(function_name_box);
	function_name_edit->set_h_size_flags(SIZE_EXPAND);
	function_name_box->connect("gui_input", this, "_fn_name_box_input");
	function_name_box->set_expand_to_text_length(true);
	add_child(function_name_edit);

	///       Actual Graph          ///

	graph = memnew(GraphEdit);
	add_child(graph);
	graph->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	graph->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	graph->connect("node_selected", this, "_node_selected");
	graph->connect("_begin_node_move", this, "_begin_node_move");
	graph->connect("_end_node_move", this, "_end_node_move");
	graph->connect("delete_nodes_request", this, "_on_nodes_delete");
	graph->connect("duplicate_nodes_request", this, "_on_nodes_duplicate");
	graph->connect("gui_input", this, "_graph_gui_input");
	graph->set_drag_forwarding(this);
	float graph_minimap_opacity = EditorSettings::get_singleton()->get("editors/visual_editors/minimap_opacity");
	graph->set_minimap_opacity(graph_minimap_opacity);
	graph->hide();
	graph->connect("scroll_offset_changed", this, "_graph_ofs_changed");

	/// Add Buttons to Top Bar/Zoom bar.
	HBoxContainer *graph_hbc = graph->get_zoom_hbox();

	Label *base_lbl = memnew(Label);
	base_lbl->set_text(TTR("Change Base Type:") + " ");
	graph_hbc->add_child(base_lbl);

	base_type_select = memnew(Button);
	base_type_select->connect("pressed", this, "_change_base_type");
	graph_hbc->add_child(base_type_select);

	Button *add_nds = memnew(Button);
	add_nds->set_text(TTR("Add Nodes..."));
	graph_hbc->add_child(add_nds);
	add_nds->connect("pressed", this, "_add_node_dialog");

	Button *fn_btn = memnew(Button);
	fn_btn->set_text(TTR("Add Function..."));
	graph_hbc->add_child(fn_btn);
	fn_btn->connect("pressed", this, "_create_function_dialog");

	// Add Function Dialog.
	VBoxContainer *function_vb = memnew(VBoxContainer);
	function_vb->set_v_size_flags(SIZE_EXPAND_FILL);
	function_vb->set_custom_minimum_size(Size2(450, 300) * EDSCALE);

	HBoxContainer *func_name_hbox = memnew(HBoxContainer);
	function_vb->add_child(func_name_hbox);

	Label *func_name_label = memnew(Label);
	func_name_label->set_text(TTR("Name:"));
	func_name_hbox->add_child(func_name_label);

	func_name_box = memnew(LineEdit);
	func_name_box->set_h_size_flags(SIZE_EXPAND_FILL);
	func_name_box->set_placeholder(TTR("function_name"));
	func_name_box->set_text("");
	func_name_box->connect("focus_entered", this, "_deselect_input_names");
	func_name_hbox->add_child(func_name_box);

	// Add minor setting for function if needed, here!

	function_vb->add_child(memnew(HSeparator));

	Button *add_input_button = memnew(Button);
	add_input_button->set_h_size_flags(SIZE_EXPAND_FILL);
	add_input_button->set_text(TTR("Add Input"));
	add_input_button->connect("pressed", this, "_add_func_input");
	function_vb->add_child(add_input_button);

	func_input_scroll = memnew(ScrollContainer);
	func_input_scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	function_vb->add_child(func_input_scroll);

	func_input_vbox = memnew(VBoxContainer);
	func_input_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
	func_input_scroll->add_child(func_input_vbox);

	function_create_dialog = memnew(ConfirmationDialog);
	function_create_dialog->set_v_size_flags(SIZE_EXPAND_FILL);
	function_create_dialog->set_title(TTR("Create Function"));
	function_create_dialog->add_child(function_vb);
	function_create_dialog->get_ok()->set_text(TTR("Create"));
	function_create_dialog->get_ok()->connect("pressed", this, "_create_function");
	add_child(function_create_dialog);

	select_func_text = memnew(Label);
	select_func_text->set_text(TTR("Select or create a function to edit its graph."));
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

	// Allowed casts (connections).
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

	signal_editor = memnew(VisualScriptEditorSignalEdit);
	edit_signal_edit = memnew(EditorInspector);
	edit_signal_dialog->add_child(edit_signal_edit);

	edit_signal_edit->edit(signal_editor);

	edit_variable_dialog = memnew(AcceptDialog);
	edit_variable_dialog->get_ok()->set_text(TTR("Close"));
	add_child(edit_variable_dialog);

	variable_editor = memnew(VisualScriptEditorVariableEdit);
	edit_variable_edit = memnew(EditorInspector);
	edit_variable_dialog->add_child(edit_variable_edit);

	edit_variable_edit->edit(variable_editor);

	select_base_type = memnew(CreateDialog);
	select_base_type->set_base_type("Object"); // Anything goes.
	select_base_type->connect("create", this, "_change_base_type_callback");
	add_child(select_base_type);

	undo_redo = EditorNode::get_singleton()->get_undo_redo();

	updating_members = false;

	set_process_input(true);
	set_process_unhandled_input(true);

	default_value_edit = memnew(CustomPropertyEditor);
	add_child(default_value_edit);
	default_value_edit->connect("variant_changed", this, "_default_value_changed");

	method_select = memnew(VisualScriptPropertySelector);
	add_child(method_select);
	method_select->connect("selected", this, "_selected_method");
	error_line = -1;

	new_connect_node_select = memnew(VisualScriptPropertySelector);
	add_child(new_connect_node_select);
	new_connect_node_select->set_resizable(true);
	new_connect_node_select->connect("selected", this, "_selected_connect_node");
	new_connect_node_select->get_cancel()->connect("pressed", this, "_cancel_connect_node");

	new_virtual_method_select = memnew(VisualScriptPropertySelector);
	add_child(new_virtual_method_select);
	new_virtual_method_select->connect("selected", this, "_selected_new_virtual_method");
}

VisualScriptEditor::~VisualScriptEditor() {
	undo_redo->clear_history(); // Avoid crashes.
	memdelete(signal_editor);
	memdelete(variable_editor);
}

static ScriptEditorBase *create_editor(const RES &p_resource) {
	if (Object::cast_to<VisualScript>(*p_resource)) {
		return memnew(VisualScriptEditor);
	}

	return nullptr;
}

VisualScriptEditor::Clipboard *VisualScriptEditor::clipboard = nullptr;

void VisualScriptEditor::free_clipboard() {
	if (clipboard) {
		memdelete(clipboard);
	}
}

static void register_editor_callback() {
	ScriptEditor::register_create_script_editor_function(create_editor);

	ED_SHORTCUT("visual_script_editor/delete_selected", TTR("Delete Selected"), KEY_DELETE);
	ED_SHORTCUT("visual_script_editor/toggle_breakpoint", TTR("Toggle Breakpoint"), KEY_F9);
	ED_SHORTCUT("visual_script_editor/find_node_type", TTR("Find Node Type"), KEY_MASK_CMD + KEY_F);
	ED_SHORTCUT("visual_script_editor/copy_nodes", TTR("Copy Nodes"), KEY_MASK_CMD + KEY_C);
	ED_SHORTCUT("visual_script_editor/cut_nodes", TTR("Cut Nodes"), KEY_MASK_CMD + KEY_X);
	ED_SHORTCUT("visual_script_editor/paste_nodes", TTR("Paste Nodes"), KEY_MASK_CMD + KEY_V);
	ED_SHORTCUT("visual_script_editor/create_function", TTR("Make Function"), KEY_MASK_CMD + KEY_G);
	ED_SHORTCUT("visual_script_editor/refresh_nodes", TTR("Refresh Graph"), KEY_MASK_CMD + KEY_R);
	ED_SHORTCUT("visual_script_editor/edit_member", TTR("Edit Member"), KEY_MASK_CMD + KEY_E);
}

void VisualScriptEditor::register_editor() {
	// Too early to register stuff here, request a callback.
	EditorNode::add_plugin_init_callback(register_editor_callback);
}

Ref<VisualScriptNode> _VisualScriptEditor::create_node_custom(const String &p_name) {
	Ref<VisualScriptCustomNode> node;
	node.instance();
	node->set_script(singleton->custom_nodes[p_name]);
	return node;
}

_VisualScriptEditor *_VisualScriptEditor::singleton = nullptr;
Map<String, RefPtr> _VisualScriptEditor::custom_nodes;

_VisualScriptEditor::_VisualScriptEditor() {
	singleton = this;
}

_VisualScriptEditor::~_VisualScriptEditor() {
	custom_nodes.clear();
}

void _VisualScriptEditor::add_custom_node(const String &p_name, const String &p_category, const Ref<Script> &p_script) {
	String node_name = "custom/" + p_category + "/" + p_name;
	custom_nodes.insert(node_name, p_script.get_ref_ptr());
	VisualScriptLanguage::singleton->add_register_func(node_name, &_VisualScriptEditor::create_node_custom);
	emit_signal("custom_nodes_updated");
}

void _VisualScriptEditor::remove_custom_node(const String &p_name, const String &p_category) {
	String node_name = "custom/" + p_category + "/" + p_name;
	custom_nodes.erase(node_name);
	VisualScriptLanguage::singleton->remove_register_func(node_name);
	emit_signal("custom_nodes_updated");
}

void _VisualScriptEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_custom_node", "name", "category", "script"), &_VisualScriptEditor::add_custom_node);
	ClassDB::bind_method(D_METHOD("remove_custom_node", "name", "category"), &_VisualScriptEditor::remove_custom_node);
	ADD_SIGNAL(MethodInfo("custom_nodes_updated"));
}

void VisualScriptEditor::validate() {
}
#endif
