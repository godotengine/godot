/**************************************************************************/
/*  shader_graph.cpp                                                      */
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

#include "shader_graph.h"

#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/templates/rb_map.h"
#include "scene/resources/visual_shader.h"
#include "scene/resources/visual_shader_group.h"
#include "scene/resources/visual_shader_nodes.h"

void ShaderGraph::_node_changed() {
	emit_signal("graph_changed");
}

void ShaderGraph::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_node", "node", "position", "id"), &ShaderGraph::add_node);
	ClassDB::bind_method(D_METHOD("get_node", "id"), &ShaderGraph::get_node);

	ClassDB::bind_method(D_METHOD("set_node_position", "id", "position"), &ShaderGraph::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "id"), &ShaderGraph::get_node_position);

	ClassDB::bind_method(D_METHOD("get_node_list"), &ShaderGraph::get_node_ids);
	ClassDB::bind_method(D_METHOD("get_valid_node_id"), &ShaderGraph::get_valid_node_id);

	ClassDB::bind_method(D_METHOD("remove_node", "id"), &ShaderGraph::remove_node);
	ClassDB::bind_method(D_METHOD("replace_node", "id", "new_class"), &ShaderGraph::replace_node);

	ClassDB::bind_method(D_METHOD("is_node_connection", "from_node", "from_port", "to_node", "to_port"), &ShaderGraph::are_nodes_connected);
	ClassDB::bind_method(D_METHOD("can_connect_nodes", "from_node", "from_port", "to_node", "to_port"), &ShaderGraph::can_connect_nodes);

	ClassDB::bind_method(D_METHOD("connect_nodes", "from_node", "from_port", "to_node", "to_port"), &ShaderGraph::connect_nodes);
	ClassDB::bind_method(D_METHOD("disconnect_nodes", "from_node", "from_port", "to_node", "to_port"), &ShaderGraph::disconnect_nodes);
	ClassDB::bind_method(D_METHOD("connect_nodes_forced", "from_node", "from_port", "to_node", "to_port"), &ShaderGraph::connect_nodes_forced);

	ClassDB::bind_method(D_METHOD("attach_node_to_frame", "id", "frame"), &ShaderGraph::attach_node_to_frame);
	ClassDB::bind_method(D_METHOD("detach_node_from_frame", "id"), &ShaderGraph::detach_node_from_frame);

	ADD_SIGNAL(MethodInfo("graph_changed"));
}

bool ShaderGraph::_set(const StringName &p_name, const Variant &p_value) {
	const String prop_name_str = p_name;
	if (prop_name_str.begins_with("nodes/")) {
		String index = prop_name_str.get_slicec('/', 1);

		if (index == "connections") {
			Vector<int> conns = p_value;
			if (conns.size() % 4 == 0) {
				for (int i = 0; i < conns.size(); i += 4) {
					connect_nodes_forced(conns[i + 0], conns[i + 1], conns[i + 2], conns[i + 3]);
				}
			}
			return true;
		}

		const int id = index.to_int();
		const String node_info = prop_name_str.get_slicec('/', 2);

		if (node_info == "node") {
			add_node(p_value, Vector2(), id);
			return true;
		} else if (node_info == "position") {
			set_node_position(id, p_value);
			return true;
		} else if (node_info == "size") {
			VisualShaderNodeResizableBase *resizable_vn = Object::cast_to<VisualShaderNodeResizableBase>(get_node(id).ptr());
			if (resizable_vn) {
				resizable_vn->set_size(p_value);
				return true;
			}
		} else if (node_info == "input_ports") {
			VisualShaderNodeGroupBase *group_vn = Object::cast_to<VisualShaderNodeGroupBase>(get_node(id).ptr());
			if (group_vn) {
				group_vn->set_inputs(p_value);
				return true;
			}
		} else if (node_info == "output_ports") {
			VisualShaderNodeGroupBase *group_vn = Object::cast_to<VisualShaderNodeGroupBase>(get_node(id).ptr());
			if (group_vn) {
				group_vn->set_outputs(p_value);
				return true;
			}
		} else if (node_info == "expression") {
			VisualShaderNodeExpression *expression_vn = Object::cast_to<VisualShaderNodeExpression>(get_node(id).ptr());
			if (expression_vn) {
				expression_vn->set_expression(p_value);
				return true;
			}
		}
	}
	return false;
}

bool ShaderGraph::_get(const StringName &p_name, Variant &r_ret) const {
	const String prop_name = p_name;
	if (prop_name.begins_with("nodes/")) {
		const String index = prop_name.get_slicec('/', 1);
		if (index == "connections") {
			Vector<int> conns;
			for (const ShaderGraph::Connection &E : connections) {
				conns.push_back(E.from_node);
				conns.push_back(E.from_port);
				conns.push_back(E.to_node);
				conns.push_back(E.to_port);
			}

			r_ret = conns;
			return true;
		}

		const int id = index.to_int();
		const String node_info = prop_name.get_slicec('/', 2);

		if (node_info == "node") {
			r_ret = get_node(id);
			return true;
		} else if (node_info == "position") {
			r_ret = get_node_position(id);
			return true;
		} else if (node_info == "size") {
			VisualShaderNodeResizableBase *resizable_vn = Object::cast_to<VisualShaderNodeResizableBase>(get_node(id).ptr());
			if (resizable_vn) {
				r_ret = resizable_vn->get_size();
				return true;
			}
		} else if (node_info == "input_ports") {
			VisualShaderNodeGroupBase *group_vn = Object::cast_to<VisualShaderNodeGroupBase>(get_node(id).ptr());
			if (group_vn) {
				r_ret = group_vn->get_inputs();
				return true;
			}
		} else if (node_info == "output_ports") {
			VisualShaderNodeGroupBase *group_vn = Object::cast_to<VisualShaderNodeGroupBase>(get_node(id).ptr());
			if (group_vn) {
				r_ret = group_vn->get_outputs();
				return true;
			}
		} else if (node_info == "expression") {
			VisualShaderNodeExpression *expression_vn = Object::cast_to<VisualShaderNodeExpression>(get_node(id).ptr());
			if (expression_vn) {
				r_ret = expression_vn->get_expression();
				return true;
			}
		}
	}
	return false;
}

void ShaderGraph::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const KeyValue<int, ShaderGraph::Node> &E : nodes) {
		String prop_name = "nodes/";
		prop_name += itos(E.key);

		if (E.key >= reserved_node_ids) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + "/node", PROPERTY_HINT_RESOURCE_TYPE, VisualShaderNode::get_class_static(), PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_ALWAYS_DUPLICATE));
		}
		p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));

		if (Object::cast_to<VisualShaderNodeGroupBase>(E.value.node.ptr()) != nullptr) {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
			p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/input_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
			p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/output_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		}
		if (Object::cast_to<VisualShaderNodeExpression>(E.value.node.ptr()) != nullptr) {
			p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/expression", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		}
	}
	p_list->push_back(PropertyInfo(Variant::PACKED_INT32_ARRAY, "nodes/connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
}

void ShaderGraph::_write_node(
		StringBuilder *p_global_code,
		StringBuilder *p_global_code_per_node,
		HashMap<Type, StringBuilder> *p_global_code_per_func,
		StringBuilder &r_code, Vector<ShaderGraph::DefaultTextureParam> &r_def_tex_params,
		const HashMap<ConnectionKey, const List<ShaderGraph::Connection>::Element *> &p_input_connections,
		const HashMap<ConnectionKey, const List<ShaderGraph::Connection>::Element *> &p_output_connections,
		int p_node,
		HashSet<int> &r_processed,
		bool p_for_preview,
		HashSet<StringName> &r_classes,
		Type p_type,
		Shader::Mode p_mode) const {
	const Ref<VisualShaderNode> vsnode = nodes[p_node].node;

	if (vsnode->is_disabled()) {
		r_code += "// " + vsnode->get_caption() + ":" + itos(p_node) + "\n";
		r_code += "// Node is disabled and code is not generated.\n";
		return;
	}

	// Check inputs recursively first.
	int input_count = vsnode->get_input_port_count();
	for (int i = 0; i < input_count; i++) {
		ConnectionKey ck;
		ck.node = p_node;
		ck.port = i;

		if (p_input_connections.has(ck)) {
			int from_node = p_input_connections[ck]->get().from_node;
			if (r_processed.has(from_node)) {
				continue;
			}

			_write_node(p_global_code, p_global_code_per_node, p_global_code_per_func, r_code, r_def_tex_params, p_input_connections, p_output_connections, from_node, r_processed, p_for_preview, r_classes, p_type, p_mode);
		}
	}

	// Then this node.

	Vector<ShaderGraph::DefaultTextureParam> params = vsnode->get_default_texture_parameters((VisualShader::Type)p_type, p_node);
	for (int i = 0; i < params.size(); i++) {
		r_def_tex_params.push_back(params[i]);
	}

	Ref<VisualShaderNodeInput> input = vsnode;
	bool skip_global = input.is_valid() && p_for_preview;

	if (!skip_global) {
		Ref<VisualShaderNodeParameter> parameter = vsnode;
		if (!parameter.is_valid() || !parameter->is_global_code_generated()) {
			if (p_global_code) {
				*p_global_code += vsnode->generate_global(p_mode, (VisualShader::Type)p_type, p_node);
			}
		}

		String class_name = vsnode->get_class_name();
		if (class_name == "VisualShaderNodeCustom") {
			class_name = vsnode->get_script_instance()->get_script()->get_path();
		}
		if (!r_classes.has(class_name)) {
			if (p_global_code_per_node) {
				*p_global_code_per_node += vsnode->generate_global_per_node(p_mode, p_node);
			}
			for (int i = 0; i < VisualShader::TYPE_MAX; i++) {
				if (p_global_code_per_func) {
					(*p_global_code_per_func)[Type(i)] += vsnode->generate_global_per_func(p_mode, VisualShader::Type(i), p_node);
				}
			}
			r_classes.insert(class_name);
		}

		// Generate node group functions only once globally.
		Ref<VisualShaderNodeGroup> group = vsnode;
		if (group.is_valid()) {
			const String group_key = "GROUP_" + group->get_group()->get_unique_func_name();
			if (!r_classes.has(group_key)) {
				if (p_global_code_per_node) {
					*p_global_code_per_node += group->generate_group_function(p_mode, (VisualShader::Type)p_type, p_node);
				}
				r_classes.insert(group_key);
			}
		}
	}

	if (!vsnode->is_code_generated()) { // Just generate globals and ignore locals.
		r_processed.insert(p_node);
		return;
	}

	String node_name = "// " + vsnode->get_caption() + ":" + itos(p_node) + "\n";
	String node_code;
	Vector<String> input_vars;

	input_vars.resize(vsnode->get_input_port_count());
	String *inputs = input_vars.ptrw();

	for (int i = 0; i < input_count; i++) {
		ConnectionKey ck;
		ck.node = p_node;
		ck.port = i;

		if (p_input_connections.has(ck)) {
			// Connected to something, use that output.
			int from_node = p_input_connections[ck]->get().from_node;

			if (nodes[from_node].node->is_disabled()) {
				continue;
			}

			int from_port = p_input_connections[ck]->get().from_port;

			VisualShaderNode::PortType in_type = vsnode->get_input_port_type(i);
			VisualShaderNode::PortType out_type = nodes[from_node].node->get_output_port_type(from_port);

			String src_var = "n_out" + itos(from_node) + "p" + itos(from_port);

			if (in_type == VisualShaderNode::PORT_TYPE_SAMPLER && out_type == VisualShaderNode::PORT_TYPE_SAMPLER) {
				VisualShaderNode *ptr = const_cast<VisualShaderNode *>(nodes[from_node].node.ptr());
				// In preview mode, GroupInput sampler parameters don't exist (no group function).
				Ref<VisualShaderNodeGroupInput> group_input = nodes[from_node].node;
				if (group_input.is_valid() && !p_for_preview) {
					inputs[i] = "p_" + ptr->get_output_port_name(from_port);
				} else if (ptr->has_method("get_input_real_name")) {
					inputs[i] = ptr->call("get_input_real_name");
				} else if (ptr->has_method("get_parameter_name")) {
					inputs[i] = ptr->call("get_parameter_name");
				} else {
					Ref<VisualShaderNodeReroute> reroute = nodes[from_node].node;
					if (reroute.is_valid()) {
						inputs[i] = get_reroute_parameter_name(from_node);
					} else {
						inputs[i] = "";
					}
				}
			} else if (in_type == out_type) {
				inputs[i] = src_var;
			} else {
				switch (in_type) {
					case VisualShaderNode::PORT_TYPE_SCALAR: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
								inputs[i] = "float(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
								inputs[i] = "float(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_BOOLEAN: {
								inputs[i] = "(" + src_var + " ? 1.0 : 0.0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
								inputs[i] = src_var + ".x";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
								inputs[i] = src_var + ".x";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
								inputs[i] = src_var + ".x";
							} break;
							default:
								break;
						}
					} break;
					case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR: {
								inputs[i] = "int(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
								inputs[i] = "int(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_BOOLEAN: {
								inputs[i] = "(" + src_var + " ? 1 : 0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
								inputs[i] = "int(" + src_var + ".x)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
								inputs[i] = "int(" + src_var + ".x)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
								inputs[i] = "int(" + src_var + ".x)";
							} break;
							default:
								break;
						}
					} break;
					case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR: {
								inputs[i] = "uint(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
								inputs[i] = "uint(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_BOOLEAN: {
								inputs[i] = "(" + src_var + " ? 1u : 0u)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
								inputs[i] = "uint(" + src_var + ".x)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
								inputs[i] = "uint(" + src_var + ".x)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
								inputs[i] = "uint(" + src_var + ".x)";
							} break;
							default:
								break;
						}
					} break;
					case VisualShaderNode::PORT_TYPE_BOOLEAN: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR: {
								inputs[i] = src_var + " > 0.0 ? true : false";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
								inputs[i] = src_var + " > 0 ? true : false";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
								inputs[i] = src_var + " > 0u ? true : false";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
								inputs[i] = "all(bvec2(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
								inputs[i] = "all(bvec3(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
								inputs[i] = "all(bvec4(" + src_var + "))";
							} break;
							default:
								break;
						}
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR: {
								inputs[i] = "vec2(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
								inputs[i] = "vec2(float(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
								inputs[i] = "vec2(float(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_BOOLEAN: {
								inputs[i] = "vec2(" + src_var + " ? 1.0 : 0.0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_3D:
							case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
								inputs[i] = "vec2(" + src_var + ".xy)";
							} break;
							default:
								break;
						}
					} break;

					case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR: {
								inputs[i] = "vec3(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
								inputs[i] = "vec3(float(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
								inputs[i] = "vec3(float(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_BOOLEAN: {
								inputs[i] = "vec3(" + src_var + " ? 1.0 : 0.0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
								inputs[i] = "vec3(" + src_var + ", 0.0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
								inputs[i] = "vec3(" + src_var + ".xyz)";
							} break;
							default:
								break;
						}
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
						switch (out_type) {
							case VisualShaderNode::PORT_TYPE_SCALAR: {
								inputs[i] = "vec4(" + src_var + ")";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
								inputs[i] = "vec4(float(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
								inputs[i] = "vec4(float(" + src_var + "))";
							} break;
							case VisualShaderNode::PORT_TYPE_BOOLEAN: {
								inputs[i] = "vec4(" + src_var + " ? 1.0 : 0.0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
								inputs[i] = "vec4(" + src_var + ", 0.0, 0.0)";
							} break;
							case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
								inputs[i] = "vec4(" + src_var + ", 0.0)";
							} break;
							default:
								break;
						}
					} break;
					default:
						break;
				}
			}
		} else {
			if (!vsnode->is_generate_input_var(i)) {
				continue;
			}

			Variant defval = vsnode->get_input_port_default_value(i);
			if (defval.get_type() == Variant::FLOAT) {
				float val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				node_code += "	float " + inputs[i] + " = " + vformat("%.5f", val) + ";\n";
			} else if (defval.get_type() == Variant::INT) {
				int val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				if (vsnode->get_input_port_type(i) == VisualShaderNode::PORT_TYPE_SCALAR_UINT) {
					node_code += "	uint " + inputs[i] + " = " + itos(val) + "u;\n";
				} else {
					node_code += "	int " + inputs[i] + " = " + itos(val) + ";\n";
				}
			} else if (defval.get_type() == Variant::BOOL) {
				bool val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				node_code += "	bool " + inputs[i] + " = " + (val ? "true" : "false") + ";\n";
			} else if (defval.get_type() == Variant::VECTOR2) {
				Vector2 val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				node_code += "	vec2 " + inputs[i] + " = " + vformat("vec2(%.5f, %.5f);\n", val.x, val.y);
			} else if (defval.get_type() == Variant::VECTOR3) {
				Vector3 val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				node_code += "	vec3 " + inputs[i] + " = " + vformat("vec3(%.5f, %.5f, %.5f);\n", val.x, val.y, val.z);
			} else if (defval.get_type() == Variant::VECTOR4) {
				Vector4 val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				node_code += "	vec4 " + inputs[i] + " = " + vformat("vec4(%.5f, %.5f, %.5f, %.5f);\n", val.x, val.y, val.z, val.w);
			} else if (defval.get_type() == Variant::QUATERNION) {
				Quaternion val = defval;
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				node_code += "	vec4 " + inputs[i] + " = " + vformat("vec4(%.5f, %.5f, %.5f, %.5f);\n", val.x, val.y, val.z, val.w);
			} else if (defval.get_type() == Variant::TRANSFORM3D) {
				Transform3D val = defval;
				val.basis.transpose();
				inputs[i] = "n_in" + itos(p_node) + "p" + itos(i);
				Array values;
				for (int j = 0; j < 3; j++) {
					values.push_back(val.basis[j].x);
					values.push_back(val.basis[j].y);
					values.push_back(val.basis[j].z);
				}
				values.push_back(val.origin.x);
				values.push_back(val.origin.y);
				values.push_back(val.origin.z);
				bool err = false;
				node_code += "	mat4 " + inputs[i] + " = " + String("mat4(vec4(%.5f, %.5f, %.5f, 0.0), vec4(%.5f, %.5f, %.5f, 0.0), vec4(%.5f, %.5f, %.5f, 0.0), vec4(%.5f, %.5f, %.5f, 1.0));\n").sprintf(values, &err);
			} else {
				// Will go empty, node is expected to know what it is doing at this point and handle it.
			}
		}
	}

	int output_count = vsnode->get_output_port_count();
	int initial_output_count = output_count;

	HashMap<int, bool> expanded_output_ports;

	for (int i = 0; i < initial_output_count; i++) {
		bool expanded = false;

		if (vsnode->is_output_port_expandable(i) && vsnode->_is_output_port_expanded(i)) {
			expanded = true;

			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
					output_count += 2;
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
					output_count += 3;
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
					output_count += 4;
				} break;
				default:
					break;
			}
		}
		expanded_output_ports.insert(i, expanded);
	}

	Vector<String> output_vars;
	output_vars.resize(output_count);
	String *outputs = output_vars.ptrw();

	if (vsnode->is_simple_decl()) { // Less code to generate for some simple_decl nodes.
		for (int i = 0, j = 0; i < initial_output_count; i++, j++) {
			String var_name = "n_out" + itos(p_node) + "p" + itos(j);
			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_SCALAR:
					outputs[i] = "float " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_SCALAR_INT:
					outputs[i] = "int " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_SCALAR_UINT:
					outputs[i] = "uint " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_2D:
					outputs[i] = "vec2 " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D:
					outputs[i] = "vec3 " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D:
					outputs[i] = "vec4 " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_BOOLEAN:
					outputs[i] = "bool " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_TRANSFORM:
					outputs[i] = "mat4 " + var_name;
					break;
				default:
					break;
			}
			if (expanded_output_ports[i]) {
				switch (vsnode->get_output_port_type(i)) {
					case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
						j += 2;
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
						j += 3;
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
						j += 4;
					} break;
					default:
						break;
				}
			}
		}

	} else {
		for (int i = 0, j = 0; i < initial_output_count; i++, j++) {
			outputs[i] = "n_out" + itos(p_node) + "p" + itos(j);
			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_SCALAR:
					r_code += "	float " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_SCALAR_INT:
					r_code += "	int " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_SCALAR_UINT:
					r_code += "	uint " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_2D:
					r_code += "	vec2 " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D:
					r_code += "	vec3 " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D:
					r_code += "	vec4 " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_BOOLEAN:
					r_code += "	bool " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_TRANSFORM:
					r_code += "	mat4 " + outputs[i] + ";\n";
					break;
				default:
					break;
			}
			if (expanded_output_ports[i]) {
				switch (vsnode->get_output_port_type(i)) {
					case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
						j += 2;
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
						j += 3;
					} break;
					case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
						j += 4;
					} break;
					default:
						break;
				}
			}
		}
	}

	// For group nodes, add default sampler2D uniforms for unconnected sampler group input ports.
	Ref<VisualShaderNodeGroup> group_node = vsnode;
	if (group_node.is_valid() && group_node->get_group().is_valid() && p_global_code) {
		for (int i = 0; i < group_node->get_input_port_count(); i++) {
			if (group_node->get_input_port_type(i) == VisualShaderNode::PORT_TYPE_SAMPLER && inputs[i].is_empty()) {
				String sampler_name = make_unique_id((VisualShader::Type)p_type, p_node, "sampler_" + itos(i));
				*p_global_code += "uniform sampler2D " + sampler_name + ";\n";
			}
		}
	}

	node_code += vsnode->generate_code(p_mode, (VisualShader::Type)p_type, p_node, inputs, outputs, p_for_preview);
	if (!node_code.is_empty()) {
		r_code += node_name;
		r_code += node_code;
	}

	for (int i = 0; i < output_count; i++) {
		if (expanded_output_ports[i]) {
			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
					if (vsnode->is_output_port_connected(i + 1) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 1))) { // red-component
						String r = "n_out" + itos(p_node) + "p" + itos(i + 1);
						r_code += "	float " + r + " = n_out" + itos(p_node) + "p" + itos(i) + ".r;\n";
						outputs[i + 1] = r;
					}

					if (vsnode->is_output_port_connected(i + 2) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 2))) { // green-component
						String g = "n_out" + itos(p_node) + "p" + itos(i + 2);
						r_code += "	float " + g + " = n_out" + itos(p_node) + "p" + itos(i) + ".g;\n";
						outputs[i + 2] = g;
					}

					i += 2;
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
					if (vsnode->is_output_port_connected(i + 1) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 1))) { // red-component
						String r = "n_out" + itos(p_node) + "p" + itos(i + 1);
						r_code += "	float " + r + " = n_out" + itos(p_node) + "p" + itos(i) + ".r;\n";
						outputs[i + 1] = r;
					}

					if (vsnode->is_output_port_connected(i + 2) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 2))) { // green-component
						String g = "n_out" + itos(p_node) + "p" + itos(i + 2);
						r_code += "	float " + g + " = n_out" + itos(p_node) + "p" + itos(i) + ".g;\n";
						outputs[i + 2] = g;
					}

					if (vsnode->is_output_port_connected(i + 3) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 3))) { // blue-component
						String b = "n_out" + itos(p_node) + "p" + itos(i + 3);
						r_code += "	float " + b + " = n_out" + itos(p_node) + "p" + itos(i) + ".b;\n";
						outputs[i + 3] = b;
					}

					i += 3;
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
					if (vsnode->is_output_port_connected(i + 1) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 1))) { // red-component
						String r = "n_out" + itos(p_node) + "p" + itos(i + 1);
						r_code += "	float " + r + " = n_out" + itos(p_node) + "p" + itos(i) + ".r;\n";
						outputs[i + 1] = r;
					}

					if (vsnode->is_output_port_connected(i + 2) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 2))) { // green-component
						String g = "n_out" + itos(p_node) + "p" + itos(i + 2);
						r_code += "	float " + g + " = n_out" + itos(p_node) + "p" + itos(i) + ".g;\n";
						outputs[i + 2] = g;
					}

					if (vsnode->is_output_port_connected(i + 3) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 3))) { // blue-component
						String b = "n_out" + itos(p_node) + "p" + itos(i + 3);
						r_code += "	float " + b + " = n_out" + itos(p_node) + "p" + itos(i) + ".b;\n";
						outputs[i + 3] = b;
					}

					if (vsnode->is_output_port_connected(i + 4) || (p_for_preview && vsnode->get_output_port_for_preview() == (i + 4))) { // alpha-component
						String a = "n_out" + itos(p_node) + "p" + itos(i + 4);
						r_code += "	float " + a + " = n_out" + itos(p_node) + "p" + itos(i) + ".a;\n";
						outputs[i + 4] = a;
					}

					i += 4;
				} break;
				default:
					break;
			}
		}
	}

	if (!node_code.is_empty()) {
		r_code += "\n\n";
	}

	r_processed.insert(p_node);
}

bool ShaderGraph::_check_reroute_subgraph(int p_target_port_type, int p_reroute_node, List<int> *r_visited_reroute_nodes) const {
	// BFS to check whether connecting to the given subgraph (rooted at p_reroute_node) is valid.
	List<int> queue;
	queue.push_back(p_reroute_node);
	if (r_visited_reroute_nodes != nullptr) {
		r_visited_reroute_nodes->push_back(p_reroute_node);
	}
	while (!queue.is_empty()) {
		int current_node_id = queue.front()->get();
		ShaderGraph::Node current_node = nodes[current_node_id];
		queue.pop_front();
		for (const int &next_node_id : current_node.next_connected_nodes) {
			Ref<VisualShaderNodeReroute> next_vsnode = nodes[next_node_id].node;
			if (next_vsnode.is_valid()) {
				queue.push_back(next_node_id);
				if (r_visited_reroute_nodes != nullptr) {
					r_visited_reroute_nodes->push_back(next_node_id);
				}
				continue;
			}
			// Check whether all ports connected with the reroute node are compatible.
			for (const ShaderGraph::Connection &c : connections) {
				VisualShaderNode::PortType to_port_type = nodes[next_node_id].node->get_input_port_type(c.to_port);
				if (c.from_node == current_node_id &&
						c.to_node == next_node_id &&
						!is_port_types_compatible(p_target_port_type, to_port_type)) {
					return false;
				}
			}
		}
	}
	return true;
}

void ShaderGraph::add_node(const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id) {
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_id < reserved_node_ids);
	ERR_FAIL_COND(nodes.has(p_id));

	ShaderGraph::Node n;
	n.node = p_node;
	n.position = p_position;

	Ref<VisualShaderNodeCustom> custom = n.node;
	if (custom.is_valid()) {
		custom->update_ports();
	}

	nodes[p_id] = n;

	n.node->connect_changed(callable_mp(this, &ShaderGraph::_node_changed));
	emit_signal("graph_changed");
}

void ShaderGraph::set_node_position(int p_id, const Vector2 &p_position) {
	ERR_FAIL_COND(!nodes.has(p_id));
	nodes[p_id].position = p_position;
}

Vector2 ShaderGraph::get_node_position(int p_id) const {
	ERR_FAIL_COND_V(!nodes.has(p_id), Vector2());
	return nodes[p_id].position;
}

Ref<VisualShaderNode> ShaderGraph::get_node(int p_id) const {
	if (!nodes.has(p_id)) {
		return Ref<VisualShaderNode>();
	}
	ERR_FAIL_COND_V(!nodes.has(p_id), Ref<VisualShaderNode>());
	return nodes[p_id].node;
}

Vector<int> ShaderGraph::get_node_ids() const {
	Vector<int> ret;
	for (const KeyValue<int, ShaderGraph::Node> &E : nodes) {
		ret.push_back(E.key);
	}

	return ret;
}

int ShaderGraph::get_valid_node_id() const {
	return nodes.size() ? MAX(reserved_node_ids, nodes.back()->key() + 1) : reserved_node_ids;
}

int ShaderGraph::find_node_id(const Ref<VisualShaderNode> &p_node) const {
	for (const KeyValue<int, ShaderGraph::Node> &E : nodes) {
		if (E.value.node == p_node) {
			return E.key;
		}
	}

	return NODE_ID_INVALID;
}

void ShaderGraph::remove_node(int p_id) {
	ERR_FAIL_COND(!nodes.has(p_id));

	nodes.erase(p_id);

	for (List<ShaderGraph::Connection>::Element *E = connections.front(); E;) {
		List<ShaderGraph::Connection>::Element *N = E->next();
		const ShaderGraph::Connection &connection = E->get();
		if (connection.from_node == p_id || connection.to_node == p_id) {
			if (connection.from_node == p_id && nodes.has(connection.to_node)) {
				nodes[connection.to_node].prev_connected_nodes.erase(p_id);
				nodes[connection.to_node].node->set_input_port_connected(connection.to_port, false);
			} else if (connection.to_node == p_id && nodes.has(connection.from_node)) {
				nodes[connection.from_node].next_connected_nodes.erase(p_id);
				nodes[connection.from_node].node->set_output_port_connected(connection.from_port, false);
			}
			connections.erase(E);
		}
		E = N;
	}

	emit_signal("graph_changed");
}

void ShaderGraph::replace_node(int p_id, const StringName &p_new_class) {
	ERR_FAIL_COND(!nodes.has(p_id));

	if (nodes[p_id].node->get_class_name() == p_new_class) {
		return;
	}
	VisualShaderNode *vsn = Object::cast_to<VisualShaderNode>(ClassDB::instantiate(p_new_class));
	VisualShaderNode *prev_vsn = nodes[p_id].node.ptr();

	// Update connection data.
	for (int i = 0; i < vsn->get_output_port_count(); i++) {
		if (i < prev_vsn->get_output_port_count()) {
			if (prev_vsn->is_output_port_connected(i)) {
				vsn->set_output_port_connected(i, true);
			}

			if (prev_vsn->is_output_port_expandable(i) && prev_vsn->_is_output_port_expanded(i) && vsn->is_output_port_expandable(i)) {
				vsn->_set_output_port_expanded(i, true);

				int component_count = 0;
				switch (prev_vsn->get_output_port_type(i)) {
					case VisualShaderNode::PORT_TYPE_VECTOR_2D:
						component_count = 2;
						break;
					case VisualShaderNode::PORT_TYPE_VECTOR_3D:
						component_count = 3;
						break;
					case VisualShaderNode::PORT_TYPE_VECTOR_4D:
						component_count = 4;
						break;
					default:
						break;
				}

				for (int j = 0; j < component_count; j++) {
					int sub_port = i + 1 + j;

					if (prev_vsn->is_output_port_connected(sub_port)) {
						vsn->set_output_port_connected(sub_port, true);
					}
				}

				i += component_count;
			}
		} else {
			break;
		}
	}

	nodes[p_id].node = Ref<VisualShaderNode>(vsn);

	emit_signal("graph_changed");
}

bool ShaderGraph::are_nodes_connected(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	for (const ShaderGraph::Connection &E : connections) {
		if (E.from_node == p_from_node && E.from_port == p_from_port && E.to_node == p_to_node && E.to_port == p_to_port) {
			return true;
		}
	}

	return false;
}

bool ShaderGraph::is_node_reachable(int p_from, int p_target) const {
	bool result = false;

	const ShaderGraph::Node &node = nodes[p_from];

	for (const int &E : node.prev_connected_nodes) {
		if (E == p_target) {
			return true;
		}

		result = is_node_reachable(E, p_target);
		if (result) {
			break;
		}
	}
	return result;
}

bool ShaderGraph::can_connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	if (!nodes.has(p_from_node)) {
		return false;
	}

	if (p_from_node == p_to_node) {
		return false;
	}

	if (p_from_port < 0 || p_from_port >= nodes[p_from_node].node->get_expanded_output_port_count()) {
		return false;
	}

	if (!nodes.has(p_to_node)) {
		return false;
	}

	if (p_to_port < 0 || p_to_port >= nodes[p_to_node].node->get_input_port_count()) {
		return false;
	}

	VisualShaderNode::PortType from_port_type = nodes[p_from_node].node->get_output_port_type(p_from_port);
	VisualShaderNode::PortType to_port_type = nodes[p_to_node].node->get_input_port_type(p_to_port);

	Ref<VisualShaderNodeReroute> to_node_reroute = nodes[p_to_node].node;
	if (to_node_reroute.is_valid()) {
		if (!_check_reroute_subgraph(from_port_type, p_to_node)) {
			return false;
		}
	} else if (!is_port_types_compatible(from_port_type, to_port_type)) {
		return false;
	}

	for (const ShaderGraph::Connection &E : connections) {
		if (E.from_node == p_from_node && E.from_port == p_from_port && E.to_node == p_to_node && E.to_port == p_to_port) {
			return false;
		}
	}

	if (is_node_reachable(p_from_node, p_to_node)) {
		return false;
	}
	return true;
}

Error ShaderGraph::connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	ERR_FAIL_COND_V(!nodes.has(p_from_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_from_port, nodes[p_from_node].node->get_expanded_output_port_count(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!nodes.has(p_to_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_to_port, nodes[p_to_node].node->get_input_port_count(), ERR_INVALID_PARAMETER);

	Ref<VisualShaderNodeReroute> from_node_reroute = nodes[p_from_node].node;
	Ref<VisualShaderNodeReroute> to_node_reroute = nodes[p_to_node].node;

	// Allow connection with incompatible port types only if the reroute node isn't connected to anything.
	VisualShaderNode::PortType from_port_type = nodes[p_from_node].node->get_output_port_type(p_from_port);
	VisualShaderNode::PortType to_port_type = nodes[p_to_node].node->get_input_port_type(p_to_port);
	bool port_types_are_compatible = is_port_types_compatible(from_port_type, to_port_type);

	if (to_node_reroute.is_valid()) {
		List<int> visited_reroute_nodes;
		port_types_are_compatible = _check_reroute_subgraph(from_port_type, p_to_node, &visited_reroute_nodes);
		if (port_types_are_compatible) {
			// Set the port type of all reroute nodes.
			for (const int &E : visited_reroute_nodes) {
				Ref<VisualShaderNodeReroute> reroute_node = nodes[E].node;
				reroute_node->_set_port_type(from_port_type);
			}
		}
	} else if (from_node_reroute.is_valid() && !from_node_reroute->is_input_port_connected(0)) {
		from_node_reroute->_set_port_type(to_port_type);
		port_types_are_compatible = true;
	}

	ERR_FAIL_COND_V_MSG(!port_types_are_compatible, ERR_INVALID_PARAMETER, "Incompatible port types.");

	for (const ShaderGraph::Connection &E : connections) {
		if (E.from_node == p_from_node && E.from_port == p_from_port && E.to_node == p_to_node && E.to_port == p_to_port) {
			ERR_FAIL_V(ERR_ALREADY_EXISTS);
		}
	}

	ShaderGraph::Connection c;
	c.from_node = p_from_node;
	c.from_port = p_from_port;
	c.to_node = p_to_node;
	c.to_port = p_to_port;
	connections.push_back(c);
	nodes[p_from_node].next_connected_nodes.push_back(p_to_node);
	nodes[p_to_node].prev_connected_nodes.push_back(p_from_node);
	nodes[p_from_node].node->set_output_port_connected(p_from_port, true);
	nodes[p_to_node].node->set_input_port_connected(p_to_port, true);

	emit_signal("graph_changed");

	return OK;
}

void ShaderGraph::disconnect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	for (List<ShaderGraph::Connection>::Element *E = connections.front(); E; E = E->next()) {
		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			connections.erase(E);
			if (nodes.has(p_from_node) && nodes[p_from_node].node.is_valid()) {
				nodes[p_from_node].next_connected_nodes.erase(p_to_node);
				nodes[p_from_node].node->set_output_port_connected(p_from_port, false);
			}
			if (nodes.has(p_to_node) && nodes[p_to_node].node.is_valid()) {
				nodes[p_to_node].prev_connected_nodes.erase(p_from_node);
				nodes[p_to_node].node->set_input_port_connected(p_to_port, false);
			}
			emit_signal("graph_changed");
			return;
		}
	}
}

void ShaderGraph::connect_nodes_forced(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	ERR_FAIL_COND(!nodes.has(p_from_node));
	ERR_FAIL_INDEX(p_from_port, nodes[p_from_node].node->get_expanded_output_port_count());
	ERR_FAIL_COND(!nodes.has(p_to_node));
	ERR_FAIL_INDEX(p_to_port, nodes[p_to_node].node->get_input_port_count());

	for (const ShaderGraph::Connection &E : connections) {
		if (E.from_node == p_from_node && E.from_port == p_from_port && E.to_node == p_to_node && E.to_port == p_to_port) {
			return;
		}
	}

	ShaderGraph::Connection c;
	c.from_node = p_from_node;
	c.from_port = p_from_port;
	c.to_node = p_to_node;
	c.to_port = p_to_port;
	connections.push_back(c);
	nodes[p_from_node].next_connected_nodes.push_back(p_to_node);
	nodes[p_to_node].prev_connected_nodes.push_back(p_from_node);
	nodes[p_from_node].node->set_output_port_connected(p_from_port, true);
	nodes[p_to_node].node->set_input_port_connected(p_to_port, true);

	emit_signal("graph_changed");
}

bool ShaderGraph::is_port_types_compatible(int p_a, int p_b) const {
	return MAX(0, p_a - (int)VisualShaderNode::PORT_TYPE_BOOLEAN) == (MAX(0, p_b - (int)VisualShaderNode::PORT_TYPE_BOOLEAN));
}

void ShaderGraph::attach_node_to_frame(int p_node, int p_frame) {
	ERR_FAIL_COND(!nodes.has(p_node));

	nodes[p_node].node->set_frame(p_frame);

	Ref<VisualShaderNodeFrame> vsnode_frame = nodes[p_frame].node;
	if (vsnode_frame.is_valid()) {
		vsnode_frame->add_attached_node(p_node);
	}
}

void ShaderGraph::detach_node_from_frame(int p_node) {
	ERR_FAIL_COND(!nodes.has(p_node));

	int parent_frame_id = nodes[p_node].node->get_frame();
	Ref<VisualShaderNodeFrame> vsnode_frame = nodes[parent_frame_id].node;
	if (vsnode_frame.is_valid()) {
		vsnode_frame->remove_attached_node(p_node);
	}

	nodes[p_node].node->set_frame(-1);
}

String ShaderGraph::get_reroute_parameter_name(int p_reroute_node) const {
	ERR_FAIL_COND_V(!nodes.has(p_reroute_node), "");

	const ShaderGraph::Node *node = &nodes[p_reroute_node];
	while (node->prev_connected_nodes.size() > 0) {
		int connected_node_id = node->prev_connected_nodes[0];
		node = &nodes[connected_node_id];
		Ref<VisualShaderNodeParameter> parameter_node = node->node;
		if (parameter_node.is_valid() && parameter_node->get_output_port_type(0) == VisualShaderNode::PORT_TYPE_SAMPLER) {
			return parameter_node->get_parameter_name();
		}
		Ref<VisualShaderNodeInput> input_node = node->node;
		if (input_node.is_valid() && input_node->get_output_port_type(0) == VisualShaderNode::PORT_TYPE_SAMPLER) {
			return input_node->get_input_real_name();
		}
	}
	return "";
}

void ShaderGraph::get_node_connections(List<ShaderGraph::Connection> *r_connections) const {
	for (const ShaderGraph::Connection &conn : connections) {
		r_connections->push_back(conn);
	}
}

String ShaderGraph::generate_preview_shader(int p_node, int p_port, Vector<DefaultTextureParam> &r_default_tex_params, const String &p_additional_global_code) const {
	Ref<VisualShaderNode> node = get_node(p_node);
	ERR_FAIL_COND_V(node.is_null(), String());
	ERR_FAIL_COND_V(p_port < 0 || p_port >= node->get_expanded_output_port_count(), String());
	ERR_FAIL_COND_V(node->get_output_port_type(p_port) == VisualShaderNode::PORT_TYPE_TRANSFORM, String());

	StringBuilder global_code;
	StringBuilder global_code_per_node;
	HashMap<ShaderGraph::Type, StringBuilder> global_code_per_func;
	StringBuilder shader_code;
	HashSet<StringName> classes;

	global_code += String() + "shader_type canvas_item;\n";
	global_code += "\n";

	String global_expressions;
	int index = 0;
	for (const KeyValue<int, ShaderGraph::Node> &E : nodes) {
		Ref<VisualShaderNodeGlobalExpression> global_expression = E.value.node;
		if (global_expression.is_valid()) {
			String expr = "";
			expr += "// " + global_expression->get_caption() + ":" + itos(index++) + "\n";
			// Use TYPE_MAX/MODE_MAX for preview since we're not in a specific shader context.
			expr += global_expression->generate_global(Shader::MODE_MAX, VisualShader::TYPE_MAX, -1);
			expr = expr.replace("\n", "\n	");
			expr += "\n";
			global_expressions += expr;
		}
	}

	global_code += global_expressions;
	global_code += p_additional_global_code;

	// Make it faster to go around through shader.
	HashMap<ShaderGraph::ConnectionKey, const List<ShaderGraph::Connection>::Element *> input_connections;
	HashMap<ShaderGraph::ConnectionKey, const List<ShaderGraph::Connection>::Element *> output_connections;

	for (const List<ShaderGraph::Connection>::Element *E = connections.front(); E; E = E->next()) {
		ShaderGraph::ConnectionKey from_key;
		from_key.node = E->get().from_node;
		from_key.port = E->get().from_port;

		output_connections.insert(from_key, E);

		ShaderGraph::ConnectionKey to_key;
		to_key.node = E->get().to_node;
		to_key.port = E->get().to_port;

		input_connections.insert(to_key, E);
	}

	shader_code += "\nvoid fragment() {\n";

	HashSet<int> processed;
	_write_node(&global_code, &global_code_per_node, &global_code_per_func, shader_code, r_default_tex_params, input_connections, output_connections, p_node, processed, true, classes);

	switch (node->get_output_port_type(p_port)) {
		case VisualShaderNode::PORT_TYPE_SCALAR: {
			shader_code += "	COLOR.rgb = vec3(n_out" + itos(p_node) + "p" + itos(p_port) + ");\n";
		} break;
		case VisualShaderNode::PORT_TYPE_SCALAR_INT: {
			shader_code += "	COLOR.rgb = vec3(float(n_out" + itos(p_node) + "p" + itos(p_port) + "));\n";
		} break;
		case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
			shader_code += "	COLOR.rgb = vec3(float(n_out" + itos(p_node) + "p" + itos(p_port) + "));\n";
		} break;
		case VisualShaderNode::PORT_TYPE_BOOLEAN: {
			shader_code += "	COLOR.rgb = vec3(n_out" + itos(p_node) + "p" + itos(p_port) + " ? 1.0 : 0.0);\n";
		} break;
		case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
			shader_code += "	COLOR.rgb = vec3(n_out" + itos(p_node) + "p" + itos(p_port) + ", 0.0);\n";
		} break;
		case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
			shader_code += "	COLOR.rgb = n_out" + itos(p_node) + "p" + itos(p_port) + ";\n";
		} break;
		case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
			shader_code += "	COLOR = n_out" + itos(p_node) + "p" + itos(p_port) + ";\n";
		} break;
		default: {
			shader_code += "	COLOR.rgb = vec3(0.0);\n";
		} break;
	}

	shader_code += "}\n";
	global_code += "\n\n";
	String final_code = global_code;
	final_code += global_code_per_node;
	final_code += shader_code;
	return final_code;
}

String ShaderGraph::validate_port_name(const String &p_port_name, VisualShaderNode *p_node, int p_port_id, bool p_output) const {
	String port_name = p_port_name;

	if (port_name.is_empty()) {
		return String();
	}

	while (port_name.length() && !is_ascii_alphabet_char(port_name[0])) {
		port_name = port_name.substr(1);
	}

	if (!port_name.is_empty()) {
		String valid_name;

		for (int i = 0; i < port_name.length(); i++) {
			if (is_ascii_identifier_char(port_name[i])) {
				valid_name += String::chr(port_name[i]);
			} else if (port_name[i] == ' ') {
				valid_name += "_";
			}
		}

		port_name = valid_name;
	} else {
		return String();
	}

	List<String> input_names;
	List<String> output_names;

	for (int i = 0; i < p_node->get_input_port_count(); i++) {
		if (!p_output && i == p_port_id) {
			continue;
		}
		if (port_name == p_node->get_input_port_name(i)) {
			return String();
		}
	}
	for (int i = 0; i < p_node->get_output_port_count(); i++) {
		if (p_output && i == p_port_id) {
			continue;
		}
		if (port_name == p_node->get_output_port_name(i)) {
			return String();
		}
	}

	return port_name;
}

ShaderGraph::ShaderGraph(int p_reserved_node_ids) :
		reserved_node_ids(p_reserved_node_ids) {
}
