/**************************************************************************/
/*  materialx_shader.h                                                    */
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

#pragma once

#include "core/object/object.h"
#include "core/string/string_name.h"
#include "scene/resources/visual_shader.h"
#include "scene/resources/visual_shader_nodes.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Value.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/XmlIo.h>

namespace mx = MaterialX;

class MaterialXShader : public VisualShader {
	GDCLASS(MaterialXShader, VisualShader)

	HashMap<StringName, int> node_id_map;
	HashMap<StringName, int> node_output_port_map;
	HashMap<StringName, StringName> output_to_node_map;

	typedef struct {
		StringName node_out;
		StringName node_in;
		int node_in_port;
	} NodeConnection;
	Vector<NodeConnection> node_connections;

public:
	Error load_file(const String &p_path);
	Error save_file(const String &p_path = "");

private:
	void _connection_or_default(Ref<VisualShaderNode> p_shader_node, int p_port, const mx::NodePtr &p_mtlx_node, std::string p_input);

	Ref<VisualShaderNode> _read_node(const mx::NodePtr &node, int p_id);
	Ref<VisualShaderNodeConstant> _read_constant_node(const mx::NodePtr &p_node) const;
	Ref<VisualShaderNodeFloatFunc> _read_unary_operator_node(const mx::NodePtr &p_node, VisualShaderNodeFloatFunc::Function p_unary_op);
	Ref<VisualShaderNodeFloatOp> _read_binary_operator_node(const mx::NodePtr &p_node, VisualShaderNodeFloatOp::Operator p_binary_op);

	static int _get_mtlx_type_as_port_type(const std::string &p_type);
	static Variant _get_mtlx_value_as_port_value(const mx::ValuePtr &p_value);
};
