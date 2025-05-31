/**************************************************************************/
/*  resource_saver_materialx.cpp                                          */
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

#include "resource_saver_materialx.h"
#include "MaterialXCore/Interface.h"
#include "MaterialXFormat/XmlIo.h"
#include "core/error/error_macros.h"
#include "core/variant/variant.h"
#include "materialx_shader.h"
#include "scene/resources/visual_shader.h"
#include "scene/resources/visual_shader_nodes.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>

Error ResourceFormatSaverMtlx::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<MaterialXShader> shader = p_resource;
	if (!shader.is_null()) {
		return shader->save_file(p_path);
	}
	return OK;
}

bool ResourceFormatSaverMtlx::recognize(const Ref<Resource> &p_resource) const {
	Ref<MaterialXShader> shader = p_resource;
	return !shader.is_null();
}

void ResourceFormatSaverMtlx::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	Ref<MaterialXShader> shader = p_resource;
	if (!shader.is_null()) {
		p_extensions->push_back("mtlx");
	}
}

Error MaterialXShader::save_file(const String &p_path) {
	mx::DocumentPtr document = mx::createDocument();

	mx::NodeGraphPtr node_graph = document->addNodeGraph("super_cool_graph");
	for (int node_id : get_node_list(VisualShader::TYPE_FRAGMENT)) {
		Ref<VisualShaderNode> node = get_node(VisualShader::TYPE_FRAGMENT, node_id);

		Ref<VisualShaderNodeFrame> frame_node = node;
		if (!frame_node.is_null()) {
			// ¯\_(ツ)_/¯
			continue;
		}

		Ref<VisualShaderNodeReroute> reroute_node = node;
		if (!reroute_node.is_null()) {
			std::string name = vformat("dot_%d", node_id).utf8().get_data();
			mx::NodePtr mx_node = node_graph->addNode("dot", name);
			mx::InputPtr mx_in = mx_node->addInput("in");

			VisualShaderNode::PortType port_type = reroute_node->get_input_port_type(0);
			switch (port_type) {
				case VisualShaderNode::PORT_TYPE_SCALAR: {
					mx_node->setType("float");
					mx_in->setType("float");
					mx_in->setNodeName(""); // aaaaaaaaaaaaaaaaaaaa
				} break;

				case VisualShaderNode::PORT_TYPE_SCALAR_INT:
				case VisualShaderNode::PORT_TYPE_SCALAR_UINT: {
					mx_node->setType("integer");
					mx_in->setType("integer");
					mx_in->setNodeName(""); // aaaaaaaaaaaaaaaaaaaa
				} break;

				case VisualShaderNode::PORT_TYPE_VECTOR_2D: {
					mx_node->setType("vector2");
					mx_in->setType("vector2");
					mx_in->setNodeName(""); // aaaaaaaaaaaaaaaaaaaa
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_3D: {
					mx_node->setType("vector3");
					mx_in->setType("vector3");
					mx_in->setNodeName(""); // aaaaaaaaaaaaaaaaaaaa
				} break;
				case VisualShaderNode::PORT_TYPE_VECTOR_4D: {
					mx_node->setType("vector4");
					mx_in->setType("vector4");
					mx_in->setNodeName(""); // aaaaaaaaaaaaaaaaaaaa
				} break;
				case VisualShaderNode::PORT_TYPE_BOOLEAN: {
					mx_node->setType("boolean");
					mx_in->setType("boolean");
					mx_in->setNodeName(""); // aaaaaaaaaaaaaaaaaaaa
				} break;

				case VisualShaderNode::PORT_TYPE_TRANSFORM:
				case VisualShaderNode::PORT_TYPE_SAMPLER:
				case VisualShaderNode::PORT_TYPE_MAX: {
					// ¯\_(ツ)_/¯
				} break;

					break;
			}

			continue;
		}

		Ref<VisualShaderNodeFloatConstant> float_constant_node = node;
		if (!float_constant_node.is_null()) {
			std::string name = vformat("float_constant_%d", node_id).utf8().get_data();
			mx::NodePtr mx_node = node_graph->addNode("constant", name, "float");

			mx::InputPtr mx_value = mx_node->addInput("value", "float");
			mx_value->setValue(float_constant_node->get_constant());

			continue;
		}

		ERR_PRINT(vformat("Couldn't save node [%d] of type [%s]", node_id, node->get_class_name()));
	}

	String content = mx::writeToXmlString(document).c_str();
	print_line(content);

	return OK;
}
