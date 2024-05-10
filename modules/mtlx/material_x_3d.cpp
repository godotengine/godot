/**************************************************************************/
/*  material_x_3d.cpp                                                     */
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

#include "material_x_3d.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/variant/variant.h"
#include "modules/tinyexr/image_loader_tinyexr.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/visual_shader.h"
#include "thirdparty/mtlx/source/MaterialXCore/Node.h"
#include "thirdparty/mtlx/source/MaterialXCore/Traversal.h"

void MTLXLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_load", "path", "original_path", "use_sub_threads", "cache_mode"), &MTLXLoader::_load);
}

Variant MTLXLoader::get_value_as_variant(const mx::ValuePtr &p_value) {
	Variant variant_value;
	if (p_value) {
		std::string typeString = p_value->getTypeString();
		if (typeString == "float") {
			variant_value = p_value->asA<float>();
		} else if (typeString == "integer") {
			variant_value = p_value->asA<int>();
		} else if (typeString == "boolean") {
			variant_value = p_value->asA<bool>();
		} else if (typeString == "color3") {
			mx::Color3 color = p_value->asA<mx::Color3>();
			variant_value = Color(color[0], color[1], color[2]);
		} else if (typeString == "color4") {
			mx::Color4 color = p_value->asA<mx::Color4>();
			variant_value = Color(color[0], color[1], color[2], color[3]);
		} else if (typeString == "vector2") {
			mx::Vector2 vector_2 = p_value->asA<mx::Vector2>();
			variant_value = Vector2(vector_2[0], vector_2[1]);
		} else if (typeString == "vector3") {
			mx::Vector3 vector_3 = p_value->asA<mx::Vector3>();
			variant_value = Vector3(vector_3[0], vector_3[1], vector_3[2]);
		} else if (typeString == "vector4") {
			mx::Vector4 vector_4 = p_value->asA<mx::Vector4>();
			variant_value = Color(vector_4[0], vector_4[1], vector_4[2], vector_4[3]);
		} else if (typeString == "matrix33") {
			// Matrix33 m = value->asA<Matrix33>();
			// TODO: fire 2022-03-11 add basis
		} else if (typeString == "matrix44") {
			// Matrix44 m = value->asA<Matrix44>();
			// TODO: fire 2022-03-11 add transform
		}
	}
	return variant_value;
}

Variant MTLXLoader::_load(const String &p_save_path, const String &p_original_path, bool p_use_sub_threads, int64_t p_cache_mode) const {
	// Globalize paths
	String save_path = ProjectSettings::get_singleton()->globalize_path(p_save_path);
	String original_path = ProjectSettings::get_singleton()->globalize_path(p_original_path);

	// Create MaterialX document
	mx::DocumentPtr doc = mx::createDocument();

	try {
		mx::FilePath materialFilename = ProjectSettings::get_singleton()->globalize_path(p_original_path).utf8().get_data();
		std::vector<MaterialPtr> materials;
		mx::DocumentPtr dependLib = mx::createDocument();
		mx::StringSet skipLibraryFiles;
		mx::DocumentPtr stdLib;
		mx::StringSet xincludeFiles;

		mx::StringVec distanceUnitOptions;
		mx::LinearUnitConverterPtr distanceUnitConverter;

		mx::UnitConverterRegistryPtr unitRegistry =
				mx::UnitConverterRegistry::create();
		mx::FileSearchPath searchPath(ProjectSettings::get_singleton()->globalize_path(p_original_path.get_base_dir()).utf8().get_data());
		try {
			stdLib = mx::createDocument();
			mx::FilePathVec libraryFolders;
			libraryFolders.push_back(ProjectSettings::get_singleton()->globalize_path(p_original_path.get_base_dir()).utf8().get_data());
			libraryFolders.push_back(ProjectSettings::get_singleton()->globalize_path("res://libraries").utf8().get_data());
			libraryFolders.push_back(ProjectSettings::get_singleton()->globalize_path("user://libraries").utf8().get_data());
			mx::StringSet xincludeFilesLib = mx::loadLibraries(libraryFolders, searchPath, stdLib);
			if (xincludeFilesLib.empty()) {
				std::cerr << "Could not find standard data libraries on the given "
							 "search path: "
						  << searchPath.asString() << std::endl;
				return Ref<Resource>();
			}

			mx::UnitTypeDefPtr distanceTypeDef = stdLib->getUnitTypeDef("distance");
			distanceUnitConverter = mx::LinearUnitConverter::create(distanceTypeDef);
			unitRegistry->addUnitConverter(distanceTypeDef, distanceUnitConverter);
			mx::UnitTypeDefPtr angleTypeDef = stdLib->getUnitTypeDef("angle");
			mx::LinearUnitConverterPtr angleConverter =
					mx::LinearUnitConverter::create(angleTypeDef);
			unitRegistry->addUnitConverter(angleTypeDef, angleConverter);

			auto unitScales = distanceUnitConverter->getUnitScale();
			distanceUnitOptions.resize(unitScales.size());
			for (auto unitScale : unitScales) {
				int location = distanceUnitConverter->getUnitAsInteger(unitScale.first);
				distanceUnitOptions[location] = unitScale.first;
			}
		} catch (std::exception &e) {
			std::cerr << "Failed to load standard data libraries: " << e.what()
					  << std::endl;
			return Ref<Resource>();
		}
		doc->importLibrary(stdLib);
		MaterialX::FilePath parentPath = materialFilename.getParentPath();
		searchPath.append(materialFilename.getParentPath());
		// Set up read options.
		mx::XmlReadOptions readOptions;
		readOptions.readXIncludeFunction = [](mx::DocumentPtr docLambda,
												   const mx::FilePath &filenameLambda,
												   const mx::FileSearchPath &pathLambda,
												   const mx::XmlReadOptions *newReadoptions) {
			mx::FilePath resolvedFilename = pathLambda.find(filenameLambda);
			if (resolvedFilename.exists()) {
				readFromXmlFile(docLambda, resolvedFilename, pathLambda, newReadoptions);
			} else {
				std::cerr << "Include file not found: " << filenameLambda.asString()
						  << std::endl;
			}
		};
		mx::readFromXmlFile(doc, materialFilename, searchPath, &readOptions);

		std::string message;
		bool docValid = doc->validate(&message);
		ERR_FAIL_COND_V_MSG(!docValid, Ref<Resource>(), String("The MaterialX document is invalid: [") + String(doc->getSourceUri().c_str()) + "] " + String(message.c_str()));

		Ref<ShaderMaterial> mat;
		mat.instantiate();
		Ref<VisualShader> shader;
		shader.instantiate();

		process_node_graph(doc, shader);

		mat->set_shader(shader);
		return mat;

	} catch (std::exception &e) {
		ERR_PRINT(String("Can't load Materialx materials. Error: ") + String(e.what()));
		return Ref<Resource>();
	}
}

void MTLXLoader::process_node_graph(mx::DocumentPtr p_doc, Ref<VisualShader> p_shader) const {
	std::vector<mx::NodeGraphPtr> node_graphs = p_doc->getNodeGraphs();
	int node_i = 2;
	std::set<mx::NodePtr> processed_nodes;

	for (mx::NodeGraphPtr graph : node_graphs) {
		String graph_name = graph->getName().c_str();
		if (!graph_name.begins_with("NG_")) {
			continue;
		}

		print_line(String("MaterialX nodegraph ") + graph_name);
		std::vector<mx::ElementPtr> sorted_nodes = graph->topologicalSort();

		for (const mx::ElementPtr &element : sorted_nodes) {
			const mx::NodePtr &node = element->asA<mx::Node>();
			if (!node) {
				continue;
			}

			process_node(node, p_shader, node_i++, processed_nodes);
		}
	}
}

void MTLXLoader::process_node(const mx::NodePtr &p_node, Ref<VisualShader> p_shader, int p_node_i, std::set<mx::NodePtr> &p_processed_nodes) const {
	if (p_processed_nodes.find(p_node) != p_processed_nodes.end()) {
		return;
	}
	p_processed_nodes.insert(p_node);

	Ref<VisualShaderNodeExpression> expression_node;
	expression_node.instantiate();
	String expression_text = String(p_node->getName().c_str());
	std::map<mx::OutputPtr, mx::InputPtr> port_mapping;

	print_line(String("MaterialX node " + expression_text));
	expression_node->set_expression(String("// ") + expression_text);
	p_shader->add_node(VisualShader::TYPE_FRAGMENT, expression_node, Vector2(200, -200), p_node_i);
	int input_port_i = 0;
	for (mx::InputPtr input : p_node->getInputs()) {
		const std::string &input_name = input->getName();
		print_line(String("Adding input port: ") + itos(input_port_i) + " Name: " + String(input_name.c_str()));
		add_input_port(input, expression_node, input_port_i++);
	}
	print_line(String("Total input ports: ") + itos(p_node->getInputs().size()));
	std::vector<mx::PortElementPtr> downstream_ports = p_node->getDownstreamPorts();
	for (mx::PortElementPtr port : downstream_ports) {
		const std::string &output_name = port->getName();
		print_line(String("MaterialX output " + String(output_name.c_str())));
		int output_port_id = expression_node->get_free_output_port_id();
		int variant_type = convert_type(port->getType());
		expression_node->add_output_port(output_port_id, variant_type, String(output_name.c_str()));
	}
	print_line(String("Total output ports: ") + String::num(downstream_ports.size()));
}

void MTLXLoader::add_input_port(mx::InputPtr p_input, Ref<VisualShaderNodeExpression> p_expression_node, int p_input_port_i) const {
	const std::string &input_name = p_input->getName();
	print_line(vformat("MaterialX input %s", String(input_name.c_str())));

	mx::ValuePtr value = p_input->getValue();
	Variant variant_value = get_value_as_variant(value);

	print_line(vformat("MaterialX input value: %s", String(variant_value)));
	p_expression_node->add_input_port(p_input_port_i, variant_value, input_name.c_str());
}

int MTLXLoader::convert_type(const std::string &p_type_string) const {
	if (p_type_string == "float") {
		return Variant::FLOAT;
	} else if (p_type_string == "integer") {
		return Variant::INT;
	} else if (p_type_string == "boolean") {
		return Variant::BOOL;
	} else if (p_type_string == "color3" || p_type_string == "color4") {
		return Variant::COLOR;
	} else if (p_type_string == "vector2") {
		return Variant::VECTOR2;
	} else if (p_type_string == "vector3") {
		return Variant::VECTOR3;
	} else if (p_type_string == "vector4") {
		return Variant::VECTOR4;
	} else if (p_type_string == "matrix33" || p_type_string == "matrix44") {
		// Placeholder.
	}
	print_line(String("Unknown type: ") + String(p_type_string.c_str()));
	return -1;
}
