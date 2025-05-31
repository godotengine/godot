/**************************************************************************/
/*  resource_loader_materialx.cpp                                         */
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

#include "resource_loader_materialx.h"

#include "materialx_shader.h"
#include "visual_shader_node_standard_surface.h"

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/math/vector2.h"
#include "core/math/vector4.h"
#include "core/object/class_db.h"
#include "core/string/print_string.h"
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"
#include "scene/resources/visual_shader.h"
#include "scene/resources/visual_shader_nodes.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Element.h>
#include <MaterialXCore/Interface.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Unit.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>

Ref<Resource> ResourceFormatLoaderMtlx::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	*r_error = OK;

	// Copy thirdparty/materialx/libraries into res://addons/materialx/libraries
	// (you should probably also add a .gdignore to it)
	// Otherwise this will fail much worse than usual
	String materialx_path = ProjectSettings::get_singleton()->get_resource_path() + "/addons/materialx/";

	mx::DocumentPtr stdlib = mx::createDocument();

	mx::FilePathVec libraries;
	libraries.push_back("bxdf");
	libraries.push_back("stdlib");
	libraries.push_back("pbrlib");
	libraries.push_back("targets");

	mx::FileSearchPath mtlx_path((materialx_path + "libraries/").utf8().get_data());

	// Load node definitions, implementations and the surface shader node generator
	mx::loadLibraries(libraries, mtlx_path, stdlib);

	mx::DocumentPtr doc = mx::createDocument();
	doc->importLibrary(stdlib);

	String resource_path = ProjectSettings::get_singleton()->globalize_path(p_path);
	mx::FilePath mx_absolute_path(resource_path.utf8().get_data());

	mx::readFromXmlFile(doc, mx_absolute_path);

	std::string msg;
	bool valid = doc->validate(&msg);
	if (!valid) {
		ERR_PRINT(msg.c_str());
	}

	mx::ShaderGeneratorPtr generator = mx::GlslShaderGenerator::create();
	mx::GenContext context(generator);
	context.registerSourceCodeSearchPath(materialx_path.utf8().get_data());

	for (mx::NodePtr material : doc->getMaterialNodes()) {
		try {
			//TODO: this will generate invalid glsl for vulkan
			// what can we do about it?
			mx::ShaderPtr mx_shader = generator->generate("code", material, context);

			String fragment_src = mx_shader->getSourceCode(mx::Stage::PIXEL).c_str();
			Ref<FileAccess> fragment_file = FileAccess::open(resource_path + ".frag.glsl", FileAccess::WRITE);
			fragment_file->store_string("#[fragment]\n");
			fragment_file->store_string(fragment_src);

			String vertex_src = mx_shader->getSourceCode(mx::Stage::VERTEX).c_str();
			Ref<FileAccess> vertex_file = FileAccess::open(resource_path + ".vert.glsl", FileAccess::WRITE);
			vertex_file->store_string("#[vertex]\n");
			vertex_file->store_string(vertex_src);

		} catch (mx::ExceptionShaderGenError &e) {
			ERR_PRINT(e.what());
		}
	}

	//String mx_err;
	//Vector<uint8_t> bytecode = RD::get_singleton()->shader_compile_spirv_from_source(RD::SHADER_STAGE_VERTEX, source, RD::SHADER_LANGUAGE_GLSL, &mx_err);
	//WARN_PRINT(mx_err);

	//RID shader_rid = RD::get_singleton()->shader_create_from_bytecode(bytecode);

	Ref<ShaderMaterial> mat = memnew(ShaderMaterial);
	//RS::get_singleton()->material_set_shader(mat->get_rid(), shader_rid);
	return mat;

	Ref<MaterialXShader> shader = memnew(MaterialXShader);
	shader->set_path(p_original_path);

	Error err = shader->load_file(p_original_path);
	if (err != OK) {
		*r_error = err;
		return nullptr;
	}

	mat->set_shader(shader);
}

void ResourceFormatLoaderMtlx::get_recognized_extensions(List<String> *p_extensions) const {
	if (!p_extensions->find("mtlx")) {
		p_extensions->push_back("mtlx");
	}
}

bool ResourceFormatLoaderMtlx::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "ShaderMaterial");
}

String ResourceFormatLoaderMtlx::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "mtlx") {
		return "ShaderMaterial";
	}
	return "";
}

Error MaterialXShader::load_file(const String &p_path) {
	// Prepare + Load MaterialX document.
	mx::DocumentPtr doc = mx::createDocument();

	String resource_path = ProjectSettings::get_singleton()->globalize_path(p_path);
	mx::FilePath mx_absolute_path = mx::FilePath(resource_path.utf8().get_data());

	mx::readFromXmlFile(doc, mx_absolute_path);

	// Make sure that worked.
	std::string message;
	bool document_valid = doc->validate(&message);
	if (!document_valid) {
		String msg = vformat("The MaterialX document is invalid: [%s] %s", resource_path, message.c_str());
		ERR_FAIL_V_MSG(ERR_PARSE_ERROR, msg);
	}

	int node_i = 2;
	for (mx::NodeGraphPtr node_graph : doc->getNodeGraphs()) {
		String graph_name = node_graph->getName().c_str();
		if (!graph_name.begins_with("NG_")) {
			WARN_PRINT(graph_name);
			continue;
		}

		print_line(vformat("MaterialX nodegraph %s", graph_name));
		std::vector<mx::ElementPtr> sorted_nodes = node_graph->topologicalSort();

		for (const mx::ElementPtr &element : sorted_nodes) {
			if (element->getCategory() == "output") {
				StringName key = element->getName().c_str();
				StringName output_node = element->getAttribute("nodename").c_str();
				output_to_node_map[key] = output_node;
				continue;
			}

			const mx::NodePtr &node = element->asA<mx::Node>();
			if (!node) {
				continue;
			}

			Ref<VisualShaderNode> new_node = _read_node(node, node_i);
			if (new_node.is_null()) {
				WARN_PRINT(vformat("Skipping node [%s] of type [%s]", node->getName().c_str(), node->getCategory().c_str()));
				continue;
			}

			node_id_map[node->getName().c_str()] = node_i;

			Ref<VisualShaderNodeFrame> frame_node;
			frame_node.instantiate();
			frame_node->set_title(node->getName().c_str());

			add_node(VisualShader::TYPE_FRAGMENT, new_node, Vector2(150, -200 + 100 * node_i), node_i);
			add_node(VisualShader::TYPE_FRAGMENT, frame_node, Vector2(150, 0 + 100 * node_i), node_i + 1);
			attach_node_to_frame(VisualShader::TYPE_FRAGMENT, node_i, node_i + 1);

			node_i += 2;
		}
	}

	for (mx::ElementPtr element : doc->getChildren()) {
		std::string shader_type = element->getCategory();
		if (shader_type == "disney_principled") {
		} else if (shader_type == "gltf_pbr") {
		} else if (shader_type == "open_pbr_surface") {
		} else if (shader_type == "standard_surface") {
			mx::NodePtr shader_node = element->asA<mx::Node>();

			Ref<VisualShaderNodeStandardSurface> surface_shader = memnew(VisualShaderNodeStandardSurface);
			add_node(VisualShader::TYPE_FRAGMENT, surface_shader, Vector2(0, 0), node_i);

			for (int port = 0; port < surface_shader->get_input_port_count(); port++) {
				std::string port_name = surface_shader->get_input_port_name(port).utf8().get_data();

				mx::OutputPtr graph_output = shader_node->getConnectedOutput(port_name);
				if (graph_output) {
					StringName node_output = output_to_node_map[graph_output->getName().c_str()];
					if (node_id_map.has(node_output)) {
						int node_source_id = node_id_map.get(node_output);
						connect_nodes(VisualShader::TYPE_FRAGMENT, node_source_id, 0, node_i, port);
					} else {
						ERR_PRINT(vformat("Cannot connect nodes [%s] -> [output]", node_output));
					}
				} else {
					Variant def_value = _get_mtlx_value_as_port_value(shader_node->getInputValue(port_name));
					surface_shader->set_input_port_default_value(port, def_value);
				}
			}

			connect_nodes(VisualShader::TYPE_FRAGMENT, node_i, 0, VisualShader::NODE_ID_OUTPUT, 2);

			//!
			//_connect_or_default(shader_node, "base_color", 0); // Albedo

			//_connect_or_default(shader_node, "opacity", 1); // Alpha

			//!
			//_connect_or_default(shader_node, "metalness", 2); // Metallic
			//_connect_or_default(shader_node, "roughness", 3); // Roughness
			//_connect_or_default(shader_node, "specular", 4); // Specular

			//_connect_or_default(shader_node, "emission", 5); // Emission
			//_connect_or_default(shader_node, "base_color", 6); // AO
			//_connect_or_default(shader_node, "base_color", 7); // AO Light Effect

			//_connect_or_default(shader_node, "normal", 8); // Normal

			//!
			//_connect_or_default(shader_node, "normal", 9); // Normal Map

			//_connect_or_default(shader_node, "base_color", 10); // Normal Map Depth

			//_connect_or_default(shader_node, "base_color", 11); // Bent Normal Map

			//_connect_or_default(shader_node, "base_color", 12); // Rim
			//_connect_or_default(shader_node, "base_color", 13); // Rim Tint
			//_connect_or_default(shader_node, "base_color", 14); // Clearcoat
			//_connect_or_default(shader_node, "base_color", 15); // Clearcoat Roughness
			//_connect_or_default(shader_node, "base_color", 16); // Anisotropy
			//_connect_or_default(shader_node, "base_color", 17); // Anisotropy Flow
			//_connect_or_default(shader_node, "base_color", 18); // Subsurf Scatter
			//_connect_or_default(shader_node, "base_color", 19); // Backlight

			//_connect_or_default(shader_node, "base_color", 20); // Alpha Scissor Threshold
			//_connect_or_default(shader_node, "base_color", 21); // Alpha Hash Scale
			//_connect_or_default(shader_node, "base_color", 22); // Alpha AA Edge
			//_connect_or_default("base_color", 23); // Alpha UV

			//_connect_or_default(shader_node, "base_color", 24); // Depth
		} else if (shader_type == "usd_preview_surface") {
		}
	}

	for (NodeConnection connection : node_connections) {
		if (!node_id_map.has(connection.node_out) || !node_id_map.has(connection.node_in)) {
			ERR_PRINT(vformat("Cannot connect nodes [%s] -> [%s]", connection.node_out, connection.node_in));
			continue;
		}

		int node_out_id = node_id_map.get(connection.node_out);
		int node_in_id = node_id_map.get(connection.node_in);
		connect_nodes(VisualShader::TYPE_FRAGMENT, node_out_id, 0, node_in_id, connection.node_in_port);
	}

	return OK;
}

void MaterialXShader::_connection_or_default(Ref<VisualShaderNode> p_shader_node, int p_port, const mx::NodePtr &p_mtlx_node, std::string p_input) {
	std::string nodename = p_mtlx_node->getConnectedNodeName(p_input);
	if (!nodename.empty()) {
		MaterialXShader::NodeConnection connection;
		connection.node_out = nodename.c_str();
		connection.node_in = p_mtlx_node->getName().c_str();
		connection.node_in_port = p_port;
		node_connections.push_back(connection);
	} else {
		Variant value = _get_mtlx_value_as_port_value(p_mtlx_node->getInputValue(p_input));
		p_shader_node->set_input_port_default_value(p_port, value);
	}
}

Ref<VisualShaderNode> MaterialXShader::_read_node(const mx::NodePtr &p_node, int p_id) {
	std::string category = p_node->getCategory();
	if (category == "image") {
		Ref<VisualShaderNodeTexture> texture_node;
		texture_node.instantiate();

		String file = p_node->getInput("file")->getValueString().c_str();
		String path = vformat("%s/%s", get_path().get_base_dir(), file);
		if (FileAccess::exists(path)) {
			Ref<Texture2D> texture = ResourceLoader::load(path, "Texture2D");
			texture_node->set_texture(texture);
		}

		//_connection_or_default(texture_node, 0, p_node, "layer");
		//_connection_or_default(texture_node, 0, p_node, "default");
		_connection_or_default(texture_node, 0, p_node, "texcoord");
		//_connection_or_default(texture_node, 0, p_node, "uaddressmode");
		//_connection_or_default(texture_node, 0, p_node, "filtertype");

		return texture_node;
	} else if (category == "tiledimage") {
		return nullptr;
	} else if (category == "triplanarprojection") {
		return nullptr;
	} else if (category == "constant") {
		return _read_constant_node(p_node);
	} else if (category == "ramplr") {
		return nullptr;
	} else if (category == "ramptb") {
		return nullptr;
	} else if (category == "ramp4") {
		return nullptr;
	} else if (category == "splitlr") {
		return nullptr;
	} else if (category == "splittb") {
		return nullptr;
	} else if (category == "randomfloat") {
		return nullptr;
	} else if (category == "randomcolor") {
		return nullptr;
	} else if (category == "noise2d") {
		return nullptr;
	} else if (category == "noise3d") {
		return nullptr;
	} else if (category == "fractal3d") {
		return nullptr;
	} else if (category == "cellnoise2d") {
		return nullptr;
	} else if (category == "cellnoise3d") {
		return nullptr;
	} else if (category == "worleynoise2d") {
		return nullptr;
	} else if (category == "worleynoise3d") {
		return nullptr;
	} else if (category == "unifiednoise2d") {
		return nullptr;
	} else if (category == "unifiednoise3d") {
		return nullptr;
	} else if (category == "checkerboard") {
		return nullptr;
	} else if (category == "line") {
		return nullptr;
	} else if (category == "circle") {
		return nullptr;
	} else if (category == "cloverleaf") {
		return nullptr;
	} else if (category == "hexagon") {
		return nullptr;
	} else if (category == "grid") {
		return nullptr;
	} else if (category == "crosshatch") {
		return nullptr;
	} else if (category == "tiledcircles") {
		return nullptr;
	} else if (category == "tiledcloverleafs") {
		return nullptr;
	} else if (category == "tiledhexagons") {
		return nullptr;
	} else if (category == "position") {
		return nullptr;
	} else if (category == "normal") {
		Ref<VisualShaderNodeInput> input_node;
		input_node.instantiate();

		input_node->set_input_name("normal");
		//_connection_or_default(input_node, 0, p_node, "space");

		return input_node;
	} else if (category == "tangent") {
		Ref<VisualShaderNodeInput> input_node;
		input_node.instantiate();

		input_node->set_input_name("tangent");
		//_connection_or_default(input_node, 0, p_node, "space");
		//_connection_or_default(input_node, 0, p_node, "index");

		return input_node;
	} else if (category == "bitangent") {
		return nullptr;
	} else if (category == "bump") {
		return nullptr;
	} else if (category == "texcoord") {
		Ref<VisualShaderNodeInput> input_node;
		input_node.instantiate();

		input_node->set_input_name("uv");
		//_connection_or_default(input_node, 0, p_node, "index");

		return input_node;
	} else if (category == "geomcolor") {
		return nullptr;
	} else if (category == "geompropvalue") {
		return nullptr;
	} else if (category == "geompropvalueuniform") {
		return nullptr;
	} else if (category == "frame") {
		return nullptr;
	} else if (category == "time") {
		return nullptr;
	} else if (category == "add") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_ADD);
		} else if (type == "vector2") {
			Ref<VisualShaderNodeVectorOp> vector_op = memnew(VisualShaderNodeVectorOp);
			vector_op->set_op_type(VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D);
			vector_op->set_operator(VisualShaderNodeVectorOp::OP_ADD);

			_connection_or_default(vector_op, 0, p_node, "in1");
			_connection_or_default(vector_op, 1, p_node, "in2");

			return vector_op;
		} else if (type == "vector3") {
			Ref<VisualShaderNodeVectorOp> vector_op = memnew(VisualShaderNodeVectorOp);
			vector_op->set_op_type(VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D);
			vector_op->set_operator(VisualShaderNodeVectorOp::OP_ADD);

			_connection_or_default(vector_op, 0, p_node, "in1");
			_connection_or_default(vector_op, 1, p_node, "in2");

			return vector_op;
		}

		return nullptr;
	} else if (category == "subtract") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_SUB);
		}

		return nullptr;
	} else if (category == "multiply") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_MUL);
		} else if (type == "vector2") {
			Ref<VisualShaderNodeVectorOp> vector_op = memnew(VisualShaderNodeVectorOp);
			vector_op->set_op_type(VisualShaderNodeVectorOp::OP_TYPE_VECTOR_2D);
			vector_op->set_operator(VisualShaderNodeVectorOp::OP_MUL);

			_connection_or_default(vector_op, 0, p_node, "in1");
			_connection_or_default(vector_op, 1, p_node, "in2");

			return vector_op;
		} else if (type == "vector3") {
			Ref<VisualShaderNodeVectorOp> vector_op = memnew(VisualShaderNodeVectorOp);
			vector_op->set_op_type(VisualShaderNodeVectorOp::OP_TYPE_VECTOR_3D);
			vector_op->set_operator(VisualShaderNodeVectorOp::OP_MUL);

			_connection_or_default(vector_op, 0, p_node, "in1");
			_connection_or_default(vector_op, 1, p_node, "in2");

			return vector_op;
		}

		return nullptr;
	} else if (category == "divide") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_DIV);
		}

		return nullptr;
	} else if (category == "modulo") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_MOD);
		}

		return nullptr;
	} else if (category == "fract") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_FRACT);
		}

		return nullptr;
	} else if (category == "invert") {
		return nullptr;
	} else if (category == "absval") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_ABS);
		}

		return nullptr;
	} else if (category == "sign") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_SIGN);
		}

		return nullptr;
	} else if (category == "floor") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_FLOOR);
		}

		return nullptr;
	} else if (category == "ceil") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_CEIL);
		}

		return nullptr;
	} else if (category == "round") {
		return nullptr;
	} else if (category == "power") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_POW);
		}

		return nullptr;
	} else if (category == "safepower") {
		return nullptr;
	} else if (category == "sin") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_SIN);
		}

		return nullptr;
	} else if (category == "cos") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_COS);
		}

		return nullptr;
	} else if (category == "tan") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_TAN);
		}

		return nullptr;
	} else if (category == "atan") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_ATAN);
		}

		return nullptr;
	} else if (category == "atan2") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_ATAN2);
		}

		return nullptr;
	} else if (category == "sqrt") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_SQRT);
		}

		return nullptr;
	} else if (category == "ln") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_LOG); //! is this log10 or ln? we require log_e
		}

		return nullptr;
	} else if (category == "exp") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_unary_operator_node(p_node, VisualShaderNodeFloatFunc::FUNC_EXP);
		}

		return nullptr;
	} else if (category == "clamp") {
		Ref<VisualShaderNodeClamp> clamp_node;
		clamp_node.instantiate();

		_connection_or_default(clamp_node, 0, p_node, "in");
		_connection_or_default(clamp_node, 1, p_node, "low");
		_connection_or_default(clamp_node, 2, p_node, "high");

		std::string type = p_node->getType();
		if (type == "float") {
			clamp_node->set_op_type(VisualShaderNodeClamp::OP_TYPE_FLOAT);
		}

		return clamp_node;
	} else if (category == "trianglewave") {
		return nullptr;
	} else if (category == "min") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_MIN);
		}

		return nullptr;
	} else if (category == "max") {
		std::string type = p_node->getType();
		if (type == "float") {
			return _read_binary_operator_node(p_node, VisualShaderNodeFloatOp::OP_MAX);
		}

		return nullptr;
	} else if (category == "normalize") {
		return nullptr;
	} else if (category == "magnitude") {
		return nullptr;
	} else if (category == "distance") {
		return nullptr;
	} else if (category == "dotproduct") {
		return nullptr;
	} else if (category == "crossproduct") {
		return nullptr;
	} else if (category == "transformpoint") {
		return nullptr;
	} else if (category == "transformvector") {
		return nullptr;
	} else if (category == "transformnormal") {
		return nullptr;
	} else if (category == "transformmatrix") {
		return nullptr;
	} else if (category == "normalmap") {
		//! This is a hack based on the expectations of standard surface shaders
		// These shaders always pass their "normal" through a "normalmap" node before sending to "normal" output
		// We connect the standard shader "normal" output to godot's normal map output which is why this can be a noop
		Ref<VisualShaderNodeReroute> noop_node;
		noop_node.instantiate();

		_connection_or_default(noop_node, 0, p_node, "in");
		//_connection_or_default(noop_node, 1, p_node, "scale");
		//_connection_or_default(noop_node, 2, p_node, "normal");
		//_connection_or_default(noop_node, 3, p_node, "tangent");
		//_connection_or_default(noop_node, 4, p_node, "bitangent");

		return noop_node;
	} else if (category == "creatematrix") {
		return nullptr;
	} else if (category == "transform") {
		return nullptr;
	} else if (category == "determinant") {
		return nullptr;
	} else if (category == "invertmatrix") {
		return nullptr;
	} else if (category == "rotate2d") {
		return nullptr;
	} else if (category == "rotate3d") {
		return nullptr;
	} else if (category == "reflect") {
		return nullptr;
	} else if (category == "refract") {
		return nullptr;
	} else if (category == "place2d") {
		return nullptr;
	} else if (category == "dot") {
		Ref<VisualShaderNodeReroute> reroute;
		reroute.instantiate();

		_connection_or_default(reroute, 0, p_node, "in");

		return reroute;
	} else if (category == "and") {
		return nullptr;
	} else if (category == "or") {
		return nullptr;
	} else if (category == "xor") {
		return nullptr;
	} else if (category == "not") {
		return nullptr;
	} else if (category == "contrast") {
		return nullptr;
	} else if (category == "remap") {
		return nullptr;
	} else if (category == "range") {
		return nullptr;
	} else if (category == "smoothstep") {
		return nullptr;
	} else if (category == "luminance") {
		return nullptr;
	} else if (category == "rgbtohsv") {
		return nullptr;
	} else if (category == "hsvtorgb") {
		return nullptr;
	} else if (category == "hsvadjust") {
		return nullptr;
	} else if (category == "saturate") {
		return nullptr;
	} else if (category == "colorcorrect") {
		return nullptr;
	} else if (category == "premult") {
		return nullptr;
	} else if (category == "unpremult") {
		return nullptr;
	} else if (category == "plus") {
		return nullptr;
	} else if (category == "minus") {
		return nullptr;
	} else if (category == "difference") {
		return nullptr;
	} else if (category == "burn") {
		return nullptr;
	} else if (category == "dodge") {
		return nullptr;
	} else if (category == "screen") {
		return nullptr;
	} else if (category == "overlay") {
		return nullptr;
	} else if (category == "disjointover") {
		return nullptr;
	} else if (category == "in") {
		return nullptr;
	} else if (category == "mask") {
		return nullptr;
	} else if (category == "matte") {
		return nullptr;
	} else if (category == "out") {
		return nullptr;
	} else if (category == "over") {
		return nullptr;
	} else if (category == "inside") {
		return nullptr;
	} else if (category == "outside") {
		return nullptr;
	} else if (category == "mix") {
		Ref<VisualShaderNodeMix> mix_node;
		mix_node.instantiate();

		_connection_or_default(mix_node, 0, p_node, "fg");
		_connection_or_default(mix_node, 1, p_node, "bg");
		_connection_or_default(mix_node, 2, p_node, "mix");

		std::string type = p_node->getType();
		std::string weight_type = p_node->getInput("mix")->getType();
		if (type == "float" || type == "integer") {
			mix_node->set_op_type(VisualShaderNodeMix::OP_TYPE_SCALAR);
		} else if (type == "vector2") {
			VisualShaderNodeMix::OpType op_type = weight_type == "float" ? VisualShaderNodeMix::OP_TYPE_VECTOR_2D_SCALAR : VisualShaderNodeMix::OP_TYPE_VECTOR_2D;
			mix_node->set_op_type(op_type);
		} else if (type == "vector3" || type == "color3") {
			VisualShaderNodeMix::OpType op_type = weight_type == "float" ? VisualShaderNodeMix::OP_TYPE_VECTOR_3D_SCALAR : VisualShaderNodeMix::OP_TYPE_VECTOR_3D;
			mix_node->set_op_type(op_type);
		} else if (type == "vector4" || type == "color4") {
			VisualShaderNodeMix::OpType op_type = weight_type == "float" ? VisualShaderNodeMix::OP_TYPE_VECTOR_4D_SCALAR : VisualShaderNodeMix::OP_TYPE_VECTOR_4D;
			mix_node->set_op_type(op_type);
		}

		return mix_node;
	} else if (category == "ifgreater") {
		return nullptr;
	} else if (category == "ifgreatereq") {
		return nullptr;
	} else if (category == "ifequal") {
		return nullptr;
	} else if (category == "switch") {
		return nullptr;
	} else if (category == "extract") {
		Ref<VisualShaderNodeExtract> extract_node;
		extract_node.instantiate();

		_connection_or_default(extract_node, 0, p_node, "in");
		_connection_or_default(extract_node, 1, p_node, "index");

		std::string type = p_node->getType();
		if (type == "vector2") {
			extract_node->set_op_type(VisualShaderNodeVectorBase::OP_TYPE_VECTOR_2D);
		} else if (type == "vector3") {
			extract_node->set_op_type(VisualShaderNodeVectorBase::OP_TYPE_VECTOR_3D);
		} else {
			extract_node->set_op_type(VisualShaderNodeVectorBase::OP_TYPE_VECTOR_4D);
		}

		return extract_node;
	} else if (category == "convert") {
		return nullptr;
	} else if (category == "combine2") {
		Ref<VisualShaderNodeVectorCompose> combine_node;
		combine_node.instantiate();
		combine_node->set_op_type(VisualShaderNodeVectorCompose::OP_TYPE_VECTOR_2D);

		_connection_or_default(combine_node, 0, p_node, "in1");
		_connection_or_default(combine_node, 1, p_node, "in2");

		return combine_node;
	} else if (category == "combine3") {
		Ref<VisualShaderNodeVectorCompose> combine_node;
		combine_node.instantiate();
		combine_node->set_op_type(VisualShaderNodeVectorCompose::OP_TYPE_VECTOR_3D);

		_connection_or_default(combine_node, 0, p_node, "in1");
		_connection_or_default(combine_node, 1, p_node, "in2");
		_connection_or_default(combine_node, 2, p_node, "in3");

		return combine_node;
	} else if (category == "combine4") {
		Ref<VisualShaderNodeVectorCompose> combine_node;
		combine_node.instantiate();
		combine_node->set_op_type(VisualShaderNodeVectorCompose::OP_TYPE_VECTOR_4D);

		_connection_or_default(combine_node, 0, p_node, "in1");
		_connection_or_default(combine_node, 1, p_node, "in2");
		_connection_or_default(combine_node, 2, p_node, "in3");
		_connection_or_default(combine_node, 3, p_node, "in4");

		return combine_node;
	} else if (category == "separate2") {
		return nullptr;
	} else if (category == "separate3") {
		return nullptr;
	} else if (category == "separate4") {
		return nullptr;
	} else if (category == "blur") {
		return nullptr;
	} else if (category == "heighttonormal") {
		return nullptr;
	} else {
		ERR_PRINT(vformat("Unknown node type: [%s]", category.c_str()));
		return nullptr;
	}

	/* shader node types
	else if (category == "surface_unlit") {
		return nullptr;
	} else if (category == "displacement") {
		return nullptr;
	} else if (category == "mix") {
		return nullptr;
	}*/
}

Ref<VisualShaderNodeConstant> MaterialXShader::_read_constant_node(const mx::NodePtr &p_node) const {
	Variant value = _get_mtlx_value_as_port_value(p_node->getInput("value")->getValue());

	std::string type = p_node->getType();
	if (type == "float") {
		Ref<VisualShaderNodeFloatConstant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "integer") {
		Ref<VisualShaderNodeIntConstant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "vector2") {
		Ref<VisualShaderNodeVec2Constant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "vector3") {
		Ref<VisualShaderNodeVec3Constant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "vector4") {
		Ref<VisualShaderNodeVec4Constant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "color3") {
		//! Visual shaders don't seem to have a color3 constant
		Ref<VisualShaderNodeVec3Constant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "color4") {
		Ref<VisualShaderNodeColorConstant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else if (type == "boolean") {
		Ref<VisualShaderNodeBooleanConstant> constant_node;
		constant_node.instantiate();
		constant_node->set_constant(value);
		return constant_node;
	} else {
		WARN_PRINT(vformat("Could not return value of type %s", type.c_str()));
		return Ref<VisualShaderNodeConstant>();
	}
}

Ref<VisualShaderNodeFloatFunc> MaterialXShader::_read_unary_operator_node(const mx::NodePtr &p_node, VisualShaderNodeFloatFunc::Function p_unary_op) {
	Ref<VisualShaderNodeFloatFunc> float_op_node;
	float_op_node.instantiate();
	float_op_node->set_function(p_unary_op);

	_connection_or_default(float_op_node, 0, p_node, "in");

	return float_op_node;
}

Ref<VisualShaderNodeFloatOp> MaterialXShader::_read_binary_operator_node(const mx::NodePtr &p_node, VisualShaderNodeFloatOp::Operator p_binary_op) {
	Ref<VisualShaderNodeFloatOp> float_op_node;
	float_op_node.instantiate();
	float_op_node->set_operator(p_binary_op);

	_connection_or_default(float_op_node, 0, p_node, "in1");
	_connection_or_default(float_op_node, 1, p_node, "in2");

	return float_op_node;
}

int MaterialXShader::_get_mtlx_type_as_port_type(const std::string &p_type_string) {
	//Unused port types
	//PORT_TYPE_SCALAR_UINT
	//PORT_TYPE_TRANSFORM
	//PORT_TYPE_SAMPLER

	if (p_type_string == "float") {
		return VisualShaderNode::PORT_TYPE_SCALAR;
	} else if (p_type_string == "integer") {
		return VisualShaderNode::PORT_TYPE_SCALAR_INT;
	} else if (p_type_string == "vector2") {
		return VisualShaderNode::PORT_TYPE_VECTOR_2D;
	} else if (p_type_string == "vector3" || p_type_string == "color3") {
		return VisualShaderNode::PORT_TYPE_VECTOR_3D;
	} else if (p_type_string == "vector4" || p_type_string == "color4") {
		return VisualShaderNode::PORT_TYPE_VECTOR_4D;
	} else if (p_type_string == "boolean") {
		return VisualShaderNode::PORT_TYPE_BOOLEAN;
	} else {
		return VisualShaderNode::PORT_TYPE_MAX;
	}
}

Variant MaterialXShader::_get_mtlx_value_as_port_value(const mx::ValuePtr &p_value) {
	if (!p_value) {
		return Variant();
	}

	std::string type_string = p_value->getTypeString();
	if (type_string == "float") {
		return p_value->asA<float>();
	} else if (type_string == "integer") {
		return p_value->asA<int>();
	} else if (type_string == "vector2") {
		mx::Vector2 vector_2 = p_value->asA<mx::Vector2>();
		return Vector2(vector_2[0], vector_2[1]);
	} else if (type_string == "vector3") {
		mx::Vector3 vector_3 = p_value->asA<mx::Vector3>();
		return Vector3(vector_3[0], vector_3[1], vector_3[2]);
	} else if (type_string == "color3") {
		mx::Color3 vector_3 = p_value->asA<mx::Color3>();
		return Vector3(vector_3[0], vector_3[1], vector_3[2]);
	} else if (type_string == "vector4") {
		mx::Vector4 vector_4 = p_value->asA<mx::Vector4>();
		return Vector4(vector_4[0], vector_4[1], vector_4[2], vector_4[3]);
	} else if (type_string == "color4") {
		mx::Color4 vector_4 = p_value->asA<mx::Color4>();
		return Vector4(vector_4[0], vector_4[1], vector_4[2], vector_4[3]);
	} else if (type_string == "boolean") {
		return p_value->asA<bool>();
	} else {
		return Variant();
	}
}
