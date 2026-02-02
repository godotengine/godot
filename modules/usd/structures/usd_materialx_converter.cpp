/**************************************************************************/
/*  usd_materialx_converter.cpp                                           */
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

#include "usd_materialx_converter.h"

#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/resource_loader.h"
#include "core/io/xml_parser.h"
#include "scene/resources/image_texture.h"

// ============================================================================
// Binding
// ============================================================================

void USDMaterialXConverter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("convert_materialx_from_xml", "xml", "base_path"), &USDMaterialXConverter::convert_materialx_from_xml, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("compile_materialx_to_shader", "xml"), &USDMaterialXConverter::compile_materialx_to_shader);
	ClassDB::bind_method(D_METHOD("parse_materialx_document", "xml"), &USDMaterialXConverter::parse_materialx_document);
	ClassDB::bind_method(D_METHOD("clear_cache"), &USDMaterialXConverter::clear_cache);
	ClassDB::bind_method(D_METHOD("get_cache_size"), &USDMaterialXConverter::get_cache_size);
}

// ============================================================================
// Public API
// ============================================================================

Ref<ShaderMaterial> USDMaterialXConverter::convert_materialx_from_xml(const String &p_xml, const String &p_base_path) {
	String shader_code = compile_materialx_to_shader(p_xml);
	if (shader_code.is_empty()) {
		return Ref<ShaderMaterial>();
	}

	// Check cache.
	String cache_key = _compute_cache_key(shader_code);
	if (shader_cache.has(cache_key)) {
		return shader_cache[cache_key];
	}

	// Create shader.
	Ref<Shader> shader;
	shader.instantiate();
	shader->set_code(shader_code);

	// Create material.
	Ref<ShaderMaterial> material;
	material.instantiate();
	material->set_shader(shader);

	// Bind textures from the parsed graph.
	MtlxNodeGraph graph = _parse_materialx_xml(p_xml);
	Array tex_keys = graph.texture_paths.keys();
	for (int i = 0; i < tex_keys.size(); i++) {
		String node_name = tex_keys[i];
		String tex_path_value = graph.texture_paths[node_name];
		if (tex_path_value.is_empty()) {
			continue;
		}

		// Resolve the path relative to the base directory.
		String resolved_path = tex_path_value;
		if (!tex_path_value.is_absolute_path() && !tex_path_value.begins_with("res://") && !p_base_path.is_empty()) {
			resolved_path = p_base_path.path_join(tex_path_value);
			resolved_path = resolved_path.simplify_path();
		}

		// Try ResourceLoader first.
		Ref<Texture2D> texture;
		if (resolved_path.begins_with("res://")) {
			Ref<Resource> res = ResourceLoader::load(resolved_path);
			if (res.is_valid()) {
				texture = res;
			}
		}

		// Fall back to direct image loading.
		if (texture.is_null() && FileAccess::exists(resolved_path)) {
			Ref<Image> image;
			image.instantiate();
			Error err = image->load(resolved_path);
			if (err == OK) {
				texture = ImageTexture::create_from_image(image);
			}
		}

		if (texture.is_valid()) {
			String uniform_name = "tex_" + node_name;
			material->set_shader_parameter(uniform_name, texture);
		}
	}

	// Cache the result.
	_evict_shader_cache_if_needed();
	shader_cache[cache_key] = material;

	return material;
}

String USDMaterialXConverter::compile_materialx_to_shader(const String &p_xml) const {
	MtlxNodeGraph graph = _parse_materialx_xml(p_xml);
	if (graph.name.is_empty() && graph.nodes.is_empty()) {
		WARN_PRINT("USDMaterialXConverter: Failed to parse MaterialX document.");
		return String();
	}
	return _compile_graph_to_shader(graph);
}

Dictionary USDMaterialXConverter::parse_materialx_document(const String &p_xml) const {
	MtlxNodeGraph graph = _parse_materialx_xml(p_xml);

	Dictionary result;
	if (graph.name.is_empty()) {
		return result;
	}

	result["name"] = graph.name;
	result["output_node"] = graph.output_node;
	result["output_type"] = graph.output_type;

	Array nodes_array;
	for (int i = 0; i < graph.nodes.size(); i++) {
		const MtlxNodeInfo &node = graph.nodes[i];
		Dictionary node_dict;
		node_dict["name"] = node.name;
		node_dict["node_type"] = node.node_type;
		node_dict["output_type"] = node.output_type;
		node_dict["inputs"] = node.inputs;
		node_dict["input_types"] = node.input_types;
		node_dict["input_connections"] = node.input_connections;
		nodes_array.push_back(node_dict);
	}
	result["nodes"] = nodes_array;
	result["texture_paths"] = graph.texture_paths;

	return result;
}

void USDMaterialXConverter::clear_cache() {
	shader_cache.clear();
}

int USDMaterialXConverter::get_cache_size() const {
	return shader_cache.size();
}

// ============================================================================
// Cache helpers
// ============================================================================

String USDMaterialXConverter::_compute_cache_key(const String &p_shader_code) const {
	return String::num_int64(p_shader_code.hash());
}

void USDMaterialXConverter::_evict_shader_cache_if_needed() {
	if (shader_cache.size() < MAX_SHADER_CACHE_SIZE) {
		return;
	}

	Array keys = shader_cache.keys();
	int to_remove = keys.size() / 2;
	WARN_PRINT(vformat("USDMaterialXConverter: Shader cache exceeded %d entries, evicting %d oldest entries.", MAX_SHADER_CACHE_SIZE, to_remove));
	for (int i = 0; i < to_remove; i++) {
		shader_cache.erase(keys[i]);
	}
}

// ============================================================================
// Node lookup helper
// ============================================================================

const MtlxNodeInfo *USDMaterialXConverter::_find_node(
		const String &p_name,
		const MtlxNodeGraph &p_graph) const {
	for (int i = 0; i < p_graph.nodes.size(); i++) {
		if (p_graph.nodes[i].name == p_name) {
			return &p_graph.nodes[i];
		}
	}
	return nullptr;
}

// ============================================================================
// MaterialX XML Parsing
// ============================================================================

MtlxNodeGraph USDMaterialXConverter::_parse_materialx_xml(const String &p_xml) const {
	MtlxNodeGraph graph;

	if (p_xml.is_empty()) {
		return graph;
	}

	Ref<XMLParser> parser;
	parser.instantiate();

	Error err = parser->open_buffer(p_xml.to_utf8_buffer());
	if (err != OK) {
		WARN_PRINT("USDMaterialXConverter: Failed to parse MaterialX XML.");
		return graph;
	}

	// Parsing state.
	String current_node_graph_name;
	bool in_nodegraph = false;
	bool in_surfacematerial = false;
	String surface_material_name;
	String surface_shader_ref;

	while (parser->read() == OK) {
		XMLParser::NodeType node_type = parser->get_node_type();

		if (node_type == XMLParser::NODE_ELEMENT) {
			String tag_name = parser->get_node_name();

			// -- <materialx> root element --
			if (tag_name == "materialx") {
				if (graph.name.is_empty()) {
					graph.name = "materialx_root";
				}
				continue;
			}

			// -- <nodegraph> element --
			if (tag_name == "nodegraph") {
				in_nodegraph = true;
				current_node_graph_name = "";
				for (int i = 0; i < parser->get_attribute_count(); i++) {
					if (parser->get_attribute_name(i) == "name") {
						current_node_graph_name = parser->get_attribute_value(i);
						if (graph.name == "materialx_root") {
							graph.name = current_node_graph_name;
						}
					}
				}
				continue;
			}

			// -- <surfacematerial> element --
			if (tag_name == "surfacematerial") {
				in_surfacematerial = true;
				for (int i = 0; i < parser->get_attribute_count(); i++) {
					if (parser->get_attribute_name(i) == "name") {
						surface_material_name = parser->get_attribute_value(i);
					}
				}
				continue;
			}

			// -- <input> inside surfacematerial --
			if (tag_name == "input" && in_surfacematerial) {
				String input_name;
				String input_nodegraph;
				for (int i = 0; i < parser->get_attribute_count(); i++) {
					String attr = parser->get_attribute_name(i);
					if (attr == "name") {
						input_name = parser->get_attribute_value(i);
					} else if (attr == "nodegraph") {
						input_nodegraph = parser->get_attribute_value(i);
					}
				}
				if (input_name == "surfaceshader" && !input_nodegraph.is_empty()) {
					surface_shader_ref = input_nodegraph;
				}
				continue;
			}

			// -- <output> inside nodegraph --
			if (tag_name == "output" && in_nodegraph) {
				String output_type;
				String output_nodename;
				for (int i = 0; i < parser->get_attribute_count(); i++) {
					String attr = parser->get_attribute_name(i);
					if (attr == "type") {
						output_type = parser->get_attribute_value(i);
					} else if (attr == "nodename") {
						output_nodename = parser->get_attribute_value(i);
					}
				}
				if (!output_nodename.is_empty()) {
					graph.output_node = output_nodename;
					graph.output_type = output_type;
				}
				continue;
			}

			// -- Node elements inside <nodegraph> --
			if (in_nodegraph && tag_name != "input" && tag_name != "output") {
				MtlxNodeInfo node_info;
				node_info.node_type = tag_name;

				for (int i = 0; i < parser->get_attribute_count(); i++) {
					String attr = parser->get_attribute_name(i);
					if (attr == "name") {
						node_info.name = parser->get_attribute_value(i);
					} else if (attr == "type") {
						node_info.output_type = parser->get_attribute_value(i);
					}
				}

				// Parse child <input> elements.
				if (!parser->is_empty()) {
					int depth = 1;
					while (parser->read() == OK && depth > 0) {
						XMLParser::NodeType inner_type = parser->get_node_type();

						if (inner_type == XMLParser::NODE_ELEMENT) {
							String inner_tag = parser->get_node_name();
							if (inner_tag == "input") {
								String inp_name;
								String inp_type;
								String inp_value;
								String inp_nodename;
								String inp_output;

								for (int j = 0; j < parser->get_attribute_count(); j++) {
									String a = parser->get_attribute_name(j);
									if (a == "name") {
										inp_name = parser->get_attribute_value(j);
									} else if (a == "type") {
										inp_type = parser->get_attribute_value(j);
									} else if (a == "value") {
										inp_value = parser->get_attribute_value(j);
									} else if (a == "nodename") {
										inp_nodename = parser->get_attribute_value(j);
									} else if (a == "output") {
										inp_output = parser->get_attribute_value(j);
									}
								}

								if (!inp_name.is_empty()) {
									node_info.input_types[inp_name] = inp_type;

									if (!inp_nodename.is_empty()) {
										String connection = inp_nodename;
										if (!inp_output.is_empty()) {
											connection += "." + inp_output;
										}
										node_info.input_connections[inp_name] = connection;
									} else if (!inp_value.is_empty()) {
										node_info.inputs[inp_name] = inp_value;
									}

									// Track texture file paths for image nodes.
									if (inp_name == "file" && node_info.node_type == "image") {
										graph.texture_paths[node_info.name] = inp_value;
									}
								}

								if (!parser->is_empty()) {
									depth++;
								}
							} else {
								if (!parser->is_empty()) {
									depth++;
								}
							}
						} else if (inner_type == XMLParser::NODE_ELEMENT_END) {
							depth--;
						}
					}
				}

				if (!node_info.name.is_empty()) {
					graph.nodes.push_back(node_info);
				}
				continue;
			}

			// -- Top-level node definitions (outside nodegraph) --
			if (!in_nodegraph && !in_surfacematerial &&
					tag_name != "input" && tag_name != "output" && tag_name != "materialx") {
				MtlxNodeInfo node_info;
				node_info.node_type = tag_name;

				for (int i = 0; i < parser->get_attribute_count(); i++) {
					String attr = parser->get_attribute_name(i);
					if (attr == "name") {
						node_info.name = parser->get_attribute_value(i);
					} else if (attr == "type") {
						node_info.output_type = parser->get_attribute_value(i);
					}
				}

				// Parse child <input> elements.
				if (!parser->is_empty()) {
					int depth = 1;
					while (parser->read() == OK && depth > 0) {
						XMLParser::NodeType inner_type = parser->get_node_type();

						if (inner_type == XMLParser::NODE_ELEMENT) {
							String inner_tag = parser->get_node_name();
							if (inner_tag == "input") {
								String inp_name;
								String inp_type;
								String inp_value;
								String inp_nodename;

								for (int j = 0; j < parser->get_attribute_count(); j++) {
									String a = parser->get_attribute_name(j);
									if (a == "name") {
										inp_name = parser->get_attribute_value(j);
									} else if (a == "type") {
										inp_type = parser->get_attribute_value(j);
									} else if (a == "value") {
										inp_value = parser->get_attribute_value(j);
									} else if (a == "nodename") {
										inp_nodename = parser->get_attribute_value(j);
									}
								}

								if (!inp_name.is_empty()) {
									node_info.input_types[inp_name] = inp_type;
									if (!inp_nodename.is_empty()) {
										node_info.input_connections[inp_name] = inp_nodename;
									} else if (!inp_value.is_empty()) {
										node_info.inputs[inp_name] = inp_value;
									}
								}

								if (!parser->is_empty()) {
									depth++;
								}
							} else {
								if (!parser->is_empty()) {
									depth++;
								}
							}
						} else if (inner_type == XMLParser::NODE_ELEMENT_END) {
							depth--;
						}
					}
				}

				if (!node_info.name.is_empty()) {
					graph.nodes.push_back(node_info);

					// If this is a surface shader type at top level, treat as output.
					if (node_info.output_type == "surfaceshader") {
						graph.output_node = node_info.name;
						graph.output_type = "surfaceshader";
					}
				}
				continue;
			}

		} else if (node_type == XMLParser::NODE_ELEMENT_END) {
			String tag_name = parser->get_node_name();
			if (tag_name == "nodegraph") {
				in_nodegraph = false;
			} else if (tag_name == "surfacematerial") {
				in_surfacematerial = false;
			}
		}
	}

	return graph;
}

// ============================================================================
// Input expression resolution (3-tier: connection -> constant -> default)
// ============================================================================

String USDMaterialXConverter::_resolve_input_expression(
		const String &p_input_name,
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms,
		String &r_functions) const {
	// Tier 1: Connected to another node.
	if (p_node.input_connections.has(p_input_name)) {
		String connection = p_node.input_connections[p_input_name];

		String node_name = connection;
		int dot_pos = connection.find(".");
		if (dot_pos >= 0) {
			node_name = connection.substr(0, dot_pos);
		}

		const MtlxNodeInfo *connected_node = _find_node(node_name, p_graph);
		if (connected_node) {
			return _compile_node_expression(*connected_node, p_graph, r_uniforms, r_functions);
		}

		// Node not found -- return a variable reference.
		return "mtlx_" + node_name;
	}

	// Tier 2: Constant value.
	if (p_node.inputs.has(p_input_name)) {
		String value = p_node.inputs[p_input_name];
		String type;
		if (p_node.input_types.has(p_input_name)) {
			type = p_node.input_types[p_input_name];
		}

		if (type == "float") {
			return value;
		} else if (type == "color3" || type == "vector3") {
			String cleaned = value.replace(" ", "");
			PackedStringArray parts = cleaned.split(",");
			if (parts.size() >= 3) {
				return "vec3(" + parts[0] + ", " + parts[1] + ", " + parts[2] + ")";
			}
			return "vec3(" + value + ")";
		} else if (type == "color4" || type == "vector4") {
			String cleaned = value.replace(" ", "");
			PackedStringArray parts = cleaned.split(",");
			if (parts.size() >= 4) {
				return "vec4(" + parts[0] + ", " + parts[1] + ", " + parts[2] + ", " + parts[3] + ")";
			}
			return "vec4(" + value + ")";
		} else if (type == "vector2") {
			String cleaned = value.replace(" ", "");
			PackedStringArray parts = cleaned.split(",");
			if (parts.size() >= 2) {
				return "vec2(" + parts[0] + ", " + parts[1] + ")";
			}
			return "vec2(" + value + ")";
		} else if (type == "integer" || type == "int") {
			return value;
		} else if (type == "boolean") {
			return value == "true" ? "true" : "false";
		} else if (type == "string" || type == "filename") {
			return "\"" + value + "\"";
		}

		// Default: return as-is.
		return value;
	}

	// Tier 3: Type-appropriate default.
	String input_type;
	if (p_node.input_types.has(p_input_name)) {
		input_type = p_node.input_types[p_input_name];
	}

	if (input_type == "color3" || input_type == "vector3") {
		return "vec3(0.0)";
	} else if (input_type == "color4" || input_type == "vector4") {
		return "vec4(0.0)";
	} else if (input_type == "vector2") {
		return "vec2(0.0)";
	}
	return "0.0";
}

// ============================================================================
// Node expression compiler (dispatcher)
// ============================================================================

String USDMaterialXConverter::_compile_node_expression(
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms,
		String &r_functions) const {
	String node_type = p_node.node_type;

	// Surface shader nodes.
	if (node_type == "standard_surface") {
		return _compile_standard_surface(p_node, p_graph, r_uniforms, r_functions);
	}

	// Image / texture nodes.
	if (node_type == "image" || node_type == "tiledimage") {
		return _compile_image_node(p_node, p_graph, r_uniforms);
	}

	// Math operation nodes.
	if (node_type == "multiply" || node_type == "add" || node_type == "subtract" ||
			node_type == "divide" || node_type == "mix" || node_type == "dot" ||
			node_type == "normalize" || node_type == "power" || node_type == "clamp" ||
			node_type == "min" || node_type == "max" || node_type == "abs" ||
			node_type == "floor" || node_type == "ceil" || node_type == "sqrt" ||
			node_type == "sin" || node_type == "cos" || node_type == "invert") {
		return _compile_math_node(p_node, p_graph, r_uniforms, r_functions);
	}

	// Geometry nodes.
	if (node_type == "position" || node_type == "normal" || node_type == "texcoord" ||
			node_type == "tangent" || node_type == "bitangent" || node_type == "geomcolor") {
		return _compile_geometry_node(p_node);
	}

	// Color operation nodes.
	if (node_type == "color_correct" || node_type == "luminance" ||
			node_type == "hsvadjust" || node_type == "contrast" ||
			node_type == "saturate" || node_type == "rgbtohsv" || node_type == "hsvtorgb") {
		return _compile_color_node(p_node, p_graph, r_uniforms, r_functions);
	}

	// Noise / procedural nodes.
	if (node_type == "fractal3d" || node_type == "noise3d" || node_type == "noise2d" ||
			node_type == "cellnoise3d" || node_type == "cellnoise2d" ||
			node_type == "worleynoise2d" || node_type == "worleynoise3d") {
		return _compile_noise_node(p_node, p_graph, r_uniforms, r_functions);
	}

	// Constant node.
	if (node_type == "constant") {
		if (p_node.inputs.has("value")) {
			String type = p_node.output_type;
			String val = p_node.inputs["value"];
			if (type == "float") {
				return val;
			}
			if (type == "color3" || type == "vector3") {
				String cleaned = val.replace(" ", "");
				PackedStringArray parts = cleaned.split(",");
				if (parts.size() >= 3) {
					return "vec3(" + parts[0] + ", " + parts[1] + ", " + parts[2] + ")";
				}
				return "vec3(" + val + ")";
			}
			return val;
		}
		return "0.0";
	}

	// Swizzle / extract.
	if (node_type == "swizzle" || node_type == "extract") {
		String in_expr = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		if (p_node.inputs.has("channels")) {
			return in_expr + "." + String(p_node.inputs["channels"]);
		}
		if (p_node.inputs.has("index")) {
			return in_expr + "[" + String(p_node.inputs["index"]) + "]";
		}
		return in_expr;
	}

	// Combine nodes.
	if (node_type == "combine2" || node_type == "combine3" || node_type == "combine4") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		if (node_type == "combine2") {
			return "vec2(" + in1 + ", " + in2 + ")";
		}
		String in3 = _resolve_input_expression("in3", p_node, p_graph, r_uniforms, r_functions);
		if (node_type == "combine3") {
			return "vec3(" + in1 + ", " + in2 + ", " + in3 + ")";
		}
		String in4 = _resolve_input_expression("in4", p_node, p_graph, r_uniforms, r_functions);
		return "vec4(" + in1 + ", " + in2 + ", " + in3 + ", " + in4 + ")";
	}

	// Fallback: unknown node type.
	WARN_PRINT(vformat("USDMaterialXConverter: Unsupported MaterialX node type: %s", node_type));
	if (p_node.output_type == "color3" || p_node.output_type == "vector3") {
		return "vec3(0.0)";
	}
	return "0.0";
}

// ============================================================================
// standard_surface compilation
// ============================================================================

String USDMaterialXConverter::_compile_standard_surface(
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms,
		String &r_functions) const {
	String body;

	// Base color.
	String base_color = _resolve_input_expression("base_color", p_node, p_graph, r_uniforms, r_functions);
	String base_weight = _resolve_input_expression("base", p_node, p_graph, r_uniforms, r_functions);
	body += "\t// Standard Surface: base color\n";
	body += "\tALBEDO = " + base_color + " * " + base_weight + ";\n";

	// Metallic (metalness in MaterialX).
	String metalness = _resolve_input_expression("metalness", p_node, p_graph, r_uniforms, r_functions);
	body += "\tMETALLIC = " + metalness + ";\n";

	// Roughness (specular_roughness in MaterialX).
	String spec_roughness = _resolve_input_expression("specular_roughness", p_node, p_graph, r_uniforms, r_functions);
	body += "\tROUGHNESS = " + spec_roughness + ";\n";

	// Specular weight.
	String specular = _resolve_input_expression("specular", p_node, p_graph, r_uniforms, r_functions);
	body += "\tSPECULAR = " + specular + ";\n";

	// Emission.
	String emission_color = _resolve_input_expression("emission_color", p_node, p_graph, r_uniforms, r_functions);
	String emission_weight = _resolve_input_expression("emission", p_node, p_graph, r_uniforms, r_functions);
	body += "\tEMISSION = " + emission_color + " * " + emission_weight + ";\n";

	// Normal map.
	if (p_node.input_connections.has("normal")) {
		String normal_val = _resolve_input_expression("normal", p_node, p_graph, r_uniforms, r_functions);
		body += "\tNORMAL_MAP = " + normal_val + ";\n";
	}

	// Opacity.
	String opacity_val = _resolve_input_expression("opacity", p_node, p_graph, r_uniforms, r_functions);
	if (p_node.input_types.has("opacity")) {
		String opacity_type = p_node.input_types["opacity"];
		if (opacity_type == "color3" || opacity_type == "vector3") {
			body += "\tALPHA = (" + opacity_val + ").r;\n";
		} else {
			body += "\tALPHA = " + opacity_val + ";\n";
		}
	} else {
		body += "\tALPHA = 1.0;\n";
	}

	// Clearcoat (coat in MaterialX).
	if (p_node.inputs.has("coat") || p_node.input_connections.has("coat")) {
		String coat = _resolve_input_expression("coat", p_node, p_graph, r_uniforms, r_functions);
		String coat_roughness = _resolve_input_expression("coat_roughness", p_node, p_graph, r_uniforms, r_functions);
		body += "\tCLEARCOAT = " + coat + ";\n";
		body += "\tCLEARCOAT_ROUGHNESS = " + coat_roughness + ";\n";
	}

	return body;
}

// ============================================================================
// Image node compilation
// ============================================================================

String USDMaterialXConverter::_compile_image_node(
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms) const {
	String uniform_name = "tex_" + p_node.name;

	// Determine if this is a color (sRGB) or data (linear) texture.
	bool is_color = (p_node.output_type == "color3" || p_node.output_type == "color4");
	String hint = is_color
			? " : source_color, filter_linear_mipmap, repeat_enable"
			: " : filter_linear_mipmap, repeat_enable";

	r_uniforms += "uniform sampler2D " + uniform_name + hint + ";\n";

	// UV coordinates.
	String uv_expr = "UV";
	if (p_node.input_connections.has("texcoord")) {
		String connection = p_node.input_connections["texcoord"];
		String node_name = connection;
		int dot_pos = connection.find(".");
		if (dot_pos >= 0) {
			node_name = connection.substr(0, dot_pos);
		}
		const MtlxNodeInfo *uv_node = _find_node(node_name, p_graph);
		if (uv_node) {
			uv_expr = _compile_geometry_node(*uv_node);
		}
	}

	// Return texture sample expression.
	String sample = "texture(" + uniform_name + ", " + uv_expr + ")";

	if (p_node.output_type == "color3" || p_node.output_type == "vector3") {
		return sample + ".rgb";
	} else if (p_node.output_type == "float") {
		return sample + ".r";
	}
	return sample;
}

// ============================================================================
// Math node compilation
// ============================================================================

String USDMaterialXConverter::_compile_math_node(
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms,
		String &r_functions) const {
	String node_type = p_node.node_type;

	if (node_type == "multiply") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "(" + in1 + " * " + in2 + ")";
	}
	if (node_type == "add") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "(" + in1 + " + " + in2 + ")";
	}
	if (node_type == "subtract") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "(" + in1 + " - " + in2 + ")";
	}
	if (node_type == "divide") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "(" + in1 + " / " + in2 + ")";
	}
	if (node_type == "mix") {
		String fg = _resolve_input_expression("fg", p_node, p_graph, r_uniforms, r_functions);
		String bg = _resolve_input_expression("bg", p_node, p_graph, r_uniforms, r_functions);
		String mix_val = _resolve_input_expression("mix", p_node, p_graph, r_uniforms, r_functions);
		return "mix(" + bg + ", " + fg + ", " + mix_val + ")";
	}
	if (node_type == "dot") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "dot(" + in1 + ", " + in2 + ")";
	}
	if (node_type == "normalize") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "normalize(" + in_val + ")";
	}
	if (node_type == "power") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "pow(" + in1 + ", " + in2 + ")";
	}
	if (node_type == "clamp") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		String low = _resolve_input_expression("low", p_node, p_graph, r_uniforms, r_functions);
		String high = _resolve_input_expression("high", p_node, p_graph, r_uniforms, r_functions);
		return "clamp(" + in_val + ", " + low + ", " + high + ")";
	}
	if (node_type == "min") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "min(" + in1 + ", " + in2 + ")";
	}
	if (node_type == "max") {
		String in1 = _resolve_input_expression("in1", p_node, p_graph, r_uniforms, r_functions);
		String in2 = _resolve_input_expression("in2", p_node, p_graph, r_uniforms, r_functions);
		return "max(" + in1 + ", " + in2 + ")";
	}
	if (node_type == "abs") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "abs(" + in_val + ")";
	}
	if (node_type == "floor") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "floor(" + in_val + ")";
	}
	if (node_type == "ceil") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "ceil(" + in_val + ")";
	}
	if (node_type == "sqrt") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "sqrt(" + in_val + ")";
	}
	if (node_type == "sin") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "sin(" + in_val + ")";
	}
	if (node_type == "cos") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		return "cos(" + in_val + ")";
	}
	if (node_type == "invert") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		if (p_node.output_type == "color3" || p_node.output_type == "vector3") {
			return "(vec3(1.0) - " + in_val + ")";
		}
		return "(1.0 - " + in_val + ")";
	}

	return "0.0";
}

// ============================================================================
// Geometry node compilation
// ============================================================================

String USDMaterialXConverter::_compile_geometry_node(
		const MtlxNodeInfo &p_node) const {
	String node_type = p_node.node_type;

	if (node_type == "texcoord") {
		String index = "0";
		if (p_node.inputs.has("index")) {
			index = p_node.inputs["index"];
		}
		if (index == "0" || index.is_empty()) {
			return "UV";
		} else if (index == "1") {
			return "UV2";
		}
		return "UV";
	}
	if (node_type == "position") {
		String space = "object";
		if (p_node.inputs.has("space")) {
			space = p_node.inputs["space"];
		}
		if (space == "world") {
			return "(MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz";
		}
		return "VERTEX";
	}
	if (node_type == "normal") {
		String space = "object";
		if (p_node.inputs.has("space")) {
			space = p_node.inputs["space"];
		}
		if (space == "world") {
			return "mat3(MODEL_MATRIX) * NORMAL";
		}
		return "NORMAL";
	}
	if (node_type == "tangent") {
		return "TANGENT";
	}
	if (node_type == "bitangent") {
		return "BINORMAL";
	}
	if (node_type == "geomcolor") {
		return "COLOR.rgb";
	}

	return "vec3(0.0)";
}

// ============================================================================
// Color node compilation
// ============================================================================

String USDMaterialXConverter::_compile_color_node(
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms,
		String &r_functions) const {
	String node_type = p_node.node_type;

	if (node_type == "luminance") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		// ITU-R BT.709 luminance coefficients.
		return "dot(" + in_val + ", vec3(0.2126, 0.7152, 0.0722))";
	}

	if (node_type == "color_correct") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);

		String result = in_val;

		if (p_node.inputs.has("gain") || p_node.input_connections.has("gain")) {
			String gain = _resolve_input_expression("gain", p_node, p_graph, r_uniforms, r_functions);
			result = "(" + result + " * " + gain + ")";
		}
		if (p_node.inputs.has("offset") || p_node.input_connections.has("offset")) {
			String offset = _resolve_input_expression("offset", p_node, p_graph, r_uniforms, r_functions);
			result = "(" + result + " + " + offset + ")";
		}
		if (p_node.inputs.has("gamma") || p_node.input_connections.has("gamma")) {
			String gamma = _resolve_input_expression("gamma", p_node, p_graph, r_uniforms, r_functions);
			result = "pow(" + result + ", vec3(1.0 / " + gamma + "))";
		}

		return result;
	}

	if (node_type == "saturate") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		String amount = _resolve_input_expression("amount", p_node, p_graph, r_uniforms, r_functions);
		return "mix(vec3(dot(" + in_val + ", vec3(0.2126, 0.7152, 0.0722))), " + in_val + ", " + amount + ")";
	}

	if (node_type == "contrast") {
		String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
		String amount = _resolve_input_expression("amount", p_node, p_graph, r_uniforms, r_functions);
		String pivot = _resolve_input_expression("pivot", p_node, p_graph, r_uniforms, r_functions);
		return "((" + in_val + " - " + pivot + ") * " + amount + " + " + pivot + ")";
	}

	// Fallback.
	String in_val = _resolve_input_expression("in", p_node, p_graph, r_uniforms, r_functions);
	return in_val;
}

// ============================================================================
// Noise node compilation
// ============================================================================

String USDMaterialXConverter::_compile_noise_node(
		const MtlxNodeInfo &p_node,
		const MtlxNodeGraph &p_graph,
		String &r_uniforms,
		String &r_functions) const {
	String node_type = p_node.node_type;

	// Add noise helper functions if not already added.
	if (!r_functions.contains("mtlx_hash")) {
		r_functions += "\n// MaterialX noise helper functions\n";
		r_functions += "float mtlx_hash(vec3 p) {\n";
		r_functions += "\tp = fract(p * 0.3183099 + 0.1);\n";
		r_functions += "\tp *= 17.0;\n";
		r_functions += "\treturn fract(p.x * p.y * p.z * (p.x + p.y + p.z));\n";
		r_functions += "}\n\n";

		r_functions += "float mtlx_noise3d(vec3 p) {\n";
		r_functions += "\tvec3 i = floor(p);\n";
		r_functions += "\tvec3 f = fract(p);\n";
		r_functions += "\tf = f * f * (3.0 - 2.0 * f);\n";
		r_functions += "\treturn mix(\n";
		r_functions += "\t\tmix(mix(mtlx_hash(i + vec3(0, 0, 0)), mtlx_hash(i + vec3(1, 0, 0)), f.x),\n";
		r_functions += "\t\t    mix(mtlx_hash(i + vec3(0, 1, 0)), mtlx_hash(i + vec3(1, 1, 0)), f.x), f.y),\n";
		r_functions += "\t\tmix(mix(mtlx_hash(i + vec3(0, 0, 1)), mtlx_hash(i + vec3(1, 0, 1)), f.x),\n";
		r_functions += "\t\t    mix(mtlx_hash(i + vec3(0, 1, 1)), mtlx_hash(i + vec3(1, 1, 1)), f.x), f.y),\n";
		r_functions += "\t\tf.z);\n";
		r_functions += "}\n\n";

		r_functions += "float mtlx_fractal3d(vec3 p, int octaves, float lacunarity, float diminish) {\n";
		r_functions += "\tfloat result = 0.0;\n";
		r_functions += "\tfloat amplitude = 1.0;\n";
		r_functions += "\tfor (int i = 0; i < octaves; i++) {\n";
		r_functions += "\t\tresult += amplitude * mtlx_noise3d(p);\n";
		r_functions += "\t\tamplitude *= diminish;\n";
		r_functions += "\t\tp *= lacunarity;\n";
		r_functions += "\t}\n";
		r_functions += "\treturn result;\n";
		r_functions += "}\n\n";

		r_functions += "float mtlx_cellnoise3d(vec3 p) {\n";
		r_functions += "\treturn mtlx_hash(floor(p));\n";
		r_functions += "}\n\n";
	}

	// Get position input.
	String pos_expr = "VERTEX";
	if (p_node.input_connections.has("position")) {
		pos_expr = _resolve_input_expression("position", p_node, p_graph, r_uniforms, r_functions);
	} else if (p_node.input_connections.has("texcoord")) {
		String tc = _resolve_input_expression("texcoord", p_node, p_graph, r_uniforms, r_functions);
		pos_expr = "vec3(" + tc + ", 0.0)";
	}

	// Get amplitude.
	String amplitude = "1.0";
	if (p_node.inputs.has("amplitude") || p_node.input_connections.has("amplitude")) {
		amplitude = _resolve_input_expression("amplitude", p_node, p_graph, r_uniforms, r_functions);
	}

	if (node_type == "noise3d") {
		return "(mtlx_noise3d(" + pos_expr + ") * " + amplitude + ")";
	}
	if (node_type == "noise2d") {
		return "(mtlx_noise3d(vec3(" + pos_expr + ".xy, 0.0)) * " + amplitude + ")";
	}
	if (node_type == "fractal3d") {
		String octaves = "3";
		String lacunarity = "2.0";
		String diminish = "0.5";
		if (p_node.inputs.has("octaves")) {
			octaves = p_node.inputs["octaves"];
		}
		if (p_node.inputs.has("lacunarity")) {
			lacunarity = p_node.inputs["lacunarity"];
		}
		if (p_node.inputs.has("diminish")) {
			diminish = p_node.inputs["diminish"];
		}
		return "(mtlx_fractal3d(" + pos_expr + ", " + octaves + ", " + lacunarity + ", " + diminish + ") * " + amplitude + ")";
	}
	if (node_type == "cellnoise3d") {
		return "mtlx_cellnoise3d(" + pos_expr + ")";
	}
	if (node_type == "cellnoise2d") {
		return "mtlx_cellnoise3d(vec3(" + pos_expr + ".xy, 0.0))";
	}

	// Worley noise -- approximation.
	if (node_type == "worleynoise2d" || node_type == "worleynoise3d") {
		return "(1.0 - mtlx_noise3d(" + pos_expr + "))";
	}

	return "0.0";
}

// ============================================================================
// Graph-to-shader compilation (top-level assembler)
// ============================================================================

String USDMaterialXConverter::_compile_graph_to_shader(const MtlxNodeGraph &p_graph) const {
	if (p_graph.output_node.is_empty() && p_graph.nodes.is_empty()) {
		return String();
	}

	String uniforms;
	String functions;

	// Find the output node.
	const MtlxNodeInfo *output_node = nullptr;
	if (!p_graph.output_node.is_empty()) {
		output_node = _find_node(p_graph.output_node, p_graph);
	}

	// If no explicit output node, use the first standard_surface node.
	if (!output_node) {
		for (int i = 0; i < p_graph.nodes.size(); i++) {
			if (p_graph.nodes[i].node_type == "standard_surface") {
				output_node = &p_graph.nodes[i];
				break;
			}
		}
	}

	// If still no output, use the last node.
	if (!output_node && !p_graph.nodes.is_empty()) {
		output_node = &p_graph.nodes[p_graph.nodes.size() - 1];
	}

	if (!output_node) {
		WARN_PRINT("USDMaterialXConverter: No output node found in MaterialX graph.");
		return String();
	}

	// Compile the output node.
	String fragment_body;
	if (output_node->node_type == "standard_surface") {
		fragment_body = _compile_standard_surface(*output_node, p_graph, uniforms, functions);
	} else {
		// Generic output: assign to ALBEDO.
		String expr = _compile_node_expression(*output_node, p_graph, uniforms, functions);
		if (output_node->output_type == "color3" || output_node->output_type == "vector3") {
			fragment_body = "\tALBEDO = " + expr + ";\n";
		} else if (output_node->output_type == "float") {
			fragment_body = "\tALBEDO = vec3(" + expr + ");\n";
		} else {
			fragment_body = "\tALBEDO = " + expr + ".rgb;\n";
		}
	}

	// Assemble the complete shader.
	String code;
	code += "shader_type spatial;\n";
	code += "render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;\n";
	code += "\n";

	// Uniforms.
	if (!uniforms.is_empty()) {
		code += "// MaterialX uniforms\n";
		code += uniforms;
		code += "\n";
	}

	// Helper functions.
	if (!functions.is_empty()) {
		code += functions;
		code += "\n";
	}

	// Fragment function.
	code += "void fragment() {\n";
	code += fragment_body;
	code += "}\n";

	return code;
}
