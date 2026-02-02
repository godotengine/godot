/**************************************************************************/
/*  usd_materialx_converter.h                                             */
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

#include "core/io/resource.h"
#include "core/io/xml_parser.h"
#include "core/string/ustring.h"
#include "core/variant/dictionary.h"
#include "scene/resources/material.h"
#include "scene/resources/shader.h"

// Internal representation of a single MaterialX node within a node graph.
// Captures the node definition type, named inputs (constant values or
// connections to other nodes), and the output data type.
struct MtlxNodeInfo {
	String name;
	String node_type; // e.g. "standard_surface", "image", "multiply"
	String output_type; // e.g. "color3", "float", "surfaceshader"
	Dictionary inputs; // input name -> constant value string
	Dictionary input_types; // input name -> type string
	Dictionary input_connections; // input name -> "nodename" or "nodename.output"
};

// Internal representation of a complete MaterialX node graph.
// Holds all parsed nodes, the designated output node, and a map of
// texture file paths discovered during XML parsing.
struct MtlxNodeGraph {
	String name;
	Vector<MtlxNodeInfo> nodes;
	String output_node; // Name of the surface output node.
	String output_type; // Surface output type.
	Dictionary texture_paths; // node name -> texture file path
};

// Converts MaterialX XML documents to Godot ShaderMaterial instances.
//
// This class implements a self-contained MaterialX-to-GLSL compiler that
// uses Godot's built-in XMLParser (no external MaterialX library required).
// It parses the MaterialX XML into an intermediate node graph representation,
// then compiles that graph to Godot spatial shader GLSL code.
//
// Supported MaterialX node types:
//   - standard_surface (Autodesk Standard Surface PBR)
//   - image / tiledimage (texture sampling)
//   - Math: multiply, add, subtract, divide, power, clamp, mix, min, max,
//           abs, floor, ceil, sqrt, sin, cos, invert, dot, normalize
//   - Geometry: texcoord, position, normal, tangent, bitangent, geomcolor
//   - Color: luminance, color_correct, saturate, contrast
//   - Procedural: noise3d, noise2d, fractal3d, cellnoise3d, cellnoise2d,
//                 worleynoise2d, worleynoise3d
//   - Utility: constant, swizzle, extract, combine2/3/4
class USDMaterialXConverter : public RefCounted {
	GDCLASS(USDMaterialXConverter, RefCounted);

protected:
	static void _bind_methods();

private:
	// Shader cache: hash of GLSL code -> compiled ShaderMaterial.
	Dictionary shader_cache;
	static const int MAX_SHADER_CACHE_SIZE = 256;

	// -- XML Parsing ----------------------------------------------------------

	MtlxNodeGraph _parse_materialx_xml(const String &p_xml) const;

	// -- Graph-to-GLSL Compilation --------------------------------------------

	String _compile_graph_to_shader(const MtlxNodeGraph &p_graph) const;

	String _compile_standard_surface(
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms,
			String &r_functions) const;

	String _resolve_input_expression(
			const String &p_input_name,
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms,
			String &r_functions) const;

	String _compile_node_expression(
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms,
			String &r_functions) const;

	// -- Specialized node compilers -------------------------------------------

	String _compile_image_node(
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms) const;

	String _compile_math_node(
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms,
			String &r_functions) const;

	String _compile_geometry_node(
			const MtlxNodeInfo &p_node) const;

	String _compile_color_node(
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms,
			String &r_functions) const;

	String _compile_noise_node(
			const MtlxNodeInfo &p_node,
			const MtlxNodeGraph &p_graph,
			String &r_uniforms,
			String &r_functions) const;

	// -- Helpers --------------------------------------------------------------

	const MtlxNodeInfo *_find_node(
			const String &p_name,
			const MtlxNodeGraph &p_graph) const;

	String _compute_cache_key(const String &p_shader_code) const;
	void _evict_shader_cache_if_needed();

public:
	// Convert a MaterialX XML document string into a ShaderMaterial.
	// Textures referenced in <image> nodes are loaded relative to p_base_path.
	Ref<ShaderMaterial> convert_materialx_from_xml(const String &p_xml, const String &p_base_path);

	// Compile a MaterialX XML document to Godot GLSL shader code (for inspection).
	String compile_materialx_to_shader(const String &p_xml) const;

	// Parse a MaterialX XML document and return the node graph as a Dictionary
	// (useful for debugging from GDScript).
	Dictionary parse_materialx_document(const String &p_xml) const;

	// Clear the compiled shader cache.
	void clear_cache();

	// Return the number of cached shader entries.
	int get_cache_size() const;
};
