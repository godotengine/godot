/*************************************************************************/
/*  shader_graph.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SHADER_GRAPH_H
#define SHADER_GRAPH_H

#if 0

#include "map.h"
#include "scene/resources/shader.h"

class ShaderGraph : public Resource {

	OBJ_TYPE( ShaderGraph, Resource );
	RES_BASE_EXTENSION("sgp");

public:

	enum NodeType {
		NODE_IN, ///< param 0: name
		NODE_OUT, ///< param 0: name
		NODE_CONSTANT, ///< param 0: value
		NODE_PARAMETER, ///< param 0: name
		NODE_ADD,
		NODE_SUB,
		NODE_MUL,
		NODE_DIV,
		NODE_MOD,
		NODE_SIN,
		NODE_COS,
		NODE_TAN,
		NODE_ARCSIN,
		NODE_ARCCOS,
		NODE_ARCTAN,
		NODE_POW,
		NODE_LOG,
		NODE_MAX,
		NODE_MIN,
		NODE_COMPARE,
		NODE_TEXTURE, ///< param  0: texture
		NODE_TIME, ///< param  0: interval length
		NODE_NOISE,
		NODE_PASS,
		NODE_VEC_IN, ///< param 0: name
		NODE_VEC_OUT, ///< param 0: name
		NODE_VEC_CONSTANT, ///< param  0: value
		NODE_VEC_PARAMETER, ///< param  0: name
		NODE_VEC_ADD,
		NODE_VEC_SUB,
		NODE_VEC_MUL,
		NODE_VEC_DIV,
		NODE_VEC_MOD,
		NODE_VEC_CROSS,
		NODE_VEC_DOT,
		NODE_VEC_POW,
		NODE_VEC_NORMALIZE,
		NODE_VEC_INTERPOLATE,
		NODE_VEC_SCREEN_TO_UV,
		NODE_VEC_TRANSFORM3,
		NODE_VEC_TRANSFORM4,
		NODE_VEC_COMPARE,
		NODE_VEC_TEXTURE_2D,
		NODE_VEC_TEXTURE_CUBE,
		NODE_VEC_NOISE,
		NODE_VEC_0,
		NODE_VEC_1,
		NODE_VEC_2,
		NODE_VEC_BUILD,
		NODE_VEC_PASS,
		NODE_COLOR_CONSTANT,
		NODE_COLOR_PARAMETER,
		NODE_TEXTURE_PARAMETER,
		NODE_TEXTURE_2D_PARAMETER,
		NODE_TEXTURE_CUBE_PARAMETER,
		NODE_TRANSFORM_CONSTANT,
		NODE_TRANSFORM_PARAMETER,
		NODE_LABEL,
		NODE_TYPE_MAX
	};

	enum ShaderType {
		SHADER_VERTEX,
		SHADER_FRAGMENT,
		SHADER_LIGHT
	};

private:

	struct Connection {

		int src_id;
		int src_slot;
		int dst_id;
		int dst_slot;
	};

	struct Node {

		int16_t x,y;
		NodeType type;
		Variant param;
		int id;
		mutable int order; // used for sorting
		mutable bool out_valid;
		mutable bool in_valid;
	};

	struct ShaderData {
		Map<int,Node> node_map;
		List<Connection> connections;
	} shader[3];
	uint64_t version;

protected:

/*	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;*/

	static void _bind_methods();

	Array _get_connections_helper() const;

public:


	void node_add(ShaderType p_which, NodeType p_type,int p_id);
	void node_remove(ShaderType p_which,int p_id);
	void node_set_param(ShaderType p_which, int p_id, const Variant& p_value);
	void node_set_pos(ShaderType p_which,int p_id,const Point2& p_pos);
	void node_change_type(ShaderType p_which,int p_id, NodeType p_type);
	Point2 node_get_pos(ShaderType p_which,int p_id) const;

	void get_node_list(ShaderType p_which,List<int> *p_node_list) const;
	NodeType node_get_type(ShaderType p_which,int p_id) const;
	Variant node_get_param(ShaderType p_which,int p_id) const;

	Error connect(ShaderType p_which,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);
	bool is_connected(ShaderType p_which,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) const;
	void disconnect(ShaderType p_which,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);

	void get_connections(ShaderType p_which,List<Connection> *p_connections) const;

	void clear();

	uint64_t get_version() const { return version; }

	static void get_default_input_nodes(Mode p_type,List<PropertyInfo> *p_inputs);
	static void get_default_output_nodes(Mode p_type,List<PropertyInfo> *p_outputs);

	static PropertyInfo node_get_type_info(NodeType p_type);
	static int get_input_count(NodeType p_type);
	static int get_output_count(NodeType p_type);
	static String get_input_name(NodeType p_type,int p_input);
	static String get_output_name(NodeType p_type,int p_output);
	static bool is_input_vector(NodeType p_type,int p_input);
	static bool is_output_vector(NodeType p_type,int p_input);


	ShaderGraph();
	~ShaderGraph();
};

//helper functions




VARIANT_ENUM_CAST( ShaderGraph::NodeType );

#endif
#endif // SHADER_GRAPH_H
