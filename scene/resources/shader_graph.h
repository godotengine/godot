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


#include "map.h"

#if 0

class Shader : public Resource {

	OBJ_TYPE( Shader, Resource );
	RES_BASE_EXTENSION("sgp");
	RID shader;
	Map<int,Point2> positions;
	uint64_t version;

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	static void _bind_methods();

	Array _get_connections_helper() const;

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

	void node_add(NodeType p_type,int p_id);
	void node_remove(int p_id);
	void node_set_param( int p_id, const Variant& p_value);
	void node_set_pos(int p_id,const Point2& p_pos);
	Point2 node_get_pos(int p_id) const;

	void get_node_list(List<int> *p_node_list) const;
	NodeType node_get_type(int p_id) const;
	Variant node_get_param(int p_id) const;

	void connect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);
	void disconnect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);

	struct Connection {

		int src_id;
		int src_slot;
		int dst_id;
		int dst_slot;
	};

	void get_connections(List<Connection> *p_connections) const;

	void clear();

	virtual RID get_rid() const { return shader; }

	uint64_t get_version() const { return version; }

	Shader();
	~Shader();
};

enum ShaderType {
	SHADER_VERTEX,
	SHADER_FRAGMENT,
	SHADER_POST_PROCESS
};
//helper functions

static void shader_get_default_input_nodes(ShaderType p_type,List<PropertyInfo> *p_inputs);
static void shader_get_default_output_nodes(ShaderType p_type,List<PropertyInfo> *p_outputs);

static PropertyInfo shader_node_get_type_info(ShaderNodeType p_type);
static int shader_get_input_count(ShaderNodeType p_type);
static int shader_get_output_count(ShaderNodeType p_type);
static String shader_get_input_name(ShaderNodeType p_type,int p_input);
static String shader_get_output_name(ShaderNodeType p_type,int p_output);
static bool shader_is_input_vector(ShaderNodeType p_type,int p_input);
static bool shader_is_output_vector(ShaderNodeType p_type,int p_input);


VARIANT_ENUM_CAST( Shader::NodeType );


#endif

#endif // SHADER_GRAPH_H
