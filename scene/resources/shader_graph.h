/*************************************************************************/
/*  shader_graph.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

class ShaderGraph : public Shader {

	GDCLASS( ShaderGraph, Shader );
	RES_BASE_EXTENSION("sgp");

public:

	enum NodeType {
		NODE_INPUT, // all inputs (shader type dependent)
		NODE_SCALAR_CONST, //scalar constant
		NODE_VEC_CONST, //vec3 constant
		NODE_RGB_CONST, //rgb constant (shows a color picker instead)
		NODE_XFORM_CONST, // 4x4 matrix constant
		NODE_TIME, // time in seconds
		NODE_SCREEN_TEX, // screen texture sampler (takes UV) (only usable in fragment shader)
		NODE_SCALAR_OP, // scalar vs scalar op (mul, add, div, etc)
		NODE_VEC_OP, // vec3 vs vec3 op (mul,ad,div,crossprod,etc)
		NODE_VEC_SCALAR_OP, // vec3 vs scalar op (mul, add, div, etc)
		NODE_RGB_OP, // vec3 vs vec3 rgb op (with scalar amount), like brighten, darken, burn, dodge, multiply, etc.
		NODE_XFORM_MULT, // mat4 x mat4
		NODE_XFORM_VEC_MULT, // mat4 x vec3 mult (with no-translation option)
		NODE_XFORM_VEC_INV_MULT, // mat4 x vec3 inverse mult (with no-translation option)
		NODE_SCALAR_FUNC, // scalar function (sin, cos, etc)
		NODE_VEC_FUNC, // vector function (normalize, negate, reciprocal, rgb2hsv, hsv2rgb, etc, etc)
		NODE_VEC_LEN, // vec3 length
		NODE_DOT_PROD, // vec3 . vec3 (dot product -> scalar output)
		NODE_VEC_TO_SCALAR, // 1 vec3 input, 3 scalar outputs
		NODE_SCALAR_TO_VEC, // 3 scalar input, 1 vec3 output
		NODE_XFORM_TO_VEC, // 3 vec input, 1 xform output
		NODE_VEC_TO_XFORM, // 3 vec input, 1 xform output
		NODE_SCALAR_INTERP, // scalar interpolation (with optional curve)
		NODE_VEC_INTERP, // vec3 interpolation  (with optional curve)
		NODE_COLOR_RAMP, //take scalar, output vec3
		NODE_CURVE_MAP, //take scalar, otput scalar
		NODE_SCALAR_INPUT, // scalar uniform (assignable in material)
		NODE_VEC_INPUT, // vec3 uniform (assignable in material)
		NODE_RGB_INPUT, // color uniform (assignable in material)
		NODE_XFORM_INPUT, // mat4 uniform (assignable in material)
		NODE_TEXTURE_INPUT, // texture input (assignable in material)
		NODE_CUBEMAP_INPUT, // cubemap input (assignable in material)
		NODE_DEFAULT_TEXTURE,
		NODE_OUTPUT, // output (shader type dependent)
		NODE_COMMENT, // comment
		NODE_TYPE_MAX
	};


	struct Connection {

		int src_id;
		int src_slot;
		int dst_id;
		int dst_slot;
	};

	enum SlotType {

		SLOT_TYPE_SCALAR,
		SLOT_TYPE_VEC,
		SLOT_TYPE_XFORM,
		SLOT_TYPE_TEXTURE,
		SLOT_MAX
	};

	enum ShaderType {
		SHADER_TYPE_VERTEX,
		SHADER_TYPE_FRAGMENT,
		SHADER_TYPE_LIGHT,
		SHADER_TYPE_MAX
	};

	enum SlotDir {
		SLOT_IN,
		SLOT_OUT
	};

	enum GraphError {
		GRAPH_OK,
		GRAPH_ERROR_CYCLIC,
		GRAPH_ERROR_MISSING_CONNECTIONS
	};

private:

	String _find_unique_name(const String& p_base);

	enum {SLOT_DEFAULT_VALUE = 0x7FFFFFFF};
	struct SourceSlot {

		int id;
		int slot;
		bool operator==(const SourceSlot& p_slot) const {
			return id==p_slot.id && slot==p_slot.slot;
		}
	};

	struct Node {

		Vector2 pos;
		NodeType type;
		Variant param1;
		Variant param2;
		Map<int, Variant> defaults;
		int id;
		mutable int order; // used for sorting
		int sort_order;
		Map<int,SourceSlot> connections;

	};

	struct ShaderData {
		Map<int,Node> node_map;
		GraphError error;
	} shader[3];



	struct InOutParamInfo {
		Mode shader_mode;
		ShaderType shader_type;
		const char *name;
		const char *variable;
		const char *postfix;
		SlotType slot_type;
		SlotDir dir;
	};

	static const InOutParamInfo inout_param_info[];

	struct NodeSlotInfo {

		enum { MAX_INS=3, MAX_OUTS=3 };
		NodeType type;
		const SlotType ins[MAX_INS];
		const SlotType outs[MAX_OUTS];
	};

	static const NodeSlotInfo node_slot_info[];

	bool _pending_update_shader;
	void _update_shader();
	void _request_update();

	void _plot_curve(const Vector2& p_a,const Vector2& p_b,const Vector2& p_c,const Vector2& p_d,uint8_t* p_heights,bool *p_useds);
	void _add_node_code(ShaderType p_type,Node *p_node,const Vector<String>& p_inputs,String& code);

	Array _get_node_list(ShaderType p_type) const;
	Array _get_connections(ShaderType p_type) const;

	void _set_data(const Dictionary& p_data);
	Dictionary _get_data() const;
protected:

	static void _bind_methods();

public:


	void node_add(ShaderType p_type, NodeType p_node_type, int p_id);
	void node_remove(ShaderType p_which,int p_id);
	void node_set_pos(ShaderType p_which,int p_id,const Point2& p_pos);
	Point2 node_get_pos(ShaderType p_which,int p_id) const;

	void get_node_list(ShaderType p_which,List<int> *p_node_list) const;
	NodeType node_get_type(ShaderType p_which,int p_id) const;

	void scalar_const_node_set_value(ShaderType p_which,int p_id,float p_value);
	float scalar_const_node_get_value(ShaderType p_which,int p_id) const;

	void vec_const_node_set_value(ShaderType p_which,int p_id,const Vector3& p_value);
	Vector3 vec_const_node_get_value(ShaderType p_which,int p_id) const;

	void rgb_const_node_set_value(ShaderType p_which,int p_id,const Color& p_value);
	Color rgb_const_node_get_value(ShaderType p_which,int p_id) const;

	void xform_const_node_set_value(ShaderType p_which,int p_id,const Transform& p_value);
	Transform xform_const_node_get_value(ShaderType p_which,int p_id) const;

	void texture_node_set_filter_size(ShaderType p_which,int p_id,int p_size);
	int texture_node_get_filter_size(ShaderType p_which,int p_id) const;

	void texture_node_set_filter_strength(ShaderType p_which,float p_id,float p_strength);
	float texture_node_get_filter_strength(ShaderType p_which,float p_id) const;

	void duplicate_nodes(ShaderType p_which, List<int> &p_nodes);

	List<int> generate_ids(ShaderType p_type, int count);

	enum ScalarOp {
		SCALAR_OP_ADD,
		SCALAR_OP_SUB,
		SCALAR_OP_MUL,
		SCALAR_OP_DIV,
		SCALAR_OP_MOD,
		SCALAR_OP_POW,
		SCALAR_OP_MAX,
		SCALAR_OP_MIN,
		SCALAR_OP_ATAN2,
		SCALAR_MAX_OP
	};

	void scalar_op_node_set_op(ShaderType p_which,float p_id,ScalarOp p_op);
	ScalarOp scalar_op_node_get_op(ShaderType p_which,float p_id) const;

	enum  VecOp {
		VEC_OP_ADD,
		VEC_OP_SUB,
		VEC_OP_MUL,
		VEC_OP_DIV,
		VEC_OP_MOD,
		VEC_OP_POW,
		VEC_OP_MAX,
		VEC_OP_MIN,
		VEC_OP_CROSS,
		VEC_MAX_OP
	};

	void vec_op_node_set_op(ShaderType p_which,float p_id,VecOp p_op);
	VecOp vec_op_node_get_op(ShaderType p_which,float p_id) const;

	enum VecScalarOp {
		VEC_SCALAR_OP_MUL,
		VEC_SCALAR_OP_DIV,
		VEC_SCALAR_OP_POW,
		VEC_SCALAR_MAX_OP
	};

	void vec_scalar_op_node_set_op(ShaderType p_which,float p_id,VecScalarOp p_op);
	VecScalarOp vec_scalar_op_node_get_op(ShaderType p_which,float p_id) const;

	enum RGBOp {
		RGB_OP_SCREEN,
		RGB_OP_DIFFERENCE,
		RGB_OP_DARKEN,
		RGB_OP_LIGHTEN,
		RGB_OP_OVERLAY,
		RGB_OP_DODGE,
		RGB_OP_BURN,
		RGB_OP_SOFT_LIGHT,
		RGB_OP_HARD_LIGHT,
		RGB_MAX_OP
	};

	void rgb_op_node_set_op(ShaderType p_which,float p_id,RGBOp p_op);
	RGBOp rgb_op_node_get_op(ShaderType p_which,float p_id) const;

	void xform_vec_mult_node_set_no_translation(ShaderType p_which,int p_id,bool p_no_translation);
	bool xform_vec_mult_node_get_no_translation(ShaderType p_which,int p_id) const;

	enum ScalarFunc {
		SCALAR_FUNC_SIN,
		SCALAR_FUNC_COS,
		SCALAR_FUNC_TAN,
		SCALAR_FUNC_ASIN,
		SCALAR_FUNC_ACOS,
		SCALAR_FUNC_ATAN,
		SCALAR_FUNC_SINH,
		SCALAR_FUNC_COSH,
		SCALAR_FUNC_TANH,
		SCALAR_FUNC_LOG,
		SCALAR_FUNC_EXP,
		SCALAR_FUNC_SQRT,
		SCALAR_FUNC_ABS,
		SCALAR_FUNC_SIGN,
		SCALAR_FUNC_FLOOR,
		SCALAR_FUNC_ROUND,
		SCALAR_FUNC_CEIL,
		SCALAR_FUNC_FRAC,
		SCALAR_FUNC_SATURATE,
		SCALAR_FUNC_NEGATE,
		SCALAR_MAX_FUNC
	};

	void scalar_func_node_set_function(ShaderType p_which,int p_id,ScalarFunc p_func);
	ScalarFunc scalar_func_node_get_function(ShaderType p_which,int p_id) const;

	enum VecFunc {
		VEC_FUNC_NORMALIZE,
		VEC_FUNC_SATURATE,
		VEC_FUNC_NEGATE,
		VEC_FUNC_RECIPROCAL,
		VEC_FUNC_RGB2HSV,
		VEC_FUNC_HSV2RGB,
		VEC_MAX_FUNC
	};

	void default_set_value(ShaderType p_which,int p_id,int p_param, const Variant& p_value);
	Variant default_get_value(ShaderType p_which,int p_id,int p_param);

	void vec_func_node_set_function(ShaderType p_which,int p_id,VecFunc p_func);
	VecFunc vec_func_node_get_function(ShaderType p_which,int p_id) const;

	void color_ramp_node_set_ramp(ShaderType p_which,int p_id,const PoolVector<Color>& p_colors, const PoolVector<real_t>& p_offsets);
	PoolVector<Color> color_ramp_node_get_colors(ShaderType p_which,int p_id) const;
	PoolVector<real_t> color_ramp_node_get_offsets(ShaderType p_which,int p_id) const;

	void curve_map_node_set_points(ShaderType p_which, int p_id, const PoolVector<Vector2>& p_points);
	PoolVector<Vector2> curve_map_node_get_points(ShaderType p_which,int p_id) const;

	void input_node_set_name(ShaderType p_which,int p_id,const String& p_name);
	String input_node_get_name(ShaderType p_which,int p_id);

	void scalar_input_node_set_value(ShaderType p_which,int p_id,float p_value);
	float scalar_input_node_get_value(ShaderType p_which,int p_id) const;

	void vec_input_node_set_value(ShaderType p_which,int p_id,const Vector3& p_value);
	Vector3 vec_input_node_get_value(ShaderType p_which,int p_id) const;

	void rgb_input_node_set_value(ShaderType p_which,int p_id,const Color& p_value);
	Color rgb_input_node_get_value(ShaderType p_which,int p_id) const;

	void xform_input_node_set_value(ShaderType p_which,int p_id,const Transform& p_value);
	Transform xform_input_node_get_value(ShaderType p_which,int p_id) const;

	void texture_input_node_set_value(ShaderType p_which,int p_id,const Ref<Texture>& p_texture);
	Ref<Texture> texture_input_node_get_value(ShaderType p_which,int p_id) const;

	void cubemap_input_node_set_value(ShaderType p_which,int p_id,const Ref<CubeMap>& p_cubemap);
	Ref<CubeMap> cubemap_input_node_get_value(ShaderType p_which,int p_id) const;

	void comment_node_set_text(ShaderType p_which,int p_id,const String& p_comment);
	String comment_node_get_text(ShaderType p_which,int p_id) const;

	Error connect_node(ShaderType p_which,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);
	bool is_node_connected(ShaderType p_which,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) const;
	void disconnect_node(ShaderType p_which,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);

	void get_node_connections(ShaderType p_which,List<Connection> *p_connections) const;

	bool is_slot_connected(ShaderType p_which,int p_dst_id,int slot_id);

	void clear(ShaderType p_which);

	Variant node_get_state(ShaderType p_type, int p_node) const;
	void node_set_state(ShaderType p_type, int p_id, const Variant& p_state);

	GraphError get_graph_error(ShaderType p_type) const;

	int node_count(ShaderType p_which, int p_type);

	static int get_type_input_count(NodeType p_type);
	static int get_type_output_count(NodeType p_type);
	static SlotType get_type_input_type(NodeType p_type,int p_idx);
	static SlotType get_type_output_type(NodeType p_type,int p_idx);
	static bool is_type_valid(Mode p_mode,ShaderType p_type);


	struct SlotInfo {
		String name;
		SlotType type;
		SlotDir dir;
	};

	static void get_input_output_node_slot_info(Mode p_mode, ShaderType  p_type, List<SlotInfo> *r_slots);

	static int get_node_input_slot_count(Mode p_mode, ShaderType  p_shader_type,NodeType p_type);
	static int get_node_output_slot_count(Mode p_mode, ShaderType  p_shader_type,NodeType p_type);
	static SlotType get_node_input_slot_type(Mode p_mode, ShaderType  p_shader_type,NodeType p_type,int p_idx);
	static SlotType get_node_output_slot_type(Mode p_mode, ShaderType  p_shader_type,NodeType p_type,int p_idx);


	ShaderGraph(Mode p_mode);
	~ShaderGraph();
};

//helper functions




VARIANT_ENUM_CAST( ShaderGraph::NodeType );
VARIANT_ENUM_CAST( ShaderGraph::ShaderType );
VARIANT_ENUM_CAST( ShaderGraph::SlotType );
VARIANT_ENUM_CAST( ShaderGraph::ScalarOp );
VARIANT_ENUM_CAST( ShaderGraph::VecOp );
VARIANT_ENUM_CAST( ShaderGraph::VecScalarOp );
VARIANT_ENUM_CAST( ShaderGraph::RGBOp );
VARIANT_ENUM_CAST( ShaderGraph::ScalarFunc );
VARIANT_ENUM_CAST( ShaderGraph::VecFunc );
VARIANT_ENUM_CAST( ShaderGraph::GraphError );


class MaterialShaderGraph : public ShaderGraph {

	GDCLASS( MaterialShaderGraph, ShaderGraph );

public:


	MaterialShaderGraph() : ShaderGraph(MODE_MATERIAL) {

	}
};

class CanvasItemShaderGraph : public ShaderGraph {

	GDCLASS( CanvasItemShaderGraph, ShaderGraph );

public:


	CanvasItemShaderGraph() : ShaderGraph(MODE_CANVAS_ITEM) {

	}
};

#endif
#endif // SHADER_GRAPH_H
