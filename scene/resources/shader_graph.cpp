/*************************************************************************/
/*  shader_graph.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "shader_graph.h"

#include "scene/scene_string_names.h"

// FIXME: Needs to be ported to the new 3.0 shader API
#if 0
Array ShaderGraph::_get_node_list(ShaderType p_type) const {

	List<int> nodes;
	get_node_list(p_type,&nodes);
	Array arr(true);
	for (List<int>::Element *E=nodes.front();E;E=E->next())
		arr.push_back(E->get());
	return arr;
}
Array ShaderGraph::_get_connections(ShaderType p_type) const {

	List<Connection> connections;
	get_node_connections(p_type,&connections);
	Array arr(true);
	for (List<Connection>::Element *E=connections.front();E;E=E->next()) {

		Dictionary d(true);
		d["src_id"]=E->get().src_id;
		d["src_slot"]=E->get().src_slot;
		d["dst_id"]=E->get().dst_id;
		d["dst_slot"]=E->get().dst_slot;
		arr.push_back(d);

	}
	return arr;
}

void ShaderGraph::_set_data(const Dictionary &p_data) {

	Dictionary d=p_data;
	ERR_FAIL_COND(!d.has("shaders"));
	Array sh=d["shaders"];
	ERR_FAIL_COND(sh.size()!=3);

	for(int t=0;t<3;t++) {
		Array data=sh[t];
		ERR_FAIL_COND((data.size()%6)!=0);
		shader[t].node_map.clear();
		for(int i=0;i<data.size();i+=6) {

			Node n;
			n.id=data[i+0];
			n.type=NodeType(int(data[i+1]));
			n.pos=data[i+2];
			n.param1=data[i+3];
			n.param2=data[i+4];

			Array conns=data[i+5];
			ERR_FAIL_COND((conns.size()%3)!=0);

			for(int j=0;j<conns.size();j+=3) {
				SourceSlot ss;
				int ls=conns[j+0];
				if (ls == SLOT_DEFAULT_VALUE) {
					n.defaults[conns[j+1]]=conns[j+2];
				} else {
					ss.id=conns[j+1];
					ss.slot=conns[j+2];
					n.connections[ls]=ss;
				}
			}
			shader[t].node_map[n.id]=n;

		}
	}

	_pending_update_shader=true;
	_update_shader();

}

Dictionary ShaderGraph::_get_data() const {

	Array sh;
	for(int i=0;i<3;i++) {
		Array data;
		int ec = shader[i].node_map.size();
		data.resize(ec*6);
		int idx=0;
		for (Map<int,Node>::Element*E=shader[i].node_map.front();E;E=E->next()) {

			data[idx+0]=E->key();
			data[idx+1]=E->get().type;
			data[idx+2]=E->get().pos;
			data[idx+3]=E->get().param1;
			data[idx+4]=E->get().param2;

			Array conns;
			conns.resize(E->get().connections.size()*3+E->get().defaults.size()*3);
			int idx2=0;
			for(Map<int,SourceSlot>::Element*F=E->get().connections.front();F;F=F->next()) {

				conns[idx2+0]=F->key();
				conns[idx2+1]=F->get().id;
				conns[idx2+2]=F->get().slot;
				idx2+=3;
			}
			for(Map<int,Variant>::Element*F=E->get().defaults.front();F;F=F->next()) {

				conns[idx2+0]=SLOT_DEFAULT_VALUE;
				conns[idx2+1]=F->key();
				conns[idx2+2]=F->get();
				idx2+=3;
			}

			data[idx+5]=conns;
			idx+=6;
		}
		sh.push_back(data);
	}

	Dictionary data;
	data["shaders"]=sh;
	return data;
}



ShaderGraph::GraphError ShaderGraph::get_graph_error(ShaderType p_type) const {

	ERR_FAIL_INDEX_V(p_type,3,GRAPH_OK);
	return shader[p_type].error;
}

int ShaderGraph::node_count(ShaderType p_which, int p_type)
{
	int count=0;
	for (Map<int,Node>::Element *E=shader[p_which].node_map.front();E;E=E->next())
		if (E->get().type==p_type)
			count++;
	return count;
}

void ShaderGraph::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_update_shader"),&ShaderGraph::_update_shader);

	ClassDB::bind_method(D_METHOD("node_add","shader_type","node_type","id"),&ShaderGraph::node_add);
	ClassDB::bind_method(D_METHOD("node_remove","shader_type","id"),&ShaderGraph::node_remove);
	ClassDB::bind_method(D_METHOD("node_set_position","shader_type","id","position"),&ShaderGraph::node_set_position);
	ClassDB::bind_method(D_METHOD("node_get_position","shader_type","id"),&ShaderGraph::node_get_position);

	ClassDB::bind_method(D_METHOD("node_get_type","shader_type","id"),&ShaderGraph::node_get_type);

	ClassDB::bind_method(D_METHOD("get_node_list","shader_type"),&ShaderGraph::_get_node_list);

	ClassDB::bind_method(D_METHOD("default_set_value","shader_type","id","param_id","value"), &ShaderGraph::default_set_value);
	ClassDB::bind_method(D_METHOD("default_get_value","shader_type","id","param_id"), &ShaderGraph::default_get_value);

	ClassDB::bind_method(D_METHOD("scalar_const_node_set_value","shader_type","id","value"),&ShaderGraph::scalar_const_node_set_value);
	ClassDB::bind_method(D_METHOD("scalar_const_node_get_value","shader_type","id"),&ShaderGraph::scalar_const_node_get_value);

	ClassDB::bind_method(D_METHOD("vec_const_node_set_value","shader_type","id","value"),&ShaderGraph::vec_const_node_set_value);
	ClassDB::bind_method(D_METHOD("vec_const_node_get_value","shader_type","id"),&ShaderGraph::vec_const_node_get_value);

	ClassDB::bind_method(D_METHOD("rgb_const_node_set_value","shader_type","id","value"),&ShaderGraph::rgb_const_node_set_value);
	ClassDB::bind_method(D_METHOD("rgb_const_node_get_value","shader_type","id"),&ShaderGraph::rgb_const_node_get_value);

	ClassDB::bind_method(D_METHOD("xform_const_node_set_value","shader_type","id","value"),&ShaderGraph::xform_const_node_set_value);
	ClassDB::bind_method(D_METHOD("xform_const_node_get_value","shader_type","id"),&ShaderGraph::xform_const_node_get_value);


	//void get_node_list(ShaderType p_which,List<int> *p_node_list) const;

	ClassDB::bind_method(D_METHOD("texture_node_set_filter_size","shader_type","id","filter_size"),&ShaderGraph::texture_node_set_filter_size);
	ClassDB::bind_method(D_METHOD("texture_node_get_filter_size","shader_type","id"),&ShaderGraph::texture_node_get_filter_size);

	ClassDB::bind_method(D_METHOD("texture_node_set_filter_strength","shader_type","id","filter_strength"),&ShaderGraph::texture_node_set_filter_strength);
	ClassDB::bind_method(D_METHOD("texture_node_get_filter_strength","shader_type","id"),&ShaderGraph::texture_node_get_filter_strength);

	ClassDB::bind_method(D_METHOD("scalar_op_node_set_op","shader_type","id","op"),&ShaderGraph::scalar_op_node_set_op);
	ClassDB::bind_method(D_METHOD("scalar_op_node_get_op","shader_type","id"),&ShaderGraph::scalar_op_node_get_op);

	ClassDB::bind_method(D_METHOD("vec_op_node_set_op","shader_type","id","op"),&ShaderGraph::vec_op_node_set_op);
	ClassDB::bind_method(D_METHOD("vec_op_node_get_op","shader_type","id"),&ShaderGraph::vec_op_node_get_op);

	ClassDB::bind_method(D_METHOD("vec_scalar_op_node_set_op","shader_type","id","op"),&ShaderGraph::vec_scalar_op_node_set_op);
	ClassDB::bind_method(D_METHOD("vec_scalar_op_node_get_op","shader_type","id"),&ShaderGraph::vec_scalar_op_node_get_op);

	ClassDB::bind_method(D_METHOD("rgb_op_node_set_op","shader_type","id","op"),&ShaderGraph::rgb_op_node_set_op);
	ClassDB::bind_method(D_METHOD("rgb_op_node_get_op","shader_type","id"),&ShaderGraph::rgb_op_node_get_op);


	ClassDB::bind_method(D_METHOD("xform_vec_mult_node_set_no_translation","shader_type","id","disable"),&ShaderGraph::xform_vec_mult_node_set_no_translation);
	ClassDB::bind_method(D_METHOD("xform_vec_mult_node_get_no_translation","shader_type","id"),&ShaderGraph::xform_vec_mult_node_get_no_translation);

	ClassDB::bind_method(D_METHOD("scalar_func_node_set_function","shader_type","id","func"),&ShaderGraph::scalar_func_node_set_function);
	ClassDB::bind_method(D_METHOD("scalar_func_node_get_function","shader_type","id"),&ShaderGraph::scalar_func_node_get_function);

	ClassDB::bind_method(D_METHOD("vec_func_node_set_function","shader_type","id","func"),&ShaderGraph::vec_func_node_set_function);
	ClassDB::bind_method(D_METHOD("vec_func_node_get_function","shader_type","id"),&ShaderGraph::vec_func_node_get_function);

	ClassDB::bind_method(D_METHOD("input_node_set_name","shader_type","id","name"),&ShaderGraph::input_node_set_name);
	ClassDB::bind_method(D_METHOD("input_node_get_name","shader_type","id"),&ShaderGraph::input_node_get_name);

	ClassDB::bind_method(D_METHOD("scalar_input_node_set_value","shader_type","id","value"),&ShaderGraph::scalar_input_node_set_value);
	ClassDB::bind_method(D_METHOD("scalar_input_node_get_value","shader_type","id"),&ShaderGraph::scalar_input_node_get_value);

	ClassDB::bind_method(D_METHOD("vec_input_node_set_value","shader_type","id","value"),&ShaderGraph::vec_input_node_set_value);
	ClassDB::bind_method(D_METHOD("vec_input_node_get_value","shader_type","id"),&ShaderGraph::vec_input_node_get_value);

	ClassDB::bind_method(D_METHOD("rgb_input_node_set_value","shader_type","id","value"),&ShaderGraph::rgb_input_node_set_value);
	ClassDB::bind_method(D_METHOD("rgb_input_node_get_value","shader_type","id"),&ShaderGraph::rgb_input_node_get_value);

	ClassDB::bind_method(D_METHOD("xform_input_node_set_value","shader_type","id","value"),&ShaderGraph::xform_input_node_set_value);
	ClassDB::bind_method(D_METHOD("xform_input_node_get_value","shader_type","id"),&ShaderGraph::xform_input_node_get_value);

	ClassDB::bind_method(D_METHOD("texture_input_node_set_value","shader_type","id","value"),&ShaderGraph::texture_input_node_set_value);
	ClassDB::bind_method(D_METHOD("texture_input_node_get_value","shader_type","id"),&ShaderGraph::texture_input_node_get_value);

	ClassDB::bind_method(D_METHOD("cubemap_input_node_set_value","shader_type","id","value"),&ShaderGraph::cubemap_input_node_set_value);
	ClassDB::bind_method(D_METHOD("cubemap_input_node_get_value","shader_type","id"),&ShaderGraph::cubemap_input_node_get_value);

	ClassDB::bind_method(D_METHOD("comment_node_set_text","shader_type","id","text"),&ShaderGraph::comment_node_set_text);
	ClassDB::bind_method(D_METHOD("comment_node_get_text","shader_type","id"),&ShaderGraph::comment_node_get_text);

	ClassDB::bind_method(D_METHOD("color_ramp_node_set_ramp","shader_type","id","colors","offsets"),&ShaderGraph::color_ramp_node_set_ramp);
	ClassDB::bind_method(D_METHOD("color_ramp_node_get_colors","shader_type","id"),&ShaderGraph::color_ramp_node_get_colors);
	ClassDB::bind_method(D_METHOD("color_ramp_node_get_offsets","shader_type","id"),&ShaderGraph::color_ramp_node_get_offsets);

	ClassDB::bind_method(D_METHOD("curve_map_node_set_points","shader_type","id","points"),&ShaderGraph::curve_map_node_set_points);
	ClassDB::bind_method(D_METHOD("curve_map_node_get_points","shader_type","id"),&ShaderGraph::curve_map_node_get_points);

	ClassDB::bind_method(D_METHOD("connect_node","shader_type","src_id","src_slot","dst_id","dst_slot"),&ShaderGraph::connect_node);
	ClassDB::bind_method(D_METHOD("is_node_connected","shader_type","src_id","src_slot","dst_id","dst_slot"),&ShaderGraph::is_node_connected);
	ClassDB::bind_method(D_METHOD("disconnect_node","shader_type","src_id","src_slot","dst_id","dst_slot"),&ShaderGraph::disconnect_node);
	ClassDB::bind_method(D_METHOD("get_node_connections","shader_type"),&ShaderGraph::_get_connections);

	ClassDB::bind_method(D_METHOD("clear","shader_type"),&ShaderGraph::clear);

	ClassDB::bind_method(D_METHOD("node_set_state","shader_type","id","state"),&ShaderGraph::node_set_state);
	ClassDB::bind_method(D_METHOD("node_get_state","shader_type","id"),&ShaderGraph::node_get_state);

	ClassDB::bind_method(D_METHOD("_set_data"),&ShaderGraph::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"),&ShaderGraph::_get_data);

	ADD_PROPERTY( PropertyInfo(Variant::DICTIONARY,"_data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR), "_set_data","_get_data");

	//void get_connections(ShaderType p_which,List<Connection> *p_connections) const;


	BIND_ENUM_CONSTANT( NODE_INPUT ); // all inputs (shader type dependent)
	BIND_ENUM_CONSTANT( NODE_SCALAR_CONST ); //scalar constant
	BIND_ENUM_CONSTANT( NODE_VEC_CONST ); //vec3 constant
	BIND_ENUM_CONSTANT( NODE_RGB_CONST ); //rgb constant (shows a color picker instead)
	BIND_ENUM_CONSTANT( NODE_XFORM_CONST ); // 4x4 matrix constant
	BIND_ENUM_CONSTANT( NODE_TIME ); // time in seconds
	BIND_ENUM_CONSTANT( NODE_SCREEN_TEX ); // screen texture sampler (takes UV) (only usable in fragment shader)
	BIND_ENUM_CONSTANT( NODE_SCALAR_OP ); // scalar vs scalar op (mul ); add ); div ); etc)
	BIND_ENUM_CONSTANT( NODE_VEC_OP ); // vec3 vs vec3 op (mul );ad );div );crossprod );etc)
	BIND_ENUM_CONSTANT( NODE_VEC_SCALAR_OP ); // vec3 vs scalar op (mul ); add ); div ); etc)
	BIND_ENUM_CONSTANT( NODE_RGB_OP ); // vec3 vs vec3 rgb op (with scalar amount) ); like brighten ); darken ); burn ); dodge ); multiply ); etc.
	BIND_ENUM_CONSTANT( NODE_XFORM_MULT ); // mat4 x mat4
	BIND_ENUM_CONSTANT( NODE_XFORM_VEC_MULT ); // mat4 x vec3 mult (with no-translation option)
	BIND_ENUM_CONSTANT( NODE_XFORM_VEC_INV_MULT ); // mat4 x vec3 inverse mult (with no-translation option)
	BIND_ENUM_CONSTANT( NODE_SCALAR_FUNC ); // scalar function (sin ); cos ); etc)
	BIND_ENUM_CONSTANT( NODE_VEC_FUNC ); // vector function (normalize ); negate ); reciprocal ); rgb2hsv ); hsv2rgb ); etc ); etc)
	BIND_ENUM_CONSTANT( NODE_VEC_LEN ); // vec3 length
	BIND_ENUM_CONSTANT( NODE_DOT_PROD ); // vec3 . vec3 (dot product -> scalar output)
	BIND_ENUM_CONSTANT( NODE_VEC_TO_SCALAR ); // 1 vec3 input ); 3 scalar outputs
	BIND_ENUM_CONSTANT( NODE_SCALAR_TO_VEC ); // 3 scalar input ); 1 vec3 output
	BIND_ENUM_CONSTANT( NODE_VEC_TO_XFORM ); // 3 vec input ); 1 xform output
	BIND_ENUM_CONSTANT( NODE_XFORM_TO_VEC ); // 3 vec input ); 1 xform output
	BIND_ENUM_CONSTANT( NODE_SCALAR_INTERP ); // scalar interpolation (with optional curve)
	BIND_ENUM_CONSTANT( NODE_VEC_INTERP ); // vec3 interpolation  (with optional curve)
	BIND_ENUM_CONSTANT( NODE_COLOR_RAMP );
	BIND_ENUM_CONSTANT( NODE_CURVE_MAP );
	BIND_ENUM_CONSTANT( NODE_SCALAR_INPUT ); // scalar uniform (assignable in material)
	BIND_ENUM_CONSTANT( NODE_VEC_INPUT ); // vec3 uniform (assignable in material)
	BIND_ENUM_CONSTANT( NODE_RGB_INPUT ); // color uniform (assignable in material)
	BIND_ENUM_CONSTANT( NODE_XFORM_INPUT ); // mat4 uniform (assignable in material)
	BIND_ENUM_CONSTANT( NODE_TEXTURE_INPUT ); // texture input (assignable in material)
	BIND_ENUM_CONSTANT( NODE_CUBEMAP_INPUT ); // cubemap input (assignable in material)
	BIND_ENUM_CONSTANT( NODE_DEFAULT_TEXTURE );
	BIND_ENUM_CONSTANT( NODE_OUTPUT ); // output (shader type dependent)
	BIND_ENUM_CONSTANT( NODE_COMMENT ); // comment
	BIND_ENUM_CONSTANT( NODE_TYPE_MAX );

	BIND_ENUM_CONSTANT( SLOT_TYPE_SCALAR );
	BIND_ENUM_CONSTANT( SLOT_TYPE_VEC );
	BIND_ENUM_CONSTANT( SLOT_TYPE_XFORM );
	BIND_ENUM_CONSTANT( SLOT_TYPE_TEXTURE );
	BIND_ENUM_CONSTANT( SLOT_MAX );

	BIND_ENUM_CONSTANT( SHADER_TYPE_VERTEX );
	BIND_ENUM_CONSTANT( SHADER_TYPE_FRAGMENT );
	BIND_ENUM_CONSTANT( SHADER_TYPE_LIGHT );
	BIND_ENUM_CONSTANT( SHADER_TYPE_MAX );


	BIND_ENUM_CONSTANT( SLOT_IN );
	BIND_ENUM_CONSTANT( SLOT_OUT );

	BIND_ENUM_CONSTANT( GRAPH_OK );
	BIND_ENUM_CONSTANT( GRAPH_ERROR_CYCLIC  );
	BIND_ENUM_CONSTANT( GRAPH_ERROR_MISSING_CONNECTIONS );

	BIND_ENUM_CONSTANT( SCALAR_OP_ADD );
	BIND_ENUM_CONSTANT( SCALAR_OP_SUB );
	BIND_ENUM_CONSTANT( SCALAR_OP_MUL );
	BIND_ENUM_CONSTANT( SCALAR_OP_DIV );
	BIND_ENUM_CONSTANT( SCALAR_OP_MOD );
	BIND_ENUM_CONSTANT( SCALAR_OP_POW );
	BIND_ENUM_CONSTANT( SCALAR_OP_MAX );
	BIND_ENUM_CONSTANT( SCALAR_OP_MIN );
	BIND_ENUM_CONSTANT( SCALAR_OP_ATAN2 );
	BIND_ENUM_CONSTANT( SCALAR_MAX_OP );

	BIND_ENUM_CONSTANT( VEC_OP_ADD );
	BIND_ENUM_CONSTANT( VEC_OP_SUB );
	BIND_ENUM_CONSTANT( VEC_OP_MUL );
	BIND_ENUM_CONSTANT( VEC_OP_DIV );
	BIND_ENUM_CONSTANT( VEC_OP_MOD );
	BIND_ENUM_CONSTANT( VEC_OP_POW );
	BIND_ENUM_CONSTANT( VEC_OP_MAX );
	BIND_ENUM_CONSTANT( VEC_OP_MIN );
	BIND_ENUM_CONSTANT( VEC_OP_CROSS );
	BIND_ENUM_CONSTANT( VEC_MAX_OP );

	BIND_ENUM_CONSTANT( VEC_SCALAR_OP_MUL );
	BIND_ENUM_CONSTANT( VEC_SCALAR_OP_DIV );
	BIND_ENUM_CONSTANT( VEC_SCALAR_OP_POW );
	BIND_ENUM_CONSTANT( VEC_SCALAR_MAX_OP );

	BIND_ENUM_CONSTANT( RGB_OP_SCREEN );
	BIND_ENUM_CONSTANT( RGB_OP_DIFFERENCE );
	BIND_ENUM_CONSTANT( RGB_OP_DARKEN );
	BIND_ENUM_CONSTANT( RGB_OP_LIGHTEN );
	BIND_ENUM_CONSTANT( RGB_OP_OVERLAY );
	BIND_ENUM_CONSTANT( RGB_OP_DODGE );
	BIND_ENUM_CONSTANT( RGB_OP_BURN );
	BIND_ENUM_CONSTANT( RGB_OP_SOFT_LIGHT );
	BIND_ENUM_CONSTANT( RGB_OP_HARD_LIGHT );
	BIND_ENUM_CONSTANT( RGB_MAX_OP );

	BIND_ENUM_CONSTANT( SCALAR_FUNC_SIN );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_COS );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_TAN );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_ASIN );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_ACOS );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_ATAN );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_SINH );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_COSH );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_TANH );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_LOG );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_EXP );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_SQRT );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_ABS );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_SIGN );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_FLOOR );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_ROUND );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_CEIL );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_FRAC );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_SATURATE );
	BIND_ENUM_CONSTANT( SCALAR_FUNC_NEGATE );
	BIND_ENUM_CONSTANT( SCALAR_MAX_FUNC );

	BIND_ENUM_CONSTANT( VEC_FUNC_NORMALIZE );
	BIND_ENUM_CONSTANT( VEC_FUNC_SATURATE );
	BIND_ENUM_CONSTANT( VEC_FUNC_NEGATE );
	BIND_ENUM_CONSTANT( VEC_FUNC_RECIPROCAL );
	BIND_ENUM_CONSTANT( VEC_FUNC_RGB2HSV );
	BIND_ENUM_CONSTANT( VEC_FUNC_HSV2RGB );
	BIND_ENUM_CONSTANT( VEC_MAX_FUNC );

	ADD_SIGNAL(MethodInfo("updated"));
}


String ShaderGraph::_find_unique_name(const String& p_base) {



	int idx=1;
	while(true) {
		String tocmp=p_base;
		if (idx>1) {
			tocmp+="_"+itos(idx);
		}
		bool valid=true;
		for(int i=0;i<3;i++) {
			if (!valid)
				break;
			for (Map<int,Node>::Element *E=shader[i].node_map.front();E;E=E->next()) {
				if (E->get().type!=NODE_SCALAR_INPUT && E->get().type!=NODE_VEC_INPUT && E->get().type==NODE_RGB_INPUT && E->get().type==NODE_XFORM_INPUT && E->get().type==NODE_TEXTURE_INPUT && E->get().type==NODE_CUBEMAP_INPUT)
					continue;
				String name = E->get().param1;
				if (name==tocmp) {
					valid=false;
					break;
				}

			}
		}

		if (!valid) {
			idx++;
			continue;
		}
		return tocmp;
	}
	return String();
}

void ShaderGraph::node_add(ShaderType p_type, NodeType p_node_type,int p_id) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(p_id==0);
	ERR_FAIL_COND(p_node_type==NODE_OUTPUT); //can't create output
	ERR_FAIL_COND( shader[p_type].node_map.has(p_id ) );
	ERR_FAIL_INDEX( p_node_type, NODE_TYPE_MAX );
	Node node;

	if (p_node_type==NODE_INPUT) {
		//see if it already exists
		for(Map<int,Node>::Element *E=shader[p_type].node_map.front();E;E=E->next()) {
			if (E->get().type==NODE_INPUT) {
				ERR_EXPLAIN("Only one input node can be added to the graph.");
				ERR_FAIL_COND(E->get().type==NODE_INPUT);
			}
		}
	}
	node.type=p_node_type;
	node.id=p_id;

	switch(p_node_type) {
		case NODE_INPUT: {} break; // all inputs (shader type dependent)
		case NODE_SCALAR_CONST: { node.param1=0;} break; //scalar constant
		case NODE_VEC_CONST: {node.param1=Vector3();} break; //vec3 constant
		case NODE_RGB_CONST: {node.param1=Color();} break; //rgb constant (shows a color picker instead)
		case NODE_XFORM_CONST: {node.param1=Transform();} break; // 4x4 matrix constant
		case NODE_TIME: {} break; // time in seconds
		case NODE_SCREEN_TEX: {Array arr; arr.push_back(0); arr.push_back(0); node.param2=arr;} break; // screen texture sampler (takes UV) (only usable in fragment shader)
		case NODE_SCALAR_OP: {node.param1=SCALAR_OP_ADD;} break; // scalar vs scalar op (mul: {} break; add: {} break; div: {} break; etc)
		case NODE_VEC_OP: {node.param1=VEC_OP_ADD;} break; // vec3 vs vec3 op (mul: {} break;ad: {} break;div: {} break;crossprod: {} break;etc)
		case NODE_VEC_SCALAR_OP: {node.param1=VEC_SCALAR_OP_MUL;} break; // vec3 vs scalar op (mul: {} break; add: {} break; div: {} break; etc)
		case NODE_RGB_OP: {node.param1=RGB_OP_SCREEN;} break; // vec3 vs vec3 rgb op (with scalar amount): {} break; like brighten: {} break; darken: {} break; burn: {} break; dodge: {} break; multiply: {} break; etc.
		case NODE_XFORM_MULT: {} break; // mat4 x mat4
		case NODE_XFORM_VEC_MULT: {} break; // mat4 x vec3 mult (with no-translation option)
		case NODE_XFORM_VEC_INV_MULT: {} break; // mat4 x vec3 inverse mult (with no-translation option)
		case NODE_SCALAR_FUNC: {node.param1=SCALAR_FUNC_SIN;} break; // scalar function (sin: {} break; cos: {} break; etc)
		case NODE_VEC_FUNC: {node.param1=VEC_FUNC_NORMALIZE;} break; // vector function (normalize: {} break; negate: {} break; reciprocal: {} break; rgb2hsv: {} break; hsv2rgb: {} break; etc: {} break; etc)
		case NODE_VEC_LEN: {} break; // vec3 length
		case NODE_DOT_PROD: {} break; // vec3 . vec3 (dot product -> scalar output)
		case NODE_VEC_TO_SCALAR: {} break; // 1 vec3 input: {} break; 3 scalar outputs
		case NODE_SCALAR_TO_VEC: {} break; // 3 scalar input: {} break; 1 vec3 output
		case NODE_VEC_TO_XFORM: {} break; // 3 scalar input: {} break; 1 vec3 output
		case NODE_XFORM_TO_VEC: {} break; // 3 scalar input: {} break; 1 vec3 output
		case NODE_SCALAR_INTERP: {} break; // scalar interpolation (with optional curve)
		case NODE_VEC_INTERP: {} break; // vec3 interpolation  (with optional curve)
		case NODE_COLOR_RAMP: { node.param1=PoolVector<Color>(); node.param2=PoolVector<real_t>();} break; // vec3 interpolation  (with optional curve)
		case NODE_CURVE_MAP: { node.param1=PoolVector<Vector2>();} break; // vec3 interpolation  (with optional curve)
		case NODE_SCALAR_INPUT: {node.param1=_find_unique_name("Scalar"); node.param2=0;} break; // scalar uniform (assignable in material)
		case NODE_VEC_INPUT: {node.param1=_find_unique_name("Vec3");node.param2=Vector3();} break; // vec3 uniform (assignable in material)
		case NODE_RGB_INPUT: {node.param1=_find_unique_name("Color");node.param2=Color();} break; // color uniform (assignable in material)
		case NODE_XFORM_INPUT: {node.param1=_find_unique_name("XForm"); node.param2=Transform();} break; // mat4 uniform (assignable in material)
		case NODE_TEXTURE_INPUT: {node.param1=_find_unique_name("Tex"); } break; // texture input (assignable in material)
		case NODE_CUBEMAP_INPUT: {node.param1=_find_unique_name("Cube"); } break; // cubemap input (assignable in material)
		case NODE_DEFAULT_TEXTURE: {}; break;
		case NODE_OUTPUT: {} break; // output (shader type dependent)
		case NODE_COMMENT: {} break; // comment
		case NODE_TYPE_MAX: {};
	}

	shader[p_type].node_map[p_id]=node;
	_request_update();
}

void ShaderGraph::node_set_position(ShaderType p_type,int p_id, const Vector2& p_pos) {
	ERR_FAIL_INDEX(p_type,3);

	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	shader[p_type].node_map[p_id].pos=p_pos;
	_request_update();

}
Vector2 ShaderGraph::node_get_position(ShaderType p_type,int p_id) const {
	ERR_FAIL_INDEX_V(p_type,3,Vector2());

	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Vector2());
	return shader[p_type].node_map[p_id].pos;
}


void ShaderGraph::node_remove(ShaderType p_type,int p_id) {

	ERR_FAIL_COND(p_id==0);
	ERR_FAIL_INDEX(p_type,3);

	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));

	//erase connections associated with node
	for(Map<int,Node>::Element *E=shader[p_type].node_map.front();E;E=E->next()) {
		if (E->key()==p_id)
			continue; //no self

		for (Map<int,SourceSlot>::Element *F=E->get().connections.front();F;) {
			Map<int,SourceSlot>::Element *N=F->next();

			if (F->get().id==p_id) {
				E->get().connections.erase(F);
			}

			F=N;
		}
	}

	shader[p_type].node_map.erase(p_id);

	_request_update();

}



void ShaderGraph::get_node_list(ShaderType p_type,List<int> *p_node_list) const {

	ERR_FAIL_INDEX(p_type,3);

	Map<int,Node>::Element *E = shader[p_type].node_map.front();

	while(E) {

		p_node_list->push_back(E->key());
		E=E->next();
	}
}


ShaderGraph::NodeType ShaderGraph::node_get_type(ShaderType p_type,int p_id) const {

	ERR_FAIL_INDEX_V(p_type,3,NODE_TYPE_MAX);

	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),NODE_TYPE_MAX);
	return shader[p_type].node_map[p_id].type;
}


Error ShaderGraph::connect_node(ShaderType p_type,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) {
	ERR_FAIL_INDEX_V(p_type,3,ERR_INVALID_PARAMETER);

	ERR_FAIL_COND_V(p_src_id==p_dst_id, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_src_id), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_dst_id), ERR_INVALID_PARAMETER);
	NodeType type_src=shader[p_type].node_map[p_src_id].type;
	NodeType type_dst=shader[p_type].node_map[p_dst_id].type;
	ERR_FAIL_INDEX_V( p_src_slot, get_node_output_slot_count(get_mode(),p_type,type_src), ERR_INVALID_PARAMETER );
	ERR_FAIL_INDEX_V( p_dst_slot, get_node_input_slot_count(get_mode(),p_type,type_dst), ERR_INVALID_PARAMETER );
	ERR_FAIL_COND_V(get_node_output_slot_type(get_mode(),p_type,type_src,p_src_slot) != get_node_input_slot_type(get_mode(),p_type,type_dst,p_dst_slot), ERR_INVALID_PARAMETER );


	SourceSlot ts;
	ts.id=p_src_id;
	ts.slot=p_src_slot;
	shader[p_type].node_map[p_dst_id].connections[p_dst_slot]=ts;
	_request_update();

	return OK;
}

bool ShaderGraph::is_node_connected(ShaderType p_type,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) const {

	ERR_FAIL_INDEX_V(p_type,3,false);

	SourceSlot ts;
	ts.id=p_src_id;
	ts.slot=p_src_slot;
	return shader[p_type].node_map.has(p_dst_id) && shader[p_type].node_map[p_dst_id].connections.has(p_dst_slot) &&
		shader[p_type].node_map[p_dst_id].connections[p_dst_slot]==ts;
}

void ShaderGraph::disconnect_node(ShaderType p_type,int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) {
	ERR_FAIL_INDEX(p_type,3);

	SourceSlot ts;
	ts.id=p_src_id;
	ts.slot=p_src_slot;
	if (shader[p_type].node_map.has(p_dst_id) && shader[p_type].node_map[p_dst_id].connections.has(p_dst_slot) &&
		shader[p_type].node_map[p_dst_id].connections[p_dst_slot]==ts) {
		shader[p_type].node_map[p_dst_id].connections.erase(p_dst_slot);

	}
	_request_update();

}

void ShaderGraph::get_node_connections(ShaderType p_type,List<Connection> *p_connections) const {

	ERR_FAIL_INDEX(p_type,3);

	for(const Map<int,Node>::Element *E=shader[p_type].node_map.front();E;E=E->next()) {
		for (const Map<int,SourceSlot>::Element *F=E->get().connections.front();F;F=F->next()) {

			Connection c;
			c.dst_id=E->key();
			c.dst_slot=F->key();
			c.src_id=F->get().id;
			c.src_slot=F->get().slot;
			p_connections->push_back(c);
		}
	}
}

bool ShaderGraph::is_slot_connected(ShaderGraph::ShaderType p_type, int p_dst_id, int slot_id)
{
	for(const Map<int,Node>::Element *E=shader[p_type].node_map.front();E;E=E->next()) {
		for (const Map<int,SourceSlot>::Element *F=E->get().connections.front();F;F=F->next()) {

			if (p_dst_id == E->key() && slot_id==F->key())
				return true;
		}
	}
	return false;
}


void ShaderGraph::clear(ShaderType p_type) {

	ERR_FAIL_INDEX(p_type,3);
	shader[p_type].node_map.clear();
	Node out;
	out.pos=Vector2(300,300);
	out.type=NODE_OUTPUT;
	shader[p_type].node_map.insert(0,out);

	_request_update();

}


void ShaderGraph::scalar_const_node_set_value(ShaderType p_type,int p_id,float p_value) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_SCALAR_CONST);
	n.param1=p_value;
	_request_update();

}

float ShaderGraph::scalar_const_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,0);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),0);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_SCALAR_CONST,0);
	return n.param1;
}

void ShaderGraph::vec_const_node_set_value(ShaderType p_type,int p_id,const Vector3& p_value){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_VEC_CONST);
	n.param1=p_value;
	_request_update();


}
Vector3 ShaderGraph::vec_const_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Vector3());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Vector3());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_VEC_CONST,Vector3());
	return n.param1;

}

void ShaderGraph::rgb_const_node_set_value(ShaderType p_type,int p_id,const Color& p_value){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_RGB_CONST);
	n.param1=p_value;
	_request_update();

}
Color ShaderGraph::rgb_const_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Color());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Color());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_RGB_CONST,Color());
	return n.param1;

}

void ShaderGraph::xform_const_node_set_value(ShaderType p_type,int p_id,const Transform& p_value){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_XFORM_CONST);
	n.param1=p_value;
	_request_update();

}
Transform ShaderGraph::xform_const_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Transform());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Transform());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_XFORM_CONST,Transform());
	return n.param1;

}

void ShaderGraph::texture_node_set_filter_size(ShaderType p_type,int p_id,int p_size){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_TEXTURE_INPUT && n.type!=NODE_SCREEN_TEX);
	Array arr = n.param2;
	arr[0]=p_size;
	n.param2=arr;
	_request_update();

}
int ShaderGraph::texture_node_get_filter_size(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,0);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),0);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_TEXTURE_INPUT && n.type!=NODE_SCREEN_TEX,0);
	Array arr = n.param2;
	return arr[0];

}

void ShaderGraph::texture_node_set_filter_strength(ShaderType p_type,float p_id,float p_strength){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_TEXTURE_INPUT && n.type!=NODE_SCREEN_TEX);
	Array arr = n.param2;
	arr[1]=p_strength;
	n.param2=arr;
	_request_update();

}
float ShaderGraph::texture_node_get_filter_strength(ShaderType p_type,float p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,0);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),0);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_TEXTURE_INPUT && n.type!=NODE_SCREEN_TEX,0);
	Array arr = n.param2;
	return arr[1];
}

void ShaderGraph::duplicate_nodes(ShaderType p_which, List<int> &p_nodes)
{
	//Create new node IDs
	Map<int,int> duplicates = Map<int,int>();
	int i=1;
	for(List<int>::Element *E=p_nodes.front();E; E=E->next()) {
		while (shader[p_which].node_map.has(i))
			i++;
		duplicates.insert(E->get(), i);
		i++;
	}

	for(List<int>::Element *E = p_nodes.front();E; E=E->next()) {

		const Node &n=shader[p_which].node_map[E->get()];
		Node nn=n;
		nn.id=duplicates.find(n.id)->get();
		nn.pos += Vector2(0,100);
		for (Map<int,SourceSlot>::Element *C=nn.connections.front();C;C=C->next()) {
			SourceSlot &c=C->get();
			if (p_nodes.find(c.id))
				c.id=duplicates.find(c.id)->get();
		}
		shader[p_which].node_map[nn.id]=nn;
	}
	_request_update();
}

List<int> ShaderGraph::generate_ids(ShaderType p_type, int count)
{
	List<int> ids = List<int>();
	int i=1;
	while (ids.size() < count) {
		while (shader[p_type].node_map.has(i))
			i++;
		ids.push_back(i);
		i++;
	}
	return ids;
}


void ShaderGraph::scalar_op_node_set_op(ShaderType p_type,float p_id,ScalarOp p_op){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_SCALAR_OP);
	n.param1=p_op;
	_request_update();

}
ShaderGraph::ScalarOp ShaderGraph::scalar_op_node_get_op(ShaderType p_type,float p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,SCALAR_MAX_OP);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),SCALAR_MAX_OP);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_SCALAR_OP,SCALAR_MAX_OP);
	int op = n.param1;
	return ScalarOp(op);

}


void ShaderGraph::vec_op_node_set_op(ShaderType p_type,float p_id,VecOp p_op){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_VEC_OP);
	n.param1=p_op;
	_request_update();

}
ShaderGraph::VecOp ShaderGraph::vec_op_node_get_op(ShaderType p_type,float p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,VEC_MAX_OP);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),VEC_MAX_OP);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_VEC_OP,VEC_MAX_OP);
	int op = n.param1;
	return VecOp(op);

}


void ShaderGraph::vec_scalar_op_node_set_op(ShaderType p_type,float p_id,VecScalarOp p_op){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_VEC_SCALAR_OP);
	n.param1=p_op;
	_request_update();

}
ShaderGraph::VecScalarOp ShaderGraph::vec_scalar_op_node_get_op(ShaderType p_type,float p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,VEC_SCALAR_MAX_OP);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),VEC_SCALAR_MAX_OP);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_VEC_SCALAR_OP,VEC_SCALAR_MAX_OP);
	int op = n.param1;
	return VecScalarOp(op);

}

void ShaderGraph::rgb_op_node_set_op(ShaderType p_type,float p_id,RGBOp p_op){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_RGB_OP);
	n.param1=p_op;

	_request_update();

}
ShaderGraph::RGBOp ShaderGraph::rgb_op_node_get_op(ShaderType p_type,float p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,RGB_MAX_OP);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),RGB_MAX_OP);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_RGB_OP,RGB_MAX_OP);
	int op = n.param1;
	return RGBOp(op);

}


void ShaderGraph::xform_vec_mult_node_set_no_translation(ShaderType p_type,int p_id,bool p_no_translation){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_XFORM_VEC_MULT && n.type!=NODE_XFORM_VEC_INV_MULT);
	n.param1=p_no_translation;
	_request_update();

}
bool ShaderGraph::xform_vec_mult_node_get_no_translation(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,false);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),false);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_XFORM_VEC_MULT && n.type!=NODE_XFORM_VEC_INV_MULT,false);
	return n.param1;

}

void ShaderGraph::scalar_func_node_set_function(ShaderType p_type,int p_id,ScalarFunc p_func){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_SCALAR_FUNC);
	int func = p_func;
	ERR_FAIL_INDEX(func,SCALAR_MAX_FUNC);
	n.param1=func;
	_request_update();

}
ShaderGraph::ScalarFunc ShaderGraph::scalar_func_node_get_function(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,SCALAR_MAX_FUNC);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),SCALAR_MAX_FUNC);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_SCALAR_FUNC,SCALAR_MAX_FUNC);
	int func = n.param1;
	return ScalarFunc(func);
}

void ShaderGraph::default_set_value(ShaderGraph::ShaderType p_which, int p_id, int p_param, const Variant &p_value)
{
	ERR_FAIL_INDEX(p_which,3);
	ERR_FAIL_COND(!shader[p_which].node_map.has(p_id));
	Node& n = shader[p_which].node_map[p_id];
	if(p_value.get_type()==Variant::NIL)
		n.defaults.erase(n.defaults.find(p_param));
	else
		n.defaults[p_param]=p_value;

	_request_update();

}

Variant ShaderGraph::default_get_value(ShaderGraph::ShaderType p_which, int p_id, int p_param)
{
	ERR_FAIL_INDEX_V(p_which,3,Variant());
	ERR_FAIL_COND_V(!shader[p_which].node_map.has(p_id),Variant());
	const Node& n = shader[p_which].node_map[p_id];

	if (!n.defaults.has(p_param))
		return Variant();
	return n.defaults[p_param];
}



void ShaderGraph::vec_func_node_set_function(ShaderType p_type,int p_id,VecFunc p_func){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_VEC_FUNC);
	int func = p_func;
	ERR_FAIL_INDEX(func,VEC_MAX_FUNC);
	n.param1=func;

	_request_update();

}
ShaderGraph::VecFunc ShaderGraph::vec_func_node_get_function(ShaderType p_type, int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,VEC_MAX_FUNC);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),VEC_MAX_FUNC);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_VEC_FUNC,VEC_MAX_FUNC);
	int func = n.param1;
	return VecFunc(func);
}

void ShaderGraph::color_ramp_node_set_ramp(ShaderType p_type,int p_id,const PoolVector<Color>& p_colors, const PoolVector<real_t>& p_offsets){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	ERR_FAIL_COND(p_colors.size()!=p_offsets.size());
	Node& n = shader[p_type].node_map[p_id];
	n.param1=p_colors;
	n.param2=p_offsets;
	_request_update();

}

PoolVector<Color> ShaderGraph::color_ramp_node_get_colors(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,PoolVector<Color>());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),PoolVector<Color>());
	const Node& n = shader[p_type].node_map[p_id];
	return n.param1;


}

PoolVector<real_t> ShaderGraph::color_ramp_node_get_offsets(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,PoolVector<real_t>());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),PoolVector<real_t>());
	const Node& n = shader[p_type].node_map[p_id];
	return n.param2;

}


void ShaderGraph::curve_map_node_set_points(ShaderType p_type,int p_id,const PoolVector<Vector2>& p_points) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	n.param1=p_points;
	_request_update();

}

PoolVector<Vector2> ShaderGraph::curve_map_node_get_points(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,PoolVector<Vector2>());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),PoolVector<Vector2>());
	const Node& n = shader[p_type].node_map[p_id];
	return n.param1;

}



void ShaderGraph::input_node_set_name(ShaderType p_type,int p_id,const String& p_name){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	ERR_FAIL_COND(!p_name.is_valid_identifier());
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_SCALAR_INPUT && n.type!=NODE_VEC_INPUT && n.type==NODE_RGB_INPUT && n.type==NODE_XFORM_INPUT && n.type==NODE_TEXTURE_INPUT && n.type==NODE_CUBEMAP_INPUT);

	n.param1="";
	n.param1=_find_unique_name(p_name);
	_request_update();

}
String ShaderGraph::input_node_get_name(ShaderType p_type,int p_id){

	ERR_FAIL_INDEX_V(p_type,3,String());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),String());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_SCALAR_INPUT && n.type!=NODE_VEC_INPUT && n.type==NODE_RGB_INPUT && n.type==NODE_XFORM_INPUT && n.type==NODE_TEXTURE_INPUT && n.type==NODE_CUBEMAP_INPUT,String());
	return n.param1;
}


void ShaderGraph::scalar_input_node_set_value(ShaderType p_type,int p_id,float p_value) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_SCALAR_INPUT);
	n.param2=p_value;
	_request_update();

}

float ShaderGraph::scalar_input_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,0);
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),0);
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_SCALAR_INPUT,0);

	return n.param2;
}

void ShaderGraph::vec_input_node_set_value(ShaderType p_type,int p_id,const Vector3& p_value){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_VEC_INPUT);

	n.param2=p_value;
	_request_update();

}
Vector3 ShaderGraph::vec_input_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Vector3());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Vector3());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_VEC_INPUT,Vector3());
	return n.param2;
}

void ShaderGraph::rgb_input_node_set_value(ShaderType p_type,int p_id,const Color& p_value){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_RGB_INPUT);
	n.param2=p_value;
	_request_update();

}
Color ShaderGraph::rgb_input_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Color());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Color());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_RGB_INPUT,Color());
	return n.param2;
}

void ShaderGraph::xform_input_node_set_value(ShaderType p_type,int p_id,const Transform& p_value){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_XFORM_INPUT);
	n.param2=p_value;
	_request_update();

}
Transform ShaderGraph::xform_input_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Transform());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Transform());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_XFORM_INPUT,Transform());
	return n.param2;
}


void ShaderGraph::texture_input_node_set_value(ShaderType p_type,int p_id,const Ref<Texture>& p_texture) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_TEXTURE_INPUT);
	n.param2=p_texture;
	_request_update();

}

Ref<Texture> ShaderGraph::texture_input_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Ref<Texture>());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Ref<Texture>());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_TEXTURE_INPUT,Ref<Texture>());
	return n.param2;
}

void ShaderGraph::cubemap_input_node_set_value(ShaderType p_type,int p_id,const Ref<CubeMap>& p_cubemap){

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_CUBEMAP_INPUT);
	n.param2=p_cubemap;
	_request_update();

}

Ref<CubeMap> ShaderGraph::cubemap_input_node_get_value(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,Ref<CubeMap>());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Ref<CubeMap>());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_CUBEMAP_INPUT,Ref<CubeMap>());
	return n.param2;

}


void ShaderGraph::comment_node_set_text(ShaderType p_type,int p_id,const String& p_comment) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND(n.type!=NODE_COMMENT);
	n.param1=p_comment;

}

String ShaderGraph::comment_node_get_text(ShaderType p_type,int p_id) const{

	ERR_FAIL_INDEX_V(p_type,3,String());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),String());
	const Node& n = shader[p_type].node_map[p_id];
	ERR_FAIL_COND_V(n.type!=NODE_COMMENT,String());
	return n.param1;

}

void ShaderGraph::_request_update() {

	if (_pending_update_shader)
		return;

	_pending_update_shader=true;
	call_deferred("_update_shader");

}

Variant ShaderGraph::node_get_state(ShaderType p_type,int p_id) const {

	ERR_FAIL_INDEX_V(p_type,3,Variant());
	ERR_FAIL_COND_V(!shader[p_type].node_map.has(p_id),Variant());
	const Node& n = shader[p_type].node_map[p_id];
	Dictionary s;
	s["position"]=n.pos;
	s["param1"]=n.param1;
	s["param2"]=n.param2;
	Array keys;
	for (Map<int,Variant>::Element *E=n.defaults.front();E;E=E->next()) {
		keys.append(E->key());
		s[E->key()]=E->get();
	}
	s["default_keys"]=keys;
	return s;

}
void ShaderGraph::node_set_state(ShaderType p_type,int p_id,const Variant& p_state) {

	ERR_FAIL_INDEX(p_type,3);
	ERR_FAIL_COND(!shader[p_type].node_map.has(p_id));
	Node& n = shader[p_type].node_map[p_id];
	Dictionary d = p_state;
	ERR_FAIL_COND(!d.has("position"));
	ERR_FAIL_COND(!d.has("param1"));
	ERR_FAIL_COND(!d.has("param2"));
	ERR_FAIL_COND(!d.has("default_keys"));

	n.pos=d["position"];
	n.param1=d["param1"];
	n.param2=d["param2"];
	Array keys = d["default_keys"];
	for(int i=0;i<keys.size();i++) {
		n.defaults[keys[i]]=d[keys[i]];
	}
}

ShaderGraph::ShaderGraph(Mode p_mode) : Shader(p_mode) {

	//shader = VisualServer::get_singleton()->shader_create();
	_pending_update_shader=false;

	Node input;
	input.id=1;
	input.pos=Vector2(50,40);
	input.type=NODE_INPUT;

	Node output;
	output.id=0;
	output.pos=Vector2(350,40);
	output.type=NODE_OUTPUT;

	for(int i=0;i<3;i++) {

		shader[i].node_map.insert(0,output);
		shader[i].node_map.insert(1,input);
	}
}

ShaderGraph::~ShaderGraph() {

	//VisualServer::get_singleton()->free(shader);
}


const ShaderGraph::InOutParamInfo ShaderGraph::inout_param_info[]={
	//material vertex in
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Vertex","SRC_VERTEX","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Normal","SRC_NORMAL","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Tangent","SRC_TANGENT","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"BinormalF","SRC_BINORMALF","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Color","SRC_COLOR","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Alpha","SRC_ALPHA","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"UV","SRC_UV","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"UV2","SRC_UV2","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"WorldMatrix","WORLD_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"InvCameraMatrix","INV_CAMERA_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"ProjectionMatrix","PROJECTION_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"ModelviewMatrix","MODELVIEW_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"InstanceID","INSTANCE_ID","",SLOT_TYPE_SCALAR,SLOT_IN},

	//material vertex out
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Vertex","VERTEX","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Normal","NORMAL","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Tangent","TANGENT","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Binormal","BINORMAL","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"UV","UV",".xy",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"UV2","UV2",".xy",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Color","COLOR.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Alpha","COLOR.a","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Var1","VAR1.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"Var2","VAR2.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"SpecExp","SPEC_EXP","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_VERTEX,"PointSize","POINT_SIZE","",SLOT_TYPE_SCALAR,SLOT_OUT},
	//pixel vertex in
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Vertex","VERTEX","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Position","POSITION.xyz","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Normal","IN_NORMAL","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Tangent","TANGENT","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Binormal","BINORMAL","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"UV","vec3(UV,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"UV2","vec3(UV2,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"UVScreen","vec3(SCREEN_UV,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"PointCoord","POINT_COORD","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Color","COLOR.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Alpha","COLOR.a","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"InvCameraMatrix","INV_CAMERA_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Var1","VAR1.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Var2","VAR2.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	//pixel vertex out
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Diffuse","DIFFUSE_OUT","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"DiffuseAlpha","ALPHA_OUT","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Specular","SPECULAR","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"SpecularExp","SPEC_EXP","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Emission","EMISSION","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Glow","GLOW","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"ShadeParam","SHADE_PARAM","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Normal","NORMAL","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"NormalMap","NORMALMAP","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"NormalMapDepth","NORMALMAP_DEPTH","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,"Discard","DISCARD",">0.5",SLOT_TYPE_SCALAR,SLOT_OUT},
	//light in
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"Normal","NORMAL","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"LightDir","LIGHT_DIR","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"LightDiffuse","LIGHT_DIFFUSE","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"LightSpecular","LIGHT_SPECULAR","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"EyeVec","EYE_VEC","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"Diffuse","DIFFUSE","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"Specular","SPECULAR","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"SpecExp","SPECULAR_EXP","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"ShadeParam","SHADE_PARAM","",SLOT_TYPE_SCALAR,SLOT_IN},
	//light out
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"Light","LIGHT","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_MATERIAL,SHADER_TYPE_LIGHT,"Shadow", "SHADOW", "",SLOT_TYPE_VEC, SLOT_OUT },
	//canvas item vertex in
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Vertex","vec3(SRC_VERTEX,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"UV","SRC_UV","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Color","SRC_COLOR.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Alpha","SRC_COLOR.a","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"WorldMatrix","WORLD_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"ExtraMatrix","EXTRA_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"ProjectionMatrix","PROJECTION_MATRIX","",SLOT_TYPE_XFORM,SLOT_IN},
	//canvas item vertex out
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Vertex","VERTEX",".xy",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"UV","UV",".xy",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Color","COLOR.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Alpha","COLOR.a","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Var1","VAR1.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"Var2","VAR2.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_VERTEX,"PointSize","POINT_SIZE","",SLOT_TYPE_SCALAR,SLOT_OUT},
	//canvas item fragment in
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Color","SRC_COLOR.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Alpha","SRC_COLOR.a","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"UV","vec3(UV,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"UVScreen","vec3(SCREEN_UV,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"TexPixelSize","vec3(TEXTURE_PIXEL_SIZE,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Var1","VAR1.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Var2","VAR2.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"PointCoord","POINT_COORD","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Position","POSITION","",SLOT_TYPE_VEC,SLOT_IN},
	//canvas item fragment out
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Color","COLOR.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Alpha","COLOR.a","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"Normal","NORMAL","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"NormalMap","NORMALMAP","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_FRAGMENT,"NormalMapDepth","NORMALMAP_DEPTH","",SLOT_TYPE_SCALAR,SLOT_OUT},
	//canvas item light in
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"Color","COLOR.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"Alpha","COLOR.a","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"Normal","NORMAL","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"UV","vec3(UV,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"LightColor","LIGHT_COLOR.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"LightAlpha","LIGHT_COLOR.a","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"LightHeight","LIGHT_HEIGHT","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"ShadowColor","LIGHT_SHADOW.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"ShadowAlpha","LIGHT_SHADOW.a","",SLOT_TYPE_SCALAR,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"TexPixelSize","vec3(TEXTURE_PIXEL_SIZE,0)","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"Var1","VAR1.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"Var2","VAR2.rgb","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"PointCoord","POINT_COORD","",SLOT_TYPE_VEC,SLOT_IN},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"Position","POSITION","",SLOT_TYPE_VEC,SLOT_IN},
	//canvas item light out
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"LightColor","LIGHT.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"LightAlpha","LIGHT.a","",SLOT_TYPE_SCALAR,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"ShadowColor","SHADOW.rgb","",SLOT_TYPE_VEC,SLOT_OUT},
	{MODE_CANVAS_ITEM,SHADER_TYPE_LIGHT,"ShadowAlpha","SHADOW.a","",SLOT_TYPE_SCALAR,SLOT_OUT},
	//end
	{MODE_MATERIAL,SHADER_TYPE_FRAGMENT,NULL,NULL,NULL,SLOT_TYPE_SCALAR,SLOT_OUT},



};

void ShaderGraph::get_input_output_node_slot_info(Mode p_mode, ShaderType  p_type, List<SlotInfo> *r_slots) {

	const InOutParamInfo* iop = &inout_param_info[0];
	while(iop->name) {
		if (p_mode==iop->shader_mode && p_type==iop->shader_type) {

			SlotInfo si;
			si.dir=iop->dir;
			si.name=iop->name;
			si.type=iop->slot_type;
			r_slots->push_back(si);
		}
		iop++;
	}
}


const ShaderGraph::NodeSlotInfo ShaderGraph::node_slot_info[]= {

		{NODE_SCALAR_CONST,{SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, //scalar constant
		{NODE_VEC_CONST,{SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, //vec3 constant
		{NODE_RGB_CONST,{SLOT_MAX},{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX}}, //rgb constant (shows a color picker instead)
		{NODE_XFORM_CONST,{SLOT_MAX},{SLOT_TYPE_XFORM,SLOT_MAX}}, // 4x4 matrix constant
		{NODE_TIME,{SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // time in seconds
		{NODE_SCREEN_TEX,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // screen texture sampler (takes UV) (only usable in fragment shader)
		{NODE_SCALAR_OP,{SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR,SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // scalar vs scalar op (mul,{SLOT_MAX},{SLOT_MAX}}, add,{SLOT_MAX},{SLOT_MAX}}, div,{SLOT_MAX},{SLOT_MAX}}, etc)
		{NODE_VEC_OP,{SLOT_TYPE_VEC,SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // scalar vs scalar op (mul,{SLOT_MAX},{SLOT_MAX}}, add,{SLOT_MAX},{SLOT_MAX}}, div,{SLOT_MAX},{SLOT_MAX}}, etc)
		{NODE_VEC_SCALAR_OP,{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // vec3 vs scalar op (mul,{SLOT_MAX},{SLOT_MAX}}, add,{SLOT_MAX},{SLOT_MAX}}, div,{SLOT_MAX},{SLOT_MAX}}, etc)
		{NODE_RGB_OP,{SLOT_TYPE_VEC,SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // vec3 vs scalar op (mul,{SLOT_MAX},{SLOT_MAX}}, add,{SLOT_MAX},{SLOT_MAX}}, div,{SLOT_MAX},{SLOT_MAX}}, etc)
		{NODE_XFORM_MULT,{SLOT_TYPE_XFORM,SLOT_TYPE_XFORM,SLOT_MAX},{SLOT_TYPE_XFORM,SLOT_MAX}}, // mat4 x mat4
		{NODE_XFORM_VEC_MULT,{SLOT_TYPE_XFORM,SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // mat4 x vec3 mult (with no-translation option)
		{NODE_XFORM_VEC_INV_MULT,{SLOT_TYPE_VEC,SLOT_TYPE_XFORM,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // mat4 x vec3 inverse mult (with no-translation option)
		{NODE_SCALAR_FUNC,{SLOT_TYPE_SCALAR,SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // scalar function (sin,{SLOT_MAX},{SLOT_MAX}}, cos,{SLOT_MAX},{SLOT_MAX}}, etc)
		{NODE_VEC_FUNC,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // vector function (normalize,{SLOT_MAX},{SLOT_MAX}}, negate,{SLOT_MAX},{SLOT_MAX}}, reciprocal,{SLOT_MAX},{SLOT_MAX}}, rgb2hsv,{SLOT_MAX},{SLOT_MAX}}, hsv2rgb,{SLOT_MAX},{SLOT_MAX}}, etc,{SLOT_MAX},{SLOT_MAX}}, etc)
		{NODE_VEC_LEN,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // vec3 length
		{NODE_DOT_PROD,{SLOT_TYPE_VEC,SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // vec3 . vec3 (dot product -> scalar output)
		{NODE_VEC_TO_SCALAR,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR}}, // 1 vec3 input,{SLOT_MAX},{SLOT_MAX}}, 3 scalar outputs
		{NODE_SCALAR_TO_VEC,{SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR},{SLOT_TYPE_VEC,SLOT_MAX}}, // 3 scalar input,{SLOT_MAX},{SLOT_MAX}}, 1 vec3 output
		{NODE_SCALAR_INTERP,{SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR,SLOT_TYPE_SCALAR},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // scalar interpolation (with optional curve)
		{NODE_VEC_INTERP,{SLOT_TYPE_VEC,SLOT_TYPE_VEC,SLOT_TYPE_SCALAR},{SLOT_TYPE_VEC,SLOT_MAX}}, // vec3 interpolation  (with optional curve)
		{NODE_COLOR_RAMP,{SLOT_TYPE_SCALAR,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX}}, // vec3 interpolation  (with optional curve)
		{NODE_CURVE_MAP,{SLOT_TYPE_SCALAR,SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // vec3 interpolation  (with optional curve)
		{NODE_SCALAR_INPUT,{SLOT_MAX},{SLOT_TYPE_SCALAR,SLOT_MAX}}, // scalar uniform (assignable in material)
		{NODE_VEC_INPUT,{SLOT_MAX},{SLOT_TYPE_VEC,SLOT_MAX}}, // vec3 uniform (assignable in material)
		{NODE_RGB_INPUT,{SLOT_MAX},{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX}}, // color uniform (assignable in material)
		{NODE_XFORM_INPUT,{SLOT_MAX},{SLOT_TYPE_XFORM,SLOT_MAX}}, // mat4 uniform (assignable in material)
		{NODE_TEXTURE_INPUT,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX}}, // texture input (assignable in material)
		{NODE_CUBEMAP_INPUT,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX}}, // cubemap input (assignable in material)
		{NODE_DEFAULT_TEXTURE,{SLOT_TYPE_VEC,SLOT_MAX},{SLOT_TYPE_VEC,SLOT_TYPE_SCALAR,SLOT_MAX}}, // cubemap input (assignable in material)
		{NODE_COMMENT,{SLOT_MAX},{SLOT_MAX}}, // comment
		{NODE_TYPE_MAX,{SLOT_MAX},{SLOT_MAX}}
};

int ShaderGraph::get_node_input_slot_count(Mode p_mode, ShaderType  p_shader_type,NodeType p_type) {

	if (p_type==NODE_INPUT || p_type==NODE_OUTPUT) {

		const InOutParamInfo* iop = &inout_param_info[0];
		int pc=0;
		while(iop->name) {
			if (p_mode==iop->shader_mode && p_shader_type==iop->shader_type) {

				if (iop->dir==SLOT_OUT)
					pc++;
			}
			iop++;
		}
		return pc;
	} else if (p_type==NODE_VEC_TO_XFORM){
		return 4;
	} else if (p_type==NODE_XFORM_TO_VEC){
		return 1;
	} else {

		const NodeSlotInfo*nsi=&node_slot_info[0];
		while(nsi->type!=NODE_TYPE_MAX) {

			if (nsi->type==p_type) {
				int pc=0;
				for(int i=0;i<NodeSlotInfo::MAX_INS;i++) {
					if (nsi->ins[i]==SLOT_MAX)
						break;
					pc++;
				}
				return pc;
			}

			nsi++;
		}

		return 0;

	}
}

int ShaderGraph::get_node_output_slot_count(Mode p_mode, ShaderType  p_shader_type,NodeType p_type){

	if (p_type==NODE_INPUT || p_type==NODE_OUTPUT) {

		const InOutParamInfo* iop = &inout_param_info[0];
		int pc=0;
		while(iop->name) {
			if (p_mode==iop->shader_mode && p_shader_type==iop->shader_type) {

				if (iop->dir==SLOT_IN)
					pc++;
			}
			iop++;
		}
		return pc;
	} else if (p_type==NODE_VEC_TO_XFORM){
		return 1;
	} else if (p_type==NODE_XFORM_TO_VEC){
		return 4;
	} else {

		const NodeSlotInfo*nsi=&node_slot_info[0];
		while(nsi->type!=NODE_TYPE_MAX) {

			if (nsi->type==p_type) {
				int pc=0;
				for(int i=0;i<NodeSlotInfo::MAX_OUTS;i++) {
					if (nsi->outs[i]==SLOT_MAX)
						break;
					pc++;
				}
				return pc;
			}

			nsi++;
		}

		return 0;

	}
}
ShaderGraph::SlotType ShaderGraph::get_node_input_slot_type(Mode p_mode, ShaderType  p_shader_type,NodeType p_type,int p_idx){

	if (p_type==NODE_INPUT || p_type==NODE_OUTPUT) {

		const InOutParamInfo* iop = &inout_param_info[0];
		int pc=0;
		while(iop->name) {
			if (p_mode==iop->shader_mode && p_shader_type==iop->shader_type) {

				if (iop->dir==SLOT_OUT) {
					if (pc==p_idx)
						return iop->slot_type;
					pc++;
				}
			}
			iop++;
		}
		ERR_FAIL_V(SLOT_MAX);
	} else if (p_type==NODE_VEC_TO_XFORM){
		return SLOT_TYPE_VEC;
	} else if (p_type==NODE_XFORM_TO_VEC){
		return SLOT_TYPE_XFORM;
	} else {

		const NodeSlotInfo*nsi=&node_slot_info[0];
		while(nsi->type!=NODE_TYPE_MAX) {

			if (nsi->type==p_type) {
				for(int i=0;i<NodeSlotInfo::MAX_INS;i++) {

					if (nsi->ins[i]==SLOT_MAX)
						break;
					if (i==p_idx)
						return nsi->ins[i];
				}
			}

			nsi++;
		}

		ERR_FAIL_V(SLOT_MAX);

	}
}
ShaderGraph::SlotType ShaderGraph::get_node_output_slot_type(Mode p_mode, ShaderType  p_shader_type,NodeType p_type,int p_idx){

	if (p_type==NODE_INPUT || p_type==NODE_OUTPUT) {

		const InOutParamInfo* iop = &inout_param_info[0];
		int pc=0;
		while(iop->name) {
			if (p_mode==iop->shader_mode && p_shader_type==iop->shader_type) {

				if (iop->dir==SLOT_IN) {
					if (pc==p_idx)
						return iop->slot_type;
					pc++;
				}
			}
			iop++;
		}
		ERR_FAIL_V(SLOT_MAX);
	} else if (p_type==NODE_VEC_TO_XFORM){
		return SLOT_TYPE_XFORM;
	} else if (p_type==NODE_XFORM_TO_VEC){
		return SLOT_TYPE_VEC;
	} else {

		const NodeSlotInfo*nsi=&node_slot_info[0];
		while(nsi->type!=NODE_TYPE_MAX) {

			if (nsi->type==p_type) {
				for(int i=0;i<NodeSlotInfo::MAX_OUTS;i++) {
					if (nsi->outs[i]==SLOT_MAX)
						break;
					if (i==p_idx)
						return nsi->outs[i];
				}
			}

			nsi++;
		}

		ERR_FAIL_V(SLOT_MAX);
	}
}





void ShaderGraph::_update_shader() {


	String code[3];

	List<StringName> names;
	get_default_texture_param_list(&names);

	for (List<StringName>::Element *E=names.front();E;E=E->next()) {

		set_default_texture_param(E->get(),Ref<Texture>());
	}


	for(int i=0;i<3;i++) {

		int idx=0;
		for (Map<int,Node>::Element *E=shader[i].node_map.front();E;E=E->next()) {

			E->get().sort_order=idx++;
		}
		//simple method for graph solving using bubblesort derived algorithm
		int iters=0;
		int iter_max=shader[i].node_map.size()*shader[i].node_map.size();

		while(true) {
			if (iters>iter_max)
				break;

			int swaps=0;
			for (Map<int,Node>::Element *E=shader[i].node_map.front();E;E=E->next()) {

				for(Map<int,SourceSlot>::Element *F=E->get().connections.front();F;F=F->next()) {

					//this is kinda slow, could be sped up
					Map<int,Node>::Element *G = shader[i].node_map.find(F->get().id);
					ERR_FAIL_COND(!G);
					if (G->get().sort_order > E->get().sort_order) {

						SWAP(G->get().sort_order,E->get().sort_order);
						swaps++;
					}
				}
			}

			iters++;
			if (swaps==0) {
				iters=0;
				break;
			}
		}

		if (iters>0) {

			shader[i].error=GRAPH_ERROR_CYCLIC;
			continue;
		}

		Vector<Node*> order;
		order.resize(shader[i].node_map.size());

		for (Map<int,Node>::Element *E=shader[i].node_map.front();E;E=E->next()) {

			order[E->get().sort_order]=&E->get();
		}

		//generate code for the ordered graph
		bool failed=false;

		if (i==SHADER_TYPE_FRAGMENT && get_mode()==MODE_MATERIAL) {
			code[i]+="vec3 DIFFUSE_OUT=vec3(0,0,0);\n";
			code[i]+="float ALPHA_OUT=0;\n";
		}


		Map<String,String> inputs_xlate;
		Map<String,String> input_names_xlate;
		Set<String> inputs_used;

		for(int j=0;j<order.size();j++) {

			Node *n=order[j];
			if (n->type==NODE_INPUT) {

				const InOutParamInfo* iop = &inout_param_info[0];
				int idx=0;
				while(iop->name) {
					if (get_mode()==iop->shader_mode && i==iop->shader_type && SLOT_IN==iop->dir) {

						const char *typestr[4]={"float","vec3","mat4","texture"};

						String vname=("nd"+itos(n->id)+"sl"+itos(idx));
						inputs_xlate[vname]=String(typestr[iop->slot_type])+" "+vname+"="+iop->variable+";\n";
						input_names_xlate[vname]=iop->variable;
						idx++;
					}
					iop++;
				}

			} else if (n->type==NODE_OUTPUT) {


				bool use_alpha=false;
				const InOutParamInfo* iop = &inout_param_info[0];
				int idx=0;
				while(iop->name) {
					if (get_mode()==iop->shader_mode && i==iop->shader_type && SLOT_OUT==iop->dir) {

						if (n->connections.has(idx)) {
							String iname=("nd"+itos(n->connections[idx].id)+"sl"+itos(n->connections[idx].slot));
							if (node_get_type(ShaderType(i),n->connections[idx].id)==NODE_INPUT)
								inputs_used.insert(iname);
							code[i]+=String(iop->variable)+"="+iname+String(iop->postfix)+";\n";
							if (i==SHADER_TYPE_FRAGMENT && get_mode()==MODE_MATERIAL && String(iop->name)=="DiffuseAlpha")
								use_alpha=true;
						}
						idx++;
					}
					iop++;
				}

				if (i==SHADER_TYPE_FRAGMENT && get_mode()==MODE_MATERIAL) {

					if (use_alpha) {
						code[i]+="DIFFUSE_ALPHA=vec4(DIFFUSE_OUT,ALPHA_OUT);\n";
					} else {
						code[i]+="DIFFUSE=DIFFUSE_OUT;\n";
					}
				}

			} else {
				Vector<String> inputs;
				int max = get_node_input_slot_count(get_mode(),ShaderType(i),n->type);
				for(int k=0;k<max;k++) {
					String iname;
					if (!n->connections.has(k)) {
						iname="nd"+itos(n->id)+"sl"+itos(k)+"def";
					} else {
						iname="nd"+itos(n->connections[k].id)+"sl"+itos(n->connections[k].slot);
						if (node_get_type(ShaderType(i),n->connections[k].id)==NODE_INPUT) {
							inputs_used.insert(iname);
						}

					}
					inputs.push_back(iname);
				}

				if (failed)
					break;

				if (n->type==NODE_TEXTURE_INPUT || n->type==NODE_CUBEMAP_INPUT) {

					set_default_texture_param(n->param1,n->param2);

				}
				_add_node_code(ShaderType(i),n,inputs,code[i]);
			}

		}

		if (failed)
			continue;


		for(Set<String>::Element *E=inputs_used.front();E;E=E->next()) {

			ERR_CONTINUE( !inputs_xlate.has(E->get()));
			code[i]=inputs_xlate[E->get()]+code[i];
			String name=input_names_xlate[E->get()];

			if (i==SHADER_TYPE_VERTEX && get_mode()==MODE_MATERIAL) {
				if (name==("SRC_COLOR"))
					code[i]="vec3 SRC_COLOR=COLOR.rgb;\n"+code[i];
				if (name==("SRC_ALPHA"))
					code[i]="float SRC_ALPHA=COLOR.a;\n"+code[i];
				if (name==("SRC_UV"))
					code[i]="vec3 SRC_UV=vec3(UV,0);\n"+code[i];
				if (name==("SRC_UV2"))
					code[i]="float SRC_UV2=vec3(UV2,0);\n"+code[i];
			} else if (i==SHADER_TYPE_FRAGMENT && get_mode()==MODE_MATERIAL) {
				if (name==("IN_NORMAL"))
					code[i]="vec3 IN_NORMAL=NORMAL;\n"+code[i];
			} else if (i==SHADER_TYPE_VERTEX && get_mode()==MODE_CANVAS_ITEM) {
				if (name==("SRC_COLOR"))
					code[i]="vec3 SRC_COLOR=COLOR.rgb;\n"+code[i];
				if (name==("SRC_UV"))
					code[i]="vec3 SRC_UV=vec3(UV,0);\n"+code[i];
			}

		}



		shader[i].error=GRAPH_OK;

	}

	bool all_ok=true;
	for(int i=0;i<3;i++) {
		if (shader[i].error!=GRAPH_OK)
			all_ok=false;
	}

	/*print_line("VERTEX: \n"+code[0]);
	print_line("FRAGMENT: \n"+code[1]);
	print_line("LIGHT: \n"+code[2]);*/

	if (all_ok) {
		set_code(code[0],code[1],code[2]);
	}
	//do shader here

	_pending_update_shader=false;
	emit_signal(SceneStringNames::get_singleton()->updated);
}

void ShaderGraph::_plot_curve(const Vector2& p_a,const Vector2& p_b,const Vector2& p_c,const Vector2& p_d,uint8_t* p_heights,bool *p_useds) {

	float geometry[4][4];
	float tmp1[4][4];
	float tmp2[4][4];
	float deltas[4][4];
	double x, dx, dx2, dx3;
	double y, dy, dy2, dy3;
	double d, d2, d3;
	int lastx, lasty;
	int newx, newy;
	int ntimes;
	int i,j;

	int xmax=255;
	int ymax=255;

	/* construct the geometry matrix from the segment */
	for (i = 0; i < 4; i++)	{
		geometry[i][2] = 0;
		geometry[i][3] = 0;
	}

	geometry[0][0] = (p_a[0] * xmax);
	geometry[1][0] = (p_b[0] * xmax);
	geometry[2][0] = (p_c[0] * xmax);
	geometry[3][0] = (p_d[0] * xmax);

	geometry[0][1] = (p_a[1] * ymax);
	geometry[1][1] = (p_b[1] * ymax);
	geometry[2][1] = (p_c[1] * ymax);
	geometry[3][1] = (p_d[1] * ymax);

	/* subdivide the curve ntimes (1000) times */
	ntimes = 4 * xmax;
	/* ntimes can be adjusted to give a finer or coarser curve */
	d = 1.0 / ntimes;
	d2 = d * d;
	d3 = d * d * d;

	/* construct a temporary matrix for determining the forward differencing deltas */
	tmp2[0][0] = 0;     tmp2[0][1] = 0;     tmp2[0][2] = 0;    tmp2[0][3] = 1;
	tmp2[1][0] = d3;    tmp2[1][1] = d2;    tmp2[1][2] = d;    tmp2[1][3] = 0;
	tmp2[2][0] = 6*d3;  tmp2[2][1] = 2*d2;  tmp2[2][2] = 0;    tmp2[2][3] = 0;
	tmp2[3][0] = 6*d3;  tmp2[3][1] = 0;     tmp2[3][2] = 0;    tmp2[3][3] = 0;

	/* compose the basis and geometry matrices */

	static const float CR_basis[4][4] = {
	  { -0.5,  1.5, -1.5,  0.5 },
	  {  1.0, -2.5,  2.0, -0.5 },
	  { -0.5,  0.0,  0.5,  0.0 },
	  {  0.0,  1.0,  0.0,  0.0 },
	};

	for (i = 0; i < 4; i++)
		{
		  for (j = 0; j < 4; j++)
		{
		  tmp1[i][j] = (CR_basis[i][0] * geometry[0][j] +
				  CR_basis[i][1] * geometry[1][j] +
				  CR_basis[i][2] * geometry[2][j] +
				  CR_basis[i][3] * geometry[3][j]);
		}
		}
	/* compose the above results to get the deltas matrix */

	for (i = 0; i < 4; i++)
		{
		  for (j = 0; j < 4; j++)
		{
		  deltas[i][j] = (tmp2[i][0] * tmp1[0][j] +
				  tmp2[i][1] * tmp1[1][j] +
				  tmp2[i][2] * tmp1[2][j] +
				  tmp2[i][3] * tmp1[3][j]);
		}
		}


	/* extract the x deltas */
	x = deltas[0][0];
	dx = deltas[1][0];
	dx2 = deltas[2][0];
	dx3 = deltas[3][0];

	/* extract the y deltas */
	y = deltas[0][1];
	dy = deltas[1][1];
	dy2 = deltas[2][1];
	dy3 = deltas[3][1];


	lastx = CLAMP (x, 0, xmax);
	lasty = CLAMP (y, 0, ymax);

	p_heights[lastx] = lasty;
	p_useds[lastx] = true;

	/* loop over the curve */
	for (i = 0; i < ntimes; i++)
	{
		/* increment the x values */
		x += dx;
		dx += dx2;
		dx2 += dx3;

		/* increment the y values */
		y += dy;
		dy += dy2;
		dy2 += dy3;

		newx = CLAMP ((Math::round (x)), 0, xmax);
		newy = CLAMP ((Math::round (y)), 0, ymax);

		/* if this point is different than the last one...then draw it */
		if ((lastx != newx) || (lasty != newy))
		{
			p_useds[newx]=true;
			p_heights[newx]=newy;
		}

		lastx = newx;
		lasty = newy;
	}
}


void ShaderGraph::_add_node_code(ShaderType p_type,Node *p_node,const Vector<String>& p_inputs,String& code) {


	const char *typestr[4]={"float","vec3","mat4","texture"};
#define OUTNAME(id, slot) (String(typestr[get_node_output_slot_type(get_mode(), p_type, p_node->type, slot)]) + " " + ("nd" + itos(id) + "sl" + itos(slot)))
#define OUTVAR(id, slot) ("nd" + itos(id) + "sl" + itos(slot))
#define DEF_VEC(slot)                                                              \
	if (p_inputs[slot].ends_with("def")) {                                         \
		Vector3 v = p_node->defaults[slot];                                        \
		code += String(typestr[1]) + " " + p_inputs[slot] + "=vec3(" + v + ");\n"; \
	}
#define DEF_SCALAR(slot)                                                           \
	if (p_inputs[slot].ends_with("def")) {                                         \
		double v = p_node->defaults[slot];                                         \
		code += String(typestr[0]) + " " + p_inputs[slot] + "=" + rtos(v) + ";\n"; \
	}
#define DEF_COLOR(slot)                                                                                                              \
	if (p_inputs[slot].ends_with("def")) {                                                                                           \
		Color col = p_node->defaults[slot];                                                                                          \
		code += String(typestr[1]) + " " + p_inputs[slot] + "=vec3(" + rtos(col.r) + "," + rtos(col.g) + "," + rtos(col.b) + ");\n"; \
	}
#define DEF_MATRIX(slot)                                                                                                                             \
	if (p_inputs[slot].ends_with("def")) {                                                                                                           \
		Transform xf = p_node->defaults[slot];                                                                                                       \
		code += String(typestr[2]) + " " + p_inputs[slot] + "=mat4(\n";                                                                              \
		code += "\tvec4(vec3(" + rtos(xf.basis.get_axis(0).x) + "," + rtos(xf.basis.get_axis(0).y) + "," + rtos(xf.basis.get_axis(0).z) + "),0),\n"; \
		code += "\tvec4(vec3(" + rtos(xf.basis.get_axis(1).x) + "," + rtos(xf.basis.get_axis(1).y) + "," + rtos(xf.basis.get_axis(1).z) + "),0),\n"; \
		code += "\tvec4(vec3(" + rtos(xf.basis.get_axis(2).x) + "," + rtos(xf.basis.get_axis(2).y) + "," + rtos(xf.basis.get_axis(2).z) + "),0),\n"; \
		code += "\tvec4(vec3(" + rtos(xf.origin.x) + "," + rtos(xf.origin.y) + "," + rtos(xf.origin.z) + "),1)\n";                                   \
		code += ");\n";                                                                                                                              \
	}

	switch(p_node->type) {

		case NODE_INPUT: {


		}break;
		case NODE_SCALAR_CONST: {

			double scalar = p_node->param1;
			code+=OUTNAME(p_node->id,0)+"="+rtos(scalar)+";\n";
		}break;
		case NODE_VEC_CONST: {
			Vector3 vec = p_node->param1;
			code+=OUTNAME(p_node->id,0)+"=vec3("+rtos(vec.x)+","+rtos(vec.y)+","+rtos(vec.z)+");\n";
		}break;
		case NODE_RGB_CONST: {
			Color col = p_node->param1;
			code+=OUTNAME(p_node->id,0)+"=vec3("+rtos(col.r)+","+rtos(col.g)+","+rtos(col.b)+");\n";
			code+=OUTNAME(p_node->id,1)+"="+rtos(col.a)+";\n";
		}break;
		case NODE_XFORM_CONST: {

			Transform xf = p_node->param1;
			code+=OUTNAME(p_node->id,0)+"=mat4(\n";
			code+="\tvec4(vec3("+rtos(xf.basis.get_axis(0).x)+","+rtos(xf.basis.get_axis(0).y)+","+rtos(xf.basis.get_axis(0).z)+"),0),\n";
			code+="\tvec4(vec3("+rtos(xf.basis.get_axis(1).x)+","+rtos(xf.basis.get_axis(1).y)+","+rtos(xf.basis.get_axis(1).z)+"),0),\n";
			code+="\tvec4(vec3("+rtos(xf.basis.get_axis(2).x)+","+rtos(xf.basis.get_axis(2).y)+","+rtos(xf.basis.get_axis(2).z)+"),0),\n";
			code+="\tvec4(vec3("+rtos(xf.origin.x)+","+rtos(xf.origin.y)+","+rtos(xf.origin.z)+"),1)\n";
			code+=");";

		}break;
		case NODE_TIME: {
			code+=OUTNAME(p_node->id,0)+"=TIME;\n";
		}break;
		case NODE_SCREEN_TEX: {
			DEF_VEC(0);
			code+=OUTNAME(p_node->id,0)+"=texscreen("+p_inputs[0]+".xy);\n";
		}break;
		case NODE_SCALAR_OP: {
			DEF_SCALAR(0);
			DEF_SCALAR(1);
			int op = p_node->param1;
			String optxt;
			switch(op) {

				case SCALAR_OP_ADD: optxt = p_inputs[0]+"+"+p_inputs[1]+";"; break;
				case SCALAR_OP_SUB: optxt = p_inputs[0]+"-"+p_inputs[1]+";"; break;
				case SCALAR_OP_MUL: optxt = p_inputs[0]+"*"+p_inputs[1]+";"; break;
				case SCALAR_OP_DIV: optxt = p_inputs[0]+"/"+p_inputs[1]+";"; break;
				case SCALAR_OP_MOD: optxt = "mod("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case SCALAR_OP_POW: optxt = "pow("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case SCALAR_OP_MAX: optxt = "max("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case SCALAR_OP_MIN: optxt = "min("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case SCALAR_OP_ATAN2: optxt = "atan2("+p_inputs[0]+","+p_inputs[1]+");"; break;

			}
			code+=OUTNAME(p_node->id,0)+"="+optxt+"\n";

		}break;
		case NODE_VEC_OP: {
			DEF_VEC(0);
			DEF_VEC(1);
			int op = p_node->param1;
			String optxt;
			switch(op) {
				case VEC_OP_ADD: optxt = p_inputs[0]+"+"+p_inputs[1]+";"; break;
				case VEC_OP_SUB: optxt = p_inputs[0]+"-"+p_inputs[1]+";"; break;
				case VEC_OP_MUL: optxt = p_inputs[0]+"*"+p_inputs[1]+";"; break;
				case VEC_OP_DIV: optxt = p_inputs[0]+"/"+p_inputs[1]+";"; break;
				case VEC_OP_MOD: optxt = "mod("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case VEC_OP_POW: optxt = "pow("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case VEC_OP_MAX: optxt = "max("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case VEC_OP_MIN: optxt = "min("+p_inputs[0]+","+p_inputs[1]+");"; break;
				case VEC_OP_CROSS: optxt = "cross("+p_inputs[0]+","+p_inputs[1]+");"; break;
			}
			code+=OUTNAME(p_node->id,0)+"="+optxt+"\n";

		}break;
		case NODE_VEC_SCALAR_OP: {
			DEF_VEC(0);
			DEF_SCALAR(1);
			int op = p_node->param1;
			String optxt;
			switch(op) {
				case VEC_SCALAR_OP_MUL: optxt = p_inputs[0]+"*"+p_inputs[1]+";"; break;
				case VEC_SCALAR_OP_DIV: optxt = p_inputs[0]+"/"+p_inputs[1]+";"; break;
				case VEC_SCALAR_OP_POW: optxt = "pow("+p_inputs[0]+","+p_inputs[1]+");"; break;
			}
			code+=OUTNAME(p_node->id,0)+"="+optxt+"\n";

		}break;
		case NODE_RGB_OP: {
			DEF_COLOR(0);
			DEF_COLOR(1);

			int op = p_node->param1;
			static const char*axisn[3]={"x","y","z"};
			switch(op) {
				case RGB_OP_SCREEN: {

					code += OUTNAME(p_node->id,0)+"=vec3(1.0)-(vec3(1.0)-"+p_inputs[0]+")*(vec3(1.0)-"+p_inputs[1]+");\n";
				} break;
				case RGB_OP_DIFFERENCE: {

					code += OUTNAME(p_node->id,0)+"=abs("+p_inputs[0]+"-"+p_inputs[1]+");\n";
				} break;
				case RGB_OP_DARKEN: {

					code += OUTNAME(p_node->id,0)+"=min("+p_inputs[0]+","+p_inputs[1]+");\n";
				} break;
				case RGB_OP_LIGHTEN: {

					code += OUTNAME(p_node->id,0)+"=max("+p_inputs[0]+","+p_inputs[1]+");\n";

				} break;
				case RGB_OP_OVERLAY: {

					code += OUTNAME(p_node->id,0)+";\n";
					for(int i=0;i<3;i++) {
						code += "{\n";
						code += "\tfloat base="+p_inputs[0]+"."+axisn[i]+";\n";
						code += "\tfloat blend="+p_inputs[1]+"."+axisn[i]+";\n";
						code += "\tif (base < 0.5) {\n";
						code += "\t\t"+OUTVAR(p_node->id,0)+"."+axisn[i]+" = 2.0 * base * blend;\n";
						code += "\t} else {\n";
						code += "\t\t"+OUTVAR(p_node->id,0)+"."+axisn[i]+" = 1.0 - 2.0 * (1.0 - blend) * (1.0 - base);\n";
						code += "\t}\n";
						code += "}\n";
					}

				} break;
				case RGB_OP_DODGE: {

					code += OUTNAME(p_node->id,0)+"=("+p_inputs[0]+")/(vec3(1.0)-"+p_inputs[1]+");\n";

				} break;
				case RGB_OP_BURN: {

					code += OUTNAME(p_node->id,0)+"=vec3(1.0)-(vec3(1.0)-"+p_inputs[0]+")/("+p_inputs[1]+");\n";
				} break;
				case RGB_OP_SOFT_LIGHT: {

					code += OUTNAME(p_node->id,0)+";\n";
					for(int i=0;i<3;i++) {
						code += "{\n";
						code += "\tfloat base="+p_inputs[0]+"."+axisn[i]+";\n";
						code += "\tfloat blend="+p_inputs[1]+"."+axisn[i]+";\n";
						code += "\tif (base < 0.5) {\n";
						code += "\t\t"+OUTVAR(p_node->id,0)+"."+axisn[i]+" = (base * (blend+0.5));\n";
						code += "\t} else {\n";
						code += "\t\t"+OUTVAR(p_node->id,0)+"."+axisn[i]+" = (1 - (1-base) * (1-(blend-0.5)));\n";
						code += "\t}\n";
						code += "}\n";
					}

				} break;
				case RGB_OP_HARD_LIGHT: {

					code += OUTNAME(p_node->id,0)+";\n";
					for(int i=0;i<3;i++) {
						code += "{\n";
						code += "\tfloat base="+p_inputs[0]+"."+axisn[i]+";\n";
						code += "\tfloat blend="+p_inputs[1]+"."+axisn[i]+";\n";
						code += "\tif (base < 0.5) {\n";
						code += "\t\t"+OUTVAR(p_node->id,0)+"."+axisn[i]+" = (base * (2*blend));\n";
						code += "\t} else {\n";
						code += "\t\t"+OUTVAR(p_node->id,0)+"."+axisn[i]+" = (1 - (1-base) * (1-2*(blend-0.5)));\n";
						code += "\t}\n";
						code += "}\n";
					}

				} break;
			}
		}break;
		case NODE_XFORM_MULT: {
			DEF_MATRIX(0);
			DEF_MATRIX(1);

			code += OUTNAME(p_node->id,0)+"="+p_inputs[0]+"*"+p_inputs[1]+";\n";

		}break;
		case NODE_XFORM_VEC_MULT: {
			DEF_MATRIX(0);
			DEF_VEC(1);

			bool no_translation = p_node->param1;
			if (no_translation) {
				code += OUTNAME(p_node->id,0)+"=("+p_inputs[0]+"*vec4("+p_inputs[1]+",0)).xyz;\n";
			} else {
				code += OUTNAME(p_node->id,0)+"=("+p_inputs[0]+"*vec4("+p_inputs[1]+",1)).xyz;\n";
			}

		}break;
		case NODE_XFORM_VEC_INV_MULT: {
			DEF_VEC(0);
			DEF_MATRIX(1);
			bool no_translation = p_node->param1;
			if (no_translation) {
				code += OUTNAME(p_node->id,0)+"=("+p_inputs[1]+"*vec4("+p_inputs[0]+",0)).xyz;\n";
			} else {
				code += OUTNAME(p_node->id,0)+"=("+p_inputs[1]+"*vec4("+p_inputs[0]+",1)).xyz;\n";
			}
		}break;
		case NODE_SCALAR_FUNC: {
			DEF_SCALAR(0);
			static const char*scalar_func_id[SCALAR_MAX_FUNC]={
				"sin($)",
				"cos($)",
				"tan($)",
				"asin($)",
				"acos($)",
				"atan($)",
				"sinh($)",
				"cosh($)",
				"tanh($)",
				"log($)",
				"exp($)",
				"sqrt($)",
				"abs($)",
				"sign($)",
				"floor($)",
				"round($)",
				"ceil($)",
				"fract($)",
				"min(max($,0),1)",
				"-($)",
			};

			int func = p_node->param1;
			ERR_FAIL_INDEX(func,SCALAR_MAX_FUNC);
			code += OUTNAME(p_node->id,0)+"="+String(scalar_func_id[func]).replace("$",p_inputs[0])+";\n";

		} break;
		case NODE_VEC_FUNC: {
			DEF_VEC(0);
			static const char*vec_func_id[VEC_MAX_FUNC]={
				"normalize($)",
				"max(min($,vec3(1,1,1)),vec3(0,0,0))",
				"-($)",
				"1.0/($)",
				"",
				"",
			};


			int func = p_node->param1;
			ERR_FAIL_INDEX(func,VEC_MAX_FUNC);
			if (func==VEC_FUNC_RGB2HSV) {
				code += OUTNAME(p_node->id,0)+";\n";
				code+="{\n";
				code+="\tvec3 c = "+p_inputs[0]+";\n";
				code+="\tvec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);\n";
				code+="\tvec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));\n";
				code+="\tvec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));\n";
				code+="\tfloat d = q.x - min(q.w, q.y);\n";
				code+="\tfloat e = 1.0e-10;\n";
				code+="\t"+OUTVAR(p_node->id,0)+"=vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);\n";
				code+="}\n";
			} else if (func==VEC_FUNC_HSV2RGB) {
				code += OUTNAME(p_node->id,0)+";\n";
				code+="{\n";
				code+="\tvec3 c = "+p_inputs[0]+";\n";
				code+="\tvec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\n";
				code+="\tvec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n";
				code+="\t"+OUTVAR(p_node->id,0)+"=c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n";
				code+="}\n";

			} else {
				code += OUTNAME(p_node->id,0)+"="+String(vec_func_id[func]).replace("$",p_inputs[0])+";\n";
			}
		}break;
		case NODE_VEC_LEN: {
			DEF_VEC(0);

			code += OUTNAME(p_node->id,0)+"=length("+p_inputs[0]+");\n";

		}break;
		case NODE_DOT_PROD: {
			DEF_VEC(0);
			DEF_VEC(1);
			code += OUTNAME(p_node->id,0)+"=dot("+p_inputs[1]+","+p_inputs[0]+");\n";

		}break;
		case NODE_VEC_TO_SCALAR: {
			DEF_VEC(0);
			code += OUTNAME(p_node->id,0)+"="+p_inputs[0]+".x;\n";
			code += OUTNAME(p_node->id,1)+"="+p_inputs[0]+".y;\n";
			code += OUTNAME(p_node->id,2)+"="+p_inputs[0]+".z;\n";

		}break;
		case NODE_SCALAR_TO_VEC: {
			DEF_SCALAR(0);
			DEF_SCALAR(1);
			DEF_SCALAR(2);
			code += OUTNAME(p_node->id,0)+"=vec3("+p_inputs[0]+","+p_inputs[1]+","+p_inputs[2]+""+");\n";

		}break;
		case NODE_VEC_TO_XFORM: {
			DEF_VEC(0);
			DEF_VEC(1);
			DEF_VEC(2);
			DEF_VEC(3);
			code += OUTNAME(p_node->id, 0) + "=mat4(" +
				"vec4(" + p_inputs[0] + ".x," + p_inputs[0] + ".y," + p_inputs[0] + ".z, 0.0),"
				"vec4(" + p_inputs[1] + ".x," + p_inputs[1] + ".y," + p_inputs[1] + ".z, 0.0),"
				"vec4(" + p_inputs[2] + ".x," + p_inputs[2] + ".y," + p_inputs[2] + ".z, 0.0),"
				"vec4(" + p_inputs[3] + ".x," + p_inputs[3] + ".y," + p_inputs[3] + ".z, 1.0));\n";

		}break;
		case NODE_XFORM_TO_VEC: {
			DEF_MATRIX(0);
			code += OUTNAME(p_node->id, 0) + ";\n";
			code += OUTNAME(p_node->id, 1) + ";\n";
			code += OUTNAME(p_node->id, 2) + ";\n";
			code += OUTNAME(p_node->id, 3) + ";\n";
			code += "{\n";
			code += "\tvec4 xform_row_01=" + p_inputs[0] + ".x;\n";
			code += "\tvec4 xform_row_02=" + p_inputs[0] + ".y;\n";
			code += "\tvec4 xform_row_03=" + p_inputs[0] + ".z;\n";
			code += "\tvec4 xform_row_04=" + p_inputs[0] + ".w;\n";
			code += "\t" + OUTVAR(p_node->id, 0) + "=vec3(xform_row_01.x, xform_row_01.y, xform_row_01.z);\n";
			code += "\t" + OUTVAR(p_node->id, 1) + "=vec3(xform_row_02.x, xform_row_02.y, xform_row_02.z);\n";
			code += "\t" + OUTVAR(p_node->id, 2) + "=vec3(xform_row_03.x, xform_row_03.y, xform_row_03.z);\n";
			code += "\t" + OUTVAR(p_node->id, 3) + "=vec3(xform_row_04.x, xform_row_04.y, xform_row_04.z);\n";
			code += "}\n";
		}break;
		case NODE_SCALAR_INTERP: {
			DEF_SCALAR(0);
			DEF_SCALAR(1);
			DEF_SCALAR(2);

			code += OUTNAME(p_node->id,0)+"=mix("+p_inputs[0]+","+p_inputs[1]+","+p_inputs[2]+");\n";

		}break;
		case NODE_VEC_INTERP: {
			DEF_VEC(0);
			DEF_VEC(1);
			DEF_SCALAR(2);
			code += OUTNAME(p_node->id,0)+"=mix("+p_inputs[0]+","+p_inputs[1]+","+p_inputs[2]+");\n";

		}break;
		case NODE_COLOR_RAMP: {
			DEF_SCALAR(0);

			static const int color_ramp_len=512;
			PoolVector<uint8_t> cramp;
			cramp.resize(color_ramp_len*4);
			{

				PoolVector<Color> colors=p_node->param1;
				PoolVector<real_t> offsets=p_node->param2;
				int cc =colors.size();
				PoolVector<uint8_t>::Write crw = cramp.write();
				PoolVector<Color>::Read cr = colors.read();
				PoolVector<real_t>::Read ofr = offsets.read();

				int at=0;
				Color color_at(0,0,0,1);
				for(int i=0;i<=cc;i++) {

					int pos;
					Color to;
					if (i==cc) {
						if (at==color_ramp_len)
							break;
						pos=color_ramp_len;
						to=Color(1,1,1,1);
					} else {
						to=cr[i];
						pos= MIN(ofr[i]*color_ramp_len,color_ramp_len);
					}
					for(int j=at;j<pos;j++) {
						float t = (j-at)/float(pos-at);
						Color c = color_at.linear_interpolate(to,t);
						crw[j*4+0]=Math::fast_ftoi( CLAMP(c.r*255.0,0,255) );
						crw[j*4+1]=Math::fast_ftoi( CLAMP(c.g*255.0,0,255) );
						crw[j*4+2]=Math::fast_ftoi( CLAMP(c.b*255.0,0,255) );
						crw[j*4+3]=Math::fast_ftoi( CLAMP(c.a*255.0,0,255) );
					}

					at=pos;
					color_at=to;
				}
			}

			Image gradient(color_ramp_len,1,0,Image::FORMAT_RGBA8,cramp);
			Ref<ImageTexture> it = memnew( ImageTexture );
			it->create_from_image(gradient,Texture::FLAG_FILTER|Texture::FLAG_MIPMAPS);

			String crampname= "cramp_"+itos(p_node->id);
			set_default_texture_param(crampname,it);

			code +="uniform texture "+crampname+";\n";
			code +="vec4 "+crampname+"_r=tex("+crampname+",vec2("+p_inputs[0]+",0));\n";
			code += OUTNAME(p_node->id,0)+"="+crampname+"_r.rgb;\n";
			code += OUTNAME(p_node->id,1)+"="+crampname+"_r.a;\n";

		}break;
		case NODE_CURVE_MAP: {
			DEF_SCALAR(0);
			static const int curve_map_len=256;
			bool mapped[256];
			zeromem(mapped,sizeof(mapped));
			PoolVector<uint8_t> cmap;
			cmap.resize(curve_map_len);
			{

				PoolVector<Point2> points=p_node->param1;
				int pc =points.size();
				PoolVector<uint8_t>::Write cmw = cmap.write();
				PoolVector<Point2>::Read pr = points.read();

				Vector2 prev=Vector2(0,0);
				Vector2 prev2=Vector2(0,0);

				for(int i=-1;i<pc;i++) {

					Vector2 next;
					Vector2 next2;
					if (i+1>=pc) {
						next=Vector2(1,1);
					} else {
						next=Vector2(pr[i+1].x,pr[i+1].y);
					}

					if (i+2>=pc) {
						next2=Vector2(1,1);
					} else {
						next2=Vector2(pr[i+2].x,pr[i+2].y);
					}

					/*if (i==-1 && prev.offset==next.offset) {
						prev=next;
						continue;
					}*/

					_plot_curve(prev2,prev,next,next2,cmw.ptr(),mapped);

					prev2=prev;
					prev=next;
				}

				uint8_t pp=0;
				for(int i=0;i<curve_map_len;i++) {

					if (!mapped[i]) {
						cmw[i]=pp;
					} else {
						pp=cmw[i];
					}
				}
			}



			Image gradient(curve_map_len,1,0,Image::FORMAT_L8,cmap);
			Ref<ImageTexture> it = memnew( ImageTexture );
			it->create_from_image(gradient,Texture::FLAG_FILTER|Texture::FLAG_MIPMAPS);

			String cmapname= "cmap_"+itos(p_node->id);
			set_default_texture_param(cmapname,it);

			code +="uniform texture "+cmapname+";\n";
			code += OUTNAME(p_node->id,0)+"=tex("+cmapname+",vec2("+p_inputs[0]+",0)).r;\n";

		}break;
		case NODE_SCALAR_INPUT: {
			String name = p_node->param1;
			float dv=p_node->param2;
			code +="uniform float "+name+"="+rtos(dv)+";\n";
			code += OUTNAME(p_node->id,0)+"="+name+";\n";
		}break;
		case NODE_VEC_INPUT: {

			String name = p_node->param1;
			Vector3 dv=p_node->param2;
			code +="uniform vec3 "+name+"=vec3("+rtos(dv.x)+","+rtos(dv.y)+","+rtos(dv.z)+");\n";
			code += OUTNAME(p_node->id,0)+"="+name+";\n";
		}break;
		case NODE_RGB_INPUT: {

			String name = p_node->param1;
			Color dv= p_node->param2;

			code +="uniform color "+name+"=vec4("+rtos(dv.r)+","+rtos(dv.g)+","+rtos(dv.b)+","+rtos(dv.a)+");\n";
			code += OUTNAME(p_node->id,0)+"="+name+".rgb;\n";
			code += OUTNAME(p_node->id,1)+"="+name+".a;\n";

		}break;
		case NODE_XFORM_INPUT: {

			String name = p_node->param1;
			Transform dv= p_node->param2;

			code +="uniform mat4 "+name+"=mat4(\n";
			code+="\tvec4(vec3("+rtos(dv.basis.get_axis(0).x)+","+rtos(dv.basis.get_axis(0).y)+","+rtos(dv.basis.get_axis(0).z)+"),0),\n";
			code+="\tvec4(vec3("+rtos(dv.basis.get_axis(1).x)+","+rtos(dv.basis.get_axis(1).y)+","+rtos(dv.basis.get_axis(1).z)+"),0),\n";
			code+="\tvec4(vec3("+rtos(dv.basis.get_axis(2).x)+","+rtos(dv.basis.get_axis(2).y)+","+rtos(dv.basis.get_axis(2).z)+"),0),\n";
			code+="\tvec4(vec3("+rtos(dv.origin.x)+","+rtos(dv.origin.y)+","+rtos(dv.origin.z)+"),1)\n";
			code+=");";

			code += OUTNAME(p_node->id,0)+"="+name+";\n";

		}break;
		case NODE_TEXTURE_INPUT: {
			DEF_VEC(0);
			String name = p_node->param1;
			String rname="rt_read_tex"+itos(p_node->id);
			code +="uniform texture "+name+";";
			code +="vec4 "+rname+"=tex("+name+","+p_inputs[0]+".xy);\n";
			code += OUTNAME(p_node->id,0)+"="+rname+".rgb;\n";
			code += OUTNAME(p_node->id,1)+"="+rname+".a;\n";

		}break;
		case NODE_CUBEMAP_INPUT: {
			DEF_VEC(0);
			String name = p_node->param1;
			code +="uniform cubemap "+name+";";
			String rname="rt_read_tex"+itos(p_node->id);
			code +="vec4 "+rname+"=texcube("+name+","+p_inputs[0]+".xy);\n";
			code += OUTNAME(p_node->id,0)+"="+rname+".rgb;\n";
			code += OUTNAME(p_node->id,1)+"="+rname+".a;\n";
		}break;
		case NODE_DEFAULT_TEXTURE: {
			DEF_VEC(0);

			if (get_mode()==MODE_CANVAS_ITEM && p_type==SHADER_TYPE_FRAGMENT) {

				String rname="rt_default_tex"+itos(p_node->id);
				code +="vec4 "+rname+"=tex(TEXTURE,"+p_inputs[0]+".xy);\n";
				code += OUTNAME(p_node->id,0)+"="+rname+".rgb;\n";
				code += OUTNAME(p_node->id,1)+"="+rname+".a;\n";

			} else {
				//not supported
				code += OUTNAME(p_node->id,0)+"=vec3(0,0,0);\n";
				code += OUTNAME(p_node->id,1)+"=1.0;\n";

			}
		} break;
		case NODE_OUTPUT: {


		}break;
		case NODE_COMMENT: {

		}break;
		case NODE_TYPE_MAX: {

		}
	}
#undef DEF_SCALAR
#undef DEF_COLOR
#undef DEF_MATRIX
#undef DEF_VEC
}

#endif
