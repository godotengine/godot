/*************************************************************************/
/*  shader_graph.cpp                                                     */
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
#include "shader_graph.h"



# if 0
void ShaderGraph::_set(const String& p_name, const Variant& p_value) {

	if (p_name.begins_with("nodes/")) {
		int idx=p_name.get_slice("/",1).to_int();
		Dictionary data=p_value;

		ERR_FAIL_COND(!data.has("type"));
		String type=data["type"];

		VS::NodeType node_type=VS::NODE_TYPE_MAX;
		for(int i=0;i<NODE_TYPE_MAX;i++) {

			if (type==VisualServer::shader_node_get_type_info((VS::NodeType)i).name)
				node_type=(VS::NodeType)i;
		}

		ERR_FAIL_COND(node_type==VS::NODE_TYPE_MAX);

		node_add( (NodeType)node_type, idx );
		if (data.has("param"))
			node_set_param(idx,data["param"]);
		if (data.has("pos"))
			node_set_pos(idx,data["pos"]);
	}

	if (p_name.begins_with("conns/")) {
		Dictionary data=p_value;
		ERR_FAIL_COND( !data.has("src_id") );
		ERR_FAIL_COND( !data.has("src_slot") );
		ERR_FAIL_COND( !data.has("dst_id") );
		ERR_FAIL_COND( !data.has("dst_slot") );

		connect(data["src_id"],data["src_slot"],data["dst_id"],data["dst_slot"]);
	}

	return false;
}
Variant ShaderGraph::_get(const String& p_name) const {

	if (p_name.begins_with("nodes/")) {
		int idx=p_name.get_slice("/",1).to_int();
		Dictionary data;
		data["type"]=VisualServer::shader_node_get_type_info((VS::NodeType)node_get_type(idx)).name;
		data["pos"]=node_get_pos(idx);
		data["param"]=node_get_param(idx);
		return data;
	}
	if (p_name.begins_with("conns/")) {
		int idx=p_name.get_slice("/",1).to_int();
		Dictionary data;

		List<Connection> connections;
		get_connections(&connections);
		ERR_FAIL_INDEX_V( idx,connections.size(), Variant() );
		Connection c = connections[idx];

		data["src_id"]=c.src_id;
		data["src_slot"]=c.src_slot;
		data["dst_id"]=c.dst_id;
		data["dst_slot"]=c.dst_slot;
		return data;
	}

	return Variant();
}
void ShaderGraph::_get_property_list( List<PropertyInfo> *p_list) const {

	List<int> nodes;
	get_node_list(&nodes);

	for(List<int>::Element *E=nodes.front();E;E=E->next()) {

		int idx=E->get();
		p_list->push_back(PropertyInfo( Variant::DICTIONARY , "nodes/"+itos(idx),PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NETWORK|PROPERTY_USAGE_STORAGE ) );
	}

	List<Connection> connections;
	get_connections(&connections);
	int idx=0;
	for(List<Connection>::Element *E=connections.front();E;E=E->next()) {
		p_list->push_back(PropertyInfo( Variant::DICTIONARY , "conns/"+itos(idx++),PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NETWORK|PROPERTY_USAGE_STORAGE ) );
	}

}

#endif
#if 0
Array ShaderGraph::_get_connections_helper() const {

	Array connections_ret;
	List<Connection> connections;
	get_connections(&connections);
	connections_ret.resize(connections.size());

	int idx=0;
	for(List<Connection>::Element *E=connections.front();E;E=E->next()) {

		Connection c = E->get();
		Dictionary data;
		data["src_id"]=c.src_id;
		data["src_slot"]=c.src_slot;
		data["dst_id"]=c.dst_id;
		data["dst_slot"]=c.dst_slot;
		connections_ret.set(idx++,data);
	}

	return connections_ret;
}

void ShaderGraph::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("node_add"),&ShaderGraph::node_add );
	ObjectTypeDB::bind_method(_MD("node_remove"),&ShaderGraph::node_remove );
	ObjectTypeDB::bind_method(_MD("node_set_param"),&ShaderGraph::node_set_param );
	ObjectTypeDB::bind_method(_MD("node_set_pos"),&ShaderGraph::node_set_pos );

	ObjectTypeDB::bind_method(_MD("node_get_pos"),&ShaderGraph::node_get_pos );
	ObjectTypeDB::bind_method(_MD("node_get_param"),&ShaderGraph::node_get_param);
	ObjectTypeDB::bind_method(_MD("node_get_type"),&ShaderGraph::node_get_type);

	ObjectTypeDB::bind_method(_MD("connect"),&ShaderGraph::connect );
	ObjectTypeDB::bind_method(_MD("disconnect"),&ShaderGraph::disconnect );

	ObjectTypeDB::bind_method(_MD("get_connections"),&ShaderGraph::_get_connections_helper );

	ObjectTypeDB::bind_method(_MD("clear"),&ShaderGraph::clear );

	BIND_CONSTANT( NODE_IN ); ///< param 0: name
	BIND_CONSTANT( NODE_OUT ); ///< param 0: name
	BIND_CONSTANT( NODE_CONSTANT ); ///< param 0: value
	BIND_CONSTANT( NODE_PARAMETER ); ///< param 0: name
	BIND_CONSTANT( NODE_ADD );
	BIND_CONSTANT( NODE_SUB );
	BIND_CONSTANT( NODE_MUL );
	BIND_CONSTANT( NODE_DIV );
	BIND_CONSTANT( NODE_MOD );
	BIND_CONSTANT( NODE_SIN );
	BIND_CONSTANT( NODE_COS );
	BIND_CONSTANT( NODE_TAN );
	BIND_CONSTANT( NODE_ARCSIN );
	BIND_CONSTANT( NODE_ARCCOS );
	BIND_CONSTANT( NODE_ARCTAN );
	BIND_CONSTANT( NODE_POW );
	BIND_CONSTANT( NODE_LOG );
	BIND_CONSTANT( NODE_MAX );
	BIND_CONSTANT( NODE_MIN );
	BIND_CONSTANT( NODE_COMPARE );
	BIND_CONSTANT( NODE_TEXTURE ); ///< param  0: texture
	BIND_CONSTANT( NODE_TIME ); ///< param  0: interval length
	BIND_CONSTANT( NODE_NOISE );
	BIND_CONSTANT( NODE_PASS );
	BIND_CONSTANT( NODE_VEC_IN ); ///< param 0: name
	BIND_CONSTANT( NODE_VEC_OUT ); ///< param 0: name
	BIND_CONSTANT( NODE_VEC_CONSTANT ); ///< param  0: value
	BIND_CONSTANT( NODE_VEC_PARAMETER ); ///< param  0: name
	BIND_CONSTANT( NODE_VEC_ADD );
	BIND_CONSTANT( NODE_VEC_SUB );
	BIND_CONSTANT( NODE_VEC_MUL );
	BIND_CONSTANT( NODE_VEC_DIV );
	BIND_CONSTANT( NODE_VEC_MOD );
	BIND_CONSTANT( NODE_VEC_CROSS );
	BIND_CONSTANT( NODE_VEC_DOT );
	BIND_CONSTANT( NODE_VEC_POW );
	BIND_CONSTANT( NODE_VEC_NORMALIZE );
	BIND_CONSTANT( NODE_VEC_TRANSFORM3 );
	BIND_CONSTANT( NODE_VEC_TRANSFORM4 );
	BIND_CONSTANT( NODE_VEC_COMPARE );
	BIND_CONSTANT( NODE_VEC_TEXTURE_2D );
	BIND_CONSTANT( NODE_VEC_TEXTURE_CUBE );
	BIND_CONSTANT( NODE_VEC_NOISE );
	BIND_CONSTANT( NODE_VEC_0 );
	BIND_CONSTANT( NODE_VEC_1 );
	BIND_CONSTANT( NODE_VEC_2 );
	BIND_CONSTANT( NODE_VEC_BUILD );
	BIND_CONSTANT( NODE_VEC_PASS );
	BIND_CONSTANT( NODE_COLOR_CONSTANT );
	BIND_CONSTANT( NODE_COLOR_PARAMETER );
	BIND_CONSTANT( NODE_TEXTURE_PARAMETER );
	BIND_CONSTANT( NODE_TEXTURE_2D_PARAMETER );
	BIND_CONSTANT( NODE_TEXTURE_CUBE_PARAMETER );
	BIND_CONSTANT( NODE_TYPE_MAX );
}

void ShaderGraph::node_add(NodeType p_type,int p_id) {


	ERR_FAIL_COND( node_map.has(p_id ) );
	ERR_FAIL_INDEX( p_type, NODE_TYPE_MAX );
	Node node;

	node.type=p_type;
	node.id=p_id;
	node.x=0;
	node.y=0;

	node_map[p_id]=node;

}

void ShaderGraph::node_set_pos(int p_id, const Vector2& p_pos) {

	ERR_FAIL_COND(!node_map.has(p_id));
	node_map[p_id].x=p_pos.x;
	node_map[p_id].y=p_pos.y;
}
Vector2 ShaderGraph::node_get_pos(int p_id) const {

	ERR_FAIL_COND_V(!node_map.has(p_id),Vector2());
	return Vector2(node_map[p_id].x,node_map[p_id].y);
}


void ShaderGraph::node_remove(int p_id) {

	ERR_FAIL_COND(!node_map.has(p_id));

	//erase connections associated with node
	List<Connection>::Element *N,*E=connections.front();
	while(E) {
		N=E->next();
		const Connection &c = E->get();
		if (c.src_id==p_id || c.dst_id==p_id) {

			connections.erase(E);
		}
		E=N;
	}

	node_map.erase(p_id);
}

void ShaderGraph::node_change_type(int p_id, NodeType p_type) {

	ERR_FAIL_COND(!node_map.has(p_id));
	node_map[p_id].type=p_type;
	node_map[p_id].param=Variant();

}

void ShaderGraph::node_set_param(int p_id, const Variant& p_value) {

	ERR_FAIL_COND(!node_map.has(p_id));
	node_map[p_id].param=p_value;
}

void ShaderGraph::get_node_list(List<int> *p_node_list) const {

	Map<int,Node>::Element *E = node_map.front();

	while(E) {

		p_node_list->push_back(E->key());
		E=E->next();
	}
}


ShaderGraph::NodeType ShaderGraph::node_get_type(int p_id) const {

	ERR_FAIL_COND_V(!node_map.has(p_id),NODE_TYPE_MAX);
	return node_map[p_id].type;
}

Variant ShaderGraph::node_get_param(int p_id) const {

	ERR_FAIL_COND_V(!node_map.has(p_id),Variant());
	return node_map[p_id].param;
}


Error ShaderGraph::connect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) {

	ERR_FAIL_COND_V(p_src_id==p_dst_id, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!node_map.has(p_src_id), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!node_map.has(p_dst_id), ERR_INVALID_PARAMETER);
	NodeType type_src=node_map[p_src_id].type;
	NodeType type_dst=node_map[p_dst_id].type;
	//ERR_FAIL_INDEX_V( p_src_slot, VisualServer::shader_get_output_count(type_src), ERR_INVALID_PARAMETER );
	//ERR_FAIL_INDEX_V( p_dst_slot, VisualServer::shader_get_input_count(type_dst), ERR_INVALID_PARAMETER );
	//ERR_FAIL_COND_V(VisualServer::shader_is_output_vector(type_src,p_src_slot) != VisualServer::shader_is_input_vector(type_dst,p_dst_slot), ERR_INVALID_PARAMETER );


	List<Connection>::Element *E=connections.front();
	while(E) {
		const Connection &c = E->get();
		ERR_FAIL_COND_V(c.dst_slot==p_dst_slot && c.dst_id == p_dst_id, ERR_ALREADY_EXISTS);

		E=E->next();
	}

	Connection c;
	c.src_slot=p_src_slot;
	c.src_id=p_src_id;
	c.dst_slot=p_dst_slot;
	c.dst_id=p_dst_id;

	connections.push_back(c);

	return OK;
}

bool ShaderGraph::is_connected(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) const {

	const List<Connection>::Element *E=connections.front();
	while(E) {
		const Connection &c = E->get();
		if (c.dst_slot==p_dst_slot && c.dst_id == p_dst_id && c.src_slot==p_src_slot && c.src_id == p_src_id)
			return true;

		E=E->next();
	}

	return false;
}

void ShaderGraph::disconnect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) {

	List<Connection>::Element *N,*E=connections.front();
	while(E) {
		N=E->next();
		const Connection &c = E->get();
		if (c.src_slot==p_src_slot && c.src_id==p_src_id && c.dst_slot==p_dst_slot && c.dst_id == p_dst_id) {

			connections.erase(E);
		}
		E=N;
	}


}

void ShaderGraph::get_connections(List<Connection> *p_connections) const {

	const List<Connection>::Element*E=connections.front();
	while(E) {
		p_connections->push_back(E->get());
		E=E->next();
	}


}


void ShaderGraph::clear() {

	connections.clear();
	node_map.clear();
}


#if 0
void ShaderGraph::node_add(NodeType p_type,int p_id) {

	ShaderNode sn;
	sn.type=p_type;
	nodes[p_id]=sn;
	version++;
}
void ShaderGraph::node_remove(int p_id) {

	nodes.erase(p_id);

}
void ShaderGraph::node_set_param( int p_id, const Variant& p_value) {

	VisualServer::get_singleton()->shader_node_set_param(shader,p_id,p_value);
	version++;
}

void ShaderGraph::get_node_list(List<int> *p_node_list) const {

	VisualServer::get_singleton()->shader_get_node_list(shader,p_node_list);
}
ShaderGraph::NodeType ShaderGraph::node_get_type(int p_id) const {

	return (NodeType)VisualServer::get_singleton()->shader_node_get_type(shader,p_id);
}
Variant ShaderGraph::node_get_param(int p_id) const {

	return VisualServer::get_singleton()->shader_node_get_param(shader,p_id);
}

void ShaderGraph::connect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) {

	VisualServer::get_singleton()->shader_connect(shader,p_src_id,p_src_slot,p_dst_id,p_dst_slot);
	version++;
}
void ShaderGraph::disconnect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) {

	VisualServer::get_singleton()->shader_disconnect(shader,p_src_id,p_src_slot,p_dst_id,p_dst_slot);
	version++;
}

void ShaderGraph::get_connections(List<Connection> *p_connections) const {

	List<VS::ShaderGraphConnection> connections;
	VisualServer::get_singleton()->shader_get_connections(shader,&connections);
	for( List<VS::ShaderGraphConnection>::Element *E=connections.front();E;E=E->next()) {

		Connection c;
		c.src_id=E->get().src_id;
		c.src_slot=E->get().src_slot;
		c.dst_id=E->get().dst_id;
		c.dst_slot=E->get().dst_slot;
		p_connections->push_back(c);
	}
}

void ShaderGraph::node_set_pos(int p_id,const Point2& p_pos) {

#ifdef TOOLS_ENABLED
	ERR_FAIL_COND(!positions.has(p_id));
	positions[p_id]=p_pos;
#endif
}

Point2 ShaderGraph::node_get_pos(int p_id) const {
#ifdef TOOLS_ENABLED
	ERR_FAIL_COND_V(!positions.has(p_id),Point2());
	return positions[p_id];
#endif
}

void ShaderGraph::clear() {

	VisualServer::get_singleton()->shader_clear(shader);
	version++;
}
#endif

ShaderGraph::ShaderGraph() {

	//shader = VisualServer::get_singleton()->shader_create();
	version = 1;
}

ShaderGraph::~ShaderGraph() {

	//VisualServer::get_singleton()->free(shader);
}

#if 0
void ShaderGraph::shader_get_default_input_nodes(Mode p_type,List<PropertyInfo> *p_inputs) {

	switch(p_type) {

		case SHADER_VERTEX: {

			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"vertex") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"normal") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"binormal") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"tangent") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"uv") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"color") );
			p_inputs->push_back( PropertyInfo( Variant::REAL,"alpha") );
		} break;
		case SHADER_FRAGMENT: {

			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"position") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"normal") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"binormal") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"tangent") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"uv") );
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"color") );
			p_inputs->push_back( PropertyInfo( Variant::REAL,"alpha") );

		} break;
		case SHADER_POST_PROCESS: {
			p_inputs->push_back( PropertyInfo( Variant::VECTOR3,"color") );
			p_inputs->push_back( PropertyInfo( Variant::REAL,"alpha") );
		} break;

	}

}
void ShaderGraph::shader_get_default_output_nodes(ShaderGraphType p_type,List<PropertyInfo> *p_outputs) {

	switch(p_type) {

		case SHADER_VERTEX: {

			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"vertex") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"normal") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"binormal") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"tangent") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"uv") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"color") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"alpha") );
		} break;
		case SHADER_FRAGMENT: {

			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"normal") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"diffuse") );
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"specular") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"alpha") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"emission") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"spec_exp") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"glow") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"alpha_discard") );

		} break;
		case SHADER_POST_PROCESS: {
			p_outputs->push_back( PropertyInfo( Variant::VECTOR3,"color") );
			p_outputs->push_back( PropertyInfo( Variant::REAL,"alpha") );
		} break;

	}

}


PropertyInfo ShaderGraph::shader_node_get_type_info(NodeType p_type) {

	switch(p_type) {

		case NODE_IN: return PropertyInfo(Variant::STRING,"in");
		case NODE_OUT: return PropertyInfo(Variant::STRING,"out");
		case NODE_CONSTANT: return PropertyInfo(Variant::REAL,"const");
		case NODE_PARAMETER: return PropertyInfo(Variant::STRING,"param");
		case NODE_ADD: return PropertyInfo(Variant::NIL,"add");
		case NODE_SUB: return PropertyInfo(Variant::NIL,"sub");
		case NODE_MUL: return PropertyInfo(Variant::NIL,"mul");
		case NODE_DIV: return PropertyInfo(Variant::NIL,"div");
		case NODE_MOD: return PropertyInfo(Variant::NIL,"rem");
		case NODE_SIN: return PropertyInfo(Variant::NIL,"sin");
		case NODE_COS: return PropertyInfo(Variant::NIL,"cos");
		case NODE_TAN: return PropertyInfo(Variant::NIL,"tan");
		case NODE_ARCSIN: return PropertyInfo(Variant::NIL,"arcsin");
		case NODE_ARCCOS: return PropertyInfo(Variant::NIL,"arccos");
		case NODE_ARCTAN: return PropertyInfo(Variant::NIL,"arctan");
		case NODE_POW: return PropertyInfo(Variant::NIL,"pow");
		case NODE_LOG: return PropertyInfo(Variant::NIL,"log");
		case NODE_MAX: return PropertyInfo(Variant::NIL,"max");
		case NODE_MIN: return PropertyInfo(Variant::NIL,"min");
		case NODE_COMPARE: return PropertyInfo(Variant::NIL,"cmp");
		case NODE_TEXTURE: return PropertyInfo(Variant::_RID,"texture1D",PROPERTY_HINT_RESOURCE_TYPE,"Texture");
		case NODE_TIME: return PropertyInfo(Variant::NIL,"time");
		case NODE_NOISE: return PropertyInfo(Variant::NIL,"noise");
		case NODE_PASS: return PropertyInfo(Variant::NIL,"pass");
		case NODE_VEC_IN: return PropertyInfo(Variant::STRING,"vin");
		case NODE_VEC_OUT: return PropertyInfo(Variant::STRING,"vout");
		case NODE_VEC_CONSTANT: return PropertyInfo(Variant::VECTOR3,"vconst");
		case NODE_VEC_PARAMETER: return PropertyInfo(Variant::STRING,"vparam");
		case NODE_VEC_ADD: return PropertyInfo(Variant::NIL,"vadd");
		case NODE_VEC_SUB: return PropertyInfo(Variant::NIL,"vsub");
		case NODE_VEC_MUL: return PropertyInfo(Variant::NIL,"vmul");
		case NODE_VEC_DIV: return PropertyInfo(Variant::NIL,"vdiv");
		case NODE_VEC_MOD: return PropertyInfo(Variant::NIL,"vrem");
		case NODE_VEC_CROSS: return PropertyInfo(Variant::NIL,"cross");
		case NODE_VEC_DOT: return PropertyInfo(Variant::NIL,"dot");
		case NODE_VEC_POW: return PropertyInfo(Variant::NIL,"vpow");
		case NODE_VEC_NORMALIZE: return PropertyInfo(Variant::NIL,"normalize");
		case NODE_VEC_INTERPOLATE: return PropertyInfo(Variant::NIL,"mix");
		case NODE_VEC_SCREEN_TO_UV: return PropertyInfo(Variant::NIL,"scrn2uv");
		case NODE_VEC_TRANSFORM3: return PropertyInfo(Variant::NIL,"xform3");
		case NODE_VEC_TRANSFORM4: return PropertyInfo(Variant::NIL,"xform4");
		case NODE_VEC_COMPARE: return PropertyInfo(Variant::_RID,"vcmp",PROPERTY_HINT_RESOURCE_TYPE,"Texture");
		case NODE_VEC_TEXTURE_2D: return PropertyInfo(Variant::_RID,"texture2D",PROPERTY_HINT_RESOURCE_TYPE,"Texture");
		case NODE_VEC_TEXTURE_CUBE: return PropertyInfo(Variant::NIL,"texcube");
		case NODE_VEC_NOISE:	return PropertyInfo(Variant::NIL,"vec_noise");
		case NODE_VEC_0: return PropertyInfo(Variant::NIL,"vec_0");
		case NODE_VEC_1: return PropertyInfo(Variant::NIL,"vec_1");
		case NODE_VEC_2: return PropertyInfo(Variant::NIL,"vec_2");
		case NODE_VEC_BUILD: return PropertyInfo(Variant::NIL,"vbuild");
		case NODE_VEC_PASS: return PropertyInfo(Variant::NIL,"vpass");
		case NODE_COLOR_CONSTANT: return PropertyInfo(Variant::COLOR,"color_const");
		case NODE_COLOR_PARAMETER: return PropertyInfo(Variant::STRING,"color_param");
		case NODE_TEXTURE_PARAMETER:  return PropertyInfo(Variant::STRING,"tex1D_param");
		case NODE_TEXTURE_2D_PARAMETER:  return PropertyInfo(Variant::STRING,"tex2D_param");
		case NODE_TEXTURE_CUBE_PARAMETER:  return PropertyInfo(Variant::STRING,"texcube_param");
		case NODE_TRANSFORM_CONSTANT:  return PropertyInfo(Variant::TRANSFORM,"xform_const");
		case NODE_TRANSFORM_PARAMETER: return PropertyInfo(Variant::STRING,"xform_param");
		case NODE_LABEL: return PropertyInfo(Variant::STRING,"label");

		default: {}

	}

	ERR_FAIL_V( PropertyInfo(Variant::NIL,"error") );
}
int ShaderGraph::shader_get_input_count(NodeType p_type) {

	switch(p_type) {
		case NODE_IN: return 0;
		case NODE_OUT: return 1;
		case NODE_CONSTANT: return 0;
		case NODE_PARAMETER: return 0;
		case NODE_ADD: return 2;
		case NODE_SUB: return 2;
		case NODE_MUL: return 2;
		case NODE_DIV: return 2;
		case NODE_MOD: return 2;
		case NODE_SIN: return 1;
		case NODE_COS: return 1;
		case NODE_TAN: return 1;
		case NODE_ARCSIN: return 1;
		case NODE_ARCCOS: return 1;
		case NODE_ARCTAN: return 1;
		case NODE_POW: return 2;
		case NODE_LOG: return 1;
		case NODE_MAX: return 2;
		case NODE_MIN: return 2;
		case NODE_COMPARE: return 4;
		case NODE_TEXTURE: return 1;  ///< param  0: texture
		case NODE_TIME: return 1;  ///< param  0: interval length
		case NODE_NOISE: return 0;
		case NODE_PASS: return 1;
		case NODE_VEC_IN: return 0;  ///< param 0: name
		case NODE_VEC_OUT: return 1;  ///< param 0: name
		case NODE_VEC_CONSTANT: return 0;  ///< param  0: value
		case NODE_VEC_PARAMETER: return 0;  ///< param  0: name
		case NODE_VEC_ADD: return 2;
		case NODE_VEC_SUB: return 2;
		case NODE_VEC_MUL: return 2;
		case NODE_VEC_DIV: return 2;
		case NODE_VEC_MOD: return 2;
		case NODE_VEC_CROSS: return 2;
		case NODE_VEC_DOT: return 2;
		case NODE_VEC_POW: return 2;
		case NODE_VEC_NORMALIZE: return 1;
		case NODE_VEC_INTERPOLATE: return 3;
		case NODE_VEC_SCREEN_TO_UV: return 1;
		case NODE_VEC_TRANSFORM3: return 4;
		case NODE_VEC_TRANSFORM4: return 5;
		case NODE_VEC_COMPARE: return 4;
		case NODE_VEC_TEXTURE_2D: return 1;
		case NODE_VEC_TEXTURE_CUBE: return 1;
		case NODE_VEC_NOISE: return 0;
		case NODE_VEC_0: return 1;
		case NODE_VEC_1: return 1;
		case NODE_VEC_2: return 1;
		case NODE_VEC_BUILD: return 3;
		case NODE_VEC_PASS: return 1;
		case NODE_COLOR_CONSTANT: return 0;
		case NODE_COLOR_PARAMETER: return 0;
		case NODE_TEXTURE_PARAMETER:  return 1;
		case NODE_TEXTURE_2D_PARAMETER:  return 1;
		case NODE_TEXTURE_CUBE_PARAMETER:  return 1;
		case NODE_TRANSFORM_CONSTANT: return 1;
		case NODE_TRANSFORM_PARAMETER: return 1;
		case NODE_LABEL: return 0;
		default: {}
	}
	ERR_FAIL_V( 0 );
}
int ShaderGraph::shader_get_output_count(NodeType p_type) {

	switch(p_type) {
		case NODE_IN: return 1;
		case NODE_OUT: return 0;
		case NODE_CONSTANT: return 1;
		case NODE_PARAMETER: return 1;
		case NODE_ADD: return 1;
		case NODE_SUB: return 1;
		case NODE_MUL: return 1;
		case NODE_DIV: return 1;
		case NODE_MOD: return 1;
		case NODE_SIN: return 1;
		case NODE_COS: return 1;
		case NODE_TAN: return 1;
		case NODE_ARCSIN: return 1;
		case NODE_ARCCOS: return 1;
		case NODE_ARCTAN: return 1;
		case NODE_POW: return 1;
		case NODE_LOG: return 1;
		case NODE_MAX: return 1;
		case NODE_MIN: return 1;
		case NODE_COMPARE: return 2;
		case NODE_TEXTURE: return 3;  ///< param  0: texture
		case NODE_TIME: return 1;  ///< param  0: interval length
		case NODE_NOISE: return 1;
		case NODE_PASS: return 1;
		case NODE_VEC_IN: return 1;  ///< param 0: name
		case NODE_VEC_OUT: return 0;  ///< param 0: name
		case NODE_VEC_CONSTANT: return 1;  ///< param  0: value
		case NODE_VEC_PARAMETER: return 1;  ///< param  0: name
		case NODE_VEC_ADD: return 1;
		case NODE_VEC_SUB: return 1;
		case NODE_VEC_MUL: return 1;
		case NODE_VEC_DIV: return 1;
		case NODE_VEC_MOD: return 1;
		case NODE_VEC_CROSS: return 1;
		case NODE_VEC_DOT: return 1;
		case NODE_VEC_POW: return 1;
		case NODE_VEC_NORMALIZE: return 1;
		case NODE_VEC_INTERPOLATE: return 1;
		case NODE_VEC_SCREEN_TO_UV: return 1;
		case NODE_VEC_TRANSFORM3: return 1;
		case NODE_VEC_TRANSFORM4: return 1;
		case NODE_VEC_COMPARE: return 2;
		case NODE_VEC_TEXTURE_2D: return 3;
		case NODE_VEC_TEXTURE_CUBE: return 3;
		case NODE_VEC_NOISE: return 1;
		case NODE_VEC_0: return 1;
		case NODE_VEC_1: return 1;
		case NODE_VEC_2: return 1;
		case NODE_VEC_BUILD: return 1;
		case NODE_VEC_PASS: return 1;
		case NODE_COLOR_CONSTANT: return 2;
		case NODE_COLOR_PARAMETER: return 2;
		case NODE_TEXTURE_PARAMETER:  return 3;
		case NODE_TEXTURE_2D_PARAMETER:  return 3;
		case NODE_TEXTURE_CUBE_PARAMETER:  return 3;
		case NODE_TRANSFORM_CONSTANT: return 1;
		case NODE_TRANSFORM_PARAMETER: return 1;
		case NODE_LABEL: return 0;

		default: {}
	}
	ERR_FAIL_V( 0 );

}

#define RET2(m_a,m_b)	if (p_idx==0) return m_a; else if (p_idx==1) return m_b; else return "";
#define RET3(m_a,m_b,m_c)	if (p_idx==0) return m_a; else if (p_idx==1) return m_b; else if (p_idx==2) return m_c;  else return "";
#define RET4(m_a,m_b,m_c,m_d)	if (p_idx==0) return m_a; else if (p_idx==1) return m_b; else if (p_idx==2) return m_c; else if (p_idx==3) return m_d; else return "";

#define RET5(m_a,m_b,m_c,m_d,m_e)	if (p_idx==0) return m_a; else if (p_idx==1) return m_b; else if (p_idx==2) return m_c; else if (p_idx==3) return m_d; else if (p_idx==4) return m_e; else return "";

String ShaderGraph::shader_get_input_name(NodeType p_type,int p_idx) {

	switch(p_type) {

		case NODE_IN: return "";
		case NODE_OUT: return "out";
		case NODE_CONSTANT: return "";
		case NODE_PARAMETER: return "";
		case NODE_ADD: RET2("a","b");
		case NODE_SUB: RET2("a","b");
		case NODE_MUL: RET2("a","b");
		case NODE_DIV: RET2("a","b");
		case NODE_MOD: RET2("a","b");
		case NODE_SIN: return "rad";
		case NODE_COS: return "rad";
		case NODE_TAN: return "rad";
		case NODE_ARCSIN: return "in";
		case NODE_ARCCOS: return "in";
		case NODE_ARCTAN: return "in";
		case NODE_POW: RET2("in","exp");
		case NODE_LOG: return "in";
		case NODE_MAX: return "in";
		case NODE_MIN: return "in";
		case NODE_COMPARE: RET4("a","b","ret1","ret2");
		case NODE_TEXTURE: return "u";
		case NODE_TIME: return "";
		case NODE_NOISE: return "";
		case NODE_PASS: return "in";
		case NODE_VEC_IN: return "";
		case NODE_VEC_OUT: return "out";
		case NODE_VEC_CONSTANT: return "";
		case NODE_VEC_PARAMETER: return "";
		case NODE_VEC_ADD: RET2("a","b");
		case NODE_VEC_SUB: RET2("a","b");
		case NODE_VEC_MUL: RET2("a","b");
		case NODE_VEC_DIV: RET2("a","b");
		case NODE_VEC_MOD: RET2("a","b");
		case NODE_VEC_CROSS: RET2("a","b");
		case NODE_VEC_DOT: RET2("a","b");
		case NODE_VEC_POW: RET2("a","b");
		case NODE_VEC_NORMALIZE: return "vec";
		case NODE_VEC_INTERPOLATE: RET3("a","b","c");
		case NODE_VEC_SCREEN_TO_UV: return "scr";
		case NODE_VEC_TRANSFORM3: RET4("in","col0","col1","col2");
		case NODE_VEC_TRANSFORM4: RET5("in","col0","col1","col2","col3");
		case NODE_VEC_COMPARE: RET4("a","b","ret1","ret2");
		case NODE_VEC_TEXTURE_2D: return "uv";
		case NODE_VEC_TEXTURE_CUBE: return "uvw";
		case NODE_VEC_NOISE: return "";
		case NODE_VEC_0: return "vec";
		case NODE_VEC_1: return "vec";
		case NODE_VEC_2: return "vec";
		case NODE_VEC_BUILD: RET3("x/r","y/g","z/b");
		case NODE_VEC_PASS: return "in";
		case NODE_COLOR_CONSTANT: return "";
		case NODE_COLOR_PARAMETER: return "";
		case NODE_TEXTURE_PARAMETER:  return "u";
		case NODE_TEXTURE_2D_PARAMETER:  return "uv";
		case NODE_TEXTURE_CUBE_PARAMETER:  return "uvw";
		case NODE_TRANSFORM_CONSTANT: return "in";
		case NODE_TRANSFORM_PARAMETER: return "in";
		case NODE_LABEL: return "";

		default: {}
	}

	ERR_FAIL_V("");
}
String ShaderGraph::shader_get_output_name(NodeType p_type,int p_idx) {

	switch(p_type) {

		case NODE_IN: return "in";
		case NODE_OUT: return "";
		case NODE_CONSTANT: return "out";
		case NODE_PARAMETER: return "out";
		case NODE_ADD: return "sum";
		case NODE_SUB: return "dif";
		case NODE_MUL: return "prod";
		case NODE_DIV: return "quot";
		case NODE_MOD: return "rem";
		case NODE_SIN: return "out";
		case NODE_COS: return "out";
		case NODE_TAN: return "out";
		case NODE_ARCSIN: return "rad";
		case NODE_ARCCOS: return "rad";
		case NODE_ARCTAN: return "rad";
		case NODE_POW: RET2("in","exp");
		case NODE_LOG: return "out";
		case NODE_MAX: return "out";
		case NODE_MIN: return "out";
		case NODE_COMPARE: RET2("a/b","a/b");
		case NODE_TEXTURE: RET3("rgb","a","v");
		case NODE_TIME: return "out";
		case NODE_NOISE: return "out";
		case NODE_PASS: return "out";
		case NODE_VEC_IN: return "in";
		case NODE_VEC_OUT: return "";
		case NODE_VEC_CONSTANT: return "out";
		case NODE_VEC_PARAMETER: return "out";
		case NODE_VEC_ADD: return "sum";
		case NODE_VEC_SUB: return "sub";
		case NODE_VEC_MUL: return "mul";
		case NODE_VEC_DIV: return "div";
		case NODE_VEC_MOD: return "rem";
		case NODE_VEC_CROSS: return "crs";
		case NODE_VEC_DOT: return "prod";
		case NODE_VEC_POW: return "out";
		case NODE_VEC_NORMALIZE: return "norm";
		case NODE_VEC_INTERPOLATE: return "out";
		case NODE_VEC_SCREEN_TO_UV: return "uv";
		case NODE_VEC_TRANSFORM3: return "prod";
		case NODE_VEC_TRANSFORM4: return "prod";
		case NODE_VEC_COMPARE: RET2("a/b","a/b");
		case NODE_VEC_TEXTURE_2D: RET3("rgb","a","v");
		case NODE_VEC_TEXTURE_CUBE: RET3("rgb","a","v");
		case NODE_VEC_NOISE: return "out";
		case NODE_VEC_0: return "x/r";
		case NODE_VEC_1: return "y/g";
		case NODE_VEC_2: return "z/b";
		case NODE_VEC_BUILD: return "vec";
		case NODE_VEC_PASS: return "out";
		case NODE_COLOR_CONSTANT: RET2("rgb","a");
		case NODE_COLOR_PARAMETER: RET2("rgb","a");
		case NODE_TEXTURE_PARAMETER:  RET3("rgb","a","v");
		case NODE_TEXTURE_2D_PARAMETER:  RET3("rgb","a","v");
		case NODE_TEXTURE_CUBE_PARAMETER:  RET3("rgb","a","v");
		case NODE_TRANSFORM_CONSTANT: return "out";
		case NODE_TRANSFORM_PARAMETER: return "out";
		case NODE_LABEL: return "";

		default: {}
	}

	ERR_FAIL_V("");
}
bool ShaderGraph::shader_is_input_vector(NodeType p_type,int p_input) {

	switch(p_type) {

		case NODE_IN: return false;
		case NODE_OUT: return false;
		case NODE_CONSTANT: return false;
		case NODE_PARAMETER: return false;
		case NODE_ADD: return false;
		case NODE_SUB: return false;
		case NODE_MUL: return false;
		case NODE_DIV: return false;
		case NODE_MOD: return false;
		case NODE_SIN: return false;
		case NODE_COS: return false;
		case NODE_TAN: return false;
		case NODE_ARCSIN: return false;
		case NODE_ARCCOS: return false;
		case NODE_ARCTAN: return false;
		case NODE_POW: return false;
		case NODE_LOG: return false;
		case NODE_MAX: return false;
		case NODE_MIN: return false;
		case NODE_COMPARE: return false;
		case NODE_TEXTURE: return false;
		case NODE_TIME: return false;
		case NODE_NOISE: return false;
		case NODE_PASS: return false;
		case NODE_VEC_IN: return false;
		case NODE_VEC_OUT: return true;
		case NODE_VEC_CONSTANT: return false;
		case NODE_VEC_PARAMETER: return false;
		case NODE_VEC_ADD: return true;
		case NODE_VEC_SUB: return true;
		case NODE_VEC_MUL: return true;
		case NODE_VEC_DIV: return true;
		case NODE_VEC_MOD: return true;
		case NODE_VEC_CROSS: return true;
		case NODE_VEC_DOT: return true;
		case NODE_VEC_POW: return (p_input==0)?true:false;
		case NODE_VEC_NORMALIZE: return true;
		case NODE_VEC_INTERPOLATE: return (p_input<2)?true:false;
		case NODE_VEC_SCREEN_TO_UV: return true;
		case NODE_VEC_TRANSFORM3: return true;
		case NODE_VEC_TRANSFORM4: return true;
		case NODE_VEC_COMPARE: return (p_input<2)?false:true;
		case NODE_VEC_TEXTURE_2D: return true;
		case NODE_VEC_TEXTURE_CUBE: return true;
		case NODE_VEC_NOISE: return false;
		case NODE_VEC_0: return true;
		case NODE_VEC_1: return true;
		case NODE_VEC_2: return true;
		case NODE_VEC_BUILD: return false;
		case NODE_VEC_PASS: return true;
		case NODE_COLOR_CONSTANT: return false;
		case NODE_COLOR_PARAMETER: return false;
		case NODE_TEXTURE_PARAMETER:  return false;
		case NODE_TEXTURE_2D_PARAMETER:  return true;
		case NODE_TEXTURE_CUBE_PARAMETER:  return true;
		case NODE_TRANSFORM_CONSTANT: return true;
		case NODE_TRANSFORM_PARAMETER: return true;
		case NODE_LABEL: return false;

		default: {}
	}

	ERR_FAIL_V(false);
}
bool ShaderGraph::shader_is_output_vector(NodeType p_type,int p_input) {

	switch(p_type) {

		case NODE_IN: return false;
		case NODE_OUT: return false ;
		case NODE_CONSTANT: return false;
		case NODE_PARAMETER: return false;
		case NODE_ADD: return false;
		case NODE_SUB: return false;
		case NODE_MUL: return false;
		case NODE_DIV: return false;
		case NODE_MOD: return false;
		case NODE_SIN: return false;
		case NODE_COS: return false;
		case NODE_TAN: return false;
		case NODE_ARCSIN: return false;
		case NODE_ARCCOS: return false;
		case NODE_ARCTAN: return false;
		case NODE_POW: return false;
		case NODE_LOG: return false;
		case NODE_MAX: return false;
		case NODE_MIN: return false;
		case NODE_COMPARE: return false;
		case NODE_TEXTURE: return false;
		case NODE_TIME: return false;
		case NODE_NOISE: return false;
		case NODE_PASS: return false;
		case NODE_VEC_IN: return true;
		case NODE_VEC_OUT: return false;
		case NODE_VEC_CONSTANT: return true;
		case NODE_VEC_PARAMETER: return true;
		case NODE_VEC_ADD: return true;
		case NODE_VEC_SUB: return true;
		case NODE_VEC_MUL: return true;
		case NODE_VEC_DIV: return true;
		case NODE_VEC_MOD: return true;
		case NODE_VEC_CROSS: return true;
		case NODE_VEC_DOT: return false;
		case NODE_VEC_POW: return true;
		case NODE_VEC_NORMALIZE: return true;
		case NODE_VEC_INTERPOLATE: return true;
		case NODE_VEC_SCREEN_TO_UV: return true;
		case NODE_VEC_TRANSFORM3: return true;
		case NODE_VEC_TRANSFORM4: return true;
		case NODE_VEC_COMPARE: return true;
		case NODE_VEC_TEXTURE_2D: return (p_input==0)?true:false;
		case NODE_VEC_TEXTURE_CUBE: return (p_input==0)?true:false;
		case NODE_VEC_NOISE: return true;
		case NODE_VEC_0: return false;
		case NODE_VEC_1: return false;
		case NODE_VEC_2: return false;
		case NODE_VEC_BUILD: return true;
		case NODE_VEC_PASS: return true;
		case NODE_COLOR_CONSTANT: return (p_input==0)?true:false;
		case NODE_COLOR_PARAMETER: return (p_input==0)?true:false;
		case NODE_TEXTURE_PARAMETER:  return (p_input==0)?true:false;
		case NODE_TEXTURE_2D_PARAMETER:  return (p_input==0)?true:false;
		case NODE_TEXTURE_CUBE_PARAMETER:  return (p_input==0)?true:false;
		case NODE_TRANSFORM_CONSTANT: return true;
		case NODE_TRANSFORM_PARAMETER: return true;
		case NODE_LABEL: return false;

		default: {}
	}

	ERR_FAIL_V("");
}

#endif
#endif
