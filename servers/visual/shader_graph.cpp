/*************************************************************************/
/*  shader_graph.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#if 0


struct _ConnectionKey {

	int node;
	int slot;

	_FORCE_INLINE_ _ConnectionKey(int p_node=0,int p_slot=0) { node=p_node; slot=p_slot; }

	_FORCE_INLINE_ bool operator<(const _ConnectionKey& p_other) const {

		if (node<p_other.node)
			return true;
		else if (node>p_other.node)
			return false;
		else
			return slot<p_other.slot;
	}
};

Error ShaderGraph::generate(ShaderCodeGenerator * p_generator) const {

	Map<int,Node>::Element *E = node_map.front();
	int i=0;
	while(E) {

		E->get().order=i++;
		E->get().out_valid=false;
		E->get().in_valid=false;
		E=E->next();
	}

	int worst_case=connections.size() * connections.size(); // worst bubble case
	int iterations=0;
	int swaps;

	do {
		swaps=0;
		const List<Connection>::Element *E=connections.front();

		while(E) {

			const Connection &c = E->get();

			const Node *src = &node_map[c.src_id];
			const Node *dst = &node_map[c.dst_id];

			if (src->order > dst->order) {

				SWAP(src->order, dst->order);
				swaps++;
			}

			E=E->next();
		}


		iterations++;

	} while (iterations<=worst_case && swaps>0);

	ERR_FAIL_COND_V( swaps != 0 , ERR_CYCLIC_LINK );

	//node array
	Vector<const Node*> nodes;
	nodes.resize(node_map.size());

	E = node_map.front();
	while(E) {

		ERR_FAIL_INDEX_V( E->get().order, nodes.size(), ERR_BUG);
		nodes[E->get().order]=&E->get();
		E=E->next();
	}

	//connection set

	Map<_ConnectionKey,int> in_connection_map;
	Map<_ConnectionKey,List<int> > out_connection_map;
	Map<_ConnectionKey,int> in_node_map;
	Map<_ConnectionKey,List<int> > out_node_map;

	const List<Connection>::Element *CE=connections.front();
	i=0;
	while(CE) {
		const Connection &c = CE->get();

		_ConnectionKey in_k;
		in_k.node=node_map[c.dst_id].order;
		in_k.slot=c.dst_slot;
		in_connection_map[in_k]=i;
		in_node_map[in_k]=node_map[c.src_id].order;

		_ConnectionKey out_k;
		out_k.node=node_map[c.src_id].order;
		out_k.slot=c.src_slot;
		if (!out_connection_map.has(out_k))
			out_connection_map[out_k]=List<int>();
		out_connection_map[out_k].push_back(i);
		if(!out_node_map.has(out_k))
			out_node_map[out_k]=List<int>();
		out_node_map[out_k].push_back(node_map[c.dst_id].order);

		i++;
		CE=CE->next();
	}

	// validate nodes if they are connected to an output

	for(int i=nodes.size()-1;i>=0;i--) {

		if (VisualServer::shader_get_output_count(nodes[i]->type)==0) {
			// an actual graph output

			_ConnectionKey in_k;
			in_k.node=nodes[i]->order;
			in_k.slot=0;

			if (in_node_map.has(in_k)) {
				nodes[i]->out_valid=true;
			}
		} else {
			// regular node

			bool valid=false;
			for(int j=0;j<VS::shader_get_output_count(nodes[i]->type);j++) {

				_ConnectionKey key(nodes[i]->order,j);

				if (out_node_map.has(key)) {
					for(List<int>::Element *CE=out_node_map[key].front();CE;CE=CE->next()) {

						int to_node=CE->get();
						ERR_CONTINUE(to_node<0 || to_node >=nodes.size());
						if (nodes[to_node]->out_valid) {
							valid=true;
							break;
						}


					}
				}
				if (valid)
					break;

			}

			nodes[i]->out_valid=valid;
		}
	}

	// validate nodes if they are connected to an input

	for(int i=0;i<nodes.size();i++) {

		if (VisualServer::shader_get_input_count(nodes[i]->type)==0) {
			// an actual graph input

			int out_count=VisualServer::shader_get_output_count(nodes[i]->type);


			for(int j=0;j<out_count;j++) {

				_ConnectionKey out_k;
				out_k.node=nodes[i]->order;
				out_k.slot=j;
				if (out_node_map.has(out_k)) {
					nodes[i]->in_valid=true;
					break;
				}
			}

		} else {
			// regular node
			// this is very important.. for a node to be valid, all its inputs need to be valid
			bool valid=true;
			for(int j=0;j<VS::shader_get_input_count(nodes[i]->type);j++) {


				bool in_valid=false;
				_ConnectionKey key(nodes[i]->order,j);
				if (in_node_map.has(key)) {

					int from_node=in_node_map[key];
					ERR_CONTINUE(from_node<0 || from_node>=nodes.size());
					if (nodes[from_node]->in_valid)
						in_valid=true;

				}

				if (!in_valid) {
					valid=false;
					break;
				}

			}

			nodes[i]->in_valid=valid;
		}
	}

	// write code

	p_generator->begin();

	for(int i=0;i<nodes.size();i++) {


		if (!nodes[i]->out_valid || !nodes[i]->in_valid) // valid in both ways
			continue; // skip node

		Vector<int> in_indices;
		in_indices.resize(VS::shader_get_input_count(nodes[i]->type));
		Vector<int> out_indices;
		Vector<int> out_slot_indices;

		for(int j=0;j<in_indices.size();j++) {

			_ConnectionKey key(nodes[i]->order,j);
			if (in_connection_map.has(key))
				in_indices[j]=in_connection_map[key];
			else
				in_indices[j]=-1;
		}

		for(int j=0;j<VS::shader_get_output_count(nodes[i]->type);j++) {

			_ConnectionKey key(nodes[i]->order,j);
			if (out_connection_map.has(key)) {
				for(List<int>::Element *CE=out_connection_map[key].front();CE;CE=CE->next()) {

					out_indices.push_back(CE->get());
					out_slot_indices.push_back(j);
				}
			}
		}

		Error err = p_generator->add_node(nodes[i]->type,i,nodes[i]->id,nodes[i]->param,in_indices,out_indices,out_slot_indices);
		ERR_FAIL_COND_V( err, err );
	}

	p_generator->end();


	return OK;
}

void ShaderGraph::node_add(VS::ShaderNodeType p_type,int p_id) {


	ERR_FAIL_COND( node_map.has(p_id ) );
	ERR_FAIL_INDEX( p_type, VS::NODE_TYPE_MAX );
	Node node;

	node.type=p_type;
	node.id=p_id;
	node.x=0;
	node.y=0;

	node_map[p_id]=node;

}

void ShaderGraph::node_set_pos(int p_id, int p_x,int p_y) {

	ERR_FAIL_COND(!node_map.has(p_id));
	node_map[p_id].x=p_x;
	node_map[p_id].y=p_y;
}
int ShaderGraph::node_get_pos_x(int p_id) const {

	ERR_FAIL_COND_V(!node_map.has(p_id),-1);
	return node_map[p_id].x;
}
int ShaderGraph::node_get_pos_y(int p_id) const {

	ERR_FAIL_COND_V(!node_map.has(p_id),-1);
	return node_map[p_id].y;
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

void ShaderGraph::node_change_type(int p_id, VS::ShaderNodeType p_type) {

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


VS::ShaderNodeType ShaderGraph::node_get_type(int p_id) const {

	ERR_FAIL_COND_V(!node_map.has(p_id),VS::NODE_TYPE_MAX);
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
	VisualServer::ShaderNodeType type_src=node_map[p_src_id].type;
	VisualServer::ShaderNodeType type_dst=node_map[p_dst_id].type;
	ERR_FAIL_INDEX_V( p_src_slot, VisualServer::shader_get_output_count(type_src), ERR_INVALID_PARAMETER );
	ERR_FAIL_INDEX_V( p_dst_slot, VisualServer::shader_get_input_count(type_dst), ERR_INVALID_PARAMETER );
	ERR_FAIL_COND_V(VisualServer::shader_is_output_vector(type_src,p_src_slot) != VisualServer::shader_is_input_vector(type_dst,p_dst_slot), ERR_INVALID_PARAMETER );


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


void ShaderGraph::clear() {

	connections.clear();
	node_map.clear();
}

List<ShaderGraph::Connection> ShaderGraph::get_connection_list() const {

	return connections;

}

ShaderGraph::ShaderGraph() {


}


ShaderGraph::~ShaderGraph() {

}


#endif
