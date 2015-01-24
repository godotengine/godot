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

#if 0

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#include "servers/visual_server.h"
#include "map.h"


class ShaderCodeGenerator {
public:

	virtual void begin()=0;
	virtual Error add_node(VS::ShaderNodeType p_type,int p_node_pos,int p_id,const Variant& p_param,const Vector<int>& p_in_connections,const Vector<int>& p_out_connections,const Vector<int>& p_out_connection_outputs)=0;
	virtual void end()=0;
	
	virtual ~ShaderCodeGenerator() {}
};

class ShaderGraph {
public:


	struct Connection {

		int src_id;
		int src_slot;
		int dst_id;
		int dst_slot;
	};

private:
	struct Node {
	
		int16_t x,y;
		VS::ShaderNodeType type;
		Variant param;
		int id;
		mutable int order; // used for sorting
		mutable bool out_valid;
		mutable bool in_valid;
	};

	Map<int,Node> node_map;

	List<Connection> connections;
	
public:

	Error generate(ShaderCodeGenerator * p_generator) const;

	void node_add(VS::ShaderNodeType p_type,int p_id);
	void node_remove(int p_id);
	void node_change_type(int p_id, VS::ShaderNodeType p_type);
	void node_set_param(int p_id, const Variant& p_value);

	void node_set_pos(int p_id, int p_x,int p_y);
	int node_get_pos_x(int p_id) const;
	int node_get_pos_y(int p_id) const;
	
	void get_node_list(List<int> *p_node_list) const;
	void get_sorted_node_list(List<int> *p_node_list) const;
	VS::ShaderNodeType node_get_type(int p_id) const;
	Variant node_get_param(int p_id) const;

	Error connect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);
	bool is_connected(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot) const;
	void disconnect(int p_src_id,int p_src_slot, int p_dst_id,int p_dst_slot);	

	void clear();

	List<Connection> get_connection_list() const;


	ShaderGraph();	
	~ShaderGraph();

};
#endif
