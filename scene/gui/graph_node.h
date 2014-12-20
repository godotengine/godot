#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "scene/gui/container.h"

class GraphNode : public Container {

	OBJ_TYPE(GraphNode,Container);


	String title;
	struct Slot {
		int type_left;
		int index_left;
		Color color_left;
		int type_right;
		int index_right;
		Color color_right;
	};

	Map<int,Slot> slot_info;

	void _resort();
protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	enum {
		TYPE_DISABLED=-1
	};


	void set_title(const String& p_title);
	String get_title() const;

	void set_slot(int p_idx,int p_type_left,int p_index_left,const Color& p_color_left, int p_type_right,int p_index_right,const Color& p_color_right);
	void clear_slot(int p_idx);
	void clear_all_slots();
	int get_slot_type_left(int p_idx) const;
	int get_slot_index_left(int p_idx) const;
	Color get_slot_color_left(int p_idx) const;
	int get_slot_type_right(int p_idx) const;
	int get_slot_index_right(int p_idx) const;
	Color get_slot_color_right(int p_idx) const;

	virtual Size2 get_minimum_size() const;

	GraphNode();
};


#endif // GRAPH_NODE_H
