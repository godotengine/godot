#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "scene/gui/container.h"

class GraphNode : public Container {

	OBJ_TYPE(GraphNode,Container);


	String title;
	struct Slot {
		bool enable_left;
		int type_left;
		Color color_left;
		bool enable_right;
		int type_right;
		Color color_right;


		Slot() { enable_left=false; type_left=0; color_left=Color(1,1,1,1); enable_right=false; type_right=0; color_right=Color(1,1,1,1); };
	};

	Vector<int> cache_y;

	Map<int,Slot> slot_info;

	void _resort();
protected:

	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:



	void set_title(const String& p_title);
	String get_title() const;

	void set_slot(int p_idx,bool p_enable_left,int p_type_left,const Color& p_color_left, bool p_enable_right,int p_type_right,const Color& p_color_right);
	void clear_slot(int p_idx);
	void clear_all_slots();
	bool is_slot_enabled_left(int p_idx) const;
	int get_slot_type_left(int p_idx) const;
	Color get_slot_color_left(int p_idx) const;
	bool is_slot_enabled_right(int p_idx) const;
	int get_slot_type_right(int p_idx) const;
	Color get_slot_color_right(int p_idx) const;

	virtual Size2 get_minimum_size() const;

	GraphNode();
};


#endif // GRAPH_NODE_H
