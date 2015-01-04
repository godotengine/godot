#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "scene/gui/container.h"

class GraphNode : public Container {

	OBJ_TYPE(GraphNode,Container);



	struct Slot {
		bool enable_left;
		int type_left;
		Color color_left;
		bool enable_right;
		int type_right;
		Color color_right;


		Slot() { enable_left=false; type_left=0; color_left=Color(1,1,1,1); enable_right=false; type_right=0; color_right=Color(1,1,1,1); };
	};

	String title;
	bool show_close;
	Vector2 offset;

	Rect2 close_rect;

	Vector<int> cache_y;

	struct ConnCache {
		Vector2 pos;
		int type;
		Color color;
	};

	Vector<ConnCache> conn_input_cache;
	Vector<ConnCache> conn_output_cache;

	Map<int,Slot> slot_info;

	bool connpos_dirty;

	void _connpos_update();
	void _resort();

	Vector2 drag_from;
	Vector2 drag_accum;
	bool dragging;
protected:

	void _input_event(const InputEvent& p_ev);
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:




	void set_slot(int p_idx,bool p_enable_left,int p_type_left,const Color& p_color_left, bool p_enable_right,int p_type_right,const Color& p_color_right);
	void clear_slot(int p_idx);
	void clear_all_slots();
	bool is_slot_enabled_left(int p_idx) const;
	int get_slot_type_left(int p_idx) const;
	Color get_slot_color_left(int p_idx) const;
	bool is_slot_enabled_right(int p_idx) const;
	int get_slot_type_right(int p_idx) const;
	Color get_slot_color_right(int p_idx) const;

	void set_title(const String& p_title);
	String get_title() const;

	void set_offset(const Vector2& p_offset);
	Vector2 get_offset() const;

	void set_show_close_button(bool p_enable);
	bool is_close_button_visible() const;

	int get_connection_input_count() ;
	int get_connection_output_count() ;
	Vector2 get_connection_input_pos(int p_idx);
	int get_connection_input_type(int p_idx);
	Color get_connection_input_color(int p_idx);
	Vector2 get_connection_output_pos(int p_idx);
	int get_connection_output_type(int p_idx);
	Color get_connection_output_color(int p_idx);


	virtual Size2 get_minimum_size() const;

	GraphNode();
};


#endif // GRAPH_NODE_H
