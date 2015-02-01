#ifndef GRAPH_EDIT_H
#define GRAPH_EDIT_H

#include "scene/gui/graph_node.h"
#include "scene/gui/scroll_bar.h"

class GraphEdit;

class GraphEditFilter : public Control {

	OBJ_TYPE(GraphEditFilter,Control);

friend class GraphEdit;
	GraphEdit *ge;
	virtual bool has_point(const Point2& p_point) const;

public:


	GraphEditFilter(GraphEdit *p_edit);
};

class GraphEdit : public Control {

	OBJ_TYPE(GraphEdit,Control);
public:

	struct Connection {
		StringName from;
		StringName to;
		int from_port;
		int to_port;

	};
private:

	HScrollBar* h_scroll;
	VScrollBar* v_scroll;


	bool connecting;
	String connecting_from;
	bool connecting_out;
	int connecting_index;
	int connecting_type;
	Color connecting_color;
	bool connecting_target;
	Vector2 connecting_to;
	String connecting_target_to;
	int connecting_target_index;



	bool right_disconnects;
	bool updating;
	List<Connection> connections;

	void _draw_cos_line(const Vector2& p_from, const Vector2& p_to,const Color& p_color);

	void _graph_node_raised(Node* p_gn);
	void _graph_node_moved(Node *p_gn);

	void _update_scroll();
	void _scroll_moved(double);
	void _input_event(const InputEvent& p_ev);

	GraphEditFilter *top_layer;
	void _top_layer_input(const InputEvent& p_ev);
	void _top_layer_draw();
	void _update_scroll_offset();

	Array _get_connection_list() const;

friend class GraphEditFilter;
	bool _filter_input(const Point2& p_point);
protected:

	static void _bind_methods();
	virtual void add_child_notify(Node *p_child);
	virtual void remove_child_notify(Node *p_child);
	void _notification(int p_what);

public:

	Error connect_node(const StringName& p_from, int p_from_port,const StringName& p_to,int p_to_port);
	bool is_node_connected(const StringName& p_from, int p_from_port,const StringName& p_to,int p_to_port);
	void disconnect_node(const StringName& p_from, int p_from_port,const StringName& p_to,int p_to_port);
	void clear_connections();

	GraphEditFilter *get_top_layer() const { return top_layer; }
	void get_connection_list(List<Connection> *r_connections) const;

	void set_right_disconnects(bool p_enable);
	bool is_right_disconnects_enabled() const;


	GraphEdit();
};

#endif // GRAPHEdit_H
