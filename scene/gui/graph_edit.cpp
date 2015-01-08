#include "graph_edit.h"
#include "os/input.h"
#include "os/keyboard.h"
bool GraphEditFilter::has_point(const Point2& p_point) const {

	return ge->_filter_input(p_point);
}


GraphEditFilter::GraphEditFilter(GraphEdit *p_edit) {

	ge=p_edit;
}


Error GraphEdit::connect_node(const StringName& p_from, int p_from_port,const StringName& p_to,int p_to_port) {

	if (is_node_connected(p_from,p_from_port,p_to,p_to_port))
		return OK;
	Connection c;
	c.from=p_from;
	c.from_port=p_from_port;
	c.to=p_to;
	c.to_port=p_to_port;
	connections.push_back(c);
	top_layer->update();

	return OK;
}

bool GraphEdit::is_node_connected(const StringName& p_from, int p_from_port,const StringName& p_to,int p_to_port) {

	for(List<Connection>::Element *E=connections.front();E;E=E->next()) {

		if (E->get().from==p_from && E->get().from_port==p_from_port && E->get().to==p_to && E->get().to_port==p_to_port)
			return true;
	}

	return false;

}

void GraphEdit::disconnect_node(const StringName& p_from, int p_from_port,const StringName& p_to,int p_to_port){


	for(List<Connection>::Element *E=connections.front();E;E=E->next()) {

		if (E->get().from==p_from && E->get().from_port==p_from_port && E->get().to==p_to && E->get().to_port==p_to_port) {

			connections.erase(E);
			top_layer->update();
			return;
		}
	}
}

void GraphEdit::get_connection_list(List<Connection> *r_connections) const {

	*r_connections=connections;
}


void GraphEdit::_scroll_moved(double) {


	_update_scroll_offset();
	top_layer->update();
}

void GraphEdit::_update_scroll_offset() {

	for(int i=0;i<get_child_count();i++) {

		GraphNode *gn=get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		Point2 pos=gn->get_offset();
		pos-=Point2(h_scroll->get_val(),v_scroll->get_val());
		gn->set_pos(pos);
	}

}

void GraphEdit::_update_scroll() {

	if (updating)
		return;

	updating=true;
	Rect2 screen;
	for(int i=0;i<get_child_count();i++) {

		GraphNode *gn=get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		Rect2 r;
		r.pos=gn->get_offset();
		r.size=gn->get_size();
		screen = screen.merge(r);
	}

	screen.pos-=get_size();
	screen.size+=get_size()*2.0;


	h_scroll->set_min(screen.pos.x);
	h_scroll->set_max(screen.pos.x+screen.size.x);
	h_scroll->set_page(get_size().x);
	if (h_scroll->get_max() - h_scroll->get_min() <= h_scroll->get_page())
		h_scroll->hide();
	else
		h_scroll->show();

	v_scroll->set_min(screen.pos.y);
	v_scroll->set_max(screen.pos.y+screen.size.y);
	v_scroll->set_page(get_size().y);

	if (v_scroll->get_max() - v_scroll->get_min() <= v_scroll->get_page())
		v_scroll->hide();
	else
		v_scroll->show();

	_update_scroll_offset();
	updating=false;
}


void GraphEdit::_graph_node_raised(Node* p_gn) {

	GraphNode *gn=p_gn->cast_to<GraphNode>();
	ERR_FAIL_COND(!gn);
	gn->raise();
	top_layer->raise();

}


void GraphEdit::_graph_node_moved(Node *p_gn) {

	GraphNode *gn=p_gn->cast_to<GraphNode>();
	ERR_FAIL_COND(!gn);

	//gn->set_pos(gn->get_offset()+scroll_offset);

	top_layer->update();
}

void GraphEdit::add_child_notify(Node *p_child) {

	top_layer->call_deferred("raise"); //top layer always on top!
	GraphNode *gn = p_child->cast_to<GraphNode>();
	if (gn) {
		gn->connect("offset_changed",this,"_graph_node_moved",varray(gn));
		gn->connect("raise_request",this,"_graph_node_raised",varray(gn));
		_graph_node_moved(gn);
		gn->set_stop_mouse(false);
	}
}

void GraphEdit::remove_child_notify(Node *p_child) {

	top_layer->call_deferred("raise"); //top layer always on top!
	GraphNode *gn = p_child->cast_to<GraphNode>();
	if (gn) {
		gn->disconnect("offset_changed",this,"_graph_node_moved");
		gn->disconnect("raise_request",this,"_graph_node_raised");
	}
}

void GraphEdit::_notification(int p_what) {

	if (p_what==NOTIFICATION_READY) {
		Size2 size = top_layer->get_size();
		Size2 hmin = h_scroll->get_combined_minimum_size();
		Size2 vmin = v_scroll->get_combined_minimum_size();

		v_scroll->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,vmin.width);
		v_scroll->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,0);
		v_scroll->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,0);
		v_scroll->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,0);

		h_scroll->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,0);
		h_scroll->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,0);
		h_scroll->set_anchor_and_margin(MARGIN_TOP,ANCHOR_END,hmin.height);
		h_scroll->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,0);

	}
	if (p_what==NOTIFICATION_DRAW) {
		VS::get_singleton()->canvas_item_set_clip(get_canvas_item(),true);

	}

	if (p_what==NOTIFICATION_RESIZED) {
		_update_scroll();
		top_layer->update();
	}
}

bool GraphEdit::_filter_input(const Point2& p_point) {

	Ref<Texture> port =get_icon("port","GraphNode");

	float grab_r=port->get_width()*0.5;
	for(int i=get_child_count()-1;i>=0;i--) {

		GraphNode *gn=get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		for(int j=0;j<gn->get_connection_output_count();j++) {

			Vector2 pos = gn->get_connection_output_pos(j)+gn->get_pos();
			if (pos.distance_to(p_point)<grab_r)
				return true;


		}

		for(int j=0;j<gn->get_connection_input_count();j++) {

			Vector2 pos = gn->get_connection_input_pos(j)+gn->get_pos();
			if (pos.distance_to(p_point)<grab_r)
				return true;


		}

	}

	return false;
}

void GraphEdit::_top_layer_input(const InputEvent& p_ev) {

	if (p_ev.type==InputEvent::MOUSE_BUTTON && p_ev.mouse_button.button_index==BUTTON_LEFT && p_ev.mouse_button.pressed) {

		Ref<Texture> port =get_icon("port","GraphNode");
		Vector2 mpos(p_ev.mouse_button.x,p_ev.mouse_button.y);
		float grab_r=port->get_width()*0.5;
		for(int i=get_child_count()-1;i>=0;i--) {

			GraphNode *gn=get_child(i)->cast_to<GraphNode>();
			if (!gn)
				continue;

			for(int j=0;j<gn->get_connection_output_count();j++) {

				Vector2 pos = gn->get_connection_output_pos(j)+gn->get_pos();
				if (pos.distance_to(mpos)<grab_r) {

					connecting=true;
					connecting_from=gn->get_name();
					connecting_index=j;
					connecting_out=true;
					connecting_type=gn->get_connection_output_type(j);
					connecting_color=gn->get_connection_output_color(j);
					connecting_target=false;
					connecting_to=pos;
					return;
				}


			}

			for(int j=0;j<gn->get_connection_input_count();j++) {

				Vector2 pos = gn->get_connection_input_pos(j)+gn->get_pos();

				if (pos.distance_to(mpos)<grab_r) {

					if (right_disconnects) {
						//check disconnect
						for (List<Connection>::Element*E=connections.front();E;E=E->next()) {

							if (E->get().to==gn->get_name() && E->get().to_port==j) {

								Node*fr = get_node(String(E->get().from));
								if (fr && fr->cast_to<GraphNode>()) {

									connecting_from=E->get().from;
									connecting_index=E->get().from_port;
									connecting_out=true;
									connecting_type=fr->cast_to<GraphNode>()->get_connection_output_type(E->get().from_port);
									connecting_color=fr->cast_to<GraphNode>()->get_connection_output_color(E->get().from_port);
									connecting_target=false;
									connecting_to=pos;

									emit_signal("disconnection_request",E->get().from,E->get().from_port,E->get().to,E->get().to_port);
									fr = get_node(String(connecting_from)); //maybe it was erased
									if (fr && fr->cast_to<GraphNode>()) {
										connecting=true;
									}
									return;
								}

							}
						}
					}


					connecting=true;
					connecting_from=gn->get_name();
					connecting_index=j;
					connecting_out=false;
					connecting_type=gn->get_connection_input_type(j);
					connecting_color=gn->get_connection_input_color(j);
					connecting_target=false;
					connecting_to=pos;
					return;
				}


			}
		}
	}

	if (p_ev.type==InputEvent::MOUSE_MOTION && connecting) {

		connecting_to=Vector2(p_ev.mouse_motion.x,p_ev.mouse_motion.y);
		connecting_target=false;
		top_layer->update();

		Ref<Texture> port =get_icon("port","GraphNode");
		Vector2 mpos(p_ev.mouse_button.x,p_ev.mouse_button.y);
		float grab_r=port->get_width()*0.5;
		for(int i=get_child_count()-1;i>=0;i--) {

			GraphNode *gn=get_child(i)->cast_to<GraphNode>();
			if (!gn)
				continue;

			if (!connecting_out) {
				for(int j=0;j<gn->get_connection_output_count();j++) {

					Vector2 pos = gn->get_connection_output_pos(j)+gn->get_pos();
					int type =gn->get_connection_output_type(j);
					if (type==connecting_type && pos.distance_to(mpos)<grab_r) {

						connecting_target=true;
						connecting_to=pos;
						connecting_target_to=gn->get_name();
						connecting_target_index=j;
						return;
					}


				}
			} else {

				for(int j=0;j<gn->get_connection_input_count();j++) {

					Vector2 pos = gn->get_connection_input_pos(j)+gn->get_pos();
					int type =gn->get_connection_input_type(j);
					if (type==connecting_type && pos.distance_to(mpos)<grab_r) {
						connecting_target=true;
						connecting_to=pos;
						connecting_target_to=gn->get_name();
						connecting_target_index=j;
						return;
					}
				}
			}
		}
	}

	if (p_ev.type==InputEvent::MOUSE_BUTTON && p_ev.mouse_button.button_index==BUTTON_LEFT && !p_ev.mouse_button.pressed) {

		if (connecting && connecting_target) {

			String from = connecting_from;
			int from_slot = connecting_index;
			String to =connecting_target_to;
			int to_slot = connecting_target_index;

			if (!connecting_out) {
				SWAP(from,to);
				SWAP(from_slot,to_slot);
			}
			emit_signal("connection_request",from,from_slot,to,to_slot);

		}
		connecting=false;
		top_layer->update();

	}



}

void GraphEdit::_draw_cos_line(const Vector2& p_from, const Vector2& p_to,const Color& p_color) {

	static const int steps = 20;

	Rect2 r;
	r.pos=p_from;
	r.expand_to(p_to);
	Vector2 sign=Vector2((p_from.x < p_to.x) ? 1 : -1,(p_from.y < p_to.y) ? 1 : -1);
	bool flip = sign.x * sign.y < 0;

	Vector2 prev;
	for(int i=0;i<=steps;i++) {

		float d = i/float(steps);
		float c=-Math::cos(d*Math_PI) * 0.5+0.5;
		if (flip)
			c=1.0-c;
		Vector2 p = r.pos+Vector2(d*r.size.width,c*r.size.height);

		if (i>0) {

			top_layer->draw_line(prev,p,p_color,2);
		}

		prev=p;
	}
}

void GraphEdit::_top_layer_draw() {

	_update_scroll();

	if (connecting) {

		Node *fromn = get_node(connecting_from);
		ERR_FAIL_COND(!fromn);
		GraphNode *from = fromn->cast_to<GraphNode>();
		ERR_FAIL_COND(!from);
		Vector2 pos;
		if (connecting_out)
			pos=from->get_connection_output_pos(connecting_index);
		else
			pos=from->get_connection_input_pos(connecting_index);
		pos+=from->get_pos();

		Vector2 topos;
		topos=connecting_to;

		Color col=connecting_color;

		if (connecting_target) {
			col.r+=0.4;
			col.g+=0.4;
			col.b+=0.4;
		}
		_draw_cos_line(pos,topos,col);
	}

	List<List<Connection>::Element* > to_erase;
	for(List<Connection>::Element *E=connections.front();E;E=E->next()) {

		NodePath fromnp(E->get().from);

		Node * from = get_node(fromnp);
		if (!from) {
			to_erase.push_back(E);
			continue;
		}

		GraphNode *gfrom = from->cast_to<GraphNode>();

		if (!gfrom) {
			to_erase.push_back(E);
			continue;
		}

		NodePath tonp(E->get().to);
		Node * to = get_node(tonp);
		if (!to) {
			to_erase.push_back(E);
			continue;
		}

		GraphNode *gto = to->cast_to<GraphNode>();

		if (!gto) {
			to_erase.push_back(E);
			continue;
		}

		Vector2 frompos=gfrom->get_connection_output_pos(E->get().from_port)+gfrom->get_pos();
		Color color = gfrom->get_connection_output_color(E->get().from_port);
		Vector2 topos=gto->get_connection_input_pos(E->get().to_port)+gto->get_pos();
		_draw_cos_line(frompos,topos,color);

	}

	while(to_erase.size()) {
		connections.erase(to_erase.front()->get());
		to_erase.pop_front();
	}
	//draw connections
}

void GraphEdit::_input_event(const InputEvent& p_ev) {

	if (p_ev.type==InputEvent::MOUSE_MOTION && (p_ev.mouse_motion.button_mask&BUTTON_MASK_MIDDLE || (p_ev.mouse_motion.button_mask&BUTTON_MASK_LEFT && Input::get_singleton()->is_key_pressed(KEY_SPACE)))) {
		h_scroll->set_val( h_scroll->get_val() - p_ev.mouse_motion.relative_x );
		v_scroll->set_val( v_scroll->get_val() - p_ev.mouse_motion.relative_y );
	}
}

void GraphEdit::clear_connections() {

	connections.clear();
	update();
}


void GraphEdit::set_right_disconnects(bool p_enable) {

	right_disconnects=p_enable;
}

bool GraphEdit::is_right_disconnects_enabled() const{

	return right_disconnects;
}

Array GraphEdit::_get_connection_list() const {

	List<Connection> conns;
	get_connection_list(&conns);
	Array arr;
	for(List<Connection>::Element *E=conns.front();E;E=E->next()) {
		Dictionary d;
		d["from"]=E->get().from;
		d["from_port"]=E->get().from_port;
		d["to"]=E->get().to;
		d["to_port"]=E->get().to_port;
		arr.push_back(d);
	}
	return arr;
}
void GraphEdit::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("connect_node:Error","from","from_port","to","to_port"),&GraphEdit::connect_node);
	ObjectTypeDB::bind_method(_MD("is_node_connected","from","from_port","to","to_port"),&GraphEdit::is_node_connected);
	ObjectTypeDB::bind_method(_MD("disconnect_node","from","from_port","to","to_port"),&GraphEdit::disconnect_node);
	ObjectTypeDB::bind_method(_MD("get_connection_list"),&GraphEdit::_get_connection_list);

	ObjectTypeDB::bind_method(_MD("set_right_disconnects","enable"),&GraphEdit::set_right_disconnects);
	ObjectTypeDB::bind_method(_MD("is_right_disconnects_enabled"),&GraphEdit::is_right_disconnects_enabled);

	ObjectTypeDB::bind_method(_MD("_graph_node_moved"),&GraphEdit::_graph_node_moved);
	ObjectTypeDB::bind_method(_MD("_graph_node_raised"),&GraphEdit::_graph_node_raised);

	ObjectTypeDB::bind_method(_MD("_top_layer_input"),&GraphEdit::_top_layer_input);
	ObjectTypeDB::bind_method(_MD("_top_layer_draw"),&GraphEdit::_top_layer_draw);
	ObjectTypeDB::bind_method(_MD("_scroll_moved"),&GraphEdit::_scroll_moved);

	ObjectTypeDB::bind_method(_MD("_input_event"),&GraphEdit::_input_event);

	ADD_SIGNAL(MethodInfo("connection_request",PropertyInfo(Variant::STRING,"from"),PropertyInfo(Variant::INT,"from_slot"),PropertyInfo(Variant::STRING,"to"),PropertyInfo(Variant::INT,"to_slot")));
	ADD_SIGNAL(MethodInfo("disconnection_request",PropertyInfo(Variant::STRING,"from"),PropertyInfo(Variant::INT,"from_slot"),PropertyInfo(Variant::STRING,"to"),PropertyInfo(Variant::INT,"to_slot")));

}



GraphEdit::GraphEdit() {
	top_layer=NULL;
	top_layer=memnew(GraphEditFilter(this));
	add_child(top_layer);
	top_layer->set_stop_mouse(false);
	top_layer->set_area_as_parent_rect();
	top_layer->connect("draw",this,"_top_layer_draw");
	top_layer->set_stop_mouse(false);
	top_layer->connect("input_event",this,"_top_layer_input");

	h_scroll = memnew(HScrollBar);
	h_scroll->set_name("_h_scroll");
	top_layer->add_child(h_scroll);

	v_scroll = memnew(VScrollBar);
	v_scroll->set_name("_v_scroll");
	top_layer->add_child(v_scroll);
	updating=false;
	connecting=false;
	right_disconnects=false;

	h_scroll->connect("value_changed", this,"_scroll_moved");
	v_scroll->connect("value_changed", this,"_scroll_moved");
}
