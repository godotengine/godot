#include "graph_node.h"


void GraphNode::_resort() {



	int sep=get_constant("separation");
	Ref<StyleBox> sb=get_stylebox("frame");
	bool first=true;

	Size2 minsize;

	for(int i=0;i<get_child_count();i++) {
		Control *c=get_child(i)->cast_to<Control>();
		if (!c || !c->is_visible())
			continue;
		if (c->is_set_as_toplevel())
			continue;

		Size2i size=c->get_combined_minimum_size();

		minsize.y+=size.y;
		minsize.x=MAX(minsize.x,size.x);

		if (first)
			first=false;
		else
			minsize.y+=sep;

	}

	int vofs=0;
	int w = get_size().x - sb->get_minimum_size().x;


	for(int i=0;i<get_child_count();i++) {
		Control *c=get_child(i)->cast_to<Control>();
		if (!c || !c->is_visible())
			continue;
		if (c->is_set_as_toplevel())
			continue;

		Size2i size=c->get_combined_minimum_size();

		Rect2 r(sb->get_margin(MARGIN_LEFT),sb->get_margin(MARGIN_TOP)+vofs,w,size.y);

		fit_child_in_rect(c,r);


		if (vofs>0)
			vofs+=sep;
		vofs+=size.y;

	}

}


void GraphNode::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		Ref<StyleBox> sb=get_stylebox("frame");
		draw_style_box(sb,Rect2(Point2(),get_size()));
	}
	if (p_what==NOTIFICATION_SORT_CHILDREN) {

		_resort();
	}

}

void GraphNode::set_title(const String& p_title) {

	title=p_title;
	update();
}

String GraphNode::get_title() const {

	return title;
}

void GraphNode::set_slot(int p_idx,int p_type_left,int p_index_left,const Color& p_color_left, int p_type_right,int p_index_right,const Color& p_color_right) {

	ERR_FAIL_COND(p_idx<0);
	Slot s;
	s.type_left=p_type_left;
	s.color_left=p_color_left;
	s.index_left=p_index_left;
	s.type_right=p_type_right;
	s.color_right=p_color_right;
	s.index_right=p_index_right;
	slot_info[p_idx]=s;
	update();
}

void GraphNode::clear_slot(int p_idx){

	slot_info.erase(p_idx);
	update();
}
void GraphNode::clear_all_slots(){

	slot_info.clear();
	update();
}
int GraphNode::get_slot_type_left(int p_idx) const{

	if (!slot_info.has(p_idx))
		return TYPE_DISABLED;
	return slot_info[p_idx].type_left;

}
int GraphNode::get_slot_index_left(int p_idx) const{

	if (!slot_info.has(p_idx))
		return TYPE_DISABLED;
	return slot_info[p_idx].index_left;

}
Color GraphNode::get_slot_color_left(int p_idx) const{

	if (!slot_info.has(p_idx))
		return Color();
	return slot_info[p_idx].color_left;

}

int GraphNode::get_slot_type_right(int p_idx) const{

	if (!slot_info.has(p_idx))
		return TYPE_DISABLED;
	return slot_info[p_idx].type_right;

}
int GraphNode::get_slot_index_right(int p_idx) const{

	if (!slot_info.has(p_idx))
		return TYPE_DISABLED;
	return slot_info[p_idx].index_right;

}
Color GraphNode::get_slot_color_right(int p_idx) const{

	if (!slot_info.has(p_idx))
		return Color();
	return slot_info[p_idx].color_right;

}

Size2 GraphNode::get_minimum_size() const {

	int sep=get_constant("separation");
	Ref<StyleBox> sb=get_stylebox("frame");
	bool first=true;

	Size2 minsize;

	for(int i=0;i<get_child_count();i++) {

		Control *c=get_child(i)->cast_to<Control>();
		if (!c || !c->is_visible())
			continue;
		if (c->is_set_as_toplevel())
			continue;

		Size2i size=c->get_combined_minimum_size();

		minsize.y+=size.y;
		minsize.x=MAX(minsize.x,size.x);

		if (first)
			first=false;
		else
			minsize.y+=sep;
	}

	return minsize+sb->get_minimum_size();
}


void GraphNode::_bind_methods() {


}

GraphNode::GraphNode()
{
}
