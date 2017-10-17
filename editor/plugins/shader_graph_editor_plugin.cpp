/*************************************************************************/
/*  shader_graph_editor_plugin.cpp                                       */
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

// FIXME: Godot 3.0 broke compatibility with ShaderGraphEditorPlugin,
// it needs to be ported to the new shader language.
#if 0
#include "shader_graph_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "os/keyboard.h"
#include "scene/gui/check_box.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "spatial_editor_plugin.h"

void GraphColorRampEdit::_gui_input(const InputEvent& p_event) {

	if (p_event.type==InputEvent::KEY && p_event->is_pressed() && p_event->get_scancode()==KEY_DELETE && grabbed!=-1) {

		points.remove(grabbed);
		grabbed=-1;
		update();
		emit_signal("ramp_changed");
		accept_event();
	}

	if (p_event.type==InputEvent::MOUSE_BUTTON && p_event->get_button_index()==1 && p_event->is_pressed()) {

		update();
		int x = p_event->get_position().x;
		int total_w = get_size().width-get_size().height-3;
		if (x>total_w+3) {

			if (grabbed==-1)
				return;
			Size2 ms = Size2(350, picker->get_combined_minimum_size().height+10);
			picker->set_color(points[grabbed].color);
			popup->set_position(get_global_position()-Size2(0,ms.height));
			popup->set_size(ms);
			popup->popup();
			return;
		}


		float ofs = CLAMP(x/float(total_w),0,1);

		grabbed=-1;
		grabbing=true;
		int pos=-1;
		for(int i=0;i<points.size();i++) {

			if (ABS(x-points[i].offset*total_w)<4) {
				grabbed=i;
			}
			if (points[i].offset<ofs)
				pos=i;
		}

		grabbed_at=ofs;
		//grab or select
		if (grabbed!=-1) {
			return;
		}
		//insert


		Point p;
		p.offset=ofs;

		Point prev;
		Point next;

		if (pos==-1) {

			prev.color=Color(0,0,0);
			prev.offset=0;
			if (points.size()) {
				next=points[0];
			} else {
				next.color=Color(1,1,1);
				next.offset=1.0;
			}
		} else  {

			if (pos==points.size()-1) {
				next.color=Color(1,1,1);
				next.offset=1.0;
			} else {
				next=points[pos+1];
			}
			prev=points[pos];

		}

		p.color=prev.color.linear_interpolate(next.color,(p.offset-prev.offset)/(next.offset-prev.offset));

		points.push_back(p);
		points.sort();
		for(int i=0;i<points.size();i++) {
			if (points[i].offset==ofs) {
				grabbed=i;
				break;
			}
		}

		emit_signal("ramp_changed");

	}

	if (p_event.type==InputEvent::MOUSE_BUTTON && p_event->get_button_index()==1 && !p_event->is_pressed()) {

		if (grabbing) {
			grabbing=false;
			emit_signal("ramp_changed");
		}
		update();
	}

	if (p_event.type==InputEvent::MOUSE_MOTION && grabbing) {

		int total_w = get_size().width-get_size().height-3;

		int x = p_event.mouse_motion.x;
		float newofs = CLAMP(x/float(total_w),0,1);

		bool valid=true;
		for(int i=0;i<points.size();i++) {

			if (points[i].offset==newofs && i!=grabbed) {
				valid=false;
			}
		}

		if (!valid)
			return;

		points[grabbed].offset=newofs;

		points.sort();
		for(int i=0;i<points.size();i++) {
			if (points[i].offset==newofs) {
				grabbed=i;
				break;
			}
		}

		emit_signal("ramp_changed");

		update();
	}
}

void GraphColorRampEdit::_notification(int p_what){

	if (p_what==NOTIFICATION_ENTER_TREE) {
		if (!picker->is_connected("color_changed",this,"_color_changed")) {
			picker->connect("color_changed",this,"_color_changed");
		}
	}
	if (p_what==NOTIFICATION_DRAW) {


		Point prev;
		prev.offset=0;
		prev.color=Color(0,0,0);

		int h = get_size().y;
		int total_w = get_size().width-get_size().height-3;

		for(int i=-1;i<points.size();i++) {

			Point next;
			if (i+1==points.size()) {
				next.color=Color(1,1,1);
				next.offset=1;
			} else {
				next=points[i+1];
			}

			if (prev.offset==next.offset) {
				prev=next;
				continue;
			}

			Vector<Vector2> points;
			Vector<Color> colors;
			points.push_back(Vector2(prev.offset*total_w,h));
			points.push_back(Vector2(prev.offset*total_w,0));
			points.push_back(Vector2(next.offset*total_w,0));
			points.push_back(Vector2(next.offset*total_w,h));
			colors.push_back(prev.color);
			colors.push_back(prev.color);
			colors.push_back(next.color);
			colors.push_back(next.color);
			draw_primitive(points,colors,Vector<Point2>());
			prev=next;
		}

		for(int i=0;i<points.size();i++) {

			Color col=i==grabbed?Color(1,0.0,0.0,0.9):Color(1,1,1,0.8);

			draw_line(Vector2(points[i].offset*total_w,0),Vector2(points[i].offset*total_w,h-1),Color(0,0,0,0.7));
			draw_line(Vector2(points[i].offset*total_w-1,h/2),Vector2(points[i].offset*total_w-1,h-1),col);
			draw_line(Vector2(points[i].offset*total_w+1,h/2),Vector2(points[i].offset*total_w+1,h-1),col);
			draw_line(Vector2(points[i].offset*total_w-1,h/2),Vector2(points[i].offset*total_w+1,h/2),col);
			draw_line(Vector2(points[i].offset*total_w-1,h-1),Vector2(points[i].offset*total_w+1,h-1),col);

		}

		if (grabbed!=-1) {

			draw_rect(Rect2(total_w+3,0,h,h),points[grabbed].color);
		}

		if (has_focus()) {

			draw_line(Vector2(-1,-1),Vector2(total_w+1,-1),Color(1,1,1,0.6));
			draw_line(Vector2(total_w+1,-1),Vector2(total_w+1,h+1),Color(1,1,1,0.6));
			draw_line(Vector2(total_w+1,h+1),Vector2(-1,h+1),Color(1,1,1,0.6));
			draw_line(Vector2(-1,-1),Vector2(-1,h+1),Color(1,1,1,0.6));
		}

	}
}

Size2 GraphColorRampEdit::get_minimum_size() const {

	return Vector2(0,16);
}


void GraphColorRampEdit::_color_changed(const Color& p_color) {

	if (grabbed==-1)
		return;
	points[grabbed].color=p_color;
	update();
	emit_signal("ramp_changed");

}

void GraphColorRampEdit::set_ramp(const Vector<float>& p_offsets,const Vector<Color>& p_colors) {

	ERR_FAIL_COND(p_offsets.size()!=p_colors.size());
	points.clear();
	for(int i=0;i<p_offsets.size();i++) {
		Point p;
		p.offset=p_offsets[i];
		p.color=p_colors[i];
		points.push_back(p);
	}

	points.sort();
	update();
}

Vector<float> GraphColorRampEdit::get_offsets() const{
	Vector<float> ret;
	for(int i=0;i<points.size();i++)
		ret.push_back(points[i].offset);
	return ret;
}
Vector<Color> GraphColorRampEdit::get_colors() const{

	Vector<Color> ret;
	for(int i=0;i<points.size();i++)
		ret.push_back(points[i].color);
	return ret;
}


void GraphColorRampEdit::_bind_methods(){

	ClassDB::bind_method(D_METHOD("_gui_input"),&GraphColorRampEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("_color_changed"),&GraphColorRampEdit::_color_changed);
	ADD_SIGNAL(MethodInfo("ramp_changed"));
}

GraphColorRampEdit::GraphColorRampEdit(){

	grabbed=-1;
	grabbing=false;
	set_focus_mode(FOCUS_ALL);

	popup = memnew( PopupPanel );
	picker = memnew( ColorPicker );
	popup->add_child(picker);
	/popup->set_child_rect(picker);
	add_child(popup);

}
////////////

void GraphCurveMapEdit::_gui_input(const InputEvent& p_event) {

	if (p_event.type==InputEvent::KEY && p_event->is_pressed() && p_event->get_scancode()==KEY_DELETE && grabbed!=-1) {

		points.remove(grabbed);
		grabbed=-1;
		update();
		emit_signal("curve_changed");
		accept_event();
	}

	if (p_event.type==InputEvent::MOUSE_BUTTON && p_event->get_button_index()==1 && p_event->is_pressed()) {

		update();
		Point2 p = Vector2(p_event->get_position().x,p_event->get_position().y)/get_size();
		p.y=1.0-p.y;
		grabbed=-1;
		grabbing=true;

		for(int i=0;i<points.size();i++) {

			Vector2 ps = p*get_size();
			Vector2 pt = Vector2(points[i].offset,points[i].height)*get_size();
			if (ps.distance_to(pt)<4) {
				grabbed=i;
			}

		}


		//grab or select
		if (grabbed!=-1) {
			return;
		}
		//insert


		Point np;
		np.offset=p.x;
		np.height=p.y;

		points.push_back(np);
		points.sort();
		for(int i=0;i<points.size();i++) {
			if (points[i].offset==p.x && points[i].height==p.y) {
				grabbed=i;
				break;
			}
		}

		emit_signal("curve_changed");

	}

	if (p_event.type==InputEvent::MOUSE_BUTTON && p_event->get_button_index()==1 && !p_event->is_pressed()) {

		if (grabbing) {
			grabbing=false;
			emit_signal("curve_changed");
		}
		update();
	}

	if (p_event.type==InputEvent::MOUSE_MOTION && grabbing  && grabbed != -1) {

		Point2 p = Vector2(p_event->get_position().x,p_event->get_position().y)/get_size();
		p.y=1.0-p.y;

		p.x = CLAMP(p.x,0.0,1.0);
		p.y = CLAMP(p.y,0.0,1.0);

		bool valid=true;

		for(int i=0;i<points.size();i++) {

			if (points[i].offset==p.x && points[i].height==p.y && i!=grabbed) {
				valid=false;
			}
		}

		if (!valid)
			return;

		points[grabbed].offset=p.x;
		points[grabbed].height=p.y;

		points.sort();
		for(int i=0;i<points.size();i++) {
			if (points[i].offset==p.x && points[i].height==p.y) {
				grabbed=i;
				break;
			}
		}

		emit_signal("curve_changed");

		update();
	}
}

void GraphCurveMapEdit::_plot_curve(const Vector2& p_a,const Vector2& p_b,const Vector2& p_c,const Vector2& p_d) {

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

	int xmax=get_size().x;
	int ymax=get_size().y;

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
	tmp2[0][0] = 0;		 tmp2[0][1] = 0;		 tmp2[0][2] = 0;		tmp2[0][3] = 1;
	tmp2[1][0] = d3;		tmp2[1][1] = d2;		tmp2[1][2] = d;		tmp2[1][3] = 0;
	tmp2[2][0] = 6*d3;  tmp2[2][1] = 2*d2;  tmp2[2][2] = 0;		tmp2[2][3] = 0;
	tmp2[3][0] = 6*d3;  tmp2[3][1] = 0;		 tmp2[3][2] = 0;		tmp2[3][3] = 0;

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

	/*	if (fix255)
		{
				cd->curve[cd->outline][lastx] = lasty;
		}
		else
		{
				cd->curve_ptr[cd->outline][lastx] = lasty;
				if(gb_debug) printf("bender_plot_curve xmax:%d ymax:%d\n", (int)xmax, (int)ymax);
		}
*/
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
		if ((lastx != newx) || (lasty != newy)) {
			draw_line(Vector2(lastx,ymax-lasty),Vector2(newx,ymax-newy),Color(0.8,0.8,0.8,0.8),2.0);
		}

		lastx = newx;
		lasty = newy;
	}
}


void GraphCurveMapEdit::_notification(int p_what){

	if (p_what==NOTIFICATION_DRAW) {

		draw_style_box(get_stylebox("bg","Tree"),Rect2(Point2(),get_size()));

		int w = get_size().x;
		int h = get_size().y;

		Vector2 prev=Vector2(0,0);
		Vector2 prev2=Vector2(0,0);

		for(int i=-1;i<points.size();i++) {

			Vector2 next;
			Vector2 next2;
			if (i+1>=points.size()) {
				next=Vector2(1,1);
			} else {
				next=Vector2(points[i+1].offset,points[i+1].height);
			}

			if (i+2>=points.size()) {
				next2=Vector2(1,1);
			} else {
				next2=Vector2(points[i+2].offset,points[i+2].height);
			}

			/*if (i==-1 && prev.offset==next.offset) {
								prev=next;
								continue;
						}*/

			_plot_curve(prev2,prev,next,next2);

			prev2=prev;
			prev=next;
		}

		for(int i=0;i<points.size();i++) {

			Color col=i==grabbed?Color(1,0.0,0.0,0.9):Color(1,1,1,0.8);


			draw_rect(Rect2( Vector2(points[i].offset,1.0-points[i].height)*get_size()-Vector2(2,2),Vector2(5,5)),col);
		}

		/*		if (grabbed!=-1) {

						draw_rect(Rect2(total_w+3,0,h,h),points[grabbed].color);
				}
*/
		if (has_focus()) {

			draw_line(Vector2(-1,-1),Vector2(w+1,-1),Color(1,1,1,0.6));
			draw_line(Vector2(w+1,-1),Vector2(w+1,h+1),Color(1,1,1,0.6));
			draw_line(Vector2(w+1,h+1),Vector2(-1,h+1),Color(1,1,1,0.6));
			draw_line(Vector2(-1,-1),Vector2(-1,h+1),Color(1,1,1,0.6));
		}

	}
}

Size2 GraphCurveMapEdit::get_minimum_size() const {

	return Vector2(64,64);
}



void GraphCurveMapEdit::set_points(const Vector<Vector2>& p_points) {


	points.clear();
	for(int i=0;i<p_points.size();i++) {
		Point p;
		p.offset=p_points[i].x;
		p.height=p_points[i].y;
		points.push_back(p);
	}

	points.sort();
	update();
}

Vector<Vector2> GraphCurveMapEdit::get_points() const {
	Vector<Vector2> ret;
	for(int i=0;i<points.size();i++)
		ret.push_back(Vector2(points[i].offset,points[i].height));
	return ret;
}

void GraphCurveMapEdit::_bind_methods(){

	ClassDB::bind_method(D_METHOD("_gui_input"),&GraphCurveMapEdit::_gui_input);
	ADD_SIGNAL(MethodInfo("curve_changed"));
}

GraphCurveMapEdit::GraphCurveMapEdit(){

	grabbed=-1;
	grabbing=false;
	set_focus_mode(FOCUS_ALL);

}


////cbacks
///
void ShaderGraphView::_scalar_const_changed(double p_value,int p_id) {

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Scalar Constant"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"scalar_const_node_set_value",type,p_id,p_value);
	ur->add_undo_method(graph.ptr(),"scalar_const_node_set_value",type,p_id,graph->scalar_const_node_get_value(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}

void ShaderGraphView::_vec_const_changed(double p_value, int p_id,Array p_arr){

	Vector3 val;
	for(int i=0;i<p_arr.size();i++) {
		val[i]=p_arr[i].call("get_val");
	}

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Vec Constant"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"vec_const_node_set_value",type,p_id,val);
	ur->add_undo_method(graph.ptr(),"vec_const_node_set_value",type,p_id,graph->vec_const_node_get_value(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_rgb_const_changed(const Color& p_color, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change RGB Constant"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"rgb_const_node_set_value",type,p_id,p_color);
	ur->add_undo_method(graph.ptr(),"rgb_const_node_set_value",type,p_id,graph->rgb_const_node_get_value(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_scalar_op_changed(int p_op, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Scalar Operator"));
	ur->add_do_method(graph.ptr(),"scalar_op_node_set_op",type,p_id,p_op);
	ur->add_undo_method(graph.ptr(),"scalar_op_node_set_op",type,p_id,graph->scalar_op_node_get_op(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_vec_op_changed(int p_op, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Vec Operator"));
	ur->add_do_method(graph.ptr(),"vec_op_node_set_op",type,p_id,p_op);
	ur->add_undo_method(graph.ptr(),"vec_op_node_set_op",type,p_id,graph->vec_op_node_get_op(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}
void ShaderGraphView::_vec_scalar_op_changed(int p_op, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Vec Scalar Operator"));
	ur->add_do_method(graph.ptr(),"vec_scalar_op_node_set_op",type,p_id,p_op);
	ur->add_undo_method(graph.ptr(),"vec_scalar_op_node_set_op",type,p_id,graph->vec_scalar_op_node_get_op(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_rgb_op_changed(int p_op, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change RGB Operator"));
	ur->add_do_method(graph.ptr(),"rgb_op_node_set_op",type,p_id,p_op);
	ur->add_undo_method(graph.ptr(),"rgb_op_node_set_op",type,p_id,graph->rgb_op_node_get_op(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}
void ShaderGraphView::_xform_inv_rev_changed(bool p_enabled, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Toggle Rot Only"));
	ur->add_do_method(graph.ptr(),"xform_vec_mult_node_set_no_translation",type,p_id,p_enabled);
	ur->add_undo_method(graph.ptr(),"xform_vec_mult_node_set_no_translation",type,p_id,graph->xform_vec_mult_node_get_no_translation(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}
void ShaderGraphView::_scalar_func_changed(int p_func, int p_id){


	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Scalar Function"));
	ur->add_do_method(graph.ptr(),"scalar_func_node_set_function",type,p_id,p_func);
	ur->add_undo_method(graph.ptr(),"scalar_func_node_set_function",type,p_id,graph->scalar_func_node_get_function(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}
void ShaderGraphView::_vec_func_changed(int p_func, int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Vec Function"));
	ur->add_do_method(graph.ptr(),"vec_func_node_set_function",type,p_id,p_func);
	ur->add_undo_method(graph.ptr(),"vec_func_node_set_function",type,p_id,graph->vec_func_node_get_function(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_scalar_input_changed(double p_value,int p_id){

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Scalar Uniform"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"scalar_input_node_set_value",type,p_id,p_value);
	ur->add_undo_method(graph.ptr(),"scalar_input_node_set_value",type,p_id,graph->scalar_input_node_get_value(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_vec_input_changed(double p_value, int p_id,Array p_arr){

	Vector3 val;
	for(int i=0;i<p_arr.size();i++) {
		val[i]=p_arr[i].call("get_val");
	}

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Vec Uniform"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"vec_input_node_set_value",type,p_id,val);
	ur->add_undo_method(graph.ptr(),"vec_input_node_set_value",type,p_id,graph->vec_input_node_get_value(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}
void ShaderGraphView::_xform_input_changed(int p_id, Node *p_button){


	ToolButton *tb = Object::cast_to<ToolButton>(p_button);
	ped_popup->set_position(tb->get_global_position()+Vector2(0,tb->get_size().height));
	ped_popup->set_size(tb->get_size());
	edited_id=p_id;
	edited_def=-1;
	ped_popup->edit(NULL,"",Variant::TRANSFORM,graph->xform_input_node_get_value(type,p_id),PROPERTY_HINT_NONE,"");
	ped_popup->popup();

}
void ShaderGraphView::_xform_const_changed(int p_id, Node *p_button){

	ToolButton *tb = Object::cast_to<ToolButton>(p_button);
	ped_popup->set_position(tb->get_global_position()+Vector2(0,tb->get_size().height));
	ped_popup->set_size(tb->get_size());
	edited_id=p_id;
	edited_def=-1;
	ped_popup->edit(NULL,"",Variant::TRANSFORM,graph->xform_const_node_get_value(type,p_id),PROPERTY_HINT_NONE,"");
	ped_popup->popup();

}

void ShaderGraphView::_rgb_input_changed(const Color& p_color, int p_id){


	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change RGB Uniform"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"rgb_input_node_set_value",type,p_id,p_color);
	ur->add_undo_method(graph.ptr(),"rgb_input_node_set_value",type,p_id,graph->rgb_input_node_get_value(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}
void ShaderGraphView::_tex_input_change(int p_id, Node *p_button){


}
void ShaderGraphView::_cube_input_change(int p_id){


}

void ShaderGraphView::_variant_edited() {

	if (edited_def != -1) {

		Variant v = ped_popup->get_variant();
		Variant v2 = graph->default_get_value(type,edited_id,edited_def);
		if (v2.get_type() == Variant::NIL)
			switch (v.get_type()) {
			case Variant::VECTOR3:
				v2=Vector3();
				break;
			case Variant::REAL:
				v2=0.0;
				break;
			case Variant::TRANSFORM:
				v2=Transform();
				break;
			case Variant::COLOR:
				v2=Color();
				break;
			default: {}
			}
		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Default Value"));
		ur->add_do_method(graph.ptr(),"default_set_value",type,edited_id,edited_def, v);
		ur->add_undo_method(graph.ptr(),"default_set_value",type,edited_id,edited_def, v2);
		ur->add_do_method(this,"_update_graph");
		ur->add_undo_method(this,"_update_graph");
		ur->commit_action();
		return;
	}

	if (graph->node_get_type(type,edited_id)==ShaderGraph::NODE_XFORM_CONST) {

		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change XForm Uniform"));
		ur->add_do_method(graph.ptr(),"xform_const_node_set_value",type,edited_id,ped_popup->get_variant());
		ur->add_undo_method(graph.ptr(),"xform_const_node_set_value",type,edited_id,graph->xform_const_node_get_value(type,edited_id));
		ur->add_do_method(this,"_update_graph");
		ur->add_undo_method(this,"_update_graph");
		ur->commit_action();
	}


	if (graph->node_get_type(type,edited_id)==ShaderGraph::NODE_XFORM_INPUT) {

		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change XForm Uniform"));
		ur->add_do_method(graph.ptr(),"xform_input_node_set_value",type,edited_id,ped_popup->get_variant());
		ur->add_undo_method(graph.ptr(),"xform_input_node_set_value",type,edited_id,graph->xform_input_node_get_value(type,edited_id));
		ur->add_do_method(this,"_update_graph");
		ur->add_undo_method(this,"_update_graph");
		ur->commit_action();
	}

	if (graph->node_get_type(type,edited_id)==ShaderGraph::NODE_TEXTURE_INPUT) {

		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Texture Uniform"));
		ur->add_do_method(graph.ptr(),"texture_input_node_set_value",type,edited_id,ped_popup->get_variant());
		ur->add_undo_method(graph.ptr(),"texture_input_node_set_value",type,edited_id,graph->texture_input_node_get_value(type,edited_id));
		ur->add_do_method(this,"_update_graph");
		ur->add_undo_method(this,"_update_graph");
		ur->commit_action();
	}

	if (graph->node_get_type(type,edited_id)==ShaderGraph::NODE_CUBEMAP_INPUT) {

		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Cubemap Uniform"));
		ur->add_do_method(graph.ptr(),"cubemap_input_node_set_value",type,edited_id,ped_popup->get_variant());
		ur->add_undo_method(graph.ptr(),"cubemap_input_node_set_value",type,edited_id,graph->cubemap_input_node_get_value(type,edited_id));
		ur->add_do_method(this,"_update_graph");
		ur->add_undo_method(this,"_update_graph");
		ur->commit_action();
	}

}

void ShaderGraphView::_comment_edited(int p_id,Node* p_button) {

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	TextEdit *te=Object::cast_to<TextEdit>(p_button);
	ur->create_action(TTR("Change Comment"),UndoRedo::MERGE_ENDS);
	ur->add_do_method(graph.ptr(),"comment_node_set_text",type,p_id,te->get_text());
	ur->add_undo_method(graph.ptr(),"comment_node_set_text",type,p_id,graph->comment_node_get_text(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;

}

void ShaderGraphView::_color_ramp_changed(int p_id,Node* p_ramp) {

	GraphColorRampEdit *cr=Object::cast_to<GraphColorRampEdit>(p_ramp);

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();


	Vector<float> offsets=cr->get_offsets();
	Vector<Color> colors=cr->get_colors();

	PoolVector<float> new_offsets;
	PoolVector<Color> new_colors;
	{
		new_offsets.resize(offsets.size());
		new_colors.resize(colors.size());
		PoolVector<float>::Write ow=new_offsets.write();
		PoolVector<Color>::Write cw=new_colors.write();
		for(int i=0;i<new_offsets.size();i++) {
			ow[i]=offsets[i];
			cw[i]=colors[i];
		}

	}


	PoolVector<float> old_offsets=graph->color_ramp_node_get_offsets(type,p_id);
	PoolVector<Color> old_colors=graph->color_ramp_node_get_colors(type,p_id);

	if (old_offsets.size()!=new_offsets.size())
		ur->create_action(TTR("Add/Remove to Color Ramp"));
	else
		ur->create_action(TTR("Modify Color Ramp"),UndoRedo::MERGE_ENDS);

	ur->add_do_method(graph.ptr(),"color_ramp_node_set_ramp",type,p_id,new_colors,new_offsets);
	ur->add_undo_method(graph.ptr(),"color_ramp_node_set_ramp",type,p_id,old_colors,old_offsets);
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}

void ShaderGraphView::_curve_changed(int p_id,Node* p_curve) {

	GraphCurveMapEdit *cr=Object::cast_to<GraphCurveMapEdit>(p_curve);

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();


	Vector<Point2> points=cr->get_points();

	PoolVector<Vector2> new_points;
	{
		new_points.resize(points.size());
		PoolVector<Vector2>::Write ow=new_points.write();
		for(int i=0;i<new_points.size();i++) {
			ow[i]=points[i];
		}

	}


	PoolVector<Vector2> old_points=graph->curve_map_node_get_points(type,p_id);

	if (old_points.size()!=new_points.size())
		ur->create_action(TTR("Add/Remove to Curve Map"));
	else
		ur->create_action(TTR("Modify Curve Map"),UndoRedo::MERGE_ENDS);

	ur->add_do_method(graph.ptr(),"curve_map_node_set_points",type,p_id,new_points);
	ur->add_undo_method(graph.ptr(),"curve_map_node_set_points",type,p_id,old_points);
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
}


void ShaderGraphView::_input_name_changed(const String& p_name, int p_id, Node *p_line_edit) {

	LineEdit *le=Object::cast_to<LineEdit>(p_line_edit);
	ERR_FAIL_COND(!le);

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Input Name"));
	ur->add_do_method(graph.ptr(),"input_node_set_name",type,p_id,p_name);
	ur->add_undo_method(graph.ptr(),"input_node_set_name",type,p_id,graph->input_node_get_name(type,p_id));
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	block_update=true;
	ur->commit_action();
	block_update=false;
	le->set_text(graph->input_node_get_name(type,p_id));
}

void ShaderGraphView::_tex_edited(int p_id,Node* p_button) {

	ToolButton *tb = Object::cast_to<ToolButton>(p_button);
	ped_popup->set_position(tb->get_global_position()+Vector2(0,tb->get_size().height));
	ped_popup->set_size(tb->get_size());
	edited_id=p_id;
	edited_def=-1;
	ped_popup->edit(NULL,"",Variant::OBJECT,graph->texture_input_node_get_value(type,p_id),PROPERTY_HINT_RESOURCE_TYPE,"Texture");
}

void ShaderGraphView::_cube_edited(int p_id,Node* p_button) {

	ToolButton *tb = Object::cast_to<ToolButton>(p_button);
	ped_popup->set_position(tb->get_global_position()+Vector2(0,tb->get_size().height));
	ped_popup->set_size(tb->get_size());
	edited_id=p_id;
	edited_def=-1;
	ped_popup->edit(NULL,"",Variant::OBJECT,graph->cubemap_input_node_get_value(type,p_id),PROPERTY_HINT_RESOURCE_TYPE,"CubeMap");
}


//////////////view/////////////


void ShaderGraphView::_connection_request(const String& p_from, int p_from_slot,const String& p_to,int p_to_slot) {

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();

	int from_idx=-1;
	int to_idx=-1;
	for (Map<int,GraphNode*>::Element *E=node_map.front();E;E=E->next()) {

		if (p_from==E->get()->get_name())
			from_idx=E->key();
		if (p_to==E->get()->get_name())
			to_idx=E->key();
	}

	ERR_FAIL_COND(from_idx==-1);
	ERR_FAIL_COND(to_idx==-1);

	ur->create_action(TTR("Connect Graph Nodes"));

	List<ShaderGraph::Connection> conns;

	graph->get_node_connections(type,&conns);
	//disconnect/reconnect dependencies
	ur->add_undo_method(graph.ptr(),"disconnect_node",type,from_idx,p_from_slot,to_idx,p_to_slot);
	for(List<ShaderGraph::Connection>::Element *E=conns.front();E;E=E->next()) {

		if (E->get().dst_id==to_idx && E->get().dst_slot==p_to_slot) {
			ur->add_do_method(graph.ptr(),"disconnect_node",type,E->get().src_id,E->get().src_slot,E->get().dst_id,E->get().dst_slot);
			ur->add_undo_method(graph.ptr(),"connect_node",type,E->get().src_id,E->get().src_slot,E->get().dst_id,E->get().dst_slot);
		}
	}
	ur->add_do_method(graph.ptr(),"connect_node",type,from_idx,p_from_slot,to_idx,p_to_slot);
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	ur->commit_action();


}

void ShaderGraphView::_disconnection_request(const String& p_from, int p_from_slot,const String& p_to,int p_to_slot) {

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();

	int from_idx=-1;
	int to_idx=-1;
	for (Map<int,GraphNode*>::Element *E=node_map.front();E;E=E->next()) {

		if (p_from==E->get()->get_name())
			from_idx=E->key();
		if (p_to==E->get()->get_name())
			to_idx=E->key();
	}

	ERR_FAIL_COND(from_idx==-1);
	ERR_FAIL_COND(to_idx==-1);

	if (!graph->is_node_connected(type,from_idx,p_from_slot,to_idx,p_to_slot))
		return; //nothing to disconnect

	ur->create_action(TTR("Disconnect Graph Nodes"));

	List<ShaderGraph::Connection> conns;

	graph->get_node_connections(type,&conns);
	//disconnect/reconnect dependencies
	ur->add_do_method(graph.ptr(),"disconnect_node",type,from_idx,p_from_slot,to_idx,p_to_slot);
	ur->add_undo_method(graph.ptr(),"connect_node",type,from_idx,p_from_slot,to_idx,p_to_slot);
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	ur->commit_action();


}

void ShaderGraphView::_node_removed(int p_id) {

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Remove Shader Graph Node"));

	ur->add_do_method(graph.ptr(),"node_remove",type,p_id);
	ur->add_undo_method(graph.ptr(),"node_add",type,graph->node_get_type(type,p_id),p_id);
	ur->add_undo_method(graph.ptr(),"node_set_state",type,p_id,graph->node_get_state(type,p_id));
	List<ShaderGraph::Connection> conns;

	graph->get_node_connections(type,&conns);
	for(List<ShaderGraph::Connection>::Element *E=conns.front();E;E=E->next()) {

		if (E->get().dst_id==p_id || E->get().src_id==p_id) {
			ur->add_undo_method(graph.ptr(),"connect_node",type,E->get().src_id,E->get().src_slot,E->get().dst_id,E->get().dst_slot);
		}
	}
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	ur->commit_action();

}

void ShaderGraphView::_begin_node_move()
{
	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Move Shader Graph Node"));
}

void ShaderGraphView::_node_moved(const Vector2& p_from, const Vector2& p_to,int p_id) {


	ERR_FAIL_COND(!node_map.has(p_id));
	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->add_do_method(this,"_move_node",p_id,p_to);
	ur->add_undo_method(this,"_move_node",p_id,p_from);
}

void ShaderGraphView::_end_node_move()
{
	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->commit_action();
}

void ShaderGraphView::_move_node(int p_id,const Vector2& p_to) {

	ERR_FAIL_COND(!node_map.has(p_id));
	node_map[p_id]->set_offset(p_to);
	graph->node_set_position(type,p_id,p_to);
}

void ShaderGraphView::_duplicate_nodes_request()
{
	Array s_id;

	for(Map<int,GraphNode*>::Element *E=node_map.front();E;E=E->next()) {
		ShaderGraph::NodeType t=graph->node_get_type(type, E->key());
		if (t==ShaderGraph::NODE_OUTPUT || t==ShaderGraph::NODE_INPUT)
			continue;
		GraphNode *gn = E->get();
		if (gn && gn->is_selected())
			s_id.push_back(E->key());
	}

	if (s_id.size()==0)
		return;

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Duplicate Graph Node(s)"));
	ur->add_do_method(this,"_duplicate_nodes",s_id);
	List<int> n_ids = graph->generate_ids(type, s_id.size());
	for (List<int>::Element *E=n_ids.front();E;E=E->next())
		ur->add_undo_method(graph.ptr(),"node_remove",type,E->get());
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	ur->commit_action();

}

void ShaderGraphView::_duplicate_nodes(const Array &p_nodes)
{
	List<int> n = List<int>();
	for (int i=0; i<p_nodes.size();i++)
		n.push_back(p_nodes.get(i));
	graph->duplicate_nodes(type, n);
	call_deferred("_update_graph");
}

void ShaderGraphView::_delete_nodes_request()
{
	List<int> s_id=List<int>();

	for(Map<int,GraphNode*>::Element *E=node_map.front();E;E=E->next()) {
		ShaderGraph::NodeType t=graph->node_get_type(type, E->key());
		if (t==ShaderGraph::NODE_OUTPUT)
			continue;
		GraphNode *gn = E->get();
		if (gn && gn->is_selected())
			s_id.push_back(E->key());
	}

	if (s_id.size()==0)
		return;

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Delete Shader Graph Node(s)"));

	for (List<int>::Element *N=s_id.front();N;N=N->next()) {
		ur->add_do_method(graph.ptr(),"node_remove",type,N->get());
		ur->add_undo_method(graph.ptr(),"node_add",type,graph->node_get_type(type,N->get()),N->get());
		ur->add_undo_method(graph.ptr(),"node_set_state",type,N->get(),graph->node_get_state(type,N->get()));
		List<ShaderGraph::Connection> conns;

		graph->get_node_connections(type,&conns);
		for(List<ShaderGraph::Connection>::Element *E=conns.front();E;E=E->next()) {

			if (E->get().dst_id==N->get() || E->get().src_id==N->get()) {
				ur->add_undo_method(graph.ptr(),"connect_node",type,E->get().src_id,E->get().src_slot,E->get().dst_id,E->get().dst_slot);
			}
		}
	}
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	ur->commit_action();

}

void ShaderGraphView::_default_changed(int p_id, Node *p_button, int p_param, int v_type, String p_hint)
{
	ToolButton *tb = Object::cast_to<ToolButton>(p_button);
	ped_popup->set_position(tb->get_global_position()+Vector2(0,tb->get_size().height));
	ped_popup->set_size(tb->get_size());
	edited_id=p_id;
	edited_def=p_param;
	Variant::Type vt = (Variant::Type)v_type;
	Variant v = graph->default_get_value(type,p_id,edited_def);
	int h=PROPERTY_HINT_NONE;
	if (v.get_type() == Variant::NIL)
		switch (vt) {
		case Variant::VECTOR3:
			v=Vector3();
			break;
		case Variant::REAL:
			h=PROPERTY_HINT_RANGE;
			v=0.0;
			break;
		case Variant::TRANSFORM:
			v=Transform();
			break;
		case Variant::COLOR:
			h=PROPERTY_HINT_COLOR_NO_ALPHA;
			v=Color();
			break;
		default: {}
		}

	ped_popup->edit(NULL,"",vt,v,h,p_hint);

	ped_popup->popup();
}

ToolButton *ShaderGraphView::make_label(String text, Variant::Type v_type) {
	ToolButton *l = memnew( ToolButton );
	l->set_text(text);
	l->set_text_align(ToolButton::ALIGN_LEFT);
	l->add_style_override("hover", l->get_stylebox("normal", "ToolButton"));
	l->add_style_override("pressed", l->get_stylebox("normal", "ToolButton"));
	l->add_style_override("focus", l->get_stylebox("normal", "ToolButton"));
	switch (v_type) {
	case Variant::REAL:
		l->set_icon(ped_popup->get_icon("Real", "EditorIcons"));
		break;
	case Variant::VECTOR3:
		l->set_icon(ped_popup->get_icon("Vector", "EditorIcons"));
		break;
	case Variant::TRANSFORM:
		l->set_icon(ped_popup->get_icon("Matrix", "EditorIcons"));
		break;
	case Variant::COLOR:
		l->set_icon(ped_popup->get_icon("Color", "EditorIcons"));
		break;
	default: {}
	}
	return l;
}

ToolButton *ShaderGraphView::make_editor(String text,GraphNode* gn,int p_id,int param,Variant::Type v_type, String p_hint) {
	ToolButton *edit = memnew( ToolButton );
	edit->set_text(text);
	edit->set_text_align(ToolButton::ALIGN_LEFT);
	edit->set_flat(false);
	edit->add_style_override("normal", gn->get_stylebox("defaultframe", "GraphNode"));
	edit->add_style_override("hover", gn->get_stylebox("defaultframe", "GraphNode"));
	edit->add_style_override("pressed", gn->get_stylebox("defaultframe", "GraphNode"));
	edit->add_style_override("focus", gn->get_stylebox("defaultfocus", "GraphNode"));
	edit->connect("pressed",this,"_default_changed",varray(p_id,edit,param,v_type,p_hint));

	switch (v_type) {
	case Variant::REAL:
		edit->set_icon(ped_popup->get_icon("Real", "EditorIcons"));
		break;
	case Variant::VECTOR3:
		edit->set_icon(ped_popup->get_icon("Vector", "EditorIcons"));
		break;
	case Variant::TRANSFORM:
		edit->set_icon(ped_popup->get_icon("Matrix", "EditorIcons"));
		break;
	case Variant::COLOR: {
		Image icon_color = Image(15,15,false,Image::FORMAT_RGB8);
		Color c = graph->default_get_value(type,p_id,param);
		for (int x=1;x<14;x++)
			for (int y=1;y<14;y++)
				icon_color.set_pixel(x,y,c);
		Ref<ImageTexture> t;
		t.instance();
		t->create_from_image(icon_color);
		edit->set_icon(t);
	} break;
	default: {}
	}
	return edit;
}

void ShaderGraphView::_create_node(int p_id) {


	GraphNode *gn = memnew( GraphNode );
	gn->set_show_close_button(true);
	Color typecol[4]={
		Color(0.9,0.4,1),
		Color(0.8,1,0.2),
		Color(1,0.2,0.2),
		Color(0,1,1)
	};

	const String hint_spin = "-65536,65535,0.001";
	const String hint_slider = "0.0,1.0,0.01,slider";


	switch(graph->node_get_type(type,p_id)) {

	case ShaderGraph::NODE_INPUT: {

		gn->set_title("Input");

		List<ShaderGraph::SlotInfo> si;
		ShaderGraph::get_input_output_node_slot_info(graph->get_mode(),type,&si);

		int idx=0;
		for (List<ShaderGraph::SlotInfo>::Element *E=si.front();E;E=E->next()) {
			ShaderGraph::SlotInfo& s=E->get();
			if (s.dir==ShaderGraph::SLOT_IN) {

				Label *l= memnew( Label );
				l->set_text(s.name);
				l->set_align(Label::ALIGN_RIGHT);
				gn->add_child(l);
				gn->set_slot(idx,false,0,Color(),true,s.type,typecol[s.type]);
				idx++;
			}
		}

	} break; // all inputs (case Shader type dependent)
	case ShaderGraph::NODE_SCALAR_CONST: {
		gn->set_title("Scalar");
		SpinBox *sb = memnew( SpinBox );
		sb->set_min(-100000);
		sb->set_max(100000);
		sb->set_step(0.001);
		sb->set_val(graph->scalar_const_node_get_value(type,p_id));
		sb->connect("value_changed",this,"_scalar_const_changed",varray(p_id));
		gn->add_child(sb);
		gn->set_slot(0,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; //scalar constant
	case ShaderGraph::NODE_VEC_CONST: {

		gn->set_title("Vector");
		Array v3p(true);
		for(int i=0;i<3;i++) {
			HBoxContainer *hbc = memnew( HBoxContainer );
			Label *l = memnew( Label );
			l->set_text(String::chr('X'+i));
			hbc->add_child(l);
			SpinBox *sb = memnew( SpinBox );
			sb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			sb->set_min(-100000);
			sb->set_max(100000);
			sb->set_step(0.001);
			sb->set_val(graph->vec_const_node_get_value(type,p_id)[i]);
			sb->connect("value_changed",this,"_vec_const_changed",varray(p_id,v3p));
			v3p.push_back(sb);
			hbc->add_child(sb);
			gn->add_child(hbc);
		}
		gn->set_slot(0,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);

	} break; //vec3 constant
	case ShaderGraph::NODE_RGB_CONST: {

		gn->set_title("Color");
		ColorPickerButton *cpb = memnew( ColorPickerButton );
		cpb->set_color(graph->rgb_const_node_get_value(type,p_id));
		cpb->connect("color_changed",this,"_rgb_const_changed",varray(p_id));
		gn->add_child(cpb);
		Label *l = memnew( Label );
		l->set_text("RGB");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);
		l = memnew( Label );
		l->set_text("Alpha");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);

		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; //rgb constant (shows a color picker instead)
	case ShaderGraph::NODE_XFORM_CONST: {
		gn->set_title("XForm");
		ToolButton *edit = memnew( ToolButton );
		edit->set_text("edit..");
		edit->connect("pressed",this,"_xform_const_changed",varray(p_id,edit));
		gn->add_child(edit);
		gn->set_slot(0,false,0,Color(),true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM]);

	} break; // 4x4 matrix constant
	case ShaderGraph::NODE_TIME: {

		gn->set_title("Time");
		Label *l = memnew( Label );
		l->set_text("(s)");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);
		gn->set_slot(0,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; // time in seconds
	case ShaderGraph::NODE_SCREEN_TEX: {

		gn->set_title("ScreenTex");
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (!graph->is_slot_connected(type,p_id,0)) {
			Vector3 v = graph->default_get_value(type, p_id, 0);
			hbc->add_child(make_editor("UV: " + v,gn,p_id,0,Variant::VECTOR3));
		} else {
			hbc->add_child(make_label("UV",Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("RGB")));
		gn->add_child(hbc);
		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);

	} break; // screen texture sampler (takes UV) (only usable in fragment case Shader)
	case ShaderGraph::NODE_SCALAR_OP: {

		gn->set_title("ScalarOp");
		static const char* op_name[ShaderGraph::SCALAR_MAX_OP]={
			("Add"),
			("Sub"),
			("Mul"),
			("Div"),
			("Mod"),
			("Pow"),
			("Max"),
			("Min"),
			("Atan2")
		};

		OptionButton *ob = memnew( OptionButton );
		for(int i=0;i<ShaderGraph::SCALAR_MAX_OP;i++) {

			ob->add_item(op_name[i],i);
		}

		ob->select(graph->scalar_op_node_get_op(type,p_id));
		ob->connect("item_selected",this,"_scalar_op_changed",varray(p_id));
		gn->add_child(ob);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a",Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("a: ")+Variant(v),gn,p_id,0,Variant::REAL,hint_spin));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b",Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("b: ")+Variant(v),gn,p_id,1,Variant::REAL,hint_spin));
		}

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());


	} break; // scalar vs scalar op (mul: { } break; add: { } break; div: { } break; etc)
	case ShaderGraph::NODE_VEC_OP: {

		gn->set_title("VecOp");
		static const char* op_name[ShaderGraph::VEC_MAX_OP]={
			("Add"),
			("Sub"),
			("Mul"),
			("Div"),
			("Mod"),
			("Pow"),
			("Max"),
			("Min"),
			("Cross")
		};

		OptionButton *ob = memnew( OptionButton );
		for(int i=0;i<ShaderGraph::VEC_MAX_OP;i++) {

			ob->add_item(op_name[i],i);
		}

		ob->select(graph->vec_op_node_get_op(type,p_id));
		ob->connect("item_selected",this,"_vec_op_changed",varray(p_id));
		gn->add_child(ob);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a",Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("a: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b",Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("b: ")+v,gn,p_id,1,Variant::VECTOR3));
		}

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());


	} break; // vec3 vs vec3 op (mul: { } break;ad: { } break;div: { } break;crossprod: { } break;etc)
	case ShaderGraph::NODE_VEC_SCALAR_OP: {

		gn->set_title("VecScalarOp");
		static const char* op_name[ShaderGraph::VEC_SCALAR_MAX_OP]={
			("Mul"),
			("Div"),
			("Pow"),
		};

		OptionButton *ob = memnew( OptionButton );
		for(int i=0;i<ShaderGraph::VEC_SCALAR_MAX_OP;i++) {

			ob->add_item(op_name[i],i);
		}

		ob->select(graph->vec_scalar_op_node_get_op(type,p_id));
		ob->connect("item_selected",this,"_vec_scalar_op_changed",varray(p_id));
		gn->add_child(ob);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a",Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("a: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);

		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b",Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("b: ")+Variant(v),gn,p_id,1,Variant::REAL,hint_spin));
		}
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());


	} break; // vec3 vs scalar op (mul: { } break; add: { } break; div: { } break; etc)
	case ShaderGraph::NODE_RGB_OP: {

		gn->set_title("RGB Op");
		static const char* op_name[ShaderGraph::RGB_MAX_OP]={
			("Screen"),
			("Difference"),
			("Darken"),
			("Lighten"),
			("Overlay"),
			("Dodge"),
			("Burn"),
			("SoftLight"),
			("HardLight")
		};

		OptionButton *ob = memnew( OptionButton );
		for(int i=0;i<ShaderGraph::RGB_MAX_OP;i++) {

			ob->add_item(op_name[i],i);
		}

		ob->select(graph->rgb_op_node_get_op(type,p_id));
		ob->connect("item_selected",this,"_rgb_op_changed",varray(p_id));
		gn->add_child(ob);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a",Variant::COLOR));
		} else {
			hbc->add_child(make_editor(String("a: "),gn,p_id,0,Variant::COLOR));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b",Variant::COLOR));
		} else {
			gn->add_child(make_editor(String("b: "),gn,p_id,1,Variant::COLOR));
		}

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());

	} break; // vec3 vs vec3 rgb op (with scalar amount): { } break; like brighten: { } break; darken: { } break; burn: { } break; dodge: { } break; multiply: { } break; etc.
	case ShaderGraph::NODE_XFORM_MULT: {

		gn->set_title("XFMult");
		HBoxContainer *hbc = memnew( HBoxContainer );
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a",Variant::TRANSFORM));
		} else {
			hbc->add_child(make_editor(String("a: edit..."),gn,p_id,0,Variant::TRANSFORM));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b",Variant::TRANSFORM));
		} else {
			gn->add_child(make_editor(String("b: edit..."),gn,p_id,1,Variant::TRANSFORM));
		}

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM],true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM]);
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM],false,0,Color());


	} break; // mat4 x mat4
	case ShaderGraph::NODE_XFORM_VEC_MULT: {

		gn->set_title("XFVecMult");

		CheckBox *button = memnew (CheckBox("RotOnly"));
		button->set_pressed(graph->xform_vec_mult_node_get_no_translation(type,p_id));
		button->connect("toggled",this,"_xform_inv_rev_changed",varray(p_id));

		gn->add_child(button);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("xf",Variant::TRANSFORM));
		} else {
			hbc->add_child(make_editor(String("xf: edit..."),gn,p_id,0,Variant::TRANSFORM));
		}
		hbc->add_spacer();
		Label *l = memnew(Label("out"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child( l);
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("a",Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("a: ")+v,gn,p_id,1,Variant::VECTOR3));
		}

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());

	} break;
	case ShaderGraph::NODE_XFORM_VEC_INV_MULT: {

		gn->set_title("XFVecInvMult");


		CheckBox *button = memnew( CheckBox("RotOnly"));
		button->set_pressed(graph->xform_vec_mult_node_get_no_translation(type,p_id));
		button->connect("toggled",this,"_xform_inv_rev_changed",varray(p_id));

		gn->add_child(button);

		if (graph->is_slot_connected(type, p_id, 0)) {
			gn->add_child(make_label("a",Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			gn->add_child(make_editor(String("a: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 1)) {
			hbc->add_child(make_label("xf", Variant::TRANSFORM));
		} else {
			hbc->add_child(make_editor(String("xf: edit..."),gn,p_id,1,Variant::TRANSFORM));
		}
		hbc->add_spacer();
		Label *l = memnew(Label("out"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child( l);
		gn->add_child(hbc);

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);


	} break; // mat4 x vec3 inverse mult (with no-translation option)
	case ShaderGraph::NODE_SCALAR_FUNC: {

		gn->set_title("ScalarFunc");
		static const char* func_name[ShaderGraph::SCALAR_MAX_FUNC]={
			("Sin"),
			("Cos"),
			("Tan"),
			("ASin"),
			("ACos"),
			("ATan"),
			("SinH"),
			("CosH"),
			("TanH"),
			("Log"),
			("Exp"),
			("Sqrt"),
			("Abs"),
			("Sign"),
			("Floor"),
			("Round"),
			("Ceil"),
			("Frac"),
			("Satr"),
			("Neg")
		};

		OptionButton *ob = memnew( OptionButton );
		for(int i=0;i<ShaderGraph::SCALAR_MAX_FUNC;i++) {

			ob->add_item(func_name[i],i);
		}

		ob->select(graph->scalar_func_node_get_function(type,p_id));
		ob->connect("item_selected",this,"_scalar_func_changed",varray(p_id));
		gn->add_child(ob);

		HBoxContainer *hbc = memnew( HBoxContainer );
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("in", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("in: ")+Variant(v),gn,p_id,0,Variant::REAL,hint_spin));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);


	} break; // scalar function (sin: { } break; cos: { } break; etc)
	case ShaderGraph::NODE_VEC_FUNC: {



		gn->set_title("VecFunc");
		static const char* func_name[ShaderGraph::VEC_MAX_FUNC]={
			("Normalize"),
			("Saturate"),
			("Negate"),
			("Reciprocal"),
			("RGB to HSV"),
			("HSV to RGB"),
		};

		OptionButton *ob = memnew( OptionButton );
		for(int i=0;i<ShaderGraph::VEC_MAX_FUNC;i++) {

			ob->add_item(func_name[i],i);
		}

		ob->select(graph->vec_func_node_get_function(type,p_id));
		ob->connect("item_selected",this,"_vec_func_changed",varray(p_id));
		gn->add_child(ob);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("in", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("in: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("out")));
		gn->add_child(hbc);

		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);

	} break; // vector function (normalize: { } break; negate: { } break; reciprocal: { } break; rgb2hsv: { } break; hsv2rgb: { } break; etc: { } break; etc)
	case ShaderGraph::NODE_VEC_LEN: {
		gn->set_title("VecLength");
		HBoxContainer *hbc = memnew( HBoxContainer );
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("in", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("in: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("len")));
		gn->add_child(hbc);

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; // vec3 length
	case ShaderGraph::NODE_DOT_PROD: {

		gn->set_title("DotProduct");
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("a: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("dp")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("b: ")+v,gn,p_id,1,Variant::VECTOR3));
		}

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());

	} break; // vec3 . vec3 (dot product -> scalar output)
	case ShaderGraph::NODE_VEC_TO_SCALAR: {

		gn->set_title("Vec2Scalar");
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("vec", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("vec: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("x"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child( l);
		gn->add_child(hbc);
		l=memnew(Label("y"));
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child( l );
		l=memnew(Label("z"));
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child( l);

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);
		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);
		gn->set_slot(2,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);




	} break; // 1 vec3 input: { } break; 3 scalar outputs
	case ShaderGraph::NODE_SCALAR_TO_VEC: {

		gn->set_title("Scalar2Vec");
		HBoxContainer *hbc = memnew( HBoxContainer );
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("x", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("x: ")+Variant(v),gn,p_id,0,Variant::REAL));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("vec")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("y", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("y: ")+Variant(v),gn,p_id,1,Variant::REAL));
		}
		if (graph->is_slot_connected(type, p_id, 2)) {
			gn->add_child(make_label("in", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,2);
			gn->add_child(make_editor(String("in: ")+Variant(v),gn,p_id,2,Variant::REAL));
		}

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());

	} break; // 3 scalar input: { } break; 1 vec3 output
	case ShaderGraph::NODE_VEC_TO_XFORM: {

		gn->set_title("Vec2XForm");
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("x", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("x: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("xf")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("y", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("y: ")+v,gn,p_id,1,Variant::VECTOR3));
		}
		if (graph->is_slot_connected(type, p_id, 2)) {
			gn->add_child(make_label("z", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,2);
			gn->add_child(make_editor(String("z: ")+v,gn,p_id,2,Variant::VECTOR3));
		}
		if (graph->is_slot_connected(type, p_id, 3)) {
			gn->add_child(make_label("ofs", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,3);
			gn->add_child(make_editor(String("ofs: ")+v,gn,p_id,3,Variant::VECTOR3));
		}

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM]);
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());
		gn->set_slot(3,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());

	} break; // 3 vec input: { } break; 1 xform output
	case ShaderGraph::NODE_XFORM_TO_VEC: {

		gn->set_title("XForm2Vec");

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("fx", Variant::TRANSFORM));
		} else {
			hbc->add_child(make_editor(String("fx: edit..."),gn,p_id,0,Variant::TRANSFORM));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("x"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child( l);
		gn->add_child(hbc);
		l=memnew(Label("y"));
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child( l );
		l=memnew(Label("z"));
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child( l);
		l=memnew(Label("ofs"));
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child( l);

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(3,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);

	} break; // 3 vec input: { } break; 1 xform output
	case ShaderGraph::NODE_SCALAR_INTERP: {

		gn->set_title("ScalarInterp");
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("a: ")+Variant(v),gn,p_id,0,Variant::REAL,hint_spin));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("interp")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("b: ")+Variant(v),gn,p_id,1,Variant::REAL,hint_spin));
		}
		if (graph->is_slot_connected(type, p_id, 2)) {
			gn->add_child(make_label("c", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,2);
			gn->add_child(make_editor(String("c: ")+Variant(v),gn,p_id,2,Variant::REAL,hint_slider));
		}

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());


	} break; // scalar interpolation (with optional curve)
	case ShaderGraph::NODE_VEC_INTERP: {

		gn->set_title("VecInterp");
		HBoxContainer *hbc = memnew( HBoxContainer );
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("a", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("a: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		hbc->add_child( memnew(Label("interp")));
		gn->add_child(hbc);
		if (graph->is_slot_connected(type, p_id, 1)) {
			gn->add_child(make_label("b", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,1);
			gn->add_child(make_editor(String("b: ")+v,gn,p_id,1,Variant::VECTOR3));
		}
		if (graph->is_slot_connected(type, p_id, 2)) {
			gn->add_child(make_label("c", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,2);
			gn->add_child(make_editor(String("c: ")+Variant(v),gn,p_id,2,Variant::REAL,hint_slider));
		}

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],false,0,Color());
		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],false,0,Color());

	} break; // vec3 interpolation  (with optional curve)
	case ShaderGraph::NODE_COLOR_RAMP: {

		gn->set_title("ColorRamp");
		GraphColorRampEdit * ramp  = memnew( GraphColorRampEdit );

		PoolVector<real_t> offsets = graph->color_ramp_node_get_offsets(type,p_id);
		PoolVector<Color> colors = graph->color_ramp_node_get_colors(type,p_id);

		int oc = offsets.size();

		if (oc) {
			PoolVector<real_t>::Read rofs = offsets.read();
			PoolVector<Color>::Read rcol = colors.read();

			Vector<float> ofsv;
			Vector<Color> colorv;
			for(int i=0;i<oc;i++) {
				ofsv.push_back(rofs[i]);
				colorv.push_back(rcol[i]);
			}

			ramp->set_ramp(ofsv,colorv);

		}

		ramp->connect("ramp_changed",this,"_color_ramp_changed",varray(p_id,ramp));
		ramp->set_custom_minimum_size(Size2(128,1));
		gn->add_child(ramp);


		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("c", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("c: ")+Variant(v),gn,p_id,0,Variant::REAL,hint_slider));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("rgb"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child( l);
		gn->add_child(hbc);
		l=memnew(Label("alpha"));
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child( l);


		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(2,false,ShaderGraph::SLOT_MAX,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);


	} break; // scalar interpolation (with optional curve)
	case ShaderGraph::NODE_CURVE_MAP: {

		gn->set_title("CurveMap");
		GraphCurveMapEdit * map  = memnew( GraphCurveMapEdit );

		PoolVector<Vector2> points = graph->curve_map_node_get_points(type,p_id);

		int oc = points.size();

		if (oc) {
			PoolVector<Vector2>::Read rofs = points.read();


			Vector<Vector2> ofsv;
			for(int i=0;i<oc;i++) {
				ofsv.push_back(rofs[i]);
			}

			map->set_points(ofsv);

		}
		map->connect("curve_changed",this,"_curve_changed",varray(p_id,map));

		//map->connect("map_changed",this,"_curve_map_changed",varray(p_id,map));
		map->set_custom_minimum_size(Size2(128,64));
		gn->add_child(map);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("c", Variant::REAL));
		} else {
			float v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("c: ")+Variant(v),gn,p_id,0,Variant::REAL,hint_slider));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("cmap"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child( l);
		gn->add_child(hbc);


		gn->set_slot(1,true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR],true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);


	} break; // scalar interpolation (with optional curve)

	case ShaderGraph::NODE_SCALAR_INPUT: {

		gn->set_title("ScalarUniform");
		LineEdit *le = memnew( LineEdit );
		gn->add_child(le);
		le->set_text(graph->input_node_get_name(type,p_id));
		le->connect("text_entered",this,"_input_name_changed",varray(p_id,le));
		SpinBox *sb = memnew( SpinBox );
		sb->set_min(-100000);
		sb->set_max(100000);
		sb->set_step(0.001);
		sb->set_val(graph->scalar_input_node_get_value(type,p_id));
		sb->connect("value_changed",this,"_scalar_input_changed",varray(p_id));
		gn->add_child(sb);
		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; // scalar uniform (assignable in material)
	case ShaderGraph::NODE_VEC_INPUT: {

		gn->set_title("VectorUniform");
		LineEdit *le = memnew( LineEdit );
		gn->add_child(le);
		le->set_text(graph->input_node_get_name(type,p_id));
		le->connect("text_entered",this,"_input_name_changed",varray(p_id,le));
		Array v3p(true);
		for(int i=0;i<3;i++) {
			HBoxContainer *hbc = memnew( HBoxContainer );
			Label *l = memnew( Label );
			l->set_text(String::chr('X'+i));
			hbc->add_child(l);
			SpinBox *sb = memnew( SpinBox );
			sb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			sb->set_min(-100000);
			sb->set_max(100000);
			sb->set_step(0.001);
			sb->set_val(graph->vec_input_node_get_value(type,p_id)[i]);
			sb->connect("value_changed",this,"_vec_input_changed",varray(p_id,v3p));
			v3p.push_back(sb);
			hbc->add_child(sb);
			gn->add_child(hbc);
		}
		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);

	} break; // vec3 uniform (assignable in material)
	case ShaderGraph::NODE_RGB_INPUT: {

		gn->set_title("ColorUniform");
		LineEdit *le = memnew( LineEdit );
		gn->add_child(le);
		le->set_text(graph->input_node_get_name(type,p_id));
		le->connect("text_entered",this,"_input_name_changed",varray(p_id,le));
		ColorPickerButton *cpb = memnew( ColorPickerButton );
		cpb->set_color(graph->rgb_input_node_get_value(type,p_id));
		cpb->connect("color_changed",this,"_rgb_input_changed",varray(p_id));
		gn->add_child(cpb);
		Label *l = memnew( Label );
		l->set_text("RGB");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);
		l = memnew( Label );
		l->set_text("Alpha");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);

		gn->set_slot(2,false,0,Color(),true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(3,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);


	} break; // color uniform (assignable in material)
	case ShaderGraph::NODE_XFORM_INPUT: {
		gn->set_title("XFUniform");
		LineEdit *le = memnew( LineEdit );
		gn->add_child(le);
		le->set_text(graph->input_node_get_name(type,p_id));
		le->connect("text_entered",this,"_input_name_changed",varray(p_id,le));
		ToolButton *edit = memnew( ToolButton );
		edit->set_text("edit..");
		edit->connect("pressed",this,"_xform_input_changed",varray(p_id,edit));
		gn->add_child(edit);
		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_XFORM,typecol[ShaderGraph::SLOT_TYPE_XFORM]);

	} break; // mat4 uniform (assignable in material)
	case ShaderGraph::NODE_TEXTURE_INPUT: {

		gn->set_title("TexUniform");
		LineEdit *le = memnew( LineEdit );
		gn->add_child(le);
		le->set_text(graph->input_node_get_name(type,p_id));
		le->connect("text_entered",this,"_input_name_changed",varray(p_id,le));
		TextureRect *tex = memnew( TextureRect );
		tex->set_expand(true);
		tex->set_custom_minimum_size(Size2(80,80));
		tex->set_drag_forwarding(this);
		gn->add_child(tex);
		tex->set_mouse_filter(MOUSE_FILTER_PASS);
		tex->set_texture(graph->texture_input_node_get_value(type,p_id));
		ToolButton *edit = memnew( ToolButton );
		edit->set_text("edit..");
		edit->connect("pressed",this,"_tex_edited",varray(p_id,edit));
		gn->add_child(edit);

		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("UV", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("UV: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("RGB"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child(l);
		gn->add_child(hbc);
		l = memnew( Label );
		l->set_text("Alpha");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);

		gn->set_slot(3,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(4,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; // texture input (assignable in material)
	case ShaderGraph::NODE_CUBEMAP_INPUT: {

		gn->set_title("TexUniform");
		LineEdit *le = memnew( LineEdit );
		gn->add_child(le);
		le->set_text(graph->input_node_get_name(type,p_id));
		le->connect("text_entered",this,"_input_name_changed",varray(p_id,le));

		ToolButton *edit = memnew( ToolButton );
		edit->set_text("edit..");
		edit->connect("pressed",this,"_cube_edited",varray(p_id,edit));
		gn->add_child(edit);


		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("UV", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("UV: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("RGB"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child(l);
		gn->add_child(hbc);
		l = memnew( Label );
		l->set_text("Alpha");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);

		gn->set_slot(2,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(3,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);

	} break; // cubemap input (assignable in material)
	case ShaderGraph::NODE_DEFAULT_TEXTURE: {

		gn->set_title("CanvasItemTex");
		HBoxContainer *hbc = memnew( HBoxContainer );
		hbc->add_constant_override("separation",0);
		if (graph->is_slot_connected(type, p_id, 0)) {
			hbc->add_child(make_label("UV", Variant::VECTOR3));
		} else {
			Vector3 v = graph->default_get_value(type,p_id,0);
			hbc->add_child(make_editor(String("UV: ")+v,gn,p_id,0,Variant::VECTOR3));
		}
		hbc->add_spacer();
		Label *l=memnew(Label("RGB"));
		l->set_align(Label::ALIGN_RIGHT);
		hbc->add_child(l);
		gn->add_child(hbc);
		l = memnew( Label );
		l->set_text("Alpha");
		l->set_align(Label::ALIGN_RIGHT);
		gn->add_child(l);

		gn->set_slot(0,true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC],true,ShaderGraph::SLOT_TYPE_VEC,typecol[ShaderGraph::SLOT_TYPE_VEC]);
		gn->set_slot(1,false,0,Color(),true,ShaderGraph::SLOT_TYPE_SCALAR,typecol[ShaderGraph::SLOT_TYPE_SCALAR]);


	} break; // screen texture sampler (takes UV) (only usable in fragment case Shader)

	case ShaderGraph::NODE_OUTPUT: {
		gn->set_title("Output");
		gn->set_show_close_button(false);

		List<ShaderGraph::SlotInfo> si;
		ShaderGraph::get_input_output_node_slot_info(graph->get_mode(),type,&si);

		Array colors;
		colors.push_back("Color");
		colors.push_back("LightColor");
		colors.push_back("Light");
		colors.push_back("ShadowColor");
		colors.push_back("Diffuse");
		colors.push_back("Specular");
		colors.push_back("Emission");
		Array reals;
		reals.push_back("Alpha");
		reals.push_back("DiffuseAlpha");
		reals.push_back("NormalMapDepth");
		reals.push_back("SpecExp");
		reals.push_back("Glow");
		reals.push_back("ShadeParam");
		reals.push_back("SpecularExp");
		reals.push_back("LightAlpha");
		reals.push_back("ShadowAlpha");
		reals.push_back("PointSize");
		reals.push_back("Discard");

		int idx=0;
		for (List<ShaderGraph::SlotInfo>::Element *E=si.front();E;E=E->next()) {
			ShaderGraph::SlotInfo& s=E->get();
			if (s.dir==ShaderGraph::SLOT_OUT) {
				Variant::Type v;
				if (colors.find(s.name)>=0)
					v=Variant::COLOR;
				else if (reals.find(s.name)>=0)
					v=Variant::REAL;
				else
					v=Variant::VECTOR3;
				gn->add_child(make_label(s.name, v));
				gn->set_slot(idx,true,s.type,typecol[s.type],false,0,Color());
				idx++;
			}
		}

	} break; // output (case Shader type dependent)
	case ShaderGraph::NODE_COMMENT: {
		gn->set_title("Comment");
		TextEdit *te = memnew(TextEdit);
		te->set_custom_minimum_size(Size2(100,100));
		gn->add_child(te);
		te->set_text(graph->comment_node_get_text(type,p_id));
		te->connect("text_changed",this,"_comment_edited",varray(p_id,te));

	} break; // comment



	}

	gn->connect("dragged",this,"_node_moved",varray(p_id));
	gn->connect("close_request",this,"_node_removed",varray(p_id),CONNECT_DEFERRED);
	graph_edit->add_child(gn);
	node_map[p_id]=gn;
	gn->set_offset(graph->node_get_position(type,p_id));


}

void ShaderGraphView::_update_graph() {


	if (block_update)
		return;

	for (Map<int,GraphNode*>::Element *E=node_map.front();E;E=E->next()) {

		memdelete(E->get());
	}

	node_map.clear();

	if (!graph.is_valid())
		return;


	List<int> nl;
	graph->get_node_list(type,&nl);

	for(List<int>::Element *E=nl.front();E;E=E->next()) {

		_create_node(E->get());
	}
	graph_edit->clear_connections();

	List<ShaderGraph::Connection> connections;
	graph->get_node_connections(type,&connections);
	for(List<ShaderGraph::Connection>::Element *E=connections.front();E;E=E->next()) {

		ERR_CONTINUE(!node_map.has(E->get().src_id) || !node_map.has(E->get().dst_id));
		graph_edit->connect_node(node_map[E->get().src_id]->get_name(),E->get().src_slot,node_map[E->get().dst_id]->get_name(),E->get().dst_slot);
	}

}

void ShaderGraphView::_sg_updated() {

	if (!graph.is_valid())
		return;
	switch(graph->get_graph_error(type)) {
	case ShaderGraph::GRAPH_OK: status->set_text(""); break;
	case ShaderGraph::GRAPH_ERROR_CYCLIC: status->set_text(TTR("Error: Cyclic Connection Link")); break;
	case ShaderGraph::GRAPH_ERROR_MISSING_CONNECTIONS: status->set_text(TTR("Error: Missing Input Connections")); break;
	}
}

Variant ShaderGraphView::get_drag_data_fw(const Point2 &p_point, Control *p_from)
{
	TextureRect* frame = Object::cast_to<TextureRect>(p_from);
	if (!frame)
		return Variant();

	if (!frame->get_texture().is_valid())
		return Variant();

	RES res = frame->get_texture();
	return EditorNode::get_singleton()->drag_resource(res,p_from);

	return Variant();
}

bool ShaderGraphView::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const
{
	if (p_data.get_type() != Variant::DICTIONARY)
		return false;

	Dictionary d = p_data;

	if (d.has("type")){
		if (d["type"] == "resource" && d.has("resource")) {
			Variant val = d["resource"];

			if (val.get_type()==Variant::OBJECT) {
				RES res = val;
				if (res.is_valid() && Object::cast_to<Texture>(res))
					return true;
			}
		}
		else if (d["type"] == "files" && d.has("files")) {
			Vector<String> files = d["files"];
			if (files.size() != 1)
				return false;
			return (ResourceLoader::get_resource_type(files[0]) == "ImageTexture");
		}
	}

	return false;
}

void ShaderGraphView::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from)
{
	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

	TextureRect *frame = Object::cast_to<TextureRect>(p_from);
	if (!frame)
		return;

	Dictionary d = p_data;
	Ref<Texture> tex;

	if (d.has("type")) {
		if (d["type"] == "resource" && d.has("resource")){
			Variant val = d["resource"];

			if (val.get_type()==Variant::OBJECT) {
				RES res = val;
				if (res.is_valid())
					tex = Ref<Texture>(Object::cast_to<Texture>(*res));
			}
		}
		else if (d["type"] == "files" && d.has("files")) {
			Vector<String> files = d["files"];
			RES res = ResourceLoader::load(files[0]);
			if (res.is_valid())
				tex = Ref<Texture>(Object::cast_to<Texture>(*res));
		}
	}

	if (!tex.is_valid()) return;

	GraphNode *gn = Object::cast_to<GraphNode>(frame->get_parent());
	if (!gn) return;

	int id = -1;
	for(Map<int,GraphNode*>::Element *E = node_map.front();E;E=E->next())
		if (E->get() == gn) {
			id = E->key();
			break;
		}
	print_line(String::num(double(id)));
	if (id < 0) return;

	if (graph->node_get_type(type,id)==ShaderGraph::NODE_TEXTURE_INPUT) {

		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Texture Uniform"));
		ur->add_do_method(graph.ptr(),"texture_input_node_set_value",type,id,tex);
		ur->add_undo_method(graph.ptr(),"texture_input_node_set_value",type,id,graph->texture_input_node_get_value(type,id));
		ur->add_do_method(this,"_update_graph");
		ur->add_undo_method(this,"_update_graph");
		ur->commit_action();
	}
}

void ShaderGraphView::set_graph(Ref<ShaderGraph> p_graph){


	if (graph.is_valid()) {
		graph->disconnect("updated",this,"_sg_updated");
	}
	graph=p_graph;
	if (graph.is_valid()) {
		graph->connect("updated",this,"_sg_updated");
	}
	_update_graph();
	_sg_updated();

}

void ShaderGraphView::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		ped_popup->connect("variant_changed",this,"_variant_edited");
	}
}

void ShaderGraphView::add_node(int p_type, const Vector2 &location) {

	if (p_type==ShaderGraph::NODE_INPUT && graph->node_count(type, p_type)>0)
		return;

	List<int> existing;
	graph->get_node_list(type,&existing);
	existing.sort();
	int newid=1;
	for(List<int>::Element *E=existing.front();E;E=E->next()) {
		if (!E->next() || (E->get()+1!=E->next()->get())){
			newid=E->get()+1;
			break;
		}
	}

	Vector2 init_ofs = location;
	while(true) {
		bool valid=true;
		for(List<int>::Element *E=existing.front();E;E=E->next()) {
			Vector2 pos = graph->node_get_position(type,E->get());
			if (init_ofs==pos) {
				init_ofs+=Vector2(20,20);
				valid=false;
				break;

			}
		}

		if (valid)
			break;
	}
	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Add Shader Graph Node"));
	ur->add_do_method(graph.ptr(),"node_add",type,p_type,newid);
	ur->add_do_method(graph.ptr(),"node_set_position",type,newid,init_ofs);
	ur->add_undo_method(graph.ptr(),"node_remove",type,newid);
	ur->add_do_method(this,"_update_graph");
	ur->add_undo_method(this,"_update_graph");
	ur->commit_action();

}

void ShaderGraphView::_bind_methods() {

	ClassDB::bind_method("_update_graph",&ShaderGraphView::_update_graph);
	ClassDB::bind_method("_begin_node_move", &ShaderGraphView::_begin_node_move);
	ClassDB::bind_method("_node_moved",&ShaderGraphView::_node_moved);
	ClassDB::bind_method("_end_node_move", &ShaderGraphView::_end_node_move);
	ClassDB::bind_method("_move_node",&ShaderGraphView::_move_node);
	ClassDB::bind_method("_node_removed",&ShaderGraphView::_node_removed);
	ClassDB::bind_method("_connection_request",&ShaderGraphView::_connection_request);
	ClassDB::bind_method("_disconnection_request",&ShaderGraphView::_disconnection_request);
	ClassDB::bind_method("_duplicate_nodes_request", &ShaderGraphView::_duplicate_nodes_request);
	ClassDB::bind_method("_duplicate_nodes", &ShaderGraphView::_duplicate_nodes);
	ClassDB::bind_method("_delete_nodes_request", &ShaderGraphView::_delete_nodes_request);

	ClassDB::bind_method("_default_changed",&ShaderGraphView::_default_changed);
	ClassDB::bind_method("_scalar_const_changed",&ShaderGraphView::_scalar_const_changed);
	ClassDB::bind_method("_vec_const_changed",&ShaderGraphView::_vec_const_changed);
	ClassDB::bind_method("_rgb_const_changed",&ShaderGraphView::_rgb_const_changed);
	ClassDB::bind_method("_xform_const_changed",&ShaderGraphView::_xform_const_changed);
	ClassDB::bind_method("_scalar_op_changed",&ShaderGraphView::_scalar_op_changed);
	ClassDB::bind_method("_vec_op_changed",&ShaderGraphView::_vec_op_changed);
	ClassDB::bind_method("_vec_scalar_op_changed",&ShaderGraphView::_vec_scalar_op_changed);
	ClassDB::bind_method("_rgb_op_changed",&ShaderGraphView::_rgb_op_changed);
	ClassDB::bind_method("_xform_inv_rev_changed",&ShaderGraphView::_xform_inv_rev_changed);
	ClassDB::bind_method("_scalar_func_changed",&ShaderGraphView::_scalar_func_changed);
	ClassDB::bind_method("_vec_func_changed",&ShaderGraphView::_vec_func_changed);
	ClassDB::bind_method("_scalar_input_changed",&ShaderGraphView::_scalar_input_changed);
	ClassDB::bind_method("_vec_input_changed",&ShaderGraphView::_vec_input_changed);
	ClassDB::bind_method("_xform_input_changed",&ShaderGraphView::_xform_input_changed);
	ClassDB::bind_method("_rgb_input_changed",&ShaderGraphView::_rgb_input_changed);
	ClassDB::bind_method("_tex_input_change",&ShaderGraphView::_tex_input_change);
	ClassDB::bind_method("_cube_input_change",&ShaderGraphView::_cube_input_change);
	ClassDB::bind_method("_input_name_changed",&ShaderGraphView::_input_name_changed);
	ClassDB::bind_method("_tex_edited",&ShaderGraphView::_tex_edited);
	ClassDB::bind_method("_variant_edited",&ShaderGraphView::_variant_edited);
	ClassDB::bind_method("_cube_edited",&ShaderGraphView::_cube_edited);
	ClassDB::bind_method("_comment_edited",&ShaderGraphView::_comment_edited);
	ClassDB::bind_method("_color_ramp_changed",&ShaderGraphView::_color_ramp_changed);
	ClassDB::bind_method("_curve_changed",&ShaderGraphView::_curve_changed);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &ShaderGraphView::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &ShaderGraphView::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &ShaderGraphView::drop_data_fw);

	ClassDB::bind_method("_sg_updated",&ShaderGraphView::_sg_updated);
}

ShaderGraphView::ShaderGraphView(ShaderGraph::ShaderType p_type) {

	type=p_type;
	graph_edit = memnew( GraphEdit );
	block_update=false;
	ped_popup = memnew( CustomPropertyEditor );
	graph_edit->add_child(ped_popup);
	status = memnew( Label );
	graph_edit->get_top_layer()->add_child(status);
	graph_edit->connect("_begin_node_move", this, "_begin_node_move");
	graph_edit->connect("_end_node_move", this, "_end_node_move");
	status->set_position(Vector2(5,5));
	status->add_color_override("font_color_shadow",Color(0,0,0));
	status->add_color_override("font_color",Color(1,0.4,0.3));
	status->add_constant_override("shadow_as_outline",1);
	status->add_constant_override("shadow_offset_x",2);
	status->add_constant_override("shadow_offset_y",2);
	status->set_text("");
}


//////////////edit//////////////
void ShaderGraphEditor::edit(Ref<ShaderGraph> p_shader) {

	for(int i=0;i<ShaderGraph::SHADER_TYPE_MAX;i++) {
		graph_edits[i]->set_graph(p_shader);
	}
}

void ShaderGraphEditor::_add_node(int p_type) {

	ShaderGraph::ShaderType shader_type=ShaderGraph::ShaderType(tabs->get_current_tab());
	graph_edits[shader_type]->add_node(p_type, next_location);
}

void ShaderGraphEditor::_popup_requested(const Vector2 &p_position)
{
	Vector2 scroll_ofs=graph_edits[tabs->get_current_tab()]->get_graph_edit()->get_scroll_ofs();
	next_location = get_local_mouse_position() + scroll_ofs;
	popup->set_global_position(p_position);
	popup->set_size( Size2( 200, 0) );
	popup->popup();
	popup->call_deferred("grab_click_focus");
	popup->set_invalidate_click_until_motion();
}

void ShaderGraphEditor::_notification(int p_what) {
	if (p_what==NOTIFICATION_ENTER_TREE) {

		for(int i=0;i<ShaderGraph::NODE_TYPE_MAX;i++) {

			if (i==ShaderGraph::NODE_OUTPUT)
				continue;
			if (!_2d && i==ShaderGraph::NODE_DEFAULT_TEXTURE)
				continue;

			String nn = node_names[i];
			String ic = nn.get_slice(":",0);
			String v = nn.get_slice(":",1);
			bool addsep=false;
			if (nn.ends_with(":")) {
				addsep=true;
			}
			popup->add_icon_item(get_icon(ic,"EditorIcons"),v,i);
			if (addsep)
				popup->add_separator();
		}
		popup->connect("id_pressed",this,"_add_node");


	}
}

void ShaderGraphEditor::_bind_methods() {

	ClassDB::bind_method("_add_node",&ShaderGraphEditor::_add_node);
	ClassDB::bind_method("_popup_requested",&ShaderGraphEditor::_popup_requested);
}


const char* ShaderGraphEditor::node_names[ShaderGraph::NODE_TYPE_MAX]={
	("GraphInput:Input"), // all inputs (shader type dependent)
	("GraphScalar:Scalar Constant"), //scalar constant
	("GraphVector:Vector Constant"), //vec3 constant
	("GraphRgb:RGB Constant"), //rgb constant (shows a color picker instead)
	("GraphXform:XForm Constant"), // 4x4 matrix constant
	("GraphTime:Time:"), // time in seconds
	("GraphTexscreen:Screen Sample"), // screen texture sampler (takes uv) (only usable in fragment shader)
	("GraphScalarOp:Scalar Operator"), // scalar vs scalar op (mul", add", div", etc)
	("GraphVecOp:Vector Operator"), // vec3 vs vec3 op (mul",ad",div",crossprod",etc)
	("GraphVecScalarOp:Scalar+Vector Operator"), // vec3 vs scalar op (mul", add", div", etc)
	("GraphRgbOp:RGB Operator:"), // vec3 vs vec3 rgb op (with scalar amount)", like brighten", darken", burn", dodge", multiply", etc.
	("GraphXformMult:XForm Multiply"), // mat4 x mat4
	("GraphXformVecMult:XForm+Vector Multiply"), // mat4 x vec3 mult (with no-translation option)
	("GraphXformVecImult:Form+Vector InvMultiply:"), // mat4 x vec3 inverse mult (with no-translation option)
	("GraphXformScalarFunc:Scalar Function"), // scalar function (sin", cos", etc)
	("GraphXformVecFunc:Vector Function"), // vector function (normalize", negate", reciprocal", rgb2hsv", hsv2rgb", etc", etc)
	("GraphVecLength:Vector Length"), // vec3 length
	("GraphVecDp:Dot Product:"), // vec3 . vec3 (dot product -> scalar output)
	("GraphVecToScalars:Vector -> Scalars"), // 1 vec3 input", 3 scalar outputs
	("GraphScalarsToVec:Scalars -> Vector"), // 3 scalar input", 1 vec3 output
	("GraphXformToVecs:XForm -> Vectors"), // 3 vec input", 1 xform output
	("GraphVecsToXform:Vectors -> XForm:"), // 3 vec input", 1 xform output
	("GraphScalarInterp:Scalar Interpolate"), // scalar interpolation (with optional curve)
	("GraphVecInterp:Vector Interpolate:"), // vec3 interpolation  (with optional curve)
	("GraphColorRamp:Color Ramp"), // vec3 interpolation  (with optional curve)
	("GraphCurveMap:Curve Remap:"), // vec3 interpolation  (with optional curve)
	("GraphScalarUniform:Scalar Uniform"), // scalar uniform (assignable in material)
	("GraphVectorUniform:Vector Uniform"), // vec3 uniform (assignable in material)
	("GraphRgbUniform:RGB Uniform"), // color uniform (assignable in material)
	("GraphXformUniform:XForm Uniform"), // mat4 uniform (assignable in material)
	("GraphTextureUniform:Texture Uniform"), // texture input (assignable in material)
	("GraphCubeUniform:CubeMap Uniform:"), // cubemap input (assignable in material)
	("GraphDefaultTexture:CanvasItem Texture:"), // cubemap input (assignable in material)
	("Output"), // output (shader type dependent)
	("GraphComment:Comment"), // comment


};
ShaderGraphEditor::ShaderGraphEditor(bool p_2d) {
	_2d=p_2d;

	popup = memnew( PopupMenu );
	add_child(popup);


	tabs = memnew(TabContainer);
	tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(tabs);
	const char* sname[ShaderGraph::SHADER_TYPE_MAX]={
		"Vertex",
		"Fragment",
		"Light"
	};
	for(int i=0;i<ShaderGraph::SHADER_TYPE_MAX;i++) {

		graph_edits[i]= memnew( ShaderGraphView(ShaderGraph::ShaderType(i)) );
		add_child(graph_edits[i]);
		graph_edits[i]->get_graph_edit()->set_name(sname[i]);
		tabs->add_child(graph_edits[i]->get_graph_edit());
		graph_edits[i]->get_graph_edit()->connect("connection_request",graph_edits[i],"_connection_request");
		graph_edits[i]->get_graph_edit()->connect("disconnection_request",graph_edits[i],"_disconnection_request");
		graph_edits[i]->get_graph_edit()->connect("duplicate_nodes_request", graph_edits[i], "_duplicate_nodes_request");
		graph_edits[i]->get_graph_edit()->connect("popup_request",this,"_popup_requested");
		graph_edits[i]->get_graph_edit()->connect("delete_nodes_request",graph_edits[i],"_delete_nodes_request");
		graph_edits[i]->get_graph_edit()->set_right_disconnects(true);
	}

	tabs->set_current_tab(1);

	set_custom_minimum_size(Size2(100,300));
}


void ShaderGraphEditorPlugin::edit(Object *p_object) {

	shader_editor->edit(Object::cast_to<ShaderGraph>(p_object));
}

bool ShaderGraphEditorPlugin::handles(Object *p_object) const {

	ShaderGraph *shader=Object::cast_to<ShaderGraph>(p_object);
	if (!shader)
		return false;
	if (_2d)
		return shader->get_mode()==Shader::MODE_CANVAS_ITEM;
	else
		return shader->get_mode()==Shader::MODE_MATERIAL;
}

void ShaderGraphEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		shader_editor->show();
	} else {

		shader_editor->hide();
	}

}

ShaderGraphEditorPlugin::ShaderGraphEditorPlugin(EditorNode *p_node, bool p_2d) {

	_2d=p_2d;
	editor=p_node;
	shader_editor = memnew( ShaderGraphEditor(p_2d) );
	shader_editor->hide();
	if (p_2d)
		CanvasItemEditor::get_singleton()->get_bottom_split()->add_child(shader_editor);
	else
		SpatialEditor::get_singleton()->get_shader_split()->add_child(shader_editor);


		//editor->get_viewport()->add_child(shader_editor);
		//shader_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);
		//shader_editor->hide();

}


ShaderGraphEditorPlugin::~ShaderGraphEditorPlugin()
{
}

#endif
