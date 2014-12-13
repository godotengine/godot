/*************************************************************************/
/*  collision_shape_2d.cpp                                               */
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
#include "collision_shape_2d.h"
#include "collision_object_2d.h"
#include "scene/resources/segment_shape_2d.h"
#include "scene/resources/shape_line_2d.h"
#include "scene/resources/circle_shape_2d.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/capsule_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"


void CollisionShape2D::_add_to_collision_object(Object *p_obj) {

	if (unparenting)
		return;

	CollisionObject2D *co = p_obj->cast_to<CollisionObject2D>();
	ERR_FAIL_COND(!co);
	co->add_shape(shape,get_transform());
	if (trigger)
		co->set_shape_as_trigger(co->get_shape_count()-1,true);

}

void CollisionShape2D::_shape_changed() {

	update();
	_update_parent();
}

void CollisionShape2D::_update_parent() {


	Node *parent = get_parent();
	if (!parent)
		return;
	CollisionObject2D *co = parent->cast_to<CollisionObject2D>();
	if (!co)
		return;
	co->_update_shapes_from_children();
}

void CollisionShape2D::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {
			unparenting=false;
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {

			if (!is_inside_tree())
				break;
			_update_parent();

		} break;
		/*
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (!is_inside_scene())
				break;
			_update_parent();

		} break;*/
		case NOTIFICATION_DRAW: {

			rect=Rect2();

			Color draw_col=Color(0,0.6,0.7,0.5);

			if (shape->cast_to<LineShape2D>()) {

				LineShape2D *l = shape->cast_to<LineShape2D>();
				Vector2 point = l->get_d() * l->get_normal();

				Vector2 l1[2]={point-l->get_normal().tangent()*100,point+l->get_normal().tangent()*100};
				draw_line(l1[0],l1[1],draw_col,3);
				Vector2 l2[2]={point,point+l->get_normal()*30};
				draw_line(l2[0],l2[1],draw_col,3);
				rect.pos=l1[0];
				rect.expand_to(l1[1]);
				rect.expand_to(l2[0]);
				rect.expand_to(l2[1]);

			} else if (shape->cast_to<SegmentShape2D>()) {

				SegmentShape2D *s = shape->cast_to<SegmentShape2D>();
				draw_line(s->get_a(),s->get_b(),draw_col,3);
				rect.pos=s->get_a();
				rect.expand_to(s->get_b());

			} else if (shape->cast_to<RayShape2D>()) {

				RayShape2D *s = shape->cast_to<RayShape2D>();

				Vector2 tip = Vector2(0,s->get_length());
				draw_line(Vector2(),tip,draw_col,3);
				Vector<Vector2> pts;
				float tsize=4;
				pts.push_back(tip+Vector2(0,tsize));
				pts.push_back(tip+Vector2(0.707*tsize,0));
				pts.push_back(tip+Vector2(-0.707*tsize,0));
				Vector<Color> cols;
				for(int i=0;i<3;i++)
					cols.push_back(draw_col);

				draw_primitive(pts,cols,Vector<Vector2>()); //small arrow

				rect.pos=Vector2();
				rect.expand_to(tip);
				rect=rect.grow(0.707*tsize);

			} else if (shape->cast_to<CircleShape2D>()) {

				CircleShape2D *s = shape->cast_to<CircleShape2D>();
				Vector<Vector2> points;
				for(int i=0;i<24;i++) {

					points.push_back(Vector2(Math::cos(i*Math_PI*2/24.0),Math::sin(i*Math_PI*2/24.0))*s->get_radius());
				}

				draw_colored_polygon(points,draw_col);
				rect.pos=-Point2(s->get_radius(),s->get_radius());
				rect.size=Point2(s->get_radius(),s->get_radius())*2.0;

			} else if (shape->cast_to<RectangleShape2D>()) {

				RectangleShape2D *s = shape->cast_to<RectangleShape2D>();
				Vector2 he = s->get_extents();
				rect=Rect2(-he,he*2.0);
				draw_rect(rect,draw_col);;

			} else if (shape->cast_to<CapsuleShape2D>()) {

				CapsuleShape2D *s = shape->cast_to<CapsuleShape2D>();

				Vector<Vector2> points;
				for(int i=0;i<24;i++) {
					Vector2 ofs = Vector2(0,(i>6 && i<=18) ? -s->get_height()*0.5 : s->get_height()*0.5);

					points.push_back(Vector2(Math::sin(i*Math_PI*2/24.0),Math::cos(i*Math_PI*2/24.0))*s->get_radius() + ofs);
					if (i==6 || i==18)
						points.push_back(Vector2(Math::sin(i*Math_PI*2/24.0),Math::cos(i*Math_PI*2/24.0))*s->get_radius() - ofs);
				}

				draw_colored_polygon(points,draw_col);
				Vector2 he=Point2(s->get_radius(),s->get_radius()+s->get_height()*0.5);
				rect.pos=-he;
				rect.size=he*2.0;

			} else if (shape->cast_to<ConvexPolygonShape2D>()) {

				ConvexPolygonShape2D *s = shape->cast_to<ConvexPolygonShape2D>();

				Vector<Vector2> points = s->get_points();
				for(int i=0;i<points.size();i++) {
					if (i==0)
						rect.pos=points[i];
					else
						rect.expand_to(points[i]);
				}

				draw_colored_polygon(points,draw_col);

			}

			rect=rect.grow(3);

		} break;
		case NOTIFICATION_UNPARENTED: {
			unparenting = true;
			_update_parent();
		} break;
	}

}

void CollisionShape2D::set_shape(const Ref<Shape2D>& p_shape) {

	if (shape.is_valid())
		shape->disconnect("changed",this,"_shape_changed");
	shape=p_shape;
	update();
	_update_parent();
	if (shape.is_valid())
		shape->connect("changed",this,"_shape_changed");

}

Ref<Shape2D> CollisionShape2D::get_shape() const {

	return shape;
}

Rect2 CollisionShape2D::get_item_rect() const {

	return rect;
}

void CollisionShape2D::set_trigger(bool p_trigger) {

	trigger=p_trigger;
	_update_parent();
}

bool CollisionShape2D::is_trigger() const{

	return trigger;
}

void CollisionShape2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_shape","shape"),&CollisionShape2D::set_shape);
	ObjectTypeDB::bind_method(_MD("get_shape"),&CollisionShape2D::get_shape);
	ObjectTypeDB::bind_method(_MD("_shape_changed"),&CollisionShape2D::_shape_changed);
	ObjectTypeDB::bind_method(_MD("_add_to_collision_object"),&CollisionShape2D::_add_to_collision_object);
	ObjectTypeDB::bind_method(_MD("set_trigger","enable"),&CollisionShape2D::set_trigger);
	ObjectTypeDB::bind_method(_MD("is_trigger"),&CollisionShape2D::is_trigger);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT,"shape",PROPERTY_HINT_RESOURCE_TYPE,"Shape2D"),_SCS("set_shape"),_SCS("get_shape"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"trigger"),_SCS("set_trigger"),_SCS("is_trigger"));
}

CollisionShape2D::CollisionShape2D() {

	rect=Rect2(-Point2(10,10),Point2(20,20));

	trigger=false;
	unparenting = false;
}
