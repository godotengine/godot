/*************************************************************************/
/*  collision_polygon_2d.cpp                                             */
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
#include "collision_polygon_2d.h"
#include "collision_object_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"

void CollisionPolygon2D::_add_to_collision_object(Object *p_obj) {

	CollisionObject2D *co = p_obj->cast_to<CollisionObject2D>();
	ERR_FAIL_COND(!co);

	if (polygon.size()==0)
		return;

	bool solids=build_mode==BUILD_SOLIDS;

	if (solids) {

		//here comes the sun, lalalala
		//decompose concave into multiple convex polygons and add them
		Vector< Vector<Vector2> > decomp = Geometry::decompose_polygon(polygon);
		for(int i=0;i<decomp.size();i++) {
			Ref<ConvexPolygonShape2D> convex = memnew( ConvexPolygonShape2D );
			convex->set_points(decomp[i]);
			co->add_shape(convex,get_transform());
			if (trigger)
				co->set_shape_as_trigger(co->get_shape_count()-1,true);

		}

	} else {

		Ref<ConcavePolygonShape2D> concave = memnew( ConcavePolygonShape2D );

		DVector<Vector2> segments;
		segments.resize(polygon.size()*2);
		DVector<Vector2>::Write w=segments.write();

		for(int i=0;i<polygon.size();i++) {
			w[(i<<1)+0]=polygon[i];
			w[(i<<1)+1]=polygon[(i+1)%polygon.size()];
		}

		w=DVector<Vector2>::Write();
		concave->set_segments(segments);

		co->add_shape(concave,get_transform());
		if (trigger)
			co->set_shape_as_trigger(co->get_shape_count()-1,true);

	}


	//co->add_shape(shape,get_transform());
}

void CollisionPolygon2D::_update_parent() {

	Node *parent = get_parent();
	if (!parent)
		return;
	CollisionObject2D *co = parent->cast_to<CollisionObject2D>();
	if (!co)
		return;
	co->_update_shapes_from_children();
}

void CollisionPolygon2D::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {

			if (!is_inside_tree())
				break;
			_update_parent();

		} break;

		case NOTIFICATION_DRAW: {
			for(int i=0;i<polygon.size();i++) {

				Vector2 p = polygon[i];
				Vector2 n = polygon[(i+1)%polygon.size()];
				draw_line(p,n,Color(0,0.6,0.7,0.5),3);
			}

			Vector< Vector<Vector2> > decomp = Geometry::decompose_polygon(polygon);
#define DEBUG_DECOMPOSE
#ifdef DEBUG_DECOMPOSE
			Color c(0.4,0.9,0.1);
			for(int i=0;i<decomp.size();i++) {

				c.set_hsv( Math::fmod(c.get_h() + 0.738,1),c.get_s(),c.get_v(),0.5);
				draw_colored_polygon(decomp[i],c);
			}
#endif
		} break;
	}
}

void CollisionPolygon2D::set_polygon(const Vector<Point2>& p_polygon) {

	polygon=p_polygon;

	for(int i=0;i<polygon.size();i++) {
		if (i==0)
			aabb=Rect2(polygon[i],Size2());
		else
			aabb.expand_to(polygon[i]);
	}
	if (aabb==Rect2()) {

		aabb=Rect2(-10,-10,20,20);
	} else {
		aabb.pos-=aabb.size*0.3;
		aabb.size+=aabb.size*0.6;
	}
	_update_parent();
	update();
}

Vector<Point2> CollisionPolygon2D::get_polygon() const {

	return polygon;
}

void CollisionPolygon2D::set_build_mode(BuildMode p_mode) {

	ERR_FAIL_INDEX(p_mode,2);
	build_mode=p_mode;
	_update_parent();
}

CollisionPolygon2D::BuildMode CollisionPolygon2D::get_build_mode() const{

	return build_mode;
}

Rect2 CollisionPolygon2D::get_item_rect() const {

	return aabb;
}

void CollisionPolygon2D::set_trigger(bool p_trigger) {

	trigger=p_trigger;
	_update_parent();
}

bool CollisionPolygon2D::is_trigger() const{

	return trigger;
}


void CollisionPolygon2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_add_to_collision_object"),&CollisionPolygon2D::_add_to_collision_object);
	ObjectTypeDB::bind_method(_MD("set_polygon","polygon"),&CollisionPolygon2D::set_polygon);
	ObjectTypeDB::bind_method(_MD("get_polygon"),&CollisionPolygon2D::get_polygon);

	ObjectTypeDB::bind_method(_MD("set_build_mode"),&CollisionPolygon2D::set_build_mode);
	ObjectTypeDB::bind_method(_MD("get_build_mode"),&CollisionPolygon2D::get_build_mode);

	ObjectTypeDB::bind_method(_MD("set_trigger"),&CollisionPolygon2D::set_trigger);
	ObjectTypeDB::bind_method(_MD("is_trigger"),&CollisionPolygon2D::is_trigger);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"build_mode",PROPERTY_HINT_ENUM,"Solids,Segments"),_SCS("set_build_mode"),_SCS("get_build_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2_ARRAY,"polygon"),_SCS("set_polygon"),_SCS("get_polygon"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"trigger"),_SCS("set_trigger"),_SCS("is_trigger"));
}

CollisionPolygon2D::CollisionPolygon2D() {

	aabb=Rect2(-10,-10,20,20);
	build_mode=BUILD_SOLIDS;
	trigger=false;


}
