#include "collision_polygon.h"

#include "collision_object.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"

void CollisionPolygon::_add_to_collision_object(Object *p_obj) {


	CollisionObject *co = p_obj->cast_to<CollisionObject>();
	ERR_FAIL_COND(!co);

	if (polygon.size()==0)
		return;

	bool solids=build_mode==BUILD_SOLIDS;

	Vector< Vector<Vector2> > decomp = Geometry::decompose_polygon(polygon);
	if (decomp.size()==0)
		return;

	if (true || solids) {

		//here comes the sun, lalalala
		//decompose concave into multiple convex polygons and add them
		for(int i=0;i<decomp.size();i++) {
			Ref<ConvexPolygonShape> convex = memnew( ConvexPolygonShape );
			DVector<Vector3> cp;
			int cs = decomp[i].size();
			cp.resize(cs*2);
			{
				DVector<Vector3>::Write w = cp.write();
				int idx=0;
				for(int j=0;j<cs;j++) {

					Vector2 d = decomp[i][j];
					w[idx++]=Vector3(d.x,d.y,depth*0.5);
					w[idx++]=Vector3(d.x,d.y,-depth*0.5);
				}
			}

			convex->set_points(cp);
			co->add_shape(convex,get_transform());

		}

	} else {
#if 0
		Ref<ConcavePolygonShape> concave = memnew( ConcavePolygonShape );

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
#endif
	}


	//co->add_shape(shape,get_transform());

}

void CollisionPolygon::_update_parent() {

	Node *parent = get_parent();
	if (!parent)
		return;
	CollisionObject *co = parent->cast_to<CollisionObject>();
	if (!co)
		return;
	co->_update_shapes_from_children();
}

void CollisionPolygon::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (!is_inside_tree())
				break;
			_update_parent();

		} break;
#if 0
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
#endif
	}
}

void CollisionPolygon::set_polygon(const Vector<Point2>& p_polygon) {

	polygon=p_polygon;

	for(int i=0;i<polygon.size();i++) {

		Vector3 p1(polygon[i].x,polygon[i].y,depth*0.5);

		if (i==0)
			aabb=AABB(p1,Vector3());
		else
			aabb.expand_to(p1);

		Vector3 p2(polygon[i].x,polygon[i].y,-depth*0.5);
		aabb.expand_to(p2);


	}
	if (aabb==AABB()) {

		aabb=AABB(Vector3(-1,-1,-1),Vector3(2,2,2));
	} else {
		aabb.pos-=aabb.size*0.3;
		aabb.size+=aabb.size*0.6;
	}
	_update_parent();
	update_gizmo();
}

Vector<Point2> CollisionPolygon::get_polygon() const {

	return polygon;
}

void CollisionPolygon::set_build_mode(BuildMode p_mode) {

	ERR_FAIL_INDEX(p_mode,2);
	build_mode=p_mode;
	_update_parent();
}

CollisionPolygon::BuildMode CollisionPolygon::get_build_mode() const{

	return build_mode;
}

AABB CollisionPolygon::get_item_rect() const {

	return aabb;
}

void CollisionPolygon::set_depth(float p_depth) {

	depth=p_depth;
	_update_parent();
	update_gizmo();
}

float CollisionPolygon::get_depth() const {

	return depth;
}


void CollisionPolygon::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_add_to_collision_object"),&CollisionPolygon::_add_to_collision_object);
	ObjectTypeDB::bind_method(_MD("set_polygon","polygon"),&CollisionPolygon::set_polygon);
	ObjectTypeDB::bind_method(_MD("get_polygon"),&CollisionPolygon::get_polygon);

	ObjectTypeDB::bind_method(_MD("set_depth","depth"),&CollisionPolygon::set_depth);
	ObjectTypeDB::bind_method(_MD("get_depth"),&CollisionPolygon::get_depth);

	ObjectTypeDB::bind_method(_MD("set_build_mode"),&CollisionPolygon::set_build_mode);
	ObjectTypeDB::bind_method(_MD("get_build_mode"),&CollisionPolygon::get_build_mode);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"build_mode",PROPERTY_HINT_ENUM,"Solids,Triangles"),_SCS("set_build_mode"),_SCS("get_build_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2_ARRAY,"polygon"),_SCS("set_polygon"),_SCS("get_polygon"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"depth"),_SCS("set_depth"),_SCS("get_depth"));
}

CollisionPolygon::CollisionPolygon() {

	aabb=AABB(Vector3(-1,-1,-1),Vector3(2,2,2));
	build_mode=BUILD_SOLIDS;
	depth=1.0;

}
