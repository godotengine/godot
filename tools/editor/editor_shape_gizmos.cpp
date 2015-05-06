/*************************************************************************/
/*  editor_shape_gizmos.cpp                                              */
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
#include "editor_shape_gizmos.h"






String EditableShapeSpatialGizmo::get_handle_name(int p_idx) const {

	if (es->cast_to<EditableSphere>()) {

		return "Radius";
	}
#if 0
	if (es->cast_to<EditableBox>()) {

		return "Extents";
	}

	if (es->cast_to<EditableCapsule>()) {

		return p_idx==0?"Radius":"Height";
	}

	if (es->cast_to<EditableRay>()) {

		return "Length";
	}
#endif
	return "";
}
Variant EditableShapeSpatialGizmo::get_handle_value(int p_idx) const{

	if (es->cast_to<EditableSphere>()) {

		EditableSphere *ss = es->cast_to<EditableSphere>();
		return ss->get_radius();
	}
#if 0
	if (es->cast_to<EditableBox>()) {

		EditableBox *bs = es->cast_to<EditableBox>();
		return bs->get_extents();
	}

	if (es->cast_to<EditableCapsule>()) {

		EditableCapsule *cs = es->cast_to<EditableCapsule>();
		return p_idx==0?es->get_radius():es->get_height();
	}

	if (es->cast_to<EditableRay>()) {

		EditableRay* cs = es->cast_to<EditableRay>();
		return es->get_length();
	}
#endif
	return Variant();
}
void EditableShapeSpatialGizmo::set_handle(int p_idx,Camera *p_camera, const Point2& p_point){

	Transform gt = es->get_global_transform();
	gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2]={gi.xform(ray_from),gi.xform(ray_from+ray_dir*4096)};

	if (es->cast_to<EditableSphere>()) {

		EditableSphere *ss = es->cast_to<EditableSphere>();
		Vector3 ra,rb;
		Geometry::get_closest_points_between_segments(Vector3(),Vector3(4096,0,0),sg[0],sg[1],ra,rb);
		float d = ra.x;
		if (d<0.001)
			d=0.001;

		ss->set_radius(d);
	}

#if 0
	if (es->cast_to<EditableRay>()) {

		EditableRay*cs = es->cast_to<EditableRay>();
		Vector3 ra,rb;
		Geometry::get_closest_points_between_segments(Vector3(),Vector3(0,0,4096),sg[0],sg[1],ra,rb);
		float d = ra.z;
		if (d<0.001)
			d=0.001;

		rs->set_length(d);
	}


	if (es->cast_to<EditableBox>()) {

		Vector3 axis;
		axis[p_idx]=1.0;
		EditableBox *bs = es->cast_to<EditableBox>();
		Vector3 ra,rb;
		Geometry::get_closest_points_between_segments(Vector3(),axis*4096,sg[0],sg[1],ra,rb);
		float d = ra[p_idx];
		if (d<0.001)
			d=0.001;

		Vector3 he = bs->get_extents();
		he[p_idx]=d;
		bs->set_extents(he);

	}

	if (es->cast_to<EditableCapsule>()) {

		Vector3 axis;
		axis[p_idx]=1.0;
		EditableCapsule *cs = es->cast_to<EditableCapsule>();
		Vector3 ra,rb;
		Geometry::get_closest_points_between_segments(Vector3(),axis*4096,sg[0],sg[1],ra,rb);
		float d = ra[p_idx];
		if (p_idx==1)
			d-=es->get_radius();
		if (d<0.001)
			d=0.001;

		if (p_idx==0)
			es->set_radius(d);
		else if (p_idx==1)
			es->set_height(d*2.0);

	}

#endif

}
void EditableShapeSpatialGizmo::commit_handle(int p_idx,const Variant& p_restore,bool p_cancel){


	if (es->cast_to<EditableSphere>()) {

		EditableSphere *ss = es->cast_to<EditableSphere>();
		if (p_cancel) {
			ss->set_radius(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action("Change Sphere Shape Radius");
		ur->add_do_method(ss,"set_radius",ss->get_radius());
		ur->add_undo_method(ss,"set_radius",p_restore);
		ur->commit_action();

	}
#if 0
	if (es->cast_to<EditableBox>()) {

		EditableBox *ss = es->cast_to<EditableBox>();
		if (p_cancel) {
			ss->set_extents(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action("Change Box Shape Extents");
		ur->add_do_method(ss,"set_extents",ss->get_extents());
		ur->add_undo_method(ss,"set_extents",p_restore);
		ur->commit_action();
	}

	if (es->cast_to<EditableCapsule>()) {

		EditableCapsule *cs = es->cast_to<EditableCapsule>();
		if (p_cancel) {
			if (p_idx==0)
				ss->set_radius(p_restore);
			else
				ss->set_height(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		if (p_idx==0) {
			ur->create_action("Change Capsule Shape Radius");
			ur->add_do_method(ss,"set_radius",ss->get_radius());
			ur->add_undo_method(ss,"set_radius",p_restore);
		} else {
			ur->create_action("Change Capsule Shape Height");
			ur->add_do_method(ss,"set_height",ss->get_height());
			ur->add_undo_method(ss,"set_height",p_restore);

		}

		ur->commit_action();

	}

	if (es->cast_to<EditableRay>()) {

		EditableRay*rs = es->cast_to<EditableRay>()
		if (p_cancel) {
			ss->set_length(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action("Change Ray Shape Length");
		ur->add_do_method(ss,"set_length",ss->get_length());
		ur->add_undo_method(ss,"set_length",p_restore);
		ur->commit_action();

	}
#endif
}
void EditableShapeSpatialGizmo::redraw(){

	clear();

	if (es->cast_to<EditableSphere>()) {

		EditableSphere* sp= es->cast_to<EditableSphere>();
		float r=sp->get_radius();

		Vector<Vector3> points;

		for(int i=0;i<=360;i++) {

			float ra=Math::deg2rad(i);
			float rb=Math::deg2rad(i+1);
			Point2 a = Vector2(Math::sin(ra),Math::cos(ra))*r;
			Point2 b = Vector2(Math::sin(rb),Math::cos(rb))*r;

			points.push_back(Vector3(a.x,0,a.y));
			points.push_back(Vector3(b.x,0,b.y));
			points.push_back(Vector3(0,a.x,a.y));
			points.push_back(Vector3(0,b.x,b.y));
			points.push_back(Vector3(a.x,a.y,0));
			points.push_back(Vector3(b.x,b.y,0));

		}

		Vector<Vector3> collision_segments;

		for(int i=0;i<64;i++) {

			float ra=i*Math_PI*2.0/64.0;
			float rb=(i+1)*Math_PI*2.0/64.0;
			Point2 a = Vector2(Math::sin(ra),Math::cos(ra))*r;
			Point2 b = Vector2(Math::sin(rb),Math::cos(rb))*r;

			collision_segments.push_back(Vector3(a.x,0,a.y));
			collision_segments.push_back(Vector3(b.x,0,b.y));
			collision_segments.push_back(Vector3(0,a.x,a.y));
			collision_segments.push_back(Vector3(0,b.x,b.y));
			collision_segments.push_back(Vector3(a.x,a.y,0));
			collision_segments.push_back(Vector3(b.x,b.y,0));
		}

		add_lines(points,SpatialEditorGizmos::singleton->shape_material);
		add_collision_segments(collision_segments);
		Vector<Vector3> handles;
		handles.push_back(Vector3(r,0,0));
		add_handles(handles);

	}

#if 0
	if (es->cast_to<EditableBox>()) {

		EditableBox*bs = es->cast_to<EditableBox>();
		Vector<Vector3> lines;
		AABB aabb;
		aabb.pos=-bs->get_extents();
		aabb.size=aabb.pos*-2;

		for(int i=0;i<12;i++) {
			Vector3 a,b;
			aabb.get_edge(i,a,b);
			lines.push_back(a);
			lines.push_back(b);
		}

		Vector<Vector3> handles;

		for(int i=0;i<3;i++) {

			Vector3 ax;
			ax[i]=bs->get_extents()[i];
			handles.push_back(ax);
		}

		add_lines(lines,SpatialEditorGizmos::singleton->shape_material);
		add_collision_segments(lines);
		add_handles(handles);

	}

	if (es->cast_to<EditableCapsule>()) {

		EditableCapsule *cs = es->cast_to<EditableCapsule>();
		float radius = es->get_radius();
		float height = es->get_height();


		Vector<Vector3> points;

		Vector3 d(0,height*0.5,0);
		for(int i=0;i<360;i++) {

			float ra=Math::deg2rad(i);
			float rb=Math::deg2rad(i+1);
			Point2 a = Vector2(Math::sin(ra),Math::cos(ra))*radius;
			Point2 b = Vector2(Math::sin(rb),Math::cos(rb))*radius;

			points.push_back(Vector3(a.x,0,a.y)+d);
			points.push_back(Vector3(b.x,0,b.y)+d);

			points.push_back(Vector3(a.x,0,a.y)-d);
			points.push_back(Vector3(b.x,0,b.y)-d);

			if (i%90==0) {

				points.push_back(Vector3(a.x,0,a.y)+d);
				points.push_back(Vector3(a.x,0,a.y)-d);
			}

			Vector3 dud = i<180?d:-d;

			points.push_back(Vector3(0,a.x,a.y)+dud);
			points.push_back(Vector3(0,b.x,b.y)+dud);
			points.push_back(Vector3(a.y,a.x,0)+dud);
			points.push_back(Vector3(b.y,b.x,0)+dud);

		}

		add_lines(points,SpatialEditorGizmos::singleton->shape_material);

		Vector<Vector3> collision_segments;

		for(int i=0;i<64;i++) {

			float ra=i*Math_PI*2.0/64.0;
			float rb=(i+1)*Math_PI*2.0/64.0;
			Point2 a = Vector2(Math::sin(ra),Math::cos(ra))*radius;
			Point2 b = Vector2(Math::sin(rb),Math::cos(rb))*radius;

			collision_segments.push_back(Vector3(a.x,0,a.y)+d);
			collision_segments.push_back(Vector3(b.x,0,b.y)+d);

			collision_segments.push_back(Vector3(a.x,0,a.y)-d);
			collision_segments.push_back(Vector3(b.x,0,b.y)-d);

			if (i%16==0) {

				collision_segments.push_back(Vector3(a.x,0,a.y)+d);
				collision_segments.push_back(Vector3(a.x,0,a.y)-d);
			}

			Vector3 dud = i<32?d:-d;

			collision_segments.push_back(Vector3(0,a.x,a.y)+dud);
			collision_segments.push_back(Vector3(0,b.x,b.y)+dud);
			collision_segments.push_back(Vector3(a.y,a.x,0)+dud);
			collision_segments.push_back(Vector3(b.y,b.x,0)+dud);

		}

		add_collision_segments(collision_segments);

		Vector<Vector3> handles;
		handles.push_back(Vector3(es->get_radius(),0,0));
		handles.push_back(Vector3(0,es->get_height()*0.5+es->get_radius(),0));
		add_handles(handles);


	}

	if (es->cast_to<EditablePlane>()) {

		EditablePlane* ps=es->cast_to<EditablePlane();
		Plane p = ps->get_plane();
		Vector<Vector3> points;

		Vector3 n1 = p.get_any_perpendicular_normal();
		Vector3 n2 = p.normal.cross(n1).normalized();

		Vector3 pface[4]={
			p.normal*p.d+n1*10.0+n2*10.0,
			p.normal*p.d+n1*10.0+n2*-10.0,
			p.normal*p.d+n1*-10.0+n2*-10.0,
			p.normal*p.d+n1*-10.0+n2*10.0,
		};

		points.push_back(pface[0]);
		points.push_back(pface[1]);
		points.push_back(pface[1]);
		points.push_back(pface[2]);
		points.push_back(pface[2]);
		points.push_back(pface[3]);
		points.push_back(pface[3]);
		points.push_back(pface[0]);
		points.push_back(p.normal*p.d);
		points.push_back(p.normal*p.d+p.normal*3);

		add_lines(points,SpatialEditorGizmos::singleton->shape_material);
		add_collision_segments(points);

	}


	if (es->cast_to<EditableRay>()) {

		EditableRay*cs = es->cast_to<EditableRay>();

		Vector<Vector3> points;
		points.push_back(Vector3());
		points.push_back(Vector3(0,0,rs->get_length()));
		add_lines(points,SpatialEditorGizmos::singleton->shape_material);
		add_collision_segments(points);
		Vector<Vector3> handles;
		handles.push_back(Vector3(0,0,rs->get_length()));
		add_handles(handles);


	}

#endif

}
EditableShapeSpatialGizmo::EditableShapeSpatialGizmo(EditableShape* p_cs) {

	es=p_cs;
	set_spatial_node(p_cs);
}



EditorShapeGizmos::EditorShapeGizmos()
{
}
