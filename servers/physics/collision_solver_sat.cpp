/*************************************************************************/
/*  collision_solver_sat.cpp                                             */
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
#include "collision_solver_sat.h"
#include "geometry.h"

#define _EDGE_IS_VALID_SUPPORT_TRESHOLD 0.02

struct _CollectorCallback {

	CollisionSolverSW::CallbackResult callback;
	void *userdata;
	bool swap;
	bool collided;
	Vector3 normal;
	Vector3 *prev_axis;

	_FORCE_INLINE_ void call(const Vector3& p_point_A, const Vector3& p_point_B) {

		//if (normal.dot(p_point_A) >= normal.dot(p_point_B))
		// return;
		if (swap)
			callback(p_point_B,p_point_A,userdata);
		else
			callback(p_point_A,p_point_B,userdata);
	}

};

typedef void (*GenerateContactsFunc)(const Vector3 *,int, const Vector3 *,int ,_CollectorCallback *);


static void _generate_contacts_point_point(const Vector3 * p_points_A,int p_point_count_A, const Vector3 * p_points_B,int p_point_count_B,_CollectorCallback *p_callback) {

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND( p_point_count_A != 1 );
	ERR_FAIL_COND( p_point_count_B != 1 );
#endif

	p_callback->call(*p_points_A,*p_points_B);
}

static void _generate_contacts_point_edge(const Vector3 * p_points_A,int p_point_count_A, const Vector3 * p_points_B,int p_point_count_B,_CollectorCallback *p_callback) {

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND( p_point_count_A != 1 );
	ERR_FAIL_COND( p_point_count_B != 2 );
#endif

	Vector3 closest_B = Geometry::get_closest_point_to_segment_uncapped(*p_points_A, p_points_B );
	p_callback->call(*p_points_A,closest_B);

}

static void _generate_contacts_point_face(const Vector3 * p_points_A,int p_point_count_A, const Vector3 * p_points_B,int p_point_count_B,_CollectorCallback *p_callback) {

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND( p_point_count_A != 1 );
	ERR_FAIL_COND( p_point_count_B < 3 );
#endif


	Vector3 closest_B=Plane(p_points_B[0],p_points_B[1],p_points_B[2]).project( *p_points_A );

	p_callback->call(*p_points_A,closest_B);

}


static void _generate_contacts_edge_edge(const Vector3 * p_points_A,int p_point_count_A, const Vector3 * p_points_B,int p_point_count_B,_CollectorCallback *p_callback) {

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND( p_point_count_A != 2 );
	ERR_FAIL_COND( p_point_count_B != 2 ); // circle is actually a 4x3 matrix
#endif



	Vector3 rel_A=p_points_A[1]-p_points_A[0];
	Vector3 rel_B=p_points_B[1]-p_points_B[0];

	Vector3 c=rel_A.cross(rel_B).cross(rel_B);

//	if ( Math::abs(rel_A.dot(c) )<_EDGE_IS_VALID_SUPPORT_TRESHOLD ) {
	if ( Math::abs(rel_A.dot(c) )<CMP_EPSILON ) {

		// should handle somehow..
		//ERR_PRINT("TODO FIX");
		//return;

		Vector3 axis = rel_A.normalized(); //make an axis
		Vector3 base_A = p_points_A[0] - axis * axis.dot(p_points_A[0]);
		Vector3 base_B = p_points_B[0] - axis * axis.dot(p_points_B[0]);

		//sort all 4 points in axis
		float dvec[4]={ axis.dot(p_points_A[0]), axis.dot(p_points_A[1]), axis.dot(p_points_B[0]), axis.dot(p_points_B[1]) };

		SortArray<float> sa;
		sa.sort(dvec,4);

		//use the middle ones as contacts
		p_callback->call(base_A+axis*dvec[1],base_B+axis*dvec[1]);
		p_callback->call(base_A+axis*dvec[2],base_B+axis*dvec[2]);

		return;

	}

	real_t d = (c.dot( p_points_B[0] ) - p_points_A[0].dot(c))/rel_A.dot(c);

	if (d<0.0)
		d=0.0;
	else if (d>1.0)
		d=1.0;

	Vector3 closest_A=p_points_A[0]+rel_A*d;
	Vector3 closest_B=Geometry::get_closest_point_to_segment_uncapped(closest_A, p_points_B);
	p_callback->call(closest_A,closest_B);

}

static void _generate_contacts_face_face(const Vector3 * p_points_A,int p_point_count_A, const Vector3 * p_points_B,int p_point_count_B,_CollectorCallback *p_callback) {

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND( p_point_count_A <2 );
	ERR_FAIL_COND( p_point_count_B <3 );
#endif

	static const int max_clip=32;

	Vector3 _clipbuf1[max_clip];
	Vector3 _clipbuf2[max_clip];
	Vector3 *clipbuf_src=_clipbuf1;
	Vector3 *clipbuf_dst=_clipbuf2;
	int clipbuf_len=p_point_count_A;

	// copy A points to clipbuf_src
	for (int i=0;i<p_point_count_A;i++) {

		clipbuf_src[i]=p_points_A[i];
	}

	Plane plane_B(p_points_B[0],p_points_B[1],p_points_B[2]);

	// go through all of B points
	for (int i=0;i<p_point_count_B;i++) {

		int i_n=(i+1)%p_point_count_B;

		Vector3 edge0_B=p_points_B[i];
		Vector3 edge1_B=p_points_B[i_n];

		Vector3 clip_normal = (edge0_B - edge1_B).cross( plane_B.normal ).normalized();
		// make a clip plane


		Plane clip(edge0_B,clip_normal);
		// avoid double clip if A is edge
		int dst_idx=0;
		bool edge = clipbuf_len==2;
		for (int j=0;j<clipbuf_len;j++) {

			int j_n=(j+1)%clipbuf_len;

			Vector3 edge0_A=clipbuf_src[j];
			Vector3 edge1_A=clipbuf_src[j_n];

			real_t dist0 = clip.distance_to(edge0_A);
			real_t dist1 = clip.distance_to(edge1_A);


			if ( dist0 <= 0 ) { // behind plane

				ERR_FAIL_COND( dst_idx >= max_clip );
				clipbuf_dst[dst_idx++]=clipbuf_src[j];

			}


			// check for different sides and non coplanar
//			if ( (dist0*dist1) < -CMP_EPSILON && !(edge && j)) {
			if ( (dist0*dist1) < 0 && !(edge && j)) {

				// calculate intersection
				Vector3 rel = edge1_A - edge0_A;
				real_t den=clip.normal.dot( rel );
				real_t dist=-(clip.normal.dot( edge0_A )-clip.d)/den;
				Vector3 inters = edge0_A+rel*dist;

				ERR_FAIL_COND( dst_idx >= max_clip );
				clipbuf_dst[dst_idx]=inters;
				dst_idx++;
			}
		}

		clipbuf_len=dst_idx;
		SWAP(clipbuf_src,clipbuf_dst);
	}


	// generate contacts
	//Plane plane_A(p_points_A[0],p_points_A[1],p_points_A[2]);

	int added=0;

	for (int i=0;i<clipbuf_len;i++) {

		float d = plane_B.distance_to(clipbuf_src[i]);
		if (d>CMP_EPSILON)
			continue;

		Vector3 closest_B=clipbuf_src[i] - plane_B.normal*d;

		p_callback->call(clipbuf_src[i],closest_B);
		added++;

	}

}


static void _generate_contacts_from_supports(const Vector3 * p_points_A,int p_point_count_A, const Vector3 * p_points_B,int p_point_count_B,_CollectorCallback *p_callback) {


#ifdef DEBUG_ENABLED
	ERR_FAIL_COND( p_point_count_A <1 );
	ERR_FAIL_COND( p_point_count_B <1 );
#endif


	static const GenerateContactsFunc generate_contacts_func_table[3][3]={
		{
			_generate_contacts_point_point,
			_generate_contacts_point_edge,
			_generate_contacts_point_face,
		},{
			0,
			_generate_contacts_edge_edge,
			_generate_contacts_face_face,
		},{
			0,0,
			_generate_contacts_face_face,
		}
	};

	int pointcount_B;
	int pointcount_A;
	const Vector3 *points_A;
	const Vector3 *points_B;

	if (p_point_count_A > p_point_count_B) {
		//swap
		p_callback->swap = !p_callback->swap;
		p_callback->normal = -p_callback->normal;

		pointcount_B = p_point_count_A;
		pointcount_A = p_point_count_B;
		points_A=p_points_B;
		points_B=p_points_A;
	} else {

		pointcount_B = p_point_count_B;
		pointcount_A = p_point_count_A;
		points_A=p_points_A;
		points_B=p_points_B;
	}

	int version_A = (pointcount_A > 3 ?  3 : pointcount_A) -1;
	int version_B = (pointcount_B > 3 ?  3 : pointcount_B) -1;

	GenerateContactsFunc contacts_func = generate_contacts_func_table[version_A][version_B];
	ERR_FAIL_COND(!contacts_func);
	contacts_func(points_A,pointcount_A,points_B,pointcount_B,p_callback);

}



template<class ShapeA, class ShapeB>
class SeparatorAxisTest {

	const ShapeA *shape_A;
	const ShapeB *shape_B;
	const Transform *transform_A;
	const Transform *transform_B;
	real_t best_depth;
	Vector3 best_axis;
	_CollectorCallback *callback;

	Vector3 separator_axis;

public:

	_FORCE_INLINE_ bool test_previous_axis() {

		if (callback && callback->prev_axis && *callback->prev_axis!=Vector3())
			return test_axis(*callback->prev_axis);
		else
			return true;
	}

	_FORCE_INLINE_ bool test_axis(const Vector3& p_axis) {

		Vector3 axis=p_axis;

		if (	Math::abs(axis.x)<CMP_EPSILON &&
			Math::abs(axis.y)<CMP_EPSILON &&
			Math::abs(axis.z)<CMP_EPSILON ) {
			// strange case, try an upwards separator
			axis=Vector3(0.0,1.0,0.0);
		}

		real_t min_A,max_A,min_B,max_B;

		shape_A->project_range(axis,*transform_A,min_A,max_A);
		shape_B->project_range(axis,*transform_B,min_B,max_B);

		min_B -= ( max_A - min_A ) * 0.5;
		max_B += ( max_A - min_A ) * 0.5;

		real_t dmin = min_B - ( min_A + max_A ) * 0.5;
		real_t dmax = max_B - ( min_A + max_A ) * 0.5;

		if (dmin > 0.0 || dmax < 0.0) {
			separator_axis=axis;
			return false; // doesn't contain 0
		}

		//use the smallest depth

		dmin = Math::abs(dmin);

		if ( dmax < dmin ) {
			if ( dmax < best_depth ) {
				best_depth=dmax;
				best_axis=axis;
			}
		} else {
			if ( dmin < best_depth ) {
				best_depth=dmin;
				best_axis=-axis; // keep it as A axis
			}
		}

		return true;
	}


	_FORCE_INLINE_ void generate_contacts() {

		// nothing to do, don't generate
		if (best_axis==Vector3(0.0,0.0,0.0))
			return;

		if (!callback->callback) {
			//just was checking intersection?
			callback->collided=true;
			if (callback->prev_axis)
				*callback->prev_axis=best_axis;
			return;
		}

		static const int max_supports=16;

		Vector3 supports_A[max_supports];
		int support_count_A;
		shape_A->get_supports(transform_A->basis.xform_inv(-best_axis).normalized(),max_supports,supports_A,support_count_A);
		for(int i=0;i<support_count_A;i++) {
			supports_A[i] = transform_A->xform(supports_A[i]);
		}


		Vector3 supports_B[max_supports];
		int support_count_B;
		shape_B->get_supports(transform_B->basis.xform_inv(best_axis).normalized(),max_supports,supports_B,support_count_B);
		for(int i=0;i<support_count_B;i++) {
			supports_B[i] = transform_B->xform(supports_B[i]);
		}
/*
		print_line("best depth: "+rtos(best_depth));
		for(int i=0;i<support_count_A;i++) {

			print_line("A-"+itos(i)+": "+supports_A[i]);
		}
		for(int i=0;i<support_count_B;i++) {

			print_line("B-"+itos(i)+": "+supports_B[i]);
		}
*/
		callback->normal=best_axis;
		if (callback->prev_axis)
			*callback->prev_axis=best_axis;
		_generate_contacts_from_supports(supports_A,support_count_A,supports_B,support_count_B,callback);

		callback->collided=true;
		//CollisionSolverSW::CallbackResult cbk=NULL;
		//cbk(Vector3(),Vector3(),NULL);

	}

	_FORCE_INLINE_ SeparatorAxisTest(const ShapeA *p_shape_A,const Transform& p_transform_A, const ShapeB *p_shape_B,const Transform& p_transform_B,_CollectorCallback *p_callback) {
		best_depth=1e15;
		shape_A=p_shape_A;
		shape_B=p_shape_B;
		transform_A=&p_transform_A;
		transform_B=&p_transform_B;
		callback=p_callback;
	}

};

/****** SAT TESTS *******/
/****** SAT TESTS *******/
/****** SAT TESTS *******/
/****** SAT TESTS *******/


typedef void (*CollisionFunc)(const ShapeSW*,const Transform&,const ShapeSW*,const Transform&,_CollectorCallback *p_callback);


static void _collision_sphere_sphere(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {


	const SphereShapeSW *sphere_A = static_cast<const SphereShapeSW*>(p_a);
	const SphereShapeSW *sphere_B = static_cast<const SphereShapeSW*>(p_b);

	SeparatorAxisTest<SphereShapeSW,SphereShapeSW> separator(sphere_A,p_transform_a,sphere_B,p_transform_b,p_collector);

	// previous axis

	if (!separator.test_previous_axis())
		return;

	if (!separator.test_axis( (p_transform_a.origin-p_transform_b.origin).normalized() ))
		return;

	separator.generate_contacts();
}

static void _collision_sphere_box(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {


	const SphereShapeSW *sphere_A = static_cast<const SphereShapeSW*>(p_a);
	const BoxShapeSW *box_B = static_cast<const BoxShapeSW*>(p_b);

	SeparatorAxisTest<SphereShapeSW,BoxShapeSW> separator(sphere_A,p_transform_a,box_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	// test faces

	for (int i=0;i<3;i++) {

		Vector3 axis = p_transform_b.basis.get_axis(i).normalized();

		if (!separator.test_axis( axis ))
			return;

	}

	// calculate closest point to sphere

	Vector3 cnormal=p_transform_b.xform_inv( p_transform_a.origin );

	Vector3 cpoint=p_transform_b.xform( Vector3(

		(cnormal.x<0) ? -box_B->get_half_extents().x : box_B->get_half_extents().x,
		(cnormal.y<0) ? -box_B->get_half_extents().y : box_B->get_half_extents().y,
		(cnormal.z<0) ? -box_B->get_half_extents().z : box_B->get_half_extents().z
	) );

	// use point to test axis
	Vector3 point_axis = (p_transform_a.origin - cpoint).normalized();

	if (!separator.test_axis( point_axis  ))
		return;

	// test edges

	for (int i=0;i<3;i++) {

		Vector3 axis = point_axis.cross( p_transform_b.basis.get_axis(i) ).cross( p_transform_b.basis.get_axis(i) ).normalized();

		if (!separator.test_axis( axis  ))
			return;
	}

	separator.generate_contacts();


}


static void _collision_sphere_capsule(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {

	const SphereShapeSW *sphere_A = static_cast<const SphereShapeSW*>(p_a);
	const CapsuleShapeSW *capsule_B = static_cast<const CapsuleShapeSW*>(p_b);

	SeparatorAxisTest<SphereShapeSW,CapsuleShapeSW> separator(sphere_A,p_transform_a,capsule_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	//capsule sphere 1, sphere

	Vector3 capsule_axis = p_transform_b.basis.get_axis(2) * (capsule_B->get_height() * 0.5);

	Vector3 capsule_ball_1 = p_transform_b.origin + capsule_axis;

	if (!separator.test_axis( (capsule_ball_1 - p_transform_a.origin).normalized() ) )
		return;

	//capsule sphere 2, sphere

	Vector3 capsule_ball_2 = p_transform_b.origin - capsule_axis;

	if (!separator.test_axis( (capsule_ball_1 - p_transform_a.origin).normalized() ) )
		return;

	//capsule edge, sphere

	Vector3 b2a = p_transform_a.origin - p_transform_b.origin;

	Vector3 axis = b2a.cross( capsule_axis ).cross( capsule_axis ).normalized();


	if (!separator.test_axis( axis ))
		return;

	separator.generate_contacts();
}

static void _collision_sphere_convex_polygon(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {


	const SphereShapeSW *sphere_A = static_cast<const SphereShapeSW*>(p_a);
	const ConvexPolygonShapeSW *convex_polygon_B = static_cast<const ConvexPolygonShapeSW*>(p_b);

	SeparatorAxisTest<SphereShapeSW,ConvexPolygonShapeSW> separator(sphere_A,p_transform_a,convex_polygon_B,p_transform_b,p_collector);


	if (!separator.test_previous_axis())
		return;

	const Geometry::MeshData &mesh = convex_polygon_B->get_mesh();

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();


	// faces of B
	for (int i=0;i<face_count;i++) {

		Vector3 axis = p_transform_b.xform( faces[i].plane ).normal;

		if (!separator.test_axis( axis ))
			return;
	}


	// edges of B
	for(int i=0;i<edge_count;i++) {


		Vector3 v1=p_transform_b.xform( vertices[ edges[i].a ] );
		Vector3 v2=p_transform_b.xform( vertices[ edges[i].b ] );
		Vector3 v3=p_transform_a.origin;


		Vector3 n1=v2-v1;
		Vector3 n2=v2-v3;

		Vector3 axis = n1.cross(n2).cross(n1).normalized();;

		if (!separator.test_axis( axis ))
			return;

	}

	// vertices of B
	for(int i=0;i<vertex_count;i++) {


		Vector3 v1=p_transform_b.xform( vertices[i] );
		Vector3 v2=p_transform_a.origin;

		Vector3 axis = (v2-v1).normalized();

		if (!separator.test_axis( axis ))
			return;

	}

	separator.generate_contacts();


}

static void _collision_sphere_face(const ShapeSW *p_a,const Transform &p_transform_a, const ShapeSW *p_b,const Transform& p_transform_b, _CollectorCallback *p_collector) {

	const SphereShapeSW *sphere_A = static_cast<const SphereShapeSW*>(p_a);
	const FaceShapeSW *face_B = static_cast<const FaceShapeSW*>(p_b);



	SeparatorAxisTest<SphereShapeSW,FaceShapeSW> separator(sphere_A,p_transform_a,face_B,p_transform_b,p_collector);


	Vector3 vertex[3]={
		p_transform_b.xform( face_B->vertex[0] ),
		p_transform_b.xform( face_B->vertex[1] ),
		p_transform_b.xform( face_B->vertex[2] ),
	};

	if (!separator.test_axis( (vertex[0]-vertex[2]).cross(vertex[0]-vertex[1]).normalized() ))
		return;

	// edges and points of B
	for(int i=0;i<3;i++) {


		Vector3 n1=vertex[i]-p_transform_a.origin;

		if (!separator.test_axis( n1.normalized() )) {
			return;
		}

		Vector3 n2=vertex[(i+1)%3]-vertex[i];

		Vector3 axis = n1.cross(n2).cross(n2).normalized();

		if (!separator.test_axis( axis )) {
			return;
		}

	}

	separator.generate_contacts();
}





static void _collision_box_box(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {


	const BoxShapeSW *box_A = static_cast<const BoxShapeSW*>(p_a);
	const BoxShapeSW *box_B = static_cast<const BoxShapeSW*>(p_b);

	SeparatorAxisTest<BoxShapeSW,BoxShapeSW> separator(box_A,p_transform_a,box_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	// test faces of A

	for (int i=0;i<3;i++) {

		Vector3 axis = p_transform_a.basis.get_axis(i).normalized();

		if (!separator.test_axis( axis ))
			return;

	}

	// test faces of B

	for (int i=0;i<3;i++) {

		Vector3 axis = p_transform_b.basis.get_axis(i).normalized();

		if (!separator.test_axis( axis ))
			return;

	}

	// test combined edges
	for (int i=0;i<3;i++) {

		for (int j=0;j<3;j++) {

			Vector3 axis = p_transform_a.basis.get_axis(i).cross( p_transform_b.basis.get_axis(j) );

			if (axis.length_squared()<CMP_EPSILON)
				continue;
			axis.normalize();


			if (!separator.test_axis( axis  )) {
				return;
			}
		}
	}

	separator.generate_contacts();


}


static void _collision_box_capsule(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {

	const BoxShapeSW *box_A = static_cast<const BoxShapeSW*>(p_a);
	const CapsuleShapeSW *capsule_B = static_cast<const CapsuleShapeSW*>(p_b);

	SeparatorAxisTest<BoxShapeSW,CapsuleShapeSW> separator(box_A,p_transform_a,capsule_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	// faces of A
	for (int i=0;i<3;i++) {

		Vector3 axis = p_transform_a.basis.get_axis(i);

		if (!separator.test_axis( axis ))
			return;
	}


	Vector3 cyl_axis = p_transform_b.basis.get_axis(2).normalized();

	// edges of A, capsule cylinder

	for (int i=0;i<3;i++) {

		// cylinder
		Vector3 box_axis = p_transform_a.basis.get_axis(i);
		Vector3 axis = box_axis.cross( cyl_axis );
		if (axis.length_squared() < CMP_EPSILON)
			continue;

		if (!separator.test_axis( axis.normalized() ))
			return;
	}

	// points of A, capsule cylinder
	// this sure could be made faster somehow..

	for (int i=0;i<2;i++) {
		for (int j=0;j<2;j++) {
			for (int k=0;k<2;k++) {
				Vector3 he = box_A->get_half_extents();
				he.x*=(i*2-1);
				he.y*=(j*2-1);
				he.z*=(k*2-1);
				Vector3 point=p_transform_a.origin;
				for(int l=0;l<3;l++)
					point+=p_transform_a.basis.get_axis(l)*he[l];

				//Vector3 axis = (point - cyl_axis * cyl_axis.dot(point)).normalized();
				Vector3 axis = Plane(cyl_axis,0).project(point).normalized();

				if (!separator.test_axis( axis ))
					return;
			}
		}
	}

	// capsule balls, edges of A

	for (int i=0;i<2;i++) {


		Vector3 capsule_axis = p_transform_b.basis.get_axis(2)*(capsule_B->get_height()*0.5);

		Vector3 sphere_pos = p_transform_b.origin + ((i==0)?capsule_axis:-capsule_axis);


		Vector3 cnormal=p_transform_a.xform_inv( sphere_pos );

		Vector3 cpoint=p_transform_a.xform( Vector3(

			(cnormal.x<0) ? -box_A->get_half_extents().x : box_A->get_half_extents().x,
			(cnormal.y<0) ? -box_A->get_half_extents().y : box_A->get_half_extents().y,
			(cnormal.z<0) ? -box_A->get_half_extents().z : box_A->get_half_extents().z
		) );

		// use point to test axis
		Vector3 point_axis = (sphere_pos - cpoint).normalized();

		if (!separator.test_axis( point_axis  ))
			return;

		// test edges of A

		for (int i=0;i<3;i++) {

			Vector3 axis = point_axis.cross( p_transform_a.basis.get_axis(i) ).cross( p_transform_a.basis.get_axis(i) ).normalized();

			if (!separator.test_axis( axis  ))
				return;
		}
	}


	separator.generate_contacts();
}


static void _collision_box_convex_polygon(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {



	const BoxShapeSW *box_A = static_cast<const BoxShapeSW*>(p_a);
	const ConvexPolygonShapeSW *convex_polygon_B = static_cast<const ConvexPolygonShapeSW*>(p_b);

	SeparatorAxisTest<BoxShapeSW,ConvexPolygonShapeSW> separator(box_A,p_transform_a,convex_polygon_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;


	const Geometry::MeshData &mesh = convex_polygon_B->get_mesh();

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();

	// faces of A
	for (int i=0;i<3;i++) {

		Vector3 axis = p_transform_a.basis.get_axis(i).normalized();

		if (!separator.test_axis( axis ))
			return;
	}

	// faces of B
	for (int i=0;i<face_count;i++) {

		Vector3 axis = p_transform_b.xform( faces[i].plane ).normal;

		if (!separator.test_axis( axis ))
			return;
	}

	// A<->B edges
	for (int i=0;i<3;i++) {

		Vector3 e1 = p_transform_a.basis.get_axis(i);

		for (int j=0;j<edge_count;j++) {

			Vector3 e2=p_transform_b.basis.xform(vertices[edges[j].a]) -  p_transform_b.basis.xform(vertices[edges[j].b]);

			Vector3 axis=e1.cross( e2 ).normalized();

			if (!separator.test_axis( axis ))
				return;

		}
	}

	separator.generate_contacts();


}

static void _collision_box_face(const ShapeSW *p_a,const Transform &p_transform_a, const ShapeSW *p_b,const Transform& p_transform_b, _CollectorCallback *p_collector) {


	const BoxShapeSW *box_A = static_cast<const BoxShapeSW*>(p_a);
	const FaceShapeSW *face_B = static_cast<const FaceShapeSW*>(p_b);

	SeparatorAxisTest<BoxShapeSW,FaceShapeSW> separator(box_A,p_transform_a,face_B,p_transform_b,p_collector);

	Vector3 vertex[3]={
		p_transform_b.xform( face_B->vertex[0] ),
		p_transform_b.xform( face_B->vertex[1] ),
		p_transform_b.xform( face_B->vertex[2] ),
	};

	if (!separator.test_axis( (vertex[0]-vertex[2]).cross(vertex[0]-vertex[1]).normalized() ))
		return;

	// faces of A
	for (int i=0;i<3;i++) {

		Vector3 axis = p_transform_a.basis.get_axis(i).normalized();

		if (!separator.test_axis( axis ))
			return;
	}

	// combined edges
	for(int i=0;i<3;i++) {

		Vector3 e=vertex[i]-vertex[(i+1)%3];

		for (int i=0;i<3;i++) {

			Vector3 axis = p_transform_a.basis.get_axis(i);

			if (!separator.test_axis( e.cross(axis).normalized() ))
				return;
		}

	}

	separator.generate_contacts();

}

static void _collision_capsule_capsule(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {

	const CapsuleShapeSW *capsule_A = static_cast<const CapsuleShapeSW*>(p_a);
	const CapsuleShapeSW *capsule_B = static_cast<const CapsuleShapeSW*>(p_b);

	SeparatorAxisTest<CapsuleShapeSW,CapsuleShapeSW> separator(capsule_A,p_transform_a,capsule_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	// some values

	Vector3 capsule_A_axis = p_transform_a.basis.get_axis(2) * (capsule_A->get_height() * 0.5);
	Vector3 capsule_B_axis = p_transform_b.basis.get_axis(2) * (capsule_B->get_height() * 0.5);

	Vector3 capsule_A_ball_1 = p_transform_a.origin + capsule_A_axis;
	Vector3 capsule_A_ball_2 = p_transform_a.origin - capsule_A_axis;
	Vector3 capsule_B_ball_1 = p_transform_b.origin + capsule_B_axis;
	Vector3 capsule_B_ball_2 = p_transform_b.origin - capsule_B_axis;

	//balls-balls

	if (!separator.test_axis( (capsule_A_ball_1 - capsule_B_ball_1 ).normalized() ) )
		return;
	if (!separator.test_axis( (capsule_A_ball_1 - capsule_B_ball_2 ).normalized() ) )
		return;

	if (!separator.test_axis( (capsule_A_ball_2 - capsule_B_ball_1 ).normalized() ) )
		return;
	if (!separator.test_axis( (capsule_A_ball_2 - capsule_B_ball_2 ).normalized() ) )
		return;


	// edges-balls

	if (!separator.test_axis( (capsule_A_ball_1 - capsule_B_ball_1 ).cross(capsule_A_axis).cross(capsule_A_axis).normalized() ) )
		return;

	if (!separator.test_axis( (capsule_A_ball_1 - capsule_B_ball_2 ).cross(capsule_A_axis).cross(capsule_A_axis).normalized() ) )
		return;

	if (!separator.test_axis( (capsule_B_ball_1 - capsule_A_ball_1 ).cross(capsule_B_axis).cross(capsule_B_axis).normalized() ) )
		return;

	if (!separator.test_axis( (capsule_B_ball_1 - capsule_A_ball_2 ).cross(capsule_B_axis).cross(capsule_B_axis).normalized() ) )
		return;

	// edges

	if (!separator.test_axis( capsule_A_axis.cross(capsule_B_axis).normalized() ) )
		return;


	separator.generate_contacts();

}

static void _collision_capsule_convex_polygon(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {


	const CapsuleShapeSW *capsule_A = static_cast<const CapsuleShapeSW*>(p_a);
	const ConvexPolygonShapeSW *convex_polygon_B = static_cast<const ConvexPolygonShapeSW*>(p_b);

	SeparatorAxisTest<CapsuleShapeSW,ConvexPolygonShapeSW> separator(capsule_A,p_transform_a,convex_polygon_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	const Geometry::MeshData &mesh = convex_polygon_B->get_mesh();

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();

	// faces of B
	for (int i=0;i<face_count;i++) {

		Vector3 axis = p_transform_b.xform( faces[i].plane ).normal;

		if (!separator.test_axis( axis ))
			return;
	}

	// edges of B, capsule cylinder

	for (int i=0;i<edge_count;i++) {

		// cylinder
		Vector3 edge_axis = p_transform_b.basis.xform( vertices[ edges[i].a] ) - p_transform_b.basis.xform( vertices[ edges[i].b] );
		Vector3 axis = edge_axis.cross( p_transform_a.basis.get_axis(2) ).normalized();


		if (!separator.test_axis( axis ))
			return;
	}

	// capsule balls, edges of B

	for (int i=0;i<2;i++) {

		// edges of B, capsule cylinder

		Vector3 capsule_axis = p_transform_a.basis.get_axis(2)*(capsule_A->get_height()*0.5);

		Vector3 sphere_pos = p_transform_a.origin + ((i==0)?capsule_axis:-capsule_axis);

		for (int j=0;j<edge_count;j++) {


			Vector3 n1=sphere_pos - p_transform_b.xform( vertices[ edges[j].a] );
			Vector3 n2=p_transform_b.basis.xform( vertices[ edges[j].a] ) - p_transform_b.basis.xform( vertices[ edges[j].b] );

			Vector3 axis = n1.cross(n2).cross(n2).normalized();

			if (!separator.test_axis( axis ))
				return;
		}
	}


	separator.generate_contacts();

}

static void _collision_capsule_face(const ShapeSW *p_a,const Transform &p_transform_a, const ShapeSW *p_b,const Transform& p_transform_b, _CollectorCallback *p_collector) {

	const CapsuleShapeSW *capsule_A = static_cast<const CapsuleShapeSW*>(p_a);
	const FaceShapeSW *face_B = static_cast<const FaceShapeSW*>(p_b);

	SeparatorAxisTest<CapsuleShapeSW,FaceShapeSW> separator(capsule_A,p_transform_a,face_B,p_transform_b,p_collector);



	Vector3 vertex[3]={
		p_transform_b.xform( face_B->vertex[0] ),
		p_transform_b.xform( face_B->vertex[1] ),
		p_transform_b.xform( face_B->vertex[2] ),
	};

	if (!separator.test_axis( (vertex[0]-vertex[2]).cross(vertex[0]-vertex[1]).normalized() ))
		return;

	// edges of B, capsule cylinder

	Vector3 capsule_axis = p_transform_a.basis.get_axis(2)*(capsule_A->get_height()*0.5);

	for (int i=0;i<3;i++) {

		// edge-cylinder
		Vector3 edge_axis = vertex[i]-vertex[(i+1)%3];
		Vector3 axis = edge_axis.cross( capsule_axis ).normalized();

		if (!separator.test_axis( axis ))
			return;

		if (!separator.test_axis( (p_transform_a.origin-vertex[i]).cross(capsule_axis).cross(capsule_axis).normalized() ))
			return;

		for (int j=0;j<2;j++) {

			// point-spheres
			Vector3 sphere_pos = p_transform_a.origin + ( (j==0) ? capsule_axis : -capsule_axis );

			Vector3 n1=sphere_pos - vertex[i];

			if (!separator.test_axis( n1.normalized() ))
				return;

			Vector3 n2=edge_axis;

			axis = n1.cross(n2).cross(n2);

			if (!separator.test_axis( axis.normalized() ))
				return;


		}

	}


	separator.generate_contacts();

}


static void _collision_convex_polygon_convex_polygon(const ShapeSW *p_a,const Transform &p_transform_a,const ShapeSW *p_b,const Transform &p_transform_b,_CollectorCallback *p_collector) {


	const ConvexPolygonShapeSW *convex_polygon_A = static_cast<const ConvexPolygonShapeSW*>(p_a);
	const ConvexPolygonShapeSW *convex_polygon_B = static_cast<const ConvexPolygonShapeSW*>(p_b);

	SeparatorAxisTest<ConvexPolygonShapeSW,ConvexPolygonShapeSW> separator(convex_polygon_A,p_transform_a,convex_polygon_B,p_transform_b,p_collector);

	if (!separator.test_previous_axis())
		return;

	const Geometry::MeshData &mesh_A = convex_polygon_A->get_mesh();

	const Geometry::MeshData::Face *faces_A = mesh_A.faces.ptr();
	int face_count_A = mesh_A.faces.size();
	const Geometry::MeshData::Edge *edges_A = mesh_A.edges.ptr();
	int edge_count_A = mesh_A.edges.size();
	const Vector3 *vertices_A = mesh_A.vertices.ptr();
	int vertex_count_A = mesh_A.vertices.size();

	const Geometry::MeshData &mesh_B = convex_polygon_B->get_mesh();

	const Geometry::MeshData::Face *faces_B = mesh_B.faces.ptr();
	int face_count_B = mesh_B.faces.size();
	const Geometry::MeshData::Edge *edges_B = mesh_B.edges.ptr();
	int edge_count_B = mesh_B.edges.size();
	const Vector3 *vertices_B = mesh_B.vertices.ptr();
	int vertex_count_B = mesh_B.vertices.size();

	// faces of A
	for (int i=0;i<face_count_A;i++) {

		Vector3 axis = p_transform_a.xform( faces_A[i].plane ).normal;
//		Vector3 axis = p_transform_a.basis.xform( faces_A[i].plane.normal ).normalized();

		if (!separator.test_axis( axis ))
			return;
	}

	// faces of B
	for (int i=0;i<face_count_B;i++) {

		Vector3 axis = p_transform_b.xform( faces_B[i].plane ).normal;
//		Vector3 axis = p_transform_b.basis.xform( faces_B[i].plane.normal ).normalized();


		if (!separator.test_axis( axis ))
			return;
	}

	// A<->B edges
	for (int i=0;i<edge_count_A;i++) {

		Vector3 e1=p_transform_a.basis.xform( vertices_A[ edges_A[i].a] ) -p_transform_a.basis.xform( vertices_A[ edges_A[i].b] );

		for (int j=0;j<edge_count_B;j++) {

			Vector3 e2=p_transform_b.basis.xform( vertices_B[ edges_B[j].a] ) -p_transform_b.basis.xform( vertices_B[ edges_B[j].b] );

			Vector3 axis=e1.cross( e2 ).normalized();

			if (!separator.test_axis( axis ))
				return;

		}
	}

	separator.generate_contacts();

}

static void _collision_convex_polygon_face(const ShapeSW *p_a,const Transform &p_transform_a, const ShapeSW *p_b,const Transform& p_transform_b, _CollectorCallback *p_collector) {


	const ConvexPolygonShapeSW *convex_polygon_A = static_cast<const ConvexPolygonShapeSW*>(p_a);
	const FaceShapeSW *face_B = static_cast<const FaceShapeSW*>(p_b);

	SeparatorAxisTest<ConvexPolygonShapeSW,FaceShapeSW> separator(convex_polygon_A,p_transform_a,face_B,p_transform_b,p_collector);

	const Geometry::MeshData &mesh = convex_polygon_A->get_mesh();

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int face_count = mesh.faces.size();
	const Geometry::MeshData::Edge *edges = mesh.edges.ptr();
	int edge_count = mesh.edges.size();
	const Vector3 *vertices = mesh.vertices.ptr();
	int vertex_count = mesh.vertices.size();



	Vector3 vertex[3]={
		p_transform_b.xform( face_B->vertex[0] ),
		p_transform_b.xform( face_B->vertex[1] ),
		p_transform_b.xform( face_B->vertex[2] ),
	};

	if (!separator.test_axis( (vertex[0]-vertex[2]).cross(vertex[0]-vertex[1]).normalized() ))
		return;


	// faces of A
	for (int i=0;i<face_count;i++) {

//		Vector3 axis = p_transform_a.xform( faces[i].plane ).normal;
		Vector3 axis = p_transform_a.basis.xform( faces[i].plane.normal ).normalized();

		if (!separator.test_axis( axis ))
			return;
	}


	// A<->B edges
	for (int i=0;i<edge_count;i++) {

		Vector3 e1=p_transform_a.xform( vertices[edges[i].a] ) - p_transform_a.xform( vertices[edges[i].b] );

		for (int j=0;j<3;j++) {

			Vector3 e2=vertex[j]-vertex[(j+1)%3];

			Vector3 axis=e1.cross( e2 ).normalized();

			if (!separator.test_axis( axis ))
				return;
		}
	}

	separator.generate_contacts();

}


bool sat_calculate_penetration(const ShapeSW *p_shape_A, const Transform& p_transform_A, const ShapeSW *p_shape_B, const Transform& p_transform_B, CollisionSolverSW::CallbackResult p_result_callback,void *p_userdata,bool p_swap,Vector3* r_prev_axis) {

	PhysicsServer::ShapeType type_A=p_shape_A->get_type();

	ERR_FAIL_COND_V(type_A==PhysicsServer::SHAPE_PLANE,false);
	ERR_FAIL_COND_V(type_A==PhysicsServer::SHAPE_RAY,false);
	ERR_FAIL_COND_V(p_shape_A->is_concave(),false);

	PhysicsServer::ShapeType type_B=p_shape_B->get_type();

	ERR_FAIL_COND_V(type_B==PhysicsServer::SHAPE_PLANE,false);
	ERR_FAIL_COND_V(type_B==PhysicsServer::SHAPE_RAY,false);
	ERR_FAIL_COND_V(p_shape_B->is_concave(),false);


	static const CollisionFunc collision_table[5][5]={
		{_collision_sphere_sphere,
		 _collision_sphere_box,
		 _collision_sphere_capsule,
		 _collision_sphere_convex_polygon,
		 _collision_sphere_face},
		{0,
		 _collision_box_box,
		 _collision_box_capsule,
		 _collision_box_convex_polygon,
		 _collision_box_face},
		{0,
		 0,
		 _collision_capsule_capsule,
		 _collision_capsule_convex_polygon,
		 _collision_capsule_face},
		{0,
		 0,
		 0,
		 _collision_convex_polygon_convex_polygon,
		 _collision_convex_polygon_face},
		{0,
		 0,
		 0,
		 0,
		 0},
	};

	_CollectorCallback callback;
	callback.callback=p_result_callback;
	callback.swap=p_swap;
	callback.userdata=p_userdata;
	callback.collided=false;
	callback.prev_axis=r_prev_axis;

	const ShapeSW *A=p_shape_A;
	const ShapeSW *B=p_shape_B;
	const Transform *transform_A=&p_transform_A;
	const Transform *transform_B=&p_transform_B;

	if (type_A > type_B) {
		SWAP(A,B);
		SWAP(transform_A,transform_B);
		SWAP(type_A,type_B);
		callback.swap = !callback.swap;
	}


	CollisionFunc collision_func = collision_table[type_A-2][type_B-2];
	ERR_FAIL_COND_V(!collision_func,false);


	collision_func(A,*transform_A,B,*transform_B,&callback);

	return callback.collided;

}
