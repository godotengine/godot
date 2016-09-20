/*************************************************************************/
/*  test_misc.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "test_misc.h"
#include "servers/visual_server.h"
#include "os/main_loop.h"
#include "math_funcs.h"
#include "print_string.h"


namespace TestMisc {

struct ConvexTestResult
{

	Vector3 edgeA[2];
	Vector3 edgeB[2];
	bool valid;
	Vector3 contactA;
	Vector3 contactB;
	Vector3 contactNormal;
	float depth;

	/*
	Vector3 contactA;
	Vector3 contactB;
	Vector3 contactNormal;
	Vector3 contactX;
	Vector3 contactY;
	Vector3 edgeA[2];
	Vector3 edgeB[2];
	float depth;
	bool valid;
	bool isEdgeEdge;
	bool needTransform;
	neBool ComputerEdgeContactPoint(ConvexTestResult & res);
	neBool ComputerEdgeContactPoint2(float & au, float & bu);
	void Reverse()
	{
		neSwap(contactA, contactB);
		contactNormal *= -1.0f;
	}*/
	bool ComputerEdgeContactPoint2(float & au, float & bu);
};



bool ConvexTestResult::ComputerEdgeContactPoint2(float & au, float & bu)
{
	float d1343, d4321, d1321, d4343, d2121;
	float numer, denom;

	Vector3 p13;
	Vector3 p43;
	Vector3 p21;
	Vector3 diff;

	p13 = (edgeA[0]) - (edgeB[0]);
	p43 = (edgeB[1]) - (edgeB[0]);

	if ( p43.length_squared() < CMP_EPSILON2 )
	{
		valid = false;
		goto ComputerEdgeContactPoint2_Exit;
	}

	p21 = (edgeA[1]) - (edgeA[0]);

	if ( p21.length_squared()<CMP_EPSILON2 )
	{
		valid = false;
		goto ComputerEdgeContactPoint2_Exit;
	}

	d1343 = p13.dot(p43);
	d4321 = p43.dot(p21);
	d1321 = p13.dot(p21);
	d4343 = p43.dot(p43);
	d2121 = p21.dot(p21);

	denom = d2121 * d4343 - d4321 * d4321;

	if (ABS(denom) < CMP_EPSILON)
	{
		valid = false;

		goto ComputerEdgeContactPoint2_Exit;
	}

	numer = d1343 * d4321 - d1321 * d4343;
	au = numer / denom;
	bu = (d1343 + d4321 * (au)) / d4343;

	if (au < 0.0f || au >= 1.0f)
	{
		valid = false;
	}
	else if (bu < 0.0f || bu >= 1.0f)
	{
		valid = false;
	}
	else
	{
		valid = true;
	}
	{
		Vector3 tmpv;

		tmpv = p21 * au;
		contactA = (edgeA[0]) + tmpv;

		tmpv = p43 * bu;
		contactB = (edgeB[0]) + tmpv;
	}

	diff = contactA - contactB;

	depth = Math::sqrt(diff.dot(diff));

	return true;

ComputerEdgeContactPoint2_Exit:

	return false;
}

struct neCollisionResult {

	float depth;
	bool penetrate;
	Matrix3 collisionFrame;
	Vector3 contactA;
	Vector3 contactB;
};


struct TConvex {

	float radius;
	float half_height;
	float CylinderRadius() const { return radius; }
	float CylinderHalfHeight() const { return half_height; }
};

float GetDistanceFromLine2(Vector3 v, Vector3 & project, const Vector3 & pointA, const Vector3 & pointB)
{
	Vector3 ba = pointB - pointA;

	float len = ba.length();

	if (len<CMP_EPSILON)
		ba=Vector3();
	else
		ba *= 1.0f / len;

	Vector3 pa = v - pointA;

	float k = pa.dot(ba);

	project = pointA + ba * k;

	Vector3 diff = v - project;

	return diff.length();
}

void TestCylinderVertEdge(neCollisionResult & result, Vector3 & edgeA1, Vector3 & edgeA2, Vector3 & vertB,
						  TConvex & cA, TConvex & cB, Transform & transA, Transform & transB, bool flip)
{
	Vector3 project;

	float dist = GetDistanceFromLine2(vertB,project, edgeA1, edgeA2);

	float depth = cA.CylinderRadius() + cB.CylinderRadius() - dist;

	if (depth <= 0.0f)
		return;

	if (depth <= result.depth)
		return;

	result.penetrate = true;

	result.depth = depth;

	if (!flip)
	{
		result.collisionFrame.set_axis(2,(project - vertB).normalized());

		result.contactA = project - result.collisionFrame.get_axis(2) * cA.CylinderRadius();

		result.contactB = vertB + result.collisionFrame.get_axis(2) * cB.CylinderRadius();
	}
	else
	{

		result.collisionFrame.set_axis(2,(vertB - project).normalized());

		result.contactA = vertB - result.collisionFrame.get_axis(2) * cB.CylinderRadius();

		result.contactB = project + result.collisionFrame.get_axis(2) * cA.CylinderRadius();
	}
}

void TestCylinderVertVert(neCollisionResult & result, Vector3 & vertA, Vector3 & vertB,
						  TConvex & cA, TConvex & cB, Transform & transA, Transform & transB)
{
	Vector3 diff = vertA - vertB;

	float dist = diff.length();

	float depth = cA.CylinderRadius() + cB.CylinderRadius() - dist;

	if (depth <= 0.0f)
		return;

	if (depth <= result.depth)
		return;

	result.penetrate = true;

	result.depth = depth;

	result.collisionFrame.set_axis(2, diff * (1.0f / dist));

	result.contactA = vertA - result.collisionFrame.get_axis(2) * cA.CylinderRadius();

	result.contactB = vertB + result.collisionFrame.get_axis(2) * cB.CylinderRadius();
}

void Cylinder2CylinderTest(neCollisionResult & result, TConvex & cA, Transform & transA, TConvex & cB, Transform & transB)
{
	result.penetrate = false;

	Vector3 dir = transA.basis.get_axis(1).cross(transB.basis.get_axis(1));

	float len = dir.length();

//	bool isParallel = len<CMP_EPSILON;

//	int doVertCheck = 0;

	ConvexTestResult cr;

	cr.edgeA[0] = transA.origin + transA.basis.get_axis(1) * cA.CylinderHalfHeight();
	cr.edgeA[1] = transA.origin - transA.basis.get_axis(1) * cA.CylinderHalfHeight();
	cr.edgeB[0] = transB.origin + transB.basis.get_axis(1) * cB.CylinderHalfHeight();
	cr.edgeB[1] = transB.origin - transB.basis.get_axis(1) * cB.CylinderHalfHeight();

//	float dot = transA.basis.get_axis(1).dot(transB.basis.get_axis(1));

	if (len>CMP_EPSILON)
	{
		float au, bu;

		cr.ComputerEdgeContactPoint2(au, bu);

		if (cr.valid)
		{
			float depth = cA.CylinderRadius() + cB.CylinderRadius() - cr.depth;

			if (depth <= 0.0f)
				return;

			result.depth = depth;

			result.penetrate = true;

			result.collisionFrame.set_axis(2, (cr.contactA - cr.contactB)*(1.0f / cr.depth));

			result.contactA = cr.contactA - result.collisionFrame.get_axis(2) * cA.CylinderRadius();

			result.contactB = cr.contactB + result.collisionFrame.get_axis(2) * cB.CylinderRadius();

			return;
		}
	}
	result.depth = -1.0e6f;

	int i;

	for (i = 0; i < 2; i++)
	{
		//project onto edge b

		Vector3 diff = cr.edgeA[i] - cr.edgeB[1];

		float dot = diff.dot(transB.basis.get_axis(1));

		if (dot < 0.0f)
		{
			TestCylinderVertVert(result, cr.edgeA[i], cr.edgeB[1], cA, cB, transA, transB);
		}
		else if (dot > (2.0f * cB.CylinderHalfHeight()))
		{
			TestCylinderVertVert(result, cr.edgeA[i], cr.edgeB[0], cA, cB, transA, transB);
		}
		else
		{
			TestCylinderVertEdge(result, cr.edgeB[0], cr.edgeB[1], cr.edgeA[i], cB, cA, transB, transA, true);
		}
	}
	for (i = 0; i < 2; i++)
	{
		//project onto edge b

		Vector3 diff = cr.edgeB[i] - cr.edgeA[1];

		float dot = diff.dot(transA.basis.get_axis(1));

		if (dot < 0.0f)
		{
			TestCylinderVertVert(result, cr.edgeB[i], cr.edgeA[1], cA, cB, transA, transB);
		}
		else if (dot > (2.0f * cB.CylinderHalfHeight()))
		{
			TestCylinderVertVert(result, cr.edgeB[i], cr.edgeA[0], cA, cB, transA, transB);
		}
		else
		{
			TestCylinderVertEdge(result, cr.edgeA[0], cr.edgeA[1], cr.edgeB[i], cA, cB, transA, transB, false);
		}
	}
}


class TestMainLoop : public MainLoop {

	RID meshA;
	RID meshB;
	RID poly;
	RID instance;
	RID camera;
	RID viewport;
	RID boxA;
	RID boxB;
	RID scenario;

	Transform rot_a;
	Transform rot_b;

	bool quit;
public:
	virtual void input_event(const InputEvent& p_event) {

		if (p_event.type==InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask&BUTTON_MASK_LEFT) {

			rot_b.origin.y+=-p_event.mouse_motion.relative_y/100.0;
			rot_b.origin.x+=p_event.mouse_motion.relative_x/100.0;
		}
		if (p_event.type==InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask&BUTTON_MASK_MIDDLE) {

			//rot_b.origin.x+=-p_event.mouse_motion.relative_y/100.0;
			rot_b.origin.z+=p_event.mouse_motion.relative_x/100.0;
		}
		if (p_event.type==InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask&BUTTON_MASK_RIGHT) {

			float rot_x=-p_event.mouse_motion.relative_y/100.0;
			float rot_y=p_event.mouse_motion.relative_x/100.0;
			rot_b.basis = rot_b.basis * Matrix3(Vector3(1,0,0),rot_x) * Matrix3(Vector3(0,1,0),rot_y);
		}

	}
	virtual void request_quit() {

		quit=true;
	}
	virtual void init() {

		VisualServer *vs=VisualServer::get_singleton();

		camera = vs->camera_create();

		viewport = vs->viewport_create();
		vs->viewport_attach_to_screen(viewport);
		vs->viewport_attach_camera( viewport, camera );
		vs->camera_set_transform(camera, Transform( Matrix3(), Vector3(0,0,3 ) ) );

		/* CONVEX SHAPE */

		DVector<Plane> cylinder_planes = Geometry::build_cylinder_planes(0.5,2,9,Vector3::AXIS_Y);
		RID cylinder_material = vs->fixed_material_create();
		vs->fixed_material_set_param( cylinder_material, VisualServer::FIXED_MATERIAL_PARAM_DIFFUSE, Color(0.8,0.2,0.9));
		vs->material_set_flag( cylinder_material, VisualServer::MATERIAL_FLAG_ONTOP,true);
		//vs->material_set_flag( cylinder_material, VisualServer::MATERIAL_FLAG_WIREFRAME,true);
		vs->material_set_flag( cylinder_material, VisualServer::MATERIAL_FLAG_DOUBLE_SIDED,true);
		vs->material_set_flag( cylinder_material, VisualServer::MATERIAL_FLAG_UNSHADED,true);

		RID cylinder_mesh = vs->mesh_create();
		Geometry::MeshData cylinder_data = Geometry::build_convex_mesh(cylinder_planes);
		vs->mesh_add_surface_from_mesh_data(cylinder_mesh,cylinder_data);
		vs->mesh_surface_set_material( cylinder_mesh, 0, cylinder_material );

		meshA=vs->instance_create2(cylinder_mesh,scenario);
		meshB=vs->instance_create2(cylinder_mesh,scenario);
		boxA=vs->instance_create2(vs->get_test_cube(),scenario);
		boxB=vs->instance_create2(vs->get_test_cube(),scenario);

		/*
		RID lightaux = vs->light_create( VisualServer::LIGHT_OMNI );
		vs->light_set_var( lightaux, VisualServer::LIGHT_VAR_RADIUS, 80 );
		vs->light_set_var( lightaux, VisualServer::LIGHT_VAR_ATTENUATION, 1 );
		vs->light_set_var( lightaux, VisualServer::LIGHT_VAR_ENERGY, 1.5 );
		light = vs->instance_create2( lightaux );
		*/
		RID lightaux = vs->light_create( VisualServer::LIGHT_DIRECTIONAL );
		//vs->light_set_color( lightaux, VisualServer::LIGHT_COLOR_AMBIENT, Color(0.0,0.0,0.0) );
		//vs->light_set_shadow( lightaux, true );
		vs->instance_create2( lightaux,scenario );

		//rot_a=Transform(Matrix3(Vector3(1,0,0),Math_PI/2.0),Vector3());
		rot_b=Transform(Matrix3(),Vector3(2,0,0));

		//rot_x=0;
		//rot_y=0;
		quit=false;
	}
	virtual bool idle(float p_time) {

		VisualServer *vs=VisualServer::get_singleton();

		vs->instance_set_transform(meshA,rot_a);
		vs->instance_set_transform(meshB,rot_b);


		neCollisionResult res;
		TConvex a;
		a.radius=0.5;
		a.half_height=1;
		Cylinder2CylinderTest(res,a,rot_a,a,rot_b);
		if (res.penetrate) {

			Matrix3 scale;
			scale.scale(Vector3(0.1,0.1,0.1));
			vs->instance_set_transform(boxA,Transform(scale,res.contactA));
			vs->instance_set_transform(boxB,Transform(scale,res.contactB));
			print_line("depth: "+rtos(res.depth));
		} else  {

			Matrix3 scale;
			scale.scale(Vector3());
			vs->instance_set_transform(boxA,Transform(scale,res.contactA));
			vs->instance_set_transform(boxB,Transform(scale,res.contactB));

		}
		print_line("collided: "+itos(res.penetrate));

		return false;
	}


	virtual bool iteration(float p_time) {



		return quit;
	}
	virtual void finish() {

	}

};


MainLoop* test() {

	return memnew( TestMainLoop );

}

}



