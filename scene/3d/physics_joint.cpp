/*************************************************************************/
/*  physics_joint.cpp                                                    */
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
#include "physics_joint.h"
 
#if 0

void PhysicsJoint::_set(const String& p_name, const Variant& p_value) {

	if (p_name=="body_A")
		set_body_A(p_value);
	else if (p_name=="body_B")
		set_body_B(p_value);
	else if (p_name=="active")
		set_active(p_value);
	else if (p_name=="no_collision")
		set_disable_collision(p_value);
}
Variant PhysicsJoint::_get(const String& p_name) const {

	if (p_name=="body_A")
		return get_body_A();
	else if (p_name=="body_B")
		return get_body_B();
	else if (p_name=="active")
		return is_active();
	else if (p_name=="no_collision")
		return has_disable_collision();

	return Variant();
}
void PhysicsJoint::_get_property_list( List<PropertyInfo> *p_list) const {


	p_list->push_back( PropertyInfo( Variant::NODE_PATH, "body_A" ) );
	p_list->push_back( PropertyInfo( Variant::NODE_PATH, "body_B" ) );
	p_list->push_back( PropertyInfo( Variant::BOOL, "active" ) );
	p_list->push_back( PropertyInfo( Variant::BOOL, "no_collision" ) );
}
void PhysicsJoint::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_PARENT_CONFIGURED: {

			_connect();
			if (get_root_node()->get_editor() && !indicator.is_valid()) {

				indicator=VisualServer::get_singleton()->poly_create();
				RID mat=VisualServer::get_singleton()->fixed_material_create();
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_UNSHADED, true );
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_ONTOP, true );
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_WIREFRAME, true );
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_DOUBLE_SIDED, true );
				VisualServer::get_singleton()->material_set_line_width( mat, 3 );

				VisualServer::get_singleton()->poly_set_material(indicator,mat,true);
				_update_indicator();

			}

			if (indicator.is_valid()) {

				indicator_instance=VisualServer::get_singleton()->instance_create(indicator,get_world()->get_scenario());
				VisualServer::get_singleton()->instance_attach_object_instance_ID( indicator_instance,get_instance_ID() );
			}
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (indicator_instance.is_valid()) {

				VisualServer::get_singleton()->instance_set_transform(indicator_instance,get_global_transform());
			}
		} break;
		case NOTIFICATION_EXIT_SCENE: {

			if (indicator_instance.is_valid()) {

				VisualServer::get_singleton()->free(indicator_instance);
			}
			_disconnect();

		} break;

	}
}


RID PhysicsJoint::_get_visual_instance_rid() const {

	return indicator_instance;

}

void PhysicsJoint::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_get_visual_instance_rid"),&PhysicsJoint::_get_visual_instance_rid);
	ObjectTypeDB::bind_method(_MD("set_body_A","path"),&PhysicsJoint::set_body_A);
	ObjectTypeDB::bind_method(_MD("set_body_B"),&PhysicsJoint::set_body_B);
	ObjectTypeDB::bind_method(_MD("get_body_A","path"),&PhysicsJoint::get_body_A);
	ObjectTypeDB::bind_method(_MD("get_body_B"),&PhysicsJoint::get_body_B);

	ObjectTypeDB::bind_method(_MD("set_active","active"),&PhysicsJoint::set_active);
	ObjectTypeDB::bind_method(_MD("is_active"),&PhysicsJoint::is_active);

	ObjectTypeDB::bind_method(_MD("set_disable_collision","disable"),&PhysicsJoint::set_disable_collision);
	ObjectTypeDB::bind_method(_MD("has_disable_collision"),&PhysicsJoint::has_disable_collision);


	ObjectTypeDB::bind_method("reconnect",&PhysicsJoint::reconnect);

	ObjectTypeDB::bind_method(_MD("get_rid"),&PhysicsJoint::get_rid);

}

void PhysicsJoint::set_body_A(const NodePath& p_path) {

	_disconnect();
	body_A=p_path;
	_connect();
	_change_notify("body_A");
}
void PhysicsJoint::set_body_B(const NodePath& p_path) {

	_disconnect();
	body_B=p_path;
	_connect();
	_change_notify("body_B");

}
NodePath PhysicsJoint::get_body_A() const {

	return body_A;
}
NodePath PhysicsJoint::get_body_B() const {

	return body_B;
}

void PhysicsJoint::set_active(bool p_active) {

	active=p_active;
	if (is_inside_scene()) {
		PhysicsServer::get_singleton()->joint_set_active(joint,active);
	}
	_change_notify("active");
}

void PhysicsJoint::set_disable_collision(bool p_active) {

	if (no_collision==p_active)
		return;
	_disconnect();
	no_collision=p_active;
	_connect();

	_change_notify("no_collision");
}
bool PhysicsJoint::has_disable_collision() const {

	return no_collision;
}



bool PhysicsJoint::is_active() const {

	return active;
}

void PhysicsJoint::_disconnect() {

	if (!is_inside_scene())
		return;

	if (joint.is_valid())
		PhysicsServer::get_singleton()->free(joint);

	joint=RID();

	Node *nA = get_node(body_A);
	Node *nB = get_node(body_B);

	PhysicsBody *A = nA?nA->cast_to<PhysicsBody>():NULL;
	PhysicsBody *B = nA?nB->cast_to<PhysicsBody>():NULL;

	if (!A ||!B)
		return;

	if (no_collision)
		PhysicsServer::get_singleton()->body_remove_collision_exception(A->get_body(),B->get_body());

}
void PhysicsJoint::_connect() {

	if (!is_inside_scene())
		return;

	ERR_FAIL_COND(joint.is_valid());

	Node *nA = get_node(body_A);
	Node *nB = get_node(body_B);

	PhysicsBody *A = nA?nA->cast_to<PhysicsBody>():NULL;
	PhysicsBody *B = nA?nB->cast_to<PhysicsBody>():NULL;

	if (!A && !B)
		return;

	if (B && !A)
		SWAP(B,A);

	joint = create(A,B);

	if (A<B)
		SWAP(A,B);

	if (no_collision)
		PhysicsServer::get_singleton()->body_add_collision_exception(A->get_body(),B->get_body());



}

void PhysicsJoint::reconnect() {

	_disconnect();
	_connect();

}


RID PhysicsJoint::get_rid() {

	return joint;
}


PhysicsJoint::PhysicsJoint() {

	active=true;
	no_collision=true;
}


PhysicsJoint::~PhysicsJoint() {

	if (indicator.is_valid()) {

		VisualServer::get_singleton()->free(indicator);
	}

}

/* PIN */

void PhysicsJointPin::_update_indicator() {


	VisualServer::get_singleton()->poly_clear(indicator);

	Vector<Color> colors;
	colors.push_back( Color(0.3,0.9,0.2,0.7) );
	colors.push_back( Color(0.3,0.9,0.2,0.7) );

	Vector<Vector3> points;
	points.resize(2);
	points[0]=Vector3(Vector3(-0.2,0,0));
	points[1]=Vector3(Vector3(0.2,0,0));
	VisualServer::get_singleton()->poly_add_primitive(indicator,points,Vector<Vector3>(),colors,Vector<Vector3>());

	points[0]=Vector3(Vector3(0,-0.2,0));
	points[1]=Vector3(Vector3(0,0.2,0));
	VisualServer::get_singleton()->poly_add_primitive(indicator,points,Vector<Vector3>(),colors,Vector<Vector3>());

	points[0]=Vector3(Vector3(0,0,-0.2));
	points[1]=Vector3(Vector3(0,0,0.2));
	VisualServer::get_singleton()->poly_add_primitive(indicator,points,Vector<Vector3>(),colors,Vector<Vector3>());

}

RID PhysicsJointPin::create(PhysicsBody*A,PhysicsBody*B) {

	RID body_A = A->get_body();
	RID body_B = B?B->get_body():RID();

	ERR_FAIL_COND_V( !body_A.is_valid(), RID() );

	Vector3 pin_pos = get_global_transform().origin;

	if (body_B.is_valid())
		return PhysicsServer::get_singleton()->joint_create_double_pin_global(body_A,pin_pos,body_B,pin_pos);
	else
		return PhysicsServer::get_singleton()->joint_create_pin(body_A,A->get_global_transform().xform_inv(pin_pos),pin_pos);
}

PhysicsJointPin::PhysicsJointPin() {


}
#endif
