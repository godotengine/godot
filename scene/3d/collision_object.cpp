/*************************************************************************/
/*  collision_object.cpp                                                 */
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
#include "collision_object.h"
#include "servers/physics_server.h"

void CollisionObject::_update_shapes_from_children() {

	shapes.resize(0);
	for(int i=0;i<get_child_count();i++) {

		Node* n = get_child(i);
		n->call("_add_to_collision_object",this);
	}

//	_update_shapes();
}

void CollisionObject::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			RID space = get_world()->get_space();
			if (area) {
				PhysicsServer::get_singleton()->area_set_space(rid,space);
			} else
				PhysicsServer::get_singleton()->body_set_space(rid,space);

		//get space
		}

		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (area)
				PhysicsServer::get_singleton()->area_set_transform(rid,get_global_transform());
			else
				PhysicsServer::get_singleton()->body_set_state(rid,PhysicsServer::BODY_STATE_TRANSFORM,get_global_transform());

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			if (area) {
				PhysicsServer::get_singleton()->area_set_space(rid,RID());
			} else
				PhysicsServer::get_singleton()->body_set_space(rid,RID());

		} break;
	}
}

void CollisionObject::_update_shapes() {

	if (!rid.is_valid())
		return;

	if (area)
		PhysicsServer::get_singleton()->area_clear_shapes(rid);
	else
		PhysicsServer::get_singleton()->body_clear_shapes(rid);

	for(int i=0;i<shapes.size();i++) {

		if (shapes[i].shape.is_null())
			continue;
		if (area)
			PhysicsServer::get_singleton()->area_add_shape(rid,shapes[i].shape->get_rid(),shapes[i].xform);
        else {
			PhysicsServer::get_singleton()->body_add_shape(rid,shapes[i].shape->get_rid(),shapes[i].xform);
            if (shapes[i].trigger)
                PhysicsServer::get_singleton()->body_set_shape_as_trigger(rid,i,shapes[i].trigger);
        }
	}
}


bool CollisionObject::_set(const StringName& p_name, const Variant& p_value) {
	String name=p_name;

	if (name=="shape_count") {

		shapes.resize(p_value);
		_update_shapes();
		_change_notify();

	} else if (name.begins_with("shapes/")) {

		int idx=name.get_slice("/",1).to_int();
		String what=name.get_slice("/",2);
		if (what=="shape")
			set_shape(idx,RefPtr(p_value));
		else if (what=="transform")
			set_shape_transform(idx,p_value);
        else if (what=="trigger")
            set_shape_as_trigger(idx,p_value);


	} else
		return false;

	return true;


}

bool CollisionObject::_get(const StringName& p_name,Variant &r_ret) const {

	String name=p_name;

	if (name=="shape_count") {
		r_ret= shapes.size();
	} else if (name.begins_with("shapes/")) {

		int idx=name.get_slice("/",1).to_int();
		String what=name.get_slice("/",2);
		if (what=="shape")
			r_ret= get_shape(idx);
		else if (what=="transform")
			r_ret= get_shape_transform(idx);
        else if (what=="trigger")
            r_ret= is_shape_set_as_trigger(idx);

	} else
		return false;

	return true;
}

void CollisionObject::_get_property_list( List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo(Variant::INT,"shape_count",PROPERTY_HINT_RANGE,"0,256,1",PROPERTY_USAGE_NOEDITOR) );

	for(int i=0;i<shapes.size();i++) {
		String path="shapes/"+itos(i)+"/";
		p_list->push_back( PropertyInfo(Variant::OBJECT,path+"shape",PROPERTY_HINT_RESOURCE_TYPE,"Shape",PROPERTY_USAGE_NOEDITOR) );
		p_list->push_back( PropertyInfo(Variant::TRANSFORM,path+"transform",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR) );
        p_list->push_back( PropertyInfo(Variant::BOOL,path+"trigger",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR) );

	}
}

void CollisionObject::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("add_shape","shape:Shape","transform"),&CollisionObject::add_shape,DEFVAL(Transform()));
	ObjectTypeDB::bind_method(_MD("get_shape_count"),&CollisionObject::get_shape_count);
	ObjectTypeDB::bind_method(_MD("set_shape","shape_idx","shape:Shape"),&CollisionObject::set_shape);
	ObjectTypeDB::bind_method(_MD("set_shape_transform","shape_idx","transform"),&CollisionObject::set_shape_transform);
//    ObjectTypeDB::bind_method(_MD("set_shape_transform","shape_idx","transform"),&CollisionObject::set_shape_transform);
	ObjectTypeDB::bind_method(_MD("set_shape_as_trigger","shape_idx","enable"),&CollisionObject::set_shape_as_trigger);
	ObjectTypeDB::bind_method(_MD("is_shape_set_as_trigger","shape_idx"),&CollisionObject::is_shape_set_as_trigger);
	ObjectTypeDB::bind_method(_MD("get_shape:Shape","shape_idx"),&CollisionObject::get_shape);
	ObjectTypeDB::bind_method(_MD("get_shape_transform","shape_idx"),&CollisionObject::get_shape_transform);
	ObjectTypeDB::bind_method(_MD("remove_shape","shape_idx"),&CollisionObject::remove_shape);
	ObjectTypeDB::bind_method(_MD("clear_shapes"),&CollisionObject::clear_shapes);
	ObjectTypeDB::bind_method(_MD("get_rid"),&CollisionObject::get_rid);

}


void CollisionObject::add_shape(const Ref<Shape>& p_shape, const Transform& p_transform) {

	ShapeData sdata;
	sdata.shape=p_shape;
	sdata.xform=p_transform;
	shapes.push_back(sdata);
	_update_shapes();

}
int CollisionObject::get_shape_count() const {

	return shapes.size();

}
void CollisionObject::set_shape(int p_shape_idx, const Ref<Shape>& p_shape) {

	ERR_FAIL_INDEX(p_shape_idx,shapes.size());
	shapes[p_shape_idx].shape=p_shape;
	_update_shapes();
}

void CollisionObject::set_shape_transform(int p_shape_idx, const Transform& p_transform) {

	ERR_FAIL_INDEX(p_shape_idx,shapes.size());
	shapes[p_shape_idx].xform=p_transform;

	_update_shapes();
}

Ref<Shape> CollisionObject::get_shape(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx,shapes.size(),Ref<Shape>());
	return shapes[p_shape_idx].shape;

}
Transform CollisionObject::get_shape_transform(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx,shapes.size(),Transform());
	return shapes[p_shape_idx].xform;

}
void CollisionObject::remove_shape(int p_shape_idx) {

	ERR_FAIL_INDEX(p_shape_idx,shapes.size());
	shapes.remove(p_shape_idx);

	_update_shapes();
}

void CollisionObject::clear_shapes() {

	shapes.clear();

	_update_shapes();
}

void CollisionObject::set_shape_as_trigger(int p_shape_idx, bool p_trigger) {

    ERR_FAIL_INDEX(p_shape_idx,shapes.size());
    shapes[p_shape_idx].trigger=p_trigger;
    if (!area && rid.is_valid()) {

        PhysicsServer::get_singleton()->body_set_shape_as_trigger(rid,p_shape_idx,p_trigger);

    }
}

bool CollisionObject::is_shape_set_as_trigger(int p_shape_idx) const {

    ERR_FAIL_INDEX_V(p_shape_idx,shapes.size(),false);
    return shapes[p_shape_idx].trigger;
}

CollisionObject::CollisionObject(RID p_rid, bool p_area) {

	rid=p_rid;
	area=p_area;
	if (p_area) {
		PhysicsServer::get_singleton()->area_attach_object_instance_ID(rid,get_instance_ID());
	} else {
		PhysicsServer::get_singleton()->body_attach_object_instance_ID(rid,get_instance_ID());
	}
//	set_transform_notify(true);

}


CollisionObject::CollisionObject() {


	//owner=

	//set_transform_notify(true);
}

CollisionObject::~CollisionObject() {

	PhysicsServer::get_singleton()->free(rid);
}
