/*************************************************************************/
/*  volume.cpp                                                           */
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
#include "volume.h"

#if 0
void Volume::_set(const String& p_name, const Variant& p_value) {


	if (p_name.begins_with("shapes/")) {

		int idx=p_name.get_slice("/",1).to_int()-1;
		ERR_FAIL_COND( idx != get_shape_count() );

		Dictionary shape = p_value;
		ERR_FAIL_COND( !shape.has("type") || !shape.has("data"));
		String type = shape["type"];
		Variant data=shape["data"];
		Transform transform;
		if (shape.has("transform"))
			transform=shape["transform"];

		if (type=="plane")
			add_shape(SHAPE_PLANE,data,transform);
		else if (type=="sphere")
			add_shape(SHAPE_SPHERE,data,transform);
		else if (type=="box")
			add_shape(SHAPE_BOX,data,transform);
		else if (type=="cylinder")
			add_shape(SHAPE_CYLINDER,data,transform);
		else if (type=="capsule")
			add_shape(SHAPE_CAPSULE,data,transform);
		else if (type=="convex_polygon")
			add_shape(SHAPE_CONVEX_POLYGON,data,transform);
		else if (type=="concave_polygon")
			add_shape(SHAPE_CONCAVE_POLYGON,data,transform);
		else {
			ERR_FAIL();
		}
	}
}

Variant Volume::_get(const String& p_name) const {

	if (p_name.begins_with("shapes/")) {

		int idx=p_name.get_slice("/",1).to_int()-1;
		ERR_FAIL_INDEX_V( idx, get_shape_count(), Variant() );

		Dictionary shape;

		switch( get_shape_type(idx) ) {

			case SHAPE_PLANE: shape["type"]="plane"; break;
			case SHAPE_SPHERE: shape["type"]="sphere"; break;
			case SHAPE_BOX: shape["type"]="box"; break;
			case SHAPE_CYLINDER: shape["type"]="cylinder"; break;
			case SHAPE_CAPSULE: shape["type"]="capsule"; break;
			case SHAPE_CONVEX_POLYGON: shape["type"]="convex_polygon"; break;
			case SHAPE_CONCAVE_POLYGON: shape["type"]="concave_polygon"; break;

		}

		shape["transform"]=get_shape_transform(idx);
		shape["data"]=get_shape(idx);

		return shape;
	}

	return Variant();
}

void Volume::_get_property_list( List<PropertyInfo> *p_list) const {

	int count=get_shape_count();
	for(int i=0;i<count;i++) {

		p_list->push_back( PropertyInfo( Variant::DICTIONARY, "shapes/"+itos(i+1)) );
	}
}





void Volume::add_shape(ShapeType p_shape_type, const Variant& p_data, const Transform& p_transform) {

	PhysicsServer::get_singleton()->volume_add_shape(volume,(PhysicsServer::ShapeType)p_shape_type,p_data,p_transform);
	_change_notify();
}


void Volume::add_plane_shape(const Plane& p_plane,const Transform& p_transform) {

	add_shape(SHAPE_PLANE, p_plane, p_transform );
}

void Volume::add_sphere_shape(float p_radius,const Transform& p_transform) {

	add_shape(SHAPE_SPHERE, p_radius, p_transform );
}

void Volume::add_box_shape(const Vector3& p_half_extents,const Transform& p_transform) {

	add_shape(SHAPE_BOX, p_half_extents, p_transform );
}
void Volume::add_cylinder_shape(float p_radius, float p_height,const Transform& p_transform) {

	Dictionary d;
	d["radius"]=p_radius;
	d["height"]=p_height;

	add_shape(SHAPE_CYLINDER,d,p_transform);
}
void Volume::add_capsule_shape(float p_radius, float p_height,const Transform& p_transform) {

	Dictionary d;
	d["radius"]=p_radius;
	d["height"]=p_height;

	add_shape(SHAPE_CAPSULE,d,p_transform);
}


int Volume::get_shape_count() const {

	return PhysicsServer::get_singleton()->volume_get_shape_count(volume);
}

Volume::ShapeType Volume::get_shape_type(int p_shape) const {

	return (ShapeType)PhysicsServer::get_singleton()->volume_get_shape_type(volume,p_shape);
}

Transform Volume::get_shape_transform(int p_shape) const {

	return PhysicsServer::get_singleton()->volume_get_shape_transform(volume,p_shape);
}

Variant Volume::get_shape(int p_shape) const {

	return PhysicsServer::get_singleton()->volume_get_shape(volume,p_shape);
}

void Volume::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("add_shape","type","data","transform"),&Volume::add_shape,DEFVAL( Transform() ));
	ObjectTypeDB::bind_method(_MD("add_plane_shape","plane","transform"),&Volume::add_plane_shape,DEFVAL( Transform() ));
	ObjectTypeDB::bind_method(_MD("add_sphere_shape"),&Volume::add_sphere_shape,DEFVAL( Transform() ));
	ObjectTypeDB::bind_method(_MD("add_box_shape","radius","transform"),&Volume::add_box_shape,DEFVAL( Transform() ));
	ObjectTypeDB::bind_method(_MD("add_cylinder_shape","radius","height","transform"),&Volume::add_cylinder_shape,DEFVAL( Transform() ));
	ObjectTypeDB::bind_method(_MD("add_capsule_shape","radius","height","transform"),&Volume::add_capsule_shape,DEFVAL( Transform() ));
	ObjectTypeDB::bind_method(_MD("get_shape_count"),&Volume::get_shape_count);
	ObjectTypeDB::bind_method(_MD("get_shape_type","shape_idx"),&Volume::get_shape_type);
	ObjectTypeDB::bind_method(_MD("get_shape_transform","shape_idx"),&Volume::get_shape_transform);
	ObjectTypeDB::bind_method(_MD("get_shape","shape_idx"),&Volume::get_shape);

	BIND_CONSTANT( SHAPE_PLANE );
	BIND_CONSTANT( SHAPE_SPHERE );
	BIND_CONSTANT( SHAPE_BOX );
	BIND_CONSTANT( SHAPE_CYLINDER );
	BIND_CONSTANT( SHAPE_CAPSULE );
	BIND_CONSTANT( SHAPE_CONVEX_POLYGON );
	BIND_CONSTANT( SHAPE_CONCAVE_POLYGON );

}

RID Volume::get_rid() {

	return volume;
}

Volume::Volume() {

	volume= PhysicsServer::get_singleton()->volume_create();

}


Volume::~Volume() {

	PhysicsServer::get_singleton()->free(volume);
}


#endif
