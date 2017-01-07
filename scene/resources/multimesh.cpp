/*************************************************************************/
/*  multimesh.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "multimesh.h"
#include "servers/visual_server.h"



void MultiMesh::_set_transform_array(const PoolVector<Vector3>& p_array) {

	int instance_count = get_instance_count();

	PoolVector<Vector3> xforms = p_array;
	int len=xforms.size();
	ERR_FAIL_COND((len/4) != instance_count);
	if (len==0)
		return;

	PoolVector<Vector3>::Read r = xforms.read();

	for(int i=0;i<len/4;i++) {

		Transform t;
		t.basis[0]=r[i*4+0];
		t.basis[1]=r[i*4+1];
		t.basis[2]=r[i*4+2];
		t.origin=r[i*4+3];

		set_instance_transform(i,t);
	}

}

PoolVector<Vector3> MultiMesh::_get_transform_array() const {

	int instance_count = get_instance_count();

	if (instance_count==0)
		return PoolVector<Vector3>();

	PoolVector<Vector3> xforms;
	xforms.resize(instance_count*4);

	PoolVector<Vector3>::Write w = xforms.write();

	for(int i=0;i<instance_count;i++) {

		Transform t=get_instance_transform(i);
		w[i*4+0]=t.basis[0];
		w[i*4+1]=t.basis[1];
		w[i*4+2]=t.basis[2];
		w[i*4+3]=t.origin;
	}

	return xforms;

}


void MultiMesh::_set_color_array(const PoolVector<Color>& p_array) {

	int instance_count = get_instance_count();

	PoolVector<Color> colors = p_array;
	int len=colors.size();
	ERR_FAIL_COND(len != instance_count);
	if (len==0)
		return;

	PoolVector<Color>::Read r = colors.read();

	for(int i=0;i<len;i++) {

		set_instance_color(i,r[i]);
	}

}

PoolVector<Color> MultiMesh::_get_color_array() const {

	int instance_count = get_instance_count();

	if (instance_count==0)
		return PoolVector<Color>();

	PoolVector<Color> colors;
	colors.resize(instance_count);

	for(int i=0;i<instance_count;i++) {

		colors.set(i,get_instance_color(i));
	}

	return colors;

}




void MultiMesh::set_mesh(const Ref<Mesh>& p_mesh) {

	mesh=p_mesh;
	if (!mesh.is_null())
		VisualServer::get_singleton()->multimesh_set_mesh(multimesh,mesh->get_rid());
	else
		VisualServer::get_singleton()->multimesh_set_mesh(multimesh,RID());

}

Ref<Mesh> MultiMesh::get_mesh() const {

	return mesh;

}

void MultiMesh::set_instance_count(int p_count) {

	VisualServer::get_singleton()->multimesh_allocate(multimesh,p_count,VS::MultimeshTransformFormat(transform_format),VS::MultimeshColorFormat(color_format));

}
int MultiMesh::get_instance_count() const {

	return VisualServer::get_singleton()->multimesh_get_instance_count(multimesh);

}

void MultiMesh::set_instance_transform(int p_instance, const Transform& p_transform) {

	VisualServer::get_singleton()->multimesh_instance_set_transform(multimesh,p_instance,p_transform);


}
Transform MultiMesh::get_instance_transform(int p_instance) const {

	return VisualServer::get_singleton()->multimesh_instance_get_transform(multimesh,p_instance);

}

void MultiMesh::set_instance_color(int p_instance, const Color& p_color) {


	VisualServer::get_singleton()->multimesh_instance_set_color(multimesh,p_instance,p_color);

}
Color MultiMesh::get_instance_color(int p_instance) const {

	return VisualServer::get_singleton()->multimesh_instance_get_color(multimesh,p_instance);

}


AABB MultiMesh::get_aabb() const {

	return VisualServer::get_singleton()->multimesh_get_aabb(multimesh);

}


RID MultiMesh::get_rid() const {

	return multimesh;

}


void MultiMesh::set_color_format(ColorFormat p_color_format) {

	color_format=p_color_format;
}

MultiMesh::ColorFormat MultiMesh::get_color_format() const{

	return color_format;
}

void MultiMesh::set_transform_format(TransformFormat p_transform_format){

	transform_format=p_transform_format;
}
MultiMesh::TransformFormat MultiMesh::get_transform_format() const{

	return transform_format;
}


void MultiMesh::_bind_methods() {

	ClassDB::bind_method(_MD("set_mesh","mesh:Mesh"),&MultiMesh::set_mesh);
	ClassDB::bind_method(_MD("get_mesh:Mesh"),&MultiMesh::get_mesh);
	ClassDB::bind_method(_MD("set_color_format","format"),&MultiMesh::set_color_format);
	ClassDB::bind_method(_MD("get_color_format"),&MultiMesh::get_color_format);
	ClassDB::bind_method(_MD("set_transform_format","format"),&MultiMesh::set_transform_format);
	ClassDB::bind_method(_MD("get_transform_format"),&MultiMesh::get_transform_format);

	ClassDB::bind_method(_MD("set_instance_count","count"),&MultiMesh::set_instance_count);
	ClassDB::bind_method(_MD("get_instance_count"),&MultiMesh::get_instance_count);
	ClassDB::bind_method(_MD("set_instance_transform","instance","transform"),&MultiMesh::set_instance_transform);
	ClassDB::bind_method(_MD("get_instance_transform","instance"),&MultiMesh::get_instance_transform);
	ClassDB::bind_method(_MD("set_instance_color","instance","color"),&MultiMesh::set_instance_color);
	ClassDB::bind_method(_MD("get_instance_color","instance"),&MultiMesh::get_instance_color);
	ClassDB::bind_method(_MD("get_aabb"),&MultiMesh::get_aabb);


	ClassDB::bind_method(_MD("_set_transform_array"),&MultiMesh::_set_transform_array);
	ClassDB::bind_method(_MD("_get_transform_array"),&MultiMesh::_get_transform_array);
	ClassDB::bind_method(_MD("_set_color_array"),&MultiMesh::_set_color_array);
	ClassDB::bind_method(_MD("_get_color_array"),&MultiMesh::_get_color_array);


	ADD_PROPERTY(PropertyInfo(Variant::INT,"color_format",PROPERTY_HINT_ENUM,"None,Byte,Float"), _SCS("set_color_format"), _SCS("get_color_format"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"transform_format",PROPERTY_HINT_ENUM,"2D,3D"), _SCS("set_transform_format"), _SCS("get_transform_format"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"instance_count",PROPERTY_HINT_RANGE,"0,16384,1"), _SCS("set_instance_count"), _SCS("get_instance_count"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"mesh",PROPERTY_HINT_RESOURCE_TYPE,"Mesh"), _SCS("set_mesh"), _SCS("get_mesh"));
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3_ARRAY,"transform_array",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR), _SCS("_set_transform_array"), _SCS("_get_transform_array"));
	ADD_PROPERTY(PropertyInfo(Variant::COLOR_ARRAY,"color_array",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR), _SCS("_set_color_array"), _SCS("_get_color_array"));



	BIND_CONSTANT( TRANSFORM_2D  );
	BIND_CONSTANT( TRANSFORM_3D  );
	BIND_CONSTANT( COLOR_NONE  );
	BIND_CONSTANT( COLOR_8BIT  );
	BIND_CONSTANT( COLOR_FLOAT  );

}

MultiMesh::MultiMesh() {

	multimesh = VisualServer::get_singleton()->multimesh_create();
	color_format=COLOR_NONE;
	transform_format=TRANSFORM_2D;
}

MultiMesh::~MultiMesh() {

	VisualServer::get_singleton()->free(multimesh);
}
