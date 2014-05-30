#include "immediate_geometry.h"


void ImmediateGeometry::begin(Mesh::PrimitiveType p_primitive,const Ref<Texture>& p_texture) {

	VS::get_singleton()->immediate_begin(im,(VS::PrimitiveType)p_primitive,p_texture.is_valid()?p_texture->get_rid():RID());
	if (p_texture.is_valid())
		cached_textures.push_back(p_texture);

}

void ImmediateGeometry::set_normal(const Vector3& p_normal){

	VS::get_singleton()->immediate_normal(im,p_normal);
}

void ImmediateGeometry::set_tangent(const Plane& p_tangent){

	VS::get_singleton()->immediate_tangent(im,p_tangent);

}

void ImmediateGeometry::set_color(const Color& p_color){

	VS::get_singleton()->immediate_color(im,p_color);

}

void ImmediateGeometry::set_uv(const Vector2& p_uv){

	VS::get_singleton()->immediate_uv(im,p_uv);

}

void ImmediateGeometry::set_uv2(const Vector2& p_uv2){

	VS::get_singleton()->immediate_uv2(im,p_uv2);

}

void ImmediateGeometry::add_vertex(const Vector3& p_vertex){

	VS::get_singleton()->immediate_vertex(im,p_vertex);
	if (empty) {
		aabb.pos=p_vertex;
		aabb.size=Vector3();
	} else {
		aabb.expand_to(p_vertex);
	}
}

void ImmediateGeometry::end(){

	VS::get_singleton()->immediate_end(im);

}

void ImmediateGeometry::clear(){

	VS::get_singleton()->immediate_clear(im);
	empty=true;
	cached_textures.clear();

}

AABB ImmediateGeometry::get_aabb() const {

	return aabb;
}
DVector<Face3> ImmediateGeometry::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
}

void ImmediateGeometry::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("begin","primitive","texture:Texture"),&ImmediateGeometry::begin);
	ObjectTypeDB::bind_method(_MD("set_normal","normal"),&ImmediateGeometry::set_normal);
	ObjectTypeDB::bind_method(_MD("set_tangent","tangent"),&ImmediateGeometry::set_tangent);
	ObjectTypeDB::bind_method(_MD("set_color","color"),&ImmediateGeometry::set_color);
	ObjectTypeDB::bind_method(_MD("set_uv","uv"),&ImmediateGeometry::set_uv);
	ObjectTypeDB::bind_method(_MD("set_uv2","uv"),&ImmediateGeometry::set_uv2);
	ObjectTypeDB::bind_method(_MD("add_vertex","color"),&ImmediateGeometry::add_vertex);
	ObjectTypeDB::bind_method(_MD("end"),&ImmediateGeometry::end);
	ObjectTypeDB::bind_method(_MD("clear"),&ImmediateGeometry::clear);

}

ImmediateGeometry::ImmediateGeometry() {

	im = VisualServer::get_singleton()->immediate_create();
	set_base(im);
	empty=true;

}


ImmediateGeometry::~ImmediateGeometry() {

	VisualServer::get_singleton()->free(im);

}
