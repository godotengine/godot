#include "mesh_instance_2d.h"

void MeshInstance2D::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {
		if (mesh.is_valid()) {
			draw_mesh(mesh, texture, normal_map);
		}
	}
}

void MeshInstance2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MeshInstance2D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MeshInstance2D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &MeshInstance2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &MeshInstance2D::get_texture);

	ClassDB::bind_method(D_METHOD("set_normal_map", "normal_map"), &MeshInstance2D::set_normal_map);
	ClassDB::bind_method(D_METHOD("get_normal_map"), &MeshInstance2D::get_normal_map);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "normal_map", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_normal_map", "get_normal_map");
}

void MeshInstance2D::set_mesh(const Ref<Mesh> &p_mesh) {

	mesh = p_mesh;
	update();
}

Ref<Mesh> MeshInstance2D::get_mesh() const {

	return mesh;
}

void MeshInstance2D::set_texture(const Ref<Texture> &p_texture) {

	if (p_texture == texture)
		return;
	texture = p_texture;
	update();
	emit_signal("texture_changed");
	_change_notify("texture");
}

void MeshInstance2D::set_normal_map(const Ref<Texture> &p_texture) {

	normal_map = p_texture;
	update();
}

Ref<Texture> MeshInstance2D::get_normal_map() const {

	return normal_map;
}

Ref<Texture> MeshInstance2D::get_texture() const {

	return texture;
}

Rect2 MeshInstance2D::_edit_get_rect() const {

	if (mesh.is_valid()) {
		AABB aabb = mesh->get_aabb();
		return Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);
	}

	return Node2D::_edit_get_rect();
}

MeshInstance2D::MeshInstance2D() {
}
