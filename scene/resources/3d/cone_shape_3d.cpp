#include "cone_shape_3d.h"

#include "servers/physics_server_3d.h"

Vector<Vector3> ConeShape3D::get_debug_mesh_lines() const {
	float c_radius = get_radius();
	float c_height = get_height();

	Vector<Vector3> points;

	Vector3 d(0, c_height * 0.5, 0);
	for (int i = 0; i < 360; i++) {
		float ra = Math::deg_to_rad((float)i);
		float rb = Math::deg_to_rad((float)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * c_radius;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * c_radius;

		points.push_back(Vector3(0, 0, a.y) + d);
		points.push_back(Vector3(0, 0, b.y) + d);

		points.push_back(Vector3(a.x, 0, a.y) - d);
		points.push_back(Vector3(b.x, 0, b.y) - d);

		if (i % 90 == 0) {
			points.push_back(Vector3(a.x, 0, a.y) + d);
			points.push_back(Vector3(a.x, 0, a.y) - d);
		}
	}

	return points;
}

real_t ConeShape3D::get_enclosing_radius() const {
	return Vector2(radius, height * 0.5).length();
}

void ConeShape3D::_update_shape() {
	Dictionary d;
	d["radius"] = radius;
	d["height"] = height;
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void ConeShape3D::set_radius(float p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0, "ConeShape3D radius cannot be negative.");
	radius = p_radius;
	if (radius > height * 0.5) {
		height = radius * 2.0;
	}
	_update_shape();
	emit_changed();
}

float ConeShape3D::get_radius() const {
	return radius;
}

void ConeShape3D::set_height(float p_height) {
	ERR_FAIL_COND_MSG(p_height < 0, "ConeShape3D height cannot be negative.");
	height = p_height;
	if (radius > height * 0.5) {
		radius = height * 0.5;
	}
	_update_shape();
	emit_changed();
}

float ConeShape3D::get_height() const {
	return height;
}

void ConeShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &ConeShape3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &ConeShape3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &ConeShape3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &ConeShape3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_height", "get_height");
	ADD_LINKED_PROPERTY("radius", "height");
	ADD_LINKED_PROPERTY("height", "radius");
}

ConeShape3D::ConeShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_CONE)) {
	_update_shape();
}
