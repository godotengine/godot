#include "Variant.hpp"

#include <gdnative/variant.h>

#include "CoreTypes.hpp"
#include "Defs.hpp"
#include "GodotGlobal.hpp"
#include "Object.hpp"

#include <iostream>

namespace godot {

Variant::Variant() {
	godot::api->godot_variant_new_nil(&_godot_variant);
}

Variant::Variant(const Variant &v) {
	godot::api->godot_variant_new_copy(&_godot_variant, &v._godot_variant);
}

Variant::Variant(bool p_bool) {
	godot::api->godot_variant_new_bool(&_godot_variant, p_bool);
}

Variant::Variant(signed int p_int) // real one
{
	godot::api->godot_variant_new_int(&_godot_variant, p_int);
}

Variant::Variant(unsigned int p_int) {
	godot::api->godot_variant_new_uint(&_godot_variant, p_int);
}

Variant::Variant(signed short p_short) // real one
{
	godot::api->godot_variant_new_int(&_godot_variant, (int)p_short);
}

Variant::Variant(int64_t p_char) // real one
{
	godot::api->godot_variant_new_int(&_godot_variant, p_char);
}

Variant::Variant(uint64_t p_char) {
	godot::api->godot_variant_new_uint(&_godot_variant, p_char);
}

Variant::Variant(float p_float) {
	godot::api->godot_variant_new_real(&_godot_variant, p_float);
}

Variant::Variant(double p_double) {
	godot::api->godot_variant_new_real(&_godot_variant, p_double);
}

Variant::Variant(const String &p_string) {
	godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&p_string);
}

Variant::Variant(const char *const p_cstring) {
	String s = String(p_cstring);
	godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&s);
}

Variant::Variant(const wchar_t *p_wstring) {
	String s = p_wstring;
	godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&s);
}

Variant::Variant(const Vector2 &p_vector2) {
	godot::api->godot_variant_new_vector2(&_godot_variant, (godot_vector2 *)&p_vector2);
}

Variant::Variant(const Rect2 &p_rect2) {
	godot::api->godot_variant_new_rect2(&_godot_variant, (godot_rect2 *)&p_rect2);
}

Variant::Variant(const Vector3 &p_vector3) {
	godot::api->godot_variant_new_vector3(&_godot_variant, (godot_vector3 *)&p_vector3);
}

Variant::Variant(const Plane &p_plane) {
	godot::api->godot_variant_new_plane(&_godot_variant, (godot_plane *)&p_plane);
}

Variant::Variant(const AABB &p_aabb) {
	godot::api->godot_variant_new_aabb(&_godot_variant, (godot_aabb *)&p_aabb);
}

Variant::Variant(const Quat &p_quat) {
	godot::api->godot_variant_new_quat(&_godot_variant, (godot_quat *)&p_quat);
}

Variant::Variant(const Basis &p_transform) {
	godot::api->godot_variant_new_basis(&_godot_variant, (godot_basis *)&p_transform);
}

Variant::Variant(const Transform2D &p_transform) {
	godot::api->godot_variant_new_transform2d(&_godot_variant, (godot_transform2d *)&p_transform);
}

Variant::Variant(const Transform &p_transform) {
	godot::api->godot_variant_new_transform(&_godot_variant, (godot_transform *)&p_transform);
}

Variant::Variant(const Color &p_color) {
	godot::api->godot_variant_new_color(&_godot_variant, (godot_color *)&p_color);
}

Variant::Variant(const NodePath &p_path) {
	godot::api->godot_variant_new_node_path(&_godot_variant, (godot_node_path *)&p_path);
}

Variant::Variant(const RID &p_rid) {
	godot::api->godot_variant_new_rid(&_godot_variant, (godot_rid *)&p_rid);
}

Variant::Variant(const Object *p_object) {
	if (p_object)
		godot::api->godot_variant_new_object(&_godot_variant, p_object->_owner);
	else
		godot::api->godot_variant_new_nil(&_godot_variant);
}

Variant::Variant(const Dictionary &p_dictionary) {
	godot::api->godot_variant_new_dictionary(&_godot_variant, (godot_dictionary *)&p_dictionary);
}

Variant::Variant(const Array &p_array) {
	godot::api->godot_variant_new_array(&_godot_variant, (godot_array *)&p_array);
}

Variant::Variant(const PoolByteArray &p_raw_array) {
	godot::api->godot_variant_new_pool_byte_array(&_godot_variant, (godot_pool_byte_array *)&p_raw_array);
}

Variant::Variant(const PoolIntArray &p_int_array) {
	godot::api->godot_variant_new_pool_int_array(&_godot_variant, (godot_pool_int_array *)&p_int_array);
}

Variant::Variant(const PoolRealArray &p_real_array) {
	godot::api->godot_variant_new_pool_real_array(&_godot_variant, (godot_pool_real_array *)&p_real_array);
}

Variant::Variant(const PoolStringArray &p_string_array) {
	godot::api->godot_variant_new_pool_string_array(&_godot_variant, (godot_pool_string_array *)&p_string_array);
}

Variant::Variant(const PoolVector2Array &p_vector2_array) {
	godot::api->godot_variant_new_pool_vector2_array(&_godot_variant, (godot_pool_vector2_array *)&p_vector2_array);
}

Variant::Variant(const PoolVector3Array &p_vector3_array) {
	godot::api->godot_variant_new_pool_vector3_array(&_godot_variant, (godot_pool_vector3_array *)&p_vector3_array);
}

Variant::Variant(const PoolColorArray &p_color_array) {
	godot::api->godot_variant_new_pool_color_array(&_godot_variant, (godot_pool_color_array *)&p_color_array);
}

Variant &Variant::operator=(const Variant &v) {
	godot::api->godot_variant_new_copy(&_godot_variant, &v._godot_variant);
	return *this;
}

Variant::operator bool() const {
	return booleanize();
}
Variant::operator signed int() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator unsigned int() const // this is the real one
{
	return godot::api->godot_variant_as_uint(&_godot_variant);
}
Variant::operator signed short() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator unsigned short() const {
	return godot::api->godot_variant_as_uint(&_godot_variant);
}
Variant::operator signed char() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator unsigned char() const {
	return godot::api->godot_variant_as_uint(&_godot_variant);
}
Variant::operator int64_t() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator uint64_t() const {
	return godot::api->godot_variant_as_uint(&_godot_variant);
}

Variant::operator wchar_t() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}

Variant::operator float() const {
	return godot::api->godot_variant_as_real(&_godot_variant);
}

Variant::operator double() const {
	return godot::api->godot_variant_as_real(&_godot_variant);
}
Variant::operator String() const {
	String ret;
	*(godot_string *)&ret = godot::api->godot_variant_as_string(&_godot_variant);
	return ret;
}
Variant::operator Vector2() const {
	godot_vector2 s = godot::api->godot_variant_as_vector2(&_godot_variant);
	return *(Vector2 *)&s;
}
Variant::operator Rect2() const {
	godot_rect2 s = godot::api->godot_variant_as_rect2(&_godot_variant);
	return *(Rect2 *)&s;
}
Variant::operator Vector3() const {
	godot_vector3 s = godot::api->godot_variant_as_vector3(&_godot_variant);
	return *(Vector3 *)&s;
}
Variant::operator Plane() const {
	godot_plane s = godot::api->godot_variant_as_plane(&_godot_variant);
	return *(Plane *)&s;
}
Variant::operator AABB() const {
	godot_aabb s = godot::api->godot_variant_as_aabb(&_godot_variant);
	return *(AABB *)&s;
}
Variant::operator Quat() const {
	godot_quat s = godot::api->godot_variant_as_quat(&_godot_variant);
	return *(Quat *)&s;
}
Variant::operator Basis() const {
	godot_basis s = godot::api->godot_variant_as_basis(&_godot_variant);
	return *(Basis *)&s;
}
Variant::operator Transform() const {
	godot_transform s = godot::api->godot_variant_as_transform(&_godot_variant);
	return *(Transform *)&s;
}
Variant::operator Transform2D() const {
	godot_transform2d s = godot::api->godot_variant_as_transform2d(&_godot_variant);
	return *(Transform2D *)&s;
}

Variant::operator Color() const {
	godot_color s = godot::api->godot_variant_as_color(&_godot_variant);
	return *(Color *)&s;
}
Variant::operator NodePath() const {
	NodePath ret;
	*(godot_node_path *)&ret = godot::api->godot_variant_as_node_path(&_godot_variant);
	return ret;
}
Variant::operator RID() const {
	godot_rid s = godot::api->godot_variant_as_rid(&_godot_variant);
	return *(RID *)&s;
}

Variant::operator Dictionary() const {
	Dictionary ret(godot::api->godot_variant_as_dictionary(&_godot_variant));
	return ret;
}

Variant::operator Array() const {
	Array ret(godot::api->godot_variant_as_array(&_godot_variant));
	return ret;
}

Variant::operator PoolByteArray() const {
	PoolByteArray ret;
	*(godot_pool_byte_array *)&ret = godot::api->godot_variant_as_pool_byte_array(&_godot_variant);
	return ret;
}
Variant::operator PoolIntArray() const {
	PoolIntArray ret;
	*(godot_pool_int_array *)&ret = godot::api->godot_variant_as_pool_int_array(&_godot_variant);
	return ret;
}
Variant::operator PoolRealArray() const {
	PoolRealArray ret;
	*(godot_pool_real_array *)&ret = godot::api->godot_variant_as_pool_real_array(&_godot_variant);
	return ret;
}
Variant::operator PoolStringArray() const {
	PoolStringArray ret;
	*(godot_pool_string_array *)&ret = godot::api->godot_variant_as_pool_string_array(&_godot_variant);
	return ret;
}
Variant::operator PoolVector2Array() const {
	PoolVector2Array ret;
	*(godot_pool_vector2_array *)&ret = godot::api->godot_variant_as_pool_vector2_array(&_godot_variant);
	return ret;
}
Variant::operator PoolVector3Array() const {
	PoolVector3Array ret;
	*(godot_pool_vector3_array *)&ret = godot::api->godot_variant_as_pool_vector3_array(&_godot_variant);
	return ret;
}
Variant::operator PoolColorArray() const {
	PoolColorArray ret;
	*(godot_pool_color_array *)&ret = godot::api->godot_variant_as_pool_color_array(&_godot_variant);
	return ret;
}
Variant::operator godot_object *() const {
	return godot::api->godot_variant_as_object(&_godot_variant);
}

Variant::Type Variant::get_type() const {
	return (Type)godot::api->godot_variant_get_type(&_godot_variant);
}

Variant Variant::call(const String &method, const Variant **args, const int arg_count) {
	Variant v;
	*(godot_variant *)&v = godot::api->godot_variant_call(&_godot_variant, (godot_string *)&method, (const godot_variant **)args, arg_count, nullptr);
	return v;
}

bool Variant::has_method(const String &method) {
	return godot::api->godot_variant_has_method(&_godot_variant, (godot_string *)&method);
}

bool Variant::operator==(const Variant &b) const {
	return godot::api->godot_variant_operator_equal(&_godot_variant, &b._godot_variant);
}

bool Variant::operator!=(const Variant &b) const {
	return !(*this == b);
}

bool Variant::operator<(const Variant &b) const {
	return godot::api->godot_variant_operator_less(&_godot_variant, &b._godot_variant);
}

bool Variant::operator<=(const Variant &b) const {
	return (*this < b) || (*this == b);
}

bool Variant::operator>(const Variant &b) const {
	return !(*this <= b);
}

bool Variant::operator>=(const Variant &b) const {
	return !(*this < b);
}

bool Variant::hash_compare(const Variant &b) const {
	return godot::api->godot_variant_hash_compare(&_godot_variant, &b._godot_variant);
}

bool Variant::booleanize() const {
	return godot::api->godot_variant_booleanize(&_godot_variant);
}

Variant::~Variant() {
	godot::api->godot_variant_destroy(&_godot_variant);
}

} // namespace godot
