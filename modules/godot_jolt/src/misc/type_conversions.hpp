#pragma once

_FORCE_INLINE_ Vector3 to_godot(const JPH::Vec3& p_vec) {
	return {(real_t)p_vec.GetX(), (real_t)p_vec.GetY(), (real_t)p_vec.GetZ()};
}

_FORCE_INLINE_ Vector3 to_godot(const JPH::DVec3& p_vec) {
	return {(real_t)p_vec.GetX(), (real_t)p_vec.GetY(), (real_t)p_vec.GetZ()};
}

_FORCE_INLINE_ Basis to_godot(const JPH::Quat& p_quat) {
	return {Quaternion(p_quat.GetX(), p_quat.GetY(), p_quat.GetZ(), p_quat.GetW())};
}

_FORCE_INLINE_ Transform3D to_godot(const JPH::Mat44& p_mat) {
	return {
		Vector3(p_mat(0, 0), p_mat(1, 0), p_mat(2, 0)),
		Vector3(p_mat(0, 1), p_mat(1, 1), p_mat(2, 1)),
		Vector3(p_mat(0, 2), p_mat(1, 2), p_mat(2, 2)),
		Vector3(p_mat(0, 3), p_mat(1, 3), p_mat(2, 3))};
}

_FORCE_INLINE_ Color to_godot(const JPH::Color& p_color) {
	const auto r = (float)p_color.r;
	const auto g = (float)p_color.g;
	const auto b = (float)p_color.b;
	const auto a = (float)p_color.a;

	return {
		r == 0.0f ? 0.0f : 255.0f / r,
		g == 0.0f ? 0.0f : 255.0f / g,
		b == 0.0f ? 0.0f : 255.0f / b,
		a == 0.0f ? 0.0f : 255.0f / a};
}

_FORCE_INLINE_ String to_godot(const JPH::String& p_str) {
	return String::utf8(p_str.c_str(), (int32_t)p_str.length());
}

_FORCE_INLINE_ AABB to_godot(const JPH::AABox& p_aabb) {
	return {to_godot(p_aabb.mMin), to_godot(p_aabb.mMax - p_aabb.mMin)};
}

_FORCE_INLINE_ JPH::Vec3 to_jolt(const Vector3& p_vec) {
	return {(float)p_vec.x, (float)p_vec.y, (float)p_vec.z};
}

_FORCE_INLINE_ JPH::Quat to_jolt(const Basis& p_basis) {
	const Quaternion quat = p_basis.get_quaternion().normalized();
	return {(float)quat.x, (float)quat.y, (float)quat.z, (float)quat.w};
}

_FORCE_INLINE_ JPH::Mat44 to_jolt(const Transform3D& p_transform) {
	const Basis& b = p_transform.basis;
	const Vector3& o = p_transform.origin;

	return {
		JPH::Vec4(b[0][0], b[1][0], b[2][0], 0.0f),
		JPH::Vec4(b[0][1], b[1][1], b[2][1], 0.0f),
		JPH::Vec4(b[0][2], b[1][2], b[2][2], 0.0f),
		JPH::Vec3(o.x, o.y, o.z)};
}

_FORCE_INLINE_ JPH::Color to_jolt(const Color& p_color) {
	return JPH::Color((JPH::uint32)p_color.to_abgr32());
}

_FORCE_INLINE_ JPH::String to_jolt(const String& p_str) {
	const CharString str_utf8 = p_str.utf8();
	return {str_utf8.get_data(), (size_t)str_utf8.length()};
}

_FORCE_INLINE_ JPH::AABox to_jolt(const AABB& p_aabb) {
	return {to_jolt(p_aabb.position), to_jolt(p_aabb.position + p_aabb.size)};
}

_FORCE_INLINE_ JPH::RVec3 to_jolt_r(const Vector3& p_vec) {
	return {p_vec.x, p_vec.y, p_vec.z};
}

_FORCE_INLINE_ JPH::RMat44 to_jolt_r(const Transform3D& p_transform) {
	const Basis& b = p_transform.basis;
	const Vector3& o = p_transform.origin;

	return {
		JPH::Vec4(b[0][0], b[1][0], b[2][0], 0.0f),
		JPH::Vec4(b[0][1], b[1][1], b[2][1], 0.0f),
		JPH::Vec4(b[0][2], b[1][2], b[2][2], 0.0f),
		JPH::RVec3(o.x, o.y, o.z)};
}
