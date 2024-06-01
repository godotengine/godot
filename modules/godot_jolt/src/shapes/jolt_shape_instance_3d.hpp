#pragma once
#include "../common.h"
#include "containers/inline_vector.hpp"
#include "misc/type_conversions.hpp"
#include "misc/utility_functions.hpp"
#include "misc/error_macros.hpp"
#include "misc/math.hpp"
class JoltShapedObjectImpl3D;
class JoltShapeImpl3D;

class JoltShapeInstance3D : public RefCounted {
public:
	JoltShapeInstance3D(
		JoltShapedObjectImpl3D* p_parent,
		JoltShapeImpl3D* p_shape,
		const Transform3D& p_transform = {},
		const Vector3& p_scale = {1.0f, 1.0f, 1.0f},
		bool p_disabled = false
	);

	JoltShapeInstance3D(const JoltShapeInstance3D& p_other) = delete;

	JoltShapeInstance3D(JoltShapeInstance3D&& p_other) noexcept;

	~JoltShapeInstance3D();

	uint32_t get_id() const { return id; }

	JoltShapeImpl3D* get_shape() const { return shape; }

	const JPH::Shape* get_jolt_ref() const { return jolt_ref; }

	const Transform3D& get_transform_unscaled() const { return transform; }

	Transform3D get_transform_scaled() const { return transform.scaled_local(scale); }

	void set_transform(const Transform3D& p_transform) { transform = p_transform; }

	const Vector3& get_scale() const { return scale; }

	void set_scale(const Vector3& p_scale) { scale = p_scale; }

	bool is_built() const { return jolt_ref != nullptr; }

	bool is_enabled() const { return !disabled; }

	bool is_disabled() const { return disabled; }

	void enable() { disabled = false; }

	void disable() { disabled = true; }

	bool try_build();

	JoltShapeInstance3D& operator=(const JoltShapeInstance3D& p_other) = delete;

	JoltShapeInstance3D& operator=(JoltShapeInstance3D&& p_other) noexcept;

private:
	inline static uint32_t next_id = 1;

	Transform3D transform;

	Vector3 scale;

	JPH::ShapeRefC jolt_ref;

	JoltShapedObjectImpl3D* parent = nullptr;

	JoltShapeImpl3D* shape = nullptr;

	uint32_t id = next_id++;

	bool disabled = false;
};
