#pragma once
#include "../common.h"
#include "misc/type_conversions.hpp"
#include "misc/error_macros.hpp"
#include "containers/hash_map.hpp"
#include "containers/hash_set.hpp"
#include "containers/local_vector.hpp"
#include "containers/inline_vector.hpp"

class JoltShapedObjectImpl3D;

class JoltShapeImpl3D {
public:
	using ShapeType = PhysicsServer3D::ShapeType;

	virtual ~JoltShapeImpl3D() = 0;

	RID get_rid() const { return rid; }

	void set_rid(const RID& p_rid) { rid = p_rid; }

	void add_owner(JoltShapedObjectImpl3D* p_owner);

	void remove_owner(JoltShapedObjectImpl3D* p_owner);

	void remove_self();

	virtual ShapeType get_type() const = 0;

	virtual bool is_convex() const = 0;

	virtual Variant get_data() const = 0;

	virtual void set_data(const Variant& p_data) = 0;

	virtual float get_margin() const = 0;

	virtual void set_margin(float p_margin) = 0;

	virtual AABB get_aabb() const = 0;

	float get_solver_bias() const;

	void set_solver_bias(float p_bias);

	JPH::ShapeRefC try_build();

	void destroy();

	const JPH::Shape* get_jolt_ref() const { return jolt_ref; }

	static JPH::ShapeRefC with_scale(const JPH::Shape* p_shape, const Vector3& p_scale);

	static JPH::ShapeRefC with_basis_origin(
		const JPH::Shape* p_shape,
		const Basis& p_basis,
		const Vector3& p_origin
	);

	static JPH::ShapeRefC with_center_of_mass_offset(
		const JPH::Shape* p_shape,
		const Vector3& p_offset
	);

	static JPH::ShapeRefC with_center_of_mass(
		const JPH::Shape* p_shape,
		const Vector3& p_center_of_mass
	);

	static JPH::ShapeRefC with_user_data(const JPH::Shape* p_shape, uint64_t p_user_data);

	static JPH::ShapeRefC with_double_sided(const JPH::Shape* p_shape, bool p_back_face_collision);

	static JPH::ShapeRefC without_custom_shapes(const JPH::Shape* p_shape);

	static Vector3 make_scale_valid(const JPH::Shape* p_shape, const Vector3& p_scale);

	static bool is_scale_valid(
		const Vector3& p_scale,
		const Vector3& p_valid_scale,
		real_t p_tolerance = 0.01f
	);

protected:
	virtual JPH::ShapeRefC _build() const = 0;

	String _owners_to_string() const;

	JHashMap<JoltShapedObjectImpl3D*, int32_t> ref_counts_by_owner;

	RID rid;

	JPH::ShapeRefC jolt_ref;
};

#ifdef TOOLS_ENABLED

#define ERR_PRINT_INVALID_SCALE_MSG(m_scale, m_valid_scale, m_msg)               \
	if (unlikely(!JoltShapeImpl3D::is_scale_valid(m_scale, valid_scale))) {      \
		ERR_PRINT(vformat(                                                       \
			"%s "                                                                \
			"A scale of %v is not supported by Godot Jolt for this shape/body. " \
			"The scale will instead be treated as %v.",                          \
			m_msg,                                                               \
			m_scale,                                                             \
			valid_scale                                                          \
		));                                                                      \
	} else                                                                       \
		((void)0)

#else // TOOLS_ENABLED

#define ERR_PRINT_INVALID_SCALE_MSG(m_scale, m_valid_scale, m_msg)

#endif // TOOLS_ENABLED

#define ENSURE_SCALE_VALID(m_shape, m_scale, m_msg)                                      \
	if (true) {                                                                          \
		const Vector3 valid_scale = JoltShapeImpl3D::make_scale_valid(m_shape, m_scale); \
		ERR_PRINT_INVALID_SCALE_MSG(m_scale, valid_scale, m_msg);                        \
		(m_scale) = valid_scale;                                                         \
	} else                                                                               \
		((void)0)
