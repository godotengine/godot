#pragma once

#include "objects/jolt_object_impl_3d.hpp"

class JoltShapedObjectImpl3D : public JoltObjectImpl3D {
public:
	explicit JoltShapedObjectImpl3D(ObjectType p_object_type);

	~JoltShapedObjectImpl3D() override;

	Transform3D get_transform_unscaled() const;

	Transform3D get_transform_scaled() const;

	Vector3 get_scale() const { return scale; }

	Basis get_basis() const;

	Vector3 get_position() const;

	Vector3 get_center_of_mass() const;

	Vector3 get_center_of_mass_local() const;

	Vector3 get_linear_velocity() const;

	Vector3 get_angular_velocity() const;

	virtual bool has_custom_center_of_mass() const = 0;

	virtual Vector3 get_center_of_mass_custom() const = 0;

	JPH::ShapeRefC try_build_shape();

	JPH::ShapeRefC build_shape();

	void update_shape();

	const JPH::Shape* get_jolt_shape() const { return jolt_shape; }

	const JPH::Shape* get_previous_jolt_shape() const { return previous_jolt_shape; }

	void add_shape(JoltShapeImpl3D* p_shape, Transform3D p_transform, bool p_disabled);

	void remove_shape(const JoltShapeImpl3D* p_shape);

	void remove_shape(int32_t p_index);

	JoltShapeImpl3D* get_shape(int32_t p_index) const;

	void set_shape(int32_t p_index, JoltShapeImpl3D* p_shape);

	void clear_shapes();

	int32_t get_shape_count() const { return shapes.size(); }

	int32_t find_shape_index(uint32_t p_shape_instance_id) const;

	int32_t find_shape_index(const JPH::SubShapeID& p_sub_shape_id) const;

	JoltShapeImpl3D* find_shape(uint32_t p_shape_instance_id) const;

	JoltShapeImpl3D* find_shape(const JPH::SubShapeID& p_sub_shape_id) const;

	Transform3D get_shape_transform_unscaled(int32_t p_index) const;

	Transform3D get_shape_transform_scaled(int32_t p_index) const;

	Vector3 get_shape_scale(int32_t p_index) const;

	void set_shape_transform(int32_t p_index, Transform3D p_transform);

	bool is_shape_disabled(int32_t p_index) const;

	void set_shape_disabled(int32_t p_index, bool p_disabled);

	void post_step(float p_step, JPH::Body& p_jolt_body) override;

protected:
	friend class JoltShapeImpl3D;

	virtual JPH::EMotionType _get_motion_type() const = 0;

	JPH::ShapeRefC _try_build_single_shape();

	JPH::ShapeRefC _try_build_compound_shape();

	virtual void _shapes_changed();

	virtual void _shapes_built() { }

	void _space_changing() override;

	Vector3 scale = {1.0f, 1.0f, 1.0f};

	JPH::ShapeRefC jolt_shape;

	JPH::ShapeRefC previous_jolt_shape;

	JPH::BodyCreationSettings* jolt_settings = new JPH::BodyCreationSettings();
};
