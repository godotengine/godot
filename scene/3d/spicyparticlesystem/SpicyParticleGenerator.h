#ifndef SPICY_PARTICLE_GENERATOR_H
#define SPICY_PARTICLE_GENERATOR_H

#include "SpicyProperty.h"
#include "SpicyParticleData.h"

#include "core/io/resource.h"

class SpicyParticleGenerator : public Resource {
	GDCLASS(SpicyParticleGenerator, Resource);
protected:
	static void _bind_methods();
public:
	SpicyParticleGenerator() = default;
	virtual ~SpicyParticleGenerator() {}

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId);
};

class LifetimeGenerator : public SpicyParticleGenerator {
	GDCLASS(LifetimeGenerator, SpicyParticleGenerator);
protected:
	Ref<SpicyFloatProperty> lifetime_property;
protected:
	static void _bind_methods();
public:
	LifetimeGenerator();
	virtual ~LifetimeGenerator() {}

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId) override;

	void set_lifetime_property(const Ref<SpicyFloatProperty>& p_lifetime_property);
	Ref<SpicyFloatProperty> get_lifetime_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class PositionGenerator : public SpicyParticleGenerator {
	GDCLASS(PositionGenerator, SpicyParticleGenerator);
protected:
	Ref<EmissionShape> emission_shape;
	Vector3 scale;
	Vector3 rotation;
	Transform3D shape_transform;
protected:
	static void _bind_methods();
	void _set_rotation_scale();
public:
	inline PositionGenerator() : scale(Vector3(1, 1, 1)), rotation(Vector3(0, 0, 0)), shape_transform(Transform3D()) { }
	inline virtual ~PositionGenerator() {};

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId) override;

	void set_emission_shape(const Ref<EmissionShape>& p_emission_shape);
	Ref<EmissionShape> get_emission_shape() const;

	void set_position(const Vector3& p_box_extents);
	Vector3 get_position() const;

	void set_rotation(const Vector3& p_rotation);
	Vector3 get_rotation() const;

	void set_scale(const Vector3& p_scale);
	Vector3 get_scale() const;

};

class ColorGenerator : public SpicyParticleGenerator {
	GDCLASS(ColorGenerator, SpicyParticleGenerator);
protected:
	Ref<SpicyColorProperty> color_property;
protected:
	static void _bind_methods();
public:
	ColorGenerator();
	virtual ~ColorGenerator() {}

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId) override;

	void set_color_property(const Ref<SpicyColorProperty>& p_color_property);
	Ref<SpicyColorProperty> get_color_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class SizeGenerator : public SpicyParticleGenerator {
	GDCLASS(SizeGenerator, SpicyParticleGenerator);
protected:
	Ref<SpicyFloatProperty> size_multiplier_property;
	Ref<SpicyVector3Property> size_property;
protected:
	static void _bind_methods();
public:
	SizeGenerator();
	virtual ~SizeGenerator() {}

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId) override;

	void set_size_multiplier_property(const Ref<SpicyFloatProperty>& p_size_multiplier_property);
	Ref<SpicyFloatProperty> get_size_multiplier_property() const;

	void set_size_property(const Ref<SpicyVector3Property>& p_size);
	Ref<SpicyVector3Property> get_size_generator() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class VelocityGenerator : public SpicyParticleGenerator {
	GDCLASS(VelocityGenerator, SpicyParticleGenerator);
protected:
	Ref<SpicyVector3Property> velocity_property;
protected:
	static void _bind_methods();
public:
	VelocityGenerator();
	virtual ~VelocityGenerator() {}

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId) override;

	void set_velocity_property(const Ref<SpicyVector3Property>& p_velocity_property);
	Ref<SpicyVector3Property> get_velocity_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class RotationGenerator : public SpicyParticleGenerator {
	GDCLASS(RotationGenerator, SpicyParticleGenerator);
protected:
	Ref<SpicyVector3Property> rotation_property;
protected:
	static void _bind_methods();
public:
	RotationGenerator();
	virtual ~RotationGenerator() {}

	virtual void generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId) override;

	void set_rotation_property(const Ref<SpicyVector3Property>& p_rotation_property);
	Ref<SpicyVector3Property> get_rotation_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

#endif //SPICY_PARTICLE_GENERATOR_H