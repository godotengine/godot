#ifndef  SPICY_PARTICLE_UPDATER_H
#define SPICY_PARTICLE_UPDATER_H

#include "SpicyProperty.h"
#include "SpicyParticleData.h"
#include "core/io/resource.h"


class SpicyParticleUpdater : public Resource {
	GDCLASS(SpicyParticleUpdater, Resource);
protected:
	static void _bind_methods() {}
public:
	SpicyParticleUpdater() = default;
	virtual ~SpicyParticleUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) {}
};

class LifetimeUpdater : public SpicyParticleUpdater {
	GDCLASS(LifetimeUpdater, SpicyParticleUpdater);
protected:
	static void _bind_methods();
public:
	LifetimeUpdater() = default;
	virtual ~LifetimeUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;
};

class PositionUpdater : public SpicyParticleUpdater {
	GDCLASS(PositionUpdater, SpicyParticleUpdater);
protected:
	static void _bind_methods();
public:
	PositionUpdater() = default;
	virtual ~PositionUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;
};

class VelocityUpdater : public SpicyParticleUpdater {
	GDCLASS(VelocityUpdater, SpicyParticleUpdater);
public:
	Ref<SpicyVector3Property> velocity_over_life;
	Ref<SpicyFloatProperty> radial_velocity;
	Ref<SpicyVector3Property> orbital_velocity;
protected:
	static void _bind_methods();
public:
	VelocityUpdater();
	virtual ~VelocityUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;

	void set_velocity_over_life_property(const Ref<SpicyVector3Property>& p_velocity_property);
	Ref<SpicyVector3Property> get_velocity_over_life_property() const;

	void set_radial_velocity_property(const Ref<SpicyFloatProperty>& p_radial_velocity_property);
	Ref<SpicyFloatProperty> get_radial_velocity_property() const;

	void set_orbital_velocity_property(const Ref<SpicyVector3Property>& p_orbital_velocity_property);
	Ref<SpicyVector3Property> get_orbital_velocity_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class AccelerationUpdater : public SpicyParticleUpdater {
	GDCLASS(AccelerationUpdater, SpicyParticleUpdater);
public:
	Ref<SpicyVector3Property> acceleration_property;
protected:
	static void _bind_methods();
public:
	AccelerationUpdater();
	virtual ~AccelerationUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;

	void set_acceleration_property(const Ref<SpicyVector3Property>& p_acceleration_property);
	Ref<SpicyVector3Property> get_acceleration_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class ColorUpdater : public SpicyParticleUpdater {
	GDCLASS(ColorUpdater, SpicyParticleUpdater);
public:
	Ref<Gradient> color_over_lifetime;
	Ref<Gradient> color_over_velocity;
	Vector2 speed_range;
protected:
	static void _bind_methods();
public:
	void _update_color(double dt, const Ref<ParticleData> p_data);
	ColorUpdater() : speed_range(0, 1) {};
	virtual ~ColorUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;

	void set_color_over_lifetime(const Ref<Gradient>& p_color_over_lifetime);
	Ref<Gradient> get_color_over_lifetime() const;

	void set_color_over_velocity(const Ref<Gradient>& p_color_over_velocity);
	Ref<Gradient> get_color_over_velocity() const;

	void set_speed_range(const Vector2& p_velocity_range);
	Vector2 get_speed_range() const;
};

class RotationUpdater : public SpicyParticleUpdater {
	GDCLASS(RotationUpdater, SpicyParticleUpdater);
public:
	Ref<SpicyVector3Property> rotation_property;
protected:
	static void _bind_methods();
public:
	RotationUpdater();
	virtual ~RotationUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;

	void set_rotation_property(const Ref<SpicyVector3Property>& p_rotation_property);
	Ref<SpicyVector3Property> get_rotation_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class SizeUpdater : public SpicyParticleUpdater {
	GDCLASS(SizeUpdater, SpicyParticleUpdater);
public:
	Ref<SpicyVector3Property> size_property;
protected:
	static void _bind_methods();
public:
	SizeUpdater();
	virtual ~SizeUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;

	void set_size_property(const Ref<SpicyVector3Property>& p_size_property);
	Ref<SpicyVector3Property> get_size_property() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};

class CustomDataUpdater : public SpicyParticleUpdater {
	GDCLASS(CustomDataUpdater, SpicyParticleUpdater);
public:
	Ref<SpicyFloatProperty> custom_data_property_x; //can be built-int or custom //lifetime
	Ref<SpicyFloatProperty> custom_data_property_y; //can be built-int or custom //loop phase (duration/lifetime)
	Ref<SpicyFloatProperty> custom_data_property_z; //can be built-int or custom //offset
	Ref<SpicyFloatProperty> custom_data_property_w; //can be custom

	bool use_builtin_data;
protected:
	static void _bind_methods();
public:
	CustomDataUpdater();
	virtual ~CustomDataUpdater() {}

	virtual void update(double dt, const Ref<ParticleData> p_data) override;

	void set_custom_data_property_x(const Ref<SpicyFloatProperty>& p_custom_data_property_x);
	Ref<SpicyFloatProperty> get_custom_data_property_x() const;

	void set_custom_data_property_y(const Ref<SpicyFloatProperty>& p_custom_data_property_y);
	Ref<SpicyFloatProperty> get_custom_data_property_y() const;

	void set_custom_data_property_z(const Ref<SpicyFloatProperty>& p_custom_data_property_z);
	Ref<SpicyFloatProperty> get_custom_data_property_z() const;

	void set_custom_data_property_w(const Ref<SpicyFloatProperty>& p_custom_data_property_w);
	Ref<SpicyFloatProperty> get_custom_data_property_w() const;

	void set_use_builtin_data(bool p_use_builtin_data);
	bool get_use_builtin_data() const;

	void _get_property_list(List<PropertyInfo>* r_props) const;
	bool _get(const StringName& p_property, Variant& r_value) const;
	bool _set(const StringName& p_property, const Variant& p_value);
	bool _property_can_revert(const StringName& p_name) const;
	bool _property_get_revert(const StringName& p_name, Variant& r_property) const;
};



#endif //SPICY_PARTICLE_UPDATER_H
