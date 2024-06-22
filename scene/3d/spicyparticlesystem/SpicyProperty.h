#ifndef  SPICY_PROPERTY_H
#define  SPICY_PROPERTY_H
#include "core/object/ref_counted.h"
#include "scene/resources/curve.h"
#include "scene/resources/gradient.h"
#include "core/math/random_number_generator.h"



class SpicyFloatProperty : public RefCounted {
	GDCLASS(SpicyFloatProperty, RefCounted);
public:
	enum SpicyFloatType {
		SPICY_FLOAT_TYPE_UNIFORM = 1,
		SPICY_FLOAT_TYPE_RANDOM = 2,
		SPICY_FLOAT_TYPE_CURVE = 4,
	};

protected:
	BitField<SpicyFloatType> display_prop_types = 0;
	String type_names = "Uniform:1,Random:2,Curve:4";
	StringName prop_type_str = "_type";
	StringName prop_name_str;

	SpicyFloatType prop_type = SPICY_FLOAT_TYPE_UNIFORM;

	real_t uniform;
	Vector2 random;
	Ref<Curve> curve;
	real_t curve_default;
protected:
	static void _bind_methods();
	real_t _get_uniform_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	real_t _get_random_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	real_t _get_curve_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
public:
	SpicyFloatProperty() {}
	virtual ~SpicyFloatProperty() {}

	real_t get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const;

	void set_available_types(BitField<SpicyFloatType> p_types);
	void set_prop_type(SpicyFloatType p_prop_type);
	SpicyFloatType get_prop_type() const;
	void set_property_name(const StringName& p_name);

	void set_default_type(SpicyFloatType p_prop_type);
	void set_default_uniform(real_t p_uniform);
	void set_default_random(const Vector2& p_random);

	bool get_property_list(List<PropertyInfo>* p_list, bool show_type_choice = true) const;
	bool get_property(const StringName& p_property, Variant& r_value) const;
	bool set_property(const StringName& p_property, const Variant& p_value, const Ref<Resource>& r);
	bool can_property_revert(const StringName& p_property);
	bool get_property_revert(const StringName& p_property, Variant& r_value);

	//GDScript methods
	void set_uniform(real_t p_uniform);
	real_t get_uniform() const;
	void set_random(const Vector2& p_random);
	Vector2 get_random() const;
	void set_curve(const Ref<Curve>& p_curve);
	Ref<Curve> get_curve() const;
};

class SpicyVector3Property : public RefCounted {
	GDCLASS(SpicyVector3Property, RefCounted);
public:
	enum SpicyVector3Type {
		SPICY_VECTOR_TYPE_UNIFORM = 1,
		SPICY_VECTOR_TYPE_RANDOM = 2,
		SPICY_VECTOR_TYPE_CURVE = 4,
		SPICY_VECTOR_TYPE_CURVE_XYZ = 8,
	};

protected:
	BitField<SpicyVector3Type> display_prop_types = 0;
	String type_names = "Uniform:1,Random:2,Curve:4,CurveXYZ:8";
	StringName prop_type_str = "_type";
	StringName prop_name_str;
	StringName prop_name_randmin_str;
	StringName prop_name_randmax_str;
	StringName prop_name_curve_x_str;
	StringName prop_name_curve_y_str;
	StringName prop_name_curve_z_str;

	SpicyVector3Type prop_type = SPICY_VECTOR_TYPE_UNIFORM;

	Vector3 uniform;
	Vector3 random_min;
	Vector3 random_max;
	Ref<Curve> curve_x;
	Ref<Curve> curve_y;
	Ref<Curve> curve_z;
	Vector3 curve_default;
protected:
	Vector3 _get_uniform_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	Vector3 _get_random_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	Vector3 _get_curve_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	Vector3 _get_curve_xyz_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	static void _bind_methods();
public:
	SpicyVector3Property() {}
	virtual ~SpicyVector3Property() {}

	Vector3 get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const;

	void set_available_types(BitField<SpicyVector3Type> p_types);
	void set_prop_type(SpicyVector3Type p_prop_type);
	SpicyVector3Type get_prop_type() const;
	void set_property_name(const StringName& p_name);

	void set_default_type(SpicyVector3Type p_prop_type);
	void set_default_uniform(Vector3 p_uniform);
	void set_default_random(const Vector3& p_random_min, const Vector3& p_random_max);

	bool get_property_list(List<PropertyInfo>* p_list, bool show_type_choice = true) const;
	bool get_property(const StringName& p_property, Variant& r_value) const;
	bool set_property(const StringName& p_property, const Variant& p_value, const Ref<Resource>& r);
	bool can_property_revert(const StringName& p_property);
	bool get_property_revert(const StringName& p_property, Variant& r_value);

	//GDScript methods
	void set_uniform(Vector3 p_uniform);
	Vector3 get_uniform() const;
	void set_random(const Vector3& p_random_min, const Vector3& p_random_max);
	Vector3 get_random_min() const;
	Vector3 get_random_max() const;
	void set_curve(const Ref<Curve>& p_curve);
	Ref<Curve> get_curve() const;
	void set_curve_xyz(const Ref<Curve>& p_curve_x, const Ref<Curve>& p_curve_y, const Ref<Curve>& p_curve_z);
	Ref<Curve> get_curve_x() const;
	Ref<Curve> get_curve_y() const;
	Ref<Curve> get_curve_z() const;
};

class SpicyColorProperty : public RefCounted {
	GDCLASS(SpicyColorProperty, RefCounted);
public:
	enum SpicyColorType {
		SPICY_COLOR_TYPE_UNIFORM = 1,
		SPICY_COLOR_TYPE_RANDOM = 2,
		SPICY_COLOR_TYPE_GRADIENT = 4
	};

protected:
	BitField<SpicyColorType> display_prop_types = 0;
	String type_names = "Uniform:1,Random:2,Gradient:4";
	StringName prop_type_str = " Type";
	StringName prop_name_str;

	SpicyColorType prop_type = SPICY_COLOR_TYPE_UNIFORM;

	Color uniform;
	Ref<Gradient> random;
	Ref<Gradient> gradient;
protected:
	static void _bind_methods();
	Color _get_uniform_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	Color _get_random_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
	Color _get_gradient_value(const Ref<RandomNumberGenerator>& rng, real_t sample) const;
public:
	SpicyColorProperty() {}
	virtual ~SpicyColorProperty() {}

	Color get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const;

	void set_available_types(BitField<SpicyColorType> p_types);
	void set_prop_type(SpicyColorType p_prop_type);
	SpicyColorType get_prop_type() const;
	void set_property_name(const StringName& p_name);

	void set_default_type(SpicyColorType p_prop_type);
	void set_default_uniform(Color p_uniform);

	bool get_property_list(List<PropertyInfo>* p_list, bool show_type_choice = true) const;
	bool get_property(const StringName& p_property, Variant& r_value) const;
	bool set_property(const StringName& p_property, const Variant& p_value, const Ref<Resource>& r);
	bool can_property_revert(const StringName& p_property);
	bool get_property_revert(const StringName& p_property, Variant& r_value);

	//GDScript methods
	void set_uniform(Color p_uniform);
	Color get_uniform() const;
	void set_random(const Ref<Gradient>& p_random);
	Ref<Gradient> get_random() const;
	void set_gradient(const Ref<Gradient>& p_gradient);
	Ref<Gradient> get_gradient() const;
};

///////////////////////////////////////////////////////////////////////////////////
/// // SpicyProperty base class
#pragma region BaseSpicyProperties 

class SpicyProperty : public Resource {
	GDCLASS(SpicyProperty, Resource);
protected:
	static void _bind_methods() {}
public:
	inline SpicyProperty() = default;
	virtual ~SpicyProperty() {}
};

#pragma endregion

#pragma region EmissionShapes

class EmissionShape : public SpicyProperty {
	GDCLASS(EmissionShape, SpicyProperty);
protected:
	static void _bind_methods() {}
public:
	inline EmissionShape() {}
	virtual ~EmissionShape() {}
	inline virtual Vector3 get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const { return Vector3(0, 0, 0); };
};

class PointEmissionShape : public EmissionShape {
	GDCLASS(PointEmissionShape, EmissionShape);
protected:
	Vector3 point;
protected:
	static void _bind_methods();
public:
	inline PointEmissionShape() : point(0.0, 0.0, 0.0) {}
	virtual ~PointEmissionShape() {}
	virtual Vector3 get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const override;

	void set_point(const Vector3& p_box_extents);
	Vector3 get_point() const;
};

class BoxEmissionShape : public EmissionShape {
	GDCLASS(BoxEmissionShape, EmissionShape);
protected:
	Vector3 box_extents;
protected:
	static void _bind_methods();
public:
	inline BoxEmissionShape() : box_extents(1.0, 1.0, 1.0) {}
	virtual ~BoxEmissionShape() {}
	virtual Vector3 get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const override;

	void set_box_extents(const Vector3& p_box_extents);
	Vector3 get_box_extents() const;
};

class SphereEmissionShape : public EmissionShape {
	GDCLASS(SphereEmissionShape, EmissionShape);
public:
	enum EmissionType {
		SPHERE_EMISSION_SHAPE_TYPE_VOLUME,
		SPHERE_EMISSION_SHAPE_TYPE_SURFACE
	};
protected:
	real_t radius;
	real_t radius_thickness;
	real_t radius_thickened;
protected:
	static void _bind_methods();
public:
	inline SphereEmissionShape() : radius(1.0), radius_thickness(1.0) {}
	virtual ~SphereEmissionShape() {}
	virtual Vector3 get_value(const Ref<RandomNumberGenerator>& rng, real_t sample = 0) const override;

	void set_radius(real_t p_radius);
	real_t get_radius() const;

	void set_radius_thickness(real_t p_radius_thickness);
	real_t get_radius_thickness() const;
};
#pragma endregion


VARIANT_ENUM_CAST(SpicyFloatProperty::SpicyFloatType)
VARIANT_ENUM_CAST(SpicyVector3Property::SpicyVector3Type)
VARIANT_ENUM_CAST(SpicyColorProperty::SpicyColorType)

#endif // ! SPICY_PROPERTY_H
