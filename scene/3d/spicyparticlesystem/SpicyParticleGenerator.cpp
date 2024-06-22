#include "SpicyParticleGenerator.h"




void SpicyParticleGenerator::_bind_methods()
{
}

void SpicyParticleGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	for (size_t i = startId; i < endId; i++)
	{
		//reset all p_data values
		p_data->position[i] = Vector3(0, 0, 0);
		p_data->rotation[i] = Vector3(0, 0, 0);
		p_data->scale[i] = Vector3(1, 1, 1);
		p_data->velocity[i] = Vector3(0, 0, 0);
		p_data->current_velocity[i] = Vector3(0, 0, 0);
		p_data->current_scale[i] = Vector3(1, 1, 1);
		p_data->current_rotation[i] = Vector3(0, 0, 0);
		p_data->acceleration[i] = Vector3(0, 0, 0);
		p_data->custom_data[i] = Vector4(0, 0, 0, 0);
		p_data->color[i] = Color(1, 1, 1, 1);
		p_data->current_color[i] = Color(1, 1, 1, 1);
		p_data->lifetime[i] = 0;
		p_data->life_remaining[i] = 0;
		p_data->normalized_lifetime[i] = 0;
		p_data->alive[i] = false;
	}
}

//////////////////////////////////////////////////////////////////////////

void LifetimeGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_lifetime_property", "lifetime_property"), &LifetimeGenerator::set_lifetime_property);
	ClassDB::bind_method(D_METHOD("get_lifetime_property"), &LifetimeGenerator::get_lifetime_property);
}

LifetimeGenerator::LifetimeGenerator()
{
	Ref<SpicyFloatProperty> lifetime_p(memnew(SpicyFloatProperty));
	lifetime_property = lifetime_p;
	lifetime_property->set_available_types(SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_UNIFORM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_RANDOM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_CURVE);
	lifetime_property->set_property_name("lifetime");
	lifetime_property->set_default_uniform(1);
	lifetime_property->set_default_random(Vector2(1, 2));
	Ref<Curve> curve(memnew(Curve));
	curve->add_point(Vector2(0, 1));
	curve->add_point(Vector2(1, 1));
}

void LifetimeGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	if (!lifetime_property.is_valid()) return;

	for (size_t i = startId; i < endId; i++)
	{
		p_data->lifetime[i] = lifetime_property->get_value(p_data->rng, p_data->current_duration_normalized);
		p_data->life_remaining[i] = p_data->lifetime[i];
	}
}

void LifetimeGenerator::set_lifetime_property(const Ref<SpicyFloatProperty>& p_lifetime_property)
{
	lifetime_property = p_lifetime_property;

	if (lifetime_property.is_null()) {
		Ref<SpicyFloatProperty> lifetime_p(memnew(SpicyFloatProperty));
		lifetime_property = lifetime_p;
		lifetime_property->set_available_types(SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_UNIFORM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_RANDOM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_CURVE);
		lifetime_property->set_property_name("lifetime");
		lifetime_property->set_default_uniform(1);
		lifetime_property->set_default_random(Vector2(1, 2));
	}

	notify_property_list_changed();
}

Ref<SpicyFloatProperty> LifetimeGenerator::get_lifetime_property() const
{
	return lifetime_property;
}

void LifetimeGenerator::_get_property_list(List<PropertyInfo>* r_props) const
{
	if (lifetime_property.is_valid()) {
		lifetime_property->get_property_list(r_props);
	}
}

bool LifetimeGenerator::_get(const StringName& p_property, Variant& r_value) const
{
	if (lifetime_property.is_valid()) {
		return lifetime_property->get_property(p_property, r_value);
	}
	return false;
}

bool LifetimeGenerator::_set(const StringName& p_property, const Variant& p_value)
{
	if (lifetime_property.is_valid()) {
		return lifetime_property->set_property(p_property, p_value, Ref<Resource>(this));
	}
	return false;
}

bool LifetimeGenerator::_property_can_revert(const StringName& p_name) const
{
	if (lifetime_property.is_valid()) {
		return lifetime_property->can_property_revert(p_name);
	}
	return false;
}

bool LifetimeGenerator::_property_get_revert(const StringName& p_name, Variant& r_property) const
{
	if (lifetime_property.is_valid()) {
		return lifetime_property->get_property_revert(p_name, r_property);
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////

void PositionGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_emission_shape", "generator"), &PositionGenerator::set_emission_shape);
	ClassDB::bind_method(D_METHOD("get_emission_shape"), &PositionGenerator::get_emission_shape);
	ClassDB::bind_method(D_METHOD("set_position", "position"), &PositionGenerator::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &PositionGenerator::get_position);
	ClassDB::bind_method(D_METHOD("set_rotation", "rotation"), &PositionGenerator::set_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation"), &PositionGenerator::get_rotation);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &PositionGenerator::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &PositionGenerator::get_scale);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "emission_shape", PROPERTY_HINT_RESOURCE_TYPE, "EmissionShape"), "set_emission_shape", "get_emission_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position", PROPERTY_HINT_RANGE, "-99999,99999,0.001,or_greater,or_less,hide_slider,suffix:m"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation", PROPERTY_HINT_RANGE, "-360,360,0.1,or_less,or_greater,radians"), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale", PROPERTY_HINT_LINK), "set_scale", "get_scale");
}

void PositionGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	if (!emission_shape.is_valid()) return;

	for (size_t i = startId; i < endId; i++)
	{
		p_data->position[i] = emission_shape->get_value(p_data->rng);
		p_data->position[i] = shape_transform.xform(p_data->position[i]);
		p_data->position[i] = p_data->emitter_transform.xform(p_data->position[i]);
	}
}

void PositionGenerator::set_emission_shape(const Ref<EmissionShape>& p_emission_shape)
{
	emission_shape = p_emission_shape;

	if (emission_shape.is_null()) {
		Ref<PointEmissionShape> point_emission_shape(memnew(PointEmissionShape));
		emission_shape = point_emission_shape;
	}
}

Ref<EmissionShape> PositionGenerator::get_emission_shape() const
{
	return emission_shape;
}

void PositionGenerator::set_position(const Vector3& p_position)
{
	shape_transform.origin = p_position;
}

Vector3 PositionGenerator::get_position() const
{
	return shape_transform.origin;
}

void PositionGenerator::set_rotation(const Vector3& p_rotation)
{
	rotation = p_rotation;

	_set_rotation_scale();
}

Vector3 PositionGenerator::get_rotation() const
{
	return rotation;
}

void PositionGenerator::set_scale(const Vector3& p_scale)
{
	scale = p_scale;

	_set_rotation_scale();
}

Vector3 PositionGenerator::get_scale() const
{
	return scale;
}

void PositionGenerator::_set_rotation_scale()
{
	Vector3 radians(Math::deg_to_rad(rotation.x), Math::deg_to_rad(rotation.y), Math::deg_to_rad(rotation.z));
	Quaternion q = Quaternion::from_euler(radians);

	Basis b;
	b.rotate(q);
	b.scale(scale);

	shape_transform.basis = b;
}

//////////////////////////////////////////////////////////////////////////

void ColorGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_color_property", "color_property"), &ColorGenerator::set_color_property);
	ClassDB::bind_method(D_METHOD("get_color_property"), &ColorGenerator::get_color_property);
}

ColorGenerator::ColorGenerator()
{
	Ref<SpicyColorProperty> color_p(memnew(SpicyColorProperty));
	color_property = color_p;
	color_property->set_available_types(SpicyColorProperty::SpicyColorType::SPICY_COLOR_TYPE_UNIFORM | SpicyColorProperty::SpicyColorType::SPICY_COLOR_TYPE_RANDOM | SpicyColorProperty::SpicyColorType::SPICY_COLOR_TYPE_GRADIENT);
	color_property->set_property_name("color");
	color_property->set_default_uniform(Color(1, 1, 1, 1));
}

void ColorGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	if (!color_property.is_valid()) return;

	for (size_t i = startId; i < endId; i++)
	{
		p_data->color[i] = color_property->get_value(p_data->rng, p_data->current_duration_normalized);
	}
}

void ColorGenerator::set_color_property(const Ref<SpicyColorProperty>& p_color_property)
{
	color_property = p_color_property;

	if (color_property.is_null()) {
		Ref<SpicyColorProperty> color_p(memnew(SpicyColorProperty));
		color_property = color_p;
		color_property->set_available_types(SpicyColorProperty::SpicyColorType::SPICY_COLOR_TYPE_UNIFORM | SpicyColorProperty::SpicyColorType::SPICY_COLOR_TYPE_RANDOM | SpicyColorProperty::SpicyColorType::SPICY_COLOR_TYPE_GRADIENT);
		color_property->set_property_name("color");
		color_property->set_default_uniform(Color(1, 1, 1, 1));
	}
}

Ref<SpicyColorProperty> ColorGenerator::get_color_property() const
{
	return color_property;
}

void ColorGenerator::_get_property_list(List<PropertyInfo>* r_props) const
{
	if (color_property.is_valid()) {
		color_property->get_property_list(r_props);
	}
}

bool ColorGenerator::_get(const StringName& p_property, Variant& r_value) const
{
	if (color_property.is_valid()) {
		return color_property->get_property(p_property, r_value);
	}
	return false;
}

bool ColorGenerator::_set(const StringName& p_property, const Variant& p_value)
{
	if (color_property.is_valid()) {
		return color_property->set_property(p_property, p_value, Ref<Resource>(this));
	}
	return false;
}

bool ColorGenerator::_property_can_revert(const StringName& p_name) const
{
	if (color_property.is_valid()) {
		return color_property->can_property_revert(p_name);
	}
	return false;
}

bool ColorGenerator::_property_get_revert(const StringName& p_name, Variant& r_property) const
{
	if (color_property.is_valid()) {
		return color_property->get_property_revert(p_name, r_property);
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////

void SizeGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_size_multiplier_property", "size_multiplier_property"), &SizeGenerator::set_size_multiplier_property);
	ClassDB::bind_method(D_METHOD("get_size_multiplier_property"), &SizeGenerator::get_size_multiplier_property);

	ClassDB::bind_method(D_METHOD("set_size_property", "property"), &SizeGenerator::set_size_property);
	ClassDB::bind_method(D_METHOD("get_size_generator"), &SizeGenerator::get_size_generator);
}

SizeGenerator::SizeGenerator()
{
	Ref<SpicyFloatProperty> size_multiplier_p(memnew(SpicyFloatProperty));
	size_multiplier_property = size_multiplier_p;
	size_multiplier_property->set_available_types(SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_UNIFORM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_RANDOM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_CURVE);
	size_multiplier_property->set_property_name("size_multiplier");
	size_multiplier_property->set_default_uniform(1);
	size_multiplier_property->set_default_random(Vector2(0.5, 1));

	Ref<SpicyVector3Property> size_p(memnew(SpicyVector3Property));
	size_property = size_p;
	size_property->set_available_types(SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_UNIFORM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_RANDOM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_CURVE | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_CURVE_XYZ);
	size_property->set_property_name("size");
	size_property->set_default_uniform(Vector3(1, 1, 1));
	size_property->set_default_random(Vector3(0.5, 0.5, 0.5), Vector3(1, 1, 1));
}

void SizeGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	if (!size_property.is_valid() && size_multiplier_property.is_valid()) return;

	for (size_t i = startId; i < endId; i++)
	{
		p_data->scale[i] = size_property->get_value(p_data->rng, p_data->current_duration_normalized) * size_multiplier_property->get_value(p_data->rng, p_data->current_duration_normalized);
	}
}

void SizeGenerator::set_size_multiplier_property(const Ref<SpicyFloatProperty>& p_size_multiplier_property)
{
	size_multiplier_property = p_size_multiplier_property;

	if (size_multiplier_property.is_null()) {
		Ref<SpicyFloatProperty> size_multiplier_p(memnew(SpicyFloatProperty));
		size_multiplier_property = size_multiplier_p;
		size_multiplier_property->set_available_types(SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_UNIFORM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_RANDOM | SpicyFloatProperty::SpicyFloatType::SPICY_FLOAT_TYPE_CURVE);
		size_multiplier_property->set_property_name("size_multiplier");
		size_multiplier_property->set_default_uniform(1);
		size_multiplier_property->set_default_random(Vector2(0.5, 1));
	}
}

Ref<SpicyFloatProperty> SizeGenerator::get_size_multiplier_property() const
{
	return size_multiplier_property;
}

void SizeGenerator::set_size_property(const Ref<SpicyVector3Property>& p_size_property)
{
	size_property = p_size_property;

	if (size_property.is_null()) {
		Ref<SpicyVector3Property> size_p(memnew(SpicyVector3Property));
		size_property = size_p;
		size_property->set_available_types(SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_UNIFORM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_RANDOM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_CURVE | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_CURVE_XYZ);
		size_property->set_property_name("size");
		size_property->set_default_uniform(Vector3(1, 1, 1));
		size_property->set_default_random(Vector3(0.5, 0.5, 0.5), Vector3(1, 1, 1));
	}
}

Ref<SpicyVector3Property> SizeGenerator::get_size_generator() const
{
	return size_property;
}

void SizeGenerator::_get_property_list(List<PropertyInfo>* r_props) const
{
	if (size_multiplier_property.is_valid()) {
		size_multiplier_property->get_property_list(r_props);
	}
	if (size_property.is_valid()) {
		size_property->get_property_list(r_props);
	}
}

bool SizeGenerator::_get(const StringName& p_property, Variant& r_value) const
{
	bool result = false;

	if (size_multiplier_property.is_valid()) {
		result = result || size_multiplier_property->get_property(p_property, r_value);
	}
	if (size_property.is_valid()) {
		result = result || size_property->get_property(p_property, r_value);
	}
	return result;
}

bool SizeGenerator::_set(const StringName& p_property, const Variant& p_value)
{
	bool result = false;

	if (size_multiplier_property.is_valid()) {
		result = result || size_multiplier_property->set_property(p_property, p_value, Ref<Resource>(this));
	}
	if (size_property.is_valid()) {
		result = result || size_property->set_property(p_property, p_value, Ref<Resource>(this));
	}
	return result;
}

bool SizeGenerator::_property_can_revert(const StringName& p_name) const
{
	bool result = false;

	if (size_multiplier_property.is_valid()) {
		result = result || size_multiplier_property->can_property_revert(p_name);
	}
	if (size_property.is_valid()) {
		result = result || size_property->can_property_revert(p_name);
	}
	return result;
}

bool SizeGenerator::_property_get_revert(const StringName& p_name, Variant& r_property) const
{
	bool result = false;

	if (size_multiplier_property.is_valid()) {
		result = result || size_multiplier_property->get_property_revert(p_name, r_property);
	}
	if (size_property.is_valid()) {
		result = result || size_property->get_property_revert(p_name, r_property);
	}
	return result;
}

//////////////////////////////////////////////////////////////////////////

void VelocityGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_velocity_over_life_property", "velocity_over_life"), &VelocityGenerator::set_velocity_property);
	ClassDB::bind_method(D_METHOD("get_velocity_over_life_property"), &VelocityGenerator::get_velocity_property);
}

VelocityGenerator::VelocityGenerator()
{
	Ref<SpicyVector3Property> velocity_p(memnew(SpicyVector3Property));
	velocity_property = velocity_p;
	velocity_property->set_available_types(SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_UNIFORM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_RANDOM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_CURVE_XYZ);
	velocity_property->set_property_name("velocity");
	velocity_property->set_default_uniform(Vector3(0, 0, 0));
	velocity_property->set_default_random(Vector3(-1, -1, -1), Vector3(1, 1, 1));
}

void VelocityGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	if (!velocity_property.is_valid()) return;

	for (size_t i = startId; i < endId; i++)
	{
		p_data->velocity[i] = velocity_property->get_value(p_data->rng, p_data->current_duration_normalized);
	}
}

void VelocityGenerator::set_velocity_property(const Ref<SpicyVector3Property>& p_velocity_property)
{
	velocity_property = p_velocity_property;

	if (velocity_property.is_null()) {
		Ref<SpicyVector3Property> velocity_p(memnew(SpicyVector3Property));
		velocity_property = velocity_p;
	}
}

Ref<SpicyVector3Property> VelocityGenerator::get_velocity_property() const
{
	return velocity_property;
}

void VelocityGenerator::_get_property_list(List<PropertyInfo>* r_props) const
{
	if (velocity_property.is_valid()) {
		velocity_property->get_property_list(r_props);
	}
}

bool VelocityGenerator::_get(const StringName& p_property, Variant& r_value) const
{
	if (velocity_property.is_valid()) {
		return velocity_property->get_property(p_property, r_value);
	}
	return false;
}

bool VelocityGenerator::_set(const StringName& p_property, const Variant& p_value)
{
	if (velocity_property.is_valid()) {
		return velocity_property->set_property(p_property, p_value, Ref<Resource>(this));
	}
	return false;
}

bool VelocityGenerator::_property_can_revert(const StringName& p_name) const
{
	if (velocity_property.is_valid()) {
		return velocity_property->can_property_revert(p_name);
	}
	return false;
}

bool VelocityGenerator::_property_get_revert(const StringName& p_name, Variant& r_property) const
{
	if (velocity_property.is_valid()) {
		return velocity_property->get_property_revert(p_name, r_property);
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////

void RotationGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_rotation_property", "rotation_property"), &RotationGenerator::set_rotation_property);
	ClassDB::bind_method(D_METHOD("get_rotation_property"), &RotationGenerator::get_rotation_property);
}

RotationGenerator::RotationGenerator()
{
	Ref<SpicyVector3Property> rotation_p(memnew(SpicyVector3Property));
	rotation_property = rotation_p;
	rotation_property->set_available_types(SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_UNIFORM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_RANDOM | SpicyVector3Property::SpicyVector3Type::SPICY_VECTOR_TYPE_CURVE_XYZ);
	rotation_property->set_property_name("rotation");
	rotation_property->set_default_uniform(Vector3(0, 0, 0));
	rotation_property->set_default_random(Vector3(-180, -180, -180), Vector3(180, 180, 180));
}

void RotationGenerator::generate(double dt, const Ref<ParticleData> p_data, size_t startId, size_t endId)
{
	if (!rotation_property.is_valid()) return;

	for (size_t i = startId; i < endId; i++)
	{
		p_data->rotation[i] = rotation_property->get_value(p_data->rng, p_data->current_duration_normalized);
	}
}

void RotationGenerator::set_rotation_property(const Ref<SpicyVector3Property>& p_rotation_property)
{
	rotation_property = p_rotation_property;

	if (rotation_property.is_null()) {
		Ref<SpicyVector3Property> rotation_p(memnew(SpicyVector3Property));
		rotation_property = rotation_p;
	}
}

Ref<SpicyVector3Property> RotationGenerator::get_rotation_property() const
{
	return rotation_property;
}

void RotationGenerator::_get_property_list(List<PropertyInfo>* r_props) const
{
	if (rotation_property.is_valid()) {
		rotation_property->get_property_list(r_props);
	}
}

bool RotationGenerator::_get(const StringName& p_property, Variant& r_value) const
{
	if (rotation_property.is_valid()) {
		return rotation_property->get_property(p_property, r_value);
	}
	return false;
}

bool RotationGenerator::_set(const StringName& p_property, const Variant& p_value)
{
	if (rotation_property.is_valid()) {
		return rotation_property->set_property(p_property, p_value, Ref<Resource>(this));
	}
	return false;
}

bool RotationGenerator::_property_can_revert(const StringName& p_name) const
{
	if (rotation_property.is_valid()) {
		return rotation_property->can_property_revert(p_name);
	}
	return false;
}

bool RotationGenerator::_property_get_revert(const StringName& p_name, Variant& r_property) const
{
	if (rotation_property.is_valid()) {
		return rotation_property->get_property_revert(p_name, r_property);
	}
	return false;
}