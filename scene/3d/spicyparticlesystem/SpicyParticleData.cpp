#include "SpicyParticleData.h"


void ParticleData::delete_data()
{
	initialized = false;

	memdelete_arr(position);
	memdelete_arr(rotation);
	memdelete_arr(scale);
	memdelete_arr(velocity);
	memdelete_arr(current_velocity);
	memdelete_arr(current_scale);
	memdelete_arr(current_rotation);
	memdelete_arr(acceleration);
	memdelete_arr(custom_data);
	memdelete_arr(color);
	memdelete_arr(current_color);
	memdelete_arr(lifetime);
	memdelete_arr(life_remaining);
	memdelete_arr(normalized_lifetime);
	memdelete_arr(alive);
}

ParticleData::~ParticleData()
{
	if (!initialized)
		return;
	delete_data();
}

void ParticleData::generate(size_t max_size, const Ref<RandomNumberGenerator>& p_rng)
{
	particle_count = max_size;
	count_alive = 0;

	if (initialized) {
		delete_data();
	}

	position = memnew_arr(Vector3, max_size);
	rotation = memnew_arr(Vector3, max_size);
	scale = memnew_arr(Vector3, max_size);
	velocity = memnew_arr(Vector3, max_size);
	current_velocity = memnew_arr(Vector3, max_size);
	current_scale = memnew_arr(Vector3, max_size);
	current_rotation = memnew_arr(Vector3, max_size);
	acceleration = memnew_arr(Vector3, max_size);
	custom_data = memnew_arr(Vector4, max_size);
	current_color = memnew_arr(Color, max_size);
	color = memnew_arr(Color, max_size);
	lifetime = memnew_arr(float, max_size);
	life_remaining = memnew_arr(float, max_size);
	normalized_lifetime = memnew_arr(float, max_size);
	alive = memnew_arr(bool, max_size);

	rng = p_rng;

	initialized = true;
}

void ParticleData::kill(size_t id)
{
	if (count_alive > 0)
	{
		alive[id] = false;
		swap(id, count_alive - 1);
		count_alive--;
	}
}

void ParticleData::wake(size_t id)
{
	if (count_alive < particle_count)
	{
		alive[id] = true;
		swap(id, count_alive);
		count_alive++;
	}
}


void ParticleData::swap(size_t id1, size_t id2)
{
	std::swap(position[id1], position[id2]);
	std::swap(rotation[id1], rotation[id2]);
	std::swap(scale[id1], scale[id2]);
	std::swap(velocity[id1], velocity[id2]);
	std::swap(current_velocity[id1], current_velocity[id2]);
	std::swap(current_scale[id1], current_scale[id2]);
	std::swap(current_rotation[id1], current_rotation[id2]);
	std::swap(acceleration[id1], acceleration[id2]);
	std::swap(custom_data[id1], custom_data[id2]);
	std::swap(color[id1], color[id2]);
	std::swap(current_color[id1], current_color[id2]);
	std::swap(lifetime[id1], lifetime[id2]);
	std::swap(life_remaining[id1], life_remaining[id2]);
	std::swap(normalized_lifetime[id1], normalized_lifetime[id2]);
	std::swap(alive[id1], alive[id2]);
}

void ParticleData::get_transform(size_t id, Transform3D& p_transform) const
{
	Vector3 p_euler_degrees = rotation[id] + current_rotation[id];
	Vector3 radians(Math::deg_to_rad(p_euler_degrees.x), Math::deg_to_rad(p_euler_degrees.y), Math::deg_to_rad(p_euler_degrees.z));

	Quaternion q = Quaternion::from_euler(radians);

	p_transform.scale(scale[id] * current_scale[id]);
	p_transform.basis.rotate(q);
	p_transform.origin = position[id];
	p_transform.origin = (emitter_transform.affine_inverse() * p_transform).origin;
}

void ParticleData::get_color(size_t id, Color& p_color) const
{
	p_color = color[id] * current_color[id];
}

size_t ParticleData::get_last_alive_index()
{
	return count_alive < particle_count ? count_alive : particle_count;
}
