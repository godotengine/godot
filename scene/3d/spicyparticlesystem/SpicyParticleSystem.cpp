#include "SpicyParticleSystem.h"



////////////////////////////////////////////////////////////////////////////////
// ParticleBurst
void SpicyParticleBurst::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("get_time"), &SpicyParticleBurst::get_time);
	ClassDB::bind_method(D_METHOD("set_time", "time"), &SpicyParticleBurst::set_time);
	ClassDB::bind_method(D_METHOD("get_count"), &SpicyParticleBurst::get_count);
	ClassDB::bind_method(D_METHOD("set_count", "count"), &SpicyParticleBurst::set_count);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time"), "set_time", "get_time");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "count"), "set_count", "get_count");
}


////////////////////////////////////////////////////////////////////////////////
// SpicyParticleEmitter


void SpicyParticleEmitter::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("sort_bursts", "a", "b"), &SpicyParticleEmitter::sort_bursts);
}

void SpicyParticleEmitter::emit(double dt, const Ref<ParticleData> p_data)
{
	int to_emit = 0;
	e_time_accumulator += dt;
	while (e_time_accumulator > emit_rate)
	{
		e_time_accumulator -= emit_rate;
		to_emit++;
	}

	if (emit_rate_over_distance > 0)
	{
		float distance = p_data->particle_node->get_global_position().distance_to(prev_position);// *emit_rate_over_distance;

		distance_traveled += distance;
		float rate = distance_traveled * emit_rate_over_distance;

		if (rate >= 1.0f) {
			to_emit += rate;
			distance_traveled = 0;
		}

		prev_position = p_data->particle_node->get_global_position();
	}

	bool reset_burst_time = false;

	if (burst_time_accumulator >= p_data->max_duration)
	{
		reset_burst_time = true;
	}

	for (int i = 0; i < m_bursts.size(); i++)
	{
		Ref<SpicyParticleBurst> b = m_bursts[i];

		if (b->time <= burst_time_accumulator)
		{
			if (!bursts_emit_state[i])
			{
				bursts_emit_state[i] = true;
				to_emit += b->get_count();
			}
		}
	}

	burst_time_accumulator += dt;

	if (reset_burst_time)
	{
		reset_bursts();
		burst_time_accumulator = 0;
	}

	emit_burst(dt, p_data, to_emit);
}
void SpicyParticleEmitter::emit_burst(double dt, const Ref<ParticleData> p_data, int count)
{
	const size_t startId = p_data->count_alive;
	const size_t endId = std::min(startId + count, p_data->particle_count - 1);

	for (auto& generator : m_generators)
	{
		generator->generate(dt, p_data, startId, endId);
	}

	for (size_t i = startId; i < endId; ++i)
	{
		p_data->wake(i);
	}
}

void SpicyParticleEmitter::reset()
{
	e_time_accumulator = 0.0f;
	burst_time_accumulator = 0.0f;
	reset_bursts();
}

void SpicyParticleEmitter::add_generator(const Ref<SpicyParticleGenerator>& generator)
{
	m_generators.push_back(generator);
}

void SpicyParticleEmitter::remove_generator(const Ref<SpicyParticleGenerator>& generator)
{
	m_generators.erase(generator);
}

void SpicyParticleEmitter::set_bursts(const TypedArray<SpicyParticleBurst>& bursts)
{
	m_bursts = bursts;

	reset_bursts();
}

void SpicyParticleEmitter::reset_bursts()
{
	bursts_emit_state = memnew_arr(bool, m_bursts.size());
	for (size_t i = 0; i < m_bursts.size(); i++)
	{
		bursts_emit_state[i] = false;
	}
}

bool SpicyParticleEmitter::sort_bursts(const Ref<SpicyParticleBurst>& a, const Ref<SpicyParticleBurst>& b)
{
	return a->get_time() < b->get_time();
}

SpicyParticleEmitter::~SpicyParticleEmitter()
{
	if (bursts_emit_state != nullptr)
		memdelete_arr(bursts_emit_state);
}

////////////////////////////////////////////////////////////////////////////////
// ParticleSystem

SpicyParticleSystem::SpicyParticleSystem() : is_emitting(false), count(0)
{
	particles = Ref<ParticleData>(memnew(ParticleData));
}

SpicyParticleSystem::~SpicyParticleSystem()
{

}

void SpicyParticleSystem::initialize(size_t max_count, const Ref<RandomNumberGenerator>& p_rng, const Node3D& particle_node)
{
	count = max_count;
	particles->generate(max_count, p_rng);
	particles->particle_node = &particle_node;
}


void SpicyParticleSystem::set_max_particle_count(size_t max_count, const Ref<RandomNumberGenerator>& p_rng)
{
	count = max_count;
	particles->generate(max_count, p_rng);
}

void SpicyParticleSystem::delete_particle_data()
{
	particles->delete_data();
}

void SpicyParticleSystem::update(double dt, double current_duration_normalized, const Transform3D& node_transform)
{
	particles->current_duration_normalized = current_duration_normalized;
	particles->emitter_transform = node_transform;

	if (is_emitting)
	{
		for (auto& emitter : emitters)
		{
			emitter->emit(dt, particles);
		}
	}

	for (auto& updater : updaters)
	{
		updater->update(dt, particles);
	}
}

void SpicyParticleSystem::reset()
{
	is_emitting = false;

	for (auto& emitter : emitters)
	{
		emitter->reset();
	}

	//kill all alive particles
	while (particles->count_alive > 0)
	{
		particles->kill(0);
	}
}

void SpicyParticleSystem::emit(double dt, int count)
{
	is_emitting = true;

	for (auto& emitter : emitters)
	{
		emitter->emit_burst(dt, particles, count);
	}

}

void SpicyParticleSystem::set_emitting(bool p_emitting)
{
	is_emitting = p_emitting;

	for (auto& emitter : emitters)
	{
		emitter->set_prev_position(particles->particle_node->get_global_position());
	}
}
