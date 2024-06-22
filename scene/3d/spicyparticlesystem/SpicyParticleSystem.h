# ifndef SPICYPARTICLESYSTEM_H
# define SPICYPARTICLESYSTEM_H

#include "SpicyParticleData.h"
#include "SpicyParticleGenerator.h"
#include "SpicyParticleUpdater.h"

#include "core/io/resource.h"



class SpicyParticleBurst : public Resource {
	GDCLASS(SpicyParticleBurst, Resource);
public:
	float time;
	int count;
protected:
	static void _bind_methods();
public:
	inline SpicyParticleBurst() : time(0), count(0) { }
	~SpicyParticleBurst() { }

	//inline void set_time(float p_time) { time = CLAMP(p_time, 0, max_time); normalized_time = CLAMP(time / max_time, 0.082, 1.0); }
	inline void set_time(float p_time) { time = MAX(0, p_time); /*normalized_time = CLAMP(time / max_time, 0.0, 1.0);*/ }
	inline float get_time() const { return time; }

	inline void set_count(int p_count) { count = MAX(0, p_count); }
	inline int get_count() const { return count; }
};

class SpicyParticleEmitter : public RefCounted
{
	GDCLASS(SpicyParticleEmitter, RefCounted)
private:
	float emit_rate;
	float emit_rate_over_distance;
	float e_time_accumulator;
	float burst_time_accumulator;

	Vector3 prev_position;
	float distance_traveled;

	bool* bursts_emit_state;
protected:
	TypedArray<SpicyParticleBurst> m_bursts;
	Vector<Ref<SpicyParticleBurst>> m_bursts_to_emit;
	Vector<Ref<SpicyParticleGenerator>> m_generators;
	static void _bind_methods();
public:
	inline SpicyParticleEmitter() : emit_rate(0),
		emit_rate_over_distance(0),
		e_time_accumulator(0),
		burst_time_accumulator(0),
		prev_position(Vector3(0, 0, 0)),
		distance_traveled(0) { }

	~SpicyParticleEmitter();

	void emit(double dt, const Ref<ParticleData> p_data);
	void emit_burst(double dt, const Ref<ParticleData> p_data, int count);
	void reset();
	inline void set_emit_rate(float p_emit_rate) { emit_rate = 1.0 / p_emit_rate; }
	inline void set_emit_rate_over_distance(float p_emit_rate_over_distance) { emit_rate_over_distance = p_emit_rate_over_distance; }

	void add_generator(const Ref<SpicyParticleGenerator>& generator);
	void remove_generator(const Ref<SpicyParticleGenerator>& generator);

	void set_bursts(const TypedArray<SpicyParticleBurst>& bursts);
	void reset_bursts();
	bool sort_bursts(const Ref<SpicyParticleBurst>& a, const Ref<SpicyParticleBurst>& b);
	inline void set_prev_position(const Vector3& p_prev_position) { prev_position = p_prev_position; }
};

class SpicyParticleSystem : public RefCounted
{
	GDCLASS(SpicyParticleSystem, RefCounted)
private:
	bool is_emitting;
protected:
	Ref<ParticleData> particles;

	size_t count;

	Vector<Ref<SpicyParticleEmitter>> emitters;
	Vector<Ref<SpicyParticleUpdater>> updaters;

protected:
	static void _bind_methods() {}
public:
	SpicyParticleSystem();
	~SpicyParticleSystem();

	SpicyParticleSystem(const SpicyParticleSystem&) = delete;

	void initialize(size_t max_count, const Ref<RandomNumberGenerator>& p_rng, const Node3D& particle_node);
	void set_max_particle_count(size_t max_count, const Ref<RandomNumberGenerator>& p_rng);
	void update(double dt, double current_duration_normalized, const Transform3D& node_transform);
	void reset();
	void delete_particle_data();
	void emit(double dt, int count);

	void set_emitting(bool p_emitting);

	inline size_t num_all_particles() const { return particles->particle_count; }
	inline size_t num_alive_particles() const { return particles->count_alive; }

	inline void add_emitter(const Ref<SpicyParticleEmitter> emitter) { emitters.push_back(emitter); }
	inline void add_updater(const Ref<SpicyParticleUpdater>& updater) { updaters.insert(0, updater); }
	inline void remove_updater(const Ref<SpicyParticleUpdater>& updater) { updaters.erase(updater); }

	inline const Ref<ParticleData> final_data() const { return particles; }
};


#endif