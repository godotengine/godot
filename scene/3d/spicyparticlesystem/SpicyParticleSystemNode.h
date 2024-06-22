#ifndef SpicyParticleSystemNode_H
#define SpicyParticleSystemNode_H

#include "SpicyParticleSystem.h"
#include "SpicyParticleUpdater.h"
#include "SpicyParticleRenderer.h"
#include "SpicyParticleGenerator.h"

#include "core/math/random_number_generator.h"
#include "scene/resources/mesh.h"
#include "scene/3d/visual_instance_3d.h"


// Helper macro to use with PROPERTY_HINT_ARRAY_TYPE for arrays of specific resources:
// PropertyInfo(Variant::ARRAY, "fallbacks", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("Font")
#ifndef MAKE_RESOURCE_TYPE_HINT
#define MAKE_RESOURCE_TYPE_HINT(m_type) vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, m_type)
#endif // !MAKE_RESOURCE_TYPE_HINT


class SpicyParticleSystemNode : public GeometryInstance3D {
	GDCLASS(SpicyParticleSystemNode, GeometryInstance3D)
private:
	const double SIMULATION_DELTA = 0.016667; //60fps
	bool initialized;
	real_t max_particles;
	bool _render = true;

	//Particle system
	Ref<SpicyParticleSystem> m_particle_system;
	Ref<MultiMeshParticleRenderer> m_renderer;

	//Modules
	Ref<SpicyParticleEmitter> m_emitter;
	Ref<LifetimeGenerator> m_lifetime_generator;
	Ref<PositionGenerator> m_position_generator;
	Ref<ColorGenerator> m_color_generator;
	Ref<SizeGenerator> m_size_generator;
	Ref<RotationGenerator> m_rotation_generator;
	Ref<VelocityGenerator> m_velocity_generator;

	Ref<ColorUpdater> m_color_updater;
	Ref<VelocityUpdater> m_velocity_updater;
	Ref<AccelerationUpdater> m_acceleration_updater;
	Ref<RotationUpdater> m_rotation_updater;
	Ref<SizeUpdater> m_size_updater;
	Ref<CustomDataUpdater> m_custom_data_updater;

	Ref<RandomNumberGenerator> rng;
	uint64_t rng_state;

	bool is_paused;
	bool is_playing;
	bool is_stopped;

	//Inspector properties
	bool looping;
	bool randomize_seed;;
	unsigned int seed;
	bool world_space;
	bool play_on_start;
	real_t emit_rate;
	real_t emit_rate_over_distance;;
	TypedArray<SpicyParticleBurst> m_emit_bursts;

	real_t duration;
	real_t delay;
	real_t simulation_speed;
	double simulation_time;
	double normalized_duration_time;

	Ref<Mesh> mesh;
	MultiMeshParticleRenderer::Alignment m_particle_alignment;
	NodePath m_alignment_target;
	bool use_custom_data;

	Transform3D node_transform;
protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo& property) const;
	void _update_null_properties();
	//void _update_burst_times();
	void _notification(int p_what);
	void _stop_no_signal();
public:
	SpicyParticleSystemNode();
	virtual ~SpicyParticleSystemNode();
	PackedStringArray get_configuration_warnings() const;
	void _internal_process(double delta);
	void _set_render(bool p_render, bool include_children = true);

	void initialize(size_t max_count);
	void emit_burst(int count);
	void seek(double t, bool include_children = true);

	void pause(bool include_children = true);
	void play(bool include_children = true);
	void stop(bool include_children = true);
	void restart(bool include_children = true);
	bool get_is_playing();

	void set_emit_rate(float p_emit_rate);
	float get_emit_rate() const;

	void set_emit_rate_over_distance(float p_emit_rate_over_distance);
	float get_emit_rate_over_distance() const;

	void set_emit_bursts(const TypedArray<SpicyParticleBurst>& p_emit_bursts);
	TypedArray<SpicyParticleBurst> get_emit_bursts() const;

	void set_duration(float p_duration);
	float get_duration() const;

	void set_delay(float p_preprocess_time);
	float get_delay() const;

	void set_looping(bool p_looping);
	bool get_looping() const;

	void set_world_space(bool p_world_space);
	bool get_world_space() const;

	void set_play_on_start(bool p_play_on_start);
	bool get_play_on_start() const;

	void set_randomize_seed(bool p_randomize_seed);
	bool get_randomize_seed() const;

	void set_simulation_time(double p_simulation_time);
	double get_simulation_time() const;

	void set_simulation_speed(double p_simulation_speed);
	double get_simulation_speed() const;

	void set_seed(unsigned int p_seed);
	unsigned int get_seed() const;

	//Lifetime generation
	void set_lifetime_generator(const Ref<LifetimeGenerator>& p_lifetime_generator);
	Ref<LifetimeGenerator> get_lifetime_generator() const;

	//Emission shape generation
	void set_position_generator(const Ref<PositionGenerator>& p_position_generator);
	Ref<PositionGenerator> get_position_generator() const;

	//Color generation
	void set_color_generator(const Ref<ColorGenerator>& p_color_generator);
	Ref<ColorGenerator> get_color_generator() const;

	//Scale generation
	void set_size_generator(const Ref<SizeGenerator>& p_size_generator);
	Ref<SizeGenerator> get_size_generator() const;

	//Rotation generation
	void set_rotation_generator(const Ref<RotationGenerator>& p_rotation_generator);
	Ref<RotationGenerator> get_rotation_generator() const;

	//Velocity generation
	void set_velocity_generator(const Ref<VelocityGenerator>& p_velocity_generator);
	Ref<VelocityGenerator> get_velocity_generator() const;

	//Color updater
	void set_color_updater(const Ref<ColorUpdater>& p_color_updater);
	Ref<ColorUpdater> get_color_updater() const;

	//Velocity updater
	void set_velocity_updater(const Ref<VelocityUpdater>& p_velocity_updater);
	Ref<VelocityUpdater> get_velocity_updater() const;

	//Acceleration updater
	void set_acceleration_updater(const Ref<AccelerationUpdater>& p_acceleration_updater);
	Ref<AccelerationUpdater> get_acceleration_updater() const;

	//Rotation updater
	void set_rotation_updater(const Ref<RotationUpdater>& p_rotation_updater);
	Ref<RotationUpdater> get_rotation_updater() const;

	//Scale updater
	void set_size_updater(const Ref<SizeUpdater>& p_size_updater);
	Ref<SizeUpdater> get_size_updater() const;		
	
	void set_custom_data_updater(const Ref<CustomDataUpdater>& p_custom_data_updater);
	Ref<CustomDataUpdater> get_custom_data_updater() const;

	// Render properties
	void set_mesh(const Ref<Mesh>& p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_particle_alignment(MultiMeshParticleRenderer::Alignment p_particle_alignment);
	MultiMeshParticleRenderer::Alignment get_particle_alignment() const;

	void set_alignment_target(const NodePath& p_path);
	NodePath get_alignment_target() const;

	void set_use_custom_data(bool p_use_custom_data);
	bool get_use_custom_data() const;

	void set_max_particle_count(int p_max_particle_count);
	int get_max_particle_count() const;

	//info
	int get_particle_count() const;
};



#endif // !SpicyParticleSystemNode_H