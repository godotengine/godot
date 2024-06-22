#ifndef SPICYPARTICLERENDERER_H
#define SPICYPARTICLERENDERER_H

#include "SpicyParticleSystem.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/camera_3d.h"

//class ParticleRenderer : public RefCounted
//{ 	GDCLASS(ParticleRenderer, RefCounted)
//protected:
//	static void _bind_methods() { }
//public:
//	ParticleRenderer() = default;
//	virtual ~ParticleRenderer() { }

//	virtual void generate(SpicyParticleSystem* p_system) = 0;
//	virtual void destroy() = 0;
//	virtual void update() = 0;
//	virtual void render() = 0;
//	virtual void reset() = 0;
//};

class MultiMeshParticleRenderer : public RefCounted
{
	GDCLASS(MultiMeshParticleRenderer, RefCounted)
public:
	enum Alignment
	{
		ALIGNMENT_LOCAL,
		ALIGNMENT_WORLD,
		ALIGNMENT_SCREEN,
		ALIGNMENT_CAMERA,
		ALIGNMENT_VELOCITY,
		ALIGNMENT_LOOK_AT,
		ALIGNMENT_MAX
	};
private:
	RID multimesh;
	static const int64_t mesh_data_size = 20;
	PackedFloat32Array mesh_data;

	Ref<SpicyParticleSystem> m_system;
	Alignment m_alignment = ALIGNMENT_LOCAL;
	const Node3D* m_alignment_target_node;
	Camera3D* m_camera;

	const Basis flip_xz;

protected:
	static void _bind_methods();
public:
	inline MultiMeshParticleRenderer() : flip_xz(Basis(Vector3(-1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, -1))) {}
	virtual ~MultiMeshParticleRenderer();

	void set_mesh(const Ref<Mesh>& p_mesh);

	RID get_multimesh() const;

	void apply_alignment(const Ref<ParticleData> p_data, size_t p_id, Transform3D& out_transform);
	void set_alignment(Alignment p_alignment);
	void set_alignment_target(const Node3D* p_alignment_target_node);

	virtual void generate(const Ref<SpicyParticleSystem> p_system, const Ref<Mesh>& p_mesh);
	virtual void destroy();
	virtual void update();
	virtual void render();
	virtual void reset();
	virtual void resize_buffer();
};

VARIANT_ENUM_CAST(MultiMeshParticleRenderer::Alignment)


#endif // SPICYPARTICLERENDERER_H