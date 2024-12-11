#pragma once
#include "../common.h"
#include "misc/type_conversions.hpp"
#include "containers/hash_set.hpp"
#include "containers/local_vector.hpp"
#include "containers/inline_vector.hpp"
#include "containers/hash_map.hpp"
class JoltShapedObjectImpl3D;
class JoltSpace3D;

class JoltContactListener3D final
	: public JPH::ContactListener
	, public JPH::SoftBodyContactListener {
	using Mutex = std::mutex;

	using MutexLock = std::unique_lock<Mutex>;

	struct BodyIDHasher {
		static uint32_t hash(const JPH::BodyID& p_id) {
			return hash_fmix32(p_id.GetIndexAndSequenceNumber());
		}
		static bool compare(const JPH::BodyID& p_lhs, const JPH::BodyID& p_rhs) {
			return p_lhs == p_rhs;
		}
	};

	struct ShapePairHasher {
		static uint32_t hash(const JPH::SubShapeIDPair& p_pair) {
			uint32_t hash = hash_murmur3_one_32(p_pair.GetBody1ID().GetIndexAndSequenceNumber());
			hash = hash_murmur3_one_32(p_pair.GetSubShapeID1().GetValue(), hash);
			hash = hash_murmur3_one_32(p_pair.GetBody2ID().GetIndexAndSequenceNumber(), hash);
			hash = hash_murmur3_one_32(p_pair.GetSubShapeID2().GetValue(), hash);
			return hash_fmix32(hash);
		}
		static bool compare(const JPH::SubShapeIDPair& p_lhs, const JPH::SubShapeIDPair& p_rhs) {
			return p_lhs == p_rhs;
		}
	};

	struct Contact {
		JPH::Vec3 normal = {};

		JPH::RVec3 point_self = {};

		JPH::RVec3 point_other = {};

		JPH::Vec3 velocity_self = {};

		JPH::Vec3 velocity_other = {};

		JPH::Vec3 impulse = {};
	};

	using Contacts = JLocalVector<Contact>;

	struct Manifold {
		Contacts contacts1;

		Contacts contacts2;

		float depth = 0.0f;
	};

	using BodyIDs = JHashSet<JPH::BodyID, BodyIDHasher,BodyIDHasher>;

	using Overlaps = JHashSet<JPH::SubShapeIDPair, ShapePairHasher,ShapePairHasher>;

	using ManifoldsByShapePair = HashMap<JPH::SubShapeIDPair, Manifold, ShapePairHasher>;

public:
	explicit JoltContactListener3D(JoltSpace3D* p_space)
		: space(p_space) { }

	void listen_for(JoltShapedObjectImpl3D* p_object);

	void pre_step();

	void post_step();

#ifdef TOOLS_ENABLED
	const PackedVector3Array& get_debug_contacts() const { return debug_contacts; }

	int32_t get_debug_contact_count() const { return debug_contact_count; }

	int32_t get_max_debug_contacts() const { return (int32_t)debug_contacts.size(); }

	void set_max_debug_contacts(int32_t p_count) { debug_contacts.resize(p_count); }
#endif // TOOLS_ENABLED

private:
	void OnContactAdded(
		const JPH::Body& p_body1,
		const JPH::Body& p_body2,
		const JPH::ContactManifold& p_manifold,
		JPH::ContactSettings& p_settings
	) override;

	void OnContactPersisted(
		const JPH::Body& p_body1,
		const JPH::Body& p_body2,
		const JPH::ContactManifold& p_manifold,
		JPH::ContactSettings& p_settings
	) override;

	void OnContactRemoved(const JPH::SubShapeIDPair& p_shape_pair) override;

	JPH::SoftBodyValidateResult OnSoftBodyContactValidate(
		const JPH::Body& p_soft_body,
		const JPH::Body& p_other_body,
		JPH::SoftBodyContactSettings& p_settings
	) override;

#ifdef TOOLS_ENABLED
	void OnSoftBodyContactAdded(
		const JPH::Body& p_soft_body,
		const JPH::SoftBodyManifold& p_manifold
	) override;
#endif // TOOLS_ENABLED

	bool _is_listening_for(const JPH::Body& p_body) const;

	bool _try_override_collision_response(
		const JPH::Body& p_jolt_body1,
		const JPH::Body& p_jolt_body2,
		JPH::ContactSettings& p_settings
	);

	bool _try_override_collision_response(
		const JPH::Body& p_jolt_soft_body,
		const JPH::Body& p_jolt_other_body,
		JPH::SoftBodyContactSettings& p_settings
	);

	bool _try_apply_surface_velocities(
		const JPH::Body& p_jolt_body1,
		const JPH::Body& p_jolt_body2,
		JPH::ContactSettings& p_settings
	);

	bool _try_add_contacts(
		const JPH::Body& p_body1,
		const JPH::Body& p_body2,
		const JPH::ContactManifold& p_manifold,
		JPH::ContactSettings& p_settings
	);

	bool _try_evaluate_area_overlap(
		const JPH::Body& p_body1,
		const JPH::Body& p_body2,
		const JPH::ContactManifold& p_manifold
	);

	bool _try_remove_contacts(const JPH::SubShapeIDPair& p_shape_pair);

	bool _try_remove_area_overlap(const JPH::SubShapeIDPair& p_shape_pair);

#ifdef TOOLS_ENABLED
	bool _try_add_debug_contacts(
		const JPH::Body& p_body1,
		const JPH::Body& p_body2,
		const JPH::ContactManifold& p_manifold
	);

	bool _try_add_debug_contacts(
		const JPH::Body& p_soft_body,
		const JPH::SoftBodyManifold& p_manifold
	);
#endif // TOOLS_ENABLED

	void _flush_contacts();

	void _flush_area_enters();

	void _flush_area_shifts();

	void _flush_area_exits();

	ManifoldsByShapePair manifolds_by_shape_pair;

	BodyIDs listening_for;

	Overlaps area_overlaps;

	Overlaps area_enters;

	Overlaps area_exits;

	Mutex write_mutex;

	JoltSpace3D* space = nullptr;

#ifdef TOOLS_ENABLED
	PackedVector3Array debug_contacts;

	std::atomic<int32_t> debug_contact_count = 0;
#endif // TOOLS_ENABLED
};
