#pragma once
#include "../common.h"
#include "misc/error_macros.hpp"
#include "containers/hash_map.hpp"
#include "containers/hash_set.hpp"
#include "containers/local_vector.hpp"
#include "containers/inline_vector.hpp"
class JoltLayerMapper final
	: public JPH::BroadPhaseLayerInterface
	, public JPH::ObjectLayerPairFilter
	, public JPH::ObjectVsBroadPhaseLayerFilter {
public:
	JoltLayerMapper();

	JPH::ObjectLayer to_object_layer(
		JPH::BroadPhaseLayer p_broad_phase_layer,
		uint32_t p_collision_layer,
		uint32_t p_collision_mask
	);

	void from_object_layer(
		JPH::ObjectLayer p_encoded_layer,
		JPH::BroadPhaseLayer& p_broad_phase_layer,
		uint32_t& p_collision_layer,
		uint32_t& p_collision_mask
	) const;

private:
	uint32_t GetNumBroadPhaseLayers() const override;

	JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer p_layer) const override;

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
	const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer p_layer) const override;
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

	bool ShouldCollide(JPH::ObjectLayer p_encoded_layer1, JPH::ObjectLayer p_encoded_layer2)
		const override;

	bool ShouldCollide(JPH::ObjectLayer p_encoded_layer1, JPH::BroadPhaseLayer p_broad_phase_layer2)
		const override;

	JPH::ObjectLayer _allocate_object_layer(uint64_t p_collision);

	InlineVector<uint64_t, 32> collisions_by_layer;

	JHashMap<uint64_t, JPH::ObjectLayer> layers_by_collision;

	JPH::ObjectLayer next_object_layer = 0;
};
