#include "jolt_layer_mapper.hpp"

#include "servers/jolt_project_settings.hpp"
#include "spaces/jolt_broad_phase_layer.hpp"

namespace {

template<uint8_t TSize = JoltBroadPhaseLayer::COUNT>
class JoltBroadPhaseMatrix {
	using LayerType = JPH::BroadPhaseLayer;
	using UnderlyingType = LayerType::Type;

public:
	JoltBroadPhaseMatrix() {
		using namespace JoltBroadPhaseLayer;

		allow_collision(BODY_STATIC, BODY_DYNAMIC);
		allow_collision(BODY_STATIC_BIG, BODY_DYNAMIC);
		allow_collision(BODY_DYNAMIC, BODY_STATIC);
		allow_collision(BODY_DYNAMIC, BODY_STATIC_BIG);
		allow_collision(BODY_DYNAMIC, BODY_DYNAMIC);
		allow_collision(BODY_DYNAMIC, AREA_DETECTABLE);
		allow_collision(BODY_DYNAMIC, AREA_UNDETECTABLE);
		allow_collision(AREA_DETECTABLE, BODY_DYNAMIC);
		allow_collision(AREA_DETECTABLE, AREA_DETECTABLE);
		allow_collision(AREA_DETECTABLE, AREA_UNDETECTABLE);
		allow_collision(AREA_UNDETECTABLE, BODY_DYNAMIC);
		allow_collision(AREA_UNDETECTABLE, AREA_DETECTABLE);

		if (JoltProjectSettings::areas_detect_static_bodies()) {
			allow_collision(BODY_STATIC, AREA_DETECTABLE);
			allow_collision(BODY_STATIC, AREA_UNDETECTABLE);
			allow_collision(BODY_STATIC_BIG, AREA_DETECTABLE);
			allow_collision(BODY_STATIC_BIG, AREA_UNDETECTABLE);
			allow_collision(AREA_DETECTABLE, BODY_STATIC);
			allow_collision(AREA_DETECTABLE, BODY_STATIC_BIG);
			allow_collision(AREA_UNDETECTABLE, BODY_STATIC);
			allow_collision(AREA_UNDETECTABLE, BODY_STATIC_BIG);
		}
	}

	void allow_collision(UnderlyingType p_layer1, UnderlyingType p_layer2) {
		masks[p_layer1] |= uint8_t(1U << p_layer2);
	}

	void allow_collision(LayerType p_layer1, LayerType p_layer2) {
		allow_collision((UnderlyingType)p_layer1, (UnderlyingType)p_layer2);
	}

	bool should_collide(UnderlyingType p_layer1, UnderlyingType p_layer2) const {
		return (masks[p_layer1] & uint8_t(1U << p_layer2)) != 0;
	}

	bool should_collide(LayerType p_layer1, LayerType p_layer2) const {
		return should_collide((UnderlyingType)p_layer1, (UnderlyingType)p_layer2);
	}

private:
	uint8_t masks[TSize] = {};
};

constexpr JPH::ObjectLayer encode_layers(
	JPH::BroadPhaseLayer p_broad_phase_layer,
	JPH::ObjectLayer p_object_layer
) {
	const auto upper_bits = uint16_t((uint8_t)p_broad_phase_layer << 13U);
	const auto lower_bits = uint16_t(p_object_layer);
	return JPH::ObjectLayer(upper_bits | lower_bits);
}

constexpr void decode_layers(
	JPH::ObjectLayer p_encoded_layers,
	JPH::BroadPhaseLayer& p_broad_phase_layer,
	JPH::ObjectLayer& p_object_layer
) {
	p_broad_phase_layer = JPH::BroadPhaseLayer(uint8_t(p_encoded_layers >> 13U));
	p_object_layer = JPH::ObjectLayer(p_encoded_layers & 0b0001'1111'1111'1111U);
}

constexpr uint64_t encode_collision(uint32_t p_collision_layer, uint32_t p_collision_mask) {
	const auto upper_bits = (uint64_t)p_collision_layer << 32U;
	const auto lower_bits = (uint64_t)p_collision_mask;
	return upper_bits | lower_bits;
}

constexpr void decode_collision(
	uint64_t p_collision,
	uint32_t& p_collision_layer,
	uint32_t& p_collision_mask
) {
	p_collision_layer = uint32_t(p_collision >> 32U);
	p_collision_mask = uint32_t(p_collision & 0xFFFFFFFFU);
}

} // namespace

JoltLayerMapper::JoltLayerMapper() {
	_allocate_object_layer(0);
}

JPH::ObjectLayer JoltLayerMapper::to_object_layer(
	JPH::BroadPhaseLayer p_broad_phase_layer,
	uint32_t p_collision_layer,
	uint32_t p_collision_mask
) {
	const uint64_t collision = encode_collision(p_collision_layer, p_collision_mask);

	JPH::ObjectLayer object_layer = 0;

	auto iter = layers_by_collision.find(collision);
	if (iter != layers_by_collision.end()) {
		object_layer = iter->second;
	} else {
		constexpr uint16_t object_layer_count = 1U << 13U;

		ERR_FAIL_COND_D_REPORT(
			next_object_layer == object_layer_count,
			vformat(
				"Maximum number of object layers (%d) reached. "
				"This means there are %d combinations of collision layers and masks.",
				object_layer_count,
				object_layer_count
			)
		);

		object_layer = _allocate_object_layer(collision);
	}

	return encode_layers(p_broad_phase_layer, object_layer);
}

void JoltLayerMapper::from_object_layer(
	JPH::ObjectLayer p_encoded_layer,
	JPH::BroadPhaseLayer& p_broad_phase_layer,
	uint32_t& p_collision_layer,
	uint32_t& p_collision_mask
) const {
	JPH::ObjectLayer object_layer = {};
	decode_layers(p_encoded_layer, p_broad_phase_layer, object_layer);

	const uint64_t collision = collisions_by_layer[object_layer];

	decode_collision(collision, p_collision_layer, p_collision_mask);
}

uint32_t JoltLayerMapper::GetNumBroadPhaseLayers() const {
	return JoltBroadPhaseLayer::COUNT;
}

JPH::BroadPhaseLayer JoltLayerMapper::GetBroadPhaseLayer(JPH::ObjectLayer p_layer) const {
	JPH::BroadPhaseLayer broad_phase_layer = {};
	JPH::ObjectLayer object_layer = 0;
	decode_layers(p_layer, broad_phase_layer, object_layer);

	return broad_phase_layer;
}

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)

const char* JoltLayerMapper::GetBroadPhaseLayerName(JPH::BroadPhaseLayer p_layer) const {
	switch ((JPH::BroadPhaseLayer::Type)p_layer) {
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::BODY_STATIC: {
			return "BODY_STATIC";
		}
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::BODY_STATIC_BIG: {
			return "BODY_STATIC_BIG";
		}
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::BODY_DYNAMIC: {
			return "BODY_DYNAMIC";
		}
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::AREA_DETECTABLE: {
			return "AREA_DETECTABLE";
		}
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::AREA_UNDETECTABLE: {
			return "AREA_UNDETECTABLE";
		}
		default: {
			return "UNKNOWN";
		}
	}
}

#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

bool JoltLayerMapper::ShouldCollide(
	JPH::ObjectLayer p_encoded_layer1,
	JPH::ObjectLayer p_encoded_layer2
) const {
	JPH::BroadPhaseLayer broad_phase_layer1 = {};
	uint32_t collision_layer1 = 0;
	uint32_t collision_mask1 = 0;
	from_object_layer(p_encoded_layer1, broad_phase_layer1, collision_layer1, collision_mask1);

	JPH::BroadPhaseLayer broad_phase_layer2 = {};
	uint32_t collision_layer2 = 0;
	uint32_t collision_mask2 = 0;
	from_object_layer(p_encoded_layer2, broad_phase_layer2, collision_layer2, collision_mask2);

	const bool first_scans_second = (collision_mask1 & collision_layer2) != 0;
	const bool second_scans_first = (collision_mask2 & collision_layer1) != 0;

	return first_scans_second || second_scans_first;
}

bool JoltLayerMapper::ShouldCollide(
	JPH::ObjectLayer p_encoded_layer1,
	JPH::BroadPhaseLayer p_broad_phase_layer2
) const {
	static const JoltBroadPhaseMatrix matrix;

	JPH::BroadPhaseLayer broad_phase_layer1 = {};
	JPH::ObjectLayer object_layer1 = 0;
	decode_layers(p_encoded_layer1, broad_phase_layer1, object_layer1);

	return matrix.should_collide(broad_phase_layer1, p_broad_phase_layer2);
}

JPH::ObjectLayer JoltLayerMapper::_allocate_object_layer(uint64_t p_collision) {
	const JPH::ObjectLayer new_object_layer = next_object_layer++;

	collisions_by_layer.resize(new_object_layer + 1);
	collisions_by_layer[new_object_layer] = p_collision;

	layers_by_collision[p_collision] = new_object_layer;

	return new_object_layer;
}

static_assert(sizeof(JPH::ObjectLayer) == 2);
static_assert(sizeof(JPH::BroadPhaseLayer::Type) == 1);
