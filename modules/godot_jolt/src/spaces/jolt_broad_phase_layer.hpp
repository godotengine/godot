#pragma once

// NOLINTNEXTLINE(readability-identifier-naming)
namespace JoltBroadPhaseLayer {

constexpr JPH::BroadPhaseLayer BODY_STATIC(0);
constexpr JPH::BroadPhaseLayer BODY_DYNAMIC(1);
constexpr JPH::BroadPhaseLayer AREA_DETECTABLE(2);
constexpr JPH::BroadPhaseLayer AREA_UNDETECTABLE(3);

constexpr uint32_t COUNT = 4;

static_assert(COUNT <= 8);

} // namespace JoltBroadPhaseLayer
