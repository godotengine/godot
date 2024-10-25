#pragma once
#include "../common.h"

// NOLINTNEXTLINE(readability-identifier-naming)
namespace JoltCustomShapeSubType {

constexpr JPH::EShapeSubType OVERRIDE_USER_DATA = JPH::EShapeSubType::User1;
constexpr JPH::EShapeSubType DOUBLE_SIDED = JPH::EShapeSubType::User2;
constexpr JPH::EShapeSubType RAY = JPH::EShapeSubType::UserConvex1;
constexpr JPH::EShapeSubType MOTION = JPH::EShapeSubType::UserConvex2;

} // namespace JoltCustomShapeSubType
