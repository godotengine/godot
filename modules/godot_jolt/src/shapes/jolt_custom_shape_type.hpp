#pragma once
#include "../common.h"
// NOLINTNEXTLINE(readability-identifier-naming)
namespace JoltCustomShapeType {

constexpr JPH::EShapeType EMPTY = JPH::EShapeType::User1;

} // namespace JoltCustomShapeType

// NOLINTNEXTLINE(readability-identifier-naming)
namespace JoltCustomShapeSubType {

constexpr JPH::EShapeSubType EMPTY = JPH::EShapeSubType::User1;
constexpr JPH::EShapeSubType OVERRIDE_USER_DATA = JPH::EShapeSubType::User2;
constexpr JPH::EShapeSubType DOUBLE_SIDED = JPH::EShapeSubType::User3;
constexpr JPH::EShapeSubType RAY = JPH::EShapeSubType::UserConvex1;
constexpr JPH::EShapeSubType MOTION = JPH::EShapeSubType::UserConvex2;

} // namespace JoltCustomShapeSubType
