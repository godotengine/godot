#ifndef TEST_VECTOR2_ANGLE_TO_H
#define TEST_VECTOR2_ANGLE_TO_H

#include "core/math/vector2.h"
#include "tests/test_macros.h"
#include <cmath>

namespace TestVector2AngleTo {

TEST_CASE("[Math] Vector2::angle_to with orthogonal vectors") {
    Vector2 v1(1, 0);
    Vector2 v2(0, 1);
    CHECK_MESSAGE(
        Math::is_equal_approx(v1.angle_to(v2), Math_PI / 2.0),
        "Angle between (1,0) and (0,1) should be PI/2 radians"
    );
}

TEST_CASE("[Math] Vector2::angle_to with opposite vectors") {
    Vector2 v1(1, 0);
    Vector2 v2(-1, 0);
    CHECK_MESSAGE(
        Math::is_equal_approx(v1.angle_to(v2), Math_PI),
        "Angle between opposite vectors should be PI radians"
    );
}

TEST_CASE("[Math] Vector2::angle_to with identical vectors") {
    Vector2 v1(5, 5);
    Vector2 v2(5, 5);
    CHECK_MESSAGE(
        Math::is_zero_approx(v1.angle_to(v2)),
        "Angle between identical vectors should be 0 radians"
    );
}

}

#endif
