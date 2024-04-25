//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>

namespace mx = MaterialX;

const float EPSILON = 1e-4f;
const float PI = std::acos(-1.0f);

TEST_CASE("Vectors", "[types]")
{
    mx::Vector3 v1(1, 2, 3);
    mx::Vector3 v2(2, 4, 6);

    // Indexing operators
    REQUIRE(v1[2] == 3);
    v1[2] = 4;
    REQUIRE(v1[2] == 4);
    v1[2] = 3;

    // Component-wise operators
    REQUIRE(v2 + v1 == mx::Vector3(3, 6, 9));
    REQUIRE(v2 - v1 == mx::Vector3(1, 2, 3));
    REQUIRE(v2 * v1 == mx::Vector3(2, 8, 18));
    REQUIRE(v2 / v1 == mx::Vector3(2, 2, 2));
    REQUIRE((v2 += v1) == mx::Vector3(3, 6, 9));
    REQUIRE((v2 -= v1) == mx::Vector3(2, 4, 6));
    REQUIRE((v2 *= v1) == mx::Vector3(2, 8, 18));
    REQUIRE((v2 /= v1) == mx::Vector3(2, 4, 6));
    REQUIRE(v1 * 2 == v2);
    REQUIRE(v2 / 2 == v1);
    
    // Geometric methods
    mx::Vector4 v3(4);
    REQUIRE(v3.getMagnitude() == 8);
    REQUIRE(v3.getNormalized().getMagnitude() == 1);
    REQUIRE(v1.dot(v2) == 28);
    REQUIRE(v1.cross(v2) == mx::Vector3());
}

TEST_CASE("Matrices", "[types]")
{
    // Translation and scale
    mx::Matrix44 trans = mx::Matrix44::createTranslation(mx::Vector3(1, 2, 3));
    mx::Matrix44 scale = mx::Matrix44::createScale(mx::Vector3(2));
    REQUIRE(trans == mx::Matrix44(1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  1, 2, 3, 1));
    REQUIRE(scale == mx::Matrix44(2, 0, 0, 0,
                                  0, 2, 0, 0,
                                  0, 0, 2, 0,
                                  0, 0, 0, 1));

    // Indexing operators
    REQUIRE(trans[3][2] == 3);
    trans[3][2] = 4;
    REQUIRE(trans[3][2] == 4);
    trans[3][2] = 3;

    // Matrix methods
    REQUIRE(trans.getTranspose() == mx::Matrix44(1, 0, 0, 1,
                                                 0, 1, 0, 2,
                                                 0, 0, 1, 3,
                                                 0, 0, 0, 1));
    REQUIRE(scale.getTranspose() == scale);
    REQUIRE(trans.getDeterminant() == 1);
    REQUIRE(scale.getDeterminant() == 8);
    REQUIRE(trans.getInverse() ==
            mx::Matrix44::createTranslation(mx::Vector3(-1, -2, -3)));

    // Matrix product
    mx::Matrix44 prod1 = trans * scale;
    mx::Matrix44 prod2 = scale * trans;
    mx::Matrix44 prod3 = trans * 2;
    mx::Matrix44 prod4 = trans;
    prod4 *= scale;
    REQUIRE(prod1 == mx::Matrix44(2, 0, 0, 0,
                                  0, 2, 0, 0,
                                  0, 0, 2, 0,
                                  2, 4, 6, 1));
    REQUIRE(prod2 == mx::Matrix44(2, 0, 0, 0,
                                  0, 2, 0, 0,
                                  0, 0, 2, 0,
                                  1, 2, 3, 1));
    REQUIRE(prod3 == mx::Matrix44(2, 0, 0, 0,
                                  0, 2, 0, 0,
                                  0, 0, 2, 0,
                                  2, 4, 6, 2));
    REQUIRE(prod4 == prod1);

    // Matrix division
    mx::Matrix44 quot1 = prod1 / scale;
    mx::Matrix44 quot2 = prod2 / trans;
    mx::Matrix44 quot3 = prod3 / 2;
    mx::Matrix44 quot4 = quot1;
    quot4 /= trans;
    REQUIRE(quot1 == trans);
    REQUIRE(quot2 == scale);
    REQUIRE(quot3 == trans);
    REQUIRE(quot4 == mx::Matrix44::IDENTITY);

    // 2D rotation
    mx::Matrix33 rot1 = mx::Matrix33::createRotation(PI / 2);
    mx::Matrix33 rot2 = mx::Matrix33::createRotation(PI);
    REQUIRE((rot1 * rot1).isEquivalent(rot2, EPSILON));
    REQUIRE(rot2.isEquivalent(mx::Matrix33::createScale(mx::Vector2(-1)), EPSILON));
    REQUIRE((rot2 * rot2).isEquivalent(mx::Matrix33::IDENTITY, EPSILON));

    // 3D rotation
    mx::Matrix44 rotX = mx::Matrix44::createRotationX(PI);
    mx::Matrix44 rotY = mx::Matrix44::createRotationY(PI);
    mx::Matrix44 rotZ = mx::Matrix44::createRotationZ(PI);
    REQUIRE((rotX * rotY).isEquivalent(mx::Matrix44::createScale({-1, -1, 1}), EPSILON));
    REQUIRE((rotX * rotZ).isEquivalent(mx::Matrix44::createScale({-1, 1, -1}), EPSILON));
    REQUIRE((rotY * rotZ).isEquivalent(mx::Matrix44::createScale({1, -1, -1}), EPSILON));
}
