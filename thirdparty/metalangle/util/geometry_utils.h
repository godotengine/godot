//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// geometry_utils:
//   Helper library for generating certain sets of geometry.
//

#ifndef UTIL_GEOMETRY_UTILS_H
#define UTIL_GEOMETRY_UTILS_H

#include <cstddef>
#include <vector>

#include <GLES2/gl2.h>

#include "common/vector_utils.h"
#include "util/util_export.h"

struct ANGLE_UTIL_EXPORT SphereGeometry
{
    SphereGeometry();
    ~SphereGeometry();

    std::vector<angle::Vector3> positions;
    std::vector<angle::Vector3> normals;
    std::vector<GLushort> indices;
};

ANGLE_UTIL_EXPORT void CreateSphereGeometry(size_t sliceCount,
                                            float radius,
                                            SphereGeometry *result);

struct ANGLE_UTIL_EXPORT CubeGeometry
{
    CubeGeometry();
    ~CubeGeometry();

    std::vector<angle::Vector3> positions;
    std::vector<angle::Vector3> normals;
    std::vector<angle::Vector2> texcoords;
    std::vector<GLushort> indices;
};

ANGLE_UTIL_EXPORT void GenerateCubeGeometry(float radius, CubeGeometry *result);

#endif  // UTIL_GEOMETRY_UTILS_H
