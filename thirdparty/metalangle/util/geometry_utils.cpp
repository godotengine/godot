//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// geometry_utils:
//   Helper library for generating certain sets of geometry.
//

#include "geometry_utils.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace angle;

SphereGeometry::SphereGeometry() {}

SphereGeometry::~SphereGeometry() {}

CubeGeometry::CubeGeometry() {}

CubeGeometry::~CubeGeometry() {}

void CreateSphereGeometry(size_t sliceCount, float radius, SphereGeometry *result)
{
    size_t parellelCount = sliceCount / 2;
    size_t vertexCount   = (parellelCount + 1) * (sliceCount + 1);
    size_t indexCount    = parellelCount * sliceCount * 6;
    float angleStep      = static_cast<float>(2.0f * M_PI) / sliceCount;

    result->positions.resize(vertexCount);
    result->normals.resize(vertexCount);
    for (size_t i = 0; i < parellelCount + 1; i++)
    {
        for (size_t j = 0; j < sliceCount + 1; j++)
        {
            Vector3 direction(sinf(angleStep * i) * sinf(angleStep * j), cosf(angleStep * i),
                              sinf(angleStep * i) * cosf(angleStep * j));

            size_t vertexIdx             = i * (sliceCount + 1) + j;
            result->positions[vertexIdx] = direction * radius;
            result->normals[vertexIdx]   = direction;
        }
    }

    result->indices.clear();
    result->indices.reserve(indexCount);
    for (size_t i = 0; i < parellelCount; i++)
    {
        for (size_t j = 0; j < sliceCount; j++)
        {
            result->indices.push_back(static_cast<unsigned short>(i * (sliceCount + 1) + j));
            result->indices.push_back(static_cast<unsigned short>((i + 1) * (sliceCount + 1) + j));
            result->indices.push_back(
                static_cast<unsigned short>((i + 1) * (sliceCount + 1) + (j + 1)));

            result->indices.push_back(static_cast<unsigned short>(i * (sliceCount + 1) + j));
            result->indices.push_back(
                static_cast<unsigned short>((i + 1) * (sliceCount + 1) + (j + 1)));
            result->indices.push_back(static_cast<unsigned short>(i * (sliceCount + 1) + (j + 1)));
        }
    }
}

void GenerateCubeGeometry(float radius, CubeGeometry *result)
{
    result->positions.resize(24);
    result->positions[0]  = Vector3(-radius, -radius, -radius);
    result->positions[1]  = Vector3(-radius, -radius, radius);
    result->positions[2]  = Vector3(radius, -radius, radius);
    result->positions[3]  = Vector3(radius, -radius, -radius);
    result->positions[4]  = Vector3(-radius, radius, -radius);
    result->positions[5]  = Vector3(-radius, radius, radius);
    result->positions[6]  = Vector3(radius, radius, radius);
    result->positions[7]  = Vector3(radius, radius, -radius);
    result->positions[8]  = Vector3(-radius, -radius, -radius);
    result->positions[9]  = Vector3(-radius, radius, -radius);
    result->positions[10] = Vector3(radius, radius, -radius);
    result->positions[11] = Vector3(radius, -radius, -radius);
    result->positions[12] = Vector3(-radius, -radius, radius);
    result->positions[13] = Vector3(-radius, radius, radius);
    result->positions[14] = Vector3(radius, radius, radius);
    result->positions[15] = Vector3(radius, -radius, radius);
    result->positions[16] = Vector3(-radius, -radius, -radius);
    result->positions[17] = Vector3(-radius, -radius, radius);
    result->positions[18] = Vector3(-radius, radius, radius);
    result->positions[19] = Vector3(-radius, radius, -radius);
    result->positions[20] = Vector3(radius, -radius, -radius);
    result->positions[21] = Vector3(radius, -radius, radius);
    result->positions[22] = Vector3(radius, radius, radius);
    result->positions[23] = Vector3(radius, radius, -radius);

    result->normals.resize(24);
    result->normals[0]  = Vector3(0.0f, -1.0f, 0.0f);
    result->normals[1]  = Vector3(0.0f, -1.0f, 0.0f);
    result->normals[2]  = Vector3(0.0f, -1.0f, 0.0f);
    result->normals[3]  = Vector3(0.0f, -1.0f, 0.0f);
    result->normals[4]  = Vector3(0.0f, 1.0f, 0.0f);
    result->normals[5]  = Vector3(0.0f, 1.0f, 0.0f);
    result->normals[6]  = Vector3(0.0f, 1.0f, 0.0f);
    result->normals[7]  = Vector3(0.0f, 1.0f, 0.0f);
    result->normals[8]  = Vector3(0.0f, 0.0f, -1.0f);
    result->normals[9]  = Vector3(0.0f, 0.0f, -1.0f);
    result->normals[10] = Vector3(0.0f, 0.0f, -1.0f);
    result->normals[11] = Vector3(0.0f, 0.0f, -1.0f);
    result->normals[12] = Vector3(0.0f, 0.0f, 1.0f);
    result->normals[13] = Vector3(0.0f, 0.0f, 1.0f);
    result->normals[14] = Vector3(0.0f, 0.0f, 1.0f);
    result->normals[15] = Vector3(0.0f, 0.0f, 1.0f);
    result->normals[16] = Vector3(-1.0f, 0.0f, 0.0f);
    result->normals[17] = Vector3(-1.0f, 0.0f, 0.0f);
    result->normals[18] = Vector3(-1.0f, 0.0f, 0.0f);
    result->normals[19] = Vector3(-1.0f, 0.0f, 0.0f);
    result->normals[20] = Vector3(1.0f, 0.0f, 0.0f);
    result->normals[21] = Vector3(1.0f, 0.0f, 0.0f);
    result->normals[22] = Vector3(1.0f, 0.0f, 0.0f);
    result->normals[23] = Vector3(1.0f, 0.0f, 0.0f);

    result->texcoords.resize(24);
    result->texcoords[0]  = Vector2(0.0f, 0.0f);
    result->texcoords[1]  = Vector2(0.0f, 1.0f);
    result->texcoords[2]  = Vector2(1.0f, 1.0f);
    result->texcoords[3]  = Vector2(1.0f, 0.0f);
    result->texcoords[4]  = Vector2(1.0f, 0.0f);
    result->texcoords[5]  = Vector2(1.0f, 1.0f);
    result->texcoords[6]  = Vector2(0.0f, 1.0f);
    result->texcoords[7]  = Vector2(0.0f, 0.0f);
    result->texcoords[8]  = Vector2(0.0f, 0.0f);
    result->texcoords[9]  = Vector2(0.0f, 1.0f);
    result->texcoords[10] = Vector2(1.0f, 1.0f);
    result->texcoords[11] = Vector2(1.0f, 0.0f);
    result->texcoords[12] = Vector2(0.0f, 0.0f);
    result->texcoords[13] = Vector2(0.0f, 1.0f);
    result->texcoords[14] = Vector2(1.0f, 1.0f);
    result->texcoords[15] = Vector2(1.0f, 0.0f);
    result->texcoords[16] = Vector2(0.0f, 0.0f);
    result->texcoords[17] = Vector2(0.0f, 1.0f);
    result->texcoords[18] = Vector2(1.0f, 1.0f);
    result->texcoords[19] = Vector2(1.0f, 0.0f);
    result->texcoords[20] = Vector2(0.0f, 0.0f);
    result->texcoords[21] = Vector2(0.0f, 1.0f);
    result->texcoords[22] = Vector2(1.0f, 1.0f);
    result->texcoords[23] = Vector2(1.0f, 0.0f);

    result->indices.resize(36);
    result->indices[0]  = 0;
    result->indices[1]  = 2;
    result->indices[2]  = 1;
    result->indices[3]  = 0;
    result->indices[4]  = 3;
    result->indices[5]  = 2;
    result->indices[6]  = 4;
    result->indices[7]  = 5;
    result->indices[8]  = 6;
    result->indices[9]  = 4;
    result->indices[10] = 6;
    result->indices[11] = 7;
    result->indices[12] = 8;
    result->indices[13] = 9;
    result->indices[14] = 10;
    result->indices[15] = 8;
    result->indices[16] = 10;
    result->indices[17] = 11;
    result->indices[18] = 12;
    result->indices[19] = 15;
    result->indices[20] = 14;
    result->indices[21] = 12;
    result->indices[22] = 14;
    result->indices[23] = 13;
    result->indices[24] = 16;
    result->indices[25] = 17;
    result->indices[26] = 18;
    result->indices[27] = 16;
    result->indices[28] = 18;
    result->indices[29] = 19;
    result->indices[30] = 20;
    result->indices[31] = 23;
    result->indices[32] = 22;
    result->indices[33] = 20;
    result->indices[34] = 22;
    result->indices[35] = 21;
}
