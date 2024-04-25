//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/Types.h>

MATERIALX_NAMESPACE_BEGIN

const Quaternion Quaternion::IDENTITY(0, 0, 0, 1);

//
// Quaternion methods
//

Matrix44 Quaternion::toMatrix() const
{
    Vector3 x(1.0f - 2.0f * (_arr[1] * _arr[1] + _arr[2] * _arr[2]),
              2.0f * (_arr[0] * _arr[1] + _arr[2] * _arr[3]),
              2.0f * (_arr[2] * _arr[0] - _arr[1] * _arr[3]));
    Vector3 y(2.0f * (_arr[0] * _arr[1] - _arr[2] * _arr[3]),
              1.0f - 2.0f * (_arr[2] * _arr[2] + _arr[0] * _arr[0]),
              2.0f * (_arr[1] * _arr[2] + _arr[0] * _arr[3]));
    Vector3 z(2.0f * (_arr[2] * _arr[0] + _arr[1] * _arr[3]),
              2.0f * (_arr[1] * _arr[2] - _arr[0] * _arr[3]),
              1.0f - 2.0f * (_arr[1] * _arr[1] + _arr[0] * _arr[0]));

    return Matrix44(x[0], x[1], x[2], 0.0f,
                    y[0], y[1], y[2], 0.0f,
                    z[0], z[1], z[2], 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
}

MATERIALX_NAMESPACE_END
