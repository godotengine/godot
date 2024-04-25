//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Types.h>

MATERIALX_NAMESPACE_BEGIN

const string DEFAULT_TYPE_STRING = "color3";
const string FILENAME_TYPE_STRING = "filename";
const string GEOMNAME_TYPE_STRING = "geomname";
const string STRING_TYPE_STRING = "string";
const string BSDF_TYPE_STRING = "BSDF";
const string EDF_TYPE_STRING = "EDF";
const string VDF_TYPE_STRING = "VDF";
const string SURFACE_SHADER_TYPE_STRING = "surfaceshader";
const string DISPLACEMENT_SHADER_TYPE_STRING = "displacementshader";
const string VOLUME_SHADER_TYPE_STRING = "volumeshader";
const string LIGHT_SHADER_TYPE_STRING = "lightshader";
const string MATERIAL_TYPE_STRING = "material";
const string SURFACE_MATERIAL_NODE_STRING = "surfacematerial";
const string VOLUME_MATERIAL_NODE_STRING = "volumematerial";
const string MULTI_OUTPUT_TYPE_STRING = "multioutput";
const string NONE_TYPE_STRING = "none";
const string VALUE_STRING_TRUE = "true";
const string VALUE_STRING_FALSE = "false";
const string NAME_PREFIX_SEPARATOR = ":";
const string NAME_PATH_SEPARATOR = "/";
const string ARRAY_VALID_SEPARATORS = ", ";
const string ARRAY_PREFERRED_SEPARATOR = ", ";

const Matrix33 Matrix33::IDENTITY(1, 0, 0,
                                  0, 1, 0,
                                  0, 0, 1);

const Matrix44 Matrix44::IDENTITY(1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1);

//
// Color3 methods
//

Color3 Color3::linearToSrgb() const
{
    Color3 res;
    for (size_t i = 0; i < 3; i++)
    {
        if (_arr[i] <= 0.0031308f)
        {
            res[i] = _arr[i] * 12.92f;
        }
        else
        {
            res[i] = 1.055f * std::pow(_arr[i], 1.0f / 2.4f) - 0.055f;
        }
    }
    return res;
}

Color3 Color3::srgbToLinear() const
{
    Color3 res;
    for (size_t i = 0; i < 3; i++)
    {
        if (_arr[i] <= 0.04045f)
        {
            res[i] = _arr[i] / 12.92f;
        }
        else
        {
            res[i] = std::pow((_arr[i] + 0.055f) / 1.055f, 2.4f);
        }
    }
    return res;
}

//
// Matrix33 methods
//

Matrix33 Matrix33::getTranspose() const
{
    return Matrix33(_arr[0][0], _arr[1][0], _arr[2][0],
                    _arr[0][1], _arr[1][1], _arr[2][1],
                    _arr[0][2], _arr[1][2], _arr[2][2]);
}

float Matrix33::getDeterminant() const
{
    return _arr[0][0] * (_arr[1][1]*_arr[2][2] - _arr[2][1]*_arr[1][2]) +
           _arr[0][1] * (_arr[1][2]*_arr[2][0] - _arr[2][2]*_arr[1][0]) +
           _arr[0][2] * (_arr[1][0]*_arr[2][1] - _arr[2][0]*_arr[1][1]);
}

Matrix33 Matrix33::getAdjugate() const
{
    return Matrix33(
        _arr[1][1]*_arr[2][2] - _arr[2][1]*_arr[1][2],
        _arr[2][1]*_arr[0][2] - _arr[0][1]*_arr[2][2],
        _arr[0][1]*_arr[1][2] - _arr[1][1]*_arr[0][2],
        _arr[1][2]*_arr[2][0] - _arr[2][2]*_arr[1][0],
        _arr[2][2]*_arr[0][0] - _arr[0][2]*_arr[2][0],
        _arr[0][2]*_arr[1][0] - _arr[1][2]*_arr[0][0],
        _arr[1][0]*_arr[2][1] - _arr[2][0]*_arr[1][1],
        _arr[2][0]*_arr[0][1] - _arr[0][0]*_arr[2][1],
        _arr[0][0]*_arr[1][1] - _arr[1][0]*_arr[0][1]);
}

Vector3 Matrix33::multiply(const Vector3& v) const
{
    return Vector3(
      v[0]*_arr[0][0] + v[1]*_arr[1][0] + v[2]*_arr[2][0],
      v[0]*_arr[0][1] + v[1]*_arr[1][1] + v[2]*_arr[2][1],
      v[0]*_arr[0][2] + v[1]*_arr[1][2] + v[2]*_arr[2][2]);
}

Vector2 Matrix33::transformPoint(const Vector2& v) const
{
    Vector3 res = multiply(Vector3(v[0], v[1], 1.0f));
    return Vector2(res[0], res[1]);
}

Vector2 Matrix33::transformVector(const Vector2& v) const
{
    Vector3 res = multiply(Vector3(v[0], v[1], 0.0f));
    return Vector2(res[0], res[1]);
}

Vector3 Matrix33::transformNormal(const Vector3& v) const
{
    return getInverse().getTranspose().multiply(v);
}

Matrix33 Matrix33::createTranslation(const Vector2& v)
{
    return Matrix33(1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    v[0], v[1], 1.0f);
}

Matrix33 Matrix33::createScale(const Vector2& v)
{
    return Matrix33(v[0], 0.0f, 0.0f,
                    0.0f, v[1], 0.0f,
                    0.0f, 0.0f, 1.0f);
}

Matrix33 Matrix33::createRotation(float angle)
{
    float sin = std::sin(angle);
    float cos = std::cos(angle);

    return Matrix33( cos,  sin, 0.0f,
                    -sin,  cos, 0.0f,
                    0.0f, 0.0f, 1.0f);
}

//
// Matrix44 methods
//

Matrix44 Matrix44::getTranspose() const
{
    return Matrix44(_arr[0][0], _arr[1][0], _arr[2][0], _arr[3][0],
                    _arr[0][1], _arr[1][1], _arr[2][1], _arr[3][1],
                    _arr[0][2], _arr[1][2], _arr[2][2], _arr[3][2],
                    _arr[0][3], _arr[1][3], _arr[2][3], _arr[3][3]);
}

float Matrix44::getDeterminant() const
{
    return _arr[0][0] * (_arr[1][1]*_arr[2][2]*_arr[3][3] + _arr[3][1]*_arr[1][2]*_arr[2][3] + _arr[2][1]*_arr[3][2]*_arr[1][3] -
                         _arr[1][1]*_arr[3][2]*_arr[2][3] - _arr[2][1]*_arr[1][2]*_arr[3][3] - _arr[3][1]*_arr[2][2]*_arr[1][3]) +
           _arr[0][1] * (_arr[1][2]*_arr[3][3]*_arr[2][0] + _arr[2][2]*_arr[1][3]*_arr[3][0] + _arr[3][2]*_arr[2][3]*_arr[1][0] -
                         _arr[1][2]*_arr[2][3]*_arr[3][0] - _arr[3][2]*_arr[1][3]*_arr[2][0] - _arr[2][2]*_arr[3][3]*_arr[1][0]) +
           _arr[0][2] * (_arr[1][3]*_arr[2][0]*_arr[3][1] + _arr[3][3]*_arr[1][0]*_arr[2][1] + _arr[2][3]*_arr[3][0]*_arr[1][1] -
                         _arr[1][3]*_arr[3][0]*_arr[2][1] - _arr[2][3]*_arr[1][0]*_arr[3][1] - _arr[3][3]*_arr[2][0]*_arr[1][1]) +
           _arr[0][3] * (_arr[1][0]*_arr[3][1]*_arr[2][2] + _arr[2][0]*_arr[1][1]*_arr[3][2] + _arr[3][0]*_arr[2][1]*_arr[1][2] -
                         _arr[1][0]*_arr[2][1]*_arr[3][2] - _arr[3][0]*_arr[1][1]*_arr[2][2] - _arr[2][0]*_arr[3][1]*_arr[1][2]);
}

Matrix44 Matrix44::getAdjugate() const
{
    return Matrix44(
        _arr[1][1]*_arr[2][2]*_arr[3][3] + _arr[3][1]*_arr[1][2]*_arr[2][3] + _arr[2][1]*_arr[3][2]*_arr[1][3] -
        _arr[1][1]*_arr[3][2]*_arr[2][3] - _arr[2][1]*_arr[1][2]*_arr[3][3] - _arr[3][1]*_arr[2][2]*_arr[1][3],

        _arr[0][1]*_arr[3][2]*_arr[2][3] + _arr[2][1]*_arr[0][2]*_arr[3][3] + _arr[3][1]*_arr[2][2]*_arr[0][3] -
        _arr[3][1]*_arr[0][2]*_arr[2][3] - _arr[2][1]*_arr[3][2]*_arr[0][3] - _arr[0][1]*_arr[2][2]*_arr[3][3],

        _arr[0][1]*_arr[1][2]*_arr[3][3] + _arr[3][1]*_arr[0][2]*_arr[1][3] + _arr[1][1]*_arr[3][2]*_arr[0][3] -
        _arr[0][1]*_arr[3][2]*_arr[1][3] - _arr[1][1]*_arr[0][2]*_arr[3][3] - _arr[3][1]*_arr[1][2]*_arr[0][3],

        _arr[0][1]*_arr[2][2]*_arr[1][3] + _arr[1][1]*_arr[0][2]*_arr[2][3] + _arr[2][1]*_arr[1][2]*_arr[0][3] -
        _arr[0][1]*_arr[1][2]*_arr[2][3] - _arr[2][1]*_arr[0][2]*_arr[1][3] - _arr[1][1]*_arr[2][2]*_arr[0][3],

        _arr[1][2]*_arr[3][3]*_arr[2][0] + _arr[2][2]*_arr[1][3]*_arr[3][0] + _arr[3][2]*_arr[2][3]*_arr[1][0] -
        _arr[1][2]*_arr[2][3]*_arr[3][0] - _arr[3][2]*_arr[1][3]*_arr[2][0] - _arr[2][2]*_arr[3][3]*_arr[1][0],

        _arr[0][2]*_arr[2][3]*_arr[3][0] + _arr[3][2]*_arr[0][3]*_arr[2][0] + _arr[2][2]*_arr[3][3]*_arr[0][0] -
        _arr[0][2]*_arr[3][3]*_arr[2][0] - _arr[2][2]*_arr[0][3]*_arr[3][0] - _arr[3][2]*_arr[2][3]*_arr[0][0],

        _arr[0][2]*_arr[3][3]*_arr[1][0] + _arr[1][2]*_arr[0][3]*_arr[3][0] + _arr[3][2]*_arr[1][3]*_arr[0][0] -
        _arr[0][2]*_arr[1][3]*_arr[3][0] - _arr[3][2]*_arr[0][3]*_arr[1][0] - _arr[1][2]*_arr[3][3]*_arr[0][0],

        _arr[0][2]*_arr[1][3]*_arr[2][0] + _arr[2][2]*_arr[0][3]*_arr[1][0] + _arr[1][2]*_arr[2][3]*_arr[0][0] -
        _arr[0][2]*_arr[2][3]*_arr[1][0] - _arr[1][2]*_arr[0][3]*_arr[2][0] - _arr[2][2]*_arr[1][3]*_arr[0][0],

        _arr[1][3]*_arr[2][0]*_arr[3][1] + _arr[3][3]*_arr[1][0]*_arr[2][1] + _arr[2][3]*_arr[3][0]*_arr[1][1] -
        _arr[1][3]*_arr[3][0]*_arr[2][1] - _arr[2][3]*_arr[1][0]*_arr[3][1] - _arr[3][3]*_arr[2][0]*_arr[1][1],

        _arr[0][3]*_arr[3][0]*_arr[2][1] + _arr[2][3]*_arr[0][0]*_arr[3][1] + _arr[3][3]*_arr[2][0]*_arr[0][1] -
        _arr[0][3]*_arr[2][0]*_arr[3][1] - _arr[3][3]*_arr[0][0]*_arr[2][1] - _arr[2][3]*_arr[3][0]*_arr[0][1],

        _arr[0][3]*_arr[1][0]*_arr[3][1] + _arr[3][3]*_arr[0][0]*_arr[1][1] + _arr[1][3]*_arr[3][0]*_arr[0][1] -
        _arr[0][3]*_arr[3][0]*_arr[1][1] - _arr[1][3]*_arr[0][0]*_arr[3][1] - _arr[3][3]*_arr[1][0]*_arr[0][1],

        _arr[0][3]*_arr[2][0]*_arr[1][1] + _arr[1][3]*_arr[0][0]*_arr[2][1] + _arr[2][3]*_arr[1][0]*_arr[0][1] -
        _arr[0][3]*_arr[1][0]*_arr[2][1] - _arr[2][3]*_arr[0][0]*_arr[1][1] - _arr[1][3]*_arr[2][0]*_arr[0][1],

        _arr[1][0]*_arr[3][1]*_arr[2][2] + _arr[2][0]*_arr[1][1]*_arr[3][2] + _arr[3][0]*_arr[2][1]*_arr[1][2] -
        _arr[1][0]*_arr[2][1]*_arr[3][2] - _arr[3][0]*_arr[1][1]*_arr[2][2] - _arr[2][0]*_arr[3][1]*_arr[1][2],

        _arr[0][0]*_arr[2][1]*_arr[3][2] + _arr[3][0]*_arr[0][1]*_arr[2][2] + _arr[2][0]*_arr[3][1]*_arr[0][2] -
        _arr[0][0]*_arr[3][1]*_arr[2][2] - _arr[2][0]*_arr[0][1]*_arr[3][2] - _arr[3][0]*_arr[2][1]*_arr[0][2],

        _arr[0][0]*_arr[3][1]*_arr[1][2] + _arr[1][0]*_arr[0][1]*_arr[3][2] + _arr[3][0]*_arr[1][1]*_arr[0][2] -
        _arr[0][0]*_arr[1][1]*_arr[3][2] - _arr[3][0]*_arr[0][1]*_arr[1][2] - _arr[1][0]*_arr[3][1]*_arr[0][2],

        _arr[0][0]*_arr[1][1]*_arr[2][2] + _arr[2][0]*_arr[0][1]*_arr[1][2] + _arr[1][0]*_arr[2][1]*_arr[0][2] -
        _arr[0][0]*_arr[2][1]*_arr[1][2] - _arr[1][0]*_arr[0][1]*_arr[2][2] - _arr[2][0]*_arr[1][1]*_arr[0][2]);
}

Vector4 Matrix44::multiply(const Vector4& v) const
{
    return Vector4(
      v[0]*_arr[0][0] + v[1]*_arr[1][0] + v[2]*_arr[2][0] + v[3]*_arr[3][0],
      v[0]*_arr[0][1] + v[1]*_arr[1][1] + v[2]*_arr[2][1] + v[3]*_arr[3][1],
      v[0]*_arr[0][2] + v[1]*_arr[1][2] + v[2]*_arr[2][2] + v[3]*_arr[3][2],
      v[0]*_arr[0][3] + v[1]*_arr[1][3] + v[2]*_arr[2][3] + v[3]*_arr[3][3]);
}

Vector3 Matrix44::transformPoint(const Vector3& v) const
{
    Vector4 res = multiply(Vector4(v[0], v[1], v[2], 1.0f));
    return Vector3(res[0], res[1], res[2]);
}

Vector3 Matrix44::transformVector(const Vector3& v) const
{
    Vector4 res = multiply(Vector4(v[0], v[1], v[2], 0.0f));
    return Vector3(res[0], res[1], res[2]);
}

Vector3 Matrix44::transformNormal(const Vector3& v) const
{
    return getInverse().getTranspose().transformVector(v);
}

Matrix44 Matrix44::createTranslation(const Vector3& v)
{
    return Matrix44(1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    v[0], v[1], v[2], 1.0f);
}

Matrix44 Matrix44::createScale(const Vector3& v)
{
    return Matrix44(v[0], 0.0f, 0.0f, 0.0f,
                    0.0f, v[1], 0.0f, 0.0f,
                    0.0f, 0.0f, v[2], 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix44 Matrix44::createRotationX(float angle)
{
    float sin = std::sin(angle);
    float cos = std::cos(angle);

    return Matrix44(1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f,  cos,  sin, 0.0f,
                    0.0f, -sin,  cos, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix44 Matrix44::createRotationY(float angle)
{
    float sin = std::sin(angle);
    float cos = std::cos(angle);

    return Matrix44( cos, 0.0f, -sin, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                     sin, 0.0f,  cos, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix44 Matrix44::createRotationZ(float angle)
{
    float sin = std::sin(angle);
    float cos = std::cos(angle);

    return Matrix44( cos,  sin, 0.0f, 0.0f,
                    -sin,  cos, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
}

MATERIALX_NAMESPACE_END
