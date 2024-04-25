//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/ShaderRenderer.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const float PI = std::acos(-1.0f);

const Vector3 DEFAULT_EYE_POSITION(0.0f, 0.0f, 3.0f);
const Vector3 DEFAULT_TARGET_POSITION(0.0f, 0.0f, 0.0f);
const Vector3 DEFAULT_UP_VECTOR(0.0f, 1.0f, 0.0f);
const float DEFAULT_FIELD_OF_VIEW = 45.0f;
const float DEFAULT_NEAR_PLANE = 0.05f;
const float DEFAULT_FAR_PLANE = 100.0f;

} // anonymous namespace

//
// ShaderRenderer methods
//

ShaderRenderer::ShaderRenderer(unsigned int width, unsigned int height, Image::BaseType baseType, MatrixConvention matrixConvention) :
    _width(width),
    _height(height),
    _baseType(baseType),
    _matrixConvention(matrixConvention)
{
    // Initialize a default camera.
    float fH = std::tan(DEFAULT_FIELD_OF_VIEW / 360.0f * PI) * DEFAULT_NEAR_PLANE;
    float fW = fH * 1.0f;
    _camera = Camera::create();
    _camera->setViewMatrix(Camera::createViewMatrix(DEFAULT_EYE_POSITION, DEFAULT_TARGET_POSITION, DEFAULT_UP_VECTOR));

    if (_matrixConvention == ShaderRenderer::MatrixConvention::Metal)
    {
        _camera->setProjectionMatrix(Camera::createPerspectiveMatrixZP(-fW, fW, -fH, fH, DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE));
    }
    else // MatrixConvention::OpenGL (default)
    {
        _camera->setProjectionMatrix(Camera::createPerspectiveMatrix(-fW, fW, -fH, fH, DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE));
    }
}

void ShaderRenderer::createProgram(ShaderPtr)
{
}

void ShaderRenderer::createProgram(const StageMap&)
{
}

void ShaderRenderer::setSize(unsigned int, unsigned int)
{
}

void ShaderRenderer::updateUniform(const string&, ConstValuePtr)
{
    throw ExceptionRenderError("Update uniform is not yet supported");
}

MATERIALX_NAMESPACE_END
