//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// GLES1State.cpp: Implements the GLES1State class, tracking state
// for GLES1 contexts.

#include "libANGLE/GLES1State.h"

#include "libANGLE/Context.h"
#include "libANGLE/GLES1Renderer.h"

namespace gl
{

TextureCoordF::TextureCoordF() = default;

TextureCoordF::TextureCoordF(float _s, float _t, float _r, float _q) : s(_s), t(_t), r(_r), q(_q) {}

bool TextureCoordF::operator==(const TextureCoordF &other) const
{
    return s == other.s && t == other.t && r == other.r && q == other.q;
}

MaterialParameters::MaterialParameters() = default;

LightModelParameters::LightModelParameters() = default;

LightParameters::LightParameters() = default;

LightParameters::LightParameters(const LightParameters &other) = default;

FogParameters::FogParameters() = default;

TextureEnvironmentParameters::TextureEnvironmentParameters() = default;

TextureEnvironmentParameters::TextureEnvironmentParameters(
    const TextureEnvironmentParameters &other) = default;

PointParameters::PointParameters() = default;

PointParameters::PointParameters(const PointParameters &other) = default;

ClipPlaneParameters::ClipPlaneParameters() = default;

ClipPlaneParameters::ClipPlaneParameters(bool enabled, const angle::Vector4 &equation)
    : enabled(enabled), equation(equation)
{}

ClipPlaneParameters::ClipPlaneParameters(const ClipPlaneParameters &other) = default;

GLES1State::GLES1State()
    : mGLState(nullptr),
      mVertexArrayEnabled(false),
      mNormalArrayEnabled(false),
      mColorArrayEnabled(false),
      mPointSizeArrayEnabled(false),
      mLineSmoothEnabled(false),
      mPointSmoothEnabled(false),
      mPointSpriteEnabled(false),
      mAlphaTestEnabled(false),
      mLogicOpEnabled(false),
      mLightingEnabled(false),
      mFogEnabled(false),
      mRescaleNormalEnabled(false),
      mNormalizeEnabled(false),
      mColorMaterialEnabled(false),
      mReflectionMapEnabled(false),
      mCurrentColor({0.0f, 0.0f, 0.0f, 0.0f}),
      mCurrentNormal({0.0f, 0.0f, 0.0f}),
      mClientActiveTexture(0),
      mMatrixMode(MatrixType::Modelview),
      mShadeModel(ShadingModel::Smooth),
      mAlphaTestFunc(AlphaTestFunc::AlwaysPass),
      mAlphaTestRef(0.0f),
      mLogicOp(LogicalOperation::Copy),
      mLineSmoothHint(HintSetting::DontCare),
      mPointSmoothHint(HintSetting::DontCare),
      mPerspectiveCorrectionHint(HintSetting::DontCare),
      mFogHint(HintSetting::DontCare)
{}

GLES1State::~GLES1State() = default;

// Taken from the GLES 1.x spec which specifies all initial state values.
void GLES1State::initialize(const Context *context, const State *state)
{
    mGLState = state;

    const Caps &caps = context->getCaps();

    mTexUnitEnables.resize(caps.maxMultitextureUnits);
    for (auto &enables : mTexUnitEnables)
    {
        enables.reset();
    }

    mVertexArrayEnabled    = false;
    mNormalArrayEnabled    = false;
    mColorArrayEnabled     = false;
    mPointSizeArrayEnabled = false;
    mTexCoordArrayEnabled.resize(caps.maxMultitextureUnits, false);

    mLineSmoothEnabled    = false;
    mPointSmoothEnabled   = false;
    mPointSpriteEnabled   = false;
    mLogicOpEnabled       = false;
    mAlphaTestEnabled     = false;
    mLightingEnabled      = false;
    mFogEnabled           = false;
    mRescaleNormalEnabled = false;
    mNormalizeEnabled     = false;
    mColorMaterialEnabled = false;
    mReflectionMapEnabled = false;

    mMatrixMode = MatrixType::Modelview;

    mCurrentColor  = {1.0f, 1.0f, 1.0f, 1.0f};
    mCurrentNormal = {0.0f, 0.0f, 1.0f};
    mCurrentTextureCoords.resize(caps.maxMultitextureUnits);
    mClientActiveTexture = 0;

    mTextureEnvironments.resize(caps.maxMultitextureUnits);

    mModelviewMatrices.push_back(angle::Mat4());
    mProjectionMatrices.push_back(angle::Mat4());
    mTextureMatrices.resize(caps.maxMultitextureUnits);
    for (auto &stack : mTextureMatrices)
    {
        stack.push_back(angle::Mat4());
    }

    mMaterial.ambient  = {0.2f, 0.2f, 0.2f, 1.0f};
    mMaterial.diffuse  = {0.8f, 0.8f, 0.8f, 1.0f};
    mMaterial.specular = {0.0f, 0.0f, 0.0f, 1.0f};
    mMaterial.emissive = {0.0f, 0.0f, 0.0f, 1.0f};

    mMaterial.specularExponent = 0.0f;

    mLightModel.color    = {0.2f, 0.2f, 0.2f, 1.0f};
    mLightModel.twoSided = false;

    mLights.resize(caps.maxLights);

    // GL_LIGHT0 is special and has default state that avoids all-black renderings.
    mLights[0].diffuse  = {1.0f, 1.0f, 1.0f, 1.0f};
    mLights[0].specular = {1.0f, 1.0f, 1.0f, 1.0f};

    mFog.mode    = FogMode::Exp;
    mFog.density = 1.0f;
    mFog.start   = 0.0f;
    mFog.end     = 1.0f;

    mFog.color = {0.0f, 0.0f, 0.0f, 0.0f};

    mShadeModel = ShadingModel::Smooth;

    mAlphaTestFunc = AlphaTestFunc::AlwaysPass;
    mAlphaTestRef  = 0.0f;

    mLogicOp = LogicalOperation::Copy;

    mClipPlanes.resize(caps.maxClipPlanes,
                       ClipPlaneParameters(false, angle::Vector4(0.0f, 0.0f, 0.0f, 0.0f)));

    mLineSmoothHint            = HintSetting::DontCare;
    mPointSmoothHint           = HintSetting::DontCare;
    mPerspectiveCorrectionHint = HintSetting::DontCare;
    mFogHint                   = HintSetting::DontCare;

    // The user-specified point size, GL_POINT_SIZE_MAX,
    // is initially equal to the implementation maximum.
    mPointParameters.pointSizeMax = caps.maxAliasedPointSize;

    mDirtyBits.set();
}

void GLES1State::setAlphaFunc(AlphaTestFunc func, GLfloat ref)
{
    setDirty(DIRTY_GLES1_ALPHA_TEST);
    mAlphaTestFunc = func;
    mAlphaTestRef  = ref;
}

void GLES1State::setClientTextureUnit(unsigned int unit)
{
    setDirty(DIRTY_GLES1_CLIENT_ACTIVE_TEXTURE);
    mClientActiveTexture = unit;
}

unsigned int GLES1State::getClientTextureUnit() const
{
    return mClientActiveTexture;
}

void GLES1State::setCurrentColor(const ColorF &color)
{
    setDirty(DIRTY_GLES1_CURRENT_VECTOR);
    mCurrentColor = color;
}

const ColorF &GLES1State::getCurrentColor() const
{
    return mCurrentColor;
}

void GLES1State::setCurrentNormal(const angle::Vector3 &normal)
{
    setDirty(DIRTY_GLES1_CURRENT_VECTOR);
    mCurrentNormal = normal;
}

const angle::Vector3 &GLES1State::getCurrentNormal() const
{
    return mCurrentNormal;
}

void GLES1State::setCurrentTextureCoords(unsigned int unit, const TextureCoordF &coords)
{
    setDirty(DIRTY_GLES1_CURRENT_VECTOR);
    mCurrentTextureCoords[unit] = coords;
}

const TextureCoordF &GLES1State::getCurrentTextureCoords(unsigned int unit) const
{
    return mCurrentTextureCoords[unit];
}

void GLES1State::setMatrixMode(MatrixType mode)
{
    setDirty(DIRTY_GLES1_MATRICES);
    mMatrixMode = mode;
}

MatrixType GLES1State::getMatrixMode() const
{
    return mMatrixMode;
}

GLint GLES1State::getCurrentMatrixStackDepth(GLenum queryType) const
{
    switch (queryType)
    {
        case GL_MODELVIEW_STACK_DEPTH:
            return clampCast<GLint>(mModelviewMatrices.size());
        case GL_PROJECTION_STACK_DEPTH:
            return clampCast<GLint>(mProjectionMatrices.size());
        case GL_TEXTURE_STACK_DEPTH:
            return clampCast<GLint>(mTextureMatrices[mGLState->getActiveSampler()].size());
        default:
            UNREACHABLE();
            return 0;
    }
}

void GLES1State::pushMatrix()
{
    setDirty(DIRTY_GLES1_MATRICES);
    auto &stack = currentMatrixStack();
    stack.push_back(stack.back());
}

void GLES1State::popMatrix()
{
    setDirty(DIRTY_GLES1_MATRICES);
    auto &stack = currentMatrixStack();
    stack.pop_back();
}

GLES1State::MatrixStack &GLES1State::currentMatrixStack()
{
    setDirty(DIRTY_GLES1_MATRICES);
    switch (mMatrixMode)
    {
        case MatrixType::Modelview:
            return mModelviewMatrices;
        case MatrixType::Projection:
            return mProjectionMatrices;
        case MatrixType::Texture:
            return mTextureMatrices[mGLState->getActiveSampler()];
        default:
            UNREACHABLE();
            return mModelviewMatrices;
    }
}

const angle::Mat4 &GLES1State::getModelviewMatrix() const
{
    return mModelviewMatrices.back();
}

const GLES1State::MatrixStack &GLES1State::currentMatrixStack() const
{
    switch (mMatrixMode)
    {
        case MatrixType::Modelview:
            return mModelviewMatrices;
        case MatrixType::Projection:
            return mProjectionMatrices;
        case MatrixType::Texture:
            return mTextureMatrices[mGLState->getActiveSampler()];
        default:
            UNREACHABLE();
            return mModelviewMatrices;
    }
}

void GLES1State::loadMatrix(const angle::Mat4 &m)
{
    setDirty(DIRTY_GLES1_MATRICES);
    currentMatrixStack().back() = m;
}

void GLES1State::multMatrix(const angle::Mat4 &m)
{
    setDirty(DIRTY_GLES1_MATRICES);
    angle::Mat4 currentMatrix   = currentMatrixStack().back();
    currentMatrixStack().back() = currentMatrix.product(m);
}

void GLES1State::setLogicOp(LogicalOperation opcodePacked)
{
    setDirty(DIRTY_GLES1_LOGIC_OP);
    mLogicOp = opcodePacked;
}

void GLES1State::setClientStateEnabled(ClientVertexArrayType clientState, bool enable)
{
    setDirty(DIRTY_GLES1_CLIENT_STATE_ENABLE);
    switch (clientState)
    {
        case ClientVertexArrayType::Vertex:
            mVertexArrayEnabled = enable;
            break;
        case ClientVertexArrayType::Normal:
            mNormalArrayEnabled = enable;
            break;
        case ClientVertexArrayType::Color:
            mColorArrayEnabled = enable;
            break;
        case ClientVertexArrayType::PointSize:
            mPointSizeArrayEnabled = enable;
            break;
        case ClientVertexArrayType::TextureCoord:
            mTexCoordArrayEnabled[mClientActiveTexture] = enable;
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void GLES1State::setTexCoordArrayEnabled(unsigned int unit, bool enable)
{
    setDirty(DIRTY_GLES1_CLIENT_STATE_ENABLE);
    mTexCoordArrayEnabled[unit] = enable;
}

bool GLES1State::isClientStateEnabled(ClientVertexArrayType clientState) const
{
    switch (clientState)
    {
        case ClientVertexArrayType::Vertex:
            return mVertexArrayEnabled;
        case ClientVertexArrayType::Normal:
            return mNormalArrayEnabled;
        case ClientVertexArrayType::Color:
            return mColorArrayEnabled;
        case ClientVertexArrayType::PointSize:
            return mPointSizeArrayEnabled;
        case ClientVertexArrayType::TextureCoord:
            return mTexCoordArrayEnabled[mClientActiveTexture];
        default:
            UNREACHABLE();
            return false;
    }
}

bool GLES1State::isTexCoordArrayEnabled(unsigned int unit) const
{
    ASSERT(unit < mTexCoordArrayEnabled.size());
    return mTexCoordArrayEnabled[unit];
}

bool GLES1State::isTextureTargetEnabled(unsigned int unit, const TextureType type) const
{
    return mTexUnitEnables[unit].test(type);
}

LightModelParameters &GLES1State::lightModelParameters()
{
    setDirty(DIRTY_GLES1_LIGHTS);
    return mLightModel;
}

const LightModelParameters &GLES1State::lightModelParameters() const
{
    return mLightModel;
}

LightParameters &GLES1State::lightParameters(unsigned int light)
{
    setDirty(DIRTY_GLES1_LIGHTS);
    return mLights[light];
}

const LightParameters &GLES1State::lightParameters(unsigned int light) const
{
    return mLights[light];
}

MaterialParameters &GLES1State::materialParameters()
{
    setDirty(DIRTY_GLES1_MATERIAL);
    return mMaterial;
}

const MaterialParameters &GLES1State::materialParameters() const
{
    return mMaterial;
}

bool GLES1State::isColorMaterialEnabled() const
{
    return mColorMaterialEnabled;
}

void GLES1State::setShadeModel(ShadingModel model)
{
    setDirty(DIRTY_GLES1_SHADE_MODEL);
    mShadeModel = model;
}

void GLES1State::setClipPlane(unsigned int plane, const GLfloat *equation)
{
    setDirty(DIRTY_GLES1_CLIP_PLANES);
    assert(plane < mClipPlanes.size());
    mClipPlanes[plane].equation[0] = equation[0];
    mClipPlanes[plane].equation[1] = equation[1];
    mClipPlanes[plane].equation[2] = equation[2];
    mClipPlanes[plane].equation[3] = equation[3];
}

void GLES1State::getClipPlane(unsigned int plane, GLfloat *equation) const
{
    assert(plane < mClipPlanes.size());
    equation[0] = mClipPlanes[plane].equation[0];
    equation[1] = mClipPlanes[plane].equation[1];
    equation[2] = mClipPlanes[plane].equation[2];
    equation[3] = mClipPlanes[plane].equation[3];
}

FogParameters &GLES1State::fogParameters()
{
    setDirty(DIRTY_GLES1_FOG);
    return mFog;
}

const FogParameters &GLES1State::fogParameters() const
{
    return mFog;
}

TextureEnvironmentParameters &GLES1State::textureEnvironment(unsigned int unit)
{
    setDirty(DIRTY_GLES1_TEXTURE_ENVIRONMENT);
    assert(unit < mTextureEnvironments.size());
    return mTextureEnvironments[unit];
}

const TextureEnvironmentParameters &GLES1State::textureEnvironment(unsigned int unit) const
{
    assert(unit < mTextureEnvironments.size());
    return mTextureEnvironments[unit];
}

PointParameters &GLES1State::pointParameters()
{
    setDirty(DIRTY_GLES1_POINT_PARAMETERS);
    return mPointParameters;
}

const PointParameters &GLES1State::pointParameters() const
{
    return mPointParameters;
}

AttributesMask GLES1State::getVertexArraysAttributeMask() const
{
    AttributesMask attribsMask;

    ClientVertexArrayType nonTexcoordArrays[] = {
        ClientVertexArrayType::Vertex,
        ClientVertexArrayType::Normal,
        ClientVertexArrayType::Color,
        ClientVertexArrayType::PointSize,
    };

    for (const ClientVertexArrayType attrib : nonTexcoordArrays)
    {
        attribsMask.set(GLES1Renderer::VertexArrayIndex(attrib, *this),
                        isClientStateEnabled(attrib));
    }

    for (unsigned int i = 0; i < GLES1Renderer::kTexUnitCount; i++)
    {
        attribsMask.set(GLES1Renderer::TexCoordArrayIndex(i), isTexCoordArrayEnabled(i));
    }

    return attribsMask;
}

AttributesMask GLES1State::getActiveAttributesMask() const
{
    // The program always has 8 attributes enabled.
    return AttributesMask(0xFF);
}

void GLES1State::setHint(GLenum target, GLenum mode)
{
    setDirty(DIRTY_GLES1_HINT_SETTING);
    HintSetting setting = FromGLenum<HintSetting>(mode);
    switch (target)
    {
        case GL_PERSPECTIVE_CORRECTION_HINT:
            mPerspectiveCorrectionHint = setting;
            break;
        case GL_POINT_SMOOTH_HINT:
            mPointSmoothHint = setting;
            break;
        case GL_LINE_SMOOTH_HINT:
            mLineSmoothHint = setting;
            break;
        case GL_FOG_HINT:
            mFogHint = setting;
            break;
        default:
            UNREACHABLE();
    }
}

GLenum GLES1State::getHint(GLenum target)
{
    switch (target)
    {
        case GL_PERSPECTIVE_CORRECTION_HINT:
            return ToGLenum(mPerspectiveCorrectionHint);
        case GL_POINT_SMOOTH_HINT:
            return ToGLenum(mPointSmoothHint);
        case GL_LINE_SMOOTH_HINT:
            return ToGLenum(mLineSmoothHint);
        case GL_FOG_HINT:
            return ToGLenum(mFogHint);
        default:
            UNREACHABLE();
            return 0;
    }
}

void GLES1State::setDirty(DirtyGles1Type type)
{
    mDirtyBits.set(type);
}

void GLES1State::setAllDirty()
{
    mDirtyBits.set();
}

void GLES1State::clearDirty()
{
    mDirtyBits.reset();
}

bool GLES1State::isDirty(DirtyGles1Type type) const
{
    return mDirtyBits.test(type);
}

}  // namespace gl
