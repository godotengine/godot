//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// GLES1State.h: Defines the GLES1State class holding the state of
// a GLES1 context.

#ifndef LIBANGLE_GLES1STATE_H_
#define LIBANGLE_GLES1STATE_H_

#include <unordered_set>

#include "common/FixedVector.h"
#include "common/angleutils.h"
#include "common/bitset_utils.h"
#include "common/matrix_utils.h"
#include "common/vector_utils.h"
#include "libANGLE/Caps.h"
#include "libANGLE/angletypes.h"

namespace gl
{

// State types specific to GLES1 contexts, from the OpenGL ES 1.1 spec "State Tables" section
struct TextureCoordF
{
    TextureCoordF();
    TextureCoordF(float _s, float _t, float _r, float _q);
    bool operator==(const TextureCoordF &other) const;

    GLfloat s = 0.0f;
    GLfloat t = 0.0f;
    GLfloat r = 0.0f;
    GLfloat q = 0.0f;
};

struct MaterialParameters
{
    MaterialParameters();

    ColorF ambient;
    ColorF diffuse;
    ColorF specular;
    ColorF emissive;
    GLfloat specularExponent;
};

struct LightModelParameters
{
    LightModelParameters();

    ColorF color;
    bool twoSided;
};

struct LightParameters
{
    LightParameters();
    LightParameters(const LightParameters &other);

    bool enabled                 = false;
    ColorF ambient               = {0.0f, 0.0f, 0.0f, 1.0f};
    ColorF diffuse               = {0.0f, 0.0f, 0.0f, 1.0f};
    ColorF specular              = {0.0f, 0.0f, 0.0f, 1.0f};
    angle::Vector4 position      = {0.0f, 0.0f, 1.0f, 0.0f};
    angle::Vector3 direction     = {0.0f, 0.0f, -1.0f};
    GLfloat spotlightExponent    = 0.0f;
    GLfloat spotlightCutoffAngle = 180.0f;
    GLfloat attenuationConst     = 1.0f;
    GLfloat attenuationLinear    = 0.0f;
    GLfloat attenuationQuadratic = 0.0f;
};

struct FogParameters
{
    FogParameters();

    FogMode mode;
    GLfloat density;
    GLfloat start;
    GLfloat end;
    ColorF color;
};

struct TextureEnvironmentParameters
{
    TextureEnvironmentParameters();
    TextureEnvironmentParameters(const TextureEnvironmentParameters &other);

    TextureEnvMode mode         = TextureEnvMode::Modulate;
    TextureCombine combineRgb   = TextureCombine::Modulate;
    TextureCombine combineAlpha = TextureCombine::Modulate;

    TextureSrc src0Rgb   = TextureSrc::Texture;
    TextureSrc src0Alpha = TextureSrc::Texture;

    TextureSrc src1Rgb   = TextureSrc::Previous;
    TextureSrc src1Alpha = TextureSrc::Previous;

    TextureSrc src2Rgb   = TextureSrc::Constant;
    TextureSrc src2Alpha = TextureSrc::Constant;

    TextureOp op0Rgb   = TextureOp::SrcColor;
    TextureOp op0Alpha = TextureOp::SrcAlpha;

    TextureOp op1Rgb   = TextureOp::SrcColor;
    TextureOp op1Alpha = TextureOp::SrcAlpha;

    TextureOp op2Rgb   = TextureOp::SrcAlpha;
    TextureOp op2Alpha = TextureOp::SrcAlpha;

    ColorF color       = {0.0f, 0.0f, 0.0f, 0.0f};
    GLfloat rgbScale   = 1.0f;
    GLfloat alphaScale = 1.0f;

    bool pointSpriteCoordReplace = false;
};

struct PointParameters
{
    PointParameters();
    PointParameters(const PointParameters &other);

    GLfloat pointSizeMin                    = 0.0f;
    GLfloat pointSizeMax                    = 1.0f;
    GLfloat pointFadeThresholdSize          = 1.0f;
    angle::Vector3 pointDistanceAttenuation = {1.0f, 0.0f, 0.0f};
    GLfloat pointSize                       = 1.0f;
};

struct ClipPlaneParameters
{
    ClipPlaneParameters();
    ClipPlaneParameters(bool enabled, const angle::Vector4 &equation);
    ClipPlaneParameters(const ClipPlaneParameters &other);

    bool enabled;
    angle::Vector4 equation;
};

class Context;
class GLES1Renderer;
class State;

class GLES1State final : angle::NonCopyable
{
  public:
    GLES1State();
    ~GLES1State();

    void initialize(const Context *context, const State *state);

    void setAlphaFunc(AlphaTestFunc func, GLfloat ref);
    void setClientTextureUnit(unsigned int unit);
    unsigned int getClientTextureUnit() const;

    void setCurrentColor(const ColorF &color);
    const ColorF &getCurrentColor() const;

    void setCurrentNormal(const angle::Vector3 &normal);
    const angle::Vector3 &getCurrentNormal() const;

    void setCurrentTextureCoords(unsigned int unit, const TextureCoordF &coords);
    const TextureCoordF &getCurrentTextureCoords(unsigned int unit) const;

    void setMatrixMode(MatrixType mode);
    MatrixType getMatrixMode() const;

    GLint getCurrentMatrixStackDepth(GLenum param) const;

    void pushMatrix();
    void popMatrix();

    using MatrixStack = angle::FixedVector<angle::Mat4, Caps::GlobalMatrixStackDepth>;
    MatrixStack &currentMatrixStack();
    const MatrixStack &currentMatrixStack() const;

    const angle::Mat4 &getModelviewMatrix() const;

    void loadMatrix(const angle::Mat4 &m);
    void multMatrix(const angle::Mat4 &m);

    void setLogicOp(LogicalOperation opcodePacked);

    void setClientStateEnabled(ClientVertexArrayType clientState, bool enable);
    void setTexCoordArrayEnabled(unsigned int unit, bool enable);
    bool isClientStateEnabled(ClientVertexArrayType clientState) const;
    bool isTexCoordArrayEnabled(unsigned int unit) const;
    bool isTextureTargetEnabled(unsigned int unit, const TextureType type) const;

    LightModelParameters &lightModelParameters();
    const LightModelParameters &lightModelParameters() const;

    LightParameters &lightParameters(unsigned int light);
    const LightParameters &lightParameters(unsigned int light) const;

    MaterialParameters &materialParameters();
    const MaterialParameters &materialParameters() const;
    bool isColorMaterialEnabled() const;

    void setShadeModel(ShadingModel model);

    void setClipPlane(unsigned int plane, const GLfloat *equation);
    void getClipPlane(unsigned int plane, GLfloat *equation) const;

    FogParameters &fogParameters();
    const FogParameters &fogParameters() const;

    TextureEnvironmentParameters &textureEnvironment(unsigned int unit);
    const TextureEnvironmentParameters &textureEnvironment(unsigned int unit) const;

    PointParameters &pointParameters();
    const PointParameters &pointParameters() const;

    AttributesMask getVertexArraysAttributeMask() const;
    AttributesMask getActiveAttributesMask() const;

    void setHint(GLenum target, GLenum mode);
    GLenum getHint(GLenum target);

  private:
    friend class State;
    friend class GLES1Renderer;

    // Back pointer for reading from State.
    const State *mGLState;

    enum DirtyGles1Type
    {
        DIRTY_GLES1_TEXTURE_UNIT_ENABLE = 0,
        DIRTY_GLES1_CLIENT_STATE_ENABLE,
        DIRTY_GLES1_FEATURE_ENABLE,
        DIRTY_GLES1_CURRENT_VECTOR,
        DIRTY_GLES1_CLIENT_ACTIVE_TEXTURE,
        DIRTY_GLES1_MATRICES,
        DIRTY_GLES1_TEXTURE_ENVIRONMENT,
        DIRTY_GLES1_MATERIAL,
        DIRTY_GLES1_LIGHTS,
        DIRTY_GLES1_FOG,
        DIRTY_GLES1_SHADE_MODEL,
        DIRTY_GLES1_POINT_PARAMETERS,
        DIRTY_GLES1_ALPHA_TEST,
        DIRTY_GLES1_LOGIC_OP,
        DIRTY_GLES1_CLIP_PLANES,
        DIRTY_GLES1_HINT_SETTING,
        DIRTY_GLES1_MAX,
    };
    using DirtyBits = angle::BitSet<DIRTY_GLES1_MAX>;
    DirtyBits mDirtyBits;

    void setDirty(DirtyGles1Type type);
    void setAllDirty();
    void clearDirty();
    bool isDirty(DirtyGles1Type type) const;

    // All initial state values come from the
    // OpenGL ES 1.1 spec.
    std::vector<angle::PackedEnumBitSet<TextureType>> mTexUnitEnables;

    // Table 6.4, 6.5 (IsEnabled)
    bool mVertexArrayEnabled;
    bool mNormalArrayEnabled;
    bool mColorArrayEnabled;
    bool mPointSizeArrayEnabled;
    std::vector<bool> mTexCoordArrayEnabled;

    // Table 6.7-6.16 (IsEnabled)
    bool mLineSmoothEnabled;
    bool mPointSmoothEnabled;
    bool mPointSpriteEnabled;
    bool mAlphaTestEnabled;
    bool mLogicOpEnabled;
    bool mLightingEnabled;
    bool mFogEnabled;
    bool mRescaleNormalEnabled;
    bool mNormalizeEnabled;
    bool mColorMaterialEnabled;
    bool mReflectionMapEnabled;

    // Table 6.3
    ColorF mCurrentColor;
    angle::Vector3 mCurrentNormal;
    // Invariant: mCurrentTextureCoords size is == GL_MAX_TEXTURE_UNITS.
    std::vector<TextureCoordF> mCurrentTextureCoords;

    // Table 6.4
    unsigned int mClientActiveTexture;

    // Table 6.7
    MatrixType mMatrixMode;
    MatrixStack mProjectionMatrices;
    MatrixStack mModelviewMatrices;
    std::vector<MatrixStack> mTextureMatrices;

    // Table 6.15
    using TextureEnvironments = std::vector<TextureEnvironmentParameters>;
    TextureEnvironments mTextureEnvironments;

    // Table 6.9, 2.8
    MaterialParameters mMaterial;
    LightModelParameters mLightModel;

    // Table 6.10
    std::vector<LightParameters> mLights;

    // Table 6.8
    FogParameters mFog;
    ShadingModel mShadeModel;

    // Table 6.11
    PointParameters mPointParameters;

    // Table 6.16
    AlphaTestFunc mAlphaTestFunc;
    GLfloat mAlphaTestRef;
    LogicalOperation mLogicOp;

    // Table 6.7
    std::vector<ClipPlaneParameters> mClipPlanes;

    // Table 6.19
    HintSetting mLineSmoothHint;
    HintSetting mPointSmoothHint;
    HintSetting mPerspectiveCorrectionHint;
    HintSetting mFogHint;
};

}  // namespace gl

#endif  // LIBANGLE_GLES1STATE_H_
