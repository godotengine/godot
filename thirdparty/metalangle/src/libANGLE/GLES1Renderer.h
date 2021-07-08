//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// GLES1Renderer.h: Defines GLES1 emulation rendering operations on top of a GLES3
// context. Used by Context.h.

#ifndef LIBANGLE_GLES1_RENDERER_H_
#define LIBANGLE_GLES1_RENDERER_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/angletypes.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace gl
{
class Context;
class GLES1State;
class Program;
class State;
class Shader;
class ShaderProgramManager;

class GLES1Renderer final : angle::NonCopyable
{
  public:
    GLES1Renderer();
    ~GLES1Renderer();

    void onDestroy(Context *context, State *state);

    angle::Result prepareForDraw(PrimitiveMode mode, Context *context, State *glState);

    static int VertexArrayIndex(ClientVertexArrayType type, const GLES1State &gles1);
    static int TexCoordArrayIndex(unsigned int unit);

    void drawTexture(Context *context,
                     State *glState,
                     float x,
                     float y,
                     float z,
                     float width,
                     float height);

    static constexpr int kTexUnitCount = 4;

  private:
    using Mat4Uniform = float[16];
    using Vec4Uniform = float[4];
    using Vec3Uniform = float[3];

    Shader *getShader(ShaderProgramID handle) const;
    Program *getProgram(ShaderProgramID handle) const;

    angle::Result compileShader(Context *context,
                                ShaderType shaderType,
                                const char *src,
                                ShaderProgramID *shaderOut);
    angle::Result linkProgram(Context *context,
                              State *glState,
                              ShaderProgramID vshader,
                              ShaderProgramID fshader,
                              const std::unordered_map<GLint, std::string> &attribLocs,
                              ShaderProgramID *programOut);
    angle::Result initializeRendererProgram(Context *context, State *glState);

    void setUniform1i(Context *context, Program *programObject, GLint loc, GLint value);
    void setUniform1iv(Context *context,
                       Program *programObject,
                       GLint loc,
                       GLint count,
                       const GLint *value);
    void setUniformMatrix4fv(Program *programObject,
                             GLint loc,
                             GLint count,
                             GLboolean transpose,
                             const GLfloat *value);
    void setUniform4fv(Program *programObject, GLint loc, GLint count, const GLfloat *value);
    void setUniform3fv(Program *programObject, GLint loc, GLint count, const GLfloat *value);
    void setUniform2fv(Program *programObject, GLint loc, GLint count, const GLfloat *value);
    void setUniform1f(Program *programObject, GLint loc, GLfloat value);
    void setUniform1fv(Program *programObject, GLint loc, GLint count, const GLfloat *value);

    void setAttributesEnabled(Context *context, State *glState, AttributesMask mask);

    static constexpr int kLightCount     = 8;
    static constexpr int kClipPlaneCount = 6;

    static constexpr int kVertexAttribIndex           = 0;
    static constexpr int kNormalAttribIndex           = 1;
    static constexpr int kColorAttribIndex            = 2;
    static constexpr int kPointSizeAttribIndex        = 3;
    static constexpr int kTextureCoordAttribIndexBase = 4;

    bool mRendererProgramInitialized;
    ShaderProgramManager *mShaderPrograms;

    struct GLES1ProgramState
    {
        ShaderProgramID program;

        GLint projMatrixLoc;
        GLint modelviewMatrixLoc;
        GLint textureMatrixLoc;
        GLint modelviewInvTrLoc;

        // Texturing
        GLint enableTexture2DLoc;
        GLint enableTextureCubeMapLoc;
        std::array<GLint, kTexUnitCount> tex2DSamplerLocs;
        std::array<GLint, kTexUnitCount> texCubeSamplerLocs;

        GLint textureFormatLoc;

        GLint textureEnvModeLoc;
        GLint combineRgbLoc;
        GLint combineAlphaLoc;
        GLint src0rgbLoc;
        GLint src0alphaLoc;
        GLint src1rgbLoc;
        GLint src1alphaLoc;
        GLint src2rgbLoc;
        GLint src2alphaLoc;
        GLint op0rgbLoc;
        GLint op0alphaLoc;
        GLint op1rgbLoc;
        GLint op1alphaLoc;
        GLint op2rgbLoc;
        GLint op2alphaLoc;
        GLint textureEnvColorLoc;
        GLint rgbScaleLoc;
        GLint alphaScaleLoc;
        GLint pointSpriteCoordReplaceLoc;

        // Alpha test
        GLint enableAlphaTestLoc;
        GLint alphaFuncLoc;
        GLint alphaTestRefLoc;

        // Shading, materials, and lighting
        GLint shadeModelFlatLoc;
        GLint enableLightingLoc;
        GLint enableRescaleNormalLoc;
        GLint enableNormalizeLoc;
        GLint enableColorMaterialLoc;

        GLint materialAmbientLoc;
        GLint materialDiffuseLoc;
        GLint materialSpecularLoc;
        GLint materialEmissiveLoc;
        GLint materialSpecularExponentLoc;

        GLint lightModelSceneAmbientLoc;
        GLint lightModelTwoSidedLoc;

        GLint lightEnablesLoc;
        GLint lightAmbientsLoc;
        GLint lightDiffusesLoc;
        GLint lightSpecularsLoc;
        GLint lightPositionsLoc;
        GLint lightDirectionsLoc;
        GLint lightSpotlightExponentsLoc;
        GLint lightSpotlightCutoffAnglesLoc;
        GLint lightAttenuationConstsLoc;
        GLint lightAttenuationLinearsLoc;
        GLint lightAttenuationQuadraticsLoc;

        // Fog
        GLint fogEnableLoc;
        GLint fogModeLoc;
        GLint fogDensityLoc;
        GLint fogStartLoc;
        GLint fogEndLoc;
        GLint fogColorLoc;

        // Clip planes
        GLint enableClipPlanesLoc;
        GLint clipPlaneEnablesLoc;
        GLint clipPlanesLoc;

        // Point rasterization
        GLint pointRasterizationLoc;
        GLint pointSizeMinLoc;
        GLint pointSizeMaxLoc;
        GLint pointDistanceAttenuationLoc;
        GLint pointSpriteEnabledLoc;

        // Draw texture
        GLint enableDrawTextureLoc;
        GLint drawTextureCoordsLoc;
        GLint drawTextureDimsLoc;
        GLint drawTextureNormalizedCropRectLoc;
    };

    struct GLES1UniformBuffers
    {
        std::array<Mat4Uniform, kTexUnitCount> textureMatrices;
        std::array<GLint, kTexUnitCount> tex2DEnables;
        std::array<GLint, kTexUnitCount> texCubeEnables;

        std::array<GLint, kTexUnitCount> texEnvModes;
        std::array<GLint, kTexUnitCount> texCombineRgbs;
        std::array<GLint, kTexUnitCount> texCombineAlphas;

        std::array<GLint, kTexUnitCount> texCombineSrc0Rgbs;
        std::array<GLint, kTexUnitCount> texCombineSrc0Alphas;
        std::array<GLint, kTexUnitCount> texCombineSrc1Rgbs;
        std::array<GLint, kTexUnitCount> texCombineSrc1Alphas;
        std::array<GLint, kTexUnitCount> texCombineSrc2Rgbs;
        std::array<GLint, kTexUnitCount> texCombineSrc2Alphas;
        std::array<GLint, kTexUnitCount> texCombineOp0Rgbs;
        std::array<GLint, kTexUnitCount> texCombineOp0Alphas;
        std::array<GLint, kTexUnitCount> texCombineOp1Rgbs;
        std::array<GLint, kTexUnitCount> texCombineOp1Alphas;
        std::array<GLint, kTexUnitCount> texCombineOp2Rgbs;
        std::array<GLint, kTexUnitCount> texCombineOp2Alphas;
        std::array<Vec4Uniform, kTexUnitCount> texEnvColors;
        std::array<GLfloat, kTexUnitCount> texEnvRgbScales;
        std::array<GLfloat, kTexUnitCount> texEnvAlphaScales;
        std::array<GLint, kTexUnitCount> pointSpriteCoordReplaces;

        // Lighting
        std::array<GLint, kLightCount> lightEnables;
        std::array<Vec4Uniform, kLightCount> lightAmbients;
        std::array<Vec4Uniform, kLightCount> lightDiffuses;
        std::array<Vec4Uniform, kLightCount> lightSpeculars;
        std::array<Vec4Uniform, kLightCount> lightPositions;
        std::array<Vec3Uniform, kLightCount> lightDirections;
        std::array<GLfloat, kLightCount> spotlightExponents;
        std::array<GLfloat, kLightCount> spotlightCutoffAngles;
        std::array<GLfloat, kLightCount> attenuationConsts;
        std::array<GLfloat, kLightCount> attenuationLinears;
        std::array<GLfloat, kLightCount> attenuationQuadratics;

        // Clip planes
        std::array<GLint, kClipPlaneCount> clipPlaneEnables;
        std::array<Vec4Uniform, kClipPlaneCount> clipPlanes;

        // Texture crop rectangles
        std::array<Vec4Uniform, kTexUnitCount> texCropRects;
    };

    GLES1UniformBuffers mUniformBuffers;
    GLES1ProgramState mProgramState;

    bool mDrawTextureEnabled      = false;
    GLfloat mDrawTextureCoords[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GLfloat mDrawTextureDims[2]   = {0.0f, 0.0f};
};

}  // namespace gl

#endif  // LIBANGLE_GLES1_RENDERER_H_
