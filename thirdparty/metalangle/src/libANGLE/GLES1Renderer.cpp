//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// GLES1Renderer.cpp: Implements the GLES1Renderer renderer.

#include "libANGLE/GLES1Renderer.h"

#include <string.h>
#include <iterator>
#include <sstream>
#include <vector>

#include "libANGLE/Context.h"
#include "libANGLE/Context.inl.h"
#include "libANGLE/Program.h"
#include "libANGLE/ResourceManager.h"
#include "libANGLE/Shader.h"
#include "libANGLE/State.h"
#include "libANGLE/renderer/ContextImpl.h"

namespace
{
#include "libANGLE/GLES1Shaders.inc"
}  // anonymous namespace

namespace gl
{

GLES1Renderer::GLES1Renderer() : mRendererProgramInitialized(false) {}

void GLES1Renderer::onDestroy(Context *context, State *state)
{
    if (mRendererProgramInitialized)
    {
        (void)state->setProgram(context, 0);

        mShaderPrograms->deleteProgram(context, {mProgramState.program});
        mShaderPrograms->release(context);
        mShaderPrograms             = nullptr;
        mRendererProgramInitialized = false;
    }
}

GLES1Renderer::~GLES1Renderer() = default;

angle::Result GLES1Renderer::prepareForDraw(PrimitiveMode mode, Context *context, State *glState)
{
    ANGLE_TRY(initializeRendererProgram(context, glState));

    GLES1State &gles1State = glState->gles1();

    Program *programObject = getProgram(mProgramState.program);

    GLES1UniformBuffers &uniformBuffers = mUniformBuffers;

    // If anything is dirty in gles1 or the common parts of gles1/2, just redo these parts
    // completely for now.

    // Feature enables
    {
        setUniform1i(context, programObject, mProgramState.enableAlphaTestLoc,
                     glState->getEnableFeature(GL_ALPHA_TEST));
        setUniform1i(context, programObject, mProgramState.enableLightingLoc,
                     glState->getEnableFeature(GL_LIGHTING));
        setUniform1i(context, programObject, mProgramState.enableRescaleNormalLoc,
                     glState->getEnableFeature(GL_RESCALE_NORMAL));
        setUniform1i(context, programObject, mProgramState.enableNormalizeLoc,
                     glState->getEnableFeature(GL_NORMALIZE));
        setUniform1i(context, programObject, mProgramState.enableColorMaterialLoc,
                     glState->getEnableFeature(GL_COLOR_MATERIAL));
        setUniform1i(context, programObject, mProgramState.fogEnableLoc,
                     glState->getEnableFeature(GL_FOG));

        bool enableClipPlanes = false;
        for (int i = 0; i < kClipPlaneCount; i++)
        {
            uniformBuffers.clipPlaneEnables[i] = glState->getEnableFeature(GL_CLIP_PLANE0 + i);
            enableClipPlanes = enableClipPlanes || uniformBuffers.clipPlaneEnables[i];
        }

        setUniform1i(context, programObject, mProgramState.enableClipPlanesLoc, enableClipPlanes);
    }

    // Texture unit enables and format info
    {
        std::array<GLint, kTexUnitCount> &tex2DEnables   = uniformBuffers.tex2DEnables;
        std::array<GLint, kTexUnitCount> &texCubeEnables = uniformBuffers.texCubeEnables;

        std::vector<int> tex2DFormats = {GL_RGBA, GL_RGBA, GL_RGBA, GL_RGBA};

        Vec4Uniform *cropRectBuffer = uniformBuffers.texCropRects.data();

        for (int i = 0; i < kTexUnitCount; i++)
        {
            // GL_OES_cube_map allows only one of TEXTURE_2D / TEXTURE_CUBE_MAP
            // to be enabled per unit, thankfully. From the extension text:
            //
            //  --  Section 3.8.10 "Texture Application"
            //
            //      Replace the beginning sentences of the first paragraph (page 138)
            //      with:
            //
            //      "Texturing is enabled or disabled using the generic Enable
            //      and Disable commands, respectively, with the symbolic constants
            //      TEXTURE_2D or TEXTURE_CUBE_MAP_OES to enable the two-dimensional or cube
            //      map texturing respectively.  If the cube map texture and the two-
            //      dimensional texture are enabled, then cube map texturing is used.  If
            //      texturing is disabled, a rasterized fragment is passed on unaltered to the
            //      next stage of the GL (although its texture coordinates may be discarded).
            //      Otherwise, a texture value is found according to the parameter values of
            //      the currently bound texture image of the appropriate dimensionality.

            texCubeEnables[i] = gles1State.isTextureTargetEnabled(i, TextureType::CubeMap);
            tex2DEnables[i] =
                !texCubeEnables[i] && (gles1State.isTextureTargetEnabled(i, TextureType::_2D));

            Texture *curr2DTexture = glState->getSamplerTexture(i, TextureType::_2D);
            if (curr2DTexture)
            {
                tex2DFormats[i] = gl::GetUnsizedFormat(
                    curr2DTexture->getFormat(TextureTarget::_2D, 0).info->internalFormat);

                const gl::Rectangle &cropRect = curr2DTexture->getCrop();

                GLfloat textureWidth =
                    static_cast<GLfloat>(curr2DTexture->getWidth(TextureTarget::_2D, 0));
                GLfloat textureHeight =
                    static_cast<GLfloat>(curr2DTexture->getHeight(TextureTarget::_2D, 0));

                if (textureWidth > 0.0f && textureHeight > 0.0f)
                {
                    cropRectBuffer[i][0] = cropRect.x / textureWidth;
                    cropRectBuffer[i][1] = cropRect.y / textureHeight;
                    cropRectBuffer[i][2] = cropRect.width / textureWidth;
                    cropRectBuffer[i][3] = cropRect.height / textureHeight;
                }
            }
        }

        setUniform1iv(context, programObject, mProgramState.enableTexture2DLoc, kTexUnitCount,
                      tex2DEnables.data());
        setUniform1iv(context, programObject, mProgramState.enableTextureCubeMapLoc, kTexUnitCount,
                      texCubeEnables.data());

        setUniform1iv(context, programObject, mProgramState.textureFormatLoc, kTexUnitCount,
                      tex2DFormats.data());

        setUniform4fv(programObject, mProgramState.drawTextureNormalizedCropRectLoc, kTexUnitCount,
                      reinterpret_cast<GLfloat *>(cropRectBuffer));
    }

    // Client state / current vector enables
    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_CLIENT_STATE_ENABLE) ||
        gles1State.isDirty(GLES1State::DIRTY_GLES1_CURRENT_VECTOR))
    {
        if (!gles1State.isClientStateEnabled(ClientVertexArrayType::Normal))
        {
            const angle::Vector3 normal = gles1State.getCurrentNormal();
            context->vertexAttrib3f(kNormalAttribIndex, normal.x(), normal.y(), normal.z());
        }

        if (!gles1State.isClientStateEnabled(ClientVertexArrayType::Color))
        {
            const ColorF color = gles1State.getCurrentColor();
            context->vertexAttrib4f(kColorAttribIndex, color.red, color.green, color.blue,
                                    color.alpha);
        }

        if (!gles1State.isClientStateEnabled(ClientVertexArrayType::PointSize))
        {
            GLfloat pointSize = gles1State.mPointParameters.pointSize;
            context->vertexAttrib1f(kPointSizeAttribIndex, pointSize);
        }

        for (int i = 0; i < kTexUnitCount; i++)
        {
            if (!gles1State.mTexCoordArrayEnabled[i])
            {
                const TextureCoordF texcoord = gles1State.getCurrentTextureCoords(i);
                context->vertexAttrib4f(kTextureCoordAttribIndexBase + i, texcoord.s, texcoord.t,
                                        texcoord.r, texcoord.q);
            }
        }
    }

    // Matrices
    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_MATRICES))
    {
        angle::Mat4 proj = gles1State.mProjectionMatrices.back();
        setUniformMatrix4fv(programObject, mProgramState.projMatrixLoc, 1, GL_FALSE, proj.data());

        angle::Mat4 modelview = gles1State.mModelviewMatrices.back();
        setUniformMatrix4fv(programObject, mProgramState.modelviewMatrixLoc, 1, GL_FALSE,
                            modelview.data());

        angle::Mat4 modelviewInvTr = modelview.transpose().inverse();
        setUniformMatrix4fv(programObject, mProgramState.modelviewInvTrLoc, 1, GL_FALSE,
                            modelviewInvTr.data());

        Mat4Uniform *textureMatrixBuffer = uniformBuffers.textureMatrices.data();

        for (int i = 0; i < kTexUnitCount; i++)
        {
            angle::Mat4 textureMatrix = gles1State.mTextureMatrices[i].back();
            memcpy(textureMatrixBuffer + i, textureMatrix.data(), sizeof(Mat4Uniform));
        }

        setUniformMatrix4fv(programObject, mProgramState.textureMatrixLoc, kTexUnitCount, GL_FALSE,
                            reinterpret_cast<float *>(uniformBuffers.textureMatrices.data()));
    }

    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_TEXTURE_ENVIRONMENT))
    {
        for (int i = 0; i < kTexUnitCount; i++)
        {
            const auto &env = gles1State.textureEnvironment(i);

            uniformBuffers.texEnvModes[i]      = ToGLenum(env.mode);
            uniformBuffers.texCombineRgbs[i]   = ToGLenum(env.combineRgb);
            uniformBuffers.texCombineAlphas[i] = ToGLenum(env.combineAlpha);

            uniformBuffers.texCombineSrc0Rgbs[i]   = ToGLenum(env.src0Rgb);
            uniformBuffers.texCombineSrc0Alphas[i] = ToGLenum(env.src0Alpha);
            uniformBuffers.texCombineSrc1Rgbs[i]   = ToGLenum(env.src1Rgb);
            uniformBuffers.texCombineSrc1Alphas[i] = ToGLenum(env.src1Alpha);
            uniformBuffers.texCombineSrc2Rgbs[i]   = ToGLenum(env.src2Rgb);
            uniformBuffers.texCombineSrc2Alphas[i] = ToGLenum(env.src2Alpha);

            uniformBuffers.texCombineOp0Rgbs[i]   = ToGLenum(env.op0Rgb);
            uniformBuffers.texCombineOp0Alphas[i] = ToGLenum(env.op0Alpha);
            uniformBuffers.texCombineOp1Rgbs[i]   = ToGLenum(env.op1Rgb);
            uniformBuffers.texCombineOp1Alphas[i] = ToGLenum(env.op1Alpha);
            uniformBuffers.texCombineOp2Rgbs[i]   = ToGLenum(env.op2Rgb);
            uniformBuffers.texCombineOp2Alphas[i] = ToGLenum(env.op2Alpha);

            uniformBuffers.texEnvColors[i][0] = env.color.red;
            uniformBuffers.texEnvColors[i][1] = env.color.green;
            uniformBuffers.texEnvColors[i][2] = env.color.blue;
            uniformBuffers.texEnvColors[i][3] = env.color.alpha;

            uniformBuffers.texEnvRgbScales[i]   = env.rgbScale;
            uniformBuffers.texEnvAlphaScales[i] = env.alphaScale;

            uniformBuffers.pointSpriteCoordReplaces[i] = env.pointSpriteCoordReplace;
        }

        setUniform1iv(context, programObject, mProgramState.textureEnvModeLoc, kTexUnitCount,
                      uniformBuffers.texEnvModes.data());
        setUniform1iv(context, programObject, mProgramState.combineRgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineRgbs.data());
        setUniform1iv(context, programObject, mProgramState.combineAlphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineAlphas.data());

        setUniform1iv(context, programObject, mProgramState.src0rgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineSrc0Rgbs.data());
        setUniform1iv(context, programObject, mProgramState.src0alphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineSrc0Alphas.data());
        setUniform1iv(context, programObject, mProgramState.src1rgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineSrc1Rgbs.data());
        setUniform1iv(context, programObject, mProgramState.src1alphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineSrc1Alphas.data());
        setUniform1iv(context, programObject, mProgramState.src2rgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineSrc2Rgbs.data());
        setUniform1iv(context, programObject, mProgramState.src2alphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineSrc2Alphas.data());

        setUniform1iv(context, programObject, mProgramState.op0rgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineOp0Rgbs.data());
        setUniform1iv(context, programObject, mProgramState.op0alphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineOp0Alphas.data());
        setUniform1iv(context, programObject, mProgramState.op1rgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineOp1Rgbs.data());
        setUniform1iv(context, programObject, mProgramState.op1alphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineOp1Alphas.data());
        setUniform1iv(context, programObject, mProgramState.op2rgbLoc, kTexUnitCount,
                      uniformBuffers.texCombineOp2Rgbs.data());
        setUniform1iv(context, programObject, mProgramState.op2alphaLoc, kTexUnitCount,
                      uniformBuffers.texCombineOp2Alphas.data());

        setUniform4fv(programObject, mProgramState.textureEnvColorLoc, kTexUnitCount,
                      reinterpret_cast<float *>(uniformBuffers.texEnvColors.data()));
        setUniform1fv(programObject, mProgramState.rgbScaleLoc, kTexUnitCount,
                      uniformBuffers.texEnvRgbScales.data());
        setUniform1fv(programObject, mProgramState.alphaScaleLoc, kTexUnitCount,
                      uniformBuffers.texEnvAlphaScales.data());

        setUniform1iv(context, programObject, mProgramState.pointSpriteCoordReplaceLoc,
                      kTexUnitCount, uniformBuffers.pointSpriteCoordReplaces.data());
    }

    // Alpha test
    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_ALPHA_TEST))
    {
        setUniform1i(context, programObject, mProgramState.alphaFuncLoc,
                     ToGLenum(gles1State.mAlphaTestFunc));
        setUniform1f(programObject, mProgramState.alphaTestRefLoc, gles1State.mAlphaTestRef);
    }

    // Shading, materials, and lighting
    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_SHADE_MODEL))
    {
        setUniform1i(context, programObject, mProgramState.shadeModelFlatLoc,
                     gles1State.mShadeModel == ShadingModel::Flat);
    }

    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_MATERIAL))
    {
        const auto &material = gles1State.mMaterial;

        setUniform4fv(programObject, mProgramState.materialAmbientLoc, 1, material.ambient.data());
        setUniform4fv(programObject, mProgramState.materialDiffuseLoc, 1, material.diffuse.data());
        setUniform4fv(programObject, mProgramState.materialSpecularLoc, 1,
                      material.specular.data());
        setUniform4fv(programObject, mProgramState.materialEmissiveLoc, 1,
                      material.emissive.data());
        setUniform1f(programObject, mProgramState.materialSpecularExponentLoc,
                     material.specularExponent);
    }

    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_LIGHTS))
    {
        const auto &lightModel = gles1State.mLightModel;

        setUniform4fv(programObject, mProgramState.lightModelSceneAmbientLoc, 1,
                      lightModel.color.data());

        // TODO (lfy@google.com): Implement two-sided lighting model
        // gl->uniform1i(mProgramState.lightModelTwoSidedLoc, lightModel.twoSided);

        for (int i = 0; i < kLightCount; i++)
        {
            const auto &light              = gles1State.mLights[i];
            uniformBuffers.lightEnables[i] = light.enabled;
            memcpy(uniformBuffers.lightAmbients.data() + i, light.ambient.data(),
                   sizeof(Vec4Uniform));
            memcpy(uniformBuffers.lightDiffuses.data() + i, light.diffuse.data(),
                   sizeof(Vec4Uniform));
            memcpy(uniformBuffers.lightSpeculars.data() + i, light.specular.data(),
                   sizeof(Vec4Uniform));
            memcpy(uniformBuffers.lightPositions.data() + i, light.position.data(),
                   sizeof(Vec4Uniform));
            memcpy(uniformBuffers.lightDirections.data() + i, light.direction.data(),
                   sizeof(Vec3Uniform));
            uniformBuffers.spotlightExponents[i]    = light.spotlightExponent;
            uniformBuffers.spotlightCutoffAngles[i] = light.spotlightCutoffAngle;
            uniformBuffers.attenuationConsts[i]     = light.attenuationConst;
            uniformBuffers.attenuationLinears[i]    = light.attenuationLinear;
            uniformBuffers.attenuationQuadratics[i] = light.attenuationQuadratic;
        }

        setUniform1iv(context, programObject, mProgramState.lightEnablesLoc, kLightCount,
                      uniformBuffers.lightEnables.data());
        setUniform4fv(programObject, mProgramState.lightAmbientsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.lightAmbients.data()));
        setUniform4fv(programObject, mProgramState.lightDiffusesLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.lightDiffuses.data()));
        setUniform4fv(programObject, mProgramState.lightSpecularsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.lightSpeculars.data()));
        setUniform4fv(programObject, mProgramState.lightPositionsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.lightPositions.data()));
        setUniform3fv(programObject, mProgramState.lightDirectionsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.lightDirections.data()));
        setUniform1fv(programObject, mProgramState.lightSpotlightExponentsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.spotlightExponents.data()));
        setUniform1fv(programObject, mProgramState.lightSpotlightCutoffAnglesLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.spotlightCutoffAngles.data()));
        setUniform1fv(programObject, mProgramState.lightAttenuationConstsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.attenuationConsts.data()));
        setUniform1fv(programObject, mProgramState.lightAttenuationLinearsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.attenuationLinears.data()));
        setUniform1fv(programObject, mProgramState.lightAttenuationQuadraticsLoc, kLightCount,
                      reinterpret_cast<float *>(uniformBuffers.attenuationQuadratics.data()));
    }

    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_FOG))
    {
        const FogParameters &fog = gles1State.fogParameters();
        setUniform1i(context, programObject, mProgramState.fogModeLoc, ToGLenum(fog.mode));
        setUniform1f(programObject, mProgramState.fogDensityLoc, fog.density);
        setUniform1f(programObject, mProgramState.fogStartLoc, fog.start);
        setUniform1f(programObject, mProgramState.fogEndLoc, fog.end);
        setUniform4fv(programObject, mProgramState.fogColorLoc, 1, fog.color.data());
    }

    // Clip planes
    if (gles1State.isDirty(GLES1State::DIRTY_GLES1_CLIP_PLANES))
    {
        bool enableClipPlanes = false;
        for (int i = 0; i < kClipPlaneCount; i++)
        {
            uniformBuffers.clipPlaneEnables[i] = glState->getEnableFeature(GL_CLIP_PLANE0 + i);
            enableClipPlanes = enableClipPlanes || uniformBuffers.clipPlaneEnables[i];
            gles1State.getClipPlane(
                i, reinterpret_cast<float *>(uniformBuffers.clipPlanes.data() + i));
        }

        setUniform1i(context, programObject, mProgramState.enableClipPlanesLoc, enableClipPlanes);
        setUniform1iv(context, programObject, mProgramState.clipPlaneEnablesLoc, kClipPlaneCount,
                      uniformBuffers.clipPlaneEnables.data());
        setUniform4fv(programObject, mProgramState.clipPlanesLoc, kClipPlaneCount,
                      reinterpret_cast<float *>(uniformBuffers.clipPlanes.data()));
    }

    // Point rasterization
    {
        const PointParameters &pointParams = gles1State.mPointParameters;

        setUniform1i(context, programObject, mProgramState.pointRasterizationLoc,
                     mode == PrimitiveMode::Points);
        setUniform1i(context, programObject, mProgramState.pointSpriteEnabledLoc,
                     glState->getEnableFeature(GL_POINT_SPRITE_OES));
        setUniform1f(programObject, mProgramState.pointSizeMinLoc, pointParams.pointSizeMin);
        setUniform1f(programObject, mProgramState.pointSizeMaxLoc, pointParams.pointSizeMax);
        setUniform3fv(programObject, mProgramState.pointDistanceAttenuationLoc, 1,
                      pointParams.pointDistanceAttenuation.data());
    }

    // Draw texture
    {
        setUniform1i(context, programObject, mProgramState.enableDrawTextureLoc,
                     mDrawTextureEnabled ? 1 : 0);
        setUniform4fv(programObject, mProgramState.drawTextureCoordsLoc, 1, mDrawTextureCoords);
        setUniform2fv(programObject, mProgramState.drawTextureDimsLoc, 1, mDrawTextureDims);
    }

    gles1State.clearDirty();
    // None of those are changes in sampler, so there is no need to set the GL_PROGRAM dirty.
    // Otherwise, put the dirtying here.

    return angle::Result::Continue;
}

// static
int GLES1Renderer::VertexArrayIndex(ClientVertexArrayType type, const GLES1State &gles1)
{
    switch (type)
    {
        case ClientVertexArrayType::Vertex:
            return kVertexAttribIndex;
        case ClientVertexArrayType::Normal:
            return kNormalAttribIndex;
        case ClientVertexArrayType::Color:
            return kColorAttribIndex;
        case ClientVertexArrayType::PointSize:
            return kPointSizeAttribIndex;
        case ClientVertexArrayType::TextureCoord:
            return kTextureCoordAttribIndexBase + gles1.getClientTextureUnit();
        default:
            UNREACHABLE();
            return 0;
    }
}

// static
int GLES1Renderer::TexCoordArrayIndex(unsigned int unit)
{
    return kTextureCoordAttribIndexBase + unit;
}

void GLES1Renderer::drawTexture(Context *context,
                                State *glState,
                                float x,
                                float y,
                                float z,
                                float width,
                                float height)
{

    // get viewport
    const gl::Rectangle &viewport = glState->getViewport();

    // Translate from viewport to NDC for feeding the shader.
    // Recenter, rescale. (e.g., [0, 0, 1080, 1920] -> [-1, -1, 1, 1])
    float xNdc = scaleScreenCoordinateToNdc(x, static_cast<GLfloat>(viewport.width));
    float yNdc = scaleScreenCoordinateToNdc(y, static_cast<GLfloat>(viewport.height));
    float wNdc = scaleScreenDimensionToNdc(width, static_cast<GLfloat>(viewport.width));
    float hNdc = scaleScreenDimensionToNdc(height, static_cast<GLfloat>(viewport.height));

    float zNdc = 2.0f * clamp(z, 0.0f, 1.0f) - 1.0f;

    mDrawTextureCoords[0] = xNdc;
    mDrawTextureCoords[1] = yNdc;
    mDrawTextureCoords[2] = zNdc;

    mDrawTextureDims[0] = wNdc;
    mDrawTextureDims[1] = hNdc;

    mDrawTextureEnabled = true;

    AttributesMask prevAttributesMask = glState->gles1().getVertexArraysAttributeMask();

    setAttributesEnabled(context, glState, AttributesMask());

    glState->gles1().setAllDirty();

    context->drawArrays(PrimitiveMode::Triangles, 0, 6);

    setAttributesEnabled(context, glState, prevAttributesMask);

    mDrawTextureEnabled = false;
}

Shader *GLES1Renderer::getShader(ShaderProgramID handle) const
{
    return mShaderPrograms->getShader(handle);
}

Program *GLES1Renderer::getProgram(ShaderProgramID handle) const
{
    return mShaderPrograms->getProgram(handle);
}

angle::Result GLES1Renderer::compileShader(Context *context,
                                           ShaderType shaderType,
                                           const char *src,
                                           ShaderProgramID *shaderOut)
{
    rx::ContextImpl *implementation = context->getImplementation();
    const Limitations &limitations  = implementation->getNativeLimitations();

    ShaderProgramID shader = mShaderPrograms->createShader(implementation, limitations, shaderType);

    Shader *shaderObject = getShader(shader);
    ANGLE_CHECK(context, shaderObject, "Missing shader object", GL_INVALID_OPERATION);

    shaderObject->setSource(1, &src, nullptr);
    shaderObject->compile(context);

    *shaderOut = shader;

    if (!shaderObject->isCompiled())
    {
        GLint infoLogLength = shaderObject->getInfoLogLength();
        std::vector<char> infoLog(infoLogLength, 0);
        shaderObject->getInfoLog(infoLogLength - 1, nullptr, infoLog.data());

        ERR() << "Internal GLES 1 shader compile failed. Info log: " << infoLog.data();
        ANGLE_CHECK(context, false, "GLES1Renderer shader compile failed.", GL_INVALID_OPERATION);
        return angle::Result::Stop;
    }

    return angle::Result::Continue;
}

angle::Result GLES1Renderer::linkProgram(Context *context,
                                         State *glState,
                                         ShaderProgramID vertexShader,
                                         ShaderProgramID fragmentShader,
                                         const std::unordered_map<GLint, std::string> &attribLocs,
                                         ShaderProgramID *programOut)
{
    ShaderProgramID program = mShaderPrograms->createProgram(context->getImplementation());

    Program *programObject = getProgram(program);
    ANGLE_CHECK(context, programObject, "Missing program object", GL_INVALID_OPERATION);

    *programOut = program;

    programObject->attachShader(getShader(vertexShader));
    programObject->attachShader(getShader(fragmentShader));

    for (auto it : attribLocs)
    {
        GLint index             = it.first;
        const std::string &name = it.second;
        programObject->bindAttributeLocation(index, name.c_str());
    }

    ANGLE_TRY(programObject->link(context));
    programObject->resolveLink(context);

    ANGLE_TRY(glState->onProgramExecutableChange(context, programObject));

    if (!programObject->isLinked())
    {
        GLint infoLogLength = programObject->getInfoLogLength();
        std::vector<char> infoLog(infoLogLength, 0);
        programObject->getInfoLog(infoLogLength - 1, nullptr, infoLog.data());

        ERR() << "Internal GLES 1 shader link failed. Info log: " << infoLog.data();
        ANGLE_CHECK(context, false, "GLES1Renderer program link failed.", GL_INVALID_OPERATION);
        return angle::Result::Stop;
    }

    programObject->detachShader(context, getShader(vertexShader));
    programObject->detachShader(context, getShader(fragmentShader));

    return angle::Result::Continue;
}

angle::Result GLES1Renderer::initializeRendererProgram(Context *context, State *glState)
{
    if (mRendererProgramInitialized)
    {
        return angle::Result::Continue;
    }

    mShaderPrograms = new ShaderProgramManager();

    ShaderProgramID vertexShader;
    ShaderProgramID fragmentShader;

    ANGLE_TRY(compileShader(context, ShaderType::Vertex, kGLES1DrawVShader, &vertexShader));

    std::stringstream fragmentStream;
    fragmentStream << kGLES1DrawFShaderHeader;
    fragmentStream << kGLES1DrawFShaderUniformDefs;
    fragmentStream << kGLES1DrawFShaderFunctions;
    fragmentStream << kGLES1DrawFShaderMultitexturing;
    fragmentStream << kGLES1DrawFShaderMain;

    ANGLE_TRY(compileShader(context, ShaderType::Fragment, fragmentStream.str().c_str(),
                            &fragmentShader));

    std::unordered_map<GLint, std::string> attribLocs;

    attribLocs[(GLint)kVertexAttribIndex]    = "pos";
    attribLocs[(GLint)kNormalAttribIndex]    = "normal";
    attribLocs[(GLint)kColorAttribIndex]     = "color";
    attribLocs[(GLint)kPointSizeAttribIndex] = "pointsize";

    for (int i = 0; i < kTexUnitCount; i++)
    {
        std::stringstream ss;
        ss << "texcoord" << i;
        attribLocs[kTextureCoordAttribIndexBase + i] = ss.str();
    }

    ANGLE_TRY(linkProgram(context, glState, vertexShader, fragmentShader, attribLocs,
                          &mProgramState.program));

    mShaderPrograms->deleteShader(context, vertexShader);
    mShaderPrograms->deleteShader(context, fragmentShader);

    Program *programObject = getProgram(mProgramState.program);

    mProgramState.projMatrixLoc      = programObject->getUniformLocation("projection");
    mProgramState.modelviewMatrixLoc = programObject->getUniformLocation("modelview");
    mProgramState.textureMatrixLoc   = programObject->getUniformLocation("texture_matrix");
    mProgramState.modelviewInvTrLoc  = programObject->getUniformLocation("modelview_invtr");

    for (int i = 0; i < kTexUnitCount; i++)
    {
        std::stringstream ss2d;
        std::stringstream sscube;

        ss2d << "tex_sampler" << i;
        sscube << "tex_cube_sampler" << i;

        mProgramState.tex2DSamplerLocs[i] = programObject->getUniformLocation(ss2d.str().c_str());
        mProgramState.texCubeSamplerLocs[i] =
            programObject->getUniformLocation(sscube.str().c_str());
    }

    mProgramState.enableTexture2DLoc = programObject->getUniformLocation("enable_texture_2d");
    mProgramState.enableTextureCubeMapLoc =
        programObject->getUniformLocation("enable_texture_cube_map");

    mProgramState.textureFormatLoc   = programObject->getUniformLocation("texture_format");
    mProgramState.textureEnvModeLoc  = programObject->getUniformLocation("texture_env_mode");
    mProgramState.combineRgbLoc      = programObject->getUniformLocation("combine_rgb");
    mProgramState.combineAlphaLoc    = programObject->getUniformLocation("combine_alpha");
    mProgramState.src0rgbLoc         = programObject->getUniformLocation("src0_rgb");
    mProgramState.src0alphaLoc       = programObject->getUniformLocation("src0_alpha");
    mProgramState.src1rgbLoc         = programObject->getUniformLocation("src1_rgb");
    mProgramState.src1alphaLoc       = programObject->getUniformLocation("src1_alpha");
    mProgramState.src2rgbLoc         = programObject->getUniformLocation("src2_rgb");
    mProgramState.src2alphaLoc       = programObject->getUniformLocation("src2_alpha");
    mProgramState.op0rgbLoc          = programObject->getUniformLocation("op0_rgb");
    mProgramState.op0alphaLoc        = programObject->getUniformLocation("op0_alpha");
    mProgramState.op1rgbLoc          = programObject->getUniformLocation("op1_rgb");
    mProgramState.op1alphaLoc        = programObject->getUniformLocation("op1_alpha");
    mProgramState.op2rgbLoc          = programObject->getUniformLocation("op2_rgb");
    mProgramState.op2alphaLoc        = programObject->getUniformLocation("op2_alpha");
    mProgramState.textureEnvColorLoc = programObject->getUniformLocation("texture_env_color");
    mProgramState.rgbScaleLoc        = programObject->getUniformLocation("texture_env_rgb_scale");
    mProgramState.alphaScaleLoc      = programObject->getUniformLocation("texture_env_alpha_scale");
    mProgramState.pointSpriteCoordReplaceLoc =
        programObject->getUniformLocation("point_sprite_coord_replace");

    mProgramState.enableAlphaTestLoc = programObject->getUniformLocation("enable_alpha_test");
    mProgramState.alphaFuncLoc       = programObject->getUniformLocation("alpha_func");
    mProgramState.alphaTestRefLoc    = programObject->getUniformLocation("alpha_test_ref");

    mProgramState.shadeModelFlatLoc = programObject->getUniformLocation("shade_model_flat");
    mProgramState.enableLightingLoc = programObject->getUniformLocation("enable_lighting");
    mProgramState.enableRescaleNormalLoc =
        programObject->getUniformLocation("enable_rescale_normal");
    mProgramState.enableNormalizeLoc = programObject->getUniformLocation("enable_normalize");
    mProgramState.enableColorMaterialLoc =
        programObject->getUniformLocation("enable_color_material");

    mProgramState.materialAmbientLoc  = programObject->getUniformLocation("material_ambient");
    mProgramState.materialDiffuseLoc  = programObject->getUniformLocation("material_diffuse");
    mProgramState.materialSpecularLoc = programObject->getUniformLocation("material_specular");
    mProgramState.materialEmissiveLoc = programObject->getUniformLocation("material_emissive");
    mProgramState.materialSpecularExponentLoc =
        programObject->getUniformLocation("material_specular_exponent");

    mProgramState.lightModelSceneAmbientLoc =
        programObject->getUniformLocation("light_model_scene_ambient");
    mProgramState.lightModelTwoSidedLoc =
        programObject->getUniformLocation("light_model_two_sided");

    mProgramState.lightEnablesLoc    = programObject->getUniformLocation("light_enables");
    mProgramState.lightAmbientsLoc   = programObject->getUniformLocation("light_ambients");
    mProgramState.lightDiffusesLoc   = programObject->getUniformLocation("light_diffuses");
    mProgramState.lightSpecularsLoc  = programObject->getUniformLocation("light_speculars");
    mProgramState.lightPositionsLoc  = programObject->getUniformLocation("light_positions");
    mProgramState.lightDirectionsLoc = programObject->getUniformLocation("light_directions");
    mProgramState.lightSpotlightExponentsLoc =
        programObject->getUniformLocation("light_spotlight_exponents");
    mProgramState.lightSpotlightCutoffAnglesLoc =
        programObject->getUniformLocation("light_spotlight_cutoff_angles");
    mProgramState.lightAttenuationConstsLoc =
        programObject->getUniformLocation("light_attenuation_consts");
    mProgramState.lightAttenuationLinearsLoc =
        programObject->getUniformLocation("light_attenuation_linears");
    mProgramState.lightAttenuationQuadraticsLoc =
        programObject->getUniformLocation("light_attenuation_quadratics");

    mProgramState.fogEnableLoc  = programObject->getUniformLocation("enable_fog");
    mProgramState.fogModeLoc    = programObject->getUniformLocation("fog_mode");
    mProgramState.fogDensityLoc = programObject->getUniformLocation("fog_density");
    mProgramState.fogStartLoc   = programObject->getUniformLocation("fog_start");
    mProgramState.fogEndLoc     = programObject->getUniformLocation("fog_end");
    mProgramState.fogColorLoc   = programObject->getUniformLocation("fog_color");

    mProgramState.enableClipPlanesLoc = programObject->getUniformLocation("enable_clip_planes");
    mProgramState.clipPlaneEnablesLoc = programObject->getUniformLocation("clip_plane_enables");
    mProgramState.clipPlanesLoc       = programObject->getUniformLocation("clip_planes");

    mProgramState.pointRasterizationLoc = programObject->getUniformLocation("point_rasterization");
    mProgramState.pointSizeMinLoc       = programObject->getUniformLocation("point_size_min");
    mProgramState.pointSizeMaxLoc       = programObject->getUniformLocation("point_size_max");
    mProgramState.pointDistanceAttenuationLoc =
        programObject->getUniformLocation("point_distance_attenuation");
    mProgramState.pointSpriteEnabledLoc = programObject->getUniformLocation("point_sprite_enabled");

    mProgramState.enableDrawTextureLoc = programObject->getUniformLocation("enable_draw_texture");
    mProgramState.drawTextureCoordsLoc = programObject->getUniformLocation("draw_texture_coords");
    mProgramState.drawTextureDimsLoc   = programObject->getUniformLocation("draw_texture_dims");
    mProgramState.drawTextureNormalizedCropRectLoc =
        programObject->getUniformLocation("draw_texture_normalized_crop_rect");

    ANGLE_TRY(glState->setProgram(context, programObject));

    for (int i = 0; i < kTexUnitCount; i++)
    {
        setUniform1i(context, programObject, mProgramState.tex2DSamplerLocs[i], i);
        setUniform1i(context, programObject, mProgramState.texCubeSamplerLocs[i],
                     i + kTexUnitCount);
    }

    glState->setObjectDirty(GL_PROGRAM);

    mRendererProgramInitialized = true;
    return angle::Result::Continue;
}

void GLES1Renderer::setUniform1i(Context *context, Program *programObject, GLint loc, GLint value)
{
    if (loc == -1)
        return;
    programObject->setUniform1iv(context, loc, 1, &value);
}

void GLES1Renderer::setUniform1iv(Context *context,
                                  Program *programObject,
                                  GLint loc,
                                  GLint count,
                                  const GLint *value)
{
    if (loc == -1)
        return;
    programObject->setUniform1iv(context, loc, count, value);
}

void GLES1Renderer::setUniformMatrix4fv(Program *programObject,
                                        GLint loc,
                                        GLint count,
                                        GLboolean transpose,
                                        const GLfloat *value)
{
    if (loc == -1)
        return;
    programObject->setUniformMatrix4fv(loc, count, transpose, value);
}

void GLES1Renderer::setUniform4fv(Program *programObject,
                                  GLint loc,
                                  GLint count,
                                  const GLfloat *value)
{
    if (loc == -1)
        return;
    programObject->setUniform4fv(loc, count, value);
}

void GLES1Renderer::setUniform3fv(Program *programObject,
                                  GLint loc,
                                  GLint count,
                                  const GLfloat *value)
{
    if (loc == -1)
        return;
    programObject->setUniform3fv(loc, count, value);
}

void GLES1Renderer::setUniform2fv(Program *programObject,
                                  GLint loc,
                                  GLint count,
                                  const GLfloat *value)
{
    if (loc == -1)
        return;
    programObject->setUniform2fv(loc, count, value);
}

void GLES1Renderer::setUniform1f(Program *programObject, GLint loc, GLfloat value)
{
    if (loc == -1)
        return;
    programObject->setUniform1fv(loc, 1, &value);
}

void GLES1Renderer::setUniform1fv(Program *programObject,
                                  GLint loc,
                                  GLint count,
                                  const GLfloat *value)
{
    if (loc == -1)
        return;
    programObject->setUniform1fv(loc, count, value);
}

void GLES1Renderer::setAttributesEnabled(Context *context, State *glState, AttributesMask mask)
{
    GLES1State &gles1 = glState->gles1();

    ClientVertexArrayType nonTexcoordArrays[] = {
        ClientVertexArrayType::Vertex,
        ClientVertexArrayType::Normal,
        ClientVertexArrayType::Color,
        ClientVertexArrayType::PointSize,
    };

    for (const ClientVertexArrayType attrib : nonTexcoordArrays)
    {
        int index = VertexArrayIndex(attrib, glState->gles1());

        if (mask.test(index))
        {
            gles1.setClientStateEnabled(attrib, true);
            context->enableVertexAttribArray(index);
        }
        else
        {
            gles1.setClientStateEnabled(attrib, false);
            context->disableVertexAttribArray(index);
        }
    }

    for (unsigned int i = 0; i < kTexUnitCount; i++)
    {
        int index = TexCoordArrayIndex(i);

        if (mask.test(index))
        {
            gles1.setTexCoordArrayEnabled(i, true);
            context->enableVertexAttribArray(index);
        }
        else
        {
            gles1.setTexCoordArrayEnabled(i, false);
            context->disableVertexAttribArray(index);
        }
    }
}

}  // namespace gl
