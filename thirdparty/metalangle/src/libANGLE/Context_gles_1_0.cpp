//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Context_gles_1_0.cpp: Implements the GLES1-specific parts of Context.

#include "libANGLE/Context.h"

#include "common/mathutil.h"
#include "common/utilities.h"

#include "libANGLE/GLES1Renderer.h"
#include "libANGLE/queryconversions.h"
#include "libANGLE/queryutils.h"

namespace
{

angle::Mat4 FixedMatrixToMat4(const GLfixed *m)
{
    angle::Mat4 matrixAsFloat;
    GLfloat *floatData = matrixAsFloat.data();

    for (int i = 0; i < 16; i++)
    {
        floatData[i] = gl::ConvertFixedToFloat(m[i]);
    }

    return matrixAsFloat;
}

}  // namespace

namespace gl
{

void Context::alphaFunc(AlphaTestFunc func, GLfloat ref)
{
    mState.gles1().setAlphaFunc(func, ref);
}

void Context::alphaFuncx(AlphaTestFunc func, GLfixed ref)
{
    mState.gles1().setAlphaFunc(func, ConvertFixedToFloat(ref));
}

void Context::clearColorx(GLfixed red, GLfixed green, GLfixed blue, GLfixed alpha)
{
    UNIMPLEMENTED();
}

void Context::clearDepthx(GLfixed depth)
{
    UNIMPLEMENTED();
}

void Context::clientActiveTexture(GLenum texture)
{
    mState.gles1().setClientTextureUnit(texture - GL_TEXTURE0);
    mStateCache.onGLES1ClientStateChange(this);
}

void Context::clipPlanef(GLenum p, const GLfloat *eqn)
{
    mState.gles1().setClipPlane(p - GL_CLIP_PLANE0, eqn);
}

void Context::clipPlanex(GLenum plane, const GLfixed *equation)
{
    const GLfloat equationf[4] = {
        ConvertFixedToFloat(equation[0]),
        ConvertFixedToFloat(equation[1]),
        ConvertFixedToFloat(equation[2]),
        ConvertFixedToFloat(equation[3]),
    };

    mState.gles1().setClipPlane(plane - GL_CLIP_PLANE0, equationf);
}

void Context::color4f(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    mState.gles1().setCurrentColor({red, green, blue, alpha});
}

void Context::color4ub(GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha)
{
    mState.gles1().setCurrentColor(
        {normalizedToFloat<uint8_t>(red), normalizedToFloat<uint8_t>(green),
         normalizedToFloat<uint8_t>(blue), normalizedToFloat<uint8_t>(alpha)});
}

void Context::color4x(GLfixed red, GLfixed green, GLfixed blue, GLfixed alpha)
{
    mState.gles1().setCurrentColor({ConvertFixedToFloat(red), ConvertFixedToFloat(green),
                                    ConvertFixedToFloat(blue), ConvertFixedToFloat(alpha)});
}

void Context::colorPointer(GLint size, VertexAttribType type, GLsizei stride, const void *ptr)
{
    vertexAttribPointer(vertexArrayIndex(ClientVertexArrayType::Color), size, type, GL_FALSE,
                        stride, ptr);
}

void Context::depthRangex(GLfixed n, GLfixed f)
{
    UNIMPLEMENTED();
}

void Context::disableClientState(ClientVertexArrayType clientState)
{
    mState.gles1().setClientStateEnabled(clientState, false);
    disableVertexAttribArray(vertexArrayIndex(clientState));
    mStateCache.onGLES1ClientStateChange(this);
}

void Context::enableClientState(ClientVertexArrayType clientState)
{
    mState.gles1().setClientStateEnabled(clientState, true);
    enableVertexAttribArray(vertexArrayIndex(clientState));
    mStateCache.onGLES1ClientStateChange(this);
}

void Context::fogf(GLenum pname, GLfloat param)
{
    SetFogParameters(&mState.gles1(), pname, &param);
}

void Context::fogfv(GLenum pname, const GLfloat *params)
{
    SetFogParameters(&mState.gles1(), pname, params);
}

void Context::fogx(GLenum pname, GLfixed param)
{
    if (GetFogParameterCount(pname) == 1)
    {
        GLfloat paramf = pname == GL_FOG_MODE ? ConvertToGLenum(param) : ConvertFixedToFloat(param);
        fogf(pname, paramf);
    }
    else
    {
        UNREACHABLE();
    }
}

void Context::fogxv(GLenum pname, const GLfixed *params)
{
    int paramCount = GetFogParameterCount(pname);

    if (paramCount > 0)
    {
        GLfloat paramsf[4];
        for (int i = 0; i < paramCount; i++)
        {
            paramsf[i] =
                pname == GL_FOG_MODE ? ConvertToGLenum(params[i]) : ConvertFixedToFloat(params[i]);
        }
        fogfv(pname, paramsf);
    }
    else
    {
        UNREACHABLE();
    }
}

void Context::frustumf(GLfloat l, GLfloat r, GLfloat b, GLfloat t, GLfloat n, GLfloat f)
{
    mState.gles1().multMatrix(angle::Mat4::Frustum(l, r, b, t, n, f));
}

void Context::frustumx(GLfixed l, GLfixed r, GLfixed b, GLfixed t, GLfixed n, GLfixed f)
{
    mState.gles1().multMatrix(angle::Mat4::Frustum(ConvertFixedToFloat(l), ConvertFixedToFloat(r),
                                                   ConvertFixedToFloat(b), ConvertFixedToFloat(t),
                                                   ConvertFixedToFloat(n), ConvertFixedToFloat(f)));
}

void Context::getClipPlanef(GLenum plane, GLfloat *equation)
{
    mState.gles1().getClipPlane(plane - GL_CLIP_PLANE0, equation);
}

void Context::getClipPlanex(GLenum plane, GLfixed *equation)
{
    GLfloat equationf[4] = {};

    mState.gles1().getClipPlane(plane - GL_CLIP_PLANE0, equationf);

    for (int i = 0; i < 4; i++)
    {
        equation[i] = ConvertFloatToFixed(equationf[i]);
    }
}

void Context::getFixedv(GLenum pname, GLfixed *params)
{
    UNIMPLEMENTED();
}

void Context::getLightfv(GLenum light, LightParameter pname, GLfloat *params)
{
    GetLightParameters(&mState.gles1(), light, pname, params);
}

void Context::getLightxv(GLenum light, LightParameter pname, GLfixed *params)
{
    GLfloat paramsf[4];
    getLightfv(light, pname, paramsf);

    for (unsigned int i = 0; i < GetLightParameterCount(pname); i++)
    {
        params[i] = ConvertFloatToFixed(paramsf[i]);
    }
}

void Context::getMaterialfv(GLenum face, MaterialParameter pname, GLfloat *params)
{
    GetMaterialParameters(&mState.gles1(), face, pname, params);
}

void Context::getMaterialxv(GLenum face, MaterialParameter pname, GLfixed *params)
{
    GLfloat paramsf[4];
    getMaterialfv(face, pname, paramsf);

    for (unsigned int i = 0; i < GetMaterialParameterCount(pname); i++)
    {
        params[i] = ConvertFloatToFixed(paramsf[i]);
    }
}

void Context::getTexEnvfv(TextureEnvTarget target, TextureEnvParameter pname, GLfloat *params)
{
    GetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, params);
}

void Context::getTexEnviv(TextureEnvTarget target, TextureEnvParameter pname, GLint *params)
{
    GLfloat paramsf[4];
    GetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, paramsf);
    ConvertTextureEnvToInt(pname, paramsf, params);
}

void Context::getTexEnvxv(TextureEnvTarget target, TextureEnvParameter pname, GLfixed *params)
{
    GLfloat paramsf[4];
    GetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, paramsf);
    ConvertTextureEnvToFixed(pname, paramsf, params);
}

void Context::getTexParameterxv(TextureType target, GLenum pname, GLfixed *params)
{
    UNIMPLEMENTED();
}

void Context::lightModelf(GLenum pname, GLfloat param)
{
    SetLightModelParameters(&mState.gles1(), pname, &param);
}

void Context::lightModelfv(GLenum pname, const GLfloat *params)
{
    SetLightModelParameters(&mState.gles1(), pname, params);
}

void Context::lightModelx(GLenum pname, GLfixed param)
{
    lightModelf(pname, ConvertFixedToFloat(param));
}

void Context::lightModelxv(GLenum pname, const GLfixed *param)
{
    GLfloat paramsf[4];

    for (unsigned int i = 0; i < GetLightModelParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(param[i]);
    }

    lightModelfv(pname, paramsf);
}

void Context::lightf(GLenum light, LightParameter pname, GLfloat param)
{
    SetLightParameters(&mState.gles1(), light, pname, &param);
}

void Context::lightfv(GLenum light, LightParameter pname, const GLfloat *params)
{
    SetLightParameters(&mState.gles1(), light, pname, params);
}

void Context::lightx(GLenum light, LightParameter pname, GLfixed param)
{
    lightf(light, pname, ConvertFixedToFloat(param));
}

void Context::lightxv(GLenum light, LightParameter pname, const GLfixed *params)
{
    GLfloat paramsf[4];

    for (unsigned int i = 0; i < GetLightParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }

    lightfv(light, pname, paramsf);
}

void Context::lineWidthx(GLfixed width)
{
    UNIMPLEMENTED();
}

void Context::loadIdentity()
{
    mState.gles1().loadMatrix(angle::Mat4());
}

void Context::loadMatrixf(const GLfloat *m)
{
    mState.gles1().loadMatrix(angle::Mat4(m));
}

void Context::loadMatrixx(const GLfixed *m)
{
    mState.gles1().loadMatrix(FixedMatrixToMat4(m));
}

void Context::logicOp(LogicalOperation opcodePacked)
{
    mState.gles1().setLogicOp(opcodePacked);
}

void Context::materialf(GLenum face, MaterialParameter pname, GLfloat param)
{
    SetMaterialParameters(&mState.gles1(), face, pname, &param);
}

void Context::materialfv(GLenum face, MaterialParameter pname, const GLfloat *params)
{
    SetMaterialParameters(&mState.gles1(), face, pname, params);
}

void Context::materialx(GLenum face, MaterialParameter pname, GLfixed param)
{
    materialf(face, pname, ConvertFixedToFloat(param));
}

void Context::materialxv(GLenum face, MaterialParameter pname, const GLfixed *param)
{
    GLfloat paramsf[4];

    for (unsigned int i = 0; i < GetMaterialParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(param[i]);
    }

    materialfv(face, pname, paramsf);
}

void Context::matrixMode(MatrixType mode)
{
    mState.gles1().setMatrixMode(mode);
}

void Context::multMatrixf(const GLfloat *m)
{
    mState.gles1().multMatrix(angle::Mat4(m));
}

void Context::multMatrixx(const GLfixed *m)
{
    mState.gles1().multMatrix(FixedMatrixToMat4(m));
}

void Context::multiTexCoord4f(GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q)
{
    unsigned int unit = target - GL_TEXTURE0;
    ASSERT(target >= GL_TEXTURE0 && unit < getCaps().maxMultitextureUnits);
    mState.gles1().setCurrentTextureCoords(unit, {s, t, r, q});
}

void Context::multiTexCoord4x(GLenum target, GLfixed s, GLfixed t, GLfixed r, GLfixed q)
{
    unsigned int unit = target - GL_TEXTURE0;
    ASSERT(target >= GL_TEXTURE0 && unit < getCaps().maxMultitextureUnits);
    mState.gles1().setCurrentTextureCoords(unit, {ConvertFixedToFloat(s), ConvertFixedToFloat(t),
                                                  ConvertFixedToFloat(r), ConvertFixedToFloat(q)});
}

void Context::normal3f(GLfloat nx, GLfloat ny, GLfloat nz)
{
    mState.gles1().setCurrentNormal({nx, ny, nz});
}

void Context::normal3x(GLfixed nx, GLfixed ny, GLfixed nz)
{
    mState.gles1().setCurrentNormal(
        {ConvertFixedToFloat(nx), ConvertFixedToFloat(ny), ConvertFixedToFloat(nz)});
}

void Context::normalPointer(VertexAttribType type, GLsizei stride, const void *ptr)
{
    vertexAttribPointer(vertexArrayIndex(ClientVertexArrayType::Normal), 3, type, GL_FALSE, stride,
                        ptr);
}

void Context::orthof(GLfloat left,
                     GLfloat right,
                     GLfloat bottom,
                     GLfloat top,
                     GLfloat zNear,
                     GLfloat zFar)
{
    mState.gles1().multMatrix(angle::Mat4::Ortho(left, right, bottom, top, zNear, zFar));
}

void Context::orthox(GLfixed l, GLfixed r, GLfixed b, GLfixed t, GLfixed n, GLfixed f)
{
    mState.gles1().multMatrix(angle::Mat4::Ortho(ConvertFixedToFloat(l), ConvertFixedToFloat(r),
                                                 ConvertFixedToFloat(b), ConvertFixedToFloat(t),
                                                 ConvertFixedToFloat(n), ConvertFixedToFloat(f)));
}

void Context::pointParameterf(PointParameter pname, GLfloat param)
{
    SetPointParameter(&mState.gles1(), pname, &param);
}

void Context::pointParameterfv(PointParameter pname, const GLfloat *params)
{
    SetPointParameter(&mState.gles1(), pname, params);
}

void Context::pointParameterx(PointParameter pname, GLfixed param)
{
    GLfloat paramf = ConvertFixedToFloat(param);
    SetPointParameter(&mState.gles1(), pname, &paramf);
}

void Context::pointParameterxv(PointParameter pname, const GLfixed *params)
{
    GLfloat paramsf[4] = {};
    for (unsigned int i = 0; i < GetPointParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }
    SetPointParameter(&mState.gles1(), pname, paramsf);
}

void Context::pointSize(GLfloat size)
{
    SetPointSize(&mState.gles1(), size);
}

void Context::pointSizex(GLfixed size)
{
    SetPointSize(&mState.gles1(), ConvertFixedToFloat(size));
}

void Context::polygonOffsetx(GLfixed factor, GLfixed units)
{
    UNIMPLEMENTED();
}

void Context::popMatrix()
{
    mState.gles1().popMatrix();
}

void Context::pushMatrix()
{
    mState.gles1().pushMatrix();
}

void Context::rotatef(float angle, float x, float y, float z)
{
    mState.gles1().multMatrix(angle::Mat4::Rotate(angle, angle::Vector3(x, y, z)));
}

void Context::rotatex(GLfixed angle, GLfixed x, GLfixed y, GLfixed z)
{
    mState.gles1().multMatrix(angle::Mat4::Rotate(
        ConvertFixedToFloat(angle),
        angle::Vector3(ConvertFixedToFloat(x), ConvertFixedToFloat(y), ConvertFixedToFloat(z))));
}

void Context::sampleCoveragex(GLclampx value, GLboolean invert)
{
    UNIMPLEMENTED();
}

void Context::scalef(float x, float y, float z)
{
    mState.gles1().multMatrix(angle::Mat4::Scale(angle::Vector3(x, y, z)));
}

void Context::scalex(GLfixed x, GLfixed y, GLfixed z)
{
    mState.gles1().multMatrix(angle::Mat4::Scale(
        angle::Vector3(ConvertFixedToFloat(x), ConvertFixedToFloat(y), ConvertFixedToFloat(z))));
}

void Context::shadeModel(ShadingModel model)
{
    mState.gles1().setShadeModel(model);
}

void Context::texCoordPointer(GLint size, VertexAttribType type, GLsizei stride, const void *ptr)
{
    vertexAttribPointer(vertexArrayIndex(ClientVertexArrayType::TextureCoord), size, type, GL_FALSE,
                        stride, ptr);
}

void Context::texEnvf(TextureEnvTarget target, TextureEnvParameter pname, GLfloat param)
{
    SetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, &param);
}

void Context::texEnvfv(TextureEnvTarget target, TextureEnvParameter pname, const GLfloat *params)
{
    SetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, params);
}

void Context::texEnvi(TextureEnvTarget target, TextureEnvParameter pname, GLint param)
{
    GLfloat paramsf[4] = {};
    ConvertTextureEnvFromInt(pname, &param, paramsf);
    SetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, paramsf);
}

void Context::texEnviv(TextureEnvTarget target, TextureEnvParameter pname, const GLint *params)
{
    GLfloat paramsf[4] = {};
    ConvertTextureEnvFromInt(pname, params, paramsf);
    SetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, paramsf);
}

void Context::texEnvx(TextureEnvTarget target, TextureEnvParameter pname, GLfixed param)
{
    GLfloat paramsf[4] = {};
    ConvertTextureEnvFromFixed(pname, &param, paramsf);
    SetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, paramsf);
}

void Context::texEnvxv(TextureEnvTarget target, TextureEnvParameter pname, const GLfixed *params)
{
    GLfloat paramsf[4] = {};
    ConvertTextureEnvFromFixed(pname, params, paramsf);
    SetTextureEnv(mState.getActiveSampler(), &mState.gles1(), target, pname, paramsf);
}

void Context::texParameterx(TextureType target, GLenum pname, GLfixed param)
{
    UNIMPLEMENTED();
}

void Context::texParameterxv(TextureType target, GLenum pname, const GLfixed *params)
{
    UNIMPLEMENTED();
}

void Context::translatef(float x, float y, float z)
{
    mState.gles1().multMatrix(angle::Mat4::Translate(angle::Vector3(x, y, z)));
}

void Context::translatex(GLfixed x, GLfixed y, GLfixed z)
{
    mState.gles1().multMatrix(angle::Mat4::Translate(
        angle::Vector3(ConvertFixedToFloat(x), ConvertFixedToFloat(y), ConvertFixedToFloat(z))));
}

void Context::vertexPointer(GLint size, VertexAttribType type, GLsizei stride, const void *ptr)
{
    vertexAttribPointer(vertexArrayIndex(ClientVertexArrayType::Vertex), size, type, GL_FALSE,
                        stride, ptr);
}

// GL_OES_draw_texture
void Context::drawTexf(float x, float y, float z, float width, float height)
{
    mGLES1Renderer->drawTexture(this, &mState, x, y, z, width, height);
}

void Context::drawTexfv(const GLfloat *coords)
{
    mGLES1Renderer->drawTexture(this, &mState, coords[0], coords[1], coords[2], coords[3],
                                coords[4]);
}

void Context::drawTexi(GLint x, GLint y, GLint z, GLint width, GLint height)
{
    mGLES1Renderer->drawTexture(this, &mState, static_cast<GLfloat>(x), static_cast<GLfloat>(y),
                                static_cast<GLfloat>(z), static_cast<GLfloat>(width),
                                static_cast<GLfloat>(height));
}

void Context::drawTexiv(const GLint *coords)
{
    mGLES1Renderer->drawTexture(this, &mState, static_cast<GLfloat>(coords[0]),
                                static_cast<GLfloat>(coords[1]), static_cast<GLfloat>(coords[2]),
                                static_cast<GLfloat>(coords[3]), static_cast<GLfloat>(coords[4]));
}

void Context::drawTexs(GLshort x, GLshort y, GLshort z, GLshort width, GLshort height)
{
    mGLES1Renderer->drawTexture(this, &mState, static_cast<GLfloat>(x), static_cast<GLfloat>(y),
                                static_cast<GLfloat>(z), static_cast<GLfloat>(width),
                                static_cast<GLfloat>(height));
}

void Context::drawTexsv(const GLshort *coords)
{
    mGLES1Renderer->drawTexture(this, &mState, static_cast<GLfloat>(coords[0]),
                                static_cast<GLfloat>(coords[1]), static_cast<GLfloat>(coords[2]),
                                static_cast<GLfloat>(coords[3]), static_cast<GLfloat>(coords[4]));
}

void Context::drawTexx(GLfixed x, GLfixed y, GLfixed z, GLfixed width, GLfixed height)
{
    mGLES1Renderer->drawTexture(this, &mState, ConvertFixedToFloat(x), ConvertFixedToFloat(y),
                                ConvertFixedToFloat(z), ConvertFixedToFloat(width),
                                ConvertFixedToFloat(height));
}

void Context::drawTexxv(const GLfixed *coords)
{
    mGLES1Renderer->drawTexture(this, &mState, ConvertFixedToFloat(coords[0]),
                                ConvertFixedToFloat(coords[1]), ConvertFixedToFloat(coords[2]),
                                ConvertFixedToFloat(coords[3]), ConvertFixedToFloat(coords[4]));
}

// GL_OES_matrix_palette
void Context::currentPaletteMatrix(GLuint matrixpaletteindex)
{
    UNIMPLEMENTED();
}

void Context::loadPaletteFromModelViewMatrix()
{
    UNIMPLEMENTED();
}

void Context::matrixIndexPointer(GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    UNIMPLEMENTED();
}

void Context::weightPointer(GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    UNIMPLEMENTED();
}

// GL_OES_point_size_array
void Context::pointSizePointer(VertexAttribType type, GLsizei stride, const void *ptr)
{
    vertexAttribPointer(vertexArrayIndex(ClientVertexArrayType::PointSize), 1, type, GL_FALSE,
                        stride, ptr);
}

// GL_OES_query_matrix
GLbitfield Context::queryMatrixx(GLfixed *mantissa, GLint *exponent)
{
    UNIMPLEMENTED();
    return 0;
}

// GL_OES_texture_cube_map
void Context::getTexGenfv(GLenum coord, GLenum pname, GLfloat *params)
{
    UNIMPLEMENTED();
}

void Context::getTexGeniv(GLenum coord, GLenum pname, GLint *params)
{
    UNIMPLEMENTED();
}

void Context::getTexGenxv(GLenum coord, GLenum pname, GLfixed *params)
{
    UNIMPLEMENTED();
}

void Context::texGenf(GLenum coord, GLenum pname, GLfloat param)
{
    UNIMPLEMENTED();
}

void Context::texGenfv(GLenum coord, GLenum pname, const GLfloat *params)
{
    UNIMPLEMENTED();
}

void Context::texGeni(GLenum coord, GLenum pname, GLint param)
{
    UNIMPLEMENTED();
}

void Context::texGeniv(GLenum coord, GLenum pname, const GLint *params)
{
    UNIMPLEMENTED();
}

void Context::texGenx(GLenum coord, GLenum pname, GLfixed param)
{
    UNIMPLEMENTED();
}

void Context::texGenxv(GLenum coord, GLenum pname, const GLint *params)
{
    UNIMPLEMENTED();
}

int Context::vertexArrayIndex(ClientVertexArrayType type) const
{
    return GLES1Renderer::VertexArrayIndex(type, mState.gles1());
}

// static
int Context::TexCoordArrayIndex(unsigned int unit)
{
    return GLES1Renderer::TexCoordArrayIndex(unit);
}
}  // namespace gl
