//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef SAMPLE_UTIL_SHADER_UTILS_H
#define SAMPLE_UTIL_SHADER_UTILS_H

#include <functional>
#include <string>
#include <vector>

#include "util/util_export.h"
#include "util/util_gl.h"

ANGLE_UTIL_EXPORT GLuint CheckLinkStatusAndReturnProgram(GLuint program, bool outputErrorMessages);
ANGLE_UTIL_EXPORT GLuint CompileShader(GLenum type, const char *source);
ANGLE_UTIL_EXPORT GLuint CompileShaderFromFile(GLenum type, const std::string &sourcePath);

ANGLE_UTIL_EXPORT GLuint
CompileProgramWithTransformFeedback(const char *vsSource,
                                    const char *fsSource,
                                    const std::vector<std::string> &transformFeedbackVaryings,
                                    GLenum bufferMode);

ANGLE_UTIL_EXPORT GLuint CompileProgram(const char *vsSource, const char *fsSource);

ANGLE_UTIL_EXPORT GLuint CompileProgram(const char *vsSource,
                                        const char *fsSource,
                                        const std::function<void(GLuint)> &preLinkCallback);

ANGLE_UTIL_EXPORT GLuint CompileProgramWithGS(const char *vsSource,
                                              const char *gsSource,
                                              const char *fsSource);
ANGLE_UTIL_EXPORT GLuint CompileProgramFromFiles(const std::string &vsPath,
                                                 const std::string &fsPath);
ANGLE_UTIL_EXPORT GLuint CompileComputeProgram(const char *csSource,
                                               bool outputErrorMessages = true);
ANGLE_UTIL_EXPORT bool LinkAttachedProgram(GLuint program);

ANGLE_UTIL_EXPORT GLuint LoadBinaryProgramOES(const std::vector<uint8_t> &binary,
                                              GLenum binaryFormat);
ANGLE_UTIL_EXPORT GLuint LoadBinaryProgramES3(const std::vector<uint8_t> &binary,
                                              GLenum binaryFormat);

namespace angle
{

namespace essl1_shaders
{

ANGLE_UTIL_EXPORT const char *PositionAttrib();
ANGLE_UTIL_EXPORT const char *ColorUniform();
ANGLE_UTIL_EXPORT const char *Texture2DUniform();

namespace vs
{

// A shader that sets gl_Position to zero.
ANGLE_UTIL_EXPORT const char *Zero();

// A shader that sets gl_Position to attribute a_position.
ANGLE_UTIL_EXPORT const char *Simple();

// A shader that passes through attribute a_position, setting it to gl_Position and varying
// v_position.
ANGLE_UTIL_EXPORT const char *Passthrough();

// A shader that simply passes through attribute a_position, setting it to gl_Position and varying
// texcoord.
ANGLE_UTIL_EXPORT const char *Texture2D();

}  // namespace vs

namespace fs
{

// A shader that renders a simple checker pattern of red and green. X axis and y axis separate the
// different colors. Needs varying v_position.
ANGLE_UTIL_EXPORT const char *Checkered();

// A shader that fills with color taken from uniform named "color".
ANGLE_UTIL_EXPORT const char *UniformColor();

// A shader that fills with 100% opaque red.
ANGLE_UTIL_EXPORT const char *Red();

// A shader that fills with 100% opaque green.
ANGLE_UTIL_EXPORT const char *Green();

// A shader that fills with 100% opaque blue.
ANGLE_UTIL_EXPORT const char *Blue();

// A shader that samples the texture
ANGLE_UTIL_EXPORT const char *Texture2D();

}  // namespace fs
}  // namespace essl1_shaders

namespace essl3_shaders
{

ANGLE_UTIL_EXPORT const char *PositionAttrib();

namespace vs
{

// A shader that sets gl_Position to zero.
ANGLE_UTIL_EXPORT const char *Zero();

// A shader that sets gl_Position to attribute a_position.
ANGLE_UTIL_EXPORT const char *Simple();

// A shader that simply passes through attribute a_position, setting it to gl_Position and varying
// v_position.
ANGLE_UTIL_EXPORT const char *Passthrough();

}  // namespace vs

namespace fs
{

// A shader that fills with 100% opaque red.
ANGLE_UTIL_EXPORT const char *Red();

// A shader that fills with 100% opaque green.
ANGLE_UTIL_EXPORT const char *Green();

// A shader that fills with 100% opaque blue.
ANGLE_UTIL_EXPORT const char *Blue();

}  // namespace fs
}  // namespace essl3_shaders

namespace essl31_shaders
{

ANGLE_UTIL_EXPORT const char *PositionAttrib();

namespace vs
{

// A shader that sets gl_Position to zero.
ANGLE_UTIL_EXPORT const char *Zero();

// A shader that sets gl_Position to attribute a_position.
ANGLE_UTIL_EXPORT const char *Simple();

// A shader that simply passes through attribute a_position, setting it to gl_Position and varying
// v_position.
ANGLE_UTIL_EXPORT const char *Passthrough();

}  // namespace vs

namespace fs
{

// A shader that fills with 100% opaque red.
ANGLE_UTIL_EXPORT const char *Red();

}  // namespace fs
}  // namespace essl31_shaders
}  // namespace angle

#endif  // SAMPLE_UTIL_SHADER_UTILS_H
