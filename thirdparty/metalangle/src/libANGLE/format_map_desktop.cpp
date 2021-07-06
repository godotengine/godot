// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// format_map_desktop:
//   Determining the sized internal format from a (format,type) pair.
//   Also check DesktopGL format combinations for validity.

#include "angle_gl.h"
#include "common/debug.h"
#include "formatutils.h"
#include "renderer/gl/functionsgl_enums.h"

// TODO(http://anglebug.com/3730): switch ANGLE to generate its own GL enum types from gl.xml

namespace gl
{

bool ValidDesktopFormat(GLenum format)
{
    switch (format)
    {
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_RED:
        case GL_GREEN:
        case GL_BLUE:
        case GL_RG:
        case GL_RGB:
        case GL_RGBA:
        case GL_BGR:
        case GL_BGRA:
        case GL_RED_INTEGER:
        case GL_GREEN_INTEGER:
        case GL_BLUE_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGR_INTEGER:
        case GL_BGRA_INTEGER:
            return true;

        default:
            return false;
    }
}

bool ValidDesktopType(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_HALF_FLOAT:
        case GL_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return true;

        default:
            return false;
    }
}

// From OpenGL 4.6 spec section 8.4
bool ValidDesktopFormatCombination(GLenum format, GLenum type, GLenum internalFormat)
{
    ASSERT(ValidDesktopFormat(format) && ValidDesktopType(type));
    const InternalFormat &internalFormatInfo = GetInternalFormatInfo(internalFormat, type);
    const InternalFormat &formatInfo         = GetInternalFormatInfo(format, type);

    switch (format)
    {
        case GL_RED_INTEGER:
        case GL_GREEN_INTEGER:
        case GL_BLUE_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGR_INTEGER:
        case GL_BGRA_INTEGER:
            switch (type)
            {
                case GL_HALF_FLOAT:
                case GL_FLOAT:
                case GL_UNSIGNED_INT_10F_11F_11F_REV:
                case GL_UNSIGNED_INT_5_9_9_9_REV:
                    return false;
                default:
                    break;
            }
            if (!internalFormatInfo.isInt())
                return false;
            break;
        default:
            // format is not an integer
            if (internalFormatInfo.isInt())
                return false;

            if (formatInfo.isDepthOrStencil() != internalFormatInfo.isDepthOrStencil())
                return false;

            if (format == GL_STENCIL_INDEX && internalFormat != GL_STENCIL_INDEX)
                return false;
            break;
    }

    return true;
}

}  // namespace gl
