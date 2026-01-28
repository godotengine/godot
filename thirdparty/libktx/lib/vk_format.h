/*
================================================================================================

Description	:	Vulkan format properties and conversion from OpenGL.
Author		:	J.M.P. van Waveren
Date		:	07/17/2016
Language	:	C99
Format		:	Real tabs with the tab size equal to 4 spaces.
Copyright	:	Copyright (c) 2016 Oculus VR, LLC. All Rights reserved.


LICENSE
=======

Copyright 2016 Oculus VR, LLC.
SPDX-License-Identifier: Apache-2.0


DESCRIPTION
===========

This header implements several support routines to convert OpenGL formats/types
to Vulkan formats. These routines are particularly useful for loading file
formats that store OpenGL formats/types such as KTX and glTF.

The functions in this header file convert the format, internalFormat and type
that are used as parameters to the following OpenGL functions:

void glTexImage2D( GLenum target, GLint level, GLint internalFormat,
	GLsizei width, GLsizei height, GLint border,
	GLenum format, GLenum type, const GLvoid * data );
void glTexImage3D( GLenum target, GLint level, GLint internalFormat,
	GLsizei width, GLsizei height, GLsizei depth, GLint border,
	GLenum format, GLenum type, const GLvoid * data );
void glCompressedTexImage2D( GLenum target, GLint level, GLenum internalformat,
	GLsizei width, GLsizei height, GLint border,
	GLsizei imageSize, const GLvoid * data );
void glCompressedTexImage3D( GLenum target, GLint level, GLenum internalformat,
	GLsizei width, GLsizei height, GLsizei depth, GLint border,
	GLsizei imageSize, const GLvoid * data );
void glTexStorage2D( GLenum target, GLsizei levels, GLenum internalformat,
	GLsizei width, GLsizei height );
void glTexStorage3D( GLenum target, GLsizei levels, GLenum internalformat,
	GLsizei width, GLsizei height, GLsizei depth );
void glVertexAttribPointer( GLuint index, GLint size, GLenum type, GLboolean normalized,
	GLsizei stride, const GLvoid * pointer);


IMPLEMENTATION
==============

This file does not include OpenGL / OpenGL ES headers because:

  1. Including OpenGL / OpenGL ES headers is platform dependent and
     may require a separate installation of an OpenGL SDK.
  2. The OpenGL format/type constants are the same between extensions and core.
  3. The OpenGL format/type constants are the same between OpenGL and OpenGL ES.
  4. File formats like KTX and glTF may use OpenGL formats and types that
     are not supported by the OpenGL implementation on the platform but are
     supported by the Vulkan implementation.


ENTRY POINTS
============

static inline VkFormat vkGetFormatFromOpenGLFormat( const GLenum format, const GLenum type );
static inline VkFormat vkGetFormatFromOpenGLType( const GLenum type, const GLuint numComponents, const GLboolean normalized );
static inline VkFormat vkGetFormatFromOpenGLInternalFormat( const GLenum internalFormat );
static inline void vkGetFormatSize( const VkFormat format, VkFormatSize * pFormatSize );

MODIFICATIONS for use in libktx
===============================

2019.5.30 Use common ktxFormatSize to return results. Mark Callow, github.com/MarkCallow.
2019.6.12 Add mapping of PVRTC formats.                             "

================================================================================================
*/

#if !defined( VK_FORMAT_H )
#define VK_FORMAT_H

#include "gl_format.h"

static inline VkFormat vkGetFormatFromOpenGLFormat( const GLenum format, const GLenum type )
{
	switch ( type )
	{
		//
		// 8 bits per component
		//
		case GL_UNSIGNED_BYTE:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R8_UNORM;
				case GL_RG:						return VK_FORMAT_R8G8_UNORM;
				case GL_RGB:					return VK_FORMAT_R8G8B8_UNORM;
				case GL_BGR:					return VK_FORMAT_B8G8R8_UNORM;
				case GL_RGBA:					return VK_FORMAT_R8G8B8A8_UNORM;
				case GL_BGRA:					return VK_FORMAT_B8G8R8A8_UNORM;
				case GL_RED_INTEGER:			return VK_FORMAT_R8_UINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R8G8_UINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R8G8B8_UINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_B8G8R8_UINT;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R8G8B8A8_UINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_B8G8R8A8_UINT;
				case GL_STENCIL_INDEX:			return VK_FORMAT_S8_UINT;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}
		case GL_BYTE:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R8_SNORM;
				case GL_RG:						return VK_FORMAT_R8G8_SNORM;
				case GL_RGB:					return VK_FORMAT_R8G8B8_SNORM;
				case GL_BGR:					return VK_FORMAT_B8G8R8_SNORM;
				case GL_RGBA:					return VK_FORMAT_R8G8B8A8_SNORM;
				case GL_BGRA:					return VK_FORMAT_B8G8R8A8_SNORM;
				case GL_RED_INTEGER:			return VK_FORMAT_R8_SINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R8G8_SINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R8G8B8_SINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_B8G8R8_SINT;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R8G8B8A8_SINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_B8G8R8A8_SINT;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}

		//
		// 16 bits per component
		//
		case GL_UNSIGNED_SHORT:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R16_UNORM;
				case GL_RG:						return VK_FORMAT_R16G16_UNORM;
				case GL_RGB:					return VK_FORMAT_R16G16B16_UNORM;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R16G16B16A16_UNORM;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_R16_UINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R16G16_UINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R16G16B16_UINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R16G16B16A16_UINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_D16_UNORM;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_D16_UNORM_S8_UINT;
			}
			break;
		}
		case GL_SHORT:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R16_SNORM;
				case GL_RG:						return VK_FORMAT_R16G16_SNORM;
				case GL_RGB:					return VK_FORMAT_R16G16B16_SNORM;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R16G16B16A16_SNORM;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_R16_SINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R16G16_SINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R16G16B16_SINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R16G16B16A16_SINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}
		case GL_HALF_FLOAT:
		case GL_HALF_FLOAT_OES:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R16_SFLOAT;
				case GL_RG:						return VK_FORMAT_R16G16_SFLOAT;
				case GL_RGB:					return VK_FORMAT_R16G16B16_SFLOAT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R16G16B16A16_SFLOAT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RG_INTEGER:				return VK_FORMAT_UNDEFINED;
				case GL_RGB_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}

		//
		// 32 bits per component
		//
		case GL_UNSIGNED_INT:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R32_UINT;
				case GL_RG:						return VK_FORMAT_R32G32_UINT;
				case GL_RGB:					return VK_FORMAT_R32G32B32_UINT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R32G32B32A32_UINT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_R32_UINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R32G32_UINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R32G32B32_UINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R32G32B32A32_UINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_X8_D24_UNORM_PACK32;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_D24_UNORM_S8_UINT;
			}
			break;
		}
		case GL_INT:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R32_SINT;
				case GL_RG:						return VK_FORMAT_R32G32_SINT;
				case GL_RGB:					return VK_FORMAT_R32G32B32_SINT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R32G32B32A32_SINT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_R32_SINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R32G32_SINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R32G32B32_SINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R32G32B32A32_SINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}
		case GL_FLOAT:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R32_SFLOAT;
				case GL_RG:						return VK_FORMAT_R32G32_SFLOAT;
				case GL_RGB:					return VK_FORMAT_R32G32B32_SFLOAT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R32G32B32A32_SFLOAT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RG_INTEGER:				return VK_FORMAT_UNDEFINED;
				case GL_RGB_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_D32_SFLOAT;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_D32_SFLOAT_S8_UINT;
			}
			break;
		}

		//
		// 64 bits per component
		//
		case GL_UNSIGNED_INT64:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R64_UINT;
				case GL_RG:						return VK_FORMAT_R64G64_UINT;
				case GL_RGB:					return VK_FORMAT_R64G64B64_UINT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R64G64B64A64_UINT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RG_INTEGER:				return VK_FORMAT_UNDEFINED;
				case GL_RGB_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}
		case GL_INT64:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R64_SINT;
				case GL_RG:						return VK_FORMAT_R64G64_SINT;
				case GL_RGB:					return VK_FORMAT_R64G64B64_SINT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R64G64B64A64_SINT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_R64_SINT;
				case GL_RG_INTEGER:				return VK_FORMAT_R64G64_SINT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R64G64B64_SINT;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R64G64B64A64_SINT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}
		case GL_DOUBLE:
		{
			switch ( format )
			{
				case GL_RED:					return VK_FORMAT_R64_SFLOAT;
				case GL_RG:						return VK_FORMAT_R64G64_SFLOAT;
				case GL_RGB:					return VK_FORMAT_R64G64B64_SFLOAT;
				case GL_BGR:					return VK_FORMAT_UNDEFINED;
				case GL_RGBA:					return VK_FORMAT_R64G64B64A64_SFLOAT;
				case GL_BGRA:					return VK_FORMAT_UNDEFINED;
				case GL_RED_INTEGER:			return VK_FORMAT_R64_SFLOAT;
				case GL_RG_INTEGER:				return VK_FORMAT_R64G64_SFLOAT;
				case GL_RGB_INTEGER:			return VK_FORMAT_R64G64B64_SFLOAT;
				case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_RGBA_INTEGER:			return VK_FORMAT_R64G64B64A64_SFLOAT;
				case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
				case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
				case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
			}
			break;
		}

		//
		// Packed
		//
		case GL_UNSIGNED_BYTE_3_3_2:
			assert( format == GL_RGB || format == GL_RGB_INTEGER );
			return VK_FORMAT_UNDEFINED;
		case GL_UNSIGNED_BYTE_2_3_3_REV:
			assert( format == GL_BGR || format == GL_BGR_INTEGER );
			return VK_FORMAT_UNDEFINED;
		case GL_UNSIGNED_SHORT_5_6_5:
			assert( format == GL_RGB || format == GL_RGB_INTEGER );
			return VK_FORMAT_R5G6B5_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_5_6_5_REV:
			assert( format == GL_BGR || format == GL_BGR_INTEGER );
			return VK_FORMAT_B5G6R5_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_4_4_4_4:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_4_4_4_4_REV:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_5_5_5_1:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_1_5_5_5_REV:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
		case GL_UNSIGNED_INT_8_8_8_8:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return ( format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER ) ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_UNORM;
		case GL_UNSIGNED_INT_8_8_8_8_REV:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return ( format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER ) ? VK_FORMAT_A8B8G8R8_UINT_PACK32 : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
		case GL_UNSIGNED_INT_10_10_10_2:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return ( format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER ) ? VK_FORMAT_A2R10G10B10_UINT_PACK32 : VK_FORMAT_A2R10G10B10_UNORM_PACK32;
		case GL_UNSIGNED_INT_2_10_10_10_REV:
			assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
			return ( format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER ) ? VK_FORMAT_A2B10G10R10_UINT_PACK32 : VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		case GL_UNSIGNED_INT_10F_11F_11F_REV:
			assert( format == GL_RGB || format == GL_BGR );
			return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case GL_UNSIGNED_INT_5_9_9_9_REV:
			assert( format == GL_RGB || format == GL_BGR );
			return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
		case GL_UNSIGNED_INT_24_8:
			assert( format == GL_DEPTH_STENCIL );
			return VK_FORMAT_D24_UNORM_S8_UINT;
		case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
			assert( format == GL_DEPTH_STENCIL );
			return VK_FORMAT_D32_SFLOAT_S8_UINT;
	}

	return VK_FORMAT_UNDEFINED;
}

#if defined(NEED_VK_GET_FORMAT_FROM_OPENGL_TYPE)
static inline VkFormat vkGetFormatFromOpenGLType( const GLenum type, const GLuint numComponents, const GLboolean normalized )
{
	switch ( type )
	{
		//
		// 8 bits per component
		//
		case GL_UNSIGNED_BYTE:
		{
			switch ( numComponents )
			{
				case 1:							return normalized ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8_UINT;
				case 2:							return normalized ? VK_FORMAT_R8G8_UNORM : VK_FORMAT_R8G8_UINT;
				case 3:							return normalized ? VK_FORMAT_R8G8B8_UNORM : VK_FORMAT_R8G8B8_UINT;
				case 4:							return normalized ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_UINT;
			}
			break;
		}
		case GL_BYTE:
		{
			switch ( numComponents )
			{
				case 1:							return normalized ? VK_FORMAT_R8_SNORM : VK_FORMAT_R8_SINT;
				case 2:							return normalized ? VK_FORMAT_R8G8_SNORM : VK_FORMAT_R8G8_SINT;
				case 3:							return normalized ? VK_FORMAT_R8G8B8_SNORM : VK_FORMAT_R8G8B8_SINT;
				case 4:							return normalized ? VK_FORMAT_R8G8B8A8_SNORM : VK_FORMAT_R8G8B8A8_SINT;
			}
			break;
		}

		//
		// 16 bits per component
		//
		case GL_UNSIGNED_SHORT:
		{
			switch ( numComponents )
			{
				case 1:							return normalized ? VK_FORMAT_R16_UNORM : VK_FORMAT_R16_UINT;
				case 2:							return normalized ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R16G16_UINT;
				case 3:							return normalized ? VK_FORMAT_R16G16B16_UNORM : VK_FORMAT_R16G16B16_UINT;
				case 4:							return normalized ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R16G16B16A16_UINT;
			}
			break;
		}
		case GL_SHORT:
		{
			switch ( numComponents )
			{
				case 1:							return normalized ? VK_FORMAT_R16_SNORM : VK_FORMAT_R16_SINT;
				case 2:							return normalized ? VK_FORMAT_R16G16_SNORM : VK_FORMAT_R16G16_SINT;
				case 3:							return normalized ? VK_FORMAT_R16G16B16_SNORM : VK_FORMAT_R16G16B16_SINT;
				case 4:							return normalized ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R16G16B16A16_SINT;
			}
			break;
		}
		case GL_HALF_FLOAT:
		case GL_HALF_FLOAT_OES:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R16_SFLOAT;
				case 2:							return VK_FORMAT_R16G16_SFLOAT;
				case 3:							return VK_FORMAT_R16G16B16_SFLOAT;
				case 4:							return VK_FORMAT_R16G16B16A16_SFLOAT;
			}
			break;
		}

		//
		// 32 bits per component
		//
		case GL_UNSIGNED_INT:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R32_UINT;
				case 2:							return VK_FORMAT_R32G32_UINT;
				case 3:							return VK_FORMAT_R32G32B32_UINT;
				case 4:							return VK_FORMAT_R32G32B32A32_UINT;
			}
			break;
		}
		case GL_INT:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R32_SINT;
				case 2:							return VK_FORMAT_R32G32_SINT;
				case 3:							return VK_FORMAT_R32G32B32_SINT;
				case 4:							return VK_FORMAT_R32G32B32A32_SINT;
			}
			break;
		}
		case GL_FLOAT:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R32_SFLOAT;
				case 2:							return VK_FORMAT_R32G32_SFLOAT;
				case 3:							return VK_FORMAT_R32G32B32_SFLOAT;
				case 4:							return VK_FORMAT_R32G32B32A32_SFLOAT;
			}
			break;
		}

		//
		// 64 bits per component
		//
		case GL_UNSIGNED_INT64:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R64_UINT;
				case 2:							return VK_FORMAT_R64G64_UINT;
				case 3:							return VK_FORMAT_R64G64B64_UINT;
				case 4:							return VK_FORMAT_R64G64B64A64_UINT;
			}
			break;
		}
		case GL_INT64:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R64_SINT;
				case 2:							return VK_FORMAT_R64G64_SINT;
				case 3:							return VK_FORMAT_R64G64B64_SINT;
				case 4:							return VK_FORMAT_R64G64B64A64_SINT;
			}
			break;
		}
		case GL_DOUBLE:
		{
			switch ( numComponents )
			{
				case 1:							return VK_FORMAT_R64_SFLOAT;
				case 2:							return VK_FORMAT_R64G64_SFLOAT;
				case 3:							return VK_FORMAT_R64G64B64_SFLOAT;
				case 4:							return VK_FORMAT_R64G64B64A64_SFLOAT;
			}
			break;
		}

		//
		// Packed
		//
		case GL_UNSIGNED_BYTE_3_3_2:			return VK_FORMAT_UNDEFINED;
		case GL_UNSIGNED_BYTE_2_3_3_REV:		return VK_FORMAT_UNDEFINED;
		case GL_UNSIGNED_SHORT_5_6_5:			return VK_FORMAT_R5G6B5_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_5_6_5_REV:		return VK_FORMAT_B5G6R5_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_4_4_4_4:			return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_4_4_4_4_REV:		return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_5_5_5_1:			return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
		case GL_UNSIGNED_SHORT_1_5_5_5_REV:		return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
		case GL_UNSIGNED_INT_8_8_8_8:			return normalized ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_UINT;
		case GL_UNSIGNED_INT_8_8_8_8_REV:		return normalized ? VK_FORMAT_A8B8G8R8_UNORM_PACK32 : VK_FORMAT_A8B8G8R8_UINT_PACK32;
		case GL_UNSIGNED_INT_10_10_10_2:		return normalized ? VK_FORMAT_A2R10G10B10_UNORM_PACK32 : VK_FORMAT_A2R10G10B10_UINT_PACK32;
		case GL_UNSIGNED_INT_2_10_10_10_REV:	return normalized ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_A2B10G10R10_UINT_PACK32;
		case GL_UNSIGNED_INT_10F_11F_11F_REV:	return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case GL_UNSIGNED_INT_5_9_9_9_REV:		return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
		case GL_UNSIGNED_INT_24_8:				return VK_FORMAT_D24_UNORM_S8_UINT;
		case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:	return VK_FORMAT_D32_SFLOAT_S8_UINT;
	}

	return VK_FORMAT_UNDEFINED;
}
#endif

static inline VkFormat vkGetFormatFromOpenGLInternalFormat( const GLenum internalFormat )
{
	switch ( internalFormat )
	{
		//
		// 8 bits per component
		//
		case GL_R8:												return VK_FORMAT_R8_UNORM;					// 1-component, 8-bit unsigned normalized
		case GL_RG8:											return VK_FORMAT_R8G8_UNORM;				// 2-component, 8-bit unsigned normalized
		case GL_RGB8:											return VK_FORMAT_R8G8B8_UNORM;				// 3-component, 8-bit unsigned normalized
		case GL_RGBA8:											return VK_FORMAT_R8G8B8A8_UNORM;			// 4-component, 8-bit unsigned normalized

		case GL_R8_SNORM:										return VK_FORMAT_R8_SNORM;					// 1-component, 8-bit signed normalized
		case GL_RG8_SNORM:										return VK_FORMAT_R8G8_SNORM;				// 2-component, 8-bit signed normalized
		case GL_RGB8_SNORM:										return VK_FORMAT_R8G8B8_SNORM;				// 3-component, 8-bit signed normalized
		case GL_RGBA8_SNORM:									return VK_FORMAT_R8G8B8A8_SNORM;			// 4-component, 8-bit signed normalized

		case GL_R8UI:											return VK_FORMAT_R8_UINT;					// 1-component, 8-bit unsigned integer
		case GL_RG8UI:											return VK_FORMAT_R8G8_UINT;					// 2-component, 8-bit unsigned integer
		case GL_RGB8UI:											return VK_FORMAT_R8G8B8_UINT;				// 3-component, 8-bit unsigned integer
		case GL_RGBA8UI:										return VK_FORMAT_R8G8B8A8_UINT;				// 4-component, 8-bit unsigned integer

		case GL_R8I:											return VK_FORMAT_R8_SINT;					// 1-component, 8-bit signed integer
		case GL_RG8I:											return VK_FORMAT_R8G8_SINT;					// 2-component, 8-bit signed integer
		case GL_RGB8I:											return VK_FORMAT_R8G8B8_SINT;				// 3-component, 8-bit signed integer
		case GL_RGBA8I:											return VK_FORMAT_R8G8B8A8_SINT;				// 4-component, 8-bit signed integer

		case GL_SR8:											return VK_FORMAT_R8_SRGB;					// 1-component, 8-bit sRGB
		case GL_SRG8:											return VK_FORMAT_R8G8_SRGB;					// 2-component, 8-bit sRGB
		case GL_SRGB8:											return VK_FORMAT_R8G8B8_SRGB;				// 3-component, 8-bit sRGB
		case GL_SRGB8_ALPHA8:									return VK_FORMAT_R8G8B8A8_SRGB;				// 4-component, 8-bit sRGB

		//
		// 16 bits per component
		//
		case GL_R16:											return VK_FORMAT_R16_UNORM;					// 1-component, 16-bit unsigned normalized
		case GL_RG16:											return VK_FORMAT_R16G16_UNORM;				// 2-component, 16-bit unsigned normalized
		case GL_RGB16:											return VK_FORMAT_R16G16B16_UNORM;			// 3-component, 16-bit unsigned normalized
		case GL_RGBA16:											return VK_FORMAT_R16G16B16A16_UNORM;		// 4-component, 16-bit unsigned normalized

		case GL_R16_SNORM:										return VK_FORMAT_R16_SNORM;					// 1-component, 16-bit signed normalized
		case GL_RG16_SNORM:										return VK_FORMAT_R16G16_SNORM;				// 2-component, 16-bit signed normalized
		case GL_RGB16_SNORM:									return VK_FORMAT_R16G16B16_SNORM;			// 3-component, 16-bit signed normalized
		case GL_RGBA16_SNORM:									return VK_FORMAT_R16G16B16A16_SNORM;		// 4-component, 16-bit signed normalized

		case GL_R16UI:											return VK_FORMAT_R16_UINT;					// 1-component, 16-bit unsigned integer
		case GL_RG16UI:											return VK_FORMAT_R16G16_UINT;				// 2-component, 16-bit unsigned integer
		case GL_RGB16UI:										return VK_FORMAT_R16G16B16_UINT;			// 3-component, 16-bit unsigned integer
		case GL_RGBA16UI:										return VK_FORMAT_R16G16B16A16_UINT;			// 4-component, 16-bit unsigned integer

		case GL_R16I:											return VK_FORMAT_R16_SINT;					// 1-component, 16-bit signed integer
		case GL_RG16I:											return VK_FORMAT_R16G16_SINT;				// 2-component, 16-bit signed integer
		case GL_RGB16I:											return VK_FORMAT_R16G16B16_SINT;			// 3-component, 16-bit signed integer
		case GL_RGBA16I:										return VK_FORMAT_R16G16B16A16_SINT;			// 4-component, 16-bit signed integer

		case GL_R16F:											return VK_FORMAT_R16_SFLOAT;				// 1-component, 16-bit floating-point
		case GL_RG16F:											return VK_FORMAT_R16G16_SFLOAT;				// 2-component, 16-bit floating-point
		case GL_RGB16F:											return VK_FORMAT_R16G16B16_SFLOAT;			// 3-component, 16-bit floating-point
		case GL_RGBA16F:										return VK_FORMAT_R16G16B16A16_SFLOAT;		// 4-component, 16-bit floating-point

		//
		// 32 bits per component
		//
		case GL_R32UI:											return VK_FORMAT_R32_UINT;					// 1-component, 32-bit unsigned integer
		case GL_RG32UI:											return VK_FORMAT_R32G32_UINT;				// 2-component, 32-bit unsigned integer
		case GL_RGB32UI:										return VK_FORMAT_R32G32B32_UINT;			// 3-component, 32-bit unsigned integer
		case GL_RGBA32UI:										return VK_FORMAT_R32G32B32A32_UINT;			// 4-component, 32-bit unsigned integer

		case GL_R32I:											return VK_FORMAT_R32_SINT;					// 1-component, 32-bit signed integer
		case GL_RG32I:											return VK_FORMAT_R32G32_SINT;				// 2-component, 32-bit signed integer
		case GL_RGB32I:											return VK_FORMAT_R32G32B32_SINT;			// 3-component, 32-bit signed integer
		case GL_RGBA32I:										return VK_FORMAT_R32G32B32A32_SINT;			// 4-component, 32-bit signed integer

		case GL_R32F:											return VK_FORMAT_R32_SFLOAT;				// 1-component, 32-bit floating-point
		case GL_RG32F:											return VK_FORMAT_R32G32_SFLOAT;				// 2-component, 32-bit floating-point
		case GL_RGB32F:											return VK_FORMAT_R32G32B32_SFLOAT;			// 3-component, 32-bit floating-point
		case GL_RGBA32F:										return VK_FORMAT_R32G32B32A32_SFLOAT;		// 4-component, 32-bit floating-point

		//
		// Packed
		//
		case GL_R3_G3_B2:										return VK_FORMAT_UNDEFINED;					// 3-component 3:3:2,       unsigned normalized
		case GL_RGB4:											return VK_FORMAT_UNDEFINED;					// 3-component 4:4:4,       unsigned normalized
		case GL_RGB5:											return VK_FORMAT_R5G5B5A1_UNORM_PACK16;		// 3-component 5:5:5,       unsigned normalized
		case GL_RGB565:											return VK_FORMAT_R5G6B5_UNORM_PACK16;		// 3-component 5:6:5,       unsigned normalized
		case GL_RGB10:											return VK_FORMAT_A2R10G10B10_UNORM_PACK32;	// 3-component 10:10:10,    unsigned normalized
		case GL_RGB12:											return VK_FORMAT_UNDEFINED;					// 3-component 12:12:12,    unsigned normalized
		case GL_RGBA2:											return VK_FORMAT_UNDEFINED;					// 4-component 2:2:2:2,     unsigned normalized
		case GL_RGBA4:											return VK_FORMAT_R4G4B4A4_UNORM_PACK16;		// 4-component 4:4:4:4,     unsigned normalized
		case GL_RGBA12:											return VK_FORMAT_UNDEFINED;					// 4-component 12:12:12:12, unsigned normalized
		case GL_RGB5_A1:										return VK_FORMAT_A1R5G5B5_UNORM_PACK16;		// 4-component 5:5:5:1,     unsigned normalized
		case GL_RGB10_A2:										return VK_FORMAT_A2R10G10B10_UNORM_PACK32;	// 4-component 10:10:10:2,  unsigned normalized
		case GL_RGB10_A2UI:										return VK_FORMAT_A2R10G10B10_UINT_PACK32;	// 4-component 10:10:10:2,  unsigned integer
		case GL_R11F_G11F_B10F:									return VK_FORMAT_B10G11R11_UFLOAT_PACK32;	// 3-component 11:11:10,    floating-point
		case GL_RGB9_E5:										return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;	// 3-component/exp 9:9:9/5, floating-point

		//
		// S3TC/DXT/BC
		//

		case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:					return VK_FORMAT_BC1_RGB_UNORM_BLOCK;		// line through 3D space, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:					return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;		// line through 3D space plus 1-bit alpha, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:					return VK_FORMAT_BC2_UNORM_BLOCK;			// line through 3D space plus line through 1D space, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:					return VK_FORMAT_BC3_UNORM_BLOCK;			// line through 3D space plus 4-bit alpha, 4x4 blocks, unsigned normalized

		case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:					return VK_FORMAT_BC1_RGB_SRGB_BLOCK;		// line through 3D space, 4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:			return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;		// line through 3D space plus 1-bit alpha, 4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:			return VK_FORMAT_BC2_SRGB_BLOCK;			// line through 3D space plus line through 1D space, 4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:			return VK_FORMAT_BC3_SRGB_BLOCK;			// line through 3D space plus 4-bit alpha, 4x4 blocks, sRGB

		case GL_COMPRESSED_LUMINANCE_LATC1_EXT:					return VK_FORMAT_BC4_UNORM_BLOCK;			// line through 1D space, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT:			return VK_FORMAT_BC5_UNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT:			return VK_FORMAT_BC4_SNORM_BLOCK;			// line through 1D space, 4x4 blocks, signed normalized
		case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT:	return VK_FORMAT_BC5_SNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, signed normalized

		case GL_COMPRESSED_RED_RGTC1:							return VK_FORMAT_BC4_UNORM_BLOCK;			// line through 1D space, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RG_RGTC2:							return VK_FORMAT_BC5_UNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_SIGNED_RED_RGTC1:					return VK_FORMAT_BC4_SNORM_BLOCK;			// line through 1D space, 4x4 blocks, signed normalized
		case GL_COMPRESSED_SIGNED_RG_RGTC2:						return VK_FORMAT_BC5_SNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, signed normalized

		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:				return VK_FORMAT_BC6H_UFLOAT_BLOCK;			// 3-component, 4x4 blocks, unsigned floating-point
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:				return VK_FORMAT_BC6H_SFLOAT_BLOCK;			// 3-component, 4x4 blocks, signed floating-point
		case GL_COMPRESSED_RGBA_BPTC_UNORM:						return VK_FORMAT_BC7_UNORM_BLOCK;			// 4-component, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:				return VK_FORMAT_BC7_SRGB_BLOCK;			// 4-component, 4x4 blocks, sRGB

		//
		// ETC
		//
		case GL_ETC1_RGB8_OES:									return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;	// 3-component ETC1, 4x4 blocks, unsigned normalized

		case GL_COMPRESSED_RGB8_ETC2:							return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;	// 3-component ETC2, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:		return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;	// 4-component ETC2 with 1-bit alpha, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA8_ETC2_EAC:						return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;	// 4-component ETC2, 4x4 blocks, unsigned normalized

		case GL_COMPRESSED_SRGB8_ETC2:							return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;	// 3-component ETC2, 4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:		return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;	// 4-component ETC2 with 1-bit alpha, 4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:				return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;	// 4-component ETC2, 4x4 blocks, sRGB

		case GL_COMPRESSED_R11_EAC:								return VK_FORMAT_EAC_R11_UNORM_BLOCK;		// 1-component ETC, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RG11_EAC:							return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;	// 2-component ETC, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_SIGNED_R11_EAC:						return VK_FORMAT_EAC_R11_SNORM_BLOCK;		// 1-component ETC, 4x4 blocks, signed normalized
		case GL_COMPRESSED_SIGNED_RG11_EAC:						return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;	// 2-component ETC, 4x4 blocks, signed normalized

		//
		// PVRTC
		//
		case GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG:				return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;	// 3-component PVRTC, 16x8 blocks, unsigned normalized
		case GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG:				return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;	// 3-component PVRTC,  8x8 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG:				return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC, 16x8 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG:				return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC,  8x8 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_PVRTC_2BPPV2_IMG:				return VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC,  8x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_PVRTC_4BPPV2_IMG:				return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC,  4x4 blocks, unsigned normalized

		case GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT:				return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;	// 3-component PVRTC, 16x8 blocks, sRGB
		case GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT:				return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;	// 3-component PVRTC,  8x8 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT:			return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC, 16x8 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT:			return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC,  8x8 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV2_IMG:			return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC,  8x4 blocks, sRGB
		case GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV2_IMG:			return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC,  4x4 blocks, sRGB

		//
		// ASTC
		//
		case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:					return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;		// 4-component ASTC, 4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:					return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;		// 4-component ASTC, 5x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:					return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;		// 4-component ASTC, 5x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:					return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;		// 4-component ASTC, 6x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:					return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;		// 4-component ASTC, 6x6 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:					return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;		// 4-component ASTC, 8x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:					return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;		// 4-component ASTC, 8x6 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:					return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;		// 4-component ASTC, 8x8 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:					return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;		// 4-component ASTC, 10x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:					return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;		// 4-component ASTC, 10x6 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:					return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;		// 4-component ASTC, 10x8 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:					return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;	// 4-component ASTC, 10x10 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:					return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;	// 4-component ASTC, 12x10 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:					return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;	// 4-component ASTC, 12x12 blocks, unsigned normalized

		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:			return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;		// 4-component ASTC, 4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:			return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;		// 4-component ASTC, 5x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:			return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;		// 4-component ASTC, 5x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:			return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;		// 4-component ASTC, 6x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:			return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;		// 4-component ASTC, 6x6 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:			return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;		// 4-component ASTC, 8x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:			return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;		// 4-component ASTC, 8x6 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:			return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;		// 4-component ASTC, 8x8 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:			return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;		// 4-component ASTC, 10x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:			return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;		// 4-component ASTC, 10x6 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:			return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;		// 4-component ASTC, 10x8 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:			return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;		// 4-component ASTC, 10x10 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:			return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;		// 4-component ASTC, 12x10 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:			return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;		// 4-component ASTC, 12x12 blocks, sRGB

		case GL_COMPRESSED_RGBA_ASTC_3x3x3_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 3x3x3 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_4x3x3_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x3x3 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_4x4x3_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x3 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_4x4x4_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_5x4x4_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x4x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_5x5x4_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x4 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_5x5x5_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_6x5x5_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x5x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_6x6x5_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x5 blocks, unsigned normalized
		case GL_COMPRESSED_RGBA_ASTC_6x6x6_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x6 blocks, unsigned normalized

		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 3x3x3 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x3x3 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x3 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x4x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x4 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x5x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x5 blocks, sRGB
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x6 blocks, sRGB

		//
		// ATC
		//
		case GL_ATC_RGB_AMD:									return VK_FORMAT_UNDEFINED;					// 3-component, 4x4 blocks, unsigned normalized
		case GL_ATC_RGBA_EXPLICIT_ALPHA_AMD:					return VK_FORMAT_UNDEFINED;					// 4-component, 4x4 blocks, unsigned normalized
		case GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD:				return VK_FORMAT_UNDEFINED;					// 4-component, 4x4 blocks, unsigned normalized

		//
		// Palletized
		//
		case GL_PALETTE4_RGB8_OES:								return VK_FORMAT_UNDEFINED;					// 3-component 8:8:8,   4-bit palette, unsigned normalized
		case GL_PALETTE4_RGBA8_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 8:8:8:8, 4-bit palette, unsigned normalized
		case GL_PALETTE4_R5_G6_B5_OES:							return VK_FORMAT_UNDEFINED;					// 3-component 5:6:5,   4-bit palette, unsigned normalized
		case GL_PALETTE4_RGBA4_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 4:4:4:4, 4-bit palette, unsigned normalized
		case GL_PALETTE4_RGB5_A1_OES:							return VK_FORMAT_UNDEFINED;					// 4-component 5:5:5:1, 4-bit palette, unsigned normalized
		case GL_PALETTE8_RGB8_OES:								return VK_FORMAT_UNDEFINED;					// 3-component 8:8:8,   8-bit palette, unsigned normalized
		case GL_PALETTE8_RGBA8_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 8:8:8:8, 8-bit palette, unsigned normalized
		case GL_PALETTE8_R5_G6_B5_OES:							return VK_FORMAT_UNDEFINED;					// 3-component 5:6:5,   8-bit palette, unsigned normalized
		case GL_PALETTE8_RGBA4_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 4:4:4:4, 8-bit palette, unsigned normalized
		case GL_PALETTE8_RGB5_A1_OES:							return VK_FORMAT_UNDEFINED;					// 4-component 5:5:5:1, 8-bit palette, unsigned normalized

		//
		// Depth/stencil
		//
		case GL_DEPTH_COMPONENT16:								return VK_FORMAT_D16_UNORM;
		case GL_DEPTH_COMPONENT24:								return VK_FORMAT_X8_D24_UNORM_PACK32;
		case GL_DEPTH_COMPONENT32:								return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT32F:								return VK_FORMAT_D32_SFLOAT;
		case GL_DEPTH_COMPONENT32F_NV:							return VK_FORMAT_D32_SFLOAT;
		case GL_STENCIL_INDEX1:									return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX4:									return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX8:									return VK_FORMAT_S8_UINT;
		case GL_STENCIL_INDEX16:								return VK_FORMAT_UNDEFINED;
		case GL_DEPTH24_STENCIL8:								return VK_FORMAT_D24_UNORM_S8_UINT;
		case GL_DEPTH32F_STENCIL8:								return VK_FORMAT_D32_SFLOAT_S8_UINT;
		case GL_DEPTH32F_STENCIL8_NV:							return VK_FORMAT_D32_SFLOAT_S8_UINT;

		default:												return VK_FORMAT_UNDEFINED;
	}
}

#if defined(NEED_VK_GET_FORMAT_SIZE)
static inline void vkGetFormatSize( const VkFormat format, ktxFormatSize * pFormatSize )
{
	pFormatSize->minBlocksX = pFormatSize->minBlocksY = 1;
	switch ( format )
	{
		case VK_FORMAT_R4G4_UNORM_PACK8:
			pFormatSize->flags = KTX_FORMAT_SIZE_PACKED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 1 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
		case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
		case VK_FORMAT_R5G6B5_UNORM_PACK16:
		case VK_FORMAT_B5G6R5_UNORM_PACK16:
		case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
		case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
		case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
			pFormatSize->flags = KTX_FORMAT_SIZE_PACKED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 2 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R8_UNORM:
		case VK_FORMAT_R8_SNORM:
		case VK_FORMAT_R8_USCALED:
		case VK_FORMAT_R8_SSCALED:
		case VK_FORMAT_R8_UINT:
		case VK_FORMAT_R8_SINT:
		case VK_FORMAT_R8_SRGB:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 1 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R8G8_UNORM:
		case VK_FORMAT_R8G8_SNORM:
		case VK_FORMAT_R8G8_USCALED:
		case VK_FORMAT_R8G8_SSCALED:
		case VK_FORMAT_R8G8_UINT:
		case VK_FORMAT_R8G8_SINT:
		case VK_FORMAT_R8G8_SRGB:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 2 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R8G8B8_UNORM:
		case VK_FORMAT_R8G8B8_SNORM:
		case VK_FORMAT_R8G8B8_USCALED:
		case VK_FORMAT_R8G8B8_SSCALED:
		case VK_FORMAT_R8G8B8_UINT:
		case VK_FORMAT_R8G8B8_SINT:
		case VK_FORMAT_R8G8B8_SRGB:
		case VK_FORMAT_B8G8R8_UNORM:
		case VK_FORMAT_B8G8R8_SNORM:
		case VK_FORMAT_B8G8R8_USCALED:
		case VK_FORMAT_B8G8R8_SSCALED:
		case VK_FORMAT_B8G8R8_UINT:
		case VK_FORMAT_B8G8R8_SINT:
		case VK_FORMAT_B8G8R8_SRGB:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 3 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R8G8B8A8_UNORM:
		case VK_FORMAT_R8G8B8A8_SNORM:
		case VK_FORMAT_R8G8B8A8_USCALED:
		case VK_FORMAT_R8G8B8A8_SSCALED:
		case VK_FORMAT_R8G8B8A8_UINT:
		case VK_FORMAT_R8G8B8A8_SINT:
		case VK_FORMAT_R8G8B8A8_SRGB:
		case VK_FORMAT_B8G8R8A8_UNORM:
		case VK_FORMAT_B8G8R8A8_SNORM:
		case VK_FORMAT_B8G8R8A8_USCALED:
		case VK_FORMAT_B8G8R8A8_SSCALED:
		case VK_FORMAT_B8G8R8A8_UINT:
		case VK_FORMAT_B8G8R8A8_SINT:
		case VK_FORMAT_B8G8R8A8_SRGB:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
		case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
		case VK_FORMAT_A8B8G8R8_USCALED_PACK32:
		case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:
		case VK_FORMAT_A8B8G8R8_UINT_PACK32:
		case VK_FORMAT_A8B8G8R8_SINT_PACK32:
		case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
			pFormatSize->flags = KTX_FORMAT_SIZE_PACKED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
		case VK_FORMAT_A2R10G10B10_SNORM_PACK32:
		case VK_FORMAT_A2R10G10B10_USCALED_PACK32:
		case VK_FORMAT_A2R10G10B10_SSCALED_PACK32:
		case VK_FORMAT_A2R10G10B10_UINT_PACK32:
		case VK_FORMAT_A2R10G10B10_SINT_PACK32:
		case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
		case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
		case VK_FORMAT_A2B10G10R10_USCALED_PACK32:
		case VK_FORMAT_A2B10G10R10_SSCALED_PACK32:
		case VK_FORMAT_A2B10G10R10_UINT_PACK32:
		case VK_FORMAT_A2B10G10R10_SINT_PACK32:
			pFormatSize->flags = KTX_FORMAT_SIZE_PACKED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R16_UNORM:
		case VK_FORMAT_R16_SNORM:
		case VK_FORMAT_R16_USCALED:
		case VK_FORMAT_R16_SSCALED:
		case VK_FORMAT_R16_UINT:
		case VK_FORMAT_R16_SINT:
		case VK_FORMAT_R16_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 2 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R16G16_UNORM:
		case VK_FORMAT_R16G16_SNORM:
		case VK_FORMAT_R16G16_USCALED:
		case VK_FORMAT_R16G16_SSCALED:
		case VK_FORMAT_R16G16_UINT:
		case VK_FORMAT_R16G16_SINT:
		case VK_FORMAT_R16G16_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R16G16B16_UNORM:
		case VK_FORMAT_R16G16B16_SNORM:
		case VK_FORMAT_R16G16B16_USCALED:
		case VK_FORMAT_R16G16B16_SSCALED:
		case VK_FORMAT_R16G16B16_UINT:
		case VK_FORMAT_R16G16B16_SINT:
		case VK_FORMAT_R16G16B16_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 6 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R16G16B16A16_UNORM:
		case VK_FORMAT_R16G16B16A16_SNORM:
		case VK_FORMAT_R16G16B16A16_USCALED:
		case VK_FORMAT_R16G16B16A16_SSCALED:
		case VK_FORMAT_R16G16B16A16_UINT:
		case VK_FORMAT_R16G16B16A16_SINT:
		case VK_FORMAT_R16G16B16A16_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R32_UINT:
		case VK_FORMAT_R32_SINT:
		case VK_FORMAT_R32_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R32G32_UINT:
		case VK_FORMAT_R32G32_SINT:
		case VK_FORMAT_R32G32_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R32G32B32_UINT:
		case VK_FORMAT_R32G32B32_SINT:
		case VK_FORMAT_R32G32B32_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 12 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R32G32B32A32_UINT:
		case VK_FORMAT_R32G32B32A32_SINT:
		case VK_FORMAT_R32G32B32A32_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R64_UINT:
		case VK_FORMAT_R64_SINT:
		case VK_FORMAT_R64_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R64G64_UINT:
		case VK_FORMAT_R64G64_SINT:
		case VK_FORMAT_R64G64_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R64G64B64_UINT:
		case VK_FORMAT_R64G64B64_SINT:
		case VK_FORMAT_R64G64B64_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 24 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_R64G64B64A64_UINT:
		case VK_FORMAT_R64G64B64A64_SINT:
		case VK_FORMAT_R64G64B64A64_SFLOAT:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 32 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
		case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
			pFormatSize->flags = KTX_FORMAT_SIZE_PACKED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_D16_UNORM:
			pFormatSize->flags = KTX_FORMAT_SIZE_DEPTH_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 2 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_X8_D24_UNORM_PACK32:
			pFormatSize->flags = KTX_FORMAT_SIZE_PACKED_BIT | KTX_FORMAT_SIZE_DEPTH_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_D32_SFLOAT:
			pFormatSize->flags = KTX_FORMAT_SIZE_DEPTH_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_S8_UINT:
			pFormatSize->flags = KTX_FORMAT_SIZE_STENCIL_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 1 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_D16_UNORM_S8_UINT:
			pFormatSize->flags = KTX_FORMAT_SIZE_DEPTH_BIT | KTX_FORMAT_SIZE_STENCIL_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 3 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_D24_UNORM_S8_UINT:
			pFormatSize->flags = KTX_FORMAT_SIZE_DEPTH_BIT | KTX_FORMAT_SIZE_STENCIL_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 4 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			pFormatSize->flags = KTX_FORMAT_SIZE_DEPTH_BIT | KTX_FORMAT_SIZE_STENCIL_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
		case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
		case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
		case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
		case VK_FORMAT_BC4_UNORM_BLOCK:
		case VK_FORMAT_BC4_SNORM_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_BC2_UNORM_BLOCK:
		case VK_FORMAT_BC2_SRGB_BLOCK:
		case VK_FORMAT_BC3_UNORM_BLOCK:
		case VK_FORMAT_BC3_SRGB_BLOCK:
		case VK_FORMAT_BC5_UNORM_BLOCK:
		case VK_FORMAT_BC5_SNORM_BLOCK:
		case VK_FORMAT_BC6H_UFLOAT_BLOCK:
		case VK_FORMAT_BC6H_SFLOAT_BLOCK:
		case VK_FORMAT_BC7_UNORM_BLOCK:
		case VK_FORMAT_BC7_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
		case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
		case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
		case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
		case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
		case VK_FORMAT_EAC_R11_UNORM_BLOCK:
		case VK_FORMAT_EAC_R11_SNORM_BLOCK:
		case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
		case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG:
		case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 8;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			pFormatSize->minBlocksX = 2;
			pFormatSize->minBlocksY = 2;
			break;
		case VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG:
		case VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 8;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG:
		case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			pFormatSize->minBlocksX = 2;
			pFormatSize->minBlocksY = 2;
			break;
		case VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG:
		case VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 8 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
		case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 4;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
		case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 5;
			pFormatSize->blockHeight = 4;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
		case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 5;
			pFormatSize->blockHeight = 5;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
		case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 6;
			pFormatSize->blockHeight = 5;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
		case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 6;
			pFormatSize->blockHeight = 6;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
		case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 8;
			pFormatSize->blockHeight = 5;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
		case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 8;
			pFormatSize->blockHeight = 6;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
		case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 8;
			pFormatSize->blockHeight = 8;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
		case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 10;
			pFormatSize->blockHeight = 5;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
		case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 10;
			pFormatSize->blockHeight = 6;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
		case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 10;
			pFormatSize->blockHeight = 8;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
		case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 10;
			pFormatSize->blockHeight = 10;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
		case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 12;
			pFormatSize->blockHeight = 10;
			pFormatSize->blockDepth = 1;
			break;
		case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
		case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
			pFormatSize->flags = KTX_FORMAT_SIZE_COMPRESSED_BIT;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 16 * 8;
			pFormatSize->blockWidth = 12;
			pFormatSize->blockHeight = 12;
			pFormatSize->blockDepth = 1;
			break;
		default:
			pFormatSize->flags = 0;
			pFormatSize->paletteSizeInBits = 0;
			pFormatSize->blockSizeInBits = 0 * 8;
			pFormatSize->blockWidth = 1;
			pFormatSize->blockHeight = 1;
			pFormatSize->blockDepth = 1;
			break;
	}
}
#endif

#endif // !VK_FORMAT_H
