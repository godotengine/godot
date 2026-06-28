// Adapted from mono-public-types.h and image-types.h.

// https://github.com/dotnet/runtime/blob/27a7fe5c4bbe0762c231b2a46162e60ee04f3cde/src/native/public/mono/utils/details/mono-publib-types.h
// https://github.com/dotnet/runtime/blob/27a7fe5c4bbe0762c231b2a46162e60ee04f3cde/src/native/public/mono/metadata/details/image-types.h

#ifndef _MONO_TYPES_H_
#define _MONO_TYPES_H_

#include <stdint.h>
#include <stdlib.h>

typedef int32_t mono_bool;

typedef void MonoAssembly;
typedef void MonoAssemblyName;
typedef void MonoImage;

typedef enum {
	MONO_IMAGE_OK,
	MONO_IMAGE_ERROR_ERRNO,
	MONO_IMAGE_MISSING_ASSEMBLYREF,
	MONO_IMAGE_IMAGE_INVALID,
	MONO_IMAGE_NOT_SUPPORTED, ///< \since net7
} MonoImageOpenStatus;

#endif // _MONO_TYPES_H_
