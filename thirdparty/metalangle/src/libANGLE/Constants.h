//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Contants.h: Defines some implementation specific and gl constants

#ifndef LIBANGLE_CONSTANTS_H_
#define LIBANGLE_CONSTANTS_H_

#include "common/platform.h"

#include <stdint.h>

namespace gl
{

// The binary cache is currently left disable by default, and the application can enable it.
const size_t kDefaultMaxProgramCacheMemoryBytes = 0;

enum
{
    // Implementation upper limits, real maximums depend on the hardware
    MAX_SAMPLE_MASK_WORDS = 2,

    MAX_VERTEX_ATTRIBS         = 16,
    MAX_VERTEX_ATTRIB_BINDINGS = 16,

    // Implementation upper limits, real maximums depend on the hardware
    IMPLEMENTATION_MAX_VARYING_VECTORS = 32,
    IMPLEMENTATION_MAX_DRAW_BUFFERS    = 8,
    IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS =
        IMPLEMENTATION_MAX_DRAW_BUFFERS + 2,  // 2 extra for depth and/or stencil buffers

    IMPLEMENTATION_MAX_VERTEX_SHADER_UNIFORM_BUFFERS   = 16,
    IMPLEMENTATION_MAX_GEOMETRY_SHADER_UNIFORM_BUFFERS = 16,
    IMPLEMENTATION_MAX_FRAGMENT_SHADER_UNIFORM_BUFFERS = 16,
    IMPLEMENTATION_MAX_COMPUTE_SHADER_UNIFORM_BUFFERS  = 16,
    // GL_EXT_geometry_shader increases the minimum value of GL_MAX_COMBINED_UNIFORM_BLOCKS to 36.
    IMPLEMENTATION_MAX_COMBINED_SHADER_UNIFORM_BUFFERS = 36,

    // GL_EXT_geometry_shader increases the minimum value of GL_MAX_UNIFORM_BUFFER_BINDINGS to 48.
    IMPLEMENTATION_MAX_UNIFORM_BUFFER_BINDINGS = 48,

    // Transform feedback limits set to the minimum required by the spec.
    IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS = 64,
    IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS       = 4,
    IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS    = 4,
    IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_BUFFERS                = 4,

    // Maximum number of views which are supported by the implementation of ANGLE_multiview.
    IMPLEMENTATION_ANGLE_MULTIVIEW_MAX_VIEWS = 4,

    // These are the maximums the implementation can support
    // The actual GL caps are limited by the device caps
    // and should be queried from the Context
    IMPLEMENTATION_MAX_2D_TEXTURE_SIZE         = 16384,
    IMPLEMENTATION_MAX_CUBE_MAP_TEXTURE_SIZE   = 16384,
    IMPLEMENTATION_MAX_3D_TEXTURE_SIZE         = 2048,
    IMPLEMENTATION_MAX_2D_ARRAY_TEXTURE_LAYERS = 2048,

    // 1+log2 of max of MAX_*_TEXTURE_SIZE
    IMPLEMENTATION_MAX_TEXTURE_LEVELS = 15,

    // Limit active textures so we can use fast bitsets.
    IMPLEMENTATION_MAX_SHADER_TEXTURES = 32,
    IMPLEMENTATION_MAX_ACTIVE_TEXTURES = IMPLEMENTATION_MAX_SHADER_TEXTURES * 2,
    IMPLEMENTATION_MAX_IMAGE_UNITS     = IMPLEMENTATION_MAX_ACTIVE_TEXTURES,

    // Maximum number of slots allocated for atomic counter buffers.
    IMPLEMENTATION_MAX_ATOMIC_COUNTER_BUFFERS = 8,

    // Implementation upper limits, real maximums depend on the hardware.
    IMPLEMENTATION_MAX_SHADER_STORAGE_BUFFER_BINDINGS = 64,

    // Implementation upper limits of max number of clip distances
    IMPLEMENTATION_MAX_CLIP_DISTANCES = 32,
};

namespace limits
{
// Some of the minimums required by GL, used to detect if the backend meets the minimum requirement.
// Currently, there's no need to separate these values per spec version.
constexpr uint32_t kMinimumComputeStorageBuffers = 4;
}  // namespace limits

}  // namespace gl

#endif  // LIBANGLE_CONSTANTS_H_
