//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ExtensionBehavior.h: Extension name enumeration and data structures for storing extension
// behavior.

#ifndef COMPILER_TRANSLATOR_EXTENSIONBEHAVIOR_H_
#define COMPILER_TRANSLATOR_EXTENSIONBEHAVIOR_H_

#include <map>

namespace sh
{

enum class TExtension
{
    UNDEFINED,  // Special value used to indicate no extension.

    ARB_texture_rectangle,
    ANGLE_texture_multisample,
    ARM_shader_framebuffer_fetch,
    EXT_blend_func_extended,
    EXT_draw_buffers,
    EXT_frag_depth,
    EXT_geometry_shader,
    EXT_shader_framebuffer_fetch,
    EXT_shader_texture_lod,
    EXT_YUV_target,
    NV_EGL_stream_consumer_external,
    NV_shader_framebuffer_fetch,
    OES_EGL_image_external,
    OES_EGL_image_external_essl3,
    OES_standard_derivatives,
    OES_texture_storage_multisample_2d_array,
    OES_texture_3D,
    OVR_multiview,
    OVR_multiview2,
    ANGLE_multi_draw,
    ANGLE_base_vertex_base_instance,
    APPLE_clip_distance,
};

enum TBehavior
{
    EBhRequire,
    EBhEnable,
    EBhWarn,
    EBhDisable,
    EBhUndefined
};

const char *GetExtensionNameString(TExtension extension);
TExtension GetExtensionByName(const char *extension);

const char *GetBehaviorString(TBehavior b);

// Mapping between extension id and behavior.
typedef std::map<TExtension, TBehavior> TExtensionBehavior;

bool IsExtensionEnabled(const TExtensionBehavior &extBehavior, TExtension extension);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_EXTENSIONBEHAVIOR_H_
