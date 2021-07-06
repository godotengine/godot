//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FrontendFeatures.h: Features/workarounds for driver bugs and other behaviors seen
// on all platforms.

#ifndef ANGLE_PLATFORM_FRONTENDFEATURES_H_
#define ANGLE_PLATFORM_FRONTENDFEATURES_H_

#include "platform/Feature.h"

namespace angle
{

struct FrontendFeatures : angle::FeatureSetBase
{
    FrontendFeatures();
    ~FrontendFeatures();

    // Force the context to be lost (via KHR_robustness) if a GL_OUT_OF_MEMORY error occurs. The
    // driver may be in an inconsistent state if this happens, and some users of ANGLE rely on this
    // notification to prevent further execution.
    angle::Feature loseContextOnOutOfMemory = {
        "lose_context_on_out_of_memory", angle::FeatureCategory::FrontendWorkarounds,
        "Some users rely on a lost context notification if a GL_OUT_OF_MEMORY "
        "error occurs",
        &members};

    // Program binaries don't contain transform feedback varyings on Qualcomm GPUs.
    // Work around this by disabling the program cache for programs with transform feedback.
    angle::Feature disableProgramCachingForTransformFeedback = {
        "disable_program_caching_for_transform_feedback",
        angle::FeatureCategory::FrontendWorkarounds,
        "On some GPUs, program binaries don't contain transform feedback varyings", &members};

    // On Windows Intel OpenGL drivers TexImage sometimes seems to interact with the Framebuffer.
    // Flaky crashes can occur unless we sync the Framebuffer bindings. The workaround is to add
    // Framebuffer binding dirty bits to TexImage updates. See http://anglebug.com/2906
    angle::Feature syncFramebufferBindingsOnTexImage = {
        "sync_framebuffer_bindings_on_tex_image", angle::FeatureCategory::FrontendWorkarounds,
        "On some drivers TexImage sometimes seems to interact "
        "with the Framebuffer",
        &members};

    angle::Feature scalarizeVecAndMatConstructorArgs = {
        "scalarize_vec_and_mat_constructor_args", angle::FeatureCategory::FrontendWorkarounds,
        "Always rewrite vec/mat constructors to be consistent", &members,
        "http://crbug.com/398694"};
};

inline FrontendFeatures::FrontendFeatures()  = default;
inline FrontendFeatures::~FrontendFeatures() = default;

}  // namespace angle

#endif  // ANGLE_PLATFORM_FRONTENDFEATURES_H_
