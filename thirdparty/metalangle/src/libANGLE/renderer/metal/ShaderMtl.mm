//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderMtl.mm:
//    Implements the class methods for ShaderMtl.
//

#include "libANGLE/renderer/metal/ShaderMtl.h"

#include "common/debug.h"
#include "compiler/translator/TranslatorMetal.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"

namespace rx
{

ShaderMtl::ShaderMtl(const gl::ShaderState &data) : ShaderImpl(data) {}

ShaderMtl::~ShaderMtl() {}

std::shared_ptr<WaitableCompileEvent> ShaderMtl::compile(const gl::Context *context,
                                                         gl::ShCompilerInstance *compilerInstance,
                                                         ShCompileOptions options)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    if (mData.getShaderType() == gl::ShaderType::Vertex &&
        !contextMtl->getDisplay()->getFeatures().hasBaseVertexInstancedDraw.enabled)
    {
        // Emulate gl_InstanceID
        sh::TShHandleBase *base = static_cast<sh::TShHandleBase *>(compilerInstance->getHandle());
        auto translatorMetal    = static_cast<sh::TranslatorMetal *>(base->getAsCompiler());
        translatorMetal->enableEmulatedInstanceID(true);
    }
    ShCompileOptions compileOptions = SH_INITIALIZE_UNINITIALIZED_LOCALS;

    bool isWebGL = context->getExtensions().webglCompatibility;
    if (isWebGL && mData.getShaderType() != gl::ShaderType::Compute)
    {
        compileOptions |= SH_INIT_OUTPUT_VARIABLES;
    }

    if (contextMtl->getDisplay()->getFeatures().emulateDepthRangeMappingInShader.enabled)
    {
        compileOptions |= SH_METAL_EMULATE_LINEAR_DEPTH_RANGE_MAP;
    }

    compileOptions |= SH_CLAMP_POINT_SIZE;

    return compileImpl(context, compilerInstance, mData.getSource(), compileOptions | options);
}

std::string ShaderMtl::getDebugInfo() const
{
    return mData.getTranslatedSource();
}

}  // namespace rx
