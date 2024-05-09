//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(ShaderGenerator)
{
    ems::class_<mx::ShaderGenerator>("ShaderGenerator")
        .smart_ptr<std::shared_ptr<mx::ShaderGenerator>>("ShaderGeneratorPtr")
        .function("getTarget", &mx::ShaderGenerator::getTarget)
        .function("generate", &mx::ShaderGenerator::generate)
        ;
}
