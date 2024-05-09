//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/Helpers.h>

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(Shader)
{
    ems::class_<mx::Shader>("Shader")
        .smart_ptr<std::shared_ptr<mx::Shader>>("ShaderPtr")
        .function("getSourceCode", &mx::Shader::getSourceCode)
        .function("getStage", PTR_RETURN_OVERLOAD(mx::ShaderStage& (mx::Shader::*)(const std::string&), &mx::Shader::getStage), ems::allow_raw_pointers())
        ;
}
