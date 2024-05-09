//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include "../MapHelper.h"

#include <MaterialXGenShader/ShaderStage.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

mx::ShaderPort* get(mx::VariableBlock& self, const ems::val& value) {
    if (value.typeOf().as<std::string>() == "string")
        return self[value.as<std::string>()];
    else
        return self[value.as<size_t>()];
}

EMSCRIPTEN_BINDINGS(ShaderStage)
{
    ems::class_<mx::VariableBlock>("VariableBlock")
        .smart_ptr<std::shared_ptr<mx::VariableBlock>>("VariableBlockPtr")
        .function("empty", &mx::VariableBlock::empty)
        .function("size", &mx::VariableBlock::size)
        .function("get", &get, ems::allow_raw_pointers())
        .function("find", ems::select_overload<const mx::ShaderPort* (const std::string&) const>(&mx::VariableBlock::find), ems::allow_raw_pointers())
        ;

    ems::class_<mx::ShaderStage>("ShaderStage")
        .function("getUniformBlocks", &mx::ShaderStage::getUniformBlocks);
        ;
}
