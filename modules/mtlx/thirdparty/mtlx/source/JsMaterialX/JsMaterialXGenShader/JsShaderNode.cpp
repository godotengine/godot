//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include "../MapHelper.h"

#include <MaterialXGenShader/ShaderNode.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(ShaderPort)
{
    ems::class_<mx::ShaderPort>("ShaderPort")
        .smart_ptr<std::shared_ptr<mx::ShaderPort>>("ShaderPortPtr")
        .function("getVariable", &mx::ShaderPort::getVariable)
        .function("getType", &mx::ShaderPort::getType, ems::allow_raw_pointers())
        .function("getValue", &mx::ShaderPort::getValue)
        .function("getPath", &mx::ShaderPort::getPath)
        .function("getUnit", &mx::ShaderPort::getUnit)
        .function("getColorSpace", &mx::ShaderPort::getColorSpace)
        .function("setGeomProp", &mx::ShaderPort::setGeomProp)        
        ;
}
