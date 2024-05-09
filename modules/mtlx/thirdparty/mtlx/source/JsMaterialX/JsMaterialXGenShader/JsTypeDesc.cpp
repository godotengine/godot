//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/TypeDesc.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(TypeDesc)
{
    ems::enum_<mx::TypeDesc::BaseType>("BaseType")
        .value("BASETYPE_NONE", mx::TypeDesc::BaseType::BASETYPE_NONE)
        .value("BASETYPE_BOOLEAN", mx::TypeDesc::BaseType::BASETYPE_BOOLEAN)
        .value("BASETYPE_INTEGER", mx::TypeDesc::BaseType::BASETYPE_INTEGER)
        .value("BASETYPE_FLOAT", mx::TypeDesc::BaseType::BASETYPE_FLOAT)
        .value("BASETYPE_STRING", mx::TypeDesc::BaseType::BASETYPE_STRING)
        .value("BASETYPE_STRUCT", mx::TypeDesc::BaseType::BASETYPE_STRUCT)
        .value("BASETYPE_LAST", mx::TypeDesc::BaseType::BASETYPE_LAST)
        ;

    ems::class_<mx::TypeDesc>("TypeDesc")
        .function("getBaseType", &mx::TypeDesc::getBaseType)
        .function("isAggregate", &mx::TypeDesc::isAggregate)
        .function("getName", &mx::TypeDesc::getName)
        ;
}
