//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/Environ.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(environ)
{
    ems::function("getEnviron", &mx::getEnviron);
    ems::function("setEnviron", &mx::setEnviron);
    ems::function("removeEnviron", &mx::removeEnviron);
}
