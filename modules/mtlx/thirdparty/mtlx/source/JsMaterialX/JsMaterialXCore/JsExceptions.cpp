//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <emscripten/bind.h>

#include <string>

namespace ems = emscripten;

namespace jsexceptions
{
std::string getExceptionMessage(int exceptionPtr)
{
    return std::string(reinterpret_cast<std::exception *>(exceptionPtr)->what());
}
} // namespace jsexceptions

EMSCRIPTEN_BINDINGS(exceptions)
{
    ems::function("getExceptionMessage", &jsexceptions::getExceptionMessage);
}
