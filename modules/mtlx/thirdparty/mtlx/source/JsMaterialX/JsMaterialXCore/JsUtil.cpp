//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>

#include <MaterialXCore/Util.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(util)
{
    ems::constant("EMPTY_STRING", mx::EMPTY_STRING);

    ems::function("getVersionString", &mx::getVersionString);

    ems::value_array<std::pair<int, int>>("IntegerPair")
        .element(&std::pair<int, int>::first)
        .element(&std::pair<int, int>::second);

    ems::value_array<std::array<int, 3>>("Integer3Array")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>());

    ems::function("getVersionIntegers", ems::optional_override([]() {
        std::tuple<int, int, int> version = mx::getVersionIntegers();
        return std::array<int, 3> { std::get<0>(version), std::get<1>(version), std::get<2>(version) };
    }));

    // Emscripten expects to provide a number from JS for a cpp 'char' parameter. 
    // Using a string seems to be the better interface for JS
    ems::function("createValidName", ems::optional_override([](std::string name) {
        return mx::createValidName(name);
    }));
    ems::function("createValidName", ems::optional_override([](std::string name, std::string replaceChar) {
        return mx::createValidName(name, replaceChar.front());
    }));

    ems::function("isValidName", &mx::isValidName);
    ems::function("incrementName", &mx::incrementName);

    ems::function("splitNamePath", &mx::splitNamePath);
    ems::function("createNamePath", &mx::createNamePath);
    ems::function("parentNamePath", &mx::parentNamePath);
}
