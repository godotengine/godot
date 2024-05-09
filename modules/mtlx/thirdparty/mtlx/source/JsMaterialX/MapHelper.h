//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

/**
 * Include this in every file that defines Emscripten bindings for functions with
 * std::unordered_map parameters or return types, to automatically convert them to / from JS objects.
 * It actually doesn't hurt to include this in every binding file ;)
 * Note that this only works for types that are known to Emscripten, i.e. primitive (built-in) types
 * and types that have bindings defined.
 */

#ifndef JSMATERIALX_MAP_HELPER_H
#define JSMATERIALX_MAP_HELPER_H

#ifdef __EMSCRIPTEN__

#include <emscripten/bind.h>

#include <memory>
#include <unordered_map>

namespace emscripten {
namespace internal {

template <typename T>
std::unordered_map<std::string, T> unorderedMapFromJSObject(const val& m) {

    val keys = val::global("Object").call<val>("entries", m);
    size_t length = keys["length"].as<size_t>();
    std::unordered_map<std::string, T> rm;
    for (size_t i = 0; i < length; ++i) {
        rm.set(m[i][0].as<T>(), m[i][1].as<T>());
    }
    
    return rm;
}

template<typename T>
struct TypeID<std::unordered_map<std::string, T>> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct TypeID<const std::unordered_map<std::string, T>> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct TypeID<std::unordered_map<std::string, T>&> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct TypeID<const std::unordered_map<std::string, T>&> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct BindingType<std::unordered_map<std::string, T>> {
    using ValBinding = BindingType<val>;
    using WireType = ValBinding::WireType;

    static WireType toWireType(const std::unordered_map<std::string, T> &map) {        
        val obj = val::object();
        for (std::pair<std::string, T> element : map)
        {
            obj.set(element.first, element.second);
        }
        return ValBinding::toWireType(obj);
    }

    static std::unordered_map<std::string, T> fromWireType(WireType value) {
        return unorderedMapFromJSObject<T>(ValBinding::fromWireType(value));
    }
};

}  // namespace internal
}  // namespace emscripten

#endif // __EMSCRIPTEN__
#endif // JSMATERIALX_MAP_HELPER_H
