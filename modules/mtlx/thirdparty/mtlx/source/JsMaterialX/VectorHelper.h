//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

/**
 * Include this in every file that defines Emscripten bindings for functions with
 * std::vector parameters or return types, to automatically convert them to / from JS arrays.
 * It actually doesn't hurt to include this in every binding file ;)
 * Note that this only works for types that are known to Emscripten, i.e. primitive (built-in) types
 * and types that have bindings defined.
 */

#ifndef JSMATERIALX_VECTOR_HELPERS_H
#define JSMATERIALX_VECTOR_HELPERS_H

#ifdef __EMSCRIPTEN__

#include <emscripten/bind.h>

#include <memory>
#include <vector>

namespace emscripten {
namespace internal {

template<typename T>
struct TypeID<std::vector<T>> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct TypeID<const std::vector<T>> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct TypeID<std::vector<T>&> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct TypeID<const std::vector<T>&> {
    static constexpr TYPEID get() {
        return LightTypeID<val>::get();
    }
};

template<typename T>
struct BindingType<std::vector<T>> {
    using ValBinding = BindingType<val>;
    using WireType = ValBinding::WireType;

    static WireType toWireType(const std::vector<T> &vec) {        
        WireType result = ValBinding::toWireType(val::array(vec));
        return result;
    }

    static std::vector<T> fromWireType(WireType value) {
        return vecFromJSArray<T>(ValBinding::fromWireType(value));
    }
};

// std<bool> are stored using bits, therefore the default emscripten
// implementation doesn't work for them and need to be specialized
template <>
struct BindingType<std::vector<bool>> {
    using ValBinding = BindingType<val>;
    using WireType = ValBinding::WireType;

     static WireType toWireType(const std::vector<bool> &vec) {
        val out = val::array();
        for (auto i: vec) {
          out.call<void>("push", i == 1 ? true : false);
        }
        WireType result = ValBinding::toWireType(out);
        return result;
    }

    static std::vector<bool> fromWireType(WireType value) {        
        return vecFromJSArray<bool>(ValBinding::fromWireType(value));
    }
};

/**
 * Vectors of smart pointers need special treatment. The above generic toWireType definition uses val::array(vec),
 * which constructs a val::array using val::array.call<void>("push", element). This leads to invalid (deleted) smart
 * pointers on the JS side, since the generated code constructs the smart pointer object, pushes it into a JS array,
 * and then deletes the smart pointer object (i.e. sets the ref count to 0). Using val::array.set() doesn't suffer from
 * this issue.
 */
template<typename T>
struct BindingType<std::vector<std::shared_ptr<T>>> {
    using ValBinding = BindingType<val>;
    using WireType = ValBinding::WireType;

    static WireType toWireType(const std::vector<std::shared_ptr<T>> &vec) {
        auto arr = val::array();
        for (int i = 0; i < vec.size(); ++i) {
            arr.set(i, vec.at(i));
        }
        return ValBinding::toWireType(arr);
    }

    static std::vector<std::shared_ptr<T>> fromWireType(WireType value) {
        return vecFromJSArray<std::shared_ptr<T>>(ValBinding::fromWireType(value));
    }
};

}  // namespace internal
}  // namespace emscripten

#endif // __EMSCRIPTEN__
#endif // JSMATERIALX_VECTOR_HELPERS_H