/*
    pybind11/detail/typeid.h: Compiler-independent access to type identifiers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(__GNUG__)
#    include <cxxabi.h>
#endif

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Erase all occurrences of a substring
inline void erase_all(std::string &string, const std::string &search) {
    for (size_t pos = 0;;) {
        pos = string.find(search, pos);
        if (pos == std::string::npos) {
            break;
        }
        string.erase(pos, search.length());
    }
}

PYBIND11_NOINLINE void clean_type_id(std::string &name) {
#if defined(__GNUG__)
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free};
    if (status == 0) {
        name = res.get();
    }
#else
    detail::erase_all(name, "class ");
    detail::erase_all(name, "struct ");
    detail::erase_all(name, "enum ");
#endif
    detail::erase_all(name, "pybind11::");
}

inline std::string clean_type_id(const char *typeid_name) {
    std::string name(typeid_name);
    detail::clean_type_id(name);
    return name;
}

PYBIND11_NAMESPACE_END(detail)

/// Return a string representation of a C++ type
template <typename T>
static std::string type_id() {
    return detail::clean_type_id(typeid(T).name());
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
