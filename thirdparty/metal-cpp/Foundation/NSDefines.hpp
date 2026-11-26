//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSDefines.hpp
//
// Copyright 2020-2024 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

// Forward declarations to avoid conflicts with Godot types (String, Object, Error)
namespace NS {
class Array;
class Dictionary;
class Error;
class Object;
class String;
class URL;
} // namespace NS

#define _NS_WEAK_IMPORT __attribute__((weak_import))
#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _NS_EXPORT __attribute__((visibility("hidden")))
#else
#define _NS_EXPORT __attribute__((visibility("default")))
#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _NS_EXTERN extern "C" _NS_EXPORT
#define _NS_INLINE inline __attribute__((always_inline))
#define _NS_PACKED __attribute__((packed))

#define _NS_CONST(type, name) _NS_EXTERN type const name
#define _NS_ENUM(type, name) enum name : type
#define _NS_OPTIONS(type, name) \
    using name = type;          \
    enum : name

#define _NS_CAST_TO_UINT(value) static_cast<NS::UInteger>(value)
#define _NS_VALIDATE_SIZE(ns, name) static_assert(sizeof(ns::name) == sizeof(ns##name), "size mismatch " #ns "::" #name)
#define _NS_VALIDATE_ENUM(ns, name) static_assert(_NS_CAST_TO_UINT(ns::name) == _NS_CAST_TO_UINT(ns##name), "value mismatch " #ns "::" #name)

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
