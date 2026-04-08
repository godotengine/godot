//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLPrivate.hpp
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

#include "MTLDefines.hpp"

#include <objc/runtime.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define _MTL_PRIVATE_CLS(symbol) (MTL::Private::Class::s_k##symbol)
#define _MTL_PRIVATE_SEL(accessor) (MTL::Private::Selector::s_k##accessor)

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#if defined(MTL_PRIVATE_IMPLEMENTATION)

#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _MTL_PRIVATE_VISIBILITY __attribute__((visibility("hidden")))
#else
#define _MTL_PRIVATE_VISIBILITY __attribute__((visibility("default")))
#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN

#define _MTL_PRIVATE_IMPORT __attribute__((weak_import))

#ifdef __OBJC__
#define _MTL_PRIVATE_OBJC_LOOKUP_CLASS(symbol) ((__bridge void*)objc_lookUpClass(#symbol))
#define _MTL_PRIVATE_OBJC_GET_PROTOCOL(symbol) ((__bridge void*)objc_getProtocol(#symbol))
#else
#define _MTL_PRIVATE_OBJC_LOOKUP_CLASS(symbol) objc_lookUpClass(#symbol)
#define _MTL_PRIVATE_OBJC_GET_PROTOCOL(symbol) objc_getProtocol(#symbol)
#endif // __OBJC__

#define _MTL_PRIVATE_DEF_CLS(symbol) void* s_k##symbol _MTL_PRIVATE_VISIBILITY = _MTL_PRIVATE_OBJC_LOOKUP_CLASS(symbol)
#define _MTL_PRIVATE_DEF_PRO(symbol) void* s_k##symbol _MTL_PRIVATE_VISIBILITY = _MTL_PRIVATE_OBJC_GET_PROTOCOL(symbol)
#define _MTL_PRIVATE_DEF_SEL(accessor, symbol) SEL s_k##accessor _MTL_PRIVATE_VISIBILITY = sel_registerName(symbol)

#include <dlfcn.h>
#define MTL_DEF_FUNC( name, signature ) \
    using Fn##name = signature; \
    Fn##name name = reinterpret_cast< Fn##name >( dlsym( RTLD_DEFAULT, #name ) )

namespace MTL::Private
{
    template <typename _Type>
    inline _Type const LoadSymbol(const char* pSymbol)
    {
        const _Type* pAddress = static_cast<_Type*>(dlsym(RTLD_DEFAULT, pSymbol));

        return pAddress ? *pAddress : nullptr;
    }
} // MTL::Private

#if defined(__MAC_26_0) || defined(__IPHONE_26_0) || defined(__TVOS_26_0)

#define _MTL_PRIVATE_DEF_STR(type, symbol)                  \
    _MTL_EXTERN type const MTL##symbol _MTL_PRIVATE_IMPORT; \
    type const                         MTL::symbol = (nullptr != &MTL##symbol) ? MTL##symbol : nullptr

#define _MTL_PRIVATE_DEF_CONST(type, symbol)              \
    _MTL_EXTERN type const MTL##symbol _MTL_PRIVATE_IMPORT; \
    type const                         MTL::symbol = (nullptr != &MTL##symbol) ? MTL##symbol : nullptr

#define _MTL_PRIVATE_DEF_WEAK_CONST(type, symbol) \
    _MTL_EXTERN type const MTL##symbol;    \
    type const             MTL::symbol = MTL::Private::LoadSymbol<type>("MTL" #symbol)

#else

#define _MTL_PRIVATE_DEF_STR(type, symbol) \
    _MTL_EXTERN type const MTL##symbol;    \
    type const             MTL::symbol = MTL::Private::LoadSymbol<type>("MTL" #symbol)

#define _MTL_PRIVATE_DEF_CONST(type, symbol) \
    _MTL_EXTERN type const MTL##symbol;    \
    type const             MTL::symbol = MTL::Private::LoadSymbol<type>("MTL" #symbol)

#define _MTL_PRIVATE_DEF_WEAK_CONST(type, symbol) _MTL_PRIVATE_DEF_CONST(type, symbol)

#endif

#else

#define _MTL_PRIVATE_DEF_CLS(symbol) extern void* s_k##symbol
#define _MTL_PRIVATE_DEF_PRO(symbol) extern void* s_k##symbol
#define _MTL_PRIVATE_DEF_SEL(accessor, symbol) extern SEL s_k##accessor
#define _MTL_PRIVATE_DEF_STR(type, symbol) extern type const MTL::symbol
#define _MTL_PRIVATE_DEF_CONST(type, symbol) extern type const MTL::symbol
#define _MTL_PRIVATE_DEF_WEAK_CONST(type, symbol) extern type const MTL::symbol

#endif // MTL_PRIVATE_IMPLEMENTATION

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL
{
namespace Private
{
    namespace Class
    {

    } // Class
} // Private
} // MTL

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL
{
namespace Private
{
    namespace Protocol
    {

    } // Protocol
} // Private
} // MTL

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL
{
namespace Private
{
    namespace Selector
    {

        _MTL_PRIVATE_DEF_SEL(beginScope,
            "beginScope");
        _MTL_PRIVATE_DEF_SEL(endScope,
            "endScope");
    } // Class
} // Private
} // MTL

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
