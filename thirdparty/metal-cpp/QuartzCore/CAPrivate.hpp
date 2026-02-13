//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// QuartzCore/CAPrivate.hpp
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

#include "CADefines.hpp"

#include <objc/runtime.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define _CA_PRIVATE_CLS(symbol) (Private::Class::s_k##symbol)
#define _CA_PRIVATE_SEL(accessor) (Private::Selector::s_k##accessor)

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#if defined(CA_PRIVATE_IMPLEMENTATION)

#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _CA_PRIVATE_VISIBILITY __attribute__((visibility("hidden")))
#else
#define _CA_PRIVATE_VISIBILITY __attribute__((visibility("default")))
#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN

#define _CA_PRIVATE_IMPORT __attribute__((weak_import))

#ifdef __OBJC__
#define _CA_PRIVATE_OBJC_LOOKUP_CLASS(symbol) ((__bridge void*)objc_lookUpClass(#symbol))
#define _CA_PRIVATE_OBJC_GET_PROTOCOL(symbol) ((__bridge void*)objc_getProtocol(#symbol))
#else
#define _CA_PRIVATE_OBJC_LOOKUP_CLASS(symbol) objc_lookUpClass(#symbol)
#define _CA_PRIVATE_OBJC_GET_PROTOCOL(symbol) objc_getProtocol(#symbol)
#endif // __OBJC__

#define _CA_PRIVATE_DEF_CLS(symbol) void* s_k##symbol _CA_PRIVATE_VISIBILITY = _CA_PRIVATE_OBJC_LOOKUP_CLASS(symbol)
#define _CA_PRIVATE_DEF_PRO(symbol) void* s_k##symbol _CA_PRIVATE_VISIBILITY = _CA_PRIVATE_OBJC_GET_PROTOCOL(symbol)
#define _CA_PRIVATE_DEF_SEL(accessor, symbol) SEL s_k##accessor _CA_PRIVATE_VISIBILITY = sel_registerName(symbol)
#define _CA_PRIVATE_DEF_STR(type, symbol)                \
    _CA_EXTERN type const CA##symbol _CA_PRIVATE_IMPORT; \
    type const                       CA::symbol = (nullptr != &CA##symbol) ? CA##symbol : nullptr

#else

#define _CA_PRIVATE_DEF_CLS(symbol) extern void* s_k##symbol
#define _CA_PRIVATE_DEF_PRO(symbol) extern void* s_k##symbol
#define _CA_PRIVATE_DEF_SEL(accessor, symbol) extern SEL s_k##accessor
#define _CA_PRIVATE_DEF_STR(type, symbol) extern type const CA::symbol

#endif // CA_PRIVATE_IMPLEMENTATION

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace CA
{
namespace Private
{
    namespace Class
    {
        _CA_PRIVATE_DEF_CLS(CAMetalLayer);
    } // Class
} // Private
} // CA

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace CA
{
namespace Private
{
    namespace Protocol
    {

        _CA_PRIVATE_DEF_PRO(CAMetalDrawable);

    } // Protocol
} // Private
} // CA

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace CA
{
namespace Private
{
    namespace Selector
    {
        _CA_PRIVATE_DEF_SEL(allowsNextDrawableTimeout,
            "allowsNextDrawableTimeout");
        _CA_PRIVATE_DEF_SEL(colorspace,
            "colorspace");
        _CA_PRIVATE_DEF_SEL(device,
            "device");
        _CA_PRIVATE_DEF_SEL(displaySyncEnabled,
            "displaySyncEnabled");
        _CA_PRIVATE_DEF_SEL(drawableSize,
            "drawableSize");
        _CA_PRIVATE_DEF_SEL(framebufferOnly,
            "framebufferOnly");
        _CA_PRIVATE_DEF_SEL(layer,
            "layer");
        _CA_PRIVATE_DEF_SEL(maximumDrawableCount,
            "maximumDrawableCount");
        _CA_PRIVATE_DEF_SEL(nextDrawable,
            "nextDrawable");
        _CA_PRIVATE_DEF_SEL(pixelFormat,
            "pixelFormat");
        _CA_PRIVATE_DEF_SEL(residencySet,
            "residencySet");
        _CA_PRIVATE_DEF_SEL(setAllowsNextDrawableTimeout_,
            "setAllowsNextDrawableTimeout:");
        _CA_PRIVATE_DEF_SEL(setColorspace_,
            "setColorspace:");
        _CA_PRIVATE_DEF_SEL(setDevice_,
            "setDevice:");
        _CA_PRIVATE_DEF_SEL(setDisplaySyncEnabled_,
            "setDisplaySyncEnabled:");
        _CA_PRIVATE_DEF_SEL(setDrawableSize_,
            "setDrawableSize:");
        _CA_PRIVATE_DEF_SEL(setFramebufferOnly_,
            "setFramebufferOnly:");
        _CA_PRIVATE_DEF_SEL(setMaximumDrawableCount_,
            "setMaximumDrawableCount:");
        _CA_PRIVATE_DEF_SEL(setPixelFormat_,
            "setPixelFormat:");
        _CA_PRIVATE_DEF_SEL(texture,
            "texture");
    } // Class
} // Private
} // CA

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
