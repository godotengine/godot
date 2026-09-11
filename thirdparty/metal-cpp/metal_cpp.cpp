/**************************************************************************/
/*  metal_cpp.cpp                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// Only NS_PRIVATE_IMPLEMENTATION is still meaningful — it flips
// Foundation/NSPrivate.hpp from `extern SEL s_k…` declarations to definitions
// so the SEL globals consumed by upstream NSObject.hpp's base ops (retain /
// release / autorelease / description / hash / isEqual:) get instantiated
// here. The Metal / MetalFX / QuartzCore framework wrappers dispatch through
// linker-synthesized `_objc_msgSend$<sel>` stubs and require no per-TU
// definitions, so their private-implementation macros are gone.
#define NS_PRIVATE_IMPLEMENTATION

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "MetalFX/MetalFX.hpp"
#include "QuartzCore/QuartzCore.hpp"

// Definition of MTL::CreateSystemDefaultDevice — kept here so the system
// `MTLCreateSystemDefaultDevice` extern (which carries an ARC-relevant
// `NS_RETURNS_RETAINED` attribute in Apple's MTLDevice.h) is never visible
// in headers that .mm translation units transitively include.
extern "C" MTL::Device* MTLCreateSystemDefaultDevice();

MTL::Device* MTL::CreateSystemDefaultDevice()
{
    return ::MTLCreateSystemDefaultDevice();
}
