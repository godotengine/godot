#pragma once

// Consolidated extern "C" trampoline decls for this framework.
// One entry per (return, args, selector) — identical C++ signatures
// across multiple classes collapse to a single linker alias of
// `_objc_msgSend$<selector>`. Per-class headers include this file
// instead of declaring their own externs.

#include "MTL4FXDefines.hpp"
#include <objc/objc.h>
#include "../Foundation/NSTypes.hpp"

namespace MTL4 {
    class CommandBuffer;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

extern "C" {
void _MTL4FX_msg_v_encodeToCommandBuffer__MTL4__CommandBufferp(const void*, SEL, MTL4::CommandBuffer*) __asm__("_objc_msgSend$" "encodeToCommandBuffer:");
} // extern "C"

#pragma clang diagnostic pop
