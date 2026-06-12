#pragma once

// Consolidated extern "C" trampoline decls for this framework.
// One entry per (return, args, selector) — identical C++ signatures
// across multiple classes collapse to a single linker alias of
// `_objc_msgSend$<selector>`. Per-class headers include this file
// instead of declaring their own externs.

#include "CADefines.hpp"
#include <objc/objc.h>
#include "../Foundation/NSTypes.hpp"
#include "CAStructs.hpp"
#include <CoreGraphics/CoreGraphics.h>

namespace CA {
    class Layer;
    class MetalDrawable;
    class MetalLayer;
}
namespace MTL {
    class Device;
    class ResidencySet;
    class Texture;
    enum PixelFormat : NS::UInteger;
}
namespace NS {
    class String;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

extern "C" {
CA::Layer* _CA_msg_CA__Layerp_alloc(const void*, SEL) __asm__("_objc_msgSend$" "alloc");
CA::MetalLayer* _CA_msg_CA__MetalLayerp_alloc(const void*, SEL) __asm__("_objc_msgSend$" "alloc");
bool _CA_msg_bool_allowsNextDrawableTimeout(const void*, SEL) __asm__("_objc_msgSend$" "allowsNextDrawableTimeout");
CGColorSpaceRef _CA_msg_CGColorSpaceRef_colorspace(const void*, SEL) __asm__("_objc_msgSend$" "colorspace");
CGFloat _CA_msg_CGFloat_contentsHeadroom(const void*, SEL) __asm__("_objc_msgSend$" "contentsHeadroom");
MTL::Device* _CA_msg_MTL__Devicep_device(const void*, SEL) __asm__("_objc_msgSend$" "device");
bool _CA_msg_bool_displaySyncEnabled(const void*, SEL) __asm__("_objc_msgSend$" "displaySyncEnabled");
CGSize _CA_msg_CGSize_drawableSize(const void*, SEL) __asm__("_objc_msgSend$" "drawableSize");
bool _CA_msg_bool_framebufferOnly(const void*, SEL) __asm__("_objc_msgSend$" "framebufferOnly");
CA::Layer* _CA_msg_CA__Layerp_init(const void*, SEL) __asm__("_objc_msgSend$" "init");
CA::MetalLayer* _CA_msg_CA__MetalLayerp_init(const void*, SEL) __asm__("_objc_msgSend$" "init");
CA::MetalLayer* _CA_msg_CA__MetalLayerp_layer(const void*, SEL) __asm__("_objc_msgSend$" "layer");
NS::UInteger _CA_msg_NS__UInteger_maximumDrawableCount(const void*, SEL) __asm__("_objc_msgSend$" "maximumDrawableCount");
CA::MetalDrawable* _CA_msg_CA__MetalDrawablep_nextDrawable(const void*, SEL) __asm__("_objc_msgSend$" "nextDrawable");
bool _CA_msg_bool_opaque(const void*, SEL) __asm__("_objc_msgSend$" "opaque");
MTL::PixelFormat _CA_msg_MTL__PixelFormat_pixelFormat(const void*, SEL) __asm__("_objc_msgSend$" "pixelFormat");
NS::String* _CA_msg_NS__Stringp_preferredDynamicRange(const void*, SEL) __asm__("_objc_msgSend$" "preferredDynamicRange");
MTL::ResidencySet* _CA_msg_MTL__ResidencySetp_residencySet(const void*, SEL) __asm__("_objc_msgSend$" "residencySet");
void _CA_msg_v_setAllowsNextDrawableTimeout__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setAllowsNextDrawableTimeout:");
void _CA_msg_v_setColorspace__CGColorSpaceRef(const void*, SEL, CGColorSpaceRef) __asm__("_objc_msgSend$" "setColorspace:");
void _CA_msg_v_setContentsHeadroom__CGFloat(const void*, SEL, CGFloat) __asm__("_objc_msgSend$" "setContentsHeadroom:");
void _CA_msg_v_setDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "setDevice:");
void _CA_msg_v_setDisplaySyncEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setDisplaySyncEnabled:");
void _CA_msg_v_setDrawableSize__CGSize(const void*, SEL, CGSize) __asm__("_objc_msgSend$" "setDrawableSize:");
void _CA_msg_v_setFramebufferOnly__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setFramebufferOnly:");
void _CA_msg_v_setMaximumDrawableCount__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setMaximumDrawableCount:");
void _CA_msg_v_setOpaque__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setOpaque:");
void _CA_msg_v_setPixelFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setPixelFormat:");
void _CA_msg_v_setPreferredDynamicRange__NS__Stringp(const void*, SEL, NS::String*) __asm__("_objc_msgSend$" "setPreferredDynamicRange:");
void _CA_msg_v_setWantsExtendedDynamicRangeContent__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setWantsExtendedDynamicRangeContent:");
MTL::Texture* _CA_msg_MTL__Texturep_texture(const void*, SEL) __asm__("_objc_msgSend$" "texture");
bool _CA_msg_bool_wantsExtendedDynamicRangeContent(const void*, SEL) __asm__("_objc_msgSend$" "wantsExtendedDynamicRangeContent");
} // extern "C"

#pragma clang diagnostic pop
