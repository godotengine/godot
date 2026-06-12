#pragma once

// Hand-written supplement to the generated `Metal/` headers. Surfaces
// `MTL::CreateSystemDefaultDevice()`, whose definition lives in
// `thirdparty/metal-cpp/metal_cpp.cpp` (kept out of any `.hpp` so that
// `.mm` translation units that also import Apple's `<Metal/Metal.h>`
// don't see two declarations of `MTLCreateSystemDefaultDevice` —
// Apple's carries an ARC-relevant `NS_RETURNS_RETAINED` attribute that
// would conflict with a plain `extern "C"` redeclaration here).

namespace MTL
{
class Device;

Device* CreateSystemDefaultDevice();
}  // namespace MTL
