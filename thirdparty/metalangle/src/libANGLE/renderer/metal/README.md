# Current limitations (TODO list)
- No Command Buffer reordering like Vulkan backend.
- ~~No ES 3.0 support (multiple render targets, instanced draw, etc).~~
- No transform feedback support.
- ~~No triangle fan, line loop support.~~
- ~~iOS pre iPhone 6S: No array of samplers support in shader.~~
- ~~No multisample support.~~
- unsigned byte index is not natively supported. Metal backend will do CPU or GPU (whenever
  possible) conversion during draw calls.
- offset passed to glVertexAttribPointer() must be multiple of attribute's format's bytes.
  Otherwise, ~~a CPU conversion will take place.~~ it will be converted (using GPU whenever
  possible) during draw calls.
- stride passed to glVertexAttribPointer() must be multiple of attribute's format's bytes.
  Otherwise, ~~a CPU conversion will take place.~~ it will be converted (using GPU whenever
  possible) during draw calls.
- indices offset passed to glDrawElements() must be multiple of 4 bytes. Otherwise, ~~a CPU
  conversion will take place.~~ it will be converted (using GPU whenever possible) during draw
  calls.
- Only support iOS 11.0+ vs MacOS 10.13+. Technically, iOS 9.0 to 10.0 can still use MetalANGLE framework. However, if pre iOS 11.0 runtime is detected, the framework will fallback to use native OpenGL ES implementation instead of translating draw calls to Metal.
This is because pre iOS 11.0 version of Metal runtime doesn't support some essencial features needed by the Metal backend.

# Failed ANGLE end2end tests
- DifferentStencilMasksTest.DrawWithDifferentMask
- ~~MipmapTest.DefineValidExtraLevelAndUseItLater~~
- PointSpritesTest.PointSizeAboveMaxIsClamped (point outside framebuffer won't get drawn)
- ~~SimpleStateChangeTest.CopyTexSubImageOnTextureBoundToFrambuffer (GL_ANGLE_framebuffer_blit not implemented)~~
- ~~WebGLReadOutsideFramebufferTest.*~~
