//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Config.h: Defines the egl::Config class, describing the format, type
// and size for an egl::Surface. Implements EGLConfig and related functionality.
// [EGL 1.5] section 3.4 page 19.

#ifndef INCLUDE_CONFIG_H_
#define INCLUDE_CONFIG_H_

#include "libANGLE/AttributeMap.h"

#include "common/angleutils.h"

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <map>
#include <vector>

namespace egl
{

struct Config
{
    Config();
    ~Config();
    Config(const Config &other);
    Config &operator=(const Config &other);

    GLenum renderTargetFormat;  // TODO(geofflang): remove this
    GLenum depthStencilFormat;  // TODO(geofflang): remove this

    EGLint bufferSize;             // Depth of the color buffer
    EGLint redSize;                // Bits of Red in the color buffer
    EGLint greenSize;              // Bits of Green in the color buffer
    EGLint blueSize;               // Bits of Blue in the color buffer
    EGLint luminanceSize;          // Bits of Luminance in the color buffer
    EGLint alphaSize;              // Bits of Alpha in the color buffer
    EGLint alphaMaskSize;          // Bits of Alpha Mask in the mask buffer
    EGLBoolean bindToTextureRGB;   // True if bindable to RGB textures.
    EGLBoolean bindToTextureRGBA;  // True if bindable to RGBA textures.
    EGLenum colorBufferType;       // Color buffer type
    EGLenum configCaveat;          // Any caveats for the configuration
    EGLint configID;               // Unique EGLConfig identifier
    EGLint conformant;             // Whether contexts created with this config are conformant
    EGLint depthSize;              // Bits of Z in the depth buffer
    EGLint level;                  // Frame buffer level
    EGLBoolean matchNativePixmap;  // Match the native pixmap format
    EGLint maxPBufferWidth;        // Maximum width of pbuffer
    EGLint maxPBufferHeight;       // Maximum height of pbuffer
    EGLint maxPBufferPixels;       // Maximum size of pbuffer
    EGLint maxSwapInterval;        // Maximum swap interval
    EGLint minSwapInterval;        // Minimum swap interval
    EGLBoolean nativeRenderable;   // EGL_TRUE if native rendering APIs can render to surface
    EGLint nativeVisualID;         // Handle of corresponding native visual
    EGLint nativeVisualType;       // Native visual type of the associated visual
    EGLint renderableType;         // Which client rendering APIs are supported.
    EGLint sampleBuffers;          // Number of multisample buffers
    EGLint samples;                // Number of samples per pixel
    EGLint stencilSize;            // Bits of Stencil in the stencil buffer
    EGLint surfaceType;            // Which types of EGL surfaces are supported.
    EGLenum transparentType;       // Type of transparency supported
    EGLint transparentRedValue;    // Transparent red value
    EGLint transparentGreenValue;  // Transparent green value
    EGLint transparentBlueValue;   // Transparent blue value
    EGLint optimalOrientation;     // Optimal window surface orientation
    EGLenum colorComponentType;    // Color component type
    EGLBoolean recordable;         // EGL_TRUE if a surface can support recording on Android
};

class ConfigSet
{
  private:
    typedef std::map<EGLint, Config> ConfigMap;

  public:
    ConfigSet();
    ConfigSet(const ConfigSet &other);
    ~ConfigSet();
    ConfigSet &operator=(const ConfigSet &other);

    EGLint add(const Config &config);
    const Config &get(EGLint id) const;

    void clear();

    size_t size() const;

    bool contains(const Config *config) const;

    // Filter configurations based on the table in [EGL 1.5] section 3.4.1.2 page 29
    std::vector<const Config *> filter(const AttributeMap &attributeMap) const;

    ConfigMap::iterator begin();
    ConfigMap::iterator end();

  private:
    ConfigMap mConfigs;
};

}  // namespace egl

#endif  // INCLUDE_CONFIG_H_
