//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// angle_windowsstore.h:

#ifndef ANGLE_WINDOWSSTORE_H_
#define ANGLE_WINDOWSSTORE_H_

// The following properties can be set on the CoreApplication to support additional
// ANGLE configuration options.
//
// The Visual Studio sample templates provided with this version of ANGLE have examples
// of how to set these property values.

//
// Property: EGLNativeWindowTypeProperty
// Type: IInspectable
// Description: Set this property to specify the window type to use for creating a surface.
//              If this property is missing, surface creation will fail.
//
const wchar_t EGLNativeWindowTypeProperty[] = L"EGLNativeWindowTypeProperty";

//
// Property: EGLRenderSurfaceSizeProperty
// Type: Size
// Description: Set this property to specify a preferred size in pixels of the render surface.
//              The render surface size width and height must be greater than 0.
//              If this property is set, then the render surface size is fixed.
//              The render surface will then be scaled to the window dimensions.
//              If this property is missing, a default behavior will be provided.
//              The default behavior uses the window size if a CoreWindow is specified or
//              the size of the SwapChainPanel control if one is specified.
//
const wchar_t EGLRenderSurfaceSizeProperty[] = L"EGLRenderSurfaceSizeProperty";

//
// Property: EGLRenderResolutionScaleProperty
// Type: Single
// Description: Use this to specify a preferred scale for the render surface compared to the window.
//              For example, if the window is 800x480, and:
//                - scale is set to 0.5f then the surface will be 400x240
//                - scale is set to 1.2f then the surface will be 960x576
//              If the window resizes or rotates then the surface will resize accordingly.
//              EGLRenderResolutionScaleProperty and EGLRenderSurfaceSizeProperty cannot both be set.
//              The scale factor should be > 0.0f.
//
const wchar_t EGLRenderResolutionScaleProperty[] = L"EGLRenderResolutionScaleProperty";

#endif // ANGLE_WINDOWSSTORE_H_
