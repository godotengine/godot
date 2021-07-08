//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// export.h : Defines ANGLE_EXPORT, a macro for exporting functions from the DLL

#ifndef LIBGLESV2_EXPORT_H_
#define LIBGLESV2_EXPORT_H_

#if !defined(ANGLE_EXPORT)
#    if defined(_WIN32)
#        if defined(LIBGLESV2_IMPLEMENTATION) || defined(LIBANGLE_IMPLEMENTATION) || \
            defined(LIBFEATURE_SUPPORT_IMPLEMENTATION)
#            define ANGLE_EXPORT __declspec(dllexport)
#        else
#            define ANGLE_EXPORT __declspec(dllimport)
#        endif
#    elif defined(__GNUC__)
#        if defined(LIBGLESV2_IMPLEMENTATION) || defined(LIBANGLE_IMPLEMENTATION) || \
            defined(LIBFEATURE_SUPPORT_IMPLEMENTATION)
#            define ANGLE_EXPORT __attribute__((visibility("default")))
#        else
#            define ANGLE_EXPORT
#        endif
#    else
#        define ANGLE_EXPORT
#    endif
#endif  // !defined(ANGLE_EXPORT)

#if !defined(ANGLE_NO_EXPORT)
#    if defined(__GNUC__)
#        define ANGLE_NO_EXPORT __attribute__((visibility("hidden")))
#    else
#        define ANGLE_NO_EXPORT
#    endif
#endif  // !defined(ANGLE_NO_EXPORT)

#endif  // LIBGLESV2_EXPORT_H_
