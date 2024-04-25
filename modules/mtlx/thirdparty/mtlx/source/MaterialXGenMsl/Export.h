//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GENMSL_EXPORT_H
#define MATERIALX_GENMSL_EXPORT_H

#include <MaterialXCore/Library.h>

/// @file
/// Macros for declaring imported and exported symbols.

#if defined(MATERIALX_GENMSL_EXPORTS)
    #define MX_GENMSL_API MATERIALX_SYMBOL_EXPORT
    #define MX_GENMSL_EXTERN_TEMPLATE(...) MATERIALX_EXPORT_EXTERN_TEMPLATE(__VA_ARGS__)
#else
    #define MX_GENMSL_API MATERIALX_SYMBOL_IMPORT
    #define MX_GENMSL_EXTERN_TEMPLATE(...) MATERIALX_IMPORT_EXTERN_TEMPLATE(__VA_ARGS__)
#endif

#endif
