//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_PYMATERIALX_H
#define MATERIALX_PYMATERIALX_H

//
// This header is used to include PyBind11 headers consistently across the
// translation units in the PyMaterialX library, and it should be the first
// include within any PyMaterialX source file.
//

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

// Define a macro to import a PyMaterialX module, e.g. `PyMaterialXCore`,
// either within the `MaterialX` Python package, e.g. in `installed/python/`,
// or as a standalone module, e.g. in `lib/`
#define PYMATERIALX_IMPORT_MODULE(MODULE_NAME)                               \
    try                                                                      \
    {                                                                        \
        pybind11::module::import("MaterialX." #MODULE_NAME);                 \
    }                                                                        \
    catch (const py::error_already_set&)                                     \
    {                                                                        \
        pybind11::module::import(#MODULE_NAME);                              \
    }
#endif
