//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Exception.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyException(py::module& mod)
{
    static py::exception<mx::Exception> pyException(mod, "Exception");

    py::register_exception_translator(
        [](std::exception_ptr errPtr)
        {
            try
            {
                if (errPtr != NULL)
                    std::rethrow_exception(errPtr);
            }
            catch (const mx::Exception& err)
            {
                PyErr_SetString(PyExc_LookupError, err.what());
            }
        }
    );

}