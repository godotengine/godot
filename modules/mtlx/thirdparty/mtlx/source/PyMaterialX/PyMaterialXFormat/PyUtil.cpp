//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Node.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyUtil(py::module& mod)
{
    mod.def("readFile", &mx::readFile);
    mod.def("getSubdirectories", &mx::getSubdirectories);
    mod.def("loadDocuments", &mx::loadDocuments,
        py::arg("rootPath"), py::arg("searchPath"), py::arg("skipFiles"), py::arg("includeFiles"), py::arg("documents"), py::arg("documentsPaths"),
        py::arg("readOptions") = (mx::XmlReadOptions*) nullptr, py::arg("errors") = (mx::StringVec*) nullptr);
    mod.def("loadLibrary", &mx::loadLibrary,
        py::arg("file"), py::arg("doc"), py::arg("searchPath") = mx::FileSearchPath(), py::arg("readOptions") = (mx::XmlReadOptions*) nullptr);
    mod.def("loadLibraries", &mx::loadLibraries,
        py::arg("libraryFolders"), py::arg("searchPath"), py::arg("doc"), py::arg("excludeFiles") = mx::StringSet(), py::arg("readOptions") = (mx::XmlReadOptions*) nullptr);
    mod.def("flattenFilenames", &mx::flattenFilenames,
        py::arg("doc"), py::arg("searchPath") = mx::FileSearchPath(), py::arg("customResolver") = (mx::StringResolverPtr) nullptr);
    mod.def("getSourceSearchPath", &mx::getSourceSearchPath);
}
