//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXFormat/File.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyFile(py::module& mod)
{
    py::enum_<mx::FilePath::Type>(mod, "Type")
        .value("TypeRelative", mx::FilePath::Type::TypeRelative)
        .value("TypeAbsolute", mx::FilePath::Type::TypeAbsolute)
        .value("TypeNetwork", mx::FilePath::Type::TypeNetwork)
        .export_values();

    py::enum_<mx::FilePath::Format>(mod, "Format")
        .value("FormatWindows", mx::FilePath::Format::FormatWindows)
        .value("FormatPosix", mx::FilePath::Format::FormatPosix)
        .value("FormatNative", mx::FilePath::Format::FormatNative)
        .export_values();

    py::class_<mx::FilePath>(mod, "FilePath")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self / py::self)
        .def("asString", &mx::FilePath::asString,
             py::arg("format") = mx::FilePath::Format::FormatNative)
        .def("isEmpty", &mx::FilePath::isEmpty)
        .def("isAbsolute", &mx::FilePath::isAbsolute)
        .def("getBaseName", &mx::FilePath::getBaseName)
        .def("getParentPath", &mx::FilePath::getParentPath)
        .def("getExtension", &mx::FilePath::getExtension)
        .def("addExtension", &mx::FilePath::addExtension)
        .def("removeExtension", &mx::FilePath::removeExtension)
        .def("size", &mx::FilePath::size)
        .def("getNormalized", &mx::FilePath::getNormalized)        
        .def("exists", &mx::FilePath::exists)
        .def("isDirectory", &mx::FilePath::isDirectory)
        .def("getFilesInDirectory", &mx::FilePath::getFilesInDirectory)
        .def("getSubDirectories", &mx::FilePath::getSubDirectories)
        .def("createDirectory", &mx::FilePath::createDirectory)
        .def_static("getCurrentPath", &mx::FilePath::getCurrentPath)
        .def_static("getModulePath", &mx::FilePath::getModulePath);

    py::class_<mx::FileSearchPath>(mod, "FileSearchPath")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("searchPath"), py::arg("sep") = mx::PATH_LIST_SEPARATOR)
        .def("asString", &mx::FileSearchPath::asString,
             py::arg("sep") = mx::PATH_LIST_SEPARATOR)
        .def("append", static_cast<void (mx::FileSearchPath::*)(const mx::FilePath&)>(&mx::FileSearchPath::append))
        .def("append", static_cast<void (mx::FileSearchPath::*)(const mx::FileSearchPath&)>(&mx::FileSearchPath::append))
        .def("prepend", &mx::FileSearchPath::prepend)
        .def("clear", &mx::FileSearchPath::clear)
        .def("size", &mx::FileSearchPath::size)
        .def("isEmpty", &mx::FileSearchPath::isEmpty)
        .def("find", &mx::FileSearchPath::find);

    py::implicitly_convertible<std::string, mx::FilePath>();
    py::implicitly_convertible<std::string, mx::FileSearchPath>();

    mod.attr("PATH_LIST_SEPARATOR") = mx::PATH_LIST_SEPARATOR;
    mod.attr("MATERIALX_SEARCH_PATH_ENV_VAR") = mx::MATERIALX_SEARCH_PATH_ENV_VAR;
}
