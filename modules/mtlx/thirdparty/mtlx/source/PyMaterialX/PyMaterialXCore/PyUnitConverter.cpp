//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Unit.h>

namespace py = pybind11;
namespace mx = MaterialX;

class PyUnitConverter : public mx::UnitConverter
{
  public:
    float convert(float input, const std::string& inputUnit, const std::string& outputUnit) const override
    {
        PYBIND11_OVERLOAD_PURE(
            float,
            mx::UnitConverter,
            convert,
            input,
            inputUnit,
            outputUnit
        );
    }

    mx::Vector2 convert(const mx::Vector2& input, const std::string& inputUnit, const std::string& outputUnit) const override
    {
        PYBIND11_OVERLOAD_PURE(
            mx::Vector2,
            mx::UnitConverter,
            convert,
            input,
            inputUnit,
            outputUnit
        );
    }

    mx::Vector3 convert(const mx::Vector3& input, const std::string& inputUnit, const std::string& outputUnit) const override
    {
        PYBIND11_OVERLOAD_PURE(
            mx::Vector3,
            mx::UnitConverter,
            convert,
            input,
            inputUnit,
            outputUnit
        );
    }

    mx::Vector4 convert(const mx::Vector4& input, const std::string& inputUnit, const std::string& outputUnit) const override
    {
        PYBIND11_OVERLOAD_PURE(
            mx::Vector4,
            mx::UnitConverter,
            convert,
            input,
            inputUnit,
            outputUnit
        );
    }
};

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

void bindPyUnitConverters(py::module& mod)
{
    py::class_<mx::UnitConverter, PyUnitConverter, mx::UnitConverterPtr>(mod, "UnitConverter")
        .def("convert", (float       (mx::UnitConverter::*)(float, const std::string&, const std::string&)const) &mx::UnitConverter::convert)
        .def("convert", (mx::Vector2 (mx::UnitConverter::*)(const mx::Vector2&, const std::string&, const std::string&)const) &mx::UnitConverter::convert)
        .def("convert", (mx::Vector3 (mx::UnitConverter::*)(const mx::Vector3&, const std::string&, const std::string&)const) &mx::UnitConverter::convert)
        .def("convert", (mx::Vector4 (mx::UnitConverter::*)(const mx::Vector4&, const std::string&, const std::string&)const) &mx::UnitConverter::convert)
        .def("getUnitAsInteger", &mx::UnitConverter::getUnitAsInteger)
        .def("getUnitFromInteger", &mx::UnitConverter::getUnitFromInteger);

    py::class_<mx::LinearUnitConverter, mx::UnitConverter, mx::LinearUnitConverterPtr>(mod, "LinearUnitConverter")
        .def_static("create", &mx::LinearUnitConverter::create)
        .def("getUnitScale", &mx::LinearUnitConverter::getUnitScale)
        .def("convert", (float       (mx::LinearUnitConverter::*)(float, const std::string&, const std::string&)const) &mx::LinearUnitConverter::convert)
        .def("convert", (mx::Vector2 (mx::LinearUnitConverter::*)(const mx::Vector2&, const std::string&, const std::string&)const) &mx::LinearUnitConverter::convert)
        .def("convert", (mx::Vector3 (mx::LinearUnitConverter::*)(const mx::Vector3&, const std::string&, const std::string&)const) &mx::LinearUnitConverter::convert)
        .def("convert", (mx::Vector4 (mx::LinearUnitConverter::*)(const mx::Vector4&, const std::string&, const std::string&)const) &mx::LinearUnitConverter::convert)
        .def("getUnitAsInteger", &mx::LinearUnitConverter::getUnitAsInteger)
        .def("getUnitFromInteger", &mx::LinearUnitConverter::getUnitFromInteger);

    py::class_<mx::UnitConverterRegistry, mx::UnitConverterRegistryPtr>(mod, "UnitConverterRegistry")
        .def_static("create", &mx::UnitConverterRegistry::create)
        .def("addUnitConverter", &mx::UnitConverterRegistry::addUnitConverter)
        .def("removeUnitConverter", &mx::UnitConverterRegistry::removeUnitConverter)
        .def("getUnitConverter", &mx::UnitConverterRegistry::getUnitConverter)
        .def("clearUnitConverters", &mx::UnitConverterRegistry::clearUnitConverters);
}
