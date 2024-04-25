//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/ShaderStage.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyShaderStage(py::module& mod)
{
    mod.attr("PIXEL_STAGE") = &mx::Stage::PIXEL;

    py::class_<mx::ShaderPortPredicate>(mod, "ShaderPortPredicate");

    py::class_<mx::VariableBlock, mx::VariableBlockPtr>(mod, "VariableBlock")
        .def(py::init<const std::string&, const std::string&>())
        .def("getName", &mx::VariableBlock::getName)
        .def("getInstance", &mx::VariableBlock::getInstance)
        .def("empty", &mx::VariableBlock::empty)
        .def("size", &mx::VariableBlock::size)
        .def("find", static_cast<mx::ShaderPort* (mx::VariableBlock::*)(const std::string&)>(&mx::VariableBlock::find))
        .def("find", (mx::ShaderPort* (mx::VariableBlock::*)(const mx::ShaderPortPredicate& )) &mx::VariableBlock::find)
        .def("__len__", &mx::VariableBlock::size)
        .def("__getitem__", [](const mx::VariableBlock &vb, size_t i)
        {
            if (i >= vb.size()) throw py::index_error();
            return vb[i];
        }, py::return_value_policy::reference_internal);

    py::class_<mx::ShaderStage>(mod, "ShaderStage")
        .def(py::init<const std::string&, mx::ConstSyntaxPtr>())
        .def("getName", &mx::ShaderStage::getName)
        .def("getFunctionName", &mx::ShaderStage::getFunctionName)
        .def("getSourceCode", &mx::ShaderStage::getSourceCode)
        .def("getUniformBlock", static_cast<mx::VariableBlock& (mx::ShaderStage::*)(const std::string&)>(&mx::ShaderStage::getUniformBlock))
        .def("getInputBlock", static_cast<mx::VariableBlock& (mx::ShaderStage::*)(const std::string&)>(&mx::ShaderStage::getInputBlock))
        .def("getOutputBlock", static_cast<mx::VariableBlock& (mx::ShaderStage::*)(const std::string&)>(&mx::ShaderStage::getOutputBlock))
        .def("getConstantBlock", static_cast<mx::VariableBlock& (mx::ShaderStage::*)()>(&mx::ShaderStage::getConstantBlock))
        .def("getUniformBlocks", &mx::ShaderStage::getUniformBlocks)
        .def("getInputBlocks", &mx::ShaderStage::getInputBlocks)
        .def("getIncludes", &mx::ShaderStage::getIncludes)
        .def("getSourceDependencies", &mx::ShaderStage::getSourceDependencies)
        .def("getOutputBlocks", &mx::ShaderStage::getOutputBlocks);
}
