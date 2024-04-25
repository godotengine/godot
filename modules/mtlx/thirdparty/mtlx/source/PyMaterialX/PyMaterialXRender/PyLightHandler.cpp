//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Document.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXRender/LightHandler.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyLightHandler(py::module& mod)
{
    py::class_<mx::LightHandler, mx::LightHandlerPtr>(mod, "LightHandler")
        .def_static("create", &mx::LightHandler::create)
        .def(py::init<>())
        .def("setLightTransform", &mx::LightHandler::setLightTransform)
        .def("getLightTransform", &mx::LightHandler::getLightTransform)
        .def("setDirectLighting", &mx::LightHandler::setDirectLighting)
        .def("getDirectLighting", &mx::LightHandler::getDirectLighting)
        .def("setIndirectLighting", &mx::LightHandler::setIndirectLighting)
        .def("getIndirectLighting", &mx::LightHandler::getIndirectLighting)
        .def("setEnvRadianceMap", &mx::LightHandler::setEnvRadianceMap)
        .def("getEnvRadianceMap", &mx::LightHandler::getEnvRadianceMap)
        .def("setEnvIrradianceMap", &mx::LightHandler::setEnvIrradianceMap)
        .def("getEnvIrradianceMap", &mx::LightHandler::getEnvIrradianceMap)
        .def("setAlbedoTable", &mx::LightHandler::setAlbedoTable)
        .def("getAlbedoTable", &mx::LightHandler::getAlbedoTable)
        .def("setEnvSampleCount", &mx::LightHandler::setEnvSampleCount)
        .def("getEnvSampleCount", &mx::LightHandler::getEnvSampleCount)
        .def("setRefractionTwoSided", &mx::LightHandler::setRefractionTwoSided)
        .def("getRefractionTwoSided", &mx::LightHandler::getRefractionTwoSided)        
        .def("addLightSource", &mx::LightHandler::addLightSource)
        .def("setLightSources", &mx::LightHandler::setLightSources)
        .def("getLightSources", &mx::LightHandler::getLightSources)
        .def("getFirstLightOfCategory", &mx::LightHandler::getFirstLightOfCategory)
        .def("getLightIdMap", &mx::LightHandler::getLightIdMap)
        .def("computeLightIdMap", &mx::LightHandler::computeLightIdMap)
        .def("findLights", &mx::LightHandler::findLights)
        .def("registerLights", &mx::LightHandler::registerLights);
}
