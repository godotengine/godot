//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderMsl/TextureBaker.h>
#include <MaterialXCore/Material.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyTextureBaker(py::module& mod)
{
    py::class_<mx::TextureBakerMsl, mx::MslRenderer, mx::TextureBakerPtr>(mod, "TextureBaker")
        .def_static("create", &mx::TextureBakerMsl::create)
        .def("setExtension", &mx::TextureBakerMsl::setExtension)
        .def("getExtension", &mx::TextureBakerMsl::getExtension)
        .def("setColorSpace", &mx::TextureBakerMsl::setColorSpace)
        .def("getColorSpace", &mx::TextureBakerMsl::getColorSpace)
        .def("setDistanceUnit", &mx::TextureBakerMsl::setDistanceUnit)
        .def("getDistanceUnit", &mx::TextureBakerMsl::getDistanceUnit)
        .def("setAverageImages", &mx::TextureBakerMsl::setAverageImages)
        .def("getAverageImages", &mx::TextureBakerMsl::getAverageImages)
        .def("setOptimizeConstants", &mx::TextureBakerMsl::setOptimizeConstants)
        .def("getOptimizeConstants", &mx::TextureBakerMsl::getOptimizeConstants)
        .def("setOutputImagePath", &mx::TextureBakerMsl::setOutputImagePath)
        .def("getOutputImagePath", &mx::TextureBakerMsl::getOutputImagePath)
        .def("setBakedGraphName", &mx::TextureBakerMsl::setBakedGraphName)
        .def("getBakedGraphName", &mx::TextureBakerMsl::getBakedGraphName)
        .def("setBakedGeomInfoName", &mx::TextureBakerMsl::setBakedGeomInfoName)
        .def("getBakedGeomInfoName", &mx::TextureBakerMsl::getBakedGeomInfoName)
        .def("setTextureFilenameTemplate", &mx::TextureBakerMsl::setTextureFilenameTemplate)
        .def("getTextureFilenameTemplate", &mx::TextureBakerMsl::getTextureFilenameTemplate)
        .def("setFilenameTemplateVarOverride", &mx::TextureBakerMsl::setFilenameTemplateVarOverride)
        .def("setHashImageNames", &mx::TextureBakerMsl::setHashImageNames)
        .def("getHashImageNames", &mx::TextureBakerMsl::getHashImageNames)
        .def("setTextureSpaceMin", &mx::TextureBakerMsl::setTextureSpaceMin)
        .def("getTextureSpaceMin", &mx::TextureBakerMsl::getTextureSpaceMin)
        .def("setTextureSpaceMax", &mx::TextureBakerMsl::setTextureSpaceMax)
        .def("getTextureSpaceMax", &mx::TextureBakerMsl::getTextureSpaceMax)
        .def("setupUnitSystem", &mx::TextureBakerMsl::setupUnitSystem)
        .def("bakeMaterialToDoc", &mx::TextureBakerMsl::bakeMaterialToDoc)
        .def("bakeAllMaterials", &mx::TextureBakerMsl::bakeAllMaterials)
        .def("writeDocumentPerMaterial", &mx::TextureBakerMsl::writeDocumentPerMaterial);
}
