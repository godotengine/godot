//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderGlsl/TextureBaker.h>
#include <MaterialXCore/Material.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyTextureBaker(py::module& mod)
{
    py::class_<mx::TextureBakerGlsl, mx::GlslRenderer, mx::TextureBakerPtr>(mod, "TextureBaker")
        .def_static("create", &mx::TextureBakerGlsl::create)
        .def("setExtension", &mx::TextureBakerGlsl::setExtension)
        .def("getExtension", &mx::TextureBakerGlsl::getExtension)
        .def("setColorSpace", &mx::TextureBakerGlsl::setColorSpace)
        .def("getColorSpace", &mx::TextureBakerGlsl::getColorSpace)
        .def("setDistanceUnit", &mx::TextureBakerGlsl::setDistanceUnit)
        .def("getDistanceUnit", &mx::TextureBakerGlsl::getDistanceUnit)
        .def("setAverageImages", &mx::TextureBakerGlsl::setAverageImages)
        .def("getAverageImages", &mx::TextureBakerGlsl::getAverageImages)
        .def("setOptimizeConstants", &mx::TextureBakerGlsl::setOptimizeConstants)
        .def("getOptimizeConstants", &mx::TextureBakerGlsl::getOptimizeConstants)
        .def("setOutputImagePath", &mx::TextureBakerGlsl::setOutputImagePath)
        .def("getOutputImagePath", &mx::TextureBakerGlsl::getOutputImagePath)
        .def("setBakedGraphName", &mx::TextureBakerGlsl::setBakedGraphName)
        .def("getBakedGraphName", &mx::TextureBakerGlsl::getBakedGraphName)
        .def("setBakedGeomInfoName", &mx::TextureBakerGlsl::setBakedGeomInfoName)
        .def("getBakedGeomInfoName", &mx::TextureBakerGlsl::getBakedGeomInfoName)
        .def("setTextureFilenameTemplate", &mx::TextureBakerGlsl::setTextureFilenameTemplate)
        .def("getTextureFilenameTemplate", &mx::TextureBakerGlsl::getTextureFilenameTemplate)
        .def("setFilenameTemplateVarOverride", &mx::TextureBakerGlsl::setFilenameTemplateVarOverride)
        .def("setHashImageNames", &mx::TextureBakerGlsl::setHashImageNames)
        .def("getHashImageNames", &mx::TextureBakerGlsl::getHashImageNames)
        .def("setTextureSpaceMin", &mx::TextureBakerGlsl::setTextureSpaceMin)
        .def("getTextureSpaceMin", &mx::TextureBakerGlsl::getTextureSpaceMin)
        .def("setTextureSpaceMax", &mx::TextureBakerGlsl::setTextureSpaceMax)
        .def("getTextureSpaceMax", &mx::TextureBakerGlsl::getTextureSpaceMax)
        .def("setupUnitSystem", &mx::TextureBakerGlsl::setupUnitSystem)
        .def("bakeMaterialToDoc", &mx::TextureBakerGlsl::bakeMaterialToDoc)
        .def("bakeAllMaterials", &mx::TextureBakerGlsl::bakeAllMaterials)
        .def("writeDocumentPerMaterial", &mx::TextureBakerGlsl::writeDocumentPerMaterial);
}
