//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRender/ImageHandler.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyImageHandler(py::module& mod)
{
    py::class_<mx::ImageSamplingProperties>(mod, "ImageSamplingProperties")
        .def_readwrite("uaddressMode", &mx::ImageSamplingProperties::uaddressMode)
        .def_readwrite("vaddressMode", &mx::ImageSamplingProperties::vaddressMode)
        .def_readwrite("filterType", &mx::ImageSamplingProperties::filterType)
        .def_readwrite("defaultColor", &mx::ImageSamplingProperties::defaultColor);

    py::class_<mx::ImageLoader, mx::ImageLoaderPtr>(mod, "ImageLoader")
        .def_readonly_static("BMP_EXTENSION", &mx::ImageLoader::BMP_EXTENSION)
        .def_readonly_static("EXR_EXTENSION", &mx::ImageLoader::EXR_EXTENSION)
        .def_readonly_static("GIF_EXTENSION", &mx::ImageLoader::GIF_EXTENSION)
        .def_readonly_static("HDR_EXTENSION", &mx::ImageLoader::HDR_EXTENSION)
        .def_readonly_static("JPG_EXTENSION", &mx::ImageLoader::JPG_EXTENSION)
        .def_readonly_static("JPEG_EXTENSION", &mx::ImageLoader::JPEG_EXTENSION)
        .def_readonly_static("PIC_EXTENSION", &mx::ImageLoader::PIC_EXTENSION)
        .def_readonly_static("PNG_EXTENSION", &mx::ImageLoader::PNG_EXTENSION)
        .def_readonly_static("PSD_EXTENSION", &mx::ImageLoader::PSD_EXTENSION)
        .def_readonly_static("TGA_EXTENSION", &mx::ImageLoader::TGA_EXTENSION)
        .def_readonly_static("TIF_EXTENSION", &mx::ImageLoader::TIF_EXTENSION)
        .def_readonly_static("TIFF_EXTENSION", &mx::ImageLoader::TIFF_EXTENSION)
        .def_readonly_static("TXT_EXTENSION", &mx::ImageLoader::TXT_EXTENSION)
        .def("supportedExtensions", &mx::ImageLoader::supportedExtensions)
        .def("saveImage", &mx::ImageLoader::saveImage)
        .def("loadImage", &mx::ImageLoader::loadImage);

    py::class_<mx::ImageHandler, mx::ImageHandlerPtr>(mod, "ImageHandler")
        .def_static("create", &mx::ImageHandler::create)
        .def("addLoader", &mx::ImageHandler::addLoader)
        .def("saveImage", &mx::ImageHandler::saveImage,
            py::arg("filePath"), py::arg("image"), py::arg("verticalFlip") = false)
        .def("acquireImage", &mx::ImageHandler::acquireImage,
            py::arg("filePath"), py::arg("defaultColor") = mx::Color4(0.0f))
        .def("bindImage", &mx::ImageHandler::bindImage)
        .def("unbindImage", &mx::ImageHandler::unbindImage)
        .def("unbindImages", &mx::ImageHandler::unbindImages)
        .def("setSearchPath", &mx::ImageHandler::setSearchPath)
        .def("getSearchPath", &mx::ImageHandler::getSearchPath)
        .def("setFilenameResolver", &mx::ImageHandler::setFilenameResolver)
        .def("getFilenameResolver", &mx::ImageHandler::getFilenameResolver)
        .def("createRenderResources", &mx::ImageHandler::createRenderResources)
        .def("releaseRenderResources", &mx::ImageHandler::releaseRenderResources,
            py::arg("image") = nullptr)
        .def("clearImageCache", &mx::ImageHandler::clearImageCache)
        .def("getZeroImage", &mx::ImageHandler::getZeroImage)
        .def("getReferencedImages", &mx::ImageHandler::getReferencedImages);
}
