//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/Helpers.h>
#include "./StrContainerTypeRegistration.h"
#include <MaterialXFormat/XmlIo.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(xmlio)
{
    ems::constant("MTLX_EXTENSION", mx::MTLX_EXTENSION);
    ems::class_<mx::XmlReadOptions>("XmlReadOptions")
        .constructor<>()
        .property<bool>("readXIncludes",
            [](const mx::XmlReadOptions &self) {
                return self.readXIncludeFunction == nullptr;
            },
            [](mx::XmlReadOptions &self, bool useIncludes) {
                if (useIncludes) {
                    self.readXIncludeFunction = &mx::readFromXmlFile;
                } else {
                    self.readXIncludeFunction = nullptr;
                }
            })
        .property("readComments", &mx::XmlReadOptions::readComments)
        .property("upgradeVersion", &mx::XmlReadOptions::upgradeVersion)                
        .property("parentXIncludes", &mx::XmlReadOptions::parentXIncludes);

    ems::class_<mx::XmlWriteOptions>("XmlWriteOptions")
        .constructor<>()
        .property("writeXIncludeEnable", &mx::XmlWriteOptions::writeXIncludeEnable);

    BIND_FUNC_RAW_PTR("_readFromXmlFile", mx::readFromXmlFile, 2, 4, mx::DocumentPtr, mx::FilePath,
        mx::FileSearchPath, const mx::XmlReadOptions*);
    BIND_FUNC_RAW_PTR("writeToXmlFile", mx::writeToXmlFile, 2, 3, mx::DocumentPtr, const mx::FilePath&, const mx::XmlWriteOptions *);
    BIND_FUNC_RAW_PTR("writeToXmlString", mx::writeToXmlString, 1, 2, mx::DocumentPtr, const mx::XmlWriteOptions *);
    ems::function("prependXInclude", &mx::prependXInclude);
}
