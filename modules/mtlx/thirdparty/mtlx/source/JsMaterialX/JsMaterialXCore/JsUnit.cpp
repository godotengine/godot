//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Unit.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

// The following 3 functions should not be necessary, but due to unexpected behaviour of emscripten,
// it reports signature mismatches if the functions are bound directly

mx::LinearUnitConverterPtr LinearUnitConverterCreate(mx::UnitTypeDefPtr unitTypeDef) {
    return mx::LinearUnitConverter::create(unitTypeDef);
}

mx::UnitConverterPtr UnitConverterRegistryGetUnitConverter(mx::UnitConverterRegistry& registry,mx::UnitTypeDefPtr def){
    return registry.getUnitConverter(def);
}

mx::UnitConverterRegistryPtr UnitConverterRegistrycreate() {
    return mx::UnitConverterRegistry::create();
}

EMSCRIPTEN_BINDINGS(unit)
{

    ems::class_<mx::UnitConverter>("UnitConverter")
        .smart_ptr<std::shared_ptr<mx::UnitConverter>>("UnitConverter")
        .smart_ptr<std::shared_ptr<const mx::UnitConverter>>("UnitConverter")
        .function("convertFloat", ems::select_overload<float(float, const std::string&, const std::string&) const>(&mx::UnitConverter::convert))
        .function("getUnitAsInteger", &mx::UnitConverter::getUnitAsInteger)
        .function("getUnitFromInteger", &mx::UnitConverter::getUnitFromInteger)
        .function("convertVector2", ems::select_overload<mx::Vector2(const mx::Vector2&, const std::string&, const std::string&) const>(&mx::UnitConverter::convert))
        .function("convertVector3", ems::select_overload<mx::Vector3(const mx::Vector3&, const std::string&, const std::string&) const>(&mx::UnitConverter::convert))
        .function("convertVector4", ems::select_overload<mx::Vector4(const mx::Vector4&, const std::string&, const std::string&) const>(&mx::UnitConverter::convert))
        .function("write", &mx::UnitConverter::write, ems::pure_virtual());

    ems::class_<mx::LinearUnitConverter, ems::base<mx::UnitConverter>>("LinearUnitConverter")
        .smart_ptr<std::shared_ptr<mx::LinearUnitConverter>>("LinearUnitConverter")
        .smart_ptr<std::shared_ptr<const mx::LinearUnitConverter>>("LinearUnitConverter")
        .class_function("create", &LinearUnitConverterCreate)
        .function("getUnitType", &mx::LinearUnitConverter::getUnitType)
        .function("write", &mx::LinearUnitConverter::write)
        .function("getUnitScale", &mx::LinearUnitConverter::getUnitScale)
        .function("conversionRatio", &mx::LinearUnitConverter::conversionRatio)
        .function("convertFloat", ems::select_overload<float(float, const std::string&, const std::string&) const>(&mx::LinearUnitConverter::convert))
        .function("convertVector2", ems::select_overload<mx::Vector2(const mx::Vector2&, const std::string&, const std::string&) const>(&mx::LinearUnitConverter::convert))
        .function("convertVector3", ems::select_overload<mx::Vector3(const mx::Vector3&, const std::string&, const std::string&) const>(&mx::LinearUnitConverter::convert))
        .function("convertVector4", ems::select_overload<mx::Vector4(const mx::Vector4&, const std::string&, const std::string&) const>(&mx::LinearUnitConverter::convert))
        .function("getUnitAsInteger", &mx::LinearUnitConverter::getUnitAsInteger)
        .function("getUnitFromInteger", &mx::LinearUnitConverter::getUnitFromInteger);

    ems::class_<mx::UnitConverterRegistry>("UnitConverterRegistry")
        .smart_ptr<std::shared_ptr<mx::UnitConverterRegistry>>("UnitConverterRegistry")
        .smart_ptr<std::shared_ptr<const mx::UnitConverterRegistry>>("UnitConverterRegistry")
        .class_function("create", &UnitConverterRegistrycreate)
        .function("addUnitConverter", &mx::UnitConverterRegistry::addUnitConverter)
        .function("removeUnitConverter", &mx::UnitConverterRegistry::removeUnitConverter)
        .function("getUnitConverter", &UnitConverterRegistryGetUnitConverter)
        .function("clearUnitConverters", &mx::UnitConverterRegistry::clearUnitConverters)
        .function("getUnitAsInteger", &mx::UnitConverterRegistry::getUnitAsInteger)
        .function("write", &mx::UnitConverterRegistry::write);
}
