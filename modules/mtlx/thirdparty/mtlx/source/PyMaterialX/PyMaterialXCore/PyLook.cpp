//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Look.h>

namespace py = pybind11;
namespace mx = MaterialX;

#define BIND_LOOK_FUNC_INSTANCE(NAME, T) \
.def("_setPropertyValue" #NAME, &mx::Look::setPropertyValue<T>, py::arg("name"), py::arg("value"), py::arg("type") = mx::EMPTY_STRING)

void bindPyLook(py::module& mod)
{
    py::class_<mx::Look, mx::LookPtr, mx::Element>(mod, "Look")
        .def("addMaterialAssign", &mx::Look::addMaterialAssign,
            py::arg("name") = mx::EMPTY_STRING, py::arg("material") = mx::EMPTY_STRING)
        .def("getMaterialAssign", &mx::Look::getMaterialAssign)
        .def("getMaterialAssigns", &mx::Look::getMaterialAssigns)
        .def("getActiveMaterialAssigns", &mx::Look::getActiveMaterialAssigns)
        .def("removeMaterialAssign", &mx::Look::removeMaterialAssign)
        .def("addPropertyAssign", &mx::Look::addPropertyAssign,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getPropertyAssign", &mx::Look::getPropertyAssign)
        .def("getPropertyAssigns", &mx::Look::getPropertyAssigns)
        .def("getActivePropertyAssigns", &mx::Look::getActivePropertyAssigns)
        .def("removePropertyAssign", &mx::Look::removePropertyAssign)
        .def("addPropertySetAssign", &mx::Look::addPropertySetAssign,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getPropertySetAssign", &mx::Look::getPropertySetAssign)
        .def("getPropertySetAssigns", &mx::Look::getPropertySetAssigns)
        .def("getActivePropertySetAssigns", &mx::Look::getActivePropertySetAssigns)
        .def("removePropertySetAssign", &mx::Look::removePropertySetAssign)
        .def("addVariantAssign", &mx::Look::addVariantAssign,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getVariantAssign", &mx::Look::getVariantAssign)
        .def("getVariantAssigns", &mx::Look::getVariantAssigns)
        .def("getActiveVariantAssigns", &mx::Look::getActiveVariantAssigns)
        .def("removeVariantAssign", &mx::Look::removeVariantAssign)
        .def("addVisibility", &mx::Look::addVisibility,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getVisibility", &mx::Look::getVisibility)
        .def("getVisibilities", &mx::Look::getVisibilities)
        .def("getActiveVisibilities", &mx::Look::getActiveVisibilities)
        .def("removeVisibility", &mx::Look::removeVisibility)
        .def_readonly_static("CATEGORY", &mx::Look::CATEGORY);

    py::class_<mx::LookGroup, mx::LookGroupPtr, mx::Element>(mod, "LookGroup")
        .def("getLooks", &mx::LookGroup::getLooks)
        .def("setLooks", &mx::LookGroup::setLooks)
        .def("getActiveLook", &mx::LookGroup::getActiveLook)
        .def("setActiveLook", &mx::LookGroup::setActiveLook)
        .def_readonly_static("CATEGORY", &mx::LookGroup::CATEGORY)
        .def_readonly_static("LOOKS_ATTRIBUTE", &mx::LookGroup::LOOKS_ATTRIBUTE)
        .def_readonly_static("ACTIVE_ATTRIBUTE", &mx::LookGroup::ACTIVE_ATTRIBUTE);

    py::class_<mx::MaterialAssign, mx::MaterialAssignPtr, mx::GeomElement>(mod, "MaterialAssign")
        .def("setMaterial", &mx::MaterialAssign::setMaterial)
        .def("hasMaterial", &mx::MaterialAssign::hasMaterial)
        .def("getMaterial", &mx::MaterialAssign::getMaterial)
        .def("getMaterialOutputs", &mx::MaterialAssign::getMaterialOutputs)        
        .def("setExclusive", &mx::MaterialAssign::setExclusive)
        .def("getExclusive", &mx::MaterialAssign::getExclusive)
        .def("getReferencedMaterial", &mx::MaterialAssign::getReferencedMaterial)
        .def_readonly_static("CATEGORY", &mx::MaterialAssign::CATEGORY);

    py::class_<mx::Visibility, mx::VisibilityPtr, mx::GeomElement>(mod, "Visibility")
        .def("setViewerGeom", &mx::Visibility::setViewerGeom)
        .def("hasViewerGeom", &mx::Visibility::hasViewerGeom)
        .def("getViewerGeom", &mx::Visibility::getViewerGeom)
        .def("setViewerCollection", &mx::Visibility::setViewerCollection)
        .def("hasViewerCollection", &mx::Visibility::hasViewerCollection)
        .def("getViewerCollection", &mx::Visibility::getViewerCollection)
        .def("setVisibilityType", &mx::Visibility::setVisibilityType)
        .def("hasVisibilityType", &mx::Visibility::hasVisibilityType)
        .def("getVisibilityType", &mx::Visibility::getVisibilityType)
        .def("setVisible", &mx::Visibility::setVisible)
        .def("getVisible", &mx::Visibility::getVisible)
        .def_readonly_static("CATEGORY", &mx::Visibility::CATEGORY);

    mod.def("getGeometryBindings", &mx::getGeometryBindings,
        py::arg("materialNode") , py::arg("geom") = mx::UNIVERSAL_GEOM_NAME);
}
