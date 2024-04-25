//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/Util.h>
#include <MaterialXGenShader/ShaderGenerator.h>

namespace py = pybind11;
namespace mx = MaterialX;

std::vector<mx::TypedElementPtr> findRenderableMaterialNodes(mx::ConstDocumentPtr doc)
{
    return mx::findRenderableMaterialNodes(doc);
}

std::vector<mx::TypedElementPtr> findRenderableElements(mx::ConstDocumentPtr doc, bool includeReferencedGraphs)
{
    (void) includeReferencedGraphs;
    return mx::findRenderableElements(doc);
}

void bindPyUtil(py::module& mod)
{
    mod.def("isTransparentSurface", &mx::isTransparentSurface);
    mod.def("mapValueToColor", &mx::mapValueToColor);
    mod.def("requiresImplementation", &mx::requiresImplementation);
    mod.def("elementRequiresShading", &mx::elementRequiresShading);
    mod.def("findRenderableMaterialNodes", &findRenderableMaterialNodes);
    mod.def("findRenderableElements", &findRenderableElements, py::arg("doc"), py::arg("includeReferencedGraphs") = false);
    mod.def("getNodeDefInput", &mx::getNodeDefInput);
    mod.def("tokenSubstitution", &mx::tokenSubstitution);
    mod.def("getUdimCoordinates", &mx::getUdimCoordinates);
    mod.def("getUdimScaleAndOffset", &mx::getUdimScaleAndOffset);
    mod.def("connectsToWorldSpaceNode", &mx::connectsToWorldSpaceNode);
    mod.def("hasElementAttributes", &mx::hasElementAttributes);
}
