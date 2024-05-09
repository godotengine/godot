//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/Helpers.h>
#include <MaterialXCore/Element.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>

#include <string>
#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

/// Returns the first renderable element from the given document. This element can be used to generate a shader.
mx::ElementPtr findRenderableElement(mx::DocumentPtr doc)
{
    mx::StringVec renderablePaths;
    std::vector<mx::TypedElementPtr> elems = mx::findRenderableElements(doc);

    for (mx::TypedElementPtr elem : elems)
    {
        mx::TypedElementPtr renderableElem = elem;
        mx::NodePtr node = elem->asA<mx::Node>();
        if (node && node->getType() == mx::MATERIAL_TYPE_STRING)
        {
            std::vector<mx::NodePtr> shaderNodes = getShaderNodes(node, mx::SURFACE_SHADER_TYPE_STRING);
            if (!shaderNodes.empty())
            {
                renderableElem = *shaderNodes.begin();
            }
        }

        const auto& renderablePath = renderableElem->getNamePath();
        mx::ElementPtr renderableElement = doc->getDescendant(renderablePath);
        mx::TypedElementPtr typedElem = renderableElement ? renderableElement->asA<mx::TypedElement>() : nullptr;
        if (typedElem)
        {
            return renderableElement;
        }
    }

    return nullptr;
}

EMSCRIPTEN_BINDINGS(Util)
{
    BIND_FUNC("isTransparentSurface", mx::isTransparentSurface, 1, 2, mx::ElementPtr, const std::string&);

    ems::function("findRenderableElement", &findRenderableElement);
}
