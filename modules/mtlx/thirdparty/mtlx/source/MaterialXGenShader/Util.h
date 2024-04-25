//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SHADERGEN_UTIL_H
#define MATERIALX_SHADERGEN_UTIL_H

/// @file
/// Shader generation utility methods

#include <MaterialXGenShader/Export.h>

#include <MaterialXCore/Document.h>

#include <unordered_set>

MATERIALX_NAMESPACE_BEGIN

class ShaderGenerator;

/// Returns true if the given element is a surface shader with the potential
/// of being transparent. This can be used by HW shader generators to determine
/// if a shader will require transparency handling.
///
/// Note: This function will check some common cases for how a surface
/// shader can be transparent. It is not covering all possible cases for
/// how transparency can be done and target applications might need to do
/// additional checks to track transparency correctly. For example, custom
/// surface shader nodes implemented in source code will not be tracked by this
/// function and transprency for such nodes must be tracked separately by the
/// target application.
///
MX_GENSHADER_API bool isTransparentSurface(ElementPtr element, const string& target = EMPTY_STRING);

/// Maps a value to a four channel color if it is of the appropriate type.
/// Supported types include float, Vector2, Vector3, Vector4,
/// and Color4. If not mapping is possible the color value is
/// set to opaque black.
MX_GENSHADER_API void mapValueToColor(ConstValuePtr value, Color4& color);

/// Return whether a nodedef requires an implementation
MX_GENSHADER_API bool requiresImplementation(ConstNodeDefPtr nodeDef);

/// Determine if a given element requires shading / lighting for rendering
MX_GENSHADER_API bool elementRequiresShading(ConstTypedElementPtr element);

/// Find all renderable material nodes in the given document.
/// @param doc Document to examine
/// @return A vector of renderable material nodes.
MX_GENSHADER_API vector<TypedElementPtr> findRenderableMaterialNodes(ConstDocumentPtr doc);

/// Find all renderable elements in the given document, including material nodes if present,
/// or graph outputs of renderable types if no material nodes are found.
/// @param doc Document to examine
/// @return A vector of renderable elements
MX_GENSHADER_API vector<TypedElementPtr> findRenderableElements(ConstDocumentPtr doc);

/// Given a node input, return the corresponding input within its matching nodedef.
/// The optional target string can be used to guide the selection of nodedef declarations.
MX_GENSHADER_API InputPtr getNodeDefInput(InputPtr nodeInput, const string& target);

/// Perform token substitutions on the given source string, using the given substituation map.
/// Tokens are required to start with '$' and can only consist of alphanumeric characters.
/// The full token name, including '$' and all following alphanumeric character, will be replaced
/// by the corresponding string in the substitution map, if the token exists in the map.
MX_GENSHADER_API void tokenSubstitution(const StringMap& substitutions, string& source);

/// Compute the UDIM coordinates for a set of UDIM identifiers
/// @return List of UDIM coordinates
MX_GENSHADER_API vector<Vector2> getUdimCoordinates(const StringVec& udimIdentifiers);

/// Get the UV scale and offset to transform uv coordinates from UDIM uv space to
/// 0..1 space.
MX_GENSHADER_API void getUdimScaleAndOffset(const vector<Vector2>& udimCoordinates, Vector2& scaleUV, Vector2& offsetUV);

/// Determine whether the given output is directly connected to a node that
/// generates world-space coordinates (e.g. the "normalmap" node).
/// @param output Output to check
/// @return Return the node if found.
MX_GENSHADER_API NodePtr connectsToWorldSpaceNode(OutputPtr output);

/// Returns true if there is are any value elements with a given set of attributes either on the
/// starting node or any graph upsstream of that node.
/// @param output Starting node
/// @param attributes Attributes to test for
MX_GENSHADER_API bool hasElementAttributes(OutputPtr output, const StringVec& attributes);

//
// These are deprecated wrappers for older versions of the function interfaces in this module.
// Clients using these interfaces should update them to the latest API.
//
MX_GENSHADER_API [[deprecated]] void findRenderableMaterialNodes(ConstDocumentPtr doc, vector<TypedElementPtr>& elements, bool, std::unordered_set<ElementPtr>&);
MX_GENSHADER_API [[deprecated]] void findRenderableElements(ConstDocumentPtr doc, vector<TypedElementPtr>& elements, bool includeReferencedGraphs = false);

MATERIALX_NAMESPACE_END

#endif
