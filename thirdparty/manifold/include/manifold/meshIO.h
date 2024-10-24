// Copyright 2021 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <string>

#include "manifold/manifold.h"

namespace manifold {

/** @defgroup MeshIO
 *  @brief 3D model file I/O based on Assimp
 * @{
 */

/**
 * PBR material properties for GLB/glTF files.
 */
struct Material {
  /// Roughness value between 0 (shiny) and 1 (matte).
  double roughness = 0.2;
  /// Metalness value, generally either 0 (dielectric) or 1 (metal).
  double metalness = 1;
  /// Color (RGBA) multiplier to apply to the whole mesh (each value between 0
  /// and 1).
  vec4 color = vec4(1.0);
  /// Optional: If non-empty, must match Mesh.vertPos. Provides an RGBA color
  /// for each vertex, linearly interpolated across triangles. Colors are
  /// linear, not sRGB. Only used with Mesh export, not MeshGL.
  std::vector<vec4> vertColor;
  /// For MeshGL export, gives the property indicies where the normal channels
  /// can be found. Must be >= 3, since the first three are position.
  ivec3 normalChannels = ivec3(-1);
  /// For MeshGL export, gives the property indicies where the color channels
  /// can be found. Any index < 0 will output all 1.0 for that channel.
  ivec4 colorChannels = ivec4(-1);
};

/**
 * These options only currently affect .glb and .gltf files.
 */
struct ExportOptions {
  /// When false, vertex normals are exported, causing the mesh to appear smooth
  /// through normal interpolation.
  bool faceted = true;
  /// PBR material properties.
  Material mat = {};
};

MeshGL ImportMesh(const std::string& filename, bool forceCleanup = false);

void ExportMesh(const std::string& filename, const MeshGL& mesh,
                const ExportOptions& options);
/** @} */
}  // namespace manifold
