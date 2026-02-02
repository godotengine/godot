// SPDX-License-Identifier: MIT
// 
// Built-in MagicaVoxel .vox import plugIn.
// Import only. Writing voxel data as .vox is not supported(yet).
//
// example usage 
//
// def VoxAsset "volume" (
//   prepend references = @bunny.vox@
// )
// {
//    ...
// }

#pragma once

#include <string>

class GPrim;

namespace tinyusdz {
namespace usdVox {

//bool ReadVoxFromString(const std::string &str, GPrim *prim, std::string *err = nullptr);
bool ReadVoxFromFile(const std::string &filepath, GPrim *prim, std::string *err = nullptr);

} // namespace usdVox
} // namespace tinyusdz
