// Copyright 2015 The Shaderc Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GLSLC_FILE_H_
#define GLSLC_FILE_H_

#include "libshaderc_util/string_piece.h"

namespace glslc {

// Given a file name, returns its extension. If no extension exists,
// returns an empty string_piece.
shaderc_util::string_piece GetFileExtension(
    const shaderc_util::string_piece& filename);

// Returns true if the given file name ends with a known shader file extension.
inline bool IsStageFile(const shaderc_util::string_piece& filename) {
  const shaderc_util::string_piece extension =
      glslc::GetFileExtension(filename);
  return extension == "vert" || extension == "frag" || extension == "tesc" ||
         extension == "tese" || extension == "geom" || extension == "comp";
}

// Returns the file extension if is either "glsl" or "hlsl", or an empty
// string otherwise.
inline std::string GetGlslOrHlslExtension(
    const shaderc_util::string_piece& filename) {
  auto extension = glslc::GetFileExtension(filename);
  if ((extension == "glsl") || (extension == "hlsl")) return extension.str();
  return "";
}

}  // namespace glslc

#endif  // GLSLC_FILE_H_
