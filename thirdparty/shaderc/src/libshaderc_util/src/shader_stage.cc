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

#include "libshaderc_util/shader_stage.h"

namespace {

// Maps an identifier to a language.
struct LanguageMapping {
  const char* id;
  EShLanguage language;
};

}  // anonymous namespace

namespace shaderc_util {

EShLanguage MapStageNameToLanguage(const string_piece& stage_name) {
  const LanguageMapping string_to_stage[] = {
      {"vertex", EShLangVertex},
      {"fragment", EShLangFragment},
      {"tesscontrol", EShLangTessControl},
      {"tesseval", EShLangTessEvaluation},
      {"geometry", EShLangGeometry},
      {"compute", EShLangCompute},
#ifdef NV_EXTENSIONS
      {"raygen", EShLangRayGenNV},
      {"intersect", EShLangIntersectNV},
      {"anyhit", EShLangAnyHitNV},
      {"closest", EShLangClosestHitNV},
      {"miss", EShLangMissNV},
      {"callable", EShLangCallableNV},
      {"task", EShLangTaskNV},
      {"mesh", EShLangMeshNV},
#endif
  };

  for (const auto& entry : string_to_stage) {
    if (stage_name == entry.id) return entry.language;
  }
  return EShLangCount;
}

}  // namespace shaderc_util
