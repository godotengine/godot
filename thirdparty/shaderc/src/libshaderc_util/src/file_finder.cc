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

#include "libshaderc_util/file_finder.h"
#include "libshaderc_util/string_piece.h"

#include <cassert>
#include <fstream>
#include <ios>

namespace {

// Returns "" if path is empty or ends in '/'.  Otherwise, returns "/".
std::string MaybeSlash(const shaderc_util::string_piece& path) {
  return (path.empty() || path.back() == '/') ? "" : "/";
}

}  // anonymous namespace

namespace shaderc_util {

std::string FileFinder::FindReadableFilepath(
    const std::string& filename) const {
  assert(!filename.empty());
  static const auto for_reading = std::ios_base::in;
  std::filebuf opener;
  for (const auto& prefix : search_path_) {
    const std::string prefixed_filename =
        prefix + MaybeSlash(prefix) + filename;
    if (opener.open(prefixed_filename, for_reading)) return prefixed_filename;
  }
  return "";
}

std::string FileFinder::FindRelativeReadableFilepath(
    const std::string& requesting_file, const std::string& filename) const {
  assert(!filename.empty());

  string_piece dir_name(requesting_file);

  size_t last_slash = requesting_file.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    dir_name = string_piece(requesting_file.c_str(),
                            requesting_file.c_str() + last_slash);
  }

  if (dir_name.size() == requesting_file.size()) {
    dir_name.clear();
  }

  static const auto for_reading = std::ios_base::in;
  std::filebuf opener;
  const std::string relative_filename =
      dir_name.str() + MaybeSlash(dir_name) + filename;
  if (opener.open(relative_filename, for_reading)) return relative_filename;

  return FindReadableFilepath(filename);
}

}  // namespace shaderc_util
