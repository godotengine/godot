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

#include "file_includer.h"

#include <mutex>
#include <utility>

#include "libshaderc_util/io.h"

namespace glslc {

shaderc_include_result* MakeErrorIncludeResult(const char* message) {
  return new shaderc_include_result{"", 0, message, strlen(message)};
}

FileIncluder::~FileIncluder() = default;

shaderc_include_result* FileIncluder::GetInclude(
    const char* requested_source, shaderc_include_type include_type,
    const char* requesting_source, size_t) {

  const std::string full_path =
      (include_type == shaderc_include_type_relative)
          ? file_finder_.FindRelativeReadableFilepath(requesting_source,
                                                      requested_source)
          : file_finder_.FindReadableFilepath(requested_source);

  if (full_path.empty())
    return MakeErrorIncludeResult("Cannot find or open include file.");

  // In principle, several threads could be resolving includes at the same
  // time.  Protect the included_files.

  // Read the file and save its full path and contents into stable addresses.
  FileInfo* new_file_info = new FileInfo{full_path, {}};
  if (!shaderc_util::ReadFile(full_path, &(new_file_info->contents))) {
    return MakeErrorIncludeResult("Cannot read file");
  }

  included_files_.insert(full_path);

  return new shaderc_include_result{
      new_file_info->full_path.data(), new_file_info->full_path.length(),
      new_file_info->contents.data(), new_file_info->contents.size(),
      new_file_info};
}

void FileIncluder::ReleaseInclude(shaderc_include_result* include_result) {
  FileInfo* info = static_cast<FileInfo*>(include_result->user_data);
  delete info;
  delete include_result;
}

}  // namespace glslc
