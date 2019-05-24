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

#ifndef GLSLC_FILE_INCLUDER_H_
#define GLSLC_FILE_INCLUDER_H_

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unordered_set>

#include "libshaderc_util/file_finder.h"
#include "shaderc/shaderc.hpp"

namespace glslc {

// An includer for files implementing shaderc's includer interface. It responds
// to the file including query from the compiler with the full path and content
// of the file to be included. In the case that the file is not found or cannot
// be opened, the full path field of in the response will point to an empty
// string, and error message will be passed to the content field.
// This class provides the basic thread-safety guarantee.
class FileIncluder : public shaderc::CompileOptions::IncluderInterface {
 public:
  explicit FileIncluder(const shaderc_util::FileFinder* file_finder)
      : file_finder_(*file_finder) {}

  ~FileIncluder() override;

  // Resolves a requested source file of a given type from a requesting
  // source into a shaderc_include_result whose contents will remain valid
  // until it's released.
  shaderc_include_result* GetInclude(const char* requested_source,
                                     shaderc_include_type type,
                                     const char* requesting_source,
                                     size_t include_depth) override;
  // Releases an include result.
  void ReleaseInclude(shaderc_include_result* include_result) override;

  // Returns a reference to the member storing the set of included files.
  const std::unordered_set<std::string>& file_path_trace() const {
    return included_files_;
  };

 private:
  // Used by GetInclude() to get the full filepath.
  const shaderc_util::FileFinder& file_finder_;
  // The full path and content of a source file.
  struct FileInfo {
    const std::string full_path;
    std::vector<char> contents;
  };

  // The set of full paths of included files.
  std::unordered_set<std::string> included_files_;
};

}  // namespace glslc

#endif  // GLSLC_FILE_INCLUDER_H_
