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

#include "dependency_info.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "file.h"
#include "libshaderc_util/io.h"

namespace glslc {

DependencyInfoDumpingHandler::DependencyInfoDumpingHandler() : mode_(not_set) {}

bool DependencyInfoDumpingHandler::DumpDependencyInfo(
    std::string compilation_output_file_name, std::string source_file_name,
    std::string* compilation_output_ptr,
    const std::unordered_set<std::string>& dependent_files) {
  std::string dep_target_label = GetTarget(compilation_output_file_name);
  std::string dep_file_name =
      GetDependencyFileName(compilation_output_file_name);

  // Dump everything to a string stream first, then dump its content to either
  // a file or compilation output string, depends on current dumping mode.
  std::stringstream dep_string_stream;
  // dump target label and the source_file_name.
  dep_string_stream << dep_target_label << ": " << source_file_name;
  // dump the dependent file names.
  for (auto& dependent_file_name : dependent_files) {
    dep_string_stream << " " << dependent_file_name;
  }
  dep_string_stream << std::endl;

  if (mode_ == dump_as_compilation_output) {
    compilation_output_ptr->assign(dep_string_stream.str());
  } else if (mode_ == dump_as_extra_file) {
    std::ofstream potential_file_stream_for_dep_info_dump;
    std::ostream* dep_file_stream = shaderc_util::GetOutputStream(
        dep_file_name, &potential_file_stream_for_dep_info_dump, &std::cerr);
    *dep_file_stream << dep_string_stream.str();
    if (dep_file_stream->fail()) {
      std::cerr << "glslc: error: error writing dependent_files info to output "
                   "file: '"
                << dep_file_name << "'" << std::endl;
      return false;
    }
  } else {
    // mode_ should not be 'not_set', we should never be here.
    return false;
  }
  return true;
}

std::string DependencyInfoDumpingHandler::GetTarget(
    const std::string& compilation_output_file_name) {
  if (!user_specified_dep_target_label_.empty()) {
    return user_specified_dep_target_label_;
  }

  return compilation_output_file_name;
}

std::string DependencyInfoDumpingHandler::GetDependencyFileName(
    const std::string& compilation_output_file_name) {
  if (!user_specified_dep_file_name_.empty()) {
    return user_specified_dep_file_name_;
  }

  return compilation_output_file_name + ".d";
}

bool DependencyInfoDumpingHandler::IsValid(std::string* error_msg_ptr,
                                           size_t num_files) {
  if (DumpingModeNotSet()) {
    *error_msg_ptr =
        "to generate dependencies you must specify either -M (-MM) or -MD";
    return false;
  }

  if (!user_specified_dep_file_name_.empty() ||
      !user_specified_dep_target_label_.empty()) {
    if (num_files > 1) {
      *error_msg_ptr =
          "to specify dependency info file name or dependency info target, "
          "only one input file is allowed.";
      return false;
    }
  }
  return true;
}
}
