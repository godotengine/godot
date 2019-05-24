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

#ifndef GLSLC_DEPENDENCY_INFO_H
#define GLSLC_DEPENDENCY_INFO_H

#include <unordered_set>
#include <string>
#include <string>

namespace glslc {

// An object to handle everything about dumping dependency info. Internally it
// has two valid dumping mode: 1) Dump to extra dependency info files, such as
// *.d files. This mode is used when we want to generate dependency info and
// also compile. 2) Overwrite the original compilation output and dump
// dependency info as compilation output. This mode is used when we do not want
// to compile the source code and want the dependency info only.
class DependencyInfoDumpingHandler {
 public:
  DependencyInfoDumpingHandler();

  // Sets the dependency target explicitly. It's the same as the argument to
  // -MT.
  void SetTarget(const std::string& target_label) {
    user_specified_dep_target_label_ = target_label;
  }

  // Sets the name of the file where dependency info will be written.
  void SetDependencyFileName(const std::string& dep_file_name) {
    user_specified_dep_file_name_ = dep_file_name;
  };

  // Dump depdendency info to a) an extra dependency info file, b) an string
  // which holds the compilation output. The choice depends on the dump
  // mode of the handler. Returns true if dumping is succeeded, false otherwise.
  //
  // The dependency file name and target are deduced based on 1) user
  // specified dependency file name and target name, 2) the output filename when
  // the compiler is in 'does not need linking' and 'not preprocessing-only'
  // mode. It is passed through compilation_output_file_name.
  //
  // When the handler is set to dump dependency info as extra dependency info
  // files, this method will open a file with the dependency file name and write
  // the dependency info to it. Error messages caused by writing to the file are
  // emitted to stderr.
  //
  // When the handler is set to dump dependency info as compilation output, the
  // compilation output string, which is passed through compilation_output_ptr,
  // will be cleared and this method will write dependency info to it. Then the
  // dependency info should be emitted as normal compilation output.
  //
  // If the dump mode is not set when this method is called, return false.
  bool DumpDependencyInfo(std::string compilation_output_file_name,
                          std::string source_file_name,
                          std::string* compilation_output_ptr,
                          const std::unordered_set<std::string>& dependent_files);

  // Sets to always dump dependency info as an extra file, instead of the normal
  // compilation output. This means the output name specified by -o options
  // won't be used for the dependency info file.
  void SetDumpToExtraDependencyInfoFiles() { mode_ = dump_as_extra_file; };

  // Sets to dump dependency info as normal compilation output. The dependency
  // info will be either saved in a file with -o option specified file, or, if
  // no output file name specified, to stdout.
  void SetDumpAsNormalCompilationOutput() {
    mode_ = dump_as_compilation_output;
  }

  // Returns true if the handler's dumping mode is set to dump dependency info
  // as extra dependency info files.
  bool DumpingToExtraDependencyInfoFiles() {
    return mode_ == dump_as_extra_file;
  }

  // Returns true if the handler's dumping mode is set to dump dependency info
  // as normal compilation output.
  bool DumpingAsCompilationOutput() {
    return mode_ == dump_as_compilation_output;
  }

  // Returns true if the handler's dumping mode is not set.
  bool DumpingModeNotSet() { return mode_ == not_set; }

  // Returns true if the handler is at valid state for dumping dependency info.
  bool IsValid(std::string* error_msg_ptr, size_t num_files);

 private:
  typedef enum {
    // not_set mode tells that the dumping mode is not set yet, so the handler
    // is not ready for dumping dependency info. Calling DumpDependencyInfo when
    // the handler is in this mode will cause failure.
    not_set = 0,
    // Dumping dependency info as normal compilation output mode.  In this mode,
    // the dependency info will be dumped as compilation output by overwriting
    // the string which holds the compilation output.
    dump_as_compilation_output,
    // Dumping dependency info as extra dependency info files mode. In this
    // mode, dependency info will be dumped to a user specified dependency info
    // file or a *.d file. Compilation output will still be generated along with
    // the dependency info.
    dump_as_extra_file,
  } dump_mode;

  // Returns the target file label to be used in depdendency info file. If -MT
  // defined a label, use that string as the label. Otherwise returns the
  // compilation output filename deduced in 'doesn't need linking' and 'not
  // preprocessing-only' mode.
  std::string GetTarget(const std::string& compilation_output_file_name);

  // Returns the dependency file name to be used. If -MF defined a file name
  // before, use it. Othwise, returns a filename formed by appending .d to the
  // output filename deduced in 'doesn't need linking' and 'no
  // preprocessing-only' mode.
  std::string GetDependencyFileName(
      const std::string& compilation_output_file_name);

  std::string user_specified_dep_file_name_;
  std::string user_specified_dep_target_label_;
  dump_mode mode_;
};
}

#endif  // GLSLC_DEPENDENCY_INFO_H
