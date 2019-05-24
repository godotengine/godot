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

#include "libshaderc_util/io.h"

#include "libshaderc_util/universal_unistd.h"

#if _WIN32
// Need _fileno from stdio.h
// Need _O_BINARY and _O_TEXT from fcntl.h
#include <fcntl.h>
#include <stdio.h>
#endif

#include <errno.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {

// Outputs a descriptive message for errno_value to cerr.
// This may be truncated to 1023 bytes on certain platforms.
void OutputFileErrorMessage(int errno_value) {
#ifdef _MSC_VER
  // If the error message is more than 1023 bytes it will be truncated.
  char buffer[1024];
  strerror_s(buffer, errno_value);
  std::cerr << ": " << buffer << std::endl;
#else
  std::cerr << ": " << strerror(errno_value) << std::endl;
#endif
}

}  // anonymous namespace

namespace shaderc_util {

bool IsAbsolutePath(const std::string& path) {
  if (path.empty()) return false;
  // Unix-like OS: /path/to/file
  if (path.front() == '/') return true;
  // Windows: \\server\user\file
  if (path.size() > 1 && path[0] == '\\' && path[1] == '\\') {
    return true;
  }
  // Windows: X:\path\to\file
  if (path.size() > 2 && ::isalpha(path[0]) && path[1] == ':' &&
      path[2] == '\\') {
    return true;
  }
  return false;
}

std::string GetBaseFileName(const std::string& file_path) {
  size_t loc_slash = file_path.find_last_of("/\\");
  std::string base_name =
      file_path.substr((loc_slash == std::string::npos ? -1 : loc_slash) + 1);
  if (base_name == ".." || base_name == ".") {
    base_name = "";
  }
  return base_name;
}

bool ReadFile(const std::string& input_file_name,
              std::vector<char>* input_data) {
  std::istream* stream = &std::cin;
  std::ifstream input_file;
  if (input_file_name != "-") {
    input_file.open(input_file_name, std::ios_base::binary);
    stream = &input_file;
    if (input_file.fail()) {
      std::cerr << "glslc: error: cannot open input file: '" << input_file_name
                << "'";
      if (access(input_file_name.c_str(), R_OK) != 0) {
        OutputFileErrorMessage(errno);
        return false;
      }
      std::cerr << std::endl;
      return false;
    }
  }
  *input_data = std::vector<char>((std::istreambuf_iterator<char>(*stream)),
                                  std::istreambuf_iterator<char>());
  return true;
}

std::ostream* GetOutputStream(const string_piece& output_filename,
                              std::ofstream* file_stream, std::ostream* err) {
  std::ostream* stream = &std::cout;
  if (output_filename != "-") {
    file_stream->open(output_filename.str(), std::ios_base::binary);
    stream = file_stream;
    if (file_stream->fail()) {
      *err << "glslc: error: cannot open output file: '" << output_filename
                << "'";
      if (access(output_filename.str().c_str(), W_OK) != 0) {
        OutputFileErrorMessage(errno);
        return nullptr;
      }
      std::cerr << std::endl;
      return nullptr;
    }
  }
  return stream;
}

bool WriteFile(std::ostream* stream, const string_piece& output_data) {
  if (output_data.size() > 0) {
    stream->write(output_data.data(), output_data.size());
    if (!stream->good()) {
      return false;
    }
  }
  stream->flush();
  return true;
}

void FlushAndSetBinaryModeOnStdout() {
  std::fflush(stdout);
#if _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif
}

void FlushAndSetTextModeOnStdout() {
  std::fflush(stdout);
#if _WIN32
  _setmode(_fileno(stdout), _O_TEXT);
#endif
}

}  // namespace shaderc_util
