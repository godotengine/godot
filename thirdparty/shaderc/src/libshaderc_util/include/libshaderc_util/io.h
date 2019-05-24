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

#ifndef LIBSHADERC_UTIL_IO_H_
#define LIBSHADERC_UTIL_IO_H_

#include <string>
#include <vector>

#include "string_piece.h"

namespace shaderc_util {

// Returns true if the given path is an absolute path.
bool IsAbsolutePath(const std::string& path);

// A helper function to return the base file name from either absolute path or
// relative path representation of a file. It keeps the component from the last
// '/' or '\' to the end of the given string. If the component is '..' or '.',
// returns an empty string. If '/' or '\' is the last char of the given string,
// also returns an empty string.
// e.g.: dir_a/dir_b/file_c.vert => file_c.vert
//       dir_a/dir_b/.. => <empty string>
//       dir_a/dir_b/.  => <empty string>
//       dir_a/dirb/c/  => <empty string>
// Note that this method doesn't check whether the given path is a valid one or
// not.
std::string GetBaseFileName(const std::string& file_path);

// Reads all of the characters in a given file into input_data.  Outputs an
// error message to std::cerr if the file could not be read and returns false if
// there was an error.  If the input_file is "-", then input is read from
// std::cin.
bool ReadFile(const std::string& input_file_name,
              std::vector<char>* input_data);

// Returns and initializes the file_stream parameter if the output_filename
// refers to a file, or returns &std::cout if the output_filename is "-".
// Returns nullptr and emits an error message to err if the file could
// not be opened for writing.  If the output refers to a file, and the open
// failed for writing, file_stream is left with its fail_bit set.
std::ostream* GetOutputStream(const string_piece& output_filename,
                              std::ofstream* file_stream, std::ostream* err);

// Writes output_data to a file, overwriting if it exists.  If output_file_name
// is "-", writes to std::cout.
bool WriteFile(std::ostream* output_stream, const string_piece& output_data);

// Flush the standard output stream and set it to binary mode.  Subsequent
// output will not translate newlines to carriage-return newline pairs.
void FlushAndSetBinaryModeOnStdout();
// Flush the standard output stream and set it to text mode.  Subsequent
// output will translate newlines to carriage-return newline pairs.
void FlushAndSetTextModeOnStdout();

}  // namespace shaderc_util

#endif  // LIBSHADERC_UTIL_IO_H_
