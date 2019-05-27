// Copyright (c) 2016 Google Inc.
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

#ifndef TOOLS_IO_H_
#define TOOLS_IO_H_

#include <cstdint>
#include <cstdio>
#include <vector>

// Appends the content from the file named as |filename| to |data|, assuming
// each element in the file is of type |T|. The file is opened with the given
// |mode|. If |filename| is nullptr or "-", reads from the standard input, but
// reopened with the given mode. If any error occurs, writes error messages to
// standard error and returns false.
template <typename T>
bool ReadFile(const char* filename, const char* mode, std::vector<T>* data) {
  const int buf_size = 1024;
  const bool use_file = filename && strcmp("-", filename);
  if (FILE* fp =
          (use_file ? fopen(filename, mode) : freopen(nullptr, mode, stdin))) {
    T buf[buf_size];
    while (size_t len = fread(buf, sizeof(T), buf_size, fp)) {
      data->insert(data->end(), buf, buf + len);
    }
    if (ftell(fp) == -1L) {
      if (ferror(fp)) {
        fprintf(stderr, "error: error reading file '%s'\n", filename);
        return false;
      }
    } else {
      if (sizeof(T) != 1 && (ftell(fp) % sizeof(T))) {
        fprintf(
            stderr,
            "error: file size should be a multiple of %zd; file '%s' corrupt\n",
            sizeof(T), filename);
        return false;
      }
    }
    if (use_file) fclose(fp);
  } else {
    fprintf(stderr, "error: file does not exist '%s'\n", filename);
    return false;
  }
  return true;
}

// Writes the given |data| into the file named as |filename| using the given
// |mode|, assuming |data| is an array of |count| elements of type |T|. If
// |filename| is nullptr or "-", writes to standard output. If any error occurs,
// returns false and outputs error message to standard error.
template <typename T>
bool WriteFile(const char* filename, const char* mode, const T* data,
               size_t count) {
  const bool use_stdout =
      !filename || (filename[0] == '-' && filename[1] == '\0');
  if (FILE* fp = (use_stdout ? stdout : fopen(filename, mode))) {
    size_t written = fwrite(data, sizeof(T), count, fp);
    if (count != written) {
      fprintf(stderr, "error: could not write to file '%s'\n", filename);
      return false;
    }
    if (!use_stdout) fclose(fp);
  } else {
    fprintf(stderr, "error: could not open file '%s'\n", filename);
    return false;
  }
  return true;
}

#endif  // TOOLS_IO_H_
