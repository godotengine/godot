// Copyright 2011 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Utility class that can persist a SimpleStringDictionary to disk.

#import <Foundation/Foundation.h>

#include "common/simple_string_dictionary.h"

namespace google_breakpad {

BOOL EnsureDirectoryPathExists(NSString* dirPath);

//=============================================================================
class ConfigFile {
 public:
  ConfigFile() {
    config_file_ = -1;
    config_file_path_[0] = 0;
    has_created_file_ = false;
  }

  ~ConfigFile() {
  }

  void WriteFile(const char* directory,
                 const SimpleStringDictionary* configurationParameters,
                 const char* dump_dir,
                 const char* minidump_id);

  const char* GetFilePath() { return config_file_path_; }

  void Unlink() {
    if (config_file_ != -1)
      unlink(config_file_path_);

    config_file_ = -1;
  }

 private:
  BOOL WriteData(const void* data, size_t length);

  BOOL AppendConfigData(const char* key,
                        const void* data,
                        size_t length);

  BOOL AppendConfigString(const char* key,
                          const char* value);

  BOOL AppendCrashTimeParameters(const char* processStartTimeString);

  int   config_file_;                    // descriptor for config file
  char  config_file_path_[PATH_MAX];     // Path to configuration file
  bool  has_created_file_;
};

} // namespace google_breakpad
