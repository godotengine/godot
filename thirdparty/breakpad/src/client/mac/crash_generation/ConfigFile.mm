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

#import "client/mac/crash_generation/ConfigFile.h"

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <sys/time.h>

#import "client/apple/Framework/BreakpadDefines.h"
#import "common/mac/GTMDefines.h"


namespace google_breakpad {

//=============================================================================
BOOL EnsureDirectoryPathExists(NSString* dirPath) {
  NSFileManager* mgr = [NSFileManager defaultManager];

  NSDictionary* attrs =
    [NSDictionary dictionaryWithObject:[NSNumber numberWithUnsignedLong:0750]
                                forKey:NSFilePosixPermissions];

  return [mgr createDirectoryAtPath:dirPath
        withIntermediateDirectories:YES
                         attributes:attrs
                              error:nil];
}

//=============================================================================
BOOL ConfigFile::WriteData(const void* data, size_t length) {
  size_t result = write(config_file_, data, length);

  return result == length;
}

//=============================================================================
BOOL ConfigFile::AppendConfigData(const char* key,
                                  const void* data, size_t length) {
  assert(config_file_ != -1);

  if (!key) {
    return NO;
  }

  if (!data) {
    return NO;
  }

  // Write the key, \n, length of data (ascii integer), \n, data
  char buffer[16];
  char nl = '\n';
  BOOL result = WriteData(key, strlen(key));

  snprintf(buffer, sizeof(buffer) - 1, "\n%lu\n", length);
  result &= WriteData(buffer, strlen(buffer));
  result &= WriteData(data, length);
  result &= WriteData(&nl, 1);
  return result;
}

//=============================================================================
BOOL ConfigFile::AppendConfigString(const char* key,
                                    const char* value) {
  return AppendConfigData(key, value, strlen(value));
}

//=============================================================================
BOOL ConfigFile::AppendCrashTimeParameters(const char* processStartTimeString) {
  // Set process uptime parameter
  struct timeval tv;
  gettimeofday(&tv, NULL);

  char processUptimeString[32], processCrashtimeString[32];
  // Set up time if we've received the start time.
  if (processStartTimeString) {
    time_t processStartTime = strtol(processStartTimeString, NULL, 10);
    time_t processUptime = tv.tv_sec - processStartTime;
    // Store the uptime in milliseconds.
    snprintf(processUptimeString, sizeof(processUptimeString), "%llu",
             static_cast<unsigned long long int>(processUptime) * 1000);
    if (!AppendConfigString(BREAKPAD_PROCESS_UP_TIME, processUptimeString))
      return false;
  }

  snprintf(processCrashtimeString, sizeof(processCrashtimeString), "%llu",
           static_cast<unsigned long long int>(tv.tv_sec));
  return AppendConfigString(BREAKPAD_PROCESS_CRASH_TIME,
                            processCrashtimeString);
}

//=============================================================================
void ConfigFile::WriteFile(const char* directory,
                           const SimpleStringDictionary* configurationParameters,
                           const char* dump_dir,
                           const char* minidump_id) {

  assert(config_file_ == -1);

  // Open and write out configuration file preamble
  if (directory) {
    snprintf(config_file_path_, sizeof(config_file_path_), "%s/Config-XXXXXX",
             directory);
  } else {
    strlcpy(config_file_path_, "/tmp/Config-XXXXXX",
            sizeof(config_file_path_));
  }
  config_file_ = mkstemp(config_file_path_);

  if (config_file_ == -1) {
    return;
  }

  has_created_file_ = true;

  // Add the minidump dir
  AppendConfigString(kReporterMinidumpDirectoryKey, dump_dir);
  AppendConfigString(kReporterMinidumpIDKey, minidump_id);

  // Write out the configuration parameters
  BOOL result = YES;
  const SimpleStringDictionary& dictionary = *configurationParameters;

  const SimpleStringDictionary::Entry* entry = NULL;
  SimpleStringDictionary::Iterator iter(dictionary);

  while ((entry = iter.Next())) {
    result = AppendConfigString(entry->key, entry->value);

    if (!result)
      break;
  }
  AppendCrashTimeParameters(
      configurationParameters->GetValueForKey(BREAKPAD_PROCESS_START_TIME));

  close(config_file_);
  config_file_ = -1;
}

} // namespace google_breakpad
