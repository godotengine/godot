// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/posix/process_info.h"

#include <stdio.h>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "util/file/delimited_file_reader.h"
#include "util/file/file_reader.h"
#include "util/file/string_file.h"
#include "util/linux/proc_stat_reader.h"
#include "util/misc/lexing.h"

namespace crashpad {

ProcessInfo::ProcessInfo()
    : connection_(),
      supplementary_groups_(),
      start_time_(),
      pid_(-1),
      ppid_(-1),
      uid_(-1),
      euid_(-1),
      suid_(-1),
      gid_(-1),
      egid_(-1),
      sgid_(-1),
      start_time_initialized_(),
      initialized_() {}

ProcessInfo::~ProcessInfo() {}

bool ProcessInfo::InitializeWithPtrace(PtraceConnection* connection) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  DCHECK(connection);

  connection_ = connection;
  pid_ = connection->GetProcessID();
  is_64_bit_ = connection->Is64Bit();

  {
    char path[32];
    snprintf(path, sizeof(path), "/proc/%d/status", pid_);
    std::string contents;
    if (!connection->ReadFileContents(base::FilePath(path), &contents)) {
      return false;
    }
    StringFile status_file;
    status_file.SetString(contents);

    DelimitedFileReader status_file_line_reader(&status_file);

    bool have_ppid = false;
    bool have_uids = false;
    bool have_gids = false;
    bool have_groups = false;
    std::string line;
    DelimitedFileReader::Result result;
    while ((result = status_file_line_reader.GetLine(&line)) ==
           DelimitedFileReader::Result::kSuccess) {
      if (line.back() != '\n') {
        LOG(ERROR) << "format error: unterminated line at EOF";
        return false;
      }

      bool understood_line = false;
      const char* line_c = line.c_str();
      if (AdvancePastPrefix(&line_c, "PPid:\t")) {
        if (have_ppid) {
          LOG(ERROR) << "format error: multiple PPid lines";
          return false;
        }
        have_ppid = AdvancePastNumber(&line_c, &ppid_);
        if (!have_ppid) {
          LOG(ERROR) << "format error: unrecognized PPid format";
          return false;
        }
        understood_line = true;
      } else if (AdvancePastPrefix(&line_c, "Uid:\t")) {
        if (have_uids) {
          LOG(ERROR) << "format error: multiple Uid lines";
          return false;
        }
        uid_t fsuid;
        have_uids = AdvancePastNumber(&line_c, &uid_) &&
                    AdvancePastPrefix(&line_c, "\t") &&
                    AdvancePastNumber(&line_c, &euid_) &&
                    AdvancePastPrefix(&line_c, "\t") &&
                    AdvancePastNumber(&line_c, &suid_) &&
                    AdvancePastPrefix(&line_c, "\t") &&
                    AdvancePastNumber(&line_c, &fsuid);
        if (!have_uids) {
          LOG(ERROR) << "format error: unrecognized Uid format";
          return false;
        }
        understood_line = true;
      } else if (AdvancePastPrefix(&line_c, "Gid:\t")) {
        if (have_gids) {
          LOG(ERROR) << "format error: multiple Gid lines";
          return false;
        }
        gid_t fsgid;
        have_gids = AdvancePastNumber(&line_c, &gid_) &&
                    AdvancePastPrefix(&line_c, "\t") &&
                    AdvancePastNumber(&line_c, &egid_) &&
                    AdvancePastPrefix(&line_c, "\t") &&
                    AdvancePastNumber(&line_c, &sgid_) &&
                    AdvancePastPrefix(&line_c, "\t") &&
                    AdvancePastNumber(&line_c, &fsgid);
        if (!have_gids) {
          LOG(ERROR) << "format error: unrecognized Gid format";
          return false;
        }
        understood_line = true;
      } else if (AdvancePastPrefix(&line_c, "Groups:\t")) {
        if (have_groups) {
          LOG(ERROR) << "format error: multiple Groups lines";
          return false;
        }
        if (!AdvancePastPrefix(&line_c, " ")) {
          // In Linux 4.10, even an empty Groups: line has a trailing space.
          gid_t group;
          while (AdvancePastNumber(&line_c, &group)) {
            supplementary_groups_.insert(group);
            if (!AdvancePastPrefix(&line_c, " ")) {
              LOG(ERROR) << "format error: unrecognized Groups format";
              return false;
            }
          }
        }
        have_groups = true;
        understood_line = true;
      }

      if (understood_line && line_c != &line.back()) {
        LOG(ERROR) << "format error: unconsumed trailing data";
        return false;
      }
    }
    if (result != DelimitedFileReader::Result::kEndOfFile) {
      return false;
    }
    if (!have_ppid || !have_uids || !have_gids || !have_groups) {
      LOG(ERROR) << "format error: missing fields";
      return false;
    }
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

pid_t ProcessInfo::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return pid_;
}

pid_t ProcessInfo::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ppid_;
}

uid_t ProcessInfo::RealUserID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return uid_;
}

uid_t ProcessInfo::EffectiveUserID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return euid_;
}

uid_t ProcessInfo::SavedUserID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return suid_;
}

gid_t ProcessInfo::RealGroupID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return gid_;
}

gid_t ProcessInfo::EffectiveGroupID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return egid_;
}

gid_t ProcessInfo::SavedGroupID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return sgid_;
}

std::set<gid_t> ProcessInfo::SupplementaryGroups() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return supplementary_groups_;
}

std::set<gid_t> ProcessInfo::AllGroups() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::set<gid_t> all_groups = SupplementaryGroups();
  all_groups.insert(RealGroupID());
  all_groups.insert(EffectiveGroupID());
  all_groups.insert(SavedGroupID());
  return all_groups;
}

bool ProcessInfo::DidChangePrivileges() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(jperaza): Is this possible to determine?
  return false;
}

bool ProcessInfo::Is64Bit() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return is_64_bit_;
}

bool ProcessInfo::StartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (start_time_initialized_.is_uninitialized()) {
    start_time_initialized_.set_invalid();
    ProcStatReader reader;
    if (!reader.Initialize(connection_, pid_)) {
      return false;
    }
    if (!reader.StartTime(&start_time_)) {
      return false;
    }
    start_time_initialized_.set_valid();
  }

  if (!start_time_initialized_.is_valid()) {
    return false;
  }

  *start_time = start_time_;
  return true;
}

bool ProcessInfo::Arguments(std::vector<std::string>* argv) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  char path[32];
  snprintf(path, sizeof(path), "/proc/%d/cmdline", pid_);
  std::string contents;
  if (!connection_->ReadFileContents(base::FilePath(path), &contents)) {
    return false;
  }
  StringFile cmdline_file;
  cmdline_file.SetString(contents);

  DelimitedFileReader cmdline_file_field_reader(&cmdline_file);

  std::vector<std::string> local_argv;
  std::string argument;
  DelimitedFileReader::Result result;
  while ((result = cmdline_file_field_reader.GetDelim('\0', &argument)) ==
         DelimitedFileReader::Result::kSuccess) {
    if (argument.back() != '\0') {
      LOG(ERROR) << "format error";
      return false;
    }
    argument.pop_back();
    local_argv.push_back(argument);
  }
  if (result != DelimitedFileReader::Result::kEndOfFile) {
    return false;
  }

  argv->swap(local_argv);
  return true;
}

}  // namespace crashpad
