// Copyright (c) 2007, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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
// Interface file between the Breakpad.framework and
// the Inspector process.

#include "common/simple_string_dictionary.h"

#import <Foundation/Foundation.h>
#include <mach/mach.h>

#import "client/mac/crash_generation/ConfigFile.h"
#import "client/mac/handler/minidump_generator.h"


// Types of mach messsages (message IDs)
enum {
  kMsgType_InspectorInitialInfo = 0,    // data is InspectorInfo
  kMsgType_InspectorKeyValuePair = 1,   // data is KeyValueMessageData
  kMsgType_InspectorAcknowledgement = 2 // no data sent
};

// Initial information sent from the crashed process by
// Breakpad.framework to the Inspector process
// The mach message with this struct as data will also include
// several descriptors for sending mach port rights to the crashed
// task, etc.
struct InspectorInfo {
  int           exception_type;
  int           exception_code;
  int           exception_subcode;
  unsigned int  parameter_count;  // key-value pairs
};

// Key/value message data to be sent to the Inspector
struct KeyValueMessageData {
 public:
  KeyValueMessageData() {}
  explicit KeyValueMessageData(
      const google_breakpad::SimpleStringDictionary::Entry& inEntry) {
    strlcpy(key, inEntry.key, sizeof(key) );
    strlcpy(value, inEntry.value, sizeof(value) );
  }

  char key[google_breakpad::SimpleStringDictionary::key_size];
  char value[google_breakpad::SimpleStringDictionary::value_size];
};

using std::string;
using google_breakpad::MinidumpGenerator;

namespace google_breakpad {

//=============================================================================
class MinidumpLocation {
 public:
  MinidumpLocation(NSString* minidumpDir) {
    // Ensure that the path exists.  Fallback to /tmp if unable to locate path.
    assert(minidumpDir);
    if (!EnsureDirectoryPathExists(minidumpDir)) {
      minidumpDir = @"/tmp";
    }

    strlcpy(minidump_dir_path_, [minidumpDir fileSystemRepresentation],
            sizeof(minidump_dir_path_));

    // now generate a unique ID
    string dump_path(minidump_dir_path_);
    string next_minidump_id;

    string next_minidump_path_ =
      (MinidumpGenerator::UniqueNameInDirectory(dump_path, &next_minidump_id));

    strlcpy(minidump_id_, next_minidump_id.c_str(), sizeof(minidump_id_));
  }

  const char* GetPath() { return minidump_dir_path_; }
  const char* GetID() { return minidump_id_; }

 private:
  char minidump_dir_path_[PATH_MAX];             // Path to minidump directory
  char minidump_id_[128];
};

//=============================================================================
class Inspector {
 public:
  Inspector() {}

  // given a bootstrap service name, receives mach messages
  // from a crashed process, then inspects it, creates a minidump file
  // and asks the user if he wants to upload it to a server.
  void            Inspect(const char* receive_port_name);

 private:
  // The Inspector is invoked with its bootstrap port set to the bootstrap
  // subset established in OnDemandServer.mm OnDemandServer::Initialize.
  // For proper communication with the system, the sender (which will inherit
  // the Inspector's bootstrap port) needs the per-session bootstrap namespace
  // available directly in its bootstrap port. OnDemandServer stashed this
  // port into the subset namespace under a special name. ResetBootstrapPort
  // recovers this port and switches this task to use it as its own bootstrap
  // (ensuring that children like the sender will inherit it), and saves the
  // subset in bootstrap_subset_port_ for use by ServiceCheckIn and
  // ServiceCheckOut.
  kern_return_t   ResetBootstrapPort();

  kern_return_t   ServiceCheckIn(const char* receive_port_name);
  kern_return_t   ServiceCheckOut(const char* receive_port_name);

  kern_return_t   ReadMessages();

  bool            InspectTask();
  kern_return_t   SendAcknowledgement();

  // The bootstrap port in which the inspector is registered and into which it
  // must check in.
  mach_port_t     bootstrap_subset_port_;

  mach_port_t     service_rcv_port_;

  int             exception_type_;
  int             exception_code_;
  int             exception_subcode_;
  mach_port_t     remote_task_;
  mach_port_t     crashing_thread_;
  mach_port_t     handler_thread_;
  mach_port_t     ack_port_;

  SimpleStringDictionary config_params_;

  ConfigFile      config_file_;
};


} // namespace google_breakpad
