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
// Utility that can inspect another process and write a crash dump

#include <cstdio>
#include <iostream>
#include <servers/bootstrap.h>
#include <stdio.h>
#include <string.h>
#include <string>

#import "client/mac/crash_generation/Inspector.h"

#import "client/mac/Framework/Breakpad.h"
#import "client/mac/handler/minidump_generator.h"

#import "common/mac/MachIPC.h"
#include "common/mac/bootstrap_compat.h"
#include "common/mac/launch_reporter.h"

#import "GTMDefines.h"

#import <Foundation/Foundation.h>

namespace google_breakpad {

//=============================================================================
void Inspector::Inspect(const char* receive_port_name) {
  kern_return_t result = ResetBootstrapPort();
  if (result != KERN_SUCCESS) {
    return;
  }

  result = ServiceCheckIn(receive_port_name);

  if (result == KERN_SUCCESS) {
    result = ReadMessages();

    if (result == KERN_SUCCESS) {
      // Inspect the task and write a minidump file.
      bool wrote_minidump = InspectTask();

      // Send acknowledgement to the crashed process that the inspection
      // has finished.  It will then be able to cleanly exit.
      // The return value is ignored because failure isn't fatal. If the process
      // didn't get the message there's nothing we can do, and we still want to
      // send the report.
      SendAcknowledgement();

      if (wrote_minidump) {
        // Ask the user if he wants to upload the crash report to a server,
        // and do so if he agrees.
        LaunchReporter(
            config_params_.GetValueForKey(BREAKPAD_REPORTER_EXE_LOCATION),
            config_file_.GetFilePath());
      } else {
        fprintf(stderr, "Inspection of crashed process failed\n");
      }

      // Now that we're done reading messages, cleanup the service, but only
      // if there was an actual exception
      // Otherwise, it means the dump was generated on demand and the process
      // lives on, and we might be needed again in the future.
      if (exception_code_) {
        ServiceCheckOut(receive_port_name);
      }
    } else {
        PRINT_MACH_RESULT(result, "Inspector: WaitForMessage()");
    }
  }
}

//=============================================================================
kern_return_t Inspector::ResetBootstrapPort() {
  // A reasonable default, in case anything fails.
  bootstrap_subset_port_ = bootstrap_port;

  mach_port_t self_task = mach_task_self();

  kern_return_t kr = task_get_bootstrap_port(self_task,
                                             &bootstrap_subset_port_);
  if (kr != KERN_SUCCESS) {
    NSLog(@"ResetBootstrapPort: task_get_bootstrap_port failed: %s (%d)",
          mach_error_string(kr), kr);
    return kr;
  }

  mach_port_t bootstrap_parent_port;
  kr = bootstrap_look_up(bootstrap_subset_port_,
                         const_cast<char*>(BREAKPAD_BOOTSTRAP_PARENT_PORT),
                         &bootstrap_parent_port);
  if (kr != BOOTSTRAP_SUCCESS) {
    NSLog(@"ResetBootstrapPort: bootstrap_look_up failed: %s (%d)",
#if defined(MAC_OS_X_VERSION_10_5) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_5
          bootstrap_strerror(kr),
#else
          mach_error_string(kr),
#endif
          kr);
    return kr;
  }

  kr = task_set_bootstrap_port(self_task, bootstrap_parent_port);
  if (kr != KERN_SUCCESS) {
    NSLog(@"ResetBootstrapPort: task_set_bootstrap_port failed: %s (%d)",
          mach_error_string(kr), kr);
    return kr;
  }

  // Some things access the bootstrap port through this global variable
  // instead of calling task_get_bootstrap_port.
  bootstrap_port = bootstrap_parent_port;

  return KERN_SUCCESS;
}

//=============================================================================
kern_return_t Inspector::ServiceCheckIn(const char* receive_port_name) {
  // We need to get the mach port representing this service, so we can
  // get information from the crashed process.
  kern_return_t kr = bootstrap_check_in(bootstrap_subset_port_,
                                        (char*)receive_port_name,
                                        &service_rcv_port_);

  if (kr != KERN_SUCCESS) {
#if VERBOSE
    PRINT_MACH_RESULT(kr, "Inspector: bootstrap_check_in()");
#endif
  }

  return kr;
}

//=============================================================================
kern_return_t Inspector::ServiceCheckOut(const char* receive_port_name) {
  // We're done receiving mach messages from the crashed process,
  // so clean up a bit.
  kern_return_t kr;

  // DO NOT use mach_port_deallocate() here -- it will fail and the
  // following bootstrap_register() will also fail leaving our service
  // name hanging around forever (until reboot)
  kr = mach_port_destroy(mach_task_self(), service_rcv_port_);

  if (kr != KERN_SUCCESS) {
    PRINT_MACH_RESULT(kr,
      "Inspector: UNREGISTERING: service_rcv_port mach_port_deallocate()");
    return kr;
  }

  // Unregister the service associated with the receive port.
  kr = breakpad::BootstrapRegister(bootstrap_subset_port_,
                                   (char*)receive_port_name,
                                   MACH_PORT_NULL);

  if (kr != KERN_SUCCESS) {
    PRINT_MACH_RESULT(kr, "Inspector: UNREGISTERING: bootstrap_register()");
  }

  return kr;
}

//=============================================================================
kern_return_t Inspector::ReadMessages() {
  // Wait for an initial message from the crashed process containing basic
  // information about the crash.
  ReceivePort receive_port(service_rcv_port_);

  MachReceiveMessage message;
  kern_return_t result = receive_port.WaitForMessage(&message, 1000);

  if (result == KERN_SUCCESS) {
    InspectorInfo& info = (InspectorInfo&)*message.GetData();
    exception_type_ = info.exception_type;
    exception_code_ = info.exception_code;
    exception_subcode_ = info.exception_subcode;

#if VERBOSE
    printf("message ID = %d\n", message.GetMessageID());
#endif

    remote_task_ = message.GetTranslatedPort(0);
    crashing_thread_ = message.GetTranslatedPort(1);
    handler_thread_ = message.GetTranslatedPort(2);
    ack_port_ = message.GetTranslatedPort(3);

#if VERBOSE
    printf("exception_type = %d\n", exception_type_);
    printf("exception_code = %d\n", exception_code_);
    printf("exception_subcode = %d\n", exception_subcode_);
    printf("remote_task = %d\n", remote_task_);
    printf("crashing_thread = %d\n", crashing_thread_);
    printf("handler_thread = %d\n", handler_thread_);
    printf("ack_port_ = %d\n", ack_port_);
    printf("parameter count = %d\n", info.parameter_count);
#endif

    // In certain situations where multiple crash requests come
    // through quickly, we can end up with the mach IPC messages not
    // coming through correctly.  Since we don't know what parameters
    // we've missed, we can't do much besides abort the crash dump
    // situation in this case.
    unsigned int parameters_read = 0;
    // The initial message contains the number of key value pairs that
    // we are expected to read.
    // Read each key/value pair, one mach message per key/value pair.
    for (unsigned int i = 0; i < info.parameter_count; ++i) {
      MachReceiveMessage parameter_message;
      result = receive_port.WaitForMessage(&parameter_message, 1000);

      if(result == KERN_SUCCESS) {
        KeyValueMessageData& key_value_data =
          (KeyValueMessageData&)*parameter_message.GetData();
        // If we get a blank key, make sure we don't increment the
        // parameter count; in some cases (notably on-demand generation
        // many times in a short period of time) caused the Mach IPC
        // messages to not come through correctly.
        if (strlen(key_value_data.key) == 0) {
          continue;
        }
        parameters_read++;

        config_params_.SetKeyValue(key_value_data.key, key_value_data.value);
      } else {
        PRINT_MACH_RESULT(result, "Inspector: key/value message");
        break;
      }
    }
    if (parameters_read != info.parameter_count) {
      return KERN_FAILURE;
    }
  }

  return result;
}

//=============================================================================
bool Inspector::InspectTask() {
  // keep the task quiet while we're looking at it
  task_suspend(remote_task_);

  NSString* minidumpDir;

  const char* minidumpDirectory =
    config_params_.GetValueForKey(BREAKPAD_DUMP_DIRECTORY);

  // If the client app has not specified a minidump directory,
  // use a default of Library/<kDefaultLibrarySubdirectory>/<Product Name>
  if (!minidumpDirectory || 0 == strlen(minidumpDirectory)) {
    NSArray* libraryDirectories =
      NSSearchPathForDirectoriesInDomains(NSLibraryDirectory,
                                          NSUserDomainMask,
                                          YES);

    NSString* applicationSupportDirectory =
        [libraryDirectories objectAtIndex:0];
    NSString* library_subdirectory = [NSString
        stringWithUTF8String:kDefaultLibrarySubdirectory];
    NSString* breakpad_product = [NSString
        stringWithUTF8String:config_params_.GetValueForKey(BREAKPAD_PRODUCT)];

    NSArray* path_components = [NSArray
        arrayWithObjects:applicationSupportDirectory,
                         library_subdirectory,
                         breakpad_product,
                         nil];

    minidumpDir = [NSString pathWithComponents:path_components];
  } else {
    minidumpDir = [[NSString stringWithUTF8String:minidumpDirectory]
                    stringByExpandingTildeInPath];
  }

  MinidumpLocation minidumpLocation(minidumpDir);

  // Obscure bug alert:
  // Don't use [NSString stringWithFormat] to build up the path here since it
  // assumes system encoding and in RTL locales will prepend an LTR override
  // character for paths beginning with '/' which fileSystemRepresentation does
  // not remove. Filed as rdar://6889706 .
  NSString* path_ns = [NSString
      stringWithUTF8String:minidumpLocation.GetPath()];
  NSString* pathid_ns = [NSString
      stringWithUTF8String:minidumpLocation.GetID()];
  NSString* minidumpPath = [path_ns stringByAppendingPathComponent:pathid_ns];
  minidumpPath = [minidumpPath
      stringByAppendingPathExtension:@"dmp"];

  config_file_.WriteFile( 0,
                          &config_params_,
                          minidumpLocation.GetPath(),
                          minidumpLocation.GetID());


  MinidumpGenerator generator(remote_task_, handler_thread_);

  if (exception_type_ && exception_code_) {
    generator.SetExceptionInformation(exception_type_,
                                      exception_code_,
                                      exception_subcode_,
                                      crashing_thread_);
  }


  bool result = generator.Write([minidumpPath fileSystemRepresentation]);

  // let the task continue
  task_resume(remote_task_);

  return result;
}

//=============================================================================
// The crashed task needs to be told that the inspection has finished.
// It will wait on a mach port (with timeout) until we send acknowledgement.
kern_return_t Inspector::SendAcknowledgement() {
  if (ack_port_ != MACH_PORT_DEAD) {
    MachPortSender sender(ack_port_);
    MachSendMessage ack_message(kMsgType_InspectorAcknowledgement);

    kern_return_t result = sender.SendMessage(ack_message, 2000);

#if VERBOSE
    PRINT_MACH_RESULT(result, "Inspector: sent acknowledgement");
#endif

    return result;
  }

  return KERN_INVALID_NAME;
}

} // namespace google_breakpad

