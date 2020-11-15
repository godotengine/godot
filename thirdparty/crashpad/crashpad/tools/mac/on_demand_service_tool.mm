// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#include <getopt.h>
#include <launch.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "base/mac/foundation_util.h"
#include "base/strings/sys_string_conversions.h"
#include "tools/tool_support.h"
#include "util/mac/service_management.h"
#include "util/stdlib/objc.h"

namespace crashpad {
namespace {

void Usage(const std::string& me) {
  fprintf(stderr,
"Usage: %s -L -l LABEL [OPTION]... COMMAND [ARG]...\n"
"       %s -U -l LABEL\n"
"Load and unload on-demand Mach services from launchd.\n"
"\n"
"  -L, --load                  load (submit) the job identified by --label;\n"
"                              COMMAND must be specified\n"
"  -U, --unload                unload (remove) the job identified by --label\n"
"  -l, --label=LABEL           identify the job to launchd with LABEL\n"
"  -m, --mach-service=SERVICE  register SERVICE with the bootstrap server\n"
"      --help                  display this help and exit\n"
"      --version               output version information and exit\n",
          me.c_str(),
          me.c_str());
  ToolSupport::UsageTail(me);
}

int OnDemandServiceToolMain(int argc, char* argv[]) {
  const std::string me(basename(argv[0]));

  enum Operation {
    kOperationUnknown = 0,
    kOperationLoadJob,
    kOperationUnloadJob,
  };

  enum OptionFlags {
    // “Short” (single-character) options.
    kOptionLoadJob = 'L',
    kOptionUnloadJob = 'U',
    kOptionJobLabel = 'l',
    kOptionMachService = 'm',

    // Long options without short equivalents.
    kOptionLastChar = 255,

    // Standard options.
    kOptionHelp = -2,
    kOptionVersion = -3,
  };

  struct {
    Operation operation;
    std::string job_label;
    std::vector<std::string> mach_services;
  } options = {};

  static constexpr option long_options[] = {
      {"load", no_argument, nullptr, kOptionLoadJob},
      {"unload", no_argument, nullptr, kOptionUnloadJob},
      {"label", required_argument, nullptr, kOptionJobLabel},
      {"mach-service", required_argument, nullptr, kOptionMachService},
      {"help", no_argument, nullptr, kOptionHelp},
      {"version", no_argument, nullptr, kOptionVersion},
      {nullptr, 0, nullptr, 0},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "+LUl:m:", long_options, nullptr)) !=
          -1) {
    switch (opt) {
      case kOptionLoadJob:
        options.operation = kOperationLoadJob;
        break;
      case kOptionUnloadJob:
        options.operation = kOperationUnloadJob;
        break;
      case kOptionJobLabel:
        options.job_label = optarg;
        break;
      case kOptionMachService:
        options.mach_services.push_back(optarg);
        break;
      case kOptionHelp:
        Usage(me);
        return EXIT_SUCCESS;
      case kOptionVersion:
        ToolSupport::Version(me);
        return EXIT_SUCCESS;
      default:
        ToolSupport::UsageHint(me, nullptr);
        return EXIT_FAILURE;
    }
  }
  argc -= optind;
  argv += optind;

  if (options.job_label.empty()) {
    ToolSupport::UsageHint(me, "must provide -l");
    return EXIT_FAILURE;
  }

  switch (options.operation) {
    case kOperationLoadJob: {
      if (argc == 0) {
        ToolSupport::UsageHint(me, "must provide COMMAND with -L");
        return EXIT_FAILURE;
      }

      @autoreleasepool {
        NSString* job_label = base::SysUTF8ToNSString(options.job_label);

        NSMutableArray* command = [NSMutableArray arrayWithCapacity:argc];
        for (int index = 0; index < argc; ++index) {
          NSString* argument = base::SysUTF8ToNSString(argv[index]);
          [command addObject:argument];
        }

        NSDictionary* job_dictionary = @{
          @LAUNCH_JOBKEY_LABEL : job_label,
          @LAUNCH_JOBKEY_PROGRAMARGUMENTS : command,
        };

        if (!options.mach_services.empty()) {
          NSMutableDictionary* mach_services = [NSMutableDictionary
              dictionaryWithCapacity:options.mach_services.size()];
          for (std::string mach_service : options.mach_services) {
            NSString* mach_service_ns = base::SysUTF8ToNSString(mach_service);
            [mach_services setObject:@YES forKey:mach_service_ns];
          }

          NSMutableDictionary* mutable_job_dictionary =
              [[job_dictionary mutableCopy] autorelease];
          [mutable_job_dictionary setObject:mach_services
                                     forKey:@LAUNCH_JOBKEY_MACHSERVICES];
          job_dictionary = mutable_job_dictionary;
        }

        CFDictionaryRef job_dictionary_cf =
            base::mac::NSToCFCast(job_dictionary);
        if (!ServiceManagementSubmitJob(job_dictionary_cf)) {
          fprintf(stderr, "%s: failed to submit job\n", me.c_str());
          return EXIT_FAILURE;
        }
      }

      return EXIT_SUCCESS;
    }

    case kOperationUnloadJob: {
      if (!ServiceManagementRemoveJob(options.job_label, true)) {
        fprintf(stderr, "%s: failed to remove job\n", me.c_str());
        return EXIT_FAILURE;
      }

      return EXIT_SUCCESS;
    }

    default: {
      ToolSupport::UsageHint(me, "must provide -L or -U");
      return EXIT_FAILURE;
    }
  }
}

}  // namespace
}  // namespace crashpad

int main(int argc, char* argv[]) {
  return crashpad::OnDemandServiceToolMain(argc, argv);
}
