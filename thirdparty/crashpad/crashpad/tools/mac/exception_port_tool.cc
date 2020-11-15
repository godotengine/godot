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

#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "tools/tool_support.h"
#include "util/mach/exception_ports.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/symbolic_constants_mach.h"
#include "util/mach/task_for_pid.h"
#include "util/posix/drop_privileges.h"
#include "util/stdlib/string_number_conversion.h"

namespace crashpad {
namespace {

//! \brief Manages a pool of Mach send rights, deallocating all send rights upon
//!     destruction.
//!
//! This class effectively implements what a vector of
//! base::mac::ScopedMachSendRight objects would be.
//!
//! The various “show” operations performed by this program display Mach ports
//! by their names as they are known in this task. For this to be useful, rights
//! to the same ports must have consistent names across successive calls. This
//! cannot be guaranteed if the rights are deallocated as soon as they are used,
//! because if that deallocation causes the task to lose its last right to a
//! port, subsequently regaining a right to the same port would cause it to be
//! known by a new name in this task.
//!
//! Instead of immediately deallocating send rights that are used for display,
//! they can be added to this pool. The pool collects send rights, ensuring that
//! they remain alive in this task, and that subsequent calls that obtain the
//! same rights cause them to be known by the same name. All rights are
//! deallocated upon destruction.
class MachSendRightPool {
 public:
  MachSendRightPool()
      : send_rights_() {
  }

  ~MachSendRightPool() {
    for (mach_port_t send_right : send_rights_) {
      kern_return_t kr = mach_port_deallocate(mach_task_self(), send_right);
      MACH_LOG_IF(ERROR, kr != KERN_SUCCESS, kr) << "mach_port_deallocate";
    }
  }

  //! \brief Adds a send right to the pool.
  //!
  //! \param[in] send_right The send right to be added. The pool object takes
  //!     its own reference to the send right, which remains valid until the
  //!     pool object is destroyed. The caller remains responsible for its
  //!     reference to the send right.
  //!
  //! It is possible and in fact likely that one pool will wind up owning the
  //! same send right multiple times. This is acceptable, because send rights
  //! are reference-counted.
  void AddSendRight(mach_port_t send_right) {
    kern_return_t kr = mach_port_mod_refs(mach_task_self(),
                                          send_right,
                                          MACH_PORT_RIGHT_SEND,
                                          1);
    MACH_CHECK(kr == KERN_SUCCESS, kr) << "mach_port_mod_refs";

    send_rights_.push_back(send_right);
  }

 private:
  std::vector<mach_port_t> send_rights_;

  DISALLOW_COPY_AND_ASSIGN(MachSendRightPool);
};

struct ExceptionHandlerDescription {
  ExceptionPorts::TargetType target_type;
  exception_mask_t mask;
  exception_behavior_t behavior;
  thread_state_flavor_t flavor;
  std::string handler;
};

constexpr char kHandlerNull[] = "NULL";
constexpr char kHandlerBootstrapColon[] = "bootstrap:";

// Populates |description| based on a textual representation in
// |handler_string_ro|, returning true on success and false on failure (parse
// error). The --help string describes the format of |handler_string_ro|.
// Briefly, it is a comma-separated string that allows the members of
// |description| to be specified as "field=value". Values for "target" can be
// "host", "task", or "thread"; values for "handler" are of the form
// "bootstrap:service_name" where service_name will be looked up with the
// bootstrap server; and values for the other fields are interpreted by
// SymbolicConstantsMach.
bool ParseHandlerString(const char* handler_string_ro,
                        ExceptionHandlerDescription* description) {
  static constexpr char kTargetEquals[] = "target=";
  static constexpr char kMaskEquals[] = "mask=";
  static constexpr char kBehaviorEquals[] = "behavior=";
  static constexpr char kFlavorEquals[] = "flavor=";
  static constexpr char kHandlerEquals[] = "handler=";

  std::string handler_string(handler_string_ro);
  char* handler_string_c = &handler_string[0];

  char* token;
  while ((token = strsep(&handler_string_c, ",")) != nullptr) {
    if (strncmp(token, kTargetEquals, strlen(kTargetEquals)) == 0) {
      const char* value = token + strlen(kTargetEquals);
      if (strcmp(value, "host") == 0) {
        description->target_type = ExceptionPorts::kTargetTypeHost;
      } else if (strcmp(value, "task") == 0) {
        description->target_type = ExceptionPorts::kTargetTypeTask;
      } else if (strcmp(value, "thread") == 0) {
        description->target_type = ExceptionPorts::kTargetTypeThread;
      } else {
        return false;
      }
    } else if (strncmp(token, kMaskEquals, strlen(kMaskEquals)) == 0) {
      const char* value = token + strlen(kMaskEquals);
      if (!StringToExceptionMask(
              value,
              kAllowFullName | kAllowShortName | kAllowNumber | kAllowOr,
              &description->mask)) {
        return false;
      }
    } else if (strncmp(token, kBehaviorEquals, strlen(kBehaviorEquals)) == 0) {
      const char* value = token + strlen(kBehaviorEquals);
      if (!StringToExceptionBehavior(
              value,
              kAllowFullName | kAllowShortName | kAllowNumber,
              &description->behavior)) {
        return false;
      }
    } else if (strncmp(token, kFlavorEquals, strlen(kFlavorEquals)) == 0) {
      const char* value = token + strlen(kFlavorEquals);
      if (!StringToThreadStateFlavor(
              value,
              kAllowFullName | kAllowShortName | kAllowNumber,
              &description->flavor)) {
        return false;
      }
    } else if (strncmp(token, kHandlerEquals, strlen(kHandlerEquals)) == 0) {
      const char* value = token + strlen(kHandlerEquals);
      if (strcmp(value, kHandlerNull) != 0 &&
          strncmp(value,
                  kHandlerBootstrapColon,
                  strlen(kHandlerBootstrapColon)) != 0) {
        return false;
      }
      description->handler = std::string(value);
    } else {
      return false;
    }
  }

  return true;
}

// ShowExceptionPorts() shows handlers as numeric mach_port_t values, which are
// opaque and meaningless on their own. ShowBootstrapService() can be used to
// look up a service with the bootstrap server by name and show its mach_port_t
// value, which can then be associated with handlers shown by
// ShowExceptionPorts(). Any send rights obtained by this function are added to
// |mach_send_right_pool|.
void ShowBootstrapService(const std::string& service_name,
                          MachSendRightPool* mach_send_right_pool) {
  base::mac::ScopedMachSendRight service_port(BootstrapLookUp(service_name));
  if (service_port == kMachPortNull) {
    return;
  }

  mach_send_right_pool->AddSendRight(service_port.get());

  printf("service %s %#x\n", service_name.c_str(), service_port.get());
}

// Prints information about all exception ports known for |exception_ports|. If
// |numeric| is true, all information is printed in numeric form, otherwise, it
// will be converted to symbolic constants where possible by
// SymbolicConstantsMach. If |is_new| is true, information will be presented as
// “new exception ports”, indicating that they show the state of the exception
// ports after SetExceptionPort() has been called. Any send rights obtained by
// this function are added to |mach_send_right_pool|.
void ShowExceptionPorts(const ExceptionPorts& exception_ports,
                        bool numeric,
                        bool is_new,
                        MachSendRightPool* mach_send_right_pool) {
  const char* target_name = exception_ports.TargetTypeName();

  ExceptionPorts::ExceptionHandlerVector handlers;
  if (!exception_ports.GetExceptionPorts(ExcMaskValid(), &handlers)) {
    return;
  }

  const char* age_name = is_new ? "new " : "";

  if (handlers.empty()) {
    printf("no %s%s exception ports\n", age_name, target_name);
  }

  for (size_t port_index = 0; port_index < handlers.size(); ++port_index) {
    mach_send_right_pool->AddSendRight(handlers[port_index].port);

    if (numeric) {
      printf(
          "%s%s exception port %zu, mask %#x, port %#x, "
          "behavior %#x, flavor %u\n",
          age_name,
          target_name,
          port_index,
          handlers[port_index].mask,
          handlers[port_index].port,
          handlers[port_index].behavior,
          handlers[port_index].flavor);
    } else {
      std::string mask_string = ExceptionMaskToString(
          handlers[port_index].mask, kUseShortName | kUnknownIsEmpty | kUseOr);
      if (mask_string.empty()) {
        mask_string.assign("?");
      }

      std::string behavior_string = ExceptionBehaviorToString(
          handlers[port_index].behavior, kUseShortName | kUnknownIsEmpty);
      if (behavior_string.empty()) {
        behavior_string.assign("?");
      }

      std::string flavor_string = ThreadStateFlavorToString(
          handlers[port_index].flavor, kUseShortName | kUnknownIsEmpty);
      if (flavor_string.empty()) {
        flavor_string.assign("?");
      }

      printf(
          "%s%s exception port %zu, mask %#x (%s), port %#x, "
          "behavior %#x (%s), flavor %u (%s)\n",
          age_name,
          target_name,
          port_index,
          handlers[port_index].mask,
          mask_string.c_str(),
          handlers[port_index].port,
          handlers[port_index].behavior,
          behavior_string.c_str(),
          handlers[port_index].flavor,
          flavor_string.c_str());
    }
  }
}

// Sets the exception port for |target_port|, a send right to a thread, task, or
// host port, to |description|, which identifies what type of port |target_port|
// is and describes an exception port to be set. Returns true on success.
//
// This function may be called more than once if setting different handlers is
// desired.
bool SetExceptionPort(const ExceptionHandlerDescription* description,
                      mach_port_t target_port) {
  base::mac::ScopedMachSendRight service_port;
  if (description->handler.compare(
          0, strlen(kHandlerBootstrapColon), kHandlerBootstrapColon) == 0) {
    const char* service_name =
        description->handler.c_str() + strlen(kHandlerBootstrapColon);
    service_port = BootstrapLookUp(service_name);
    if (service_port == kMachPortNull) {
      return false;
    }

    // The service port doesn’t need to be added to a MachSendRightPool because
    // it’s not used for display at all. ScopedMachSendRight is sufficient.
  } else if (description->handler != kHandlerNull) {
    return false;
  }

  ExceptionPorts exception_ports(description->target_type, target_port);
  if (!exception_ports.SetExceptionPort(description->mask,
                                        service_port.get(),
                                        description->behavior,
                                        description->flavor)) {
    return false;
  }

  return true;
}

void Usage(const std::string& me) {
  fprintf(stderr,
"Usage: %s [OPTION]... [COMMAND [ARG]...]\n"
"View and change Mach exception ports, and run COMMAND if supplied.\n"
"\n"
"  -s, --set-handler=DESCRIPTION  set an exception port to DESCRIPTION, see below\n"
"      --show-bootstrap=SERVICE   look up and display a service registered with\n"
"                                 the bootstrap server\n"
"  -p, --pid=PID                  operate on PID instead of the current task\n"
"  -h, --show-host                display original host exception ports\n"
"  -t, --show-task                display original task exception ports\n"
"      --show-thread              display original thread exception ports\n"
"  -H, --show-new-host            display modified host exception ports\n"
"  -T, --show-new-task            display modified task exception ports\n"
"      --show-new-thread          display modified thread exception ports\n"
"  -n, --numeric                  display values numerically, not symbolically\n"
"      --help                     display this help and exit\n"
"      --version                  output version information and exit\n"
"\n"
"Any operations on host exception ports require superuser permissions.\n"
"\n"
"DESCRIPTION is formatted as a comma-separated sequence of tokens, where each\n"
"token consists of a key and value separated by an equals sign. Available keys:\n"
"  target    which target's exception ports to set: host, task, or thread\n"
"  mask      the mask of exception types to handle: CRASH, ALL, or others\n"
"  behavior  the specific exception handler routine to call: DEFAULT, STATE,\n"
"            or STATE_IDENTITY, possibly with MACH_EXCEPTION_CODES.\n"
"  flavor    the thread state flavor passed to the handler: architecture-specific\n"
"  handler   the exception handler: NULL or bootstrap:SERVICE, indicating that\n"
"            the handler should be looked up with the bootstrap server\n"
"The default DESCRIPTION is\n"
"  target=task,mask=CRASH,behavior=DEFAULT|MACH,flavor=NONE,handler=NULL\n",
          me.c_str());
  ToolSupport::UsageTail(me);
}

int ExceptionPortToolMain(int argc, char* argv[]) {
  const std::string me(basename(argv[0]));

  enum ExitCode {
    kExitSuccess = EXIT_SUCCESS,

    // To differentiate this tool’s errors from errors in the programs it execs,
    // use a high exit code for ordinary failures instead of EXIT_FAILURE. This
    // is the same rationale for using the distinct exit codes for exec
    // failures.
    kExitFailure = 125,

    // Like env, use exit code 126 if the program was found but could not be
    // invoked, and 127 if it could not be found.
    // http://pubs.opengroup.org/onlinepubs/9699919799/utilities/env.html
    kExitExecFailure = 126,
    kExitExecENOENT = 127,
  };

  enum OptionFlags {
    // “Short” (single-character) options.
    kOptionSetPort = 's',
    kOptionPid = 'p',
    kOptionShowHost = 'h',
    kOptionShowTask = 't',
    kOptionShowNewHost = 'H',
    kOptionShowNewTask = 'T',
    kOptionNumeric = 'n',

    // Long options without short equivalents.
    kOptionLastChar = 255,
    kOptionShowBootstrap,
    kOptionShowThread,
    kOptionShowNewThread,

    // Standard options.
    kOptionHelp = -2,
    kOptionVersion = -3,
  };

  struct {
    std::vector<const char*> show_bootstrap;
    std::vector<ExceptionHandlerDescription> set_handler;
    pid_t pid;
    task_t alternate_task;
    bool show_host;
    bool show_task;
    bool show_thread;
    bool show_new_host;
    bool show_new_task;
    bool show_new_thread;
    bool numeric;
  } options = {};

  static constexpr option long_options[] = {
      {"set-handler", required_argument, nullptr, kOptionSetPort},
      {"show-bootstrap", required_argument, nullptr, kOptionShowBootstrap},
      {"pid", required_argument, nullptr, kOptionPid},
      {"show-host", no_argument, nullptr, kOptionShowHost},
      {"show-task", no_argument, nullptr, kOptionShowTask},
      {"show-thread", no_argument, nullptr, kOptionShowThread},
      {"show-new-host", no_argument, nullptr, kOptionShowNewHost},
      {"show-new-task", no_argument, nullptr, kOptionShowNewTask},
      {"show-new-thread", no_argument, nullptr, kOptionShowNewThread},
      {"numeric", no_argument, nullptr, kOptionNumeric},
      {"help", no_argument, nullptr, kOptionHelp},
      {"version", no_argument, nullptr, kOptionVersion},
      {nullptr, 0, nullptr, 0},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "+s:p:htHTn", long_options, nullptr)) !=
         -1) {
    switch (opt) {
      case kOptionSetPort: {
        options.set_handler.push_back({});
        ExceptionHandlerDescription* description = &options.set_handler.back();
        description->target_type = ExceptionPorts::kTargetTypeTask;
        description->mask = EXC_MASK_CRASH;
        description->behavior = EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES;
        description->flavor = THREAD_STATE_NONE;
        description->handler = "NULL";
        if (!ParseHandlerString(optarg, description)) {
          fprintf(stderr,
                  "%s: invalid exception handler: %s\n",
                  me.c_str(),
                  optarg);
          return kExitFailure;
        }
        break;
      }
      case kOptionShowBootstrap:
        options.show_bootstrap.push_back(optarg);
        break;
      case kOptionPid:
        if (!StringToNumber(optarg, &options.pid)) {
          fprintf(stderr, "%s: invalid pid: %s\n", me.c_str(), optarg);
          return kExitFailure;
        }
        break;
      case kOptionShowHost:
        options.show_host = true;
        break;
      case kOptionShowTask:
        options.show_task = true;
        break;
      case kOptionShowThread:
        options.show_thread = true;
        break;
      case kOptionShowNewHost:
        options.show_new_host = true;
        break;
      case kOptionShowNewTask:
        options.show_new_task = true;
        break;
      case kOptionShowNewThread:
        options.show_new_thread = true;
        break;
      case kOptionNumeric:
        options.numeric = true;
        break;
      case kOptionHelp:
        Usage(me);
        return kExitSuccess;
      case kOptionVersion:
        ToolSupport::Version(me);
        return kExitSuccess;
      default:
        ToolSupport::UsageHint(me, nullptr);
        return kExitFailure;
    }
  }
  argc -= optind;
  argv += optind;

  if (options.show_bootstrap.empty() && !options.show_host &&
      !options.show_task && !options.show_thread &&
      options.set_handler.empty() && argc == 0) {
    ToolSupport::UsageHint(me, "nothing to do");
    return kExitFailure;
  }

  base::mac::ScopedMachSendRight alternate_task_owner;
  if (options.pid) {
    if (argc) {
      ToolSupport::UsageHint(me, "cannot combine -p with COMMAND");
      return kExitFailure;
    }

    options.alternate_task = TaskForPID(options.pid);
    if (options.alternate_task == TASK_NULL) {
      return kExitFailure;
    }
    alternate_task_owner.reset(options.alternate_task);
  }

  // This tool may have been installed as a setuid binary so that TaskForPID()
  // could succeed. Drop any privileges now that they’re no longer necessary.
  DropPrivileges();

  MachSendRightPool mach_send_right_pool;

  // Show bootstrap services requested.
  for (const char* service : options.show_bootstrap) {
    ShowBootstrapService(service, &mach_send_right_pool);
  }

  // Show the original exception ports.
  if (options.show_host) {
    ShowExceptionPorts(
        ExceptionPorts(ExceptionPorts::kTargetTypeHost, HOST_NULL),
        options.numeric,
        false,
        &mach_send_right_pool);
  }
  if (options.show_task) {
    ShowExceptionPorts(
        ExceptionPorts(ExceptionPorts::kTargetTypeTask, options.alternate_task),
        options.numeric,
        false,
        &mach_send_right_pool);
  }
  if (options.show_thread) {
    ShowExceptionPorts(
        ExceptionPorts(ExceptionPorts::kTargetTypeThread, THREAD_NULL),
        options.numeric,
        false,
        &mach_send_right_pool);
  }

  if (!options.set_handler.empty()) {
    // Set new exception handlers.
    for (ExceptionHandlerDescription description : options.set_handler) {
      if (!SetExceptionPort(
              &description,
              description.target_type == ExceptionPorts::kTargetTypeTask
                  ? options.alternate_task
                  : TASK_NULL)) {
        return kExitFailure;
      }
    }

    // Show changed exception ports.
    if (options.show_new_host) {
      ShowExceptionPorts(
          ExceptionPorts(ExceptionPorts::kTargetTypeHost, HOST_NULL),
          options.numeric,
          true,
          &mach_send_right_pool);
    }
    if (options.show_new_task) {
      ShowExceptionPorts(
          ExceptionPorts(ExceptionPorts::kTargetTypeTask,
                         options.alternate_task),
          options.numeric,
          true,
          &mach_send_right_pool);
    }
    if (options.show_new_thread) {
      ShowExceptionPorts(
          ExceptionPorts(ExceptionPorts::kTargetTypeThread, THREAD_NULL),
          options.numeric,
          true,
          &mach_send_right_pool);
    }
  }

  if (argc) {
    // Using the remaining arguments, start a new program with the new set of
    // exception ports in effect.
    execvp(argv[0], argv);
    PLOG(ERROR) << "execvp " << argv[0];
    return errno == ENOENT ? kExitExecENOENT : kExitExecFailure;
  }

  return kExitSuccess;
}

}  // namespace
}  // namespace crashpad

int main(int argc, char* argv[]) {
  return crashpad::ExceptionPortToolMain(argc, argv);
}
