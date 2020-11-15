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

#include "snapshot/mac/mach_o_image_annotations_reader.h"

#include <dlfcn.h>
#include <mach/mach.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <map>
#include <string>
#include <vector>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "client/annotation.h"
#include "client/annotation_list.h"
#include "client/crashpad_info.h"
#include "client/simple_string_dictionary.h"
#include "gtest/gtest.h"
#include "snapshot/mac/process_reader_mac.h"
#include "test/errors.h"
#include "test/mac/mach_errors.h"
#include "test/mac/mach_multiprocess.h"
#include "test/test_paths.h"
#include "util/file/file_io.h"
#include "util/mac/mac_util.h"
#include "util/mach/exc_server_variants.h"
#include "util/mach/exception_ports.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/mach/mach_message_server.h"

namespace crashpad {
namespace test {
namespace {

// \return The path to crashpad_snapshot_test_module_crashy_initializer.so
base::FilePath ModuleWithCrashyInitializer() {
  return TestPaths::BuildArtifact("snapshot",
                                  "module_crashy_initializer",
                                  TestPaths::FileType::kLoadableModule);
}

//! \return The path to the crashpad_snapshot_test_no_op executable.
base::FilePath NoOpExecutable() {
  return TestPaths::BuildArtifact(
      "snapshot", "no_op", TestPaths::FileType::kExecutable);
}

class TestMachOImageAnnotationsReader final
    : public MachMultiprocess,
      public UniversalMachExcServer::Interface {
 public:
  enum TestType {
    // Don’t crash, just test the CrashpadInfo interface.
    kDontCrash = 0,

    // The child process should crash by calling abort(). The parent verifies
    // that the system libraries set the expected annotations.
    //
    // This test verifies that the message field in crashreporter_annotations_t
    // can be recovered. Either 10.10.2 Libc-1044.1.2/stdlib/FreeBSD/abort.c
    // abort() or 10.10.2 Libc-1044.10.1/sys/_libc_fork_child.c
    // _libc_fork_child() calls CRSetCrashLogMessage() to set the message field.
    kCrashAbort,

    // The child process should crash at module initialization time, when dyld
    // will have set an annotation matching the path of the module being
    // initialized.
    //
    // This test exists to verify that the message2 field in
    // crashreporter_annotations_t can be recovered. 10.10.2
    // dyld-353.2.1/src/ImageLoaderMachO.cpp
    // ImageLoaderMachO::doInitialization() calls CRSetCrashLogMessage2() to set
    // the message2 field.
    kCrashModuleInitialization,

    // The child process should crash by setting DYLD_INSERT_LIBRARIES to
    // contain a nonexistent library. The parent verifies that dyld sets the
    // expected annotations.
    kCrashDyld,
  };

  explicit TestMachOImageAnnotationsReader(TestType test_type)
      : MachMultiprocess(),
        UniversalMachExcServer::Interface(),
        test_type_(test_type) {
    switch (test_type_) {
      case kDontCrash:
        // SetExpectedChildTermination(kTerminationNormal, EXIT_SUCCESS) is the
        // default.
        break;

      case kCrashAbort:
        SetExpectedChildTermination(kTerminationSignal, SIGABRT);
        break;

      case kCrashModuleInitialization:
        // This crash is triggered by __builtin_trap(), which shows up as
        // SIGILL.
        SetExpectedChildTermination(kTerminationSignal, SIGILL);
        break;

      case kCrashDyld:
        // Prior to 10.12, dyld fatal errors result in the execution of an
        // int3 instruction on x86 and a trap instruction on ARM, both of
        // which raise SIGTRAP. 10.9.5 dyld-239.4/src/dyldStartup.s
        // _dyld_fatal_error. This changed in 10.12 to use
        // abort_with_payload(), which appears as SIGABRT to a waiting parent.
        SetExpectedChildTermination(
            kTerminationSignal, MacOSXMinorVersion() < 12 ? SIGTRAP : SIGABRT);
        break;
    }
  }

  ~TestMachOImageAnnotationsReader() {}

  // UniversalMachExcServer::Interface:
  kern_return_t CatchMachException(exception_behavior_t behavior,
                                   exception_handler_t exception_port,
                                   thread_t thread,
                                   task_t task,
                                   exception_type_t exception,
                                   const mach_exception_data_type_t* code,
                                   mach_msg_type_number_t code_count,
                                   thread_state_flavor_t* flavor,
                                   ConstThreadState old_state,
                                   mach_msg_type_number_t old_state_count,
                                   thread_state_t new_state,
                                   mach_msg_type_number_t* new_state_count,
                                   const mach_msg_trailer_t* trailer,
                                   bool* destroy_complex_request) override {
    *destroy_complex_request = true;

    if (test_type_ != kCrashDyld) {
      // In 10.12.1 and later, the task port will not match ChildTask() in the
      // kCrashDyld case, because kCrashDyld uses execl(), which results in a
      // new task port being assigned.
      EXPECT_EQ(task, ChildTask());
    }

    // The process ID should always compare favorably.
    pid_t task_pid;
    kern_return_t kr = pid_for_task(task, &task_pid);
    EXPECT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "pid_for_task");
    EXPECT_EQ(task_pid, ChildPID());

    ProcessReaderMac process_reader;
    bool rv = process_reader.Initialize(task);
    if (!rv) {
      ADD_FAILURE();
    } else {
      const std::vector<ProcessReaderMac::Module>& modules =
          process_reader.Modules();
      std::vector<std::string> all_annotations_vector;
      for (const ProcessReaderMac::Module& module : modules) {
        if (module.reader) {
          MachOImageAnnotationsReader module_annotations_reader(
              &process_reader, module.reader, module.name);
          std::vector<std::string> module_annotations_vector =
              module_annotations_reader.Vector();
          all_annotations_vector.insert(all_annotations_vector.end(),
                                        module_annotations_vector.begin(),
                                        module_annotations_vector.end());
        } else {
          EXPECT_TRUE(module.reader);
        }
      }

      // Mac OS X 10.6 doesn’t have support for CrashReporter annotations
      // (CrashReporterClient.h), so don’t look for any special annotations in
      // that version.
      int mac_os_x_minor_version = MacOSXMinorVersion();
      if (mac_os_x_minor_version > 7) {
        EXPECT_GE(all_annotations_vector.size(), 1u);

        std::string expected_annotation;
        switch (test_type_) {
          case kCrashAbort:
            // The child process calls abort(), so the expected annotation
            // reflects this, with a string set by 10.7.5
            // Libc-763.13/stdlib/abort-fbsd.c abort(). This string is still
            // present in 10.9.5 Libc-997.90.3/stdlib/FreeBSD/abort.c abort(),
            // but because abort() tests to see if a message is already set and
            // something else in Libc will have set a message, this string is
            // not the expectation on 10.9 or higher. Instead, after fork(), the
            // child process has a message indicating that a fork() without
            // exec() occurred. See 10.9.5 Libc-997.90.3/sys/_libc_fork_child.c
            // _libc_fork_child().
            expected_annotation =
                mac_os_x_minor_version <= 8
                    ? "abort() called"
                    : "crashed on child side of fork pre-exec";
            break;

          case kCrashModuleInitialization:
            // This message is set by dyld-353.2.1/src/ImageLoaderMachO.cpp
            // ImageLoaderMachO::doInitialization().
            expected_annotation = ModuleWithCrashyInitializer().value();
            break;

          case kCrashDyld:
            // This is independent of dyld’s error_string, which is tested
            // below.
            expected_annotation = "dyld: launch, loading dependent libraries";
            break;

          default:
            ADD_FAILURE();
            break;
        }

        bool found = false;
        for (const std::string& annotation : all_annotations_vector) {
          // Look for the expectation as a leading susbtring, because the actual
          // string that dyld uses will have the contents of the
          // DYLD_INSERT_LIBRARIES environment variable appended to it on OS X
          // 10.10.
          if (annotation.substr(0, expected_annotation.length()) ==
                  expected_annotation) {
            found = true;
            break;
          }
        }
        EXPECT_TRUE(found) << expected_annotation;
      }

      // dyld exposes its error_string at least as far back as Mac OS X 10.4.
      if (test_type_ == kCrashDyld) {
        static constexpr char kExpectedAnnotation[] =
            "could not load inserted library";
        size_t expected_annotation_length = strlen(kExpectedAnnotation);
        bool found = false;
        for (const std::string& annotation : all_annotations_vector) {
          // Look for the expectation as a leading substring, because the actual
          // string will contain the library’s pathname and, on OS X 10.9 and
          // later, a reason.
          if (annotation.substr(0, expected_annotation_length) ==
                  kExpectedAnnotation) {
            found = true;
            break;
          }
        }

        EXPECT_TRUE(found) << kExpectedAnnotation;
      }
    }

    ExcServerCopyState(
        behavior, old_state, old_state_count, new_state, new_state_count);
    return ExcServerSuccessfulReturnValue(exception, behavior, false);
  }

 private:
  // MachMultiprocess:

  void MachMultiprocessParent() override {
    ProcessReaderMac process_reader;
    ASSERT_TRUE(process_reader.Initialize(ChildTask()));

    // Wait for the child process to indicate that it’s done setting up its
    // annotations via the CrashpadInfo interface.
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));

    // Verify the “simple map” and object-based annotations set via the
    // CrashpadInfo interface.
    const std::vector<ProcessReaderMac::Module>& modules =
        process_reader.Modules();
    std::map<std::string, std::string> all_annotations_simple_map;
    std::vector<AnnotationSnapshot> all_annotations;
    for (const ProcessReaderMac::Module& module : modules) {
      MachOImageAnnotationsReader module_annotations_reader(
          &process_reader, module.reader, module.name);
      std::map<std::string, std::string> module_annotations_simple_map =
          module_annotations_reader.SimpleMap();
      all_annotations_simple_map.insert(module_annotations_simple_map.begin(),
                                        module_annotations_simple_map.end());

      std::vector<AnnotationSnapshot> annotations =
          module_annotations_reader.AnnotationsList();
      all_annotations.insert(
          all_annotations.end(), annotations.begin(), annotations.end());
    }

    EXPECT_GE(all_annotations_simple_map.size(), 5u);
    EXPECT_EQ(all_annotations_simple_map["#TEST# pad"], "crash");
    EXPECT_EQ(all_annotations_simple_map["#TEST# key"], "value");
    EXPECT_EQ(all_annotations_simple_map["#TEST# x"], "y");
    EXPECT_EQ(all_annotations_simple_map["#TEST# longer"], "shorter");
    EXPECT_EQ(all_annotations_simple_map["#TEST# empty_value"], "");

    EXPECT_EQ(all_annotations.size(), 3u);
    bool saw_same_name_3 = false, saw_same_name_4 = false;
    for (const auto& annotation : all_annotations) {
      EXPECT_EQ(annotation.type,
                static_cast<uint16_t>(Annotation::Type::kString));
      std::string value(reinterpret_cast<const char*>(annotation.value.data()),
                        annotation.value.size());

      if (annotation.name == "#TEST# one") {
        EXPECT_EQ(value, "moocow");
      } else if (annotation.name == "#TEST# same-name") {
        if (value == "same-name 3") {
          EXPECT_FALSE(saw_same_name_3);
          saw_same_name_3 = true;
        } else if (value == "same-name 4") {
          EXPECT_FALSE(saw_same_name_4);
          saw_same_name_4 = true;
        } else {
          ADD_FAILURE() << "unexpected annotation value " << value;
        }
      } else {
        ADD_FAILURE() << "unexpected annotation " << annotation.name;
      }
    }

    // Tell the child process that it’s permitted to crash.
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));

    if (test_type_ != kDontCrash) {
      // Handle the child’s crash. Further validation will be done in
      // CatchMachException().
      UniversalMachExcServer universal_mach_exc_server(this);

      mach_msg_return_t mr =
          MachMessageServer::Run(&universal_mach_exc_server,
                                 LocalPort(),
                                 MACH_MSG_OPTION_NONE,
                                 MachMessageServer::kOneShot,
                                 MachMessageServer::kReceiveLargeError,
                                 kMachMessageTimeoutWaitIndefinitely);
      EXPECT_EQ(mr, MACH_MSG_SUCCESS)
          << MachErrorMessage(mr, "MachMessageServer::Run");
    }
  }

  void MachMultiprocessChild() override {
    CrashpadInfo* crashpad_info = CrashpadInfo::GetCrashpadInfo();

    // This is “leaked” to crashpad_info.
    SimpleStringDictionary* simple_annotations = new SimpleStringDictionary();
    simple_annotations->SetKeyValue("#TEST# pad", "break");
    simple_annotations->SetKeyValue("#TEST# key", "value");
    simple_annotations->SetKeyValue("#TEST# pad", "crash");
    simple_annotations->SetKeyValue("#TEST# x", "y");
    simple_annotations->SetKeyValue("#TEST# longer", "shorter");
    simple_annotations->SetKeyValue("#TEST# empty_value", "");

    crashpad_info->set_simple_annotations(simple_annotations);

    AnnotationList::Register();  // This is “leaked” to crashpad_info.

    static StringAnnotation<32> test_annotation_one{"#TEST# one"};
    static StringAnnotation<32> test_annotation_two{"#TEST# two"};
    static StringAnnotation<32> test_annotation_three{"#TEST# same-name"};
    static StringAnnotation<32> test_annotation_four{"#TEST# same-name"};

    test_annotation_one.Set("moocow");
    test_annotation_two.Set("this will be cleared");
    test_annotation_three.Set("same-name 3");
    test_annotation_four.Set("same-name 4");
    test_annotation_two.Clear();

    // Tell the parent that the environment has been set up.
    char c = '\0';
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));

    // Wait for the parent to indicate that it’s safe to crash.
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));

    // Direct an exception message to the exception server running in the
    // parent.
    ExceptionPorts exception_ports(ExceptionPorts::kTargetTypeTask,
                                   mach_task_self());
    ASSERT_TRUE(exception_ports.SetExceptionPort(
        EXC_MASK_CRASH, RemotePort(), EXCEPTION_DEFAULT, THREAD_STATE_NONE));

    switch (test_type_) {
      case kDontCrash: {
        break;
      }

      case kCrashAbort: {
        abort();
        break;
      }

      case kCrashModuleInitialization: {
        // Load a module that crashes while executing a module initializer.
        void* dl_handle = dlopen(ModuleWithCrashyInitializer().value().c_str(),
                                 RTLD_LAZY | RTLD_LOCAL);

        // This should have crashed in the dlopen(). If dlopen() failed, the
        // ASSERT_NE() will show the message. If it succeeded without crashing,
        // the FAIL() will fail the test.
        ASSERT_NE(dl_handle, nullptr) << dlerror();
        FAIL();
        break;
      }

      case kCrashDyld: {
        // Set DYLD_INSERT_LIBRARIES to contain a library that does not exist.
        // Unable to load it, dyld will abort with a fatal error.
        ASSERT_EQ(
            setenv(
                "DYLD_INSERT_LIBRARIES", "/var/empty/NoDirectory/NoLibrary", 1),
            0)
            << ErrnoMessage("setenv");

        // The actual executable doesn’t matter very much, because dyld won’t
        // ever launch it. It just needs to be an executable that uses dyld as
        // its LC_LOAD_DYLINKER (all normal executables do). A custom no-op
        // executable is provided because DYLD_INSERT_LIBRARIES does not work
        // with system executables on OS X 10.11 due to System Integrity
        // Protection.
        base::FilePath no_op_executable = NoOpExecutable();
        ASSERT_EQ(execl(no_op_executable.value().c_str(),
                        no_op_executable.BaseName().value().c_str(),
                        nullptr),
                  0)
            << ErrnoMessage("execl");
        break;
      }

      default:
        break;
    }
  }

  TestType test_type_;

  DISALLOW_COPY_AND_ASSIGN(TestMachOImageAnnotationsReader);
};

TEST(MachOImageAnnotationsReader, DontCrash) {
  TestMachOImageAnnotationsReader test_mach_o_image_annotations_reader(
      TestMachOImageAnnotationsReader::kDontCrash);
  test_mach_o_image_annotations_reader.Run();
}

TEST(MachOImageAnnotationsReader, CrashAbort) {
  TestMachOImageAnnotationsReader test_mach_o_image_annotations_reader(
      TestMachOImageAnnotationsReader::kCrashAbort);
  test_mach_o_image_annotations_reader.Run();
}

TEST(MachOImageAnnotationsReader, CrashModuleInitialization) {
  TestMachOImageAnnotationsReader test_mach_o_image_annotations_reader(
      TestMachOImageAnnotationsReader::kCrashModuleInitialization);
  test_mach_o_image_annotations_reader.Run();
}

TEST(MachOImageAnnotationsReader, CrashDyld) {
  TestMachOImageAnnotationsReader test_mach_o_image_annotations_reader(
      TestMachOImageAnnotationsReader::kCrashDyld);
  test_mach_o_image_annotations_reader.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
