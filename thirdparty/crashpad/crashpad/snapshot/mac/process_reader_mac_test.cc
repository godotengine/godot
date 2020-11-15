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

#include "snapshot/mac/process_reader_mac.h"

#include <AvailabilityMacros.h>
#include <OpenCL/opencl.h>
#include <mach-o/dyld.h>
#include <mach-o/dyld_images.h>
#include <mach/mach.h>
#include <string.h>
#include <sys/stat.h>

#include <map>
#include <utility>

#include "base/logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/posix/eintr_wrapper.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "snapshot/mac/mach_o_image_reader.h"
#include "snapshot/mac/mach_o_image_segment_reader.h"
#include "test/errors.h"
#include "test/mac/dyld.h"
#include "test/mac/mach_errors.h"
#include "test/mac/mach_multiprocess.h"
#include "util/file/file_io.h"
#include "util/mac/mac_util.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/from_pointer_cast.h"
#include "util/synchronization/semaphore.h"

namespace crashpad {
namespace test {
namespace {

constexpr char kDyldPath[] = "/usr/lib/dyld";

TEST(ProcessReaderMac, SelfBasic) {
  ProcessReaderMac process_reader;
  ASSERT_TRUE(process_reader.Initialize(mach_task_self()));

#if !defined(ARCH_CPU_64_BITS)
  EXPECT_FALSE(process_reader.Is64Bit());
#else
  EXPECT_TRUE(process_reader.Is64Bit());
#endif

  EXPECT_EQ(process_reader.ProcessID(), getpid());
  EXPECT_EQ(process_reader.ParentProcessID(), getppid());

  static constexpr char kTestMemory[] = "Some test memory";
  char buffer[arraysize(kTestMemory)];
  ASSERT_TRUE(process_reader.Memory()->Read(
      FromPointerCast<mach_vm_address_t>(kTestMemory),
      sizeof(kTestMemory),
      &buffer));
  EXPECT_STREQ(kTestMemory, buffer);
}

constexpr char kTestMemory[] = "Read me from another process";

class ProcessReaderChild final : public MachMultiprocess {
 public:
  ProcessReaderChild() : MachMultiprocess() {}

  ~ProcessReaderChild() {}

 private:
  void MachMultiprocessParent() override {
    ProcessReaderMac process_reader;
    ASSERT_TRUE(process_reader.Initialize(ChildTask()));

#if !defined(ARCH_CPU_64_BITS)
    EXPECT_FALSE(process_reader.Is64Bit());
#else
    EXPECT_TRUE(process_reader.Is64Bit());
#endif

    EXPECT_EQ(process_reader.ParentProcessID(), getpid());
    EXPECT_EQ(process_reader.ProcessID(), ChildPID());

    FileHandle read_handle = ReadPipeHandle();

    mach_vm_address_t address;
    CheckedReadFileExactly(read_handle, &address, sizeof(address));

    std::string read_string;
    ASSERT_TRUE(process_reader.Memory()->ReadCString(address, &read_string));
    EXPECT_EQ(read_string, kTestMemory);
  }

  void MachMultiprocessChild() override {
    FileHandle write_handle = WritePipeHandle();

    mach_vm_address_t address = FromPointerCast<mach_vm_address_t>(kTestMemory);
    CheckedWriteFile(write_handle, &address, sizeof(address));

    // Wait for the parent to signal that it’s OK to exit by closing its end of
    // the pipe.
    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderChild);
};

TEST(ProcessReaderMac, ChildBasic) {
  ProcessReaderChild process_reader_child;
  process_reader_child.Run();
}

// Returns a thread ID given a pthread_t. This wraps pthread_threadid_np() but
// that function has a cumbersome interface because it returns a success value.
// This function CHECKs success and returns the thread ID directly.
uint64_t PthreadToThreadID(pthread_t pthread) {
  uint64_t thread_id;
  int rv = pthread_threadid_np(pthread, &thread_id);
  CHECK_EQ(rv, 0);
  return thread_id;
}

TEST(ProcessReaderMac, SelfOneThread) {
  ProcessReaderMac process_reader;
  ASSERT_TRUE(process_reader.Initialize(mach_task_self()));

  const std::vector<ProcessReaderMac::Thread>& threads =
      process_reader.Threads();

  // If other tests ran in this process previously, threads may have been
  // created and may still be running. This check must look for at least one
  // thread, not exactly one thread.
  ASSERT_GE(threads.size(), 1u);

  EXPECT_EQ(threads[0].id, PthreadToThreadID(pthread_self()));

  thread_t thread_self = MachThreadSelf();
  EXPECT_EQ(threads[0].port, thread_self);

  EXPECT_EQ(threads[0].suspend_count, 0);
}

class TestThreadPool {
 public:
  struct ThreadExpectation {
    mach_vm_address_t stack_address;
    int suspend_count;
  };

  TestThreadPool() : thread_infos_() {}

  // Resumes suspended threads, signals each thread’s exit semaphore asking it
  // to exit, and joins each thread, blocking until they have all exited.
  ~TestThreadPool() {
    for (const auto& thread_info : thread_infos_) {
      thread_t thread_port = pthread_mach_thread_np(thread_info->pthread);
      while (thread_info->suspend_count > 0) {
        kern_return_t kr = thread_resume(thread_port);
        EXPECT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "thread_resume");
        --thread_info->suspend_count;
      }
    }

    for (const auto& thread_info : thread_infos_) {
      thread_info->exit_semaphore.Signal();
    }

    for (const auto& thread_info : thread_infos_) {
      int rv = pthread_join(thread_info->pthread, nullptr);
      CHECK_EQ(0, rv);
    }
  }

  // Starts |thread_count| threads and waits on each thread’s ready semaphore,
  // so that when this function returns, all threads have been started and have
  // all run to the point that they’ve signalled that they are ready.
  void StartThreads(size_t thread_count) {
    ASSERT_TRUE(thread_infos_.empty());

    for (size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
      thread_infos_.push_back(std::make_unique<ThreadInfo>());
      ThreadInfo* thread_info = thread_infos_.back().get();

      int rv = pthread_create(
          &thread_info->pthread, nullptr, ThreadMain, thread_info);
      ASSERT_EQ(rv, 0);
    }

    for (const auto& thread_info : thread_infos_) {
      thread_info->ready_semaphore.Wait();
    }

    // If present, suspend the thread at indices 1 through 3 the same number of
    // times as their index. This tests reporting of suspend counts.
    for (size_t thread_index = 1;
         thread_index < thread_infos_.size() && thread_index < 4;
         ++thread_index) {
      thread_t thread_port =
          pthread_mach_thread_np(thread_infos_[thread_index]->pthread);
      for (size_t suspend_count = 0; suspend_count < thread_index;
           ++suspend_count) {
        kern_return_t kr = thread_suspend(thread_port);
        EXPECT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "thread_suspend");
        if (kr == KERN_SUCCESS) {
          ++thread_infos_[thread_index]->suspend_count;
        }
      }
    }
  }

  uint64_t GetThreadInfo(size_t thread_index, ThreadExpectation* expectation) {
    CHECK_LT(thread_index, thread_infos_.size());

    const auto& thread_info = thread_infos_[thread_index];
    expectation->stack_address = thread_info->stack_address;
    expectation->suspend_count = thread_info->suspend_count;

    return PthreadToThreadID(thread_info->pthread);
  }

 private:
  struct ThreadInfo {
    ThreadInfo()
        : pthread(nullptr),
          stack_address(0),
          ready_semaphore(0),
          exit_semaphore(0),
          suspend_count(0) {}

    ~ThreadInfo() {}

    // The thread’s ID, set at the time the thread is created.
    pthread_t pthread;

    // An address somewhere within the thread’s stack. The thread sets this in
    // its ThreadMain().
    mach_vm_address_t stack_address;

    // The worker thread signals ready_semaphore to indicate that it’s done
    // setting up its ThreadInfo structure. The main thread waits on this
    // semaphore before using any data that the worker thread is responsible for
    // setting.
    Semaphore ready_semaphore;

    // The worker thread waits on exit_semaphore to determine when it’s safe to
    // exit. The main thread signals exit_semaphore when it no longer needs the
    // worker thread.
    Semaphore exit_semaphore;

    // The thread’s suspend count.
    int suspend_count;
  };

  static void* ThreadMain(void* argument) {
    ThreadInfo* thread_info = static_cast<ThreadInfo*>(argument);

    thread_info->stack_address =
        FromPointerCast<mach_vm_address_t>(&thread_info);

    thread_info->ready_semaphore.Signal();
    thread_info->exit_semaphore.Wait();

    // Check this here after everything’s known to be synchronized, otherwise
    // there’s a race between the parent thread storing this thread’s pthread_t
    // in thread_info_pthread and this thread starting and attempting to access
    // it.
    CHECK_EQ(pthread_self(), thread_info->pthread);

    return nullptr;
  }

  // This is a vector of pointers because the address of a ThreadInfo object is
  // passed to each thread’s ThreadMain(), so they cannot move around in memory.
  std::vector<std::unique_ptr<ThreadInfo>> thread_infos_;

  DISALLOW_COPY_AND_ASSIGN(TestThreadPool);
};

using ThreadMap = std::map<uint64_t, TestThreadPool::ThreadExpectation>;

// Verifies that all of the threads in |threads|, obtained from
// ProcessReaderMac, agree with the expectation in |thread_map|. If
// |tolerate_extra_threads| is true, |threads| is allowed to contain threads
// that are not listed in |thread_map|. This is useful when testing situations
// where code outside of the test’s control (such as system libraries) may start
// threads, or may have started threads prior to a test’s execution.
void ExpectSeveralThreads(ThreadMap* thread_map,
                          const std::vector<ProcessReaderMac::Thread>& threads,
                          const bool tolerate_extra_threads) {
  if (tolerate_extra_threads) {
    ASSERT_GE(threads.size(), thread_map->size());
  } else {
    ASSERT_EQ(threads.size(), thread_map->size());
  }

  for (size_t thread_index = 0; thread_index < threads.size(); ++thread_index) {
    const ProcessReaderMac::Thread& thread = threads[thread_index];
    mach_vm_address_t thread_stack_region_end =
        thread.stack_region_address + thread.stack_region_size;

    const auto& iterator = thread_map->find(thread.id);
    if (!tolerate_extra_threads) {
      // Make sure that the thread is in the expectation map.
      ASSERT_NE(iterator, thread_map->end());
    }

    if (iterator != thread_map->end()) {
      EXPECT_GE(iterator->second.stack_address, thread.stack_region_address);
      EXPECT_LT(iterator->second.stack_address, thread_stack_region_end);

      EXPECT_EQ(thread.suspend_count, iterator->second.suspend_count);

      // Remove the thread from the expectation map since it’s already been
      // found. This makes it easy to check for duplicate thread IDs, and makes
      // it easy to check that all expected threads were found.
      thread_map->erase(iterator);
    }

    // Make sure that this thread’s ID, stack region, and port don’t conflict
    // with any other thread’s. Each thread should have a unique value for its
    // ID and port, and each should have its own stack that doesn’t touch any
    // other thread’s stack.
    for (size_t other_thread_index = 0; other_thread_index < threads.size();
         ++other_thread_index) {
      if (other_thread_index == thread_index) {
        continue;
      }

      const ProcessReaderMac::Thread& other_thread =
          threads[other_thread_index];

      EXPECT_NE(other_thread.id, thread.id);
      EXPECT_NE(other_thread.port, thread.port);

      mach_vm_address_t other_thread_stack_region_end =
          other_thread.stack_region_address + other_thread.stack_region_size;
      EXPECT_FALSE(thread.stack_region_address >=
                       other_thread.stack_region_address &&
                   thread.stack_region_address < other_thread_stack_region_end);
      EXPECT_FALSE(thread_stack_region_end >
                       other_thread.stack_region_address &&
                   thread_stack_region_end <= other_thread_stack_region_end);
    }
  }

  // Make sure that each expected thread was found.
  EXPECT_TRUE(thread_map->empty());
}

TEST(ProcessReaderMac, SelfSeveralThreads) {
  // Set up the ProcessReaderMac here, before any other threads are running.
  // This tests that the threads it returns are lazily initialized as a snapshot
  // of the threads at the time of the first call to Threads(), and not at the
  // time the ProcessReader was created or initialized.
  ProcessReaderMac process_reader;
  ASSERT_TRUE(process_reader.Initialize(mach_task_self()));

  TestThreadPool thread_pool;
  constexpr size_t kChildThreads = 16;
  ASSERT_NO_FATAL_FAILURE(thread_pool.StartThreads(kChildThreads));

  // Build a map of all expected threads, keyed by each thread’s ID. The values
  // are addresses that should lie somewhere within each thread’s stack.
  ThreadMap thread_map;
  const uint64_t self_thread_id = PthreadToThreadID(pthread_self());
  TestThreadPool::ThreadExpectation expectation;
  expectation.stack_address = FromPointerCast<mach_vm_address_t>(&thread_map);
  expectation.suspend_count = 0;
  thread_map[self_thread_id] = expectation;
  for (size_t thread_index = 0; thread_index < kChildThreads; ++thread_index) {
    uint64_t thread_id = thread_pool.GetThreadInfo(thread_index, &expectation);

    // There can’t be any duplicate thread IDs.
    EXPECT_EQ(thread_map.count(thread_id), 0u);

    thread_map[thread_id] = expectation;
  }

  const std::vector<ProcessReaderMac::Thread>& threads =
      process_reader.Threads();

  // Other tests that have run previously may have resulted in the creation of
  // threads that still exist, so pass true for |tolerate_extra_threads|.
  ExpectSeveralThreads(&thread_map, threads, true);

  // When testing in-process, verify that when this thread shows up in the
  // vector, it has the expected thread port, and that this thread port only
  // shows up once.
  thread_t thread_self = MachThreadSelf();
  bool found_thread_self = false;
  for (const ProcessReaderMac::Thread& thread : threads) {
    if (thread.port == thread_self) {
      EXPECT_FALSE(found_thread_self);
      found_thread_self = true;
      EXPECT_EQ(thread.id, self_thread_id);
    }
  }
  EXPECT_TRUE(found_thread_self);
}

class ProcessReaderThreadedChild final : public MachMultiprocess {
 public:
  explicit ProcessReaderThreadedChild(size_t thread_count)
      : MachMultiprocess(), thread_count_(thread_count) {}

  ~ProcessReaderThreadedChild() {}

 private:
  void MachMultiprocessParent() override {
    ProcessReaderMac process_reader;
    ASSERT_TRUE(process_reader.Initialize(ChildTask()));

    FileHandle read_handle = ReadPipeHandle();

    // Build a map of all expected threads, keyed by each thread’s ID, and with
    // addresses that should lie somewhere within each thread’s stack as values.
    // These IDs and addresses all come from the child process via the pipe.
    ThreadMap thread_map;
    for (size_t thread_index = 0; thread_index < thread_count_ + 1;
         ++thread_index) {
      uint64_t thread_id;
      CheckedReadFileExactly(read_handle, &thread_id, sizeof(thread_id));

      TestThreadPool::ThreadExpectation expectation;
      CheckedReadFileExactly(read_handle,
                             &expectation.stack_address,
                             sizeof(expectation.stack_address));
      CheckedReadFileExactly(read_handle,
                             &expectation.suspend_count,
                             sizeof(expectation.suspend_count));

      // There can’t be any duplicate thread IDs.
      EXPECT_EQ(thread_map.count(thread_id), 0u);

      thread_map[thread_id] = expectation;
    }

    const std::vector<ProcessReaderMac::Thread>& threads =
        process_reader.Threads();

    // The child shouldn’t have any threads other than its main thread and the
    // ones it created in its pool, so pass false for |tolerate_extra_threads|.
    ExpectSeveralThreads(&thread_map, threads, false);
  }

  void MachMultiprocessChild() override {
    TestThreadPool thread_pool;
    ASSERT_NO_FATAL_FAILURE(thread_pool.StartThreads(thread_count_));

    FileHandle write_handle = WritePipeHandle();

    // This thread isn’t part of the thread pool, but the parent will be able
    // to inspect it. Write an entry for it.
    uint64_t thread_id = PthreadToThreadID(pthread_self());

    CheckedWriteFile(write_handle, &thread_id, sizeof(thread_id));

    TestThreadPool::ThreadExpectation expectation;
    expectation.stack_address = FromPointerCast<mach_vm_address_t>(&thread_id);
    expectation.suspend_count = 0;

    CheckedWriteFile(write_handle,
                     &expectation.stack_address,
                     sizeof(expectation.stack_address));
    CheckedWriteFile(write_handle,
                     &expectation.suspend_count,
                     sizeof(expectation.suspend_count));

    // Write an entry for everything in the thread pool.
    for (size_t thread_index = 0; thread_index < thread_count_;
         ++thread_index) {
      uint64_t thread_id =
          thread_pool.GetThreadInfo(thread_index, &expectation);

      CheckedWriteFile(write_handle, &thread_id, sizeof(thread_id));
      CheckedWriteFile(write_handle,
                       &expectation.stack_address,
                       sizeof(expectation.stack_address));
      CheckedWriteFile(write_handle,
                       &expectation.suspend_count,
                       sizeof(expectation.suspend_count));
    }

    // Wait for the parent to signal that it’s OK to exit by closing its end of
    // the pipe.
    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  size_t thread_count_;

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderThreadedChild);
};

TEST(ProcessReaderMac, ChildOneThread) {
  // The main thread plus zero child threads equals one thread.
  constexpr size_t kChildThreads = 0;
  ProcessReaderThreadedChild process_reader_threaded_child(kChildThreads);
  process_reader_threaded_child.Run();
}

TEST(ProcessReaderMac, ChildSeveralThreads) {
  constexpr size_t kChildThreads = 64;
  ProcessReaderThreadedChild process_reader_threaded_child(kChildThreads);
  process_reader_threaded_child.Run();
}

// cl_kernels images (OpenCL kernels) are weird. They’re not ld output and don’t
// exist as files on disk. On OS X 10.10 and 10.11, their Mach-O structure isn’t
// perfect. They show up loaded into many executables, so these quirks should be
// tolerated.
//
// Create an object of this class to ensure that at least one cl_kernels image
// is present in a process, to be able to test that all of the process-reading
// machinery tolerates them. On systems where cl_kernels modules have known
// quirks, the image that an object of this class produces will also have those
// quirks.
//
// https://openradar.appspot.com/20239912
class ScopedOpenCLNoOpKernel {
 public:
  ScopedOpenCLNoOpKernel()
      : context_(nullptr), program_(nullptr), kernel_(nullptr) {}

  ~ScopedOpenCLNoOpKernel() {
    if (kernel_) {
      cl_int rv = clReleaseKernel(kernel_);
      EXPECT_EQ(rv, CL_SUCCESS) << "clReleaseKernel";
    }

    if (program_) {
      cl_int rv = clReleaseProgram(program_);
      EXPECT_EQ(rv, CL_SUCCESS) << "clReleaseProgram";
    }

    if (context_) {
      cl_int rv = clReleaseContext(context_);
      EXPECT_EQ(rv, CL_SUCCESS) << "clReleaseContext";
    }
  }

  void SetUp() {
    cl_platform_id platform_id;
    cl_int rv = clGetPlatformIDs(1, &platform_id, nullptr);
    ASSERT_EQ(rv, CL_SUCCESS) << "clGetPlatformIDs";

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_10 && \
    MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_10
// cl_device_id is really available in OpenCL.framework back to 10.5, but in
// the 10.10 SDK and later, OpenCL.framework includes <OpenGL/CGLDevice.h>,
// which has its own cl_device_id that was introduced in 10.10. That
// triggers erroneous availability warnings.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability"
#define DISABLED_WUNGUARDED_AVAILABILITY
#endif  // SDK >= 10.10 && DT < 10.10
    // Use CL_DEVICE_TYPE_CPU to ensure that the kernel would execute on the
    // CPU. This is the only device type that a cl_kernels image will be created
    // for.
    cl_device_id device_id;
#if defined(DISABLED_WUNGUARDED_AVAILABILITY)
#pragma clang diagnostic pop
#undef DISABLED_WUNGUARDED_AVAILABILITY
#endif  // DISABLED_WUNGUARDED_AVAILABILITY
    rv =
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);
    ASSERT_EQ(rv, CL_SUCCESS) << "clGetDeviceIDs";

    context_ = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &rv);
    ASSERT_EQ(rv, CL_SUCCESS) << "clCreateContext";

    // The goal of the program in |sources| is to produce a cl_kernels image
    // that doesn’t strictly conform to Mach-O expectations. On OS X 10.10,
    // cl_kernels modules show up with an __LD,__compact_unwind section, showing
    // up in the __TEXT segment. MachOImageSegmentReader would normally reject
    // modules for this problem, but a special exception is made when this
    // occurs in cl_kernels images. This portion of the test is aimed at making
    // sure that this exception works correctly.
    //
    // A true no-op program doesn’t actually produce unwind data, so there would
    // be no errant __LD,__compact_unwind section on 10.10, and the test
    // wouldn’t be complete. This simple no-op, which calls a built-in function,
    // does produce unwind data provided optimization is disabled.
    // "-cl-opt-disable" is given to clBuildProgram() below.
    const char* sources[] = {
        "__kernel void NoOp(void) {barrier(CLK_LOCAL_MEM_FENCE);}",
    };
    const size_t source_lengths[] = {
        strlen(sources[0]),
    };
    static_assert(arraysize(sources) == arraysize(source_lengths),
                  "arrays must be parallel");

    program_ = clCreateProgramWithSource(
        context_, arraysize(sources), sources, source_lengths, &rv);
    ASSERT_EQ(rv, CL_SUCCESS) << "clCreateProgramWithSource";

    rv = clBuildProgram(
        program_, 1, &device_id, "-cl-opt-disable", nullptr, nullptr);
    ASSERT_EQ(rv, CL_SUCCESS) << "clBuildProgram";

    kernel_ = clCreateKernel(program_, "NoOp", &rv);
    ASSERT_EQ(rv, CL_SUCCESS) << "clCreateKernel";
  }

 private:
  cl_context context_;
  cl_program program_;
  cl_kernel kernel_;

  DISALLOW_COPY_AND_ASSIGN(ScopedOpenCLNoOpKernel);
};

// Although Mac OS X 10.6 has OpenCL and can compile and execute OpenCL code,
// OpenCL kernels that run on the CPU do not result in cl_kernels images
// appearing on that OS version.
bool ExpectCLKernels() {
#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_7
  return true;
#else
  return MacOSXMinorVersion() >= 7;
#endif
}

TEST(ProcessReaderMac, SelfModules) {
  ScopedOpenCLNoOpKernel ensure_cl_kernels;
  ASSERT_NO_FATAL_FAILURE(ensure_cl_kernels.SetUp());

  ProcessReaderMac process_reader;
  ASSERT_TRUE(process_reader.Initialize(mach_task_self()));

  uint32_t dyld_image_count = _dyld_image_count();
  const std::vector<ProcessReaderMac::Module>& modules =
      process_reader.Modules();

  // There needs to be at least an entry for the main executable, for a dylib,
  // and for dyld.
  ASSERT_GE(modules.size(), 3u);

  // dyld_image_count doesn’t include an entry for dyld itself, but |modules|
  // does.
  ASSERT_EQ(modules.size(), dyld_image_count + 1);

  bool found_cl_kernels = false;
  for (uint32_t index = 0; index < dyld_image_count; ++index) {
    SCOPED_TRACE(base::StringPrintf(
        "index %u, name %s", index, modules[index].name.c_str()));

    const char* dyld_image_name = _dyld_get_image_name(index);
    EXPECT_EQ(modules[index].name, dyld_image_name);
    ASSERT_TRUE(modules[index].reader);
    EXPECT_EQ(
        modules[index].reader->Address(),
        FromPointerCast<mach_vm_address_t>(_dyld_get_image_header(index)));

    bool expect_timestamp;
    if (index == 0) {
      // dyld didn’t load the main executable, so it couldn’t record its
      // timestamp, and it is reported as 0.
      EXPECT_EQ(modules[index].timestamp, 0);
    } else if (IsMalformedCLKernelsModule(modules[index].reader->FileType(),
                                          modules[index].name,
                                          &expect_timestamp)) {
      // cl_kernels doesn’t exist as a file, but may still have a timestamp.
      if (!expect_timestamp) {
        EXPECT_EQ(modules[index].timestamp, 0);
      } else {
        EXPECT_NE(modules[index].timestamp, 0);
      }
      found_cl_kernels = true;
    } else {
      // Hope that the module didn’t change on disk.
      struct stat stat_buf;
      int rv = stat(dyld_image_name, &stat_buf);
      EXPECT_EQ(rv, 0) << ErrnoMessage("stat");
      if (rv == 0) {
        EXPECT_EQ(modules[index].timestamp, stat_buf.st_mtime);
      }
    }
  }

  EXPECT_EQ(found_cl_kernels, ExpectCLKernels());

  size_t index = modules.size() - 1;
  EXPECT_EQ(modules[index].name, kDyldPath);

  // dyld didn’t load itself either, so it couldn’t record its timestamp, and it
  // is also reported as 0.
  EXPECT_EQ(modules[index].timestamp, 0);

  const dyld_all_image_infos* dyld_image_infos = DyldGetAllImageInfos();
  if (dyld_image_infos->version >= 2) {
    ASSERT_TRUE(modules[index].reader);
    EXPECT_EQ(modules[index].reader->Address(),
              FromPointerCast<mach_vm_address_t>(
                  dyld_image_infos->dyldImageLoadAddress));
  }
}

class ProcessReaderModulesChild final : public MachMultiprocess {
 public:
  ProcessReaderModulesChild() : MachMultiprocess() {}

  ~ProcessReaderModulesChild() {}

 private:
  void MachMultiprocessParent() override {
    ProcessReaderMac process_reader;
    ASSERT_TRUE(process_reader.Initialize(ChildTask()));

    const std::vector<ProcessReaderMac::Module>& modules =
        process_reader.Modules();

    // There needs to be at least an entry for the main executable, for a dylib,
    // and for dyld.
    ASSERT_GE(modules.size(), 3u);

    FileHandle read_handle = ReadPipeHandle();

    uint32_t expect_modules;
    CheckedReadFileExactly(
        read_handle, &expect_modules, sizeof(expect_modules));

    ASSERT_EQ(modules.size(), expect_modules);

    bool found_cl_kernels = false;
    for (size_t index = 0; index < modules.size(); ++index) {
      SCOPED_TRACE(base::StringPrintf(
          "index %zu, name %s", index, modules[index].name.c_str()));

      uint32_t expect_name_length;
      CheckedReadFileExactly(
          read_handle, &expect_name_length, sizeof(expect_name_length));

      // The NUL terminator is not read.
      std::string expect_name(expect_name_length, '\0');
      CheckedReadFileExactly(read_handle, &expect_name[0], expect_name_length);
      EXPECT_EQ(modules[index].name, expect_name);

      mach_vm_address_t expect_address;
      CheckedReadFileExactly(
          read_handle, &expect_address, sizeof(expect_address));
      ASSERT_TRUE(modules[index].reader);
      EXPECT_EQ(modules[index].reader->Address(), expect_address);

      bool expect_timestamp;
      if (index == 0 || index == modules.size() - 1) {
        // dyld didn’t load the main executable or itself, so it couldn’t record
        // these timestamps, and they are reported as 0.
        EXPECT_EQ(modules[index].timestamp, 0);
      } else if (IsMalformedCLKernelsModule(modules[index].reader->FileType(),
                                            modules[index].name,
                                            &expect_timestamp)) {
        // cl_kernels doesn’t exist as a file, but may still have a timestamp.
        if (!expect_timestamp) {
          EXPECT_EQ(modules[index].timestamp, 0);
        } else {
          EXPECT_NE(modules[index].timestamp, 0);
        }
        found_cl_kernels = true;
      } else {
        // Hope that the module didn’t change on disk.
        struct stat stat_buf;
        int rv = stat(expect_name.c_str(), &stat_buf);
        EXPECT_EQ(rv, 0) << ErrnoMessage("stat");
        if (rv == 0) {
          EXPECT_EQ(modules[index].timestamp, stat_buf.st_mtime);
        }
      }
    }

    EXPECT_EQ(found_cl_kernels, ExpectCLKernels());
  }

  void MachMultiprocessChild() override {
    FileHandle write_handle = WritePipeHandle();

    uint32_t dyld_image_count = _dyld_image_count();
    const dyld_all_image_infos* dyld_image_infos = DyldGetAllImageInfos();

    uint32_t write_image_count = dyld_image_count;
    if (dyld_image_infos->version >= 2) {
      // dyld_image_count doesn’t include an entry for dyld itself, but one will
      // be written.
      ++write_image_count;
    }

    CheckedWriteFile(
        write_handle, &write_image_count, sizeof(write_image_count));

    for (size_t index = 0; index < write_image_count; ++index) {
      const char* dyld_image_name;
      mach_vm_address_t dyld_image_address;

      if (index < dyld_image_count) {
        dyld_image_name = _dyld_get_image_name(index);
        dyld_image_address =
            FromPointerCast<mach_vm_address_t>(_dyld_get_image_header(index));
      } else {
        dyld_image_name = kDyldPath;
        dyld_image_address = FromPointerCast<mach_vm_address_t>(
            dyld_image_infos->dyldImageLoadAddress);
      }

      uint32_t dyld_image_name_length = strlen(dyld_image_name);
      CheckedWriteFile(write_handle,
                       &dyld_image_name_length,
                       sizeof(dyld_image_name_length));

      // The NUL terminator is not written.
      CheckedWriteFile(write_handle, dyld_image_name, dyld_image_name_length);

      CheckedWriteFile(
          write_handle, &dyld_image_address, sizeof(dyld_image_address));
    }

    // Wait for the parent to signal that it’s OK to exit by closing its end of
    // the pipe.
    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderModulesChild);
};

TEST(ProcessReaderMac, ChildModules) {
  ScopedOpenCLNoOpKernel ensure_cl_kernels;
  ASSERT_NO_FATAL_FAILURE(ensure_cl_kernels.SetUp());

  ProcessReaderModulesChild process_reader_modules_child;
  process_reader_modules_child.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
