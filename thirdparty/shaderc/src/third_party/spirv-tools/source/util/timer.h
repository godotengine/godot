// Copyright (c) 2018 Google LLC.
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

// Contains utils for getting resource utilization

#ifndef SOURCE_UTIL_TIMER_H_
#define SOURCE_UTIL_TIMER_H_

#if defined(SPIRV_TIMER_ENABLED)

#include <sys/resource.h>
#include <cassert>
#include <iostream>

// A macro to call spvtools::utils::PrintTimerDescription(std::ostream*, bool).
// The first argument must be given as std::ostream*. If it is NULL, the
// function does nothing. Otherwise, it prints resource types measured by Timer
// class. The second is optional and if it is true, the function also prints
// resource type fields related to memory. Otherwise, it does not print memory
// related fields. Its default is false. In usual, this must be placed before
// calling Timer::Report() to inform what those fields printed by
// Timer::Report() indicate (or spvtools::utils::PrintTimerDescription() must be
// used instead).
#define SPIRV_TIMER_DESCRIPTION(...) \
  spvtools::utils::PrintTimerDescription(__VA_ARGS__)

// Creates an object of ScopedTimer to measure the resource utilization for the
// scope surrounding it as the following example:
//
//   {   // <-- beginning of this scope
//
//     /* ... code out of interest ... */
//
//     SPIRV_TIMER_SCOPED(std::cout, tag);
//
//     /* ... lines of code that we want to know its resource usage ... */
//
//   }   // <-- end of this scope. The destructor of ScopedTimer prints tag and
//              the resource utilization to std::cout.
#define SPIRV_TIMER_SCOPED(...)                                         \
  spvtools::utils::ScopedTimer<spvtools::utils::Timer> timer##__LINE__( \
      __VA_ARGS__)

namespace spvtools {
namespace utils {

// Prints the description of resource types measured by Timer class. If |out| is
// NULL, it does nothing. Otherwise, it prints resource types. The second is
// optional and if it is true, the function also prints resource type fields
// related to memory. Its default is false. In usual, this must be placed before
// calling Timer::Report() to inform what those fields printed by
// Timer::Report() indicate.
void PrintTimerDescription(std::ostream*, bool = false);

// Status of Timer. kGetrusageFailed means it failed in calling getrusage().
// kClockGettimeWalltimeFailed means it failed in getting wall time when calling
// clock_gettime(). kClockGettimeCPUtimeFailed means it failed in getting CPU
// time when calling clock_gettime().
enum UsageStatus {
  kSucceeded = 0,
  kGetrusageFailed = 1 << 0,
  kClockGettimeWalltimeFailed = 1 << 1,
  kClockGettimeCPUtimeFailed = 1 << 2,
};

// Timer measures the resource utilization for a range of code. The resource
// utilization consists of CPU time (i.e., process time), WALL time (elapsed
// time), USR time, SYS time, RSS delta, and the delta of the number of page
// faults. RSS delta and the delta of the number of page faults are measured
// only when |measure_mem_usage| given to the constructor is true. This class
// should be used as the following example:
//
//   spvtools::utils::Timer timer(std::cout);
//   timer.Start();       // <-- set |usage_before_|, |wall_before_|,
//                               and |cpu_before_|
//
//   /* ... lines of code that we want to know its resource usage ... */
//
//   timer.Stop();        // <-- set |cpu_after_|, |wall_after_|, and
//                               |usage_after_|
//   timer.Report(tag);   // <-- print tag and the resource utilization to
//                               std::cout.
class Timer {
 public:
  Timer(std::ostream* out, bool measure_mem_usage = false)
      : report_stream_(out),
        usage_status_(kSucceeded),
        measure_mem_usage_(measure_mem_usage) {}

  // Sets |usage_before_|, |wall_before_|, and |cpu_before_| as results of
  // getrusage(), clock_gettime() for the wall time, and clock_gettime() for the
  // CPU time respectively. Note that this method erases all previous state of
  // |usage_before_|, |wall_before_|, |cpu_before_|.
  virtual void Start();

  // Sets |cpu_after_|, |wall_after_|, and |usage_after_| as results of
  // clock_gettime() for the wall time, and clock_gettime() for the CPU time,
  // getrusage() respectively. Note that this method erases all previous state
  // of |cpu_after_|, |wall_after_|, |usage_after_|.
  virtual void Stop();

  // If |report_stream_| is NULL, it does nothing. Otherwise, it prints the
  // resource utilization (i.e., CPU/WALL/USR/SYS time, RSS delta) between the
  // time of calling Timer::Start() and the time of calling Timer::Stop(). If we
  // cannot get a resource usage because of failures, it prints "Failed" instead
  // for the resource.
  void Report(const char* tag);

  // Returns the measured CPU Time (i.e., process time) for a range of code
  // execution. If kClockGettimeCPUtimeFailed is set by the failure of calling
  // clock_gettime(), it returns -1.
  virtual double CPUTime() {
    if (usage_status_ & kClockGettimeCPUtimeFailed) return -1;
    return TimeDifference(cpu_before_, cpu_after_);
  }

  // Returns the measured Wall Time (i.e., elapsed time) for a range of code
  // execution. If kClockGettimeWalltimeFailed is set by the failure of
  // calling clock_gettime(), it returns -1.
  virtual double WallTime() {
    if (usage_status_ & kClockGettimeWalltimeFailed) return -1;
    return TimeDifference(wall_before_, wall_after_);
  }

  // Returns the measured USR Time for a range of code execution. If
  // kGetrusageFailed is set because of the failure of calling getrusage(), it
  // returns -1.
  virtual double UserTime() {
    if (usage_status_ & kGetrusageFailed) return -1;
    return TimeDifference(usage_before_.ru_utime, usage_after_.ru_utime);
  }

  // Returns the measured SYS Time for a range of code execution. If
  // kGetrusageFailed is set because of the failure of calling getrusage(), it
  // returns -1.
  virtual double SystemTime() {
    if (usage_status_ & kGetrusageFailed) return -1;
    return TimeDifference(usage_before_.ru_stime, usage_after_.ru_stime);
  }

  // Returns the measured RSS delta for a range of code execution. If
  // kGetrusageFailed is set because of the failure of calling getrusage(), it
  // returns -1.
  virtual long RSS() const {
    if (usage_status_ & kGetrusageFailed) return -1;
    return usage_after_.ru_maxrss - usage_before_.ru_maxrss;
  }

  // Returns the measured the delta of the number of page faults for a range of
  // code execution. If kGetrusageFailed is set because of the failure of
  // calling getrusage(), it returns -1.
  virtual long PageFault() const {
    if (usage_status_ & kGetrusageFailed) return -1;
    return (usage_after_.ru_minflt - usage_before_.ru_minflt) +
           (usage_after_.ru_majflt - usage_before_.ru_majflt);
  }

  virtual ~Timer() {}

 private:
  // Returns the time gap between |from| and |to| in seconds.
  static double TimeDifference(const timeval& from, const timeval& to) {
    assert((to.tv_sec > from.tv_sec) ||
           (to.tv_sec == from.tv_sec && to.tv_usec >= from.tv_usec));
    return static_cast<double>(to.tv_sec - from.tv_sec) +
           static_cast<double>(to.tv_usec - from.tv_usec) * .000001;
  }

  // Returns the time gap between |from| and |to| in seconds.
  static double TimeDifference(const timespec& from, const timespec& to) {
    assert((to.tv_sec > from.tv_sec) ||
           (to.tv_sec == from.tv_sec && to.tv_nsec >= from.tv_nsec));
    return static_cast<double>(to.tv_sec - from.tv_sec) +
           static_cast<double>(to.tv_nsec - from.tv_nsec) * .000000001;
  }

  // Output stream to print out the resource utilization. If it is NULL,
  // Report() does nothing.
  std::ostream* report_stream_;

  // Status to stop measurement if a system call returns an error.
  unsigned usage_status_;

  // Variable to save the result of clock_gettime(CLOCK_PROCESS_CPUTIME_ID) when
  // Timer::Start() is called. It is used as the base status of CPU time.
  timespec cpu_before_;

  // Variable to save the result of clock_gettime(CLOCK_MONOTONIC) when
  // Timer::Start() is called. It is used as the base status of WALL time.
  timespec wall_before_;

  // Variable to save the result of getrusage() when Timer::Start() is called.
  // It is used as the base status of USR time, SYS time, and RSS.
  rusage usage_before_;

  // Variable to save the result of clock_gettime(CLOCK_PROCESS_CPUTIME_ID) when
  // Timer::Stop() is called. It is used as the last status of CPU time. The
  // resouce usage is measured by subtracting |cpu_before_| from it.
  timespec cpu_after_;

  // Variable to save the result of clock_gettime(CLOCK_MONOTONIC) when
  // Timer::Stop() is called. It is used as the last status of WALL time. The
  // resouce usage is measured by subtracting |wall_before_| from it.
  timespec wall_after_;

  // Variable to save the result of getrusage() when Timer::Stop() is called. It
  // is used as the last status of USR time, SYS time, and RSS. Those resouce
  // usages are measured by subtracting |usage_before_| from it.
  rusage usage_after_;

  // If true, Timer reports the memory usage information too. Otherwise, Timer
  // reports only USR time, WALL time, SYS time.
  bool measure_mem_usage_;
};

// The purpose of ScopedTimer is to measure the resource utilization for a
// scope. Simply creating a local variable of ScopedTimer will call
// Timer::Start() and it calls Timer::Stop() and Timer::Report() at the end of
// the scope by its destructor. When we use this class, we must choose the
// proper Timer class (for class TimerType template) in advance. This class
// should be used as the following example:
//
//   {   // <-- beginning of this scope
//
//     /* ... code out of interest ... */
//
//     spvtools::utils::ScopedTimer<spvtools::utils::Timer>
//     scopedtimer(std::cout, tag);
//
//     /* ... lines of code that we want to know its resource usage ... */
//
//   }   // <-- end of this scope. The destructor of ScopedTimer prints tag and
//              the resource utilization to std::cout.
//
// The template<class TimerType> is used to choose a Timer class. Currently,
// only options for the Timer class are Timer and MockTimer in the unit test.
template <class TimerType>
class ScopedTimer {
 public:
  ScopedTimer(std::ostream* out, const char* tag,
              bool measure_mem_usage = false)
      : timer(new TimerType(out, measure_mem_usage)), tag_(tag) {
    timer->Start();
  }

  // At the end of the scope surrounding the instance of this class, this
  // destructor saves the last status of resource usage and reports it.
  virtual ~ScopedTimer() {
    timer->Stop();
    timer->Report(tag_);
    delete timer;
  }

 private:
  // Actual timer that measures the resource utilization. It must be an instance
  // of Timer class if there is no special reason to use other class.
  TimerType* timer;

  // A tag that will be printed in front of the trace reported by Timer class.
  const char* tag_;
};

// CumulativeTimer is the same as Timer class, but it supports a cumulative
// measurement as the following example:
//
//   CumulativeTimer *ctimer = new CumulativeTimer(std::cout);
//   ctimer->Start();
//
//   /* ... lines of code that we want to know its resource usage ... */
//
//   ctimer->Stop();
//
//   /* ... code out of interest ... */
//
//   ctimer->Start();
//
//   /* ... lines of code that we want to know its resource usage ... */
//
//   ctimer->Stop();
//   ctimer->Report(tag);
//   delete ctimer;
//
class CumulativeTimer : public Timer {
 public:
  CumulativeTimer(std::ostream* out, bool measure_mem_usage = false)
      : Timer(out, measure_mem_usage),
        cpu_time_(0),
        wall_time_(0),
        usr_time_(0),
        sys_time_(0),
        rss_(0),
        pgfaults_(0) {}

  // If we cannot get a resource usage because of failures, it sets -1 for the
  // resource usage.
  void Stop() override {
    Timer::Stop();

    if (cpu_time_ >= 0 && Timer::CPUTime() >= 0)
      cpu_time_ += Timer::CPUTime();
    else
      cpu_time_ = -1;

    if (wall_time_ >= 0 && Timer::WallTime() >= 0)
      wall_time_ += Timer::WallTime();
    else
      wall_time_ = -1;

    if (usr_time_ >= 0 && Timer::UserTime() >= 0)
      usr_time_ += Timer::UserTime();
    else
      usr_time_ = -1;

    if (sys_time_ >= 0 && Timer::SystemTime() >= 0)
      sys_time_ += Timer::SystemTime();
    else
      sys_time_ = -1;

    if (rss_ >= 0 && Timer::RSS() >= 0)
      rss_ += Timer::RSS();
    else
      rss_ = -1;

    if (pgfaults_ >= 0 && Timer::PageFault() >= 0)
      pgfaults_ += Timer::PageFault();
    else
      pgfaults_ = -1;
  }

  // Returns the cumulative CPU Time (i.e., process time) for a range of code
  // execution.
  double CPUTime() override { return cpu_time_; }

  // Returns the cumulative Wall Time (i.e., elapsed time) for a range of code
  // execution.
  double WallTime() override { return wall_time_; }

  // Returns the cumulative USR Time for a range of code execution.
  double UserTime() override { return usr_time_; }

  // Returns the cumulative SYS Time for a range of code execution.
  double SystemTime() override { return sys_time_; }

  // Returns the cumulative RSS delta for a range of code execution.
  long RSS() const override { return rss_; }

  // Returns the cumulative delta of number of page faults for a range of code
  // execution.
  long PageFault() const override { return pgfaults_; }

 private:
  // Variable to save the cumulative CPU time (i.e., process time).
  double cpu_time_;

  // Variable to save the cumulative wall time (i.e., elapsed time).
  double wall_time_;

  // Variable to save the cumulative user time.
  double usr_time_;

  // Variable to save the cumulative system time.
  double sys_time_;

  // Variable to save the cumulative RSS delta.
  long rss_;

  // Variable to save the cumulative delta of the number of page faults.
  long pgfaults_;
};

}  // namespace utils
}  // namespace spvtools

#else  // defined(SPIRV_TIMER_ENABLED)

#define SPIRV_TIMER_DESCRIPTION(...)
#define SPIRV_TIMER_SCOPED(...)

#endif  // defined(SPIRV_TIMER_ENABLED)

#endif  // SOURCE_UTIL_TIMER_H_
