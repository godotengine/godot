// Copyright 2015, Google Inc.
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
// Injection point for custom user configurations. See README for details
//
// ** Custom implementation starts here **

#ifndef GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_
#define GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_

// Use a stub Notification class.
//
// The built-in Notification class in GoogleTest v1.12.1 uses std::mutex and
// std::condition_variable. The <mutex> and <condition_variable> headers of
// mingw32 g++ (GNU 10.0.0) define std::mutex and std::condition_variable only
// when configured with the posix threads option but don't define them when
// configured with the win32 threads option. The Notification class is only
// used in GoogleTest's internal tests. Since we don't build GoogleTest's
// internal tests, we don't need a working Notification class. Although it's
// not hard to fix the mingw32 g++ compilation errors by implementing the
// Notification class using Windows CRITICAL_SECTION and CONDITION_VARIABLE,
// it's simpler to just use a stub Notification class on all platforms.
//
// The default constructor of the stub class is deleted and the declaration of
// the Notify() method is commented out, so that compilation will fail if any
// code actually uses the Notification class.

#define GTEST_HAS_NOTIFICATION_ 1
namespace testing {
namespace internal {
class Notification {
 public:
  Notification() = delete;
  Notification(const Notification&) = delete;
  Notification& operator=(const Notification&) = delete;
  // void Notify();
  void WaitForNotification() {}
};
}  // namespace internal
}  // namespace testing

#endif  // GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_
