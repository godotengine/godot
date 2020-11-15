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

#include "util/mach/mach_extensions.h"

#include "base/mac/scoped_mach_port.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "util/mac/mac_util.h"
#include "util/misc/random_string.h"

namespace crashpad {
namespace test {
namespace {

TEST(MachExtensions, MachThreadSelf) {
  base::mac::ScopedMachSendRight thread_self(mach_thread_self());
  EXPECT_EQ(MachThreadSelf(), thread_self);
}

TEST(MachExtensions, NewMachPort_Receive) {
  base::mac::ScopedMachReceiveRight port(NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_NE(port, kMachPortNull);

  mach_port_type_t type;
  kern_return_t kr = mach_port_type(mach_task_self(), port.get(), &type);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "mach_port_get_type");

  EXPECT_EQ(type, MACH_PORT_TYPE_RECEIVE);
}

TEST(MachExtensions, NewMachPort_PortSet) {
  base::mac::ScopedMachPortSet port(NewMachPort(MACH_PORT_RIGHT_PORT_SET));
  ASSERT_NE(port, kMachPortNull);

  mach_port_type_t type;
  kern_return_t kr = mach_port_type(mach_task_self(), port.get(), &type);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "mach_port_get_type");

  EXPECT_EQ(type, MACH_PORT_TYPE_PORT_SET);
}

TEST(MachExtensions, NewMachPort_DeadName) {
  base::mac::ScopedMachSendRight port(NewMachPort(MACH_PORT_RIGHT_DEAD_NAME));
  ASSERT_NE(port, kMachPortNull);

  mach_port_type_t type;
  kern_return_t kr = mach_port_type(mach_task_self(), port.get(), &type);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "mach_port_get_type");

  EXPECT_EQ(type, MACH_PORT_TYPE_DEAD_NAME);
}

constexpr exception_mask_t kExcMaskBasic =
    EXC_MASK_BAD_ACCESS |
    EXC_MASK_BAD_INSTRUCTION |
    EXC_MASK_ARITHMETIC |
    EXC_MASK_EMULATION |
    EXC_MASK_SOFTWARE |
    EXC_MASK_BREAKPOINT |
    EXC_MASK_SYSCALL |
    EXC_MASK_MACH_SYSCALL |
    EXC_MASK_RPC_ALERT;

TEST(MachExtensions, ExcMaskAll) {
  const exception_mask_t exc_mask_all = ExcMaskAll();
  EXPECT_EQ(exc_mask_all & kExcMaskBasic, kExcMaskBasic);

  EXPECT_FALSE(exc_mask_all & EXC_MASK_CRASH);
  EXPECT_FALSE(exc_mask_all & EXC_MASK_CORPSE_NOTIFY);

  const int mac_os_x_minor_version = MacOSXMinorVersion();
  if (mac_os_x_minor_version >= 8) {
    EXPECT_TRUE(exc_mask_all & EXC_MASK_RESOURCE);
  } else {
    EXPECT_FALSE(exc_mask_all & EXC_MASK_RESOURCE);
  }

  if (mac_os_x_minor_version >= 9) {
    EXPECT_TRUE(exc_mask_all & EXC_MASK_GUARD);
  } else {
    EXPECT_FALSE(exc_mask_all & EXC_MASK_GUARD);
  }

  // Bit 0 should not be set.
  EXPECT_FALSE(ExcMaskAll() & 1);

  // Every bit set in ExcMaskAll() must also be set in ExcMaskValid().
  EXPECT_EQ(ExcMaskAll() & ExcMaskValid(), ExcMaskAll());
}

TEST(MachExtensions, ExcMaskValid) {
  const exception_mask_t exc_mask_valid = ExcMaskValid();
  EXPECT_EQ(exc_mask_valid & kExcMaskBasic, kExcMaskBasic);

  EXPECT_TRUE(exc_mask_valid & EXC_MASK_CRASH);

  const int mac_os_x_minor_version = MacOSXMinorVersion();
  if (mac_os_x_minor_version >= 8) {
    EXPECT_TRUE(exc_mask_valid & EXC_MASK_RESOURCE);
  } else {
    EXPECT_FALSE(exc_mask_valid & EXC_MASK_RESOURCE);
  }

  if (mac_os_x_minor_version >= 9) {
    EXPECT_TRUE(exc_mask_valid & EXC_MASK_GUARD);
  } else {
    EXPECT_FALSE(exc_mask_valid & EXC_MASK_GUARD);
  }

  if (mac_os_x_minor_version >= 11) {
    EXPECT_TRUE(exc_mask_valid & EXC_MASK_CORPSE_NOTIFY);
  } else {
    EXPECT_FALSE(exc_mask_valid & EXC_MASK_CORPSE_NOTIFY);
  }

  // Bit 0 should not be set.
  EXPECT_FALSE(ExcMaskValid() & 1);

  // There must be bits set in ExcMaskValid() that are not set in ExcMaskAll().
  EXPECT_TRUE(ExcMaskValid() & ~ExcMaskAll());
}

TEST(MachExtensions, BootstrapCheckInAndLookUp) {
  // This should always exist.
  base::mac::ScopedMachSendRight
      report_crash(BootstrapLookUp("com.apple.ReportCrash"));
  EXPECT_NE(report_crash, kMachPortNull);

  std::string service_name = "org.chromium.crashpad.test.bootstrap_check_in.";
  service_name.append(RandomString());

  {
    // The new service hasn’t checked in yet, so this should fail.
    base::mac::ScopedMachSendRight send(BootstrapLookUp(service_name));
    EXPECT_EQ(send, kMachPortNull);

    // Check it in.
    base::mac::ScopedMachReceiveRight receive(BootstrapCheckIn(service_name));
    EXPECT_NE(receive, kMachPortNull);

    // Now it should be possible to look up the new service.
    send = BootstrapLookUp(service_name);
    EXPECT_NE(send, kMachPortNull);

    // It shouldn’t be possible to check the service in while it’s active.
    base::mac::ScopedMachReceiveRight receive_2(BootstrapCheckIn(service_name));
    EXPECT_EQ(receive_2, kMachPortNull);
  }

  // The new service should be gone now.
  base::mac::ScopedMachSendRight send(BootstrapLookUp(service_name));
  EXPECT_EQ(send, kMachPortNull);

  // It should be possible to check it in again.
  base::mac::ScopedMachReceiveRight receive(BootstrapCheckIn(service_name));
  EXPECT_NE(receive, kMachPortNull);
}

TEST(MachExtensions, SystemCrashReporterHandler) {
  base::mac::ScopedMachSendRight
      system_crash_reporter_handler(SystemCrashReporterHandler());
  EXPECT_TRUE(system_crash_reporter_handler.is_valid());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
