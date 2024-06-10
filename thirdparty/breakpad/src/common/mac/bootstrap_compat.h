// Copyright 2012 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

#ifndef COMMON_MAC_BOOTSTRAP_COMPAT_H_
#define COMMON_MAC_BOOTSTRAP_COMPAT_H_

#include <servers/bootstrap.h>

namespace breakpad {

// Wrapper for bootstrap_register to avoid deprecation warnings.
//
// In 10.6, it's possible to call bootstrap_check_in as the one-stop-shop for
// handling what bootstrap_register is used for. In 10.5, bootstrap_check_in
// can't check in a service whose name has not yet been registered, despite
// bootstrap_register being marked as deprecated in that OS release. Breakpad
// needs to register new service names, and in 10.5, calling
// bootstrap_register is the only way to achieve that. Attempts to call
// bootstrap_check_in for a new service name on 10.5 will result in
// BOOTSTRAP_UNKNOWN_SERVICE being returned rather than registration of the
// new service name.
kern_return_t BootstrapRegister(mach_port_t bp,
                                name_t service_name,
                                mach_port_t sp);

}  // namespace breakpad

#endif  // COMMON_MAC_BOOTSTRAP_COMPAT_H_
