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

#include <mach/mach.h>
#include <servers/bootstrap.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

//==============================================================================
// class OnDemandServer :
//    A basic on-demand server launcher supporting a single named service port
//
// Example Usage :
//
//  kern_return_t result;
//  OnDemandServer* server = OnDemandServer::Create("/tmp/myserver",
//                                                  "com.MyCompany.MyServiceName",
//                                                  true,
//                                                  &result);
//
//  if (server) {
//    server->LaunchOnDemand();
//    mach_port_t service_port = GetServicePort();
//
//    // Send a mach message to service_port and "myserver" will be launched
//  }
//
//
//                  ---- Now in the server code ----
//
//  // "myserver" should get the service port and read the message which
//  // launched it:
//  mach_port_t service_rcv_port_;
//  kern_return_t kr = bootstrap_check_in(bootstrap_port,
//                                      "com.MyCompany.MyServiceName",
//                                      &service_rcv_port_);
//  // mach_msg() read service_rcv_port_ ....
//
//  ....
//
//  // Later "myserver" may want to unregister the service if it doesn't
//  // want its bootstrap service to stick around after it exits.
//
//  // DO NOT use mach_port_deallocate() here -- it will fail and the
//  // following bootstrap_register() will also fail leaving our service
//  // name hanging around forever (until reboot)
//  kern_return_t kr = mach_port_destroy(mach_task_self(), service_rcv_port_);
//
//  kr = bootstrap_register(bootstrap_port,
//                          "com.MyCompany.MyServiceName",
//                          MACH_PORT_NULL);

class OnDemandServer {
 public:
  // must call Initialize() to be useful
  OnDemandServer()
    : server_port_(MACH_PORT_NULL),
      service_port_(MACH_PORT_NULL),
      unregister_on_cleanup_(true) {
  }

  // Creates the bootstrap server and service
  kern_return_t Initialize(const char* server_command,
                           const char* service_name,
                           bool unregister_on_cleanup);

  // Returns an OnDemandServer object if successful, or NULL if there's
  // an error.  The error result will be returned in out_result.
  //
  //    server_command : the full path name including optional command-line
  //      arguments to the executable representing the server
  //
  //    service_name : represents service name
  //      something like "com.company.ServiceName"
  //
  //    unregister_on_cleanup : if true, unregisters the service name
  //      when the OnDemandServer is deleted -- unregistering will
  //      ONLY be possible if LaunchOnDemand() has NOT been called.
  //      If false, then the service will continue to be registered
  //      even after the current process quits.
  //
  //    out_result : if non-NULL, returns the result
  //      this value will be KERN_SUCCESS if Create() returns non-NULL
  //
  static OnDemandServer* Create(const char *server_command,
                                const char* service_name,
                                bool unregister_on_cleanup,
                                kern_return_t* out_result);

  // Cleans up and if LaunchOnDemand() has not yet been called then
  // the bootstrap service will be unregistered.
  ~OnDemandServer();

  // This must be called if we intend to commit to launching the server
  // by sending a mach message to our service port.  Do not call it otherwise
  // or it will be difficult (impossible?) to unregister the service name.
  void LaunchOnDemand();

  // This is the port we need to send a mach message to after calling
  // LaunchOnDemand().  Sending a message causing an immediate launch
  // of the server
  mach_port_t GetServicePort() { return service_port_; }

 private:
  // Disallow copy constructor
  OnDemandServer(const OnDemandServer&);

  // Cleans up and if LaunchOnDemand() has not yet been called then
  // the bootstrap service will be unregistered.
  void Unregister();

  name_t      service_name_;

  mach_port_t server_port_;
  mach_port_t service_port_;
  bool        unregister_on_cleanup_;
};
