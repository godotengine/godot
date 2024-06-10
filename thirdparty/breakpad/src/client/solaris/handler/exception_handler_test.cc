// Copyright 2007 Google LLC
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

// Author: Alfred Peng

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <pthread.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "client/solaris/handler/exception_handler.h"
#include "client/solaris/handler/solaris_lwp.h"

using namespace google_breakpad;

// Thread use this to see if it should stop working.
static bool should_exit = false;

static int foo2(int arg) {
  // Stack variable, used for debugging stack dumps.
  int c = 0xcccccccc;
  fprintf(stderr, "Thread trying to crash: %x\n", getpid());
  c = *reinterpret_cast<int*>(0x5);
  return c;
}

static int foo(int arg) {
  // Stack variable, used for debugging stack dumps.
  int b = 0xbbbbbbbb;
  b = foo2(b);
  return b;
}

static void* thread_crash(void*) {
  // Stack variable, used for debugging stack dumps.
  int a = 0xaaaaaaaa;
  sleep(3);
  a = foo(a);
  printf("%x\n", a);
  return NULL;
}

static void* thread_main(void*) {
  while (!should_exit)
    sleep(1);
  return NULL;
}

static void CreateCrashThread() {
  pthread_t h;
  pthread_create(&h, NULL, thread_crash, NULL);
  pthread_detach(h);
}

// Create working threads.
static void CreateThread(int num) {
  pthread_t h;
  for (int i = 0; i < num; ++i) {
    pthread_create(&h, NULL, thread_main, NULL);
    pthread_detach(h);
  }
}

// Callback when minidump written.
static bool MinidumpCallback(const char* dump_path,
                             const char* minidump_id,
                             void* context,
                             bool succeeded) {
  int index = reinterpret_cast<int>(context);
  if (index == 0) {
    should_exit = true;
    return true;
  }
  // Don't process it.
  return false;
}

int main(int argc, char* argv[]) {
  int handler_index = 1;
  ExceptionHandler handler_ignore(".", NULL, MinidumpCallback,
                                  (void*)handler_index, true);
  CreateCrashThread();
  CreateThread(10);

  while (true)
    sleep(20);
  should_exit = true;

  return 0;
}
