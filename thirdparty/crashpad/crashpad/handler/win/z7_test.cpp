// Copyright 2015 The Crashpad Authors. All rights reserved.
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

// Build in VC++6 or older command prompt with:
//
// cl /nologo /W4 /MT /Z7 z7_test.cpp /link /dll /out:z7_test.dll /debugtype:cv /pdb:none
//
// Given that this is quite tedious to build, the result is also checked in.

#include <windows.h>
#include <stdio.h>

extern "C" __declspec(dllexport) void CrashMe() {
  volatile int* foo = reinterpret_cast<volatile int*>(7);
  *foo = 42;
}

BOOL WINAPI DllMain(HINSTANCE hinstance, DWORD reason, LPVOID) {
  printf("%p %d\n", hinstance, reason);
  return TRUE;
}
