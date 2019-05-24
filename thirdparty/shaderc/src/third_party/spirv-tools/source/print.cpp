// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "source/print.h"

#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || \
    defined(SPIRV_FREEBSD)
namespace spvtools {

clr::reset::operator const char*() { return "\x1b[0m"; }

clr::grey::operator const char*() { return "\x1b[1;30m"; }

clr::red::operator const char*() { return "\x1b[31m"; }

clr::green::operator const char*() { return "\x1b[32m"; }

clr::yellow::operator const char*() { return "\x1b[33m"; }

clr::blue::operator const char*() { return "\x1b[34m"; }

}  // namespace spvtools
#elif defined(SPIRV_WINDOWS)
#include <windows.h>

namespace spvtools {

static void SetConsoleForegroundColorPrimary(HANDLE hConsole, WORD color) {
  // Get screen buffer information from console handle
  CONSOLE_SCREEN_BUFFER_INFO bufInfo;
  GetConsoleScreenBufferInfo(hConsole, &bufInfo);

  // Get background color
  color = WORD(color | (bufInfo.wAttributes & 0xfff0));

  // Set foreground color
  SetConsoleTextAttribute(hConsole, color);
}

static void SetConsoleForegroundColor(WORD color) {
  SetConsoleForegroundColorPrimary(GetStdHandle(STD_OUTPUT_HANDLE), color);
  SetConsoleForegroundColorPrimary(GetStdHandle(STD_ERROR_HANDLE), color);
}

clr::reset::operator const char*() {
  if (isPrint) {
    SetConsoleForegroundColor(0xf);
    return "";
  }
  return "\x1b[0m";
}

clr::grey::operator const char*() {
  if (isPrint) {
    SetConsoleForegroundColor(FOREGROUND_INTENSITY);
    return "";
  }
  return "\x1b[1;30m";
}

clr::red::operator const char*() {
  if (isPrint) {
    SetConsoleForegroundColor(FOREGROUND_RED);
    return "";
  }
  return "\x1b[31m";
}

clr::green::operator const char*() {
  if (isPrint) {
    SetConsoleForegroundColor(FOREGROUND_GREEN);
    return "";
  }
  return "\x1b[32m";
}

clr::yellow::operator const char*() {
  if (isPrint) {
    SetConsoleForegroundColor(FOREGROUND_RED | FOREGROUND_GREEN);
    return "";
  }
  return "\x1b[33m";
}

clr::blue::operator const char*() {
  // Blue all by itself is hard to see against a black background (the
  // default on command shell), or a medium blue background (the default
  // on PowerShell).  So increase its intensity.

  if (isPrint) {
    SetConsoleForegroundColor(FOREGROUND_BLUE | FOREGROUND_INTENSITY);
    return "";
  }
  return "\x1b[94m";
}

}  // namespace spvtools
#else
namespace spvtools {

clr::reset::operator const char*() { return ""; }

clr::grey::operator const char*() { return ""; }

clr::red::operator const char*() { return ""; }

clr::green::operator const char*() { return ""; }

clr::yellow::operator const char*() { return ""; }

clr::blue::operator const char*() { return ""; }

}  // namespace spvtools
#endif
