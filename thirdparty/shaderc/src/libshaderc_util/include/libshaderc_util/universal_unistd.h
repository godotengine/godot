// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#ifndef LIBSHADERC_UTIL_UNIVERSAL_UNISTD_H_
#define LIBSHADERC_UTIL_UNIVERSAL_UNISTD_H_

#ifndef _MSC_VER
#include <unistd.h>
#else
// Minimal set of <unistd> needed to compile on windows.

#include <io.h>
#define access _access

// https://msdn.microsoft.com/en-us/library/1w06ktdy.aspx
// Defines these constants.
#define R_OK 4
#define W_OK 2
#endif //_MSC_VER

#endif // LIBSHADERC_UTIL_UNIVERSAL_UNISTD_H_