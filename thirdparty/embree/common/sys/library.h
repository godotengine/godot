// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "platform.h"

namespace embree
{
  /*! type for shared library */
  typedef struct opaque_lib_t* lib_t;

  /*! loads a shared library */
  lib_t openLibrary(const std::string& file);

  /*! returns address of a symbol from the library */
  void* getSymbol(lib_t lib, const std::string& sym);

  /*! unloads a shared library */
  void closeLibrary(lib_t lib);
}
