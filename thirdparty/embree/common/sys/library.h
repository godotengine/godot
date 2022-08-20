// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
