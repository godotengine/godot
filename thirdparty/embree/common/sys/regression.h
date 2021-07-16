// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"

#include <vector>

namespace embree
{
  /*! virtual interface for all regression tests */
  struct RegressionTest 
  { 
    RegressionTest (std::string name) : name(name) {}
    virtual bool run() = 0;
    std::string name;
  };
 
  /*! registers a regression test */
  void registerRegressionTest(RegressionTest* test);

  /*! run all regression tests */
  RegressionTest* getRegressionTest(size_t index);
}
