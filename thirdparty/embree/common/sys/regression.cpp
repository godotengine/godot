// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "regression.h"

namespace embree
{
  /* registerRegressionTest is invoked from static initializers, thus
   * we cannot have the regression_tests variable as global static
   * variable due to issues with static variable initialization
   * order. */
  std::vector<RegressionTest*>& get_regression_tests()
  {
    static std::vector<RegressionTest*> regression_tests;
    return regression_tests;
  } 

  void registerRegressionTest(RegressionTest* test) 
  {
    get_regression_tests().push_back(test);
  }

  RegressionTest* getRegressionTest(size_t index)
  {
    if (index >= get_regression_tests().size())
      return nullptr;
    
    return get_regression_tests()[index];
  }
}
