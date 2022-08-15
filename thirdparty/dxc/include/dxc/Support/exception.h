//===- exception.h ----------------------------------------------*- C++ -*-===//
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// exception.h                                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/ErrorCodes.h"
#include "dxc/Support/WinAdapter.h"
#include <exception>
#include <string>

namespace hlsl
{

/// <summary>
/// Exception stores off information about an error and its error message for
/// later consumption by the hlsl compiler tools.
/// </summary>
struct Exception : public std::exception
{
  /// <summary>HRESULT error code. Must be a failure.</summary>
  HRESULT hr;
  std::string msg;

  Exception(HRESULT errCode) : hr(errCode) { }
  Exception(HRESULT errCode, const std::string &errMsg) : hr(errCode), msg(errMsg) { }

  // what returns a formatted message with the error code and the message used
  // to create the message.
  virtual const char *what() const throw() { return msg.c_str(); }
};

}  // namespace hlsl
