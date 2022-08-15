///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLSLTestData.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides declarations and sample data for unit tests.           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

struct StorageClassDataItem
{
  const char* Keyword;
  bool IsValid;
};

const StorageClassDataItem StorageClassData[] =
{
  { "inline", true },
  { "extern", false },
  { "", true }
};

struct InOutParameterModifierDataItem
{
  const char* Keyword;
  bool ActsAsReference;
};

const InOutParameterModifierDataItem InOutParameterModifierData[] =
{
  { "", false },
  { "in", false },
  { "inout", true },
  { "out", true }
};

