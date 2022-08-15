///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilInterpolationMode.h                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL interpolation mode.                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DxilConstants.h"

namespace hlsl {


/// Use this class to represent signature element interpolation mode.
class InterpolationMode {
public:
  using Kind = DXIL::InterpolationMode;

  InterpolationMode();
  InterpolationMode(const InterpolationMode &Mode) = default;
  InterpolationMode(Kind Kind);
  InterpolationMode(unsigned long long Kind);
  InterpolationMode(bool bNoInterpolation, bool bLinear, bool bNoperspective, bool bCentroid, bool bSample);
  InterpolationMode &operator=(const InterpolationMode &o);
  bool operator==(const InterpolationMode &o) const;

  bool IsValid() const                        { return m_Kind >= Kind::Undefined && m_Kind < Kind::Invalid; }
  bool IsUndefined() const                    { return m_Kind == Kind::Undefined; }
  bool IsConstant() const                     { return m_Kind == Kind::Constant; }
  bool IsLinear() const                       { return m_Kind == Kind::Linear; }
  bool IsLinearCentroid() const               { return m_Kind == Kind::LinearCentroid; }
  bool IsLinearNoperspective() const          { return m_Kind == Kind::LinearNoperspective; }
  bool IsLinearNoperspectiveCentroid() const  { return m_Kind == Kind::LinearNoperspectiveCentroid; }
  bool IsLinearSample() const                 { return m_Kind == Kind::LinearSample; }
  bool IsLinearNoperspectiveSample() const    { return m_Kind == Kind::LinearNoperspectiveSample; }

  bool IsAnyLinear() const;
  bool IsAnyNoPerspective() const;
  bool IsAnyCentroid() const;
  bool IsAnySample() const;

  Kind GetKind() const                        { return m_Kind; }
  const char *GetName() const;

private:
  Kind m_Kind;
};

} // namespace hlsl
