///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilInterpolationMode.cpp                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilInterpolationMode.h"
#include "dxc/Support/Global.h"


namespace hlsl {


//------------------------------------------------------------------------------
//
// InterpolationMode class methods.
//
InterpolationMode::InterpolationMode()
: m_Kind(Kind::Undefined) {
}

InterpolationMode::InterpolationMode(Kind Kind)
: m_Kind(Kind) {
}

InterpolationMode::InterpolationMode(unsigned long long kind) {
  m_Kind = (Kind)kind;
  if (m_Kind >= Kind::Invalid) {
    m_Kind = Kind::Invalid;
  }
}

//                                                                  NoInterpolation, bLinear,     bNoperspective, bCentroid,     bSample
static const DXIL::InterpolationMode interpModeTab[] = {
DXIL::InterpolationMode::Undefined,                       //  0        , False       , False        , False        , False        , False
DXIL::InterpolationMode::LinearSample,                    //  1        , False       , False        , False        , False        , True
DXIL::InterpolationMode::LinearCentroid,                  //  2        , False       , False        , False        , True        , False
DXIL::InterpolationMode::LinearSample,                    //  3        , False       , False        , False        , True        , True
DXIL::InterpolationMode::LinearNoperspective,             //  4        , False       , False        , True        , False        , False
DXIL::InterpolationMode::LinearNoperspectiveSample,       //  5        , False       , False        , True        , False        , True
DXIL::InterpolationMode::LinearNoperspectiveCentroid,     //  6        , False       , False        , True        , True        , False
DXIL::InterpolationMode::LinearNoperspectiveSample,       //  7        , False       , False        , True        , True        , True
DXIL::InterpolationMode::Linear,                          //  8        , False       , True         , False        , False        , False
DXIL::InterpolationMode::LinearSample,                    //  9        , False       , True         , False        , False        , True
DXIL::InterpolationMode::LinearCentroid,                  // 10        , False       , True         , False        , True        , False
DXIL::InterpolationMode::LinearSample,                    // 11        , False       , True         , False        , True        , True
DXIL::InterpolationMode::LinearNoperspective,             // 12        , False       , True         , True        , False        , False
DXIL::InterpolationMode::LinearNoperspectiveSample,       // 13        , False       , True         , True        , False        , True
DXIL::InterpolationMode::LinearNoperspectiveCentroid,     // 14        , False       , True         , True        , True        , False
DXIL::InterpolationMode::LinearNoperspectiveSample,       // 15        , False       , True         , True        , True        , True
DXIL::InterpolationMode::Constant,                        // 16        , True        , False        , False        , False        , False
DXIL::InterpolationMode::Invalid,                         // 17        , True        , False        , False        , False        , True
DXIL::InterpolationMode::Invalid,                         // 18        , True        , False        , False        , True        , False
DXIL::InterpolationMode::Invalid,                         // 19        , True        , False        , False        , True        , True
DXIL::InterpolationMode::Invalid,                         // 20        , True        , False        , True        , False        , False
DXIL::InterpolationMode::Invalid,                         // 21        , True        , False        , True        , False        , True
DXIL::InterpolationMode::Invalid,                         // 22        , True        , False        , True        , True        , False
DXIL::InterpolationMode::Invalid,                         // 23        , True        , False        , True        , True        , True
DXIL::InterpolationMode::Invalid,                         // 24        , True        , True        , False        , False        , False
DXIL::InterpolationMode::Invalid,                         // 25        , True        , True        , False        , False        , True
DXIL::InterpolationMode::Invalid,                         // 26        , True        , True        , False        , True        , False
DXIL::InterpolationMode::Invalid,                         // 27        , True        , True        , False        , True        , True
DXIL::InterpolationMode::Invalid,                         // 28        , True        , True        , True        , False        , False
DXIL::InterpolationMode::Invalid,                         // 29        , True        , True        , True        , False        , True
DXIL::InterpolationMode::Invalid,                         // 30        , True        , True        , True        , True        , False
DXIL::InterpolationMode::Invalid,                         // 31        , True        , True        , True        , True        , True
};

InterpolationMode::InterpolationMode(bool bNoInterpolation, bool bLinear, bool bNoperspective, bool bCentroid, bool bSample) {
  unsigned mask = (unsigned)bNoInterpolation << 4;
  mask |= ((unsigned)bLinear) << 3;
  mask |= ((unsigned)bNoperspective) << 2;
  mask |= ((unsigned)bCentroid) << 1;
  mask |= ((unsigned)bSample);

  m_Kind = interpModeTab[mask];

  // interpModeTab is generate from following code.
  //m_Kind = Kind::Invalid; // Cases not set below are invalid.
  //bool bAnyLinear = bLinear | bNoperspective | bCentroid | bSample;
  //if (bNoInterpolation) {
  //  if (!bAnyLinear) m_Kind = Kind::Constant;
  //}
  //else if (bAnyLinear) {
  //  if (bSample) {    // warning case: sample overrides centroid.
  //    if (bNoperspective) m_Kind = Kind::LinearNoperspectiveSample;
  //    else                m_Kind = Kind::LinearSample;
  //  }
  //  else {
  //    if      (!bNoperspective  && !bCentroid) m_Kind = Kind::Linear;
  //    else if (!bNoperspective  &&  bCentroid) m_Kind = Kind::LinearCentroid;
  //    else if ( bNoperspective  && !bCentroid) m_Kind = Kind::LinearNoperspective;
  //    else if ( bNoperspective  &&  bCentroid) m_Kind = Kind::LinearNoperspectiveCentroid;
  //  }
  //}
  //else {
  //  m_Kind = Kind::Undefined;
  //}
}

InterpolationMode &InterpolationMode::operator=(const InterpolationMode &o) {
  if (this != &o) {
    m_Kind = o.m_Kind;
  }

  return *this;
}

bool InterpolationMode::operator==(const InterpolationMode &o) const {
  return m_Kind == o.m_Kind;
}

bool InterpolationMode::IsAnyLinear() const {
  return m_Kind < Kind::Invalid && m_Kind != Kind::Undefined && m_Kind != Kind::Constant;
}

bool InterpolationMode::IsAnyNoPerspective() const {
  return IsLinearNoperspective() || IsLinearNoperspectiveCentroid() || IsLinearNoperspectiveSample();
}

bool InterpolationMode::IsAnyCentroid() const {
  return IsLinearCentroid() || IsLinearNoperspectiveCentroid();
}

bool InterpolationMode::IsAnySample() const {
  return IsLinearSample() || IsLinearNoperspectiveSample();
}

const char *InterpolationMode::GetName() const {
  switch (m_Kind) {
  case Kind::Undefined:                     return "";
  case Kind::Constant:                      return "nointerpolation";
  case Kind::Linear:                        return "linear";
  case Kind::LinearCentroid:                return "centroid";
  case Kind::LinearNoperspective:           return "noperspective";
  case Kind::LinearNoperspectiveCentroid:   return "noperspective centroid";
  case Kind::LinearSample:                  return "sample";
  case Kind::LinearNoperspectiveSample:     return "noperspective sample";
  default: DXASSERT(false, "invalid interpolation mode"); return "invalid";
  }
}

} // namespace hlsl
