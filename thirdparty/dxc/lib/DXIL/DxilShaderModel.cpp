///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilShaderModel.cpp                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <limits.h>

#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSemantic.h"
#include "dxc/Support/Global.h"
#include <unordered_map>


namespace hlsl {

ShaderModel::ShaderModel(Kind Kind, unsigned Major, unsigned Minor, const char *pszName,
                         unsigned NumInputRegs, unsigned NumOutputRegs,
                         bool bUAVs, bool bTypedUavs,  unsigned NumUAVRegs)
: m_Kind(Kind)
, m_Major(Major)
, m_Minor(Minor)
, m_pszName(pszName)
, m_NumInputRegs(NumInputRegs)
, m_NumOutputRegs(NumOutputRegs)
, m_bTypedUavs(bTypedUavs)
, m_NumUAVRegs(NumUAVRegs) {
}

bool ShaderModel::operator==(const ShaderModel &other) const {
    return m_Kind          == other.m_Kind
        && m_Major         == other.m_Major
        && m_Minor         == other.m_Minor
        && strcmp(m_pszName,  other.m_pszName) == 0
        && m_NumInputRegs  == other.m_NumInputRegs
        && m_NumOutputRegs == other.m_NumOutputRegs
        && m_bTypedUavs    == other.m_bTypedUavs
        && m_NumUAVRegs    == other.m_NumUAVRegs;
}

bool ShaderModel::IsValid() const {
  DXASSERT(IsPS() || IsVS() || IsGS() || IsHS() || IsDS() || IsCS() ||
               IsLib() || IsMS() || IsAS() || m_Kind == Kind::Invalid,
           "invalid shader model");
  return m_Kind != Kind::Invalid;
}

bool ShaderModel::IsValidForDxil() const {
  if (!IsValid())
    return false;
  switch (m_Major) {
    case 6: {
      switch (m_Minor) {
      /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_is_valid_for_dxil()</py>*/
      // VALRULE-TEXT:BEGIN
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      // VALRULE-TEXT:END
        return true;
      case kOfflineMinor:
        return m_Kind == Kind::Library;
      }
    }
    break;
  }
  return false;
}

bool ShaderModel::IsValidForModule() const {
  // Ray tracing shader model should only be used on functions in a lib
  return IsValid() && !IsRay();
}

const ShaderModel *ShaderModel::Get(Kind Kind, unsigned Major, unsigned Minor) {
  /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_shader_model_get()</py>*/
  // VALRULE-TEXT:BEGIN
  const static std::unordered_map<unsigned, unsigned> hashToIdxMap = {
  {1024,0}, //ps_4_0
  {1025,1}, //ps_4_1
  {1280,2}, //ps_5_0
  {1281,3}, //ps_5_1
  {1536,4}, //ps_6_0
  {1537,5}, //ps_6_1
  {1538,6}, //ps_6_2
  {1539,7}, //ps_6_3
  {1540,8}, //ps_6_4
  {1541,9}, //ps_6_5
  {1542,10}, //ps_6_6
  {1543,11}, //ps_6_7
  {66560,12}, //vs_4_0
  {66561,13}, //vs_4_1
  {66816,14}, //vs_5_0
  {66817,15}, //vs_5_1
  {67072,16}, //vs_6_0
  {67073,17}, //vs_6_1
  {67074,18}, //vs_6_2
  {67075,19}, //vs_6_3
  {67076,20}, //vs_6_4
  {67077,21}, //vs_6_5
  {67078,22}, //vs_6_6
  {67079,23}, //vs_6_7
  {132096,24}, //gs_4_0
  {132097,25}, //gs_4_1
  {132352,26}, //gs_5_0
  {132353,27}, //gs_5_1
  {132608,28}, //gs_6_0
  {132609,29}, //gs_6_1
  {132610,30}, //gs_6_2
  {132611,31}, //gs_6_3
  {132612,32}, //gs_6_4
  {132613,33}, //gs_6_5
  {132614,34}, //gs_6_6
  {132615,35}, //gs_6_7
  {197888,36}, //hs_5_0
  {197889,37}, //hs_5_1
  {198144,38}, //hs_6_0
  {198145,39}, //hs_6_1
  {198146,40}, //hs_6_2
  {198147,41}, //hs_6_3
  {198148,42}, //hs_6_4
  {198149,43}, //hs_6_5
  {198150,44}, //hs_6_6
  {198151,45}, //hs_6_7
  {263424,46}, //ds_5_0
  {263425,47}, //ds_5_1
  {263680,48}, //ds_6_0
  {263681,49}, //ds_6_1
  {263682,50}, //ds_6_2
  {263683,51}, //ds_6_3
  {263684,52}, //ds_6_4
  {263685,53}, //ds_6_5
  {263686,54}, //ds_6_6
  {263687,55}, //ds_6_7
  {328704,56}, //cs_4_0
  {328705,57}, //cs_4_1
  {328960,58}, //cs_5_0
  {328961,59}, //cs_5_1
  {329216,60}, //cs_6_0
  {329217,61}, //cs_6_1
  {329218,62}, //cs_6_2
  {329219,63}, //cs_6_3
  {329220,64}, //cs_6_4
  {329221,65}, //cs_6_5
  {329222,66}, //cs_6_6
  {329223,67}, //cs_6_7
  {394753,68}, //lib_6_1
  {394754,69}, //lib_6_2
  {394755,70}, //lib_6_3
  {394756,71}, //lib_6_4
  {394757,72}, //lib_6_5
  {394758,73}, //lib_6_6
  {394759,74}, //lib_6_7
  // lib_6_x is for offline linking only, and relaxes restrictions
  {394767,75},//lib_6_x
  {853509,76}, //ms_6_5
  {853510,77}, //ms_6_6
  {853511,78}, //ms_6_7
  {919045,79}, //as_6_5
  {919046,80}, //as_6_6
  {919047,81}, //as_6_7
  };
  unsigned hash = (unsigned)Kind << 16 | Major << 8 | Minor;
  auto it = hashToIdxMap.find(hash);
  if (it == hashToIdxMap.end())
    return GetInvalid();
  return &ms_ShaderModels[it->second];
  // VALRULE-TEXT:END
}

const ShaderModel *ShaderModel::GetByName(const char *pszName) {
  // [ps|vs|gs|hs|ds|cs|ms|as]_[major]_[minor]
  Kind kind;
  switch (pszName[0]) {
  case 'p':   kind = Kind::Pixel;     break;
  case 'v':   kind = Kind::Vertex;    break;
  case 'g':   kind = Kind::Geometry;  break;
  case 'h':   kind = Kind::Hull;      break;
  case 'd':   kind = Kind::Domain;    break;
  case 'c':   kind = Kind::Compute;   break;
  case 'l':   kind = Kind::Library;   break;
  case 'm':   kind = Kind::Mesh;      break;
  case 'a':   kind = Kind::Amplification; break;
  default:    return GetInvalid();
  }
  unsigned Idx = 3;
  if (kind != Kind::Library) {
    if (pszName[1] != 's' || pszName[2] != '_')
      return GetInvalid();
  } else {
    if (pszName[1] != 'i' || pszName[2] != 'b' || pszName[3] != '_')
      return GetInvalid();
    Idx = 4;
  }

  unsigned Major;
  switch (pszName[Idx++]) {
  case '4': Major = 4;  break;
  case '5': Major = 5;  break;
  case '6': Major = 6;  break;
  default:  return GetInvalid();
  }
  if (pszName[Idx++] != '_')
    return GetInvalid();

  unsigned Minor;
  switch (pszName[Idx++]) {
    case '0': Minor = 0;  break;
    case '1': Minor = 1;  break;
  /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_shader_model_by_name()</py>*/
  // VALRULE-TEXT:BEGIN
  case '2':
    if (Major == 6) {
      Minor = 2;
      break;
    }
  else return GetInvalid();
  case '3':
    if (Major == 6) {
      Minor = 3;
      break;
    }
  else return GetInvalid();
  case '4':
    if (Major == 6) {
      Minor = 4;
      break;
    }
  else return GetInvalid();
  case '5':
    if (Major == 6) {
      Minor = 5;
      break;
    }
  else return GetInvalid();
  case '6':
    if (Major == 6) {
      Minor = 6;
      break;
    }
  else return GetInvalid();
  case '7':
    if (Major == 6) {
      Minor = 7;
      break;
    }
  else return GetInvalid();
  // VALRULE-TEXT:END
    case 'x':
      if (kind == Kind::Library && Major == 6) {
        Minor = kOfflineMinor;
        break;
      }
      else return GetInvalid();
    default:  return GetInvalid();
  }
  if (pszName[Idx++] != 0)
    return GetInvalid();

  return Get(kind, Major, Minor);
}

void ShaderModel::GetDxilVersion(unsigned &DxilMajor, unsigned &DxilMinor) const {
  DXASSERT(IsValidForDxil(), "invalid shader model");
  DxilMajor = 1;
  switch (m_Minor) {
  /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_dxil_version()</py>*/
  // VALRULE-TEXT:BEGIN
  case 0:
    DxilMinor = 0;
    break;
  case 1:
    DxilMinor = 1;
    break;
  case 2:
    DxilMinor = 2;
    break;
  case 3:
    DxilMinor = 3;
    break;
  case 4:
    DxilMinor = 4;
    break;
  case 5:
    DxilMinor = 5;
    break;
  case 6:
    DxilMinor = 6;
    break;
  case 7:
    DxilMinor = 7;
    break;
  case kOfflineMinor: // Always update this to highest dxil version
    DxilMinor = 7;
    break;
  // VALRULE-TEXT:END
  default:
    DXASSERT(0, "IsValidForDxil() should have caught this.");
    break;
  }
}

void ShaderModel::GetMinValidatorVersion(unsigned &ValMajor, unsigned &ValMinor) const {
  DXASSERT(IsValidForDxil(), "invalid shader model");
  ValMajor = 1;
  switch (m_Minor) {
  /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_min_validator_version()</py>*/
  // VALRULE-TEXT:BEGIN
  case 0:
    ValMinor = 0;
    break;
  case 1:
    ValMinor = 1;
    break;
  case 2:
    ValMinor = 2;
    break;
  case 3:
    ValMinor = 3;
    break;
  case 4:
    ValMinor = 4;
    break;
  case 5:
    ValMinor = 5;
    break;
  case 6:
    ValMinor = 6;
    break;
  case 7:
    ValMinor = 7;
    break;
  // VALRULE-TEXT:END
  case kOfflineMinor:
    ValMajor = 0;
    ValMinor = 0;
    break;
  default:
    DXASSERT(0, "IsValidForDxil() should have caught this.");
    break;
  }
}

static const char *ShaderModelKindNames[] = {
    "ps", "vs", "gs", "hs", "ds", "cs", "lib",
    "raygeneration", "intersection", "anyhit", "closesthit", "miss", "callable",
    "ms", "as", "invalid",
};

const char * ShaderModel::GetKindName() const {
  return GetKindName(m_Kind);
}

const char *ShaderModel::GetKindName(Kind kind) {
  static_assert(static_cast<unsigned>(Kind::Invalid) ==
                    _countof(ShaderModelKindNames) - 1,
                "Invalid kinds or names");
  return ShaderModelKindNames[static_cast<unsigned int>(kind)];
}

const ShaderModel *ShaderModel::GetInvalid() {
  return &ms_ShaderModels[kNumShaderModels - 1];
}

typedef ShaderModel SM;
typedef Semantic SE;
const ShaderModel ShaderModel::ms_ShaderModels[kNumShaderModels] = {
  //                                  IR  OR   UAV?   TyUAV? UAV base
  /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_shader_models()</py>*/
  // VALRULE-TEXT:BEGIN
  SM(Kind::Pixel, 4, 0, "ps_4_0", 32, 8, false, false, 0),
  SM(Kind::Pixel, 4, 1, "ps_4_1", 32, 8, false, false, 0),
  SM(Kind::Pixel, 5, 0, "ps_5_0", 32, 8, true, true, 64),
  SM(Kind::Pixel, 5, 1, "ps_5_1", 32, 8, true, true, 64),
  SM(Kind::Pixel, 6, 0, "ps_6_0", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 1, "ps_6_1", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 2, "ps_6_2", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 3, "ps_6_3", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 4, "ps_6_4", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 5, "ps_6_5", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 6, "ps_6_6", 32, 8, true, true, UINT_MAX),
  SM(Kind::Pixel, 6, 7, "ps_6_7", 32, 8, true, true, UINT_MAX),
  SM(Kind::Vertex, 4, 0, "vs_4_0", 16, 16, false, false, 0),
  SM(Kind::Vertex, 4, 1, "vs_4_1", 32, 32, false, false, 0),
  SM(Kind::Vertex, 5, 0, "vs_5_0", 32, 32, true, true, 64),
  SM(Kind::Vertex, 5, 1, "vs_5_1", 32, 32, true, true, 64),
  SM(Kind::Vertex, 6, 0, "vs_6_0", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 1, "vs_6_1", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 2, "vs_6_2", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 3, "vs_6_3", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 4, "vs_6_4", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 5, "vs_6_5", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 6, "vs_6_6", 32, 32, true, true, UINT_MAX),
  SM(Kind::Vertex, 6, 7, "vs_6_7", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 4, 0, "gs_4_0", 16, 32, false, false, 0),
  SM(Kind::Geometry, 4, 1, "gs_4_1", 32, 32, false, false, 0),
  SM(Kind::Geometry, 5, 0, "gs_5_0", 32, 32, true, true, 64),
  SM(Kind::Geometry, 5, 1, "gs_5_1", 32, 32, true, true, 64),
  SM(Kind::Geometry, 6, 0, "gs_6_0", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 1, "gs_6_1", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 2, "gs_6_2", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 3, "gs_6_3", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 4, "gs_6_4", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 5, "gs_6_5", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 6, "gs_6_6", 32, 32, true, true, UINT_MAX),
  SM(Kind::Geometry, 6, 7, "gs_6_7", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 5, 0, "hs_5_0", 32, 32, true, true, 64),
  SM(Kind::Hull, 5, 1, "hs_5_1", 32, 32, true, true, 64),
  SM(Kind::Hull, 6, 0, "hs_6_0", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 1, "hs_6_1", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 2, "hs_6_2", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 3, "hs_6_3", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 4, "hs_6_4", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 5, "hs_6_5", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 6, "hs_6_6", 32, 32, true, true, UINT_MAX),
  SM(Kind::Hull, 6, 7, "hs_6_7", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 5, 0, "ds_5_0", 32, 32, true, true, 64),
  SM(Kind::Domain, 5, 1, "ds_5_1", 32, 32, true, true, 64),
  SM(Kind::Domain, 6, 0, "ds_6_0", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 1, "ds_6_1", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 2, "ds_6_2", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 3, "ds_6_3", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 4, "ds_6_4", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 5, "ds_6_5", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 6, "ds_6_6", 32, 32, true, true, UINT_MAX),
  SM(Kind::Domain, 6, 7, "ds_6_7", 32, 32, true, true, UINT_MAX),
  SM(Kind::Compute, 4, 0, "cs_4_0", 0, 0, false, false, 0),
  SM(Kind::Compute, 4, 1, "cs_4_1", 0, 0, false, false, 0),
  SM(Kind::Compute, 5, 0, "cs_5_0", 0, 0, true, true, 64),
  SM(Kind::Compute, 5, 1, "cs_5_1", 0, 0, true, true, 64),
  SM(Kind::Compute, 6, 0, "cs_6_0", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 1, "cs_6_1", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 2, "cs_6_2", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 3, "cs_6_3", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 4, "cs_6_4", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 5, "cs_6_5", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 6, "cs_6_6", 0, 0, true, true, UINT_MAX),
  SM(Kind::Compute, 6, 7, "cs_6_7", 0, 0, true, true, UINT_MAX),
  SM(Kind::Library, 6, 1, "lib_6_1", 32, 32, true, true, UINT_MAX),
  SM(Kind::Library, 6, 2, "lib_6_2", 32, 32, true, true, UINT_MAX),
  SM(Kind::Library, 6, 3, "lib_6_3", 32, 32, true, true, UINT_MAX),
  SM(Kind::Library, 6, 4, "lib_6_4", 32, 32, true, true, UINT_MAX),
  SM(Kind::Library, 6, 5, "lib_6_5", 32, 32, true, true, UINT_MAX),
  SM(Kind::Library, 6, 6, "lib_6_6", 32, 32, true, true, UINT_MAX),
  SM(Kind::Library, 6, 7, "lib_6_7", 32, 32, true, true, UINT_MAX),
  // lib_6_x is for offline linking only, and relaxes restrictions
  SM(Kind::Library,  6, kOfflineMinor, "lib_6_x",  32, 32,  true,  true,  UINT_MAX),
  SM(Kind::Mesh, 6, 5, "ms_6_5", 0, 0, true, true, UINT_MAX),
  SM(Kind::Mesh, 6, 6, "ms_6_6", 0, 0, true, true, UINT_MAX),
  SM(Kind::Mesh, 6, 7, "ms_6_7", 0, 0, true, true, UINT_MAX),
  SM(Kind::Amplification, 6, 5, "as_6_5", 0, 0, true, true, UINT_MAX),
  SM(Kind::Amplification, 6, 6, "as_6_6", 0, 0, true, true, UINT_MAX),
  SM(Kind::Amplification, 6, 7, "as_6_7", 0, 0, true, true, UINT_MAX),
  // Values before Invalid must remain sorted by Kind, then Major, then Minor.
  SM(Kind::Invalid,  0, 0, "invalid", 0,  0,   false, false, 0),
  // VALRULE-TEXT:END
};

} // namespace hlsl
