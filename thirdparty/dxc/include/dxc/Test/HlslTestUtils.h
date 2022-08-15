///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HlslTestUtils.h                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides utility functions for HLSL tests.                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#include <string>
#include <sstream>
#include <fstream>
#include <atomic>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <dxgiformat.h>
#include "WexTestClass.h"
#else
#include "dxc/Support/Global.h" // DXASSERT_LOCALVAR
#include "WEXAdapter.h"
#endif
#include "dxc/Support/Unicode.h"
#include "dxc/DXIL/DxilConstants.h" // DenormMode

using namespace std;

#ifndef HLSLDATAFILEPARAM
#define HLSLDATAFILEPARAM L"HlslDataDir"
#endif

#ifndef FILECHECKDUMPDIRPARAM
#define FILECHECKDUMPDIRPARAM L"FileCheckDumpDir"
#endif

// If TAEF verify macros are available, use them to alias other legacy
// comparison macros that don't have a direct translation.
//
// Other common replacements are as follows.
//
// EXPECT_EQ -> VERIFY_ARE_EQUAL
// ASSERT_EQ -> VERIFY_ARE_EQUAL
//
// Note that whether verification throws or continues depends on
// preprocessor settings.

#ifdef VERIFY_ARE_EQUAL
#ifndef EXPECT_STREQ
#define EXPECT_STREQ(a, b) VERIFY_ARE_EQUAL(0, strcmp(a, b))
#endif
#define EXPECT_STREQW(a, b) VERIFY_ARE_EQUAL(0, wcscmp(a, b))
#define VERIFY_ARE_EQUAL_CMP(a, b, ...) VERIFY_IS_TRUE(a == b, __VA_ARGS__)
#define VERIFY_ARE_EQUAL_STR(a, b) { \
  const char *pTmpA = (a);\
  const char *pTmpB = (b);\
  if (0 != strcmp(pTmpA, pTmpB)) {\
    CA2W conv(pTmpB, CP_UTF8); WEX::Logging::Log::Comment(conv);\
    const char *pA = pTmpA; const char *pB = pTmpB; \
    while(*pA == *pB) { pA++; pB++; } \
    wchar_t diffMsg[32]; swprintf_s(diffMsg, _countof(diffMsg), L"diff at %u", (unsigned)(pA-pTmpA)); \
    WEX::Logging::Log::Comment(diffMsg); \
  } \
  VERIFY_ARE_EQUAL(0, strcmp(pTmpA, pTmpB)); \
}
#define VERIFY_ARE_EQUAL_WSTR(a, b) { \
  if (0 != wcscmp(a, b)) { WEX::Logging::Log::Comment(b);} \
  VERIFY_ARE_EQUAL(0, wcscmp(a, b)); \
}
#ifndef ASSERT_EQ
#define ASSERT_EQ(expected, actual) VERIFY_ARE_EQUAL(expected, actual)
#endif
#ifndef ASSERT_NE
#define ASSERT_NE(expected, actual) VERIFY_ARE_NOT_EQUAL(expected, actual)
#endif
#ifndef TEST_F
#define TEST_F(typeName, functionName) void typeName::functionName()
#endif
#define ASSERT_HRESULT_SUCCEEDED VERIFY_SUCCEEDED
#ifndef EXPECT_EQ
#define EXPECT_EQ(expected, actual) VERIFY_ARE_EQUAL(expected, actual)
#endif 
#endif // VERIFY_ARE_EQUAL

static constexpr char whitespaceChars[] = " \t\r\n";

inline std::string strltrim(const std::string &value) {
  size_t first = value.find_first_not_of(whitespaceChars);
  return first == string::npos ? value : value.substr(first);
}

inline std::string strrtrim(const std::string &value) {
  size_t last = value.find_last_not_of(whitespaceChars);
  return last == string::npos ? value : value.substr(0, last + 1);
}

inline std::string strtrim(const std::string &value) {
  return strltrim(strrtrim(value));
}

inline bool strstartswith(const std::string& value, const char* pattern) {
  for (size_t i = 0; ; ++i) {
    if (pattern[i] == '\0') return true;
    if (i == value.size() || value[i] != pattern[i]) return false;
  }
}

inline std::vector<std::string> strtok(const std::string &value, const char *delimiters = whitespaceChars) {
  size_t searchOffset = 0;
  std::vector<std::string> tokens;
  while (searchOffset != value.size()) {
    size_t tokenStartIndex = value.find_first_not_of(delimiters, searchOffset);
    if (tokenStartIndex == std::string::npos) break;
    size_t tokenEndIndex = value.find_first_of(delimiters, tokenStartIndex);
    if (tokenEndIndex == std::string::npos) tokenEndIndex = value.size();
    tokens.emplace_back(value.substr(tokenStartIndex, tokenEndIndex - tokenStartIndex));
    searchOffset = tokenEndIndex;
  }
  return tokens;
}

namespace hlsl_test {

inline std::wstring
vFormatToWString(_In_z_ _Printf_format_string_ const wchar_t *fmt, va_list argptr) {
  std::wstring result;
#ifdef _WIN32
  int len = _vscwprintf(fmt, argptr);
  result.resize(len + 1);
  vswprintf_s((wchar_t *)result.data(), len + 1, fmt, argptr);
#else
  wchar_t fmtOut[1000];
  int len = vswprintf(fmtOut, 1000, fmt, argptr);
  DXASSERT_LOCALVAR(len, len >= 0,
                    "Too long formatted string in vFormatToWstring");
  result = fmtOut;
#endif
  return result;
}

inline std::wstring
FormatToWString(_In_z_ _Printf_format_string_ const wchar_t *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::wstring result(vFormatToWString(fmt, args));
  va_end(args);
  return result;
}

inline void LogCommentFmt(_In_z_ _Printf_format_string_ const wchar_t *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::wstring buf(vFormatToWString(fmt, args));
  va_end(args);
  WEX::Logging::Log::Comment(buf.data());
}

inline void LogErrorFmt(_In_z_ _Printf_format_string_ const wchar_t *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::wstring buf(vFormatToWString(fmt, args));
    va_end(args);
    WEX::Logging::Log::Error(buf.data());
}

inline std::wstring GetPathToHlslDataFile(const wchar_t* relative, LPCWSTR paramName = HLSLDATAFILEPARAM) {
  WEX::TestExecution::SetVerifyOutput verifySettings(WEX::TestExecution::VerifyOutputSettings::LogOnlyFailures);
  WEX::Common::String HlslDataDirValue;
  if (std::wstring(paramName).compare(HLSLDATAFILEPARAM) != 0) {
    // Not fatal, for instance, FILECHECKDUMPDIRPARAM will dump files before running FileCheck, so they can be compared run to run
    if (FAILED(WEX::TestExecution::RuntimeParameters::TryGetValue(paramName, HlslDataDirValue)))
      return std::wstring();
  } else {
    ASSERT_HRESULT_SUCCEEDED(WEX::TestExecution::RuntimeParameters::TryGetValue(HLSLDATAFILEPARAM, HlslDataDirValue));
  }

  wchar_t envPath[MAX_PATH];
  wchar_t expanded[MAX_PATH];
  swprintf_s(envPath, _countof(envPath), L"%ls\\%ls", reinterpret_cast<const wchar_t*>(HlslDataDirValue.GetBuffer()), relative);
  VERIFY_WIN32_BOOL_SUCCEEDED(ExpandEnvironmentStringsW(envPath, expanded, _countof(expanded)));
  return std::wstring(expanded);
}

inline bool PathLooksAbsolute(LPCWSTR name) {
  // Very simplified, only for the cases we care about in the test suite.
#ifdef _WIN32
  return name && *name && ((*name == L'\\') || (name[1] == L':'));
#else
  return name && *name && (*name == L'/');
#endif
}

static bool HasRunLine(std::string &line) {
  const char *delimiters = " ;/";
  auto lineelems = strtok(line, delimiters);
  return !lineelems.empty() &&
    lineelems.front().compare("RUN:") == 0;
}

inline std::vector<std::string> GetRunLines(const LPCWSTR name) {
  const std::wstring path = PathLooksAbsolute(name)
    ? std::wstring(name)
    : hlsl_test::GetPathToHlslDataFile(name);
#ifdef _WIN32
  std::ifstream infile(path);
#else
  std::ifstream infile((CW2A(path.c_str())));
#endif
  if (infile.bad()) {
    std::wstring errMsg(L"Unable to read file ");
    errMsg += path;
    WEX::Logging::Log::Error(errMsg.c_str());
    VERIFY_FAIL();
  }

  std::vector<std::string> runlines;
  std::string line;
  constexpr size_t runlinesize = 300;
  while (std::getline(infile, line)) {
    if (!HasRunLine(line))
      continue;
    char runline[runlinesize];
    memset(runline, 0, runlinesize);
    memcpy(runline, line.c_str(), min(runlinesize, line.size()));
    runlines.emplace_back(runline);
  }
  return runlines;
}

inline std::string GetFirstLine(LPCWSTR name) {
  char firstLine[300];
  memset(firstLine, 0, sizeof(firstLine));

  const std::wstring path = PathLooksAbsolute(name)
                                ? std::wstring(name)
                                : hlsl_test::GetPathToHlslDataFile(name);
#ifdef _WIN32
  std::ifstream infile(path);
#else
  std::ifstream infile((CW2A(path.c_str())));
#endif
  if (infile.bad()) {
    std::wstring errMsg(L"Unable to read file ");
    errMsg += path;
    WEX::Logging::Log::Error(errMsg.c_str());
    VERIFY_FAIL();
  }

  infile.getline(firstLine, _countof(firstLine));
  return firstLine;
}

inline HANDLE CreateFileForReading(LPCWSTR path) {
  HANDLE sourceHandle = CreateFileW(path, GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0);
  if (sourceHandle == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    std::wstring errorMessage(FormatToWString(L"Unable to open file '%s', err=%u", path, err).c_str());
    VERIFY_SUCCEEDED(HRESULT_FROM_WIN32(err), errorMessage.c_str());
  }
  return sourceHandle;
}

inline HANDLE CreateNewFileForReadWrite(LPCWSTR path) {
  HANDLE sourceHandle = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
  if (sourceHandle == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    std::wstring errorMessage(FormatToWString(L"Unable to create file '%s', err=%u", path, err).c_str());
    VERIFY_SUCCEEDED(HRESULT_FROM_WIN32(err), errorMessage.c_str());
  }
  return sourceHandle;
}

inline bool GetTestParamBool(LPCWSTR name) {
  WEX::Common::String ParamValue;
  WEX::Common::String NameValue;
  if (FAILED(WEX::TestExecution::RuntimeParameters::TryGetValue(name,
                                                                ParamValue))) {
    return false;
  }
  if (ParamValue.IsEmpty()) {
    return false;
  }
  if (0 == wcscmp(ParamValue, L"*")) {
    return true;
  }
  VERIFY_SUCCEEDED(WEX::TestExecution::RuntimeParameters::TryGetValue(
      L"TestName", NameValue));
  if (NameValue.IsEmpty()) {
    return false;
  }
  return Unicode::IsStarMatchWide(ParamValue, ParamValue.GetLength(),
                                  NameValue, NameValue.GetLength());
}

inline bool GetTestParamUseWARP(bool defaultVal) {
  WEX::Common::String AdapterValue;
  if (FAILED(WEX::TestExecution::RuntimeParameters::TryGetValue(
        L"Adapter", AdapterValue))) {
    return defaultVal;
  }
  if ((defaultVal && AdapterValue.IsEmpty()) ||
      AdapterValue.CompareNoCase(L"WARP") == 0) {
    return true;
  }
  return false;
}

}

#ifdef FP_SUBNORMAL

inline bool isdenorm(float f) {
  return FP_SUBNORMAL == std::fpclassify(f);
}

#else

inline bool isdenorm(float f) {
  return (std::numeric_limits<float>::denorm_min() <= f && f < std::numeric_limits<float>::min()) ||
         (-std::numeric_limits<float>::min() < f && f <= -std::numeric_limits<float>::denorm_min());
}

#endif // FP_SUBNORMAL

inline float ifdenorm_flushf(float a) {
  return isdenorm(a) ? copysign(0.0f, a) : a;
}

inline bool ifdenorm_flushf_eq(float a, float b) {
  return ifdenorm_flushf(a) == ifdenorm_flushf(b);
}

static const uint16_t Float16NaN = 0xff80;
static const uint16_t Float16PosInf = 0x7c00;
static const uint16_t Float16NegInf = 0xfc00;
static const uint16_t Float16PosDenorm = 0x0008;
static const uint16_t Float16NegDenorm = 0x8008;
static const uint16_t Float16PosZero = 0x0000;
static const uint16_t Float16NegZero = 0x8000;

inline bool GetSign(float x) {
  return std::signbit(x);
}

inline int GetMantissa(float x) {
  int bits = reinterpret_cast<int &>(x);
  return bits & 0x7fffff;
}

inline int GetExponent(float x) {
  int bits = reinterpret_cast<int &>(x);
  return (bits >> 23) & 0xff;
}

#define FLOAT16_BIT_SIGN 0x8000
#define FLOAT16_BIT_EXP 0x7c00
#define FLOAT16_BIT_MANTISSA 0x03ff
#define FLOAT16_BIGGEST_DENORM FLOAT16_BIT_MANTISSA
#define FLOAT16_BIGGEST_NORMAL 0x7bff

inline bool isnanFloat16(uint16_t val) {
  return (val & FLOAT16_BIT_EXP) == FLOAT16_BIT_EXP &&
         (val & FLOAT16_BIT_MANTISSA) != 0;
}

inline uint16_t ConvertFloat32ToFloat16(float val) {
  union Bits {
    uint32_t u_bits;
    float f_bits;
  };

  static const uint32_t SignMask = 0x8000;

  // Minimum f32 value representable in f16 format without denormalizing
  static const uint32_t Min16in32 = 0x38800000;

  // Maximum f32 value (next to infinity)
  static const uint32_t Max32 = 0x7f7FFFFF;

  // Mask for f32 mantissa
  static const uint32_t Fraction32Mask = 0x007FFFFF;

  // pow(2,24)
  static const uint32_t DenormalRatio = 0x4B800000;

  static const uint32_t NormalDelta = 0x38000000;

  Bits bits;
  bits.f_bits = val;
  uint32_t sign = bits.u_bits & (SignMask << 16);
  Bits Abs;
  Abs.u_bits = bits.u_bits ^ sign;

  bool isLessThanNormal = Abs.f_bits < *(const float*)&Min16in32;
  bool isInfOrNaN = Abs.u_bits > Max32;

  if (isLessThanNormal) {
    // Compute Denormal result
    return (uint16_t)(Abs.f_bits * *(const float*)(&DenormalRatio)) | (uint16_t)(sign >> 16);
  }
  else if (isInfOrNaN) {
    // Compute Inf or Nan result
    uint32_t Fraction = Abs.u_bits & Fraction32Mask;
    uint16_t IsNaN = Fraction == 0 ? 0 : 0xffff;
    return (IsNaN & FLOAT16_BIT_MANTISSA) | FLOAT16_BIT_EXP | (uint16_t)(sign >> 16);
  }
  else {
    // Compute Normal result
    return (uint16_t)((Abs.u_bits - NormalDelta) >> 13) | (uint16_t)(sign >> 16);
  }
}

inline float ConvertFloat16ToFloat32(uint16_t x) {
 union Bits {
    float f_bits;
    uint32_t u_bits;
  };

  uint32_t Sign = (x & FLOAT16_BIT_SIGN) << 16;

  // nan -> exponent all set and mantisa is non zero
  // +/-inf -> exponent all set and mantissa is zero
  // denorm -> exponent zero and significand nonzero
  uint32_t Abs = (x & 0x7fff);
  uint32_t IsNormal = Abs > FLOAT16_BIGGEST_DENORM;
  uint32_t IsInfOrNaN = Abs > FLOAT16_BIGGEST_NORMAL;

  // Signless Result for normals
  uint32_t DenormRatio = 0x33800000;
  float DenormResult = Abs * (*(float*)&DenormRatio);

  uint32_t AbsShifted = Abs << 13;
  // Signless Result for normals
  uint32_t NormalResult = AbsShifted + 0x38000000;
  // Signless Result for int & nans
  uint32_t InfResult = AbsShifted + 0x70000000;

  Bits bits;
  bits.u_bits = 0;
  if (IsInfOrNaN)
    bits.u_bits |= InfResult;
  else if (IsNormal)
    bits.u_bits |= NormalResult;
  else
    bits.f_bits = DenormResult;
  bits.u_bits |= Sign;
  return bits.f_bits;
}
uint16_t ConvertFloat32ToFloat16(float val);
float ConvertFloat16ToFloat32(uint16_t val);

inline bool CompareFloatULP(const float &fsrc, const float &fref, int ULPTolerance,
                            hlsl::DXIL::Float32DenormMode mode = hlsl::DXIL::Float32DenormMode::Any) {
  if (fsrc == fref) {
    return true;
  }
  if (std::isnan(fsrc)) {
    return std::isnan(fref);
  }
  if (mode == hlsl::DXIL::Float32DenormMode::Any) {
    // If denorm expected, output can be sign preserved zero. Otherwise output
    // should pass the regular ulp testing.
    if (isdenorm(fref) && fsrc == 0 && std::signbit(fsrc) == std::signbit(fref))
      return true;
  }
  // For FTZ or Preserve mode, we should get the expected number within
  // ULPTolerance for any operations.
  int diff = *((const DWORD *)&fsrc) - *((const DWORD *)&fref);
  unsigned int uDiff = diff < 0 ? -diff : diff;
  return uDiff <= (unsigned int)ULPTolerance;
}

inline bool CompareFloatEpsilon(const float &fsrc, const float &fref, float epsilon,
                    hlsl::DXIL::Float32DenormMode mode = hlsl::DXIL::Float32DenormMode::Any) {
  if (fsrc == fref) {
    return true;
  }
  if (std::isnan(fsrc)) {
    return std::isnan(fref);
  }
  if (mode == hlsl::DXIL::Float32DenormMode::Any) {
    // If denorm expected, output can be sign preserved zero. Otherwise output
    // should pass the regular epsilon testing.
    if (isdenorm(fref) && fsrc == 0 && std::signbit(fsrc) == std::signbit(fref))
      return true;
  }
  // For FTZ or Preserve mode, we should get the expected number within
  // epsilon for any operations.
  return fabsf(fsrc - fref) < epsilon;
}

// Compare using relative error (relative error < 2^{nRelativeExp})
inline bool CompareFloatRelativeEpsilon(const float &fsrc, const float &fref, int nRelativeExp,
                            hlsl::DXIL::Float32DenormMode mode = hlsl::DXIL::Float32DenormMode::Any) {
  return CompareFloatULP(fsrc, fref, 23 - nRelativeExp, mode);
}

inline bool CompareHalfULP(const uint16_t &fsrc, const uint16_t &fref, float ULPTolerance) {
  if (fsrc == fref)
    return true;
  if (isnanFloat16(fsrc))
    return isnanFloat16(fref);
  // 16-bit floating point numbers must preserve denorms
  int diff = fsrc - fref;
  unsigned int uDiff = diff < 0 ? -diff : diff;
  return uDiff <= (unsigned int)ULPTolerance;
}

inline bool CompareHalfEpsilon(const uint16_t &fsrc, const uint16_t &fref, float epsilon) {
  if (fsrc == fref)
    return true;
  if (isnanFloat16(fsrc))
    return isnanFloat16(fref);
  float src_f32 = ConvertFloat16ToFloat32(fsrc);
  float ref_f32 = ConvertFloat16ToFloat32(fref);
  return std::abs(src_f32-ref_f32) < epsilon;
}

inline bool CompareHalfRelativeEpsilon(const uint16_t &fsrc, const uint16_t &fref, int nRelativeExp) {
  return CompareHalfULP(fsrc, fref, (float)(10 - nRelativeExp));
}

#ifdef _WIN32
// returns the number of bytes per pixel for a given dxgi format
// add more cases if different format needed to copy back resources
inline UINT GetByteSizeForFormat(DXGI_FORMAT value) {
    switch (value) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return 16;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    case DXGI_FORMAT_R32G32B32A32_UINT: return 16;
    case DXGI_FORMAT_R32G32B32A32_SINT: return 16;
    case DXGI_FORMAT_R32G32B32_TYPELESS: return 12;
    case DXGI_FORMAT_R32G32B32_FLOAT: return 12;
    case DXGI_FORMAT_R32G32B32_UINT: return 12;
    case DXGI_FORMAT_R32G32B32_SINT: return 12;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return 8;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R16G16B16A16_UNORM: return 8;
    case DXGI_FORMAT_R16G16B16A16_UINT: return 8;
    case DXGI_FORMAT_R16G16B16A16_SNORM: return 8;
    case DXGI_FORMAT_R16G16B16A16_SINT: return 8;
    case DXGI_FORMAT_R32G32_TYPELESS: return 8;
    case DXGI_FORMAT_R32G32_FLOAT: return 8;
    case DXGI_FORMAT_R32G32_UINT: return 8;
    case DXGI_FORMAT_R32G32_SINT: return 8;
    case DXGI_FORMAT_R32G8X24_TYPELESS: return 8;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return 4;
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return 4;
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: return 4;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return 4;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return 4;
    case DXGI_FORMAT_R10G10B10A2_UINT: return 4;
    case DXGI_FORMAT_R11G11B10_FLOAT: return 4;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return 4;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return 4;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return 4;
    case DXGI_FORMAT_R8G8B8A8_UINT: return 4;
    case DXGI_FORMAT_R8G8B8A8_SNORM: return 4;
    case DXGI_FORMAT_R8G8B8A8_SINT: return 4;
    case DXGI_FORMAT_R16G16_TYPELESS: return 4;
    case DXGI_FORMAT_R16G16_FLOAT: return 4;
    case DXGI_FORMAT_R16G16_UNORM: return 4;
    case DXGI_FORMAT_R16G16_UINT: return 4;
    case DXGI_FORMAT_R16G16_SNORM: return 4;
    case DXGI_FORMAT_R16G16_SINT: return 4;
    case DXGI_FORMAT_R32_TYPELESS: return 4;
    case DXGI_FORMAT_D32_FLOAT: return 4;
    case DXGI_FORMAT_R32_FLOAT: return 4;
    case DXGI_FORMAT_R32_UINT: return 4;
    case DXGI_FORMAT_R32_SINT: return 4;
    case DXGI_FORMAT_R24G8_TYPELESS: return 4;
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return 4;
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return 4;
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT: return 4;
    case DXGI_FORMAT_R8G8_TYPELESS: return 2;
    case DXGI_FORMAT_R8G8_UNORM: return 2;
    case DXGI_FORMAT_R8G8_UINT: return 2;
    case DXGI_FORMAT_R8G8_SNORM: return 2;
    case DXGI_FORMAT_R8G8_SINT: return 2;
    case DXGI_FORMAT_R16_TYPELESS: return 2;
    case DXGI_FORMAT_R16_FLOAT: return 2;
    case DXGI_FORMAT_D16_UNORM: return 2;
    case DXGI_FORMAT_R16_UNORM: return 2;
    case DXGI_FORMAT_R16_UINT: return 2;
    case DXGI_FORMAT_R16_SNORM: return 2;
    case DXGI_FORMAT_R16_SINT: return 2;
    case DXGI_FORMAT_R8_TYPELESS: return 1;
    case DXGI_FORMAT_R8_UNORM: return 1;
    case DXGI_FORMAT_R8_UINT: return 1;
    case DXGI_FORMAT_R8_SNORM: return 1;
    case DXGI_FORMAT_R8_SINT: return 1;
    case DXGI_FORMAT_A8_UNORM: return 1;
    case DXGI_FORMAT_R1_UNORM: return 1;
    default:
        VERIFY_FAILED(E_INVALIDARG);
        return 0;
    }
}
#endif
