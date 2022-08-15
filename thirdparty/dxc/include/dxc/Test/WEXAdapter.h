#ifndef LLVM_CLANG_UNITTESTS_WEX_ADAPTER_H
#define LLVM_CLANG_UNITTESTS_WEX_ADAPTER_H

#ifndef _WIN32

#include <unistd.h>
#include <wchar.h>

#include "dxc/Support/WinAdapter.h"
#include "dxc/Support/WinFunctions.h"
#include "gtest/gtest.h"

#define MAX_PATH 260

// Concatinate two macro fragments
#define CONCAT2(a, b) a##b
#define CONCAT1(a, b) CONCAT2(a, b)
#define CONCAT(a, b) CONCAT1(a, b)

// Determine how many arguments are passed to NARG() up to 3
#define ARG_CT(_1, _2, _3, N, ...) N
#define NARG(...) ARG_CT(__VA_ARGS__, 3, 2, 1, 0)

// Call the appropriate arity macro based on number of arguments
#define MACRO_N_(PREFIX, N, ...) CONCAT(PREFIX, N)(__VA_ARGS__)
#define MACRO_N(PREFIX, ...) MACRO_N_(PREFIX, NARG(__VA_ARGS__), __VA_ARGS__)

// Macros to convert TAEF macros to gtest equivalents
// Single and double argument versions for optional failure messages
#define VERIFY_SUCCEEDED_1(expr) EXPECT_TRUE(SUCCEEDED(expr))
#define VERIFY_SUCCEEDED_2(expr, msg) EXPECT_TRUE(SUCCEEDED(expr)) << msg
#define VERIFY_SUCCEEDED(...) MACRO_N(VERIFY_SUCCEEDED_, __VA_ARGS__)

#define VERIFY_FAILED_1(expr) EXPECT_FALSE(SUCCEEDED(expr))
#define VERIFY_FAILED_2(expr, msg) EXPECT_FALSE(SUCCEEDED(expr)) << msg
#define VERIFY_FAILED(...) MACRO_N(VERIFY_FAILED_, __VA_ARGS__)

#define VERIFY_ARE_EQUAL_2(A, B) EXPECT_EQ(A, B)
#define VERIFY_ARE_EQUAL_3(A, B, msg) EXPECT_EQ(A, B) << msg
#define VERIFY_ARE_EQUAL(...) MACRO_N(VERIFY_ARE_EQUAL_, __VA_ARGS__)

#define VERIFY_ARE_NOT_EQUAL_2(A, B) EXPECT_NE(A, B)
#define VERIFY_ARE_NOT_EQUAL_3(A, B, msg) EXPECT_NE(A, B) << msg
#define VERIFY_ARE_NOT_EQUAL(...) MACRO_N(VERIFY_ARE_NOT_EQUAL_, __VA_ARGS__)

#define VERIFY_IS_TRUE_1(expr) EXPECT_TRUE(expr)
#define VERIFY_IS_TRUE_2(expr, msg) EXPECT_TRUE(expr) << msg
#define VERIFY_IS_TRUE(...) MACRO_N(VERIFY_IS_TRUE_, __VA_ARGS__)

#define VERIFY_IS_FALSE_1(expr) EXPECT_FALSE(expr)
#define VERIFY_IS_FALSE_2(expr, msg) EXPECT_FALSE(expr) << msg
#define VERIFY_IS_FALSE(...) MACRO_N(VERIFY_IS_FALSE_, __VA_ARGS__)

#define VERIFY_IS_NULL_1(expr) EXPECT_EQ(nullptr, (expr))
#define VERIFY_IS_NULL_2(expr, msg) EXPECT_EQ(nullptr, (expr)) << msg
#define VERIFY_IS_NULL(...) MACRO_N(VERIFY_IS_NULL_, __VA_ARGS__)

#define VERIFY_IS_NOT_NULL_1(expr) EXPECT_NE(nullptr, (expr))
#define VERIFY_IS_NOT_NULL_2(expr, msg) EXPECT_NE(nullptr, (expr)) << msg
#define VERIFY_IS_NOT_NULL(...) MACRO_N(VERIFY_IS_NOT_NULL_, __VA_ARGS__)

#define VERIFY_IS_GREATER_THAN_OR_EQUAL(greater, less) EXPECT_GE(greater, less)

#define VERIFY_IS_GREATER_THAN_2(greater, less) EXPECT_GT(greater, less)
#define VERIFY_IS_GREATER_THAN_3(greater, less, msg) EXPECT_GT(greater, less) << msg
#define VERIFY_IS_GREATER_THAN(...) MACRO_N(VERIFY_IS_GREATER_THAN_, __VA_ARGS__)

#define VERIFY_IS_LESS_THAN_2(greater, less) EXPECT_LT(greater, less)
#define VERIFY_IS_LESS_THAN_3(greater, less, msg) EXPECT_LT(greater, less) << msg
#define VERIFY_IS_LESS_THAN(...) MACRO_N(VERIFY_IS_LESS_THAN_, __VA_ARGS__)

#define VERIFY_WIN32_BOOL_SUCCEEDED_1(expr) EXPECT_TRUE(expr)
#define VERIFY_WIN32_BOOL_SUCCEEDED_2(expr, msg) EXPECT_TRUE(expr) << msg
#define VERIFY_WIN32_BOOL_SUCCEEDED(...) MACRO_N(VERIFY_WIN32_BOOL_SUCCEEDED_, __VA_ARGS__)

#define VERIFY_FAIL ADD_FAILURE

#define TEST_CLASS_SETUP(method)                                               \
  bool method();                                                               \
  virtual void SetUp() { EXPECT_TRUE(method()); }
#define TEST_CLASS_CLEANUP(method)                                             \
  bool method();                                                               \
  virtual void TearDown() { EXPECT_TRUE(method()); }
#define BEGIN_TEST_CLASS(test)
#define TEST_CLASS_PROPERTY(str1, str2)
#define TEST_METHOD_PROPERTY(str1, str2)
#define END_TEST_CLASS()
#define TEST_METHOD(method)
#define BEGIN_TEST_METHOD(method)
#define END_TEST_METHOD()

// gtest lacks any module setup/cleanup. These functions are called by the
// main() function before and after tests are run. This approximates the
// behavior.
bool moduleSetup();
bool moduleTeardown();
#define MODULE_SETUP(method)                                                   \
  bool method();                                                               \
  bool moduleSetup() { return method(); }
#define MODULE_CLEANUP(method)                                                 \
  bool method();                                                               \
  bool moduleTeardown() { return method(); }

// No need to expand env vars on Unix platforms, so convert the slashes instead.
inline DWORD ExpandEnvironmentStringsW(_In_ LPCWSTR lpSrc,
                                       _Out_opt_ LPWSTR lpDst,
                                       _In_ DWORD nSize) {
  unsigned i;
  bool wasSlash = false;
  for (i = 0; i < nSize && *lpSrc; i++, lpSrc++) {
    if (*lpSrc == L'\\' || *lpSrc == L'/') {
      if (!wasSlash)
        *lpDst++ = L'/';
      wasSlash = true;
    } else {
      *lpDst++ = *lpSrc;
      wasSlash = false;
    }
  }
  *lpDst = L'\0';
  return i;
}

typedef struct _LIST_ENTRY {
  struct _LIST_ENTRY *Flink;
  struct _LIST_ENTRY *Blink;
} LIST_ENTRY, *PLIST_ENTRY, PRLIST_ENTRY;

// Minimal implementation of the WEX namespace functions and classes
// To either stub out or approximate behavior
namespace WEX {
namespace Common {
class String : public std::wstring {
public:
  size_t GetLength() { return length(); }
  bool IsEmpty() { return empty(); }
  int CompareNoCase(std::wstring str) const {
    return -1;
    assert(!"unimplemented");
  }
  operator const wchar_t *() { return c_str(); }
  const wchar_t *GetBuffer() { return *this; }
  wchar_t *Format(const wchar_t *fmt, ...) {
    static wchar_t msg[512];
    va_list args;
    va_start(args, fmt);
    vswprintf(msg, 512, fmt, args);
    va_end(args);
    return msg;
  }
};
} // namespace Common
namespace TestExecution {
enum class VerifyOutputSettings { LogOnlyFailures };
class SetVerifyOutput {
public:
  SetVerifyOutput(VerifyOutputSettings) {}
};
class DisableVerifyExceptions {
public:
  DisableVerifyExceptions() {}
};
namespace RuntimeParameters {
HRESULT TryGetValue(const wchar_t *param, Common::String &retStr);
} // namespace RuntimeParameters
} // namespace TestExecution
namespace Logging {
namespace Log {
inline void StartGroup(const wchar_t *name) { wprintf(L"BEGIN TEST(S): <%ls>\n", name); }
inline void EndGroup(const wchar_t *name) { wprintf(L"END TEST(S): <%ls>\n", name); }
inline void Comment(const wchar_t *msg) {
  fputws(msg, stdout);
  fputwc(L'\n', stdout);
}
inline void Error(const wchar_t *msg) {
  fputws(msg, stderr);
  fputwc(L'\n', stderr);
  ADD_FAILURE();
}
} // namespace Log
} // namespace Logging
} // namespace WEX

#endif // _WIN32

#endif // LLVM_CLANG_UNITTESTS_WEX_ADAPTER_H
