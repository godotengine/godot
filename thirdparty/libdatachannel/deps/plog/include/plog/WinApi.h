#pragma once

#ifdef _WIN32

// These windows structs must be in a global namespace
struct HKEY__;
struct _SECURITY_ATTRIBUTES;
struct _CONSOLE_SCREEN_BUFFER_INFO;
struct _RTL_CRITICAL_SECTION;

namespace plog
{
    typedef unsigned long DWORD;
    typedef unsigned short WORD;
    typedef unsigned char BYTE;
    typedef unsigned int UINT;
    typedef int BOOL;
    typedef long LSTATUS;
    typedef char* LPSTR;
    typedef wchar_t* LPWSTR;
    typedef const char* LPCSTR;
    typedef const wchar_t* LPCWSTR;
    typedef void* HANDLE;
    typedef HKEY__* HKEY;
    typedef size_t ULONG_PTR;

    struct CRITICAL_SECTION
    {
        void* DebugInfo;
        long LockCount;
        long RecursionCount;
        HANDLE OwningThread;
        HANDLE LockSemaphore;
        ULONG_PTR SpinCount;
    };

    struct COORD
    {
        short X;
        short Y;
    };

    struct SMALL_RECT
    {
        short Left;
        short Top;
        short Right;
        short Bottom;
    };

    struct CONSOLE_SCREEN_BUFFER_INFO
    {
        COORD dwSize;
        COORD dwCursorPosition;
        WORD  wAttributes;
        SMALL_RECT srWindow;
        COORD dwMaximumWindowSize;
    };

    namespace codePage
    {
        const UINT kActive = 0;
        const UINT kUTF8 = 65001;
#if PLOG_CHAR_IS_UTF8
        const UINT kChar = kUTF8;
#else
        const UINT kChar = kActive;
#endif
    }

    namespace eventLog
    {
        const WORD kErrorType = 0x0001;
        const WORD kWarningType = 0x0002;
        const WORD kInformationType = 0x0004;
    }

    namespace hkey
    {
        const HKEY kLocalMachine = reinterpret_cast<HKEY>(static_cast<ULONG_PTR>(0x80000002));
    }

    namespace regSam
    {
        const DWORD kQueryValue = 0x0001;
        const DWORD kSetValue = 0x0002;
    }

    namespace regType
    {
        const DWORD kExpandSz = 2;
        const DWORD kDword = 4;
    }

     namespace stdHandle
    {
        const DWORD kOutput = static_cast<DWORD>(-11);
        const DWORD kErrorOutput = static_cast<DWORD>(-12);
    }

    namespace foreground
    {
        const WORD kBlue = 0x0001;
        const WORD kGreen = 0x0002;
        const WORD kRed = 0x0004;
        const WORD kIntensity = 0x0008;
    }

    namespace background
    {
        const WORD kBlue = 0x0010;
        const WORD kGreen = 0x0020;
        const WORD kRed = 0x0040;
        const WORD kIntensity = 0x0080;
    }

    extern "C"
    {
        __declspec(dllimport) int __stdcall MultiByteToWideChar(UINT CodePage, DWORD dwFlags, LPCSTR lpMultiByteStr, int cbMultiByte, LPWSTR lpWideCharStr, int cchWideChar);
        __declspec(dllimport) int __stdcall WideCharToMultiByte(UINT CodePage, DWORD dwFlags, LPCWSTR lpWideCharStr, int cchWideChar, LPSTR lpMultiByteStr, int cbMultiByte, const char* lpDefaultChar, BOOL* lpUsedDefaultChar);

        __declspec(dllimport) DWORD __stdcall GetCurrentThreadId();

        __declspec(dllimport) BOOL __stdcall MoveFileW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName);

        __declspec(dllimport) void __stdcall InitializeCriticalSection(_RTL_CRITICAL_SECTION* lpCriticalSection);
        __declspec(dllimport) void __stdcall EnterCriticalSection(_RTL_CRITICAL_SECTION* lpCriticalSection);
        __declspec(dllimport) void __stdcall LeaveCriticalSection(_RTL_CRITICAL_SECTION* lpCriticalSection);
        __declspec(dllimport) void __stdcall DeleteCriticalSection(_RTL_CRITICAL_SECTION* lpCriticalSection);

        __declspec(dllimport) HANDLE __stdcall RegisterEventSourceW(LPCWSTR lpUNCServerName, LPCWSTR lpSourceName);
        __declspec(dllimport) BOOL __stdcall DeregisterEventSource(HANDLE hEventLog);
        __declspec(dllimport) BOOL __stdcall ReportEventW(HANDLE hEventLog, WORD wType, WORD wCategory, DWORD dwEventID, void* lpUserSid, WORD wNumStrings, DWORD dwDataSize, LPCWSTR* lpStrings, void* lpRawData);

        __declspec(dllimport) LSTATUS __stdcall RegCreateKeyExW(HKEY hKey, LPCWSTR lpSubKey, DWORD Reserved, LPWSTR lpClass, DWORD dwOptions, DWORD samDesired, _SECURITY_ATTRIBUTES* lpSecurityAttributes, HKEY* phkResult, DWORD* lpdwDisposition);
        __declspec(dllimport) LSTATUS __stdcall RegSetValueExW(HKEY hKey, LPCWSTR lpValueName, DWORD Reserved, DWORD dwType, const BYTE* lpData, DWORD cbData);
        __declspec(dllimport) LSTATUS __stdcall RegCloseKey(HKEY hKey);
        __declspec(dllimport) LSTATUS __stdcall RegOpenKeyExW(HKEY hKey, LPCWSTR lpSubKey, DWORD ulOptions, DWORD samDesired, HKEY* phkResult);
        __declspec(dllimport) LSTATUS __stdcall RegDeleteKeyW(HKEY hKey, LPCWSTR lpSubKey);

        __declspec(dllimport) HANDLE __stdcall GetStdHandle(DWORD nStdHandle);

        __declspec(dllimport) BOOL __stdcall WriteConsoleW(HANDLE hConsoleOutput, const void* lpBuffer, DWORD nNumberOfCharsToWrite, DWORD* lpNumberOfCharsWritten, void* lpReserved);
        __declspec(dllimport) BOOL __stdcall GetConsoleScreenBufferInfo(HANDLE hConsoleOutput, _CONSOLE_SCREEN_BUFFER_INFO* lpConsoleScreenBufferInfo);
        __declspec(dllimport) BOOL __stdcall SetConsoleTextAttribute(HANDLE hConsoleOutput, WORD wAttributes);

        __declspec(dllimport) void __stdcall OutputDebugStringW(LPCWSTR lpOutputString);
    }

    inline void InitializeCriticalSection(CRITICAL_SECTION* criticalSection)
    {
        plog::InitializeCriticalSection(reinterpret_cast<_RTL_CRITICAL_SECTION*>(criticalSection));
    }

    inline void EnterCriticalSection(CRITICAL_SECTION* criticalSection)
    {
        plog::EnterCriticalSection(reinterpret_cast<_RTL_CRITICAL_SECTION*>(criticalSection));
    }

    inline void LeaveCriticalSection(CRITICAL_SECTION* criticalSection)
    {
        plog::LeaveCriticalSection(reinterpret_cast<_RTL_CRITICAL_SECTION*>(criticalSection));
    }

    inline void DeleteCriticalSection(CRITICAL_SECTION* criticalSection)
    {
        plog::DeleteCriticalSection(reinterpret_cast<_RTL_CRITICAL_SECTION*>(criticalSection));
    }

    inline BOOL GetConsoleScreenBufferInfo(HANDLE consoleOutput, CONSOLE_SCREEN_BUFFER_INFO* consoleScreenBufferInfo)
    {
        return plog::GetConsoleScreenBufferInfo(consoleOutput, reinterpret_cast<_CONSOLE_SCREEN_BUFFER_INFO*>(consoleScreenBufferInfo));
    }
}
#endif // _WIN32
