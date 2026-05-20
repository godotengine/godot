// Main.cpp

#include "StdAfx.h"

#include "../../../Common/MyWindows.h"

#ifdef _WIN32

#ifndef Z7_OLD_WIN_SDK

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <psapi.h>
#else
#include <Psapi.h>
#endif

#else // Z7_OLD_WIN_SDK

typedef struct {
    DWORD cb;
    DWORD PageFaultCount;
    SIZE_T PeakWorkingSetSize;
    SIZE_T WorkingSetSize;
    SIZE_T QuotaPeakPagedPoolUsage;
    SIZE_T QuotaPagedPoolUsage;
    SIZE_T QuotaPeakNonPagedPoolUsage;
    SIZE_T QuotaNonPagedPoolUsage;
    SIZE_T PagefileUsage;
    SIZE_T PeakPagefileUsage;
} PROCESS_MEMORY_COUNTERS;
typedef PROCESS_MEMORY_COUNTERS *PPROCESS_MEMORY_COUNTERS;

#endif // Z7_OLD_WIN_SDK

#else // _WIN32
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/times.h>
#endif // _WIN32

#include "../../../../C/CpuArch.h"

#include "../../../Common/MyInitGuid.h"

#include "../../../Common/CommandLineParser.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/MyException.h"
#include "../../../Common/StdInStream.h"
#include "../../../Common/StdOutStream.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/StringToInt.h"
#include "../../../Common/UTFConvert.h"

#include "../../../Windows/ErrorMsg.h"
#include "../../../Windows/TimeUtils.h"
#include "../../../Windows/FileDir.h"

#include "../Common/ArchiveCommandLine.h"
#include "../Common/Bench.h"
#include "../Common/ExitCode.h"
#include "../Common/Extract.h"

#ifdef Z7_EXTERNAL_CODECS
#include "../Common/LoadCodecs.h"
#endif

#include "../../Common/RegisterCodec.h"

#include "BenchCon.h"
#include "ConsoleClose.h"
#include "ExtractCallbackConsole.h"
#include "HashCon.h"
#include "List.h"
#include "OpenCallbackConsole.h"
#include "UpdateCallbackConsole.h"

#ifdef Z7_PROG_VARIANT_R
#include "../../../../C/7zVersion.h"
#else
#include "../../MyVersion.h"
#endif

using namespace NWindows;
using namespace NFile;
using namespace NCommandLineParser;

#ifdef _WIN32
extern
HINSTANCE g_hInstance;
HINSTANCE g_hInstance = NULL;
#endif

extern CStdOutStream *g_StdStream;
extern CStdOutStream *g_ErrStream;

extern unsigned g_NumCodecs;
extern const CCodecInfo *g_Codecs[];

extern unsigned g_NumHashers;
extern const CHasherInfo *g_Hashers[];

#ifdef Z7_EXTERNAL_CODECS
extern
const CExternalCodecs *g_ExternalCodecs_Ptr;
const CExternalCodecs *g_ExternalCodecs_Ptr;
#endif

DECLARE_AND_SET_CLIENT_VERSION_VAR

#if defined(Z7_PROG_VARIANT_Z)
  #define PROG_POSTFIX      "z"
  #define PROG_POSTFIX_2  " (z)"
#elif defined(Z7_PROG_VARIANT_R)
  #define PROG_POSTFIX      "r"
  #define PROG_POSTFIX_2  " (r)"
#elif defined(Z7_PROG_VARIANT_A) || !defined(Z7_EXTERNAL_CODECS)
  #define PROG_POSTFIX      "a"
  #define PROG_POSTFIX_2  " (a)"
#else
  #define PROG_POSTFIX    ""
  #define PROG_POSTFIX_2  ""
#endif


static const char * const kCopyrightString = "\n7-Zip"
  PROG_POSTFIX_2
  " " MY_VERSION_CPU
  " : " MY_COPYRIGHT_DATE "\n";

static const char * const kHelpString =
    "Usage: 7z"
    PROG_POSTFIX
    " <command> [<switches>...] <archive_name> [<file_names>...] [@listfile]\n"
    "\n"
    "<Commands>\n"
    "  a : Add files to archive\n"
    "  b : Benchmark\n"
    "  d : Delete files from archive\n"
    "  e : Extract files from archive (without using directory names)\n"
    "  h : Calculate hash values for files\n"
    "  i : Show information about supported formats\n"
    "  l : List contents of archive\n"
    "  rn : Rename files in archive\n"
    "  t : Test integrity of archive\n"
    "  u : Update files to archive\n"
    "  x : eXtract files with full paths\n"
    "\n"
    "<Switches>\n"
    "  -- : Stop switches and @listfile parsing\n"
    "  -ai[r[-|0]][m[-|2]][w[-]]{@listfile|!wildcard} : Include archives\n"
    "  -ax[r[-|0]][m[-|2]][w[-]]{@listfile|!wildcard} : eXclude archives\n"
    "  -ao{a|s|t|u} : set Overwrite mode\n"
    "  -an : disable archive_name field\n"
    "  -bb[0-3] : set output log level\n"
    "  -bd : disable progress indicator\n"
    "  -bs{o|e|p}{0|1|2} : set output stream for output/error/progress line\n"
    "  -bt : show execution time statistics\n"
    "  -i[r[-|0]][m[-|2]][w[-]]{@listfile|!wildcard} : Include filenames\n"
    "  -m{Parameters} : set compression Method\n"
    "    -mmt[N] : set number of CPU threads\n"
    "    -mx[N] : set compression level: -mx1 (fastest) ... -mx9 (ultra)\n"
    "  -o{Directory} : set Output directory\n"
    #ifndef Z7_NO_CRYPTO
    "  -p{Password} : set Password\n"
    #endif
    "  -r[-|0] : Recurse subdirectories for name search\n"
    "  -sa{a|e|s} : set Archive name mode\n"
    "  -scc{UTF-8|WIN|DOS} : set charset for console input/output\n"
    "  -scs{UTF-8|UTF-16LE|UTF-16BE|WIN|DOS|{id}} : set charset for list files\n"
    "  -scrc[CRC32|CRC64|SHA256"
#ifndef Z7_PROG_VARIANT_R
    "|SHA1|XXH64"
#ifdef Z7_PROG_VARIANT_Z
    "|BLAKE2SP"
#endif
#endif
    "|*] : set hash function for x, e, h commands\n"
    "  -sdel : delete files after compression\n"
    "  -seml[.] : send archive by email\n"
    "  -sfx[{name}] : Create SFX archive\n"
    "  -si[{name}] : read data from stdin\n"
    "  -slp : set Large Pages mode\n"
    "  -slt : show technical information for l (List) command\n"
    "  -snh : store hard links as links\n"
    "  -snl : store symbolic links as links\n"
    "  -sni : store NT security information\n"
    "  -sns[-] : store NTFS alternate streams\n"
    "  -so : write data to stdout\n"
    "  -spd : disable wildcard matching for file names\n"
    "  -spe : eliminate duplication of root folder for extract command\n"
    "  -spf[2] : use fully qualified file paths\n"
    "  -ssc[-] : set sensitive case mode\n"
    "  -sse : stop archive creating, if it can't open some input file\n"
    "  -ssp : do not change Last Access Time of source files while archiving\n"
    "  -ssw : compress shared files\n"
    "  -stl : set archive timestamp from the most recently modified file\n"
    "  -stm{HexMask} : set CPU thread affinity mask (hexadecimal number)\n"
    "  -stx{Type} : exclude archive type\n"
    "  -t{Type} : Set type of archive\n"
    "  -u[-][p#][q#][r#][x#][y#][z#][!newArchiveName] : Update options\n"
    "  -v{Size}[b|k|m|g] : Create volumes\n"
    "  -w[{path}] : assign Work directory. Empty path means a temporary directory\n"
    "  -x[r[-|0]][m[-|2]][w[-]]{@listfile|!wildcard} : eXclude filenames\n"
    "  -y : assume Yes on all queries\n";

// ---------------------------
// exception messages

static const char * const kEverythingIsOk = "Everything is Ok";
static const char * const kUserErrorMessage = "Incorrect command line";
static const char * const kNoFormats = "7-Zip cannot find the code that works with archives.";
static const char * const kUnsupportedArcTypeMessage = "Unsupported archive type";
// static const char * const kUnsupportedUpdateArcType = "Can't create archive for that type";

#ifndef Z7_EXTRACT_ONLY
#define kDefaultSfxModule "7zCon.sfx"
#endif

Z7_ATTR_NORETURN
static void ShowMessageAndThrowException(LPCSTR message, NExitCode::EEnum code)
{
  if (g_ErrStream)
    *g_ErrStream << endl << "ERROR: " << message << endl;
  throw code;
}


#ifdef _WIN32
#define ShowProgInfo(so)
#else
static void ShowProgInfo(CStdOutStream *so)
{
  if (!so)
    return;

  *so
  
  /*
  #ifdef __DATE__
      << " " << __DATE__
  #endif
  #ifdef __TIME__
      << " " << __TIME__
  #endif
  */

  << " " << (unsigned)(sizeof(void *)) * 8 << "-bit"

  #ifdef __ILP32__
    << " ILP32"
  #endif

  #ifdef __ARM_ARCH
  << " arm_v:" << __ARM_ARCH
  #if (__ARM_ARCH == 8)
    // for macos:
    #if   defined(__ARM_ARCH_8_9__)
      << ".9"
    #elif defined(__ARM_ARCH_8_8__)
      << ".8"
    #elif defined(__ARM_ARCH_8_7__)
      << ".7"
    #elif defined(__ARM_ARCH_8_6__)
      << ".6"
    #elif defined(__ARM_ARCH_8_5__)
      << ".5"
    #elif defined(__ARM_ARCH_8_4__)
      << ".4"
    #elif defined(__ARM_ARCH_8_3__)
      << ".3"
    #elif defined(__ARM_ARCH_8_2__)
      << ".2"
    #elif defined(__ARM_ARCH_8_1__)
      << ".1"
    #endif
  #endif
    
    #if defined(__ARM_ARCH_PROFILE) && \
        (   __ARM_ARCH_PROFILE >= 'A' && __ARM_ARCH_PROFILE <= 'Z' \
         || __ARM_ARCH_PROFILE >= 65  && __ARM_ARCH_PROFILE <= 65 + 25)
      << "-" << (char)__ARM_ARCH_PROFILE
    #endif

  #ifdef __ARM_ARCH_ISA_THUMB
  << " thumb:" << __ARM_ARCH_ISA_THUMB
  #endif
  #endif

  #ifdef _MIPS_ARCH
  << " mips_arch:" << _MIPS_ARCH
  #endif
  #ifdef __mips_isa_rev
  << " mips_isa_rev:" << __mips_isa_rev
  #endif

  #ifdef __iset__
  << " e2k_v:" << __iset__
  #endif
  ;



  #ifdef ENV_HAVE_LOCALE
    *so << " locale=" << GetLocale();
  #endif
  #ifndef _WIN32
  {
    const bool is_IsNativeUTF8 = IsNativeUTF8();
    if (!is_IsNativeUTF8)
      *so << " UTF8=" << (is_IsNativeUTF8 ? "+" : "-");
  }
  if (!g_ForceToUTF8)
    *so << " use-UTF8=" << (g_ForceToUTF8 ? "+" : "-");
  {
    const unsigned wchar_t_size = (unsigned)sizeof(wchar_t);
    if (wchar_t_size != 4)
      *so << " wchar_t=" << wchar_t_size * 8 << "-bit";
  }
  {
    const unsigned off_t_size = (unsigned)sizeof(off_t);
    if (off_t_size != 8)
      *so << " Files=" << off_t_size * 8 << "-bit";
  }
  #endif
  
  {
    const UInt32 numCpus = NWindows::NSystem::GetNumberOfProcessors();
    *so << " Threads:" << numCpus;
    const UInt64 openMAX= NWindows::NSystem::Get_File_OPEN_MAX();
    *so << " OPEN_MAX:" << openMAX;
    {
      FString temp;
      NDir::MyGetTempPath(temp);
      if (!temp.IsEqualTo(STRING_PATH_SEPARATOR "tmp" STRING_PATH_SEPARATOR))
        *so << " temp_path:" << temp;
    }
  }

  #ifdef Z7_7ZIP_ASM
  *so << ", ASM";
  #endif

  /*
  {
    AString s;
    GetCpuName(s);
    s.Trim();
    *so << ", " << s;
  }

  #ifdef __ARM_FEATURE_CRC32
     << " CRC32"
  #endif

  
  #if (defined MY_CPU_X86_OR_AMD64 || defined(MY_CPU_ARM_OR_ARM64))
  if (CPU_IsSupported_AES()) *so << ",AES";
  #endif
  
  #ifdef MY_CPU_ARM_OR_ARM64
  if (CPU_IsSupported_CRC32()) *so << ",CRC32";
  #if defined(_WIN32)
  if (CPU_IsSupported_CRYPTO()) *so << ",CRYPTO";
  #else
  if (CPU_IsSupported_SHA1()) *so << ",SHA1";
  if (CPU_IsSupported_SHA2()) *so << ",SHA2";
  #endif
  #endif
  */

  *so << endl;
}
#endif

static void ShowCopyrightAndHelp(CStdOutStream *so, bool needHelp)
{
  if (!so)
    return;
  *so << kCopyrightString;
  // *so << "# CPUs: " << (UInt64)NWindows::NSystem::GetNumberOfProcessors() << endl;
  ShowProgInfo(so);
  *so << endl;
  if (needHelp)
    *so << kHelpString;
}


static void PrintStringRight(CStdOutStream &so, const char *s, unsigned size)
{
  unsigned len = MyStringLen(s);
  for (unsigned i = len; i < size; i++)
    so << ' ';
  so << s;
}

static void PrintUInt32(CStdOutStream &so, UInt32 val, unsigned size)
{
  char s[16];
  ConvertUInt32ToString(val, s);
  PrintStringRight(so, s, size);
}

#ifdef Z7_EXTERNAL_CODECS
static void PrintNumber(CStdOutStream &so, UInt32 val, unsigned numDigits)
{
  AString s;
  s.Add_UInt32(val);
  while (s.Len() < numDigits)
    s.InsertAtFront('0');
  so << s;
}
#endif

static void PrintLibIndex(CStdOutStream &so, int libIndex)
{
  if (libIndex >= 0)
    PrintUInt32(so, (UInt32)libIndex, 2);
  else
    so << "  ";
  so << ' ';
}

static void PrintString(CStdOutStream &so, const UString &s, unsigned size)
{
  unsigned len = s.Len();
  so << s;
  for (unsigned i = len; i < size; i++)
    so << ' ';
}

static void PrintWarningsPaths(const CErrorPathCodes &pc, CStdOutStream &so)
{
  FOR_VECTOR(i, pc.Paths)
  {
    so.NormalizePrint_UString_Path(fs2us(pc.Paths[i]));
    so << " : ";
    so << NError::MyFormatMessage(pc.Codes[i]) << endl;
  }
  so << "----------------" << endl;
}

static int WarningsCheck(HRESULT result, const CCallbackConsoleBase &callback,
    const CUpdateErrorInfo &errorInfo,
    CStdOutStream *so,
    CStdOutStream *se,
    bool showHeaders)
{
  int exitCode = NExitCode::kSuccess;
  
  if (callback.ScanErrors.Paths.Size() != 0)
  {
    if (se)
    {
      *se << endl;
      *se << "Scan WARNINGS for files and folders:" << endl << endl;
      PrintWarningsPaths(callback.ScanErrors, *se);
      *se << "Scan WARNINGS: " << callback.ScanErrors.Paths.Size();
      *se << endl;
    }
    exitCode = NExitCode::kWarning;
  }
  
  if (result != S_OK || errorInfo.ThereIsError())
  {
    if (se)
    {
      UString message;
      if (!errorInfo.Message.IsEmpty())
      {
        message += errorInfo.Message.Ptr();
        message.Add_LF();
      }
      {
        FOR_VECTOR(i, errorInfo.FileNames)
        {
          message += fs2us(errorInfo.FileNames[i]);
          message.Add_LF();
        }
      }
      if (errorInfo.SystemError != 0)
      {
        message += NError::MyFormatMessage(errorInfo.SystemError);
        message.Add_LF();
      }
      if (!message.IsEmpty())
        *se << L"\nError:\n" << message;
    }

    // we will work with (result) later
    // throw CSystemException(result);
    return NExitCode::kFatalError;
  }

  unsigned numErrors = callback.FailedFiles.Paths.Size();
  if (numErrors == 0)
  {
    if (showHeaders)
      if (callback.ScanErrors.Paths.Size() == 0)
        if (so)
        {
          if (se)
            se->Flush();
          *so << kEverythingIsOk << endl;
        }
  }
  else
  {
    if (se)
    {
      *se << endl;
      *se << "WARNINGS for files:" << endl << endl;
      PrintWarningsPaths(callback.FailedFiles, *se);
      *se << "WARNING: Cannot open " << numErrors << " file";
      if (numErrors > 1)
        *se << 's';
      *se << endl;
    }
    exitCode = NExitCode::kWarning;
  }
  
  return exitCode;
}

static void ThrowException_if_Error(HRESULT res)
{
  if (res != S_OK)
    throw CSystemException(res);
}

static void PrintNum(UInt64 val, unsigned numDigits, char c = ' ')
{
  char temp[64];
  char *p = temp + 32;
  ConvertUInt64ToString(val, p);
  unsigned len = MyStringLen(p);
  for (; len < numDigits; len++)
    *--p = c;
  *g_StdStream << p;
}

#ifdef _WIN32

static void PrintTime(const char *s, UInt64 val, UInt64 total)
{
  *g_StdStream << endl << s << " Time =";
  const UInt32 kFreq = 10000000;
  UInt64 sec = val / kFreq;
  PrintNum(sec, 6);
  *g_StdStream << '.';
  UInt32 ms = (UInt32)(val - (sec * kFreq)) / (kFreq / 1000);
  PrintNum(ms, 3, '0');
  
  while (val > ((UInt64)1 << 56))
  {
    val >>= 1;
    total >>= 1;
  }

  UInt64 percent = 0;
  if (total != 0)
    percent = val * 100 / total;
  *g_StdStream << " =";
  PrintNum(percent, 5);
  *g_StdStream << '%';
}

#ifndef UNDER_CE

#define SHIFT_SIZE_VALUE(x, num) (((x) + (1 << (num)) - 1) >> (num))

static void PrintMemUsage(const char *s, UInt64 val)
{
  *g_StdStream << "    " << s << " Memory =";
  PrintNum(SHIFT_SIZE_VALUE(val, 20), 7);
  *g_StdStream << " MB";
  /*
  *g_StdStream << " =";
  PrintNum(SHIFT_SIZE_VALUE(val, 10), 9);
  *g_StdStream << " KB";
  */
  #ifdef Z7_LARGE_PAGES
  AString lp;
  Add_LargePages_String(lp);
  if (!lp.IsEmpty())
    *g_StdStream << lp;
  #endif
}

EXTERN_C_BEGIN
typedef BOOL (WINAPI *Func_GetProcessMemoryInfo)(HANDLE Process,
    PPROCESS_MEMORY_COUNTERS ppsmemCounters, DWORD cb);
typedef BOOL (WINAPI *Func_QueryProcessCycleTime)(HANDLE Process, PULONG64 CycleTime);
EXTERN_C_END

#endif

static inline UInt64 GetTime64(const FILETIME &t) { return ((UInt64)t.dwHighDateTime << 32) | t.dwLowDateTime; }

static void PrintStat()
{
  FILETIME creationTimeFT, exitTimeFT, kernelTimeFT, userTimeFT;
  if (!
      #ifdef UNDER_CE
        ::GetThreadTimes(::GetCurrentThread()
      #else
        // NT 3.5
        ::GetProcessTimes(::GetCurrentProcess()
      #endif
      , &creationTimeFT, &exitTimeFT, &kernelTimeFT, &userTimeFT))
    return;
  FILETIME curTimeFT;
  NTime::GetCurUtc_FiTime(curTimeFT);

  #ifndef UNDER_CE
  
Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

  PROCESS_MEMORY_COUNTERS m;
  memset(&m, 0, sizeof(m));
  BOOL memDefined = FALSE;
  BOOL cycleDefined = FALSE;
  ULONG64 cycleTime = 0;
  {
    /* NT 4.0: GetProcessMemoryInfo() in Psapi.dll
       Win7: new function K32GetProcessMemoryInfo() in kernel32.dll
       It's faster to call kernel32.dll code than Psapi.dll code
       GetProcessMemoryInfo() requires Psapi.lib
       Psapi.lib in SDK7+ can link to K32GetProcessMemoryInfo in kernel32.dll
       The program with K32GetProcessMemoryInfo will not work on systems before Win7
       // memDefined = GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof(m));
    */
    const HMODULE kern = ::GetModuleHandleW(L"kernel32.dll");
    Func_GetProcessMemoryInfo
      my_GetProcessMemoryInfo = Z7_GET_PROC_ADDRESS(
    Func_GetProcessMemoryInfo, kern,
     "K32GetProcessMemoryInfo");
    if (!my_GetProcessMemoryInfo)
    {
      const HMODULE lib = LoadLibraryW(L"Psapi.dll");
      if (lib)
          my_GetProcessMemoryInfo = Z7_GET_PROC_ADDRESS(
        Func_GetProcessMemoryInfo, lib,
            "GetProcessMemoryInfo");
    }
    if (my_GetProcessMemoryInfo)
      memDefined = my_GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof(m));
    // FreeLibrary(lib);
    const
    Func_QueryProcessCycleTime
      my_QueryProcessCycleTime = Z7_GET_PROC_ADDRESS(
    Func_QueryProcessCycleTime, kern,
        "QueryProcessCycleTime");
    if (my_QueryProcessCycleTime)
      cycleDefined = my_QueryProcessCycleTime(GetCurrentProcess(), &cycleTime);
  }

  #endif

  UInt64 curTime = GetTime64(curTimeFT);
  UInt64 creationTime = GetTime64(creationTimeFT);
  UInt64 kernelTime = GetTime64(kernelTimeFT);
  UInt64 userTime = GetTime64(userTimeFT);

  UInt64 totalTime = curTime - creationTime;
  
  PrintTime("Kernel ", kernelTime, totalTime);

  const UInt64 processTime = kernelTime + userTime;
  
  #ifndef UNDER_CE
  if (cycleDefined)
  {
    *g_StdStream << "    Cnt:";
    PrintNum(cycleTime / 1000000, 15);
    *g_StdStream << " MCycles";
  }
  #endif

  PrintTime("User   ", userTime, totalTime);

  #ifndef UNDER_CE
  if (cycleDefined)
  {
    *g_StdStream << "    Freq (cnt/ptime):";
    UInt64 us = processTime / 10;
    if (us == 0)
      us = 1;
    PrintNum(cycleTime / us, 6);
    *g_StdStream << " MHz";
  }
  #endif
  
  PrintTime("Process", processTime, totalTime);
  #ifndef UNDER_CE
  if (memDefined) PrintMemUsage("Virtual ", m.PeakPagefileUsage);
  #endif
  
  PrintTime("Global ", totalTime, totalTime);
  #ifndef UNDER_CE
  if (memDefined) PrintMemUsage("Physical", m.PeakWorkingSetSize);
  #endif
  *g_StdStream << endl;
}


#else  // ! _WIN32

static UInt64 Get_timeofday_us()
{
  struct timeval now;
  if (gettimeofday(&now, NULL) == 0)
    return (UInt64)now.tv_sec * 1000000 + (UInt64)now.tv_usec;
  return 0;
}

static void PrintTime(const char *s, UInt64 val, UInt64 total_us, UInt64 kFreq)
{
  *g_StdStream << endl << s << " Time =";

  {
    UInt64 sec, ms;
    
    if (kFreq == 0)
    {
      sec = val / 1000000;
      ms  = val % 1000000 / 1000;
    }
    else
    {
      sec = val / kFreq;
      ms = (UInt32)((val - (sec * kFreq)) * 1000 / kFreq);
    }
    
    PrintNum(sec, 6);
    *g_StdStream << '.';
    PrintNum(ms, 3, '0');
  }
  
  if (total_us == 0)
    return;

  UInt64 percent = 0;
  if (kFreq == 0)
    percent = val * 100 / total_us;
  else
  {
    const UInt64 kMaxVal = (UInt64)(Int64)-1;
    UInt32 m = 100000000;
    for (;;)
    {
      if (m == 0 || kFreq == 0)
        break;
      if (kMaxVal / m > val &&
        kMaxVal / kFreq > total_us)
        break;
      if (val > m)
        val >>= 1;
      else
        m >>= 1;
      if (kFreq > total_us)
        kFreq >>= 1;
      else
        total_us >>= 1;
    }
    const UInt64 total = kFreq * total_us;
    if (total != 0)
      percent = val * m / total;
  }
  *g_StdStream << " =";
  PrintNum(percent, 5);
  *g_StdStream << '%';
}

static void PrintStat(const UInt64 startTime)
{
  tms t;
  /* clock_t res = */ times(&t);
  const UInt64 totalTime = Get_timeofday_us() - startTime;
  const UInt64 kFreq = (UInt64)sysconf(_SC_CLK_TCK);
  PrintTime("Kernel ", (UInt64)t.tms_stime, totalTime, kFreq);
  PrintTime("User   ", (UInt64)t.tms_utime, totalTime, kFreq);
  PrintTime("Process", (UInt64)t.tms_utime + (UInt64)t.tms_stime, totalTime, kFreq);
  PrintTime("Global ", totalTime, totalTime, 0);
  *g_StdStream << endl;
}

#endif // ! _WIN32





static void PrintHexId(CStdOutStream &so, UInt64 id)
{
  char s[32];
  ConvertUInt64ToHex(id, s);
  PrintStringRight(so, s, 8);
}

#ifndef _WIN32
#include <sys/resource.h>
void Set_ModuleDirPrefix_From_ProgArg0(const char *s);
#endif

int Main2(
  #ifndef _WIN32
  int numArgs, char *args[]
  #endif
);
int Main2(
  #ifndef _WIN32
  int numArgs, char *args[]
  #endif
)
{
  #if defined(MY_CPU_SIZEOF_POINTER)
    { unsigned k = sizeof(void *); if (k != MY_CPU_SIZEOF_POINTER) throw "incorrect MY_CPU_PTR_SIZE"; }
  #endif

  #if defined(_WIN32) && !defined(UNDER_CE)
  SetFileApisToOEM();
  #endif

#if 1 && !defined(_WIN32)
  {
    // We increase RLIMIT_NOFILE because a high limit makes it easier for 7-Zip to work with multi-volume archives.
    struct rlimit a;
    const int kType = RLIMIT_NOFILE;
    if (!getrlimit(kType, &a))
    {
      // printf("\nrlim_cur=%8lld rlim_max=%8lld \n", (long long)a.rlim_cur, (long long)a.rlim_max);
      unsigned newVal = 1 << 12; // rlim_t : it's new soft limit
      if (newVal > a.rlim_cur && a.rlim_cur < a.rlim_max)
      {
        if (newVal > a.rlim_max)
          newVal = (unsigned)a.rlim_max;
        a.rlim_cur = newVal;
        setrlimit(kType, &a);
        // if (!getrlimit(kType, &a)) printf("\nrlim_cur=%8lld rlim_max=%8lld \n", (long long)a.rlim_cur, (long long)a.rlim_max);
      }
    }
  }
#endif

  #ifdef ENV_HAVE_LOCALE
  // printf("\nBefore SetLocale() : %s\n", IsNativeUtf8() ? "NATIVE UTF-8" : "IS NOT NATIVE UTF-8");
  MY_SetLocale();
  // printf("\nAfter  SetLocale() : %s\n", IsNativeUtf8() ? "NATIVE UTF-8" : "IS NOT NATIVE UTF-8");
  #endif

  #ifndef _WIN32
  const UInt64 startTime = Get_timeofday_us();
  #endif

  /*
  {
    g_StdOut << "DWORD:" << (unsigned)sizeof(DWORD);
    g_StdOut << " LONG:" << (unsigned)sizeof(LONG);
    g_StdOut << " long:" << (unsigned)sizeof(long);
    #ifdef _WIN64
    // g_StdOut << " long long:" << (unsigned)sizeof(long long);
    #endif
    g_StdOut << " int:" << (unsigned)sizeof(int);
    g_StdOut << " void*:"  << (unsigned)sizeof(void *);
    g_StdOut << endl;
  }
  */

  UStringVector commandStrings;
  
  #ifdef _WIN32
  NCommandLineParser::SplitCommandLine(GetCommandLineW(), commandStrings);
  #else
  {
    if (numArgs > 0)
      Set_ModuleDirPrefix_From_ProgArg0(args[0]);

    for (int i = 0; i < numArgs; i++)
    {
      AString a (args[i]);
#if 0
      printf("\n%d %s :", i, a.Ptr());
      for (unsigned k = 0; k < a.Len(); k++)
        printf(" %2x", (unsigned)(Byte)a[k]);
#endif
      const UString s = MultiByteToUnicodeString(a);
      commandStrings.Add(s);
    }
    // printf("\n");
  }

  #endif

  #ifndef UNDER_CE
  if (commandStrings.Size() > 0)
    commandStrings.Delete(0);
  #endif

  if (commandStrings.Size() == 0)
  {
    ShowCopyrightAndHelp(g_StdStream, true);
    return 0;
  }

  CArcCmdLineOptions options;

  CArcCmdLineParser parser;

  parser.Parse1(commandStrings, options);

  g_StdOut.IsTerminalMode = options.IsStdOutTerminal;
  g_StdErr.IsTerminalMode = options.IsStdErrTerminal;

  if (options.Number_for_Out != k_OutStream_stdout)
    g_StdStream = (options.Number_for_Out == k_OutStream_stderr ? &g_StdErr : NULL);

  if (options.Number_for_Errors != k_OutStream_stderr)
    g_ErrStream = (options.Number_for_Errors == k_OutStream_stdout ? &g_StdOut : NULL);

  CStdOutStream *percentsStream = NULL;
  if (options.Number_for_Percents != k_OutStream_disabled)
    percentsStream = (options.Number_for_Percents == k_OutStream_stderr) ? &g_StdErr : &g_StdOut;
  
  if (options.HelpMode)
  {
    ShowCopyrightAndHelp(g_StdStream, true);
    return 0;
  }

  if (options.EnableHeaders)
  {
    if (g_StdStream)
    {
      ShowCopyrightAndHelp(g_StdStream, false);
      if (!parser.Parse1Log.IsEmpty())
        *g_StdStream << parser.Parse1Log;
    }
  }

  parser.Parse2(options);

  {
    int cp = options.ConsoleCodePage;
    
    int stdout_cp = cp;
    int stderr_cp = cp;
    int stdin_cp = cp;

    /*
    // these cases are complicated.
    // maybe we must use CRT functions instead of console WIN32.
    // different Windows/CRT versions also can work different ways.
    // so the following code was not enabled:
    if (cp == -1)
    {
      // we set CodePage only if stream is attached to terminal
      // maybe we should set CodePage even if is not terminal?
      #ifdef _WIN32
      {
        UINT ccp = GetConsoleOutputCP();
        if (ccp != 0)
        {
          if (options.IsStdOutTerminal) stdout_cp = ccp;
          if (options.IsStdErrTerminal) stderr_cp = ccp;
        }
      }
      if (options.IsInTerminal)
      {
        UINT ccp = GetConsoleCP();
        if (ccp != 0) stdin_cp = ccp;
      }
      #endif
    }
    */
    
    if (stdout_cp != -1) g_StdOut.CodePage = stdout_cp;
    if (stderr_cp != -1) g_StdErr.CodePage = stderr_cp;
    if (stdin_cp != -1) g_StdIn.CodePage = stdin_cp;
  }
  g_StdOut.ListPathSeparatorSlash = options.ListPathSeparatorSlash;
  g_StdErr.ListPathSeparatorSlash = options.ListPathSeparatorSlash;

  unsigned percentsNameLevel = 1;
  if (options.LogLevel == 0 || options.Number_for_Percents != options.Number_for_Out)
    percentsNameLevel = 2;

  unsigned consoleWidth = 80;

  if (percentsStream)
  {
    #ifdef _WIN32
    
    #if !defined(UNDER_CE)
    CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &consoleInfo))
      consoleWidth = (USHORT)consoleInfo.dwSize.X;
    #endif
    
    #else
    
#if !defined(__sun)
    struct winsize w;
    if (ioctl(0, TIOCGWINSZ, &w) == 0)
      consoleWidth = w.ws_col;
#endif
    #endif
  }

  CREATE_CODECS_OBJECT

  codecs->CaseSensitive_Change = options.CaseSensitive_Change;
  codecs->CaseSensitive = options.CaseSensitive;
  ThrowException_if_Error(codecs->Load());
  Codecs_AddHashArcHandler(codecs);

  #ifdef Z7_EXTERNAL_CODECS
  {
    g_ExternalCodecs_Ptr = &_externalCodecs;
    UString s;
    codecs->GetCodecsErrorMessage(s);
    if (!s.IsEmpty())
    {
      CStdOutStream &so = (g_StdStream ? *g_StdStream : g_StdOut);
      so << endl << s << endl;
    }
  }
  #endif

  const bool isExtractGroupCommand = options.Command.IsFromExtractGroup();

  if (codecs->Formats.Size() == 0 &&
        (isExtractGroupCommand
        || options.Command.CommandType == NCommandType::kList
        || options.Command.IsFromUpdateGroup()))
  {
    #ifdef Z7_EXTERNAL_CODECS
    if (!codecs->MainDll_ErrorPath.IsEmpty())
    {
      UString s ("Can't load module: ");
      s += fs2us(codecs->MainDll_ErrorPath);
      throw s;
    }
    #endif
    throw kNoFormats;
  }

  CObjectVector<COpenType> types;
  if (!ParseOpenTypes(*codecs, options.ArcType, types))
  {
    throw kUnsupportedArcTypeMessage;
  }


  CIntVector excludedFormats;
  FOR_VECTOR (k, options.ExcludedArcTypes)
  {
    CIntVector tempIndices;
    if (!codecs->FindFormatForArchiveType(options.ExcludedArcTypes[k], tempIndices)
        || tempIndices.Size() != 1)
      throw kUnsupportedArcTypeMessage;
    
    
    
    excludedFormats.AddToUniqueSorted(tempIndices[0]);
    // excludedFormats.Sort();
  }
  
  #ifdef Z7_EXTERNAL_CODECS
  if (isExtractGroupCommand
      || options.Command.IsFromUpdateGroup()
      || options.Command.CommandType == NCommandType::kHash
      || options.Command.CommandType == NCommandType::kBenchmark)
    ThrowException_if_Error(_externalCodecs.Load());
  #endif

  int retCode = NExitCode::kSuccess;
  HRESULT hresultMain = S_OK;

  // bool showStat = options.ShowTime;
  
  /*
  if (!options.EnableHeaders ||
      options.TechMode)
    showStat = false;
  */
  

  if (options.Command.CommandType == NCommandType::kInfo)
  {
    CStdOutStream &so = (g_StdStream ? *g_StdStream : g_StdOut);
    unsigned i;

    #ifdef Z7_EXTERNAL_CODECS
    so << endl << "Libs:" << endl;
    for (i = 0; i < codecs->Libs.Size(); i++)
    {
      PrintLibIndex(so, (int)i);
      const CCodecLib &lib = codecs->Libs[i];
      // if (lib.Version != 0)
      so << ": " << (lib.Version >> 16) << ".";
      PrintNumber(so, lib.Version & 0xffff, 2);
      so << " : " << lib.Path << endl;
    }
    #endif

    so << endl << "Formats:" << endl;
    
    const char * const kArcFlags = "KSNFMGOPBELHXCc+a+m+r+";
    const char * const kArcTimeFlags = "wudn";
    const unsigned kNumArcFlags = (unsigned)strlen(kArcFlags);
    const unsigned kNumArcTimeFlags = (unsigned)strlen(kArcTimeFlags);

    for (i = 0; i < codecs->Formats.Size(); i++)
    {
      const CArcInfoEx &arc = codecs->Formats[i];

      #ifdef Z7_EXTERNAL_CODECS
      PrintLibIndex(so, arc.LibIndex);
      #else
      so << "   ";
      #endif

      so << (char)(arc.UpdateEnabled ? 'C' : ' ');
      
      {
        unsigned b;
        for (b = 0; b < kNumArcFlags; b++)
          so << (char)((arc.Flags & ((UInt32)1 << b)) != 0 ? kArcFlags[b] : '.');
        so << ' ';
      }

      if (arc.TimeFlags != 0)
      {
        unsigned b;
        for (b = 0; b < kNumArcTimeFlags; b++)
          so << (char)((arc.TimeFlags & ((UInt32)1 << b)) != 0 ? kArcTimeFlags[b] : '.');
        so << arc.Get_DefaultTimePrec();
        so << ' ';
      }
      
      so << ' ';
      PrintString(so, arc.Name, 8);
      so << ' ';
      UString s;
      
      FOR_VECTOR (t, arc.Exts)
      {
        if (t != 0)
          s.Add_Space();
        const CArcExtInfo &ext = arc.Exts[t];
        s += ext.Ext;
        if (!ext.AddExt.IsEmpty())
        {
          s += " (";
          s += ext.AddExt;
          s.Add_Char(')');
        }
      }
      
      PrintString(so, s, 13);
      so << ' ';
      
      if (arc.SignatureOffset != 0)
        so << "offset=" << arc.SignatureOffset << ' ';

      // so << "numSignatures = " << arc.Signatures.Size() << " ";

      FOR_VECTOR(si, arc.Signatures)
      {
        if (si != 0)
          so << "  ||  ";

        const CByteBuffer &sig = arc.Signatures[si];
        
        for (size_t j = 0; j < sig.Size(); j++)
        {
          if (j != 0)
            so << ' ';
          const unsigned b = sig.ConstData()[j];
          if (b > 0x20 && b < 0x80)
          {
            so << (char)b;
          }
          else
          {
            so << GET_HEX_CHAR_UPPER(b >> 4);
            so << GET_HEX_CHAR_UPPER(b & 15);
          }
        }
      }
      so << endl;
    }

    so << endl << "Codecs:" << endl; //  << "Lib          ID Name" << endl;

    for (i = 0; i < g_NumCodecs; i++)
    {
      const CCodecInfo &cod = *g_Codecs[i];

      PrintLibIndex(so, -1);

      if (cod.NumStreams == 1)
        so << ' ';
      else
        so << cod.NumStreams;
      
      so << (char)(cod.CreateEncoder ? 'E' : ' ');
      so << (char)(cod.CreateDecoder ? 'D' : ' ');
      so << (char)(cod.IsFilter      ? 'F' : ' ');

      so << ' ';
      PrintHexId(so, cod.Id);
      so << ' ' << cod.Name << endl;
    }


    #ifdef Z7_EXTERNAL_CODECS

    UInt32 numMethods;
    if (_externalCodecs.GetCodecs->GetNumMethods(&numMethods) == S_OK)
    for (UInt32 j = 0; j < numMethods; j++)
    {
      PrintLibIndex(so, codecs->GetCodec_LibIndex(j));

      UInt32 numStreams = codecs->GetCodec_NumStreams(j);
      if (numStreams == 1)
        so << ' ';
      else
        so << numStreams;
      
      so << (char)(codecs->GetCodec_EncoderIsAssigned(j) ? 'E' : ' ');
      so << (char)(codecs->GetCodec_DecoderIsAssigned(j) ? 'D' : ' ');
      {
        bool isFilter_Assigned;
        const bool isFilter = codecs->GetCodec_IsFilter(j, isFilter_Assigned);
        so << (char)(isFilter ? 'F' : isFilter_Assigned ? ' ' : '*');
      }


      so << ' ';
      UInt64 id;
      HRESULT res = codecs->GetCodec_Id(j, id);
      if (res != S_OK)
        id = (UInt64)(Int64)-1;
      PrintHexId(so, id);
      so << ' ' << codecs->GetCodec_Name(j) << endl;
    }

    #endif
    

    so << endl << "Hashers:" << endl; //  << " L Size       ID Name" << endl;

    for (i = 0; i < g_NumHashers; i++)
    {
      const CHasherInfo &codec = *g_Hashers[i];
      PrintLibIndex(so, -1);
      PrintUInt32(so, codec.DigestSize, 4);
      so << ' ';
      PrintHexId(so, codec.Id);
      so << ' ' << codec.Name << endl;
    }

    #ifdef Z7_EXTERNAL_CODECS
    
    numMethods = _externalCodecs.GetHashers->GetNumHashers();
    for (UInt32 j = 0; j < numMethods; j++)
    {
      PrintLibIndex(so, codecs->GetHasherLibIndex(j));
      PrintUInt32(so, codecs->GetHasherDigestSize(j), 4);
      so << ' ';
      PrintHexId(so, codecs->GetHasherId(j));
      so << ' ' << codecs->GetHasherName(j) << endl;
    }

    #endif
    
  }
  else if (options.Command.CommandType == NCommandType::kBenchmark)
  {
    CStdOutStream &so = (g_StdStream ? *g_StdStream : g_StdOut);
    hresultMain = BenchCon(EXTERNAL_CODECS_VARS_L
        options.Properties, options.NumIterations, (FILE *)so);
    if (hresultMain == S_FALSE)
    {
      so << endl;
      if (g_ErrStream)
        *g_ErrStream << "\nDecoding ERROR\n";
      retCode = NExitCode::kFatalError;
      hresultMain = S_OK;
    }
  }
  else if (isExtractGroupCommand || options.Command.CommandType == NCommandType::kList)
  {
    UStringVector ArchivePathsSorted;
    UStringVector ArchivePathsFullSorted;

    if (options.StdInMode)
    {
      ArchivePathsSorted.Add(options.ArcName_for_StdInMode);
      ArchivePathsFullSorted.Add(options.ArcName_for_StdInMode);
    }
    else
    {
      CExtractScanConsole scan;
      
      scan.Init(options.EnableHeaders ? g_StdStream : NULL,
          g_ErrStream, percentsStream,
          options.DisablePercents);
      scan.SetWindowWidth(consoleWidth);

      if (g_StdStream && options.EnableHeaders)
        *g_StdStream << "Scanning the drive for archives:" << endl;

      CDirItemsStat st;

      scan.StartScanning();

      hresultMain = EnumerateDirItemsAndSort(
          options.arcCensor,
          NWildcard::k_RelatPath,
          UString(), // addPathPrefix
          ArchivePathsSorted,
          ArchivePathsFullSorted,
          st,
          &scan);

      scan.CloseScanning();

      if (hresultMain == S_OK)
      {
        if (options.EnableHeaders)
          scan.PrintStat(st);
      }
      else
      {
        /*
        if (res != E_ABORT)
        {
          throw CSystemException(res);
          // errorInfo.Message = "Scanning error";
        }
        return res;
        */
      }
    }

    if (hresultMain == S_OK) {
    if (isExtractGroupCommand)
    {
      CExtractCallbackConsole *ecs = new CExtractCallbackConsole;
      CMyComPtr<IFolderArchiveExtractCallback> extractCallback = ecs;

      #ifndef Z7_NO_CRYPTO
      ecs->PasswordIsDefined = options.PasswordEnabled;
      ecs->Password = options.Password;
      #endif

      ecs->Init(g_StdStream, g_ErrStream, percentsStream, options.DisablePercents);
      ecs->MultiArcMode = (ArchivePathsSorted.Size() > 1);

      ecs->LogLevel = options.LogLevel;
      ecs->PercentsNameLevel = percentsNameLevel;
      
      if (percentsStream)
        ecs->SetWindowWidth(consoleWidth);

      /*
      COpenCallbackConsole openCallback;
      openCallback.Init(g_StdStream, g_ErrStream);

      #ifndef Z7_NO_CRYPTO
      openCallback.PasswordIsDefined = options.PasswordEnabled;
      openCallback.Password = options.Password;
      #endif
      */

      CExtractOptions eo;
      (CExtractOptionsBase &)eo = options.ExtractOptions;
      
      eo.StdInMode = options.StdInMode;
      eo.StdOutMode = options.StdOutMode;
      eo.YesToAll = options.YesToAll;
      eo.TestMode = options.Command.IsTestCommand();
      
      #ifndef Z7_SFX
      eo.Properties = options.Properties;
      #endif

      UString errorMessage;
      CDecompressStat stat;
      CHashBundle hb;
      IHashCalc *hashCalc = NULL;

      if (!options.HashMethods.IsEmpty())
      {
        hashCalc = &hb;
        ThrowException_if_Error(hb.SetMethods(EXTERNAL_CODECS_VARS_L options.HashMethods));
        // hb.Init();
      }
      
      hresultMain = Extract(
          // EXTERNAL_CODECS_VARS_L
          codecs,
          types,
          excludedFormats,
          ArchivePathsSorted,
          ArchivePathsFullSorted,
          options.Censor.Pairs.Front().Head,
          eo,
          ecs, ecs, ecs,
          hashCalc, errorMessage, stat);
      
      ecs->ClosePercents();

      if (!errorMessage.IsEmpty())
      {
        if (g_ErrStream)
          *g_ErrStream << endl << "ERROR:" << endl << errorMessage << endl;
        if (hresultMain == S_OK)
          hresultMain = E_FAIL;
      }

      CStdOutStream *so = g_StdStream;

      bool isError = false;

      if (so)
      {
        *so << endl;
        
        if (ecs->NumTryArcs > 1)
        {
          *so << "Archives: " << ecs->NumTryArcs << endl;
          *so << "OK archives: " << ecs->NumOkArcs << endl;
        }
      }

      if (ecs->NumCantOpenArcs != 0)
      {
        isError = true;
        if (so)
          *so << "Can't open as archive: " << ecs->NumCantOpenArcs << endl;
      }
      
      if (ecs->NumArcsWithError != 0)
      {
        isError = true;
        if (so)
          *so << "Archives with Errors: " << ecs->NumArcsWithError << endl;
      }
      
      if (so)
      {
        if (ecs->NumArcsWithWarnings != 0)
          *so << "Archives with Warnings: " << ecs->NumArcsWithWarnings << endl;
        
        if (ecs->NumOpenArcWarnings != 0)
        {
          *so << endl;
          if (ecs->NumOpenArcWarnings != 0)
            *so << "Warnings: " << ecs->NumOpenArcWarnings << endl;
        }
      }
      
      if (ecs->NumOpenArcErrors != 0)
      {
        isError = true;
        if (so)
        {
          *so << endl;
          if (ecs->NumOpenArcErrors != 0)
            *so << "Open Errors: " << ecs->NumOpenArcErrors << endl;
        }
      }

      if (isError)
        retCode = NExitCode::kFatalError;
      
      if (so) {
      if (ecs->NumArcsWithError != 0 || ecs->NumFileErrors != 0)
      {
        // if (ecs->NumArchives > 1)
        {
          *so << endl;
          if (ecs->NumFileErrors != 0)
            *so << "Sub items Errors: " << ecs->NumFileErrors << endl;
        }
      }
      else if (hresultMain == S_OK)
      {
        if (stat.NumFolders != 0)
          *so << "Folders: " << stat.NumFolders << endl;
        if (stat.NumFiles != 1 || stat.NumFolders != 0 || stat.NumAltStreams != 0)
          *so << "Files: " << stat.NumFiles << endl;
        if (stat.NumAltStreams != 0)
        {
          *so << "Alternate Streams: " << stat.NumAltStreams << endl;
          *so << "Alternate Streams Size: " << stat.AltStreams_UnpackSize << endl;
        }
        
        *so
          << "Size:       " << stat.UnpackSize << endl
          << "Compressed: " << stat.PackSize << endl;
        if (hashCalc)
        {
          *so << endl;
          PrintHashStat(*so, hb);
        }
      }
      } // if (so)
    }
    else // if_(!isExtractGroupCommand)
    {
      UInt64 numErrors = 0;
      UInt64 numWarnings = 0;
      
      // options.ExtractNtOptions.StoreAltStreams = true, if -sns[-] is not definmed

      CListOptions lo;
      lo.ExcludeDirItems = options.Censor.ExcludeDirItems;
      lo.ExcludeFileItems = options.Censor.ExcludeFileItems;
      lo.DisablePercents = options.DisablePercents;

      hresultMain = ListArchives(
          lo,
          codecs,
          types,
          excludedFormats,
          options.StdInMode,
          ArchivePathsSorted,
          ArchivePathsFullSorted,
          options.ExtractOptions.NtOptions.AltStreams.Val,
          options.AltStreams.Val, // we don't want to show AltStreams by default
          options.Censor.Pairs.Front().Head,
          options.EnableHeaders,
          options.TechMode,
          #ifndef Z7_NO_CRYPTO
          options.PasswordEnabled,
          options.Password,
          #endif
          &options.Properties,
          numErrors, numWarnings);

      if (options.EnableHeaders)
        if (numWarnings > 0)
          g_StdOut << endl << "Warnings: " << numWarnings << endl;
      
      if (numErrors > 0)
      {
        if (options.EnableHeaders)
          g_StdOut << endl << "Errors: " << numErrors << endl;
        retCode = NExitCode::kFatalError;
      }
    } // if_(isExtractGroupCommand)
    } // if_(hresultMain == S_OK)
  }
  else if (options.Command.IsFromUpdateGroup())
  {
   #ifdef Z7_EXTRACT_ONLY
    throw "update commands are not implemented";
   #else
    CUpdateOptions &uo = options.UpdateOptions;
    if (uo.SfxMode && uo.SfxModule.IsEmpty())
      uo.SfxModule = kDefaultSfxModule;

    COpenCallbackConsole openCallback;
    openCallback.Init(g_StdStream, g_ErrStream, percentsStream, options.DisablePercents);

    #ifndef Z7_NO_CRYPTO
    bool passwordIsDefined =
        (options.PasswordEnabled && !options.Password.IsEmpty());
    openCallback.PasswordIsDefined = passwordIsDefined;
    openCallback.Password = options.Password;
    #endif

    CUpdateCallbackConsole callback;
    callback.LogLevel = options.LogLevel;
    callback.PercentsNameLevel = percentsNameLevel;

    if (percentsStream)
      callback.SetWindowWidth(consoleWidth);

    #ifndef Z7_NO_CRYPTO
    callback.PasswordIsDefined = passwordIsDefined;
    callback.AskPassword = (options.PasswordEnabled && options.Password.IsEmpty());
    callback.Password = options.Password;
    #endif

    callback.StdOutMode = uo.StdOutMode;
    callback.Init(
      // NULL,
      g_StdStream, g_ErrStream, percentsStream, options.DisablePercents);

    CUpdateErrorInfo errorInfo;

    /*
    if (!uo.Init(codecs, types, options.ArchiveName))
      throw kUnsupportedUpdateArcType;
    */
    hresultMain = UpdateArchive(codecs,
        types,
        options.ArchiveName,
        options.Censor,
        uo,
        errorInfo, &openCallback, &callback, true);

    callback.ClosePercents2();

    CStdOutStream *se = g_StdStream;
    if (!se)
      se = g_ErrStream;

    retCode = WarningsCheck(hresultMain, callback, errorInfo,
        g_StdStream, se,
        true // options.EnableHeaders
        );
   #endif
  }
  else if (options.Command.CommandType == NCommandType::kHash)
  {
    const CHashOptions &uo = options.HashOptions;

    CHashCallbackConsole callback;
    if (percentsStream)
      callback.SetWindowWidth(consoleWidth);
  
    callback.Init(g_StdStream, g_ErrStream, percentsStream, options.DisablePercents);
    callback.PrintHeaders = options.EnableHeaders;
    callback.PrintFields = options.ListFields;

    AString errorInfoString;
    hresultMain = HashCalc(EXTERNAL_CODECS_VARS_L
        options.Censor, uo,
        errorInfoString, &callback);
    CUpdateErrorInfo errorInfo;
    errorInfo.Message = errorInfoString;
    CStdOutStream *se = g_StdStream;
    if (!se)
      se = g_ErrStream;
    retCode = WarningsCheck(hresultMain, callback, errorInfo, g_StdStream, se, options.EnableHeaders);
  }
  else
    ShowMessageAndThrowException(kUserErrorMessage, NExitCode::kUserError);

  if (options.ShowTime && g_StdStream)
    PrintStat(
      #ifndef _WIN32
        startTime
      #endif
    );

  ThrowException_if_Error(hresultMain);

  return retCode;
}
