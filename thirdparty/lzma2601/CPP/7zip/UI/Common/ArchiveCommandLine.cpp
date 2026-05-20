// ArchiveCommandLine.cpp

#include "StdAfx.h"
#undef printf
#undef sprintf

#ifdef _WIN32
#ifndef UNDER_CE
#include <io.h>
#endif
#else
// for isatty()
#include <unistd.h>
#endif

#include <stdio.h>

#ifdef Z7_LARGE_PAGES
#include "../../../../C/Alloc.h"
#endif

#include "../../../Common/IntToString.h"
#include "../../../Common/ListFileUtils.h"
#include "../../../Common/MyException.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/StringToInt.h"

#include "../../../Windows/ErrorMsg.h"
#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileName.h"
#include "../../../Windows/PropVariantConv.h"
#include "../../../Windows/System.h"
#ifdef _WIN32
#include "../../../Windows/FileMapping.h"
#include "../../../Windows/MemoryLock.h"
#include "../../../Windows/Synchronization.h"
#endif

#include "ArchiveCommandLine.h"
#include "EnumDirItems.h"
#include "Update.h"
#include "UpdateAction.h"

extern bool g_CaseSensitive;
extern bool g_PathTrailReplaceMode;

#ifdef Z7_LARGE_PAGES
extern
bool g_LargePagesMode;
bool g_LargePagesMode = false;
#endif

/*
#ifdef ENV_HAVE_LSTAT
EXTERN_C_BEGIN
extern int global_use_lstat;
EXTERN_C_END
#endif
*/

#ifdef UNDER_CE

#define MY_IS_TERMINAL(x) false;

#else

static bool MY_IS_TERMINAL(FILE *x)
{
#ifdef _WIN32
  /*
crt/stdio.h:
typedef struct _iobuf FILE;
#define stdin  (&_iob[0])
#define stdout (&_iob[1])
#define stderr (&_iob[2])
*/
  // fprintf(stderr, "\nMY_IS_TERMINAL = %p", x);
  const int fd = _fileno(x);
  /* (fd) is 0, 1 or 2 in console program.
     docs: If stdout or stderr is not associated with
     an output stream (for example, in a Windows application
     without a console window), the file descriptor returned is -2.
     In previous versions, the file descriptor returned was -1.
  */
  if (fd < 0) // is not associated with an output stream application (without a console window)
    return false;
  // fprintf(stderr, "\n\nstderr _fileno(%p) = %d", x, fd);
  if (!_isatty(fd))
    return false;
  // fprintf(stderr, "\nisatty_val = true");
  const HANDLE h = (HANDLE)_get_osfhandle(fd);
  /* _get_osfhandle() returns intptr_t in new SDK, or long in MSVC6.
     Also it can return (INVALID_HANDLE_VALUE).
     docs: _get_osfhandle also returns the special value -2 when
     the file descriptor is not associated with a stream
     in old msvcrt.dll: it returns (-1) for incorrect value
  */
  // fprintf(stderr, "\n_get_osfhandle() = %p", (void *)h);
  if (h == NULL || h == INVALID_HANDLE_VALUE)
    return false;
  DWORD st;
  // fprintf(stderr, "\nGetConsoleMode() = %u", (unsigned)GetConsoleMode(h, &st));
  return GetConsoleMode(h, &st) != 0;
#else
  return isatty(fileno(x)) != 0;
#endif
}

#endif

using namespace NCommandLineParser;
using namespace NWindows;
using namespace NFile;

static bool StringToUInt32(const wchar_t *s, UInt32 &v)
{
  if (*s == 0)
    return false;
  const wchar_t *end;
  v = ConvertStringToUInt32(s, &end);
  return *end == 0;
}


namespace NKey {
enum Enum
{
  kHelp1 = 0,
  kHelp2,
  kHelp3,
  
  kDisableHeaders,
  kDisablePercents,
  kShowTime,
  kLogLevel,

  kOutStream,
  kErrStream,
  kPercentStream,

  kYes,

  kShowDialog,
  kOverwrite,

  kArchiveType,
  kExcludedArcType,

  kProperty,
  kOutputDir,
  kWorkingDir,
  
  kInclude,
  kExclude,
  kArInclude,
  kArExclude,
  kNoArName,

  kUpdate,
  kVolume,
  kRecursed,

  kAffinity,
  kSfx,
  kEmail,
  kHash,
  // kHashGenFile,
  kHashDir,
  kExtractMemLimit,
 
  kStdIn,
  kStdOut,

  kLargePages,
  kListfileCharSet,
  kConsoleCharSet,
  kTechMode,
  kListFields,
  kListPathSlash,
  kListTimestampUTC,
  
  kPreserveATime,
  kShareForWrite,
  kStopAfterOpenError,
  kCaseSensitive,
  kArcNameMode,

  kUseSlashMark,
  kDisableWildcardParsing,
  kElimDup,
  kFullPathMode,
  kOutDirMode,
  
  kHardLinks,
  kSymLinks_AllowDangerous,
  kSymLinks,
  kNtSecurity,

  kStoreOwnerId,
  kStoreOwnerName,
  
  kZoneFile,
  kAltStreams,
  kReplaceColonForAltStream,
  kWriteToAltStreamIfColon,

  kNameTrailReplace,

  kDeleteAfterCompressing,
  kSetArcMTime

  #ifndef Z7_NO_CRYPTO
  , kPassword
  #endif
};

}


static const wchar_t kRecursedIDChar = 'r';
static const char * const kRecursedPostCharSet = "0-";

static const char * const k_ArcNameMode_PostCharSet = "sea";

static const char * const k_Stream_PostCharSet = "012";

static inline EArcNameMode ParseArcNameMode(int postCharIndex)
{
  switch (postCharIndex)
  {
    case 1: return k_ArcNameMode_Exact;
    case 2: return k_ArcNameMode_Add;
    default: return k_ArcNameMode_Smart;
  }
}

namespace NRecursedPostCharIndex {
  enum EEnum
  {
    kWildcardRecursionOnly = 0,
    kNoRecursion = 1
  };
}

// static const char
#define kImmediateNameID '!'
#ifdef _WIN32
#define kMapNameID '#'
#endif
#define kFileListID '@'

static const Byte kSomeCludePostStringMinSize = 2; // at least <@|!><N>ame must be
static const Byte kSomeCludeAfterRecursedPostStringMinSize = 2; // at least <@|!><N>ame must be

static const char * const kOverwritePostCharSet = "asut";

static const NExtract::NOverwriteMode::EEnum k_OverwriteModes[] =
{
  NExtract::NOverwriteMode::kOverwrite,
  NExtract::NOverwriteMode::kSkip,
  NExtract::NOverwriteMode::kRename,
  NExtract::NOverwriteMode::kRenameExisting
};



#define SWFRM_3(t, mu, mi) t, mu, mi, NULL

#define SWFRM_1(t) SWFRM_3(t, false, 0)
#define SWFRM_SIMPLE SWFRM_1(NSwitchType::kSimple)
#define SWFRM_MINUS  SWFRM_1(NSwitchType::kMinus)
#define SWFRM_STRING SWFRM_1(NSwitchType::kString)

#define SWFRM_STRING_SINGL(mi) SWFRM_3(NSwitchType::kString, false, mi)
#define SWFRM_STRING_MULT(mi)  SWFRM_3(NSwitchType::kString, true, mi)


static const CSwitchForm kSwitchForms[] =
{
  { "?", SWFRM_SIMPLE },
  { "h", SWFRM_SIMPLE },
  { "-help", SWFRM_SIMPLE },
  
  { "ba", SWFRM_SIMPLE },
  { "bd", SWFRM_SIMPLE },
  { "bt", SWFRM_SIMPLE },
  { "bb", SWFRM_STRING_SINGL(0) },

  { "bso", NSwitchType::kChar, false, 1, k_Stream_PostCharSet },
  { "bse", NSwitchType::kChar, false, 1, k_Stream_PostCharSet },
  { "bsp", NSwitchType::kChar, false, 1, k_Stream_PostCharSet },
  
  { "y", SWFRM_SIMPLE },
  
  { "ad", SWFRM_SIMPLE },
  { "ao", NSwitchType::kChar, false, 1, kOverwritePostCharSet},

  { "t",  SWFRM_STRING_SINGL(1) },
  { "stx", SWFRM_STRING_MULT(1) },
  
  { "m",  SWFRM_STRING_MULT(1) },
  { "o",  SWFRM_STRING_SINGL(1) },
  { "w",  SWFRM_STRING },

  { "i",  SWFRM_STRING_MULT(kSomeCludePostStringMinSize) },
  { "x",  SWFRM_STRING_MULT(kSomeCludePostStringMinSize) },
  { "ai", SWFRM_STRING_MULT(kSomeCludePostStringMinSize) },
  { "ax", SWFRM_STRING_MULT(kSomeCludePostStringMinSize) },
  { "an", SWFRM_SIMPLE },
  
  { "u",  SWFRM_STRING_MULT(1) },
  { "v",  SWFRM_STRING_MULT(1) },
  { "r",  NSwitchType::kChar, false, 0, kRecursedPostCharSet },
  
  { "stm", SWFRM_STRING },
  { "sfx", SWFRM_STRING },
  { "seml", SWFRM_STRING_SINGL(0) },
  { "scrc", SWFRM_STRING_MULT(0) },
  // { "scrf", SWFRM_STRING_SINGL(1) },
  { "shd", SWFRM_STRING_SINGL(1) },
  { "smemx", SWFRM_STRING },
  
  { "si", SWFRM_STRING },
  { "so", SWFRM_SIMPLE },

  { "slp", SWFRM_STRING },
  { "scs", SWFRM_STRING },
  { "scc", SWFRM_STRING },
  { "slt", SWFRM_SIMPLE },
  { "slf", SWFRM_STRING_SINGL(1) },
  { "slsl", SWFRM_MINUS },
  { "slmu", SWFRM_MINUS },

  { "ssp", SWFRM_SIMPLE },
  { "ssw", SWFRM_SIMPLE },
  { "sse", SWFRM_SIMPLE },
  { "ssc", SWFRM_MINUS },
  { "sa",  NSwitchType::kChar, false, 1, k_ArcNameMode_PostCharSet },
  
  { "spm", SWFRM_STRING_SINGL(0) },
  { "spd", SWFRM_SIMPLE },
  { "spe", SWFRM_MINUS },
  { "spf", SWFRM_STRING_SINGL(0) },
  { "spo", NSwitchType::kChar, false, 1, "dcr" }, // kOutDirMode

  { "snh", SWFRM_MINUS },
  { "snld", SWFRM_STRING },
  { "snl", SWFRM_MINUS },
  { "sni", SWFRM_SIMPLE },

  { "snoi", SWFRM_MINUS },
  { "snon", SWFRM_MINUS },
  
  { "snz", SWFRM_STRING_SINGL(0) },
  { "sns", SWFRM_MINUS },
  { "snr", SWFRM_SIMPLE },
  { "snc", SWFRM_SIMPLE },
  
  { "snt", SWFRM_MINUS },
  
  { "sdel", SWFRM_SIMPLE },
  { "stl", SWFRM_SIMPLE }

  #ifndef Z7_NO_CRYPTO
  , { "p", SWFRM_STRING }
  #endif
};

static const char * const kUniversalWildcard = "*";
static const unsigned kMinNonSwitchWords = 1;
static const unsigned kCommandIndex = 0;

// static const char * const kUserErrorMessage  = "Incorrect command line";
// static const char * const kCannotFindListFile = "Cannot find listfile";
static const char * const kIncorrectListFile = "Incorrect item in listfile.\nCheck charset encoding and -scs switch.";
static const char * const kTerminalOutError = "I won't write compressed data to a terminal";
static const char * const kSameTerminalError = "I won't write data and program's messages to same stream";
static const char * const kEmptyFilePath = "Empty file path";

bool CArcCommand::IsFromExtractGroup() const
{
  switch ((int)CommandType)
  {
    case NCommandType::kTest:
    case NCommandType::kExtract:
    case NCommandType::kExtractFull:
      return true;
    default:
      return false;
  }
}

NExtract::NPathMode::EEnum CArcCommand::GetPathMode() const
{
  switch ((int)CommandType)
  {
    case NCommandType::kTest:
    case NCommandType::kExtractFull:
      return NExtract::NPathMode::kFullPaths;
    default:
      return NExtract::NPathMode::kNoPaths;
  }
}

bool CArcCommand::IsFromUpdateGroup() const
{
  switch ((int)CommandType)
  {
    case NCommandType::kAdd:
    case NCommandType::kUpdate:
    case NCommandType::kDelete:
    case NCommandType::kRename:
      return true;
    default:
      return false;
  }
}

static NRecursedType::EEnum GetRecursedTypeFromIndex(int index)
{
  switch (index)
  {
    case NRecursedPostCharIndex::kWildcardRecursionOnly:
      return NRecursedType::kWildcardOnlyRecursed;
    case NRecursedPostCharIndex::kNoRecursion:
      return NRecursedType::kNonRecursed;
    default:
      return NRecursedType::kRecursed;
  }
}

static const char * const g_Commands = "audtexlbih";

static bool ParseArchiveCommand(const UString &commandString, CArcCommand &command)
{
  UString s (commandString);
  s.MakeLower_Ascii();
  if (s.Len() == 1)
  {
    if (s[0] > 0x7F)
      return false;
    int index = FindCharPosInString(g_Commands, (char)s[0]);
    if (index < 0)
      return false;
    command.CommandType = (NCommandType::EEnum)index;
    return true;
  }
  if (s.Len() == 2 && s[0] == 'r' && s[1] == 'n')
  {
    command.CommandType = (NCommandType::kRename);
    return true;
  }
  return false;
}

// ------------------------------------------------------------------
// filenames functions

struct CNameOption
{
  bool Include;
  bool WildcardMatching;
  Byte MarkMode;
  NRecursedType::EEnum RecursedType;

  CNameOption():
      Include(true),
      WildcardMatching(true),
      MarkMode(NWildcard::kMark_FileOrDir),
      RecursedType(NRecursedType::kNonRecursed)
      {}
};


static void AddNameToCensor(NWildcard::CCensor &censor,
    const CNameOption &nop, const UString &name)
{
  bool recursed = false;

  switch ((int)nop.RecursedType)
  {
    case NRecursedType::kWildcardOnlyRecursed:
      recursed = DoesNameContainWildcard(name);
      break;
    case NRecursedType::kRecursed:
      recursed = true;
      break;
    default:
      break;
  }

  NWildcard::CCensorPathProps props;
  props.Recursive = recursed;
  props.WildcardMatching = nop.WildcardMatching;
  props.MarkMode = nop.MarkMode;
  censor.AddPreItem(nop.Include, name, props);
}

#ifndef Z7_EXTRACT_ONLY
static void AddRenamePair(CObjectVector<CRenamePair> *renamePairs,
    const UString &oldName, const UString &newName, NRecursedType::EEnum type,
    bool wildcardMatching)
{
  CRenamePair &pair = renamePairs->AddNew();
  pair.OldName = oldName;
  pair.NewName = newName;
  pair.RecursedType = type;
  pair.WildcardParsing = wildcardMatching;

  if (!pair.Prepare())
  {
    UString val;
    val += pair.OldName;
    val.Add_LF();
    val += pair.NewName;
    val.Add_LF();
    if (type == NRecursedType::kRecursed)
      val += "-r";
    else if (type == NRecursedType::kWildcardOnlyRecursed)
      val += "-r0";
    throw CArcCmdLineException("Unsupported rename command:", val);
  }
}
#endif

static void AddToCensorFromListFile(
    CObjectVector<CRenamePair> *renamePairs,
    NWildcard::CCensor &censor,
    const CNameOption &nop, LPCWSTR fileName, UInt32 codePage)
{
  UStringVector names;
  /*
  if (!NFind::DoesFileExist_FollowLink(us2fs(fileName)))
    throw CArcCmdLineException(kCannotFindListFile, fileName);
  */
  DWORD lastError = 0;
  if (!ReadNamesFromListFile2(us2fs(fileName), names, codePage, lastError))
  {
    if (lastError != 0)
    {
      UString m;
      m = "The file operation error for listfile";
      m.Add_LF();
      m += NError::MyFormatMessage(lastError);
      throw CArcCmdLineException(m, fileName);
    }
    throw CArcCmdLineException(kIncorrectListFile, fileName);
  }
  if (renamePairs)
  {
    #ifndef Z7_EXTRACT_ONLY
    if ((names.Size() & 1) != 0)
      throw CArcCmdLineException(kIncorrectListFile, fileName);
    for (unsigned i = 0; i < names.Size(); i += 2)
    {
      // change type !!!!
      AddRenamePair(renamePairs, names[i], names[i + 1], nop.RecursedType, nop.WildcardMatching);
    }
    #else
    throw "not implemented";
    #endif
  }
  else
    FOR_VECTOR (i, names)
      AddNameToCensor(censor, nop, names[i]);
}

static void AddToCensorFromNonSwitchesStrings(
    CObjectVector<CRenamePair> *renamePairs,
    unsigned startIndex,
    NWildcard::CCensor &censor,
    const UStringVector &nonSwitchStrings,
    int stopSwitchIndex,
    const CNameOption &nop,
    bool thereAreSwitchIncludes, UInt32 codePage)
{
  // another default
  if ((renamePairs || nonSwitchStrings.Size() == startIndex) && !thereAreSwitchIncludes)
  {
    /* for rename command: -i switch sets the mask for archive item reading.
       if (thereAreSwitchIncludes), { we don't use UniversalWildcard. }
       also for non-rename command: we set UniversalWildcard, only if there are no nonSwitches. */
    // we use default fileds in (CNameOption) for UniversalWildcard.
    CNameOption nop2;
    // recursive mode is not important for UniversalWildcard (*)
    // nop2.RecursedType = nop.RecursedType; // we don't need it
    /*
    nop2.RecursedType = NRecursedType::kNonRecursed;
    nop2.Include = true;
    nop2.WildcardMatching = true;
    nop2.MarkMode = NWildcard::kMark_FileOrDir;
    */
    AddNameToCensor(censor, nop2, UString(kUniversalWildcard));
  }

  int oldIndex = -1;
  
  if (stopSwitchIndex < 0)
    stopSwitchIndex = (int)nonSwitchStrings.Size();

  for (unsigned i = startIndex; i < nonSwitchStrings.Size(); i++)
  {
    const UString &s = nonSwitchStrings[i];
    if (s.IsEmpty())
      throw CArcCmdLineException(kEmptyFilePath);
    if (i < (unsigned)stopSwitchIndex && s[0] == kFileListID)
      AddToCensorFromListFile(renamePairs, censor, nop, s.Ptr(1), codePage);
    else if (renamePairs)
    {
      #ifndef Z7_EXTRACT_ONLY
      if (oldIndex == -1)
        oldIndex = (int)i;
      else
      {
        // NRecursedType::EEnum type is used for global wildcard (-i! switches)
        AddRenamePair(renamePairs, nonSwitchStrings[(unsigned)oldIndex], s, NRecursedType::kNonRecursed, nop.WildcardMatching);
        // AddRenamePair(renamePairs, nonSwitchStrings[oldIndex], s, type);
        oldIndex = -1;
      }
      #else
      throw "not implemented";
      #endif
    }
    else
      AddNameToCensor(censor, nop, s);
  }
  
  if (oldIndex != -1)
  {
    throw CArcCmdLineException("There is no second file name for rename pair:", nonSwitchStrings[(unsigned)oldIndex]);
  }
}

#ifdef _WIN32

struct CEventSetEnd
{
  UString Name;
  
  CEventSetEnd(const wchar_t *name): Name(name) {}
  ~CEventSetEnd()
  {
    NSynchronization::CManualResetEvent event;
    if (event.Open(EVENT_MODIFY_STATE, false, GetSystemString(Name)) == 0)
      event.Set();
  }
};

static const char * const k_IncorrectMapCommand = "Incorrect Map command";

static const char *ParseMapWithPaths(
    NWildcard::CCensor &censor,
    const UString &s2,
    const CNameOption &nop)
{
  UString s (s2);
  const int pos = s.Find(L':');
  if (pos < 0)
    return k_IncorrectMapCommand;
  const int pos2 = s.Find(L':', (unsigned)(pos + 1));
  if (pos2 < 0)
    return k_IncorrectMapCommand;

  CEventSetEnd eventSetEnd((const wchar_t *)s + (unsigned)(pos2 + 1));
  s.DeleteFrom((unsigned)pos2);
  UInt32 size;
  if (!StringToUInt32(s.Ptr((unsigned)(pos + 1)), size)
      || size < sizeof(wchar_t)
      || size > ((UInt32)1 << 31)
      || size % sizeof(wchar_t) != 0)
    return "Unsupported Map data size";

  s.DeleteFrom((unsigned)pos);
  CFileMapping map;
  if (map.Open(FILE_MAP_READ, GetSystemString(s)) != 0)
    return "Cannot open mapping";
  const LPVOID data = map.Map(FILE_MAP_READ, 0, size);
  if (!data)
    return "MapViewOfFile error";
  CFileUnmapper unmapper(data);

  UString name;
  const wchar_t *p = (const wchar_t *)data;
  if (*p != 0) // data format marker
    return "Unsupported Map data";
  const UInt32 numChars = size / sizeof(wchar_t);
  for (UInt32 i = 1; i < numChars; i++)
  {
    const wchar_t c = p[i];
    if (c == 0)
    {
      // MessageBoxW(0, name, L"7-Zip", 0);
      AddNameToCensor(censor, nop, name);
      name.Empty();
    }
    else
      name += c;
  }
  if (!name.IsEmpty())
    return "Map data error";

  return NULL;
}

#endif

static void AddSwitchWildcardsToCensor(
    NWildcard::CCensor &censor,
    const UStringVector &strings,
    const CNameOption &nop,
    UInt32 codePage)
{
  const char *errorMessage = NULL;
  unsigned i;
  for (i = 0; i < strings.Size(); i++)
  {
    const UString &name = strings[i];
    unsigned pos = 0;
    
    if (name.Len() < kSomeCludePostStringMinSize)
    {
      errorMessage = "Too short switch";
      break;
    }

    if (!nop.Include)
    {
      if (name.IsEqualTo_Ascii_NoCase("td"))
      {
        censor.ExcludeDirItems = true;
        continue;
      }
      if (name.IsEqualTo_Ascii_NoCase("tf"))
      {
        censor.ExcludeFileItems = true;
        continue;
      }
    }

    CNameOption nop2 = nop;

    bool type_WasUsed = false;
    bool recursed_WasUsed = false;
    bool matching_WasUsed = false;
    bool error = false;

    for (;;)
    {
      wchar_t c = ::MyCharLower_Ascii(name[pos]);
      if (c == kRecursedIDChar)
      {
        if (recursed_WasUsed)
        {
          error = true;
          break;
        }
        recursed_WasUsed = true;
        pos++;
        c = name[pos];
        int index = -1;
        if (c <= 0x7F)
          index = FindCharPosInString(kRecursedPostCharSet, (char)c);
        nop2.RecursedType = GetRecursedTypeFromIndex(index);
        if (index >= 0)
        {
          pos++;
          continue;
        }
      }
      
      if (c == 'w')
      {
        if (matching_WasUsed)
        {
          error = true;
          break;
        }
        matching_WasUsed = true;
        nop2.WildcardMatching = true;
        pos++;
        if (name[pos] == '-')
        {
          nop2.WildcardMatching = false;
          pos++;
        }
      }
      else if (c == 'm')
      {
        if (type_WasUsed)
        {
          error = true;
          break;
        }
        type_WasUsed = true;
        pos++;
        nop2.MarkMode = NWildcard::kMark_StrictFile;
        c = name[pos];
        if (c == '-')
        {
          nop2.MarkMode = NWildcard::kMark_FileOrDir;
          pos++;
        }
        else if (c == '2')
        {
          nop2.MarkMode = NWildcard::kMark_StrictFile_IfWildcard;
          pos++;
        }
      }
      else
        break;
    }
    
    if (error)
    {
      errorMessage = "inorrect switch";
      break;
    }

    if (name.Len() < pos + kSomeCludeAfterRecursedPostStringMinSize)
    {
      errorMessage = "Too short switch";
      break;
    }
    
    const UString tail = name.Ptr(pos + 1);
   
    const wchar_t c = name[pos];

    if (c == kImmediateNameID)
      AddNameToCensor(censor, nop2, tail);
    else if (c == kFileListID)
      AddToCensorFromListFile(NULL, censor, nop2, tail, codePage);
    #ifdef _WIN32
    else if (c == kMapNameID)
    {
      errorMessage = ParseMapWithPaths(censor, tail, nop2);
      if (errorMessage)
        break;
    }
    #endif
    else
    {
      errorMessage = "Incorrect wildcard type marker";
      break;
    }
  }

  if (i != strings.Size())
    throw CArcCmdLineException(errorMessage, strings[i]);
}

/*
static NUpdateArchive::NPairAction::EEnum GetUpdatePairActionType(int i)
{
  switch (i)
  {
    case NUpdateArchive::NPairAction::kIgnore: return NUpdateArchive::NPairAction::kIgnore;
    case NUpdateArchive::NPairAction::kCopy: return NUpdateArchive::NPairAction::kCopy;
    case NUpdateArchive::NPairAction::kCompress: return NUpdateArchive::NPairAction::kCompress;
    case NUpdateArchive::NPairAction::kCompressAsAnti: return NUpdateArchive::NPairAction::kCompressAsAnti;
  }
  throw 98111603;
}
*/

static const char * const kUpdatePairStateIDSet = "pqrxyzw";
static const int kUpdatePairStateNotSupportedActions[] = {2, 2, 1, -1, -1, -1, -1};

static const unsigned kNumUpdatePairActions = 4;
static const char * const kUpdateIgnoreItselfPostStringID = "-";
static const wchar_t kUpdateNewArchivePostCharID = '!';


static bool ParseUpdateCommandString2(const UString &command,
    NUpdateArchive::CActionSet &actionSet, UString &postString)
{
  for (unsigned i = 0; i < command.Len();)
  {
    wchar_t c = MyCharLower_Ascii(command[i]);
    int statePos = FindCharPosInString(kUpdatePairStateIDSet, (char)c);
    if (c > 0x7F || statePos < 0)
    {
      postString = command.Ptr(i);
      return true;
    }
    i++;
    if (i >= command.Len())
      return false;
    c = command[i];
    if (c < '0' || c >= (wchar_t)('0' + kNumUpdatePairActions))
      return false;
    unsigned actionPos = (unsigned)(c - '0');
    actionSet.StateActions[(unsigned)statePos] = (NUpdateArchive::NPairAction::EEnum)(actionPos);
    if (kUpdatePairStateNotSupportedActions[(unsigned)statePos] == (int)actionPos)
      return false;
    i++;
  }
  postString.Empty();
  return true;
}

static void ParseUpdateCommandString(CUpdateOptions &options,
    const UStringVector &updatePostStrings,
    const NUpdateArchive::CActionSet &defaultActionSet)
{
  const char *errorMessage = "incorrect update switch command";
  unsigned i;
  for (i = 0; i < updatePostStrings.Size(); i++)
  {
    const UString &updateString = updatePostStrings[i];
    if (updateString.IsEqualTo(kUpdateIgnoreItselfPostStringID))
    {
      if (options.UpdateArchiveItself)
      {
        options.UpdateArchiveItself = false;
        options.Commands.Delete(0);
      }
    }
    else
    {
      NUpdateArchive::CActionSet actionSet = defaultActionSet;

      UString postString;
      if (!ParseUpdateCommandString2(updateString, actionSet, postString))
        break;
      if (postString.IsEmpty())
      {
        if (options.UpdateArchiveItself)
          options.Commands[0].ActionSet = actionSet;
      }
      else
      {
        if (postString[0] != kUpdateNewArchivePostCharID)
          break;
        CUpdateArchiveCommand uc;
        UString archivePath = postString.Ptr(1);
        if (archivePath.IsEmpty())
          break;
        uc.UserArchivePath = archivePath;
        uc.ActionSet = actionSet;
        options.Commands.Add(uc);
      }
    }
  }
  if (i != updatePostStrings.Size())
    throw CArcCmdLineException(errorMessage, updatePostStrings[i]);
}

bool ParseComplexSize(const wchar_t *s, UInt64 &result);

static void SetAddCommandOptions(
    NCommandType::EEnum commandType,
    const CParser &parser,
    CUpdateOptions &options)
{
  NUpdateArchive::CActionSet defaultActionSet;
  switch ((int)commandType)
  {
    case NCommandType::kAdd:
      defaultActionSet = NUpdateArchive::k_ActionSet_Add;
      break;
    case NCommandType::kDelete:
      defaultActionSet = NUpdateArchive::k_ActionSet_Delete;
      break;
    default:
      defaultActionSet = NUpdateArchive::k_ActionSet_Update;
  }
  
  options.UpdateArchiveItself = true;
  
  options.Commands.Clear();
  CUpdateArchiveCommand updateMainCommand;
  updateMainCommand.ActionSet = defaultActionSet;
  options.Commands.Add(updateMainCommand);
  if (parser[NKey::kUpdate].ThereIs)
    ParseUpdateCommandString(options, parser[NKey::kUpdate].PostStrings,
        defaultActionSet);
  if (parser[NKey::kWorkingDir].ThereIs)
  {
    const UString &postString = parser[NKey::kWorkingDir].PostStrings[0];
    if (postString.IsEmpty())
      NDir::MyGetTempPath(options.WorkingDir);
    else
      options.WorkingDir = us2fs(postString);
  }
  options.SfxMode = parser[NKey::kSfx].ThereIs;
  if (options.SfxMode)
    options.SfxModule = us2fs(parser[NKey::kSfx].PostStrings[0]);

  if (parser[NKey::kVolume].ThereIs)
  {
    const UStringVector &sv = parser[NKey::kVolume].PostStrings;
    FOR_VECTOR (i, sv)
    {
      UInt64 size;
      if (!ParseComplexSize(sv[i], size))
        throw CArcCmdLineException("Incorrect volume size:", sv[i]);
      if (i == sv.Size() - 1 && size == 0)
        throw CArcCmdLineException("zero size last volume is not allowed");
      options.VolumesSizes.Add(size);
    }
  }
}

static void SetMethodOptions(const CParser &parser, CObjectVector<CProperty> &properties)
{
  if (parser[NKey::kProperty].ThereIs)
  {
    FOR_VECTOR (i, parser[NKey::kProperty].PostStrings)
    {
      CProperty prop;
      prop.Name = parser[NKey::kProperty].PostStrings[i];
      int index = prop.Name.Find(L'=');
      if (index >= 0)
      {
        prop.Value = prop.Name.Ptr((unsigned)(index + 1));
        prop.Name.DeleteFrom((unsigned)index);
      }
      properties.Add(prop);
    }
  }
}


static inline void SetStreamMode(const CSwitchResult &sw, unsigned &res)
{
  if (sw.ThereIs)
    res = (unsigned)sw.PostCharIndex;
}


#if defined(_WIN32) && !defined(UNDER_CE)
static void PrintHex(UString &s, UInt64 v)
{
  char temp[32];
  ConvertUInt64ToHex(v, temp);
  s += temp;
}
#endif


#if 0 && defined(Z7_LARGE_PAGES) && defined(__linux__)
bool Get_HugePageSize(UInt64 &pageSize);
extern "C" { extern size_t g_LargePageSize; }
#endif


void CArcCmdLineParser::Parse1(const UStringVector &commandStrings,
    CArcCmdLineOptions &options)
{
  Parse1Log.Empty();
  if (!parser.ParseStrings(kSwitchForms, Z7_ARRAY_SIZE(kSwitchForms), commandStrings))
    throw CArcCmdLineException(parser.ErrorMessage, parser.ErrorLine);

  options.IsInTerminal = MY_IS_TERMINAL(stdin);
  options.IsStdOutTerminal = MY_IS_TERMINAL(stdout);
  options.IsStdErrTerminal = MY_IS_TERMINAL(stderr);

  options.HelpMode = parser[NKey::kHelp1].ThereIs || parser[NKey::kHelp2].ThereIs  || parser[NKey::kHelp3].ThereIs;
  options.YesToAll = parser[NKey::kYes].ThereIs;

  options.StdInMode = parser[NKey::kStdIn].ThereIs;
  options.StdOutMode = parser[NKey::kStdOut].ThereIs;
  options.EnableHeaders = !parser[NKey::kDisableHeaders].ThereIs;
  if (parser[NKey::kListFields].ThereIs)
  {
    const UString &s = parser[NKey::kListFields].PostStrings[0];
    options.ListFields = GetAnsiString(s);
  }
  if (parser[NKey::kListPathSlash].ThereIs)
  {
    options.ListPathSeparatorSlash.Val = !parser[NKey::kListPathSlash].WithMinus;
    options.ListPathSeparatorSlash.Def = true;
  }
  if (parser[NKey::kListTimestampUTC].ThereIs)
    g_Timestamp_Show_UTC = !parser[NKey::kListTimestampUTC].WithMinus;
  options.TechMode = parser[NKey::kTechMode].ThereIs;
  options.ShowTime = parser[NKey::kShowTime].ThereIs;

  if (parser[NKey::kDisablePercents].ThereIs)
    options.DisablePercents = true;

  if (parser[NKey::kDisablePercents].ThereIs
      || options.StdOutMode
      || !options.IsStdOutTerminal)
    options.Number_for_Percents = k_OutStream_disabled;

  if (options.StdOutMode)
    options.Number_for_Out = k_OutStream_disabled;

  SetStreamMode(parser[NKey::kOutStream], options.Number_for_Out);
  SetStreamMode(parser[NKey::kErrStream], options.Number_for_Errors);
  SetStreamMode(parser[NKey::kPercentStream], options.Number_for_Percents);

  if (parser[NKey::kLogLevel].ThereIs)
  {
    const UString &s = parser[NKey::kLogLevel].PostStrings[0];
    if (s.IsEmpty())
      options.LogLevel = 1;
    else
    {
      UInt32 v;
      if (!StringToUInt32(s, v))
        throw CArcCmdLineException("Unsupported switch postfix -bb", s);
      options.LogLevel = (unsigned)v;
    }
  }

  if (parser[NKey::kCaseSensitive].ThereIs)
  {
    options.CaseSensitive =
    g_CaseSensitive = !parser[NKey::kCaseSensitive].WithMinus;
    options.CaseSensitive_Change = true;
  }


  #if defined(_WIN32) && !defined(UNDER_CE)
  NSecurity::EnablePrivilege_SymLink();
  #endif
  
  // options.LargePages = false;

#if defined(Z7_LARGE_PAGES)
  if (parser[NKey::kLargePages].ThereIs)
  {
    const UString &s = parser[NKey::kLargePages].PostStrings[0];
    UInt32 slp_Risk = 1;
    unsigned flags = 0;
    unsigned cmd = 0;
    size_t pageSize = 0;
    size_t threshold = 0;
    
    if (s.IsEqualTo("-"))
      slp_Risk = 0;
    else if (!s.IsEmpty())
    {
      unsigned index = 0;
      while (index < s.Len())
      {
        const bool isStart = (index == 0);
        UString s2;
        {
          const int pos = s.Find(L':', index);
          if (pos < 0)
          {
            s2 = s.Ptr(index);
            index = s.Len();
          }
          else
          {
            s2 = s.Mid(index, (unsigned)pos - index);
            index = (unsigned)pos + 1;
          }
        }
        if (s2.IsEmpty())
          continue;
        
        if (isStart)
        {
          if (StringToUInt32(s2, slp_Risk))
            continue;
        }
        else if (s2.IsPrefixedBy_Ascii_NoCase("ps"))
        {
          UInt32 ps = 0;
          if (StringToUInt32(s2.Ptr(2), ps))
          {
            if (ps < sizeof(size_t) * 8)
            {
              pageSize = (size_t)1 << ps;
              flags |= Z7_LARGE_PAGES_FLAG_DIRECT_PAGE_SIZE;
              continue;
            }
          }
        }
        else if (s2.IsPrefixedBy_Ascii_NoCase("min"))
        {
          UInt32 ps = 0;
          if (StringToUInt32(s2.Ptr(3), ps))
          {
            if (ps < sizeof(size_t) * 8)
            {
              threshold = (size_t)1 << ps;
              flags |= Z7_LARGE_PAGES_FLAG_DIRECT_THRESHOLD;
              continue;
            }
          }
        }
        else if (s2.IsEqualTo_Ascii_NoCase("failstop"))
        {
          flags |= Z7_LARGE_PAGES_FLAG_FAIL_STOP;
          continue;
        }
        else if (s2.IsEqualTo_Ascii_NoCase("nomadvise"))
        {
          cmd = Z7_LARGE_PAGES_FLAG_NO_MADVISE;
          continue;
        }
        else if (s2.IsEqualTo_Ascii_NoCase("nohuge"))
        {
          cmd = Z7_LARGE_PAGES_FLAG_NO_HUGEPAGE;
          continue;
        }
        throw CArcCmdLineException("Unsupported switch postfix for -slp", s);
      }
    }

    if (slp_Risk <=
          #if defined(_WIN32) && !defined(UNDER_CE)
            (unsigned)NSecurity::Get_LargePages_RiskLevel()
          #else
            0
          #endif
        )
      cmd = Z7_LARGE_PAGES_FLAG_NO_PAGECODE;
    if (cmd == 0)
      cmd = Z7_LARGE_PAGES_FLAG_USE_HUGEPAGE;
    flags |= cmd;

#if 0 && defined(Z7_LARGE_PAGES) && defined(__linux__)
    if ((flags & Z7_LARGE_PAGES_FLAG_DIRECT_PAGE_SIZE) == 0)
    {
      UInt64 pageSize64;
      if (g_LargePageSize
          && Get_HugePageSize(pageSize64)
          && (pageSize64 & (pageSize64 - 1)) == 0
          && pageSize64 <= (1u << 25))
      {
        pageSize = (size_t)pageSize64;
        flags |= Z7_LARGE_PAGES_FLAG_DIRECT_PAGE_SIZE;
        printf("\npageSize=0x%x\n", (unsigned)pageSize);
      }
    }
#endif
    
#if defined(Z7_LARGE_PAGES)
    z7_LargePage_Set(flags, pageSize, threshold);
    if (flags & Z7_LARGE_PAGES_FLAG_USE_HUGEPAGE)
    {
      // note: this process also can inherit that Privilege from parent process
      g_LargePagesMode = true;
      #if defined(_WIN32) && !defined(UNDER_CE)
      if (!NSecurity::EnablePrivilege_LockMemory())
      {
        g_LargePagesMode = false;
        if (flags & Z7_LARGE_PAGES_FLAG_FAIL_STOP)
          throw CSystemException(GetLastError_noZero_HRESULT());
      }
      #endif
    }
#endif
  }
#endif // Z7_LARGE_PAGES


#ifndef UNDER_CE

  if (parser[NKey::kAffinity].ThereIs)
  {
    const UString &s = parser[NKey::kAffinity].PostStrings[0];
    if (!s.IsEmpty())
    {
      AString a;
      a.SetFromWStr_if_Ascii(s);
      Parse1Log += "Set process affinity mask: ";

      bool isError = false;

#ifdef _WIN32

      UInt64 v = 0;
      {
        const char *end;
        v = ConvertHexStringToUInt64(a, &end);
        if (*end != 0)
          a.Empty();
      }
      if (a.IsEmpty())
        isError = true;
      else
      {
#ifndef _WIN64
        if (v >= ((UInt64)1 << 32))
          throw CArcCmdLineException("unsupported value -stm", s);
        else
#endif
        {
          PrintHex(Parse1Log, v);
          if (!SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)v))
          {
            const DWORD lastError = GetLastError();
            Parse1Log += " : ERROR : ";
            Parse1Log += NError::MyFormatMessage(lastError);
          }
        }
      }
      
#else // _WIN32
      
      if (a.Len() != s.Len())
        isError = true;
      else
      {
        Parse1Log += a;
        NSystem::CProcessAffinity aff;
        aff.CpuZero();
        unsigned cpu = 0;
        unsigned i = a.Len();
        while (i)
        {
          unsigned v = (Byte)a[--i];
          Z7_PARSE_HEX_DIGIT(v, { isError = true; break; })
          for (unsigned mask = 1; mask != 1u << 4; mask <<= 1, cpu++)
            if (v & mask)
              aff.CpuSet(cpu);
        }
        if (!isError)
        if (!aff.SetProcAffinity())
        {
          const DWORD lastError = GetLastError();
          Parse1Log += " : ERROR : ";
          Parse1Log += NError::MyFormatMessage(lastError);
        }
      }
#endif // _WIN32

      if (isError)
        throw CArcCmdLineException("Unsupported switch postfix -stm", s);

      Parse1Log.Add_LF();
    }
  }

#endif
}



struct CCodePagePair
{
  const char *Name;
  UInt32 CodePage;
};

static const unsigned kNumByteOnlyCodePages = 3;

static const CCodePagePair g_CodePagePairs[] =
{
  { "utf-8", CP_UTF8 },
  { "win", CP_ACP },
  { "dos", CP_OEMCP },
  { "utf-16le", Z7_WIN_CP_UTF16 },
  { "utf-16be", Z7_WIN_CP_UTF16BE }
};

static Int32 FindCharset(const NCommandLineParser::CParser &parser, unsigned keyIndex,
    bool byteOnlyCodePages, Int32 defaultVal)
{
  if (!parser[keyIndex].ThereIs)
    return defaultVal;

  UString name (parser[keyIndex].PostStrings.Back());
  UInt32 v;
  if (StringToUInt32(name, v))
    if (v < ((UInt32)1 << 16))
      return (Int32)v;
  name.MakeLower_Ascii();
  const unsigned num = byteOnlyCodePages ? kNumByteOnlyCodePages : Z7_ARRAY_SIZE(g_CodePagePairs);
  for (unsigned i = 0;; i++)
  {
    if (i == num) // to disable warnings from different compilers
      throw CArcCmdLineException("Unsupported charset:", name);
    const CCodePagePair &pair = g_CodePagePairs[i];
    if (name.IsEqualTo(pair.Name))
      return (Int32)pair.CodePage;
  }
}


static void SetBoolPair(NCommandLineParser::CParser &parser, unsigned switchID, CBoolPair &bp)
{
  bp.Def = parser[switchID].ThereIs;
  if (bp.Def)
    bp.Val = !parser[switchID].WithMinus;
}


static bool ParseSizeString(const wchar_t *s, UInt64 &res)
{
  const wchar_t *end;
  const UInt64 v = ConvertStringToUInt64(s, &end);
  if (s == end)
    return false;
  const wchar_t c = *end;

  if (c == 0)
  {
    res = v;
    return true;
  }
  if (end[1] != 0)
    return false;

  unsigned numBits;
  switch (MyCharLower_Ascii(c))
  {
    case 'b': numBits =  0; break;
    case 'k': numBits = 10; break;
    case 'm': numBits = 20; break;
    case 'g': numBits = 30; break;
    case 't': numBits = 40; break;
    default: return false;
  }
  const UInt64 val2 = v << numBits;
  if ((val2 >> numBits) != v)
    return false;
  res = val2;
  return true;
}

void CArcCmdLineParser::Parse2(CArcCmdLineOptions &options)
{
  const UStringVector &nonSwitchStrings = parser.NonSwitchStrings;
  const unsigned numNonSwitchStrings = nonSwitchStrings.Size();
  if (numNonSwitchStrings < kMinNonSwitchWords)
    throw CArcCmdLineException("The command must be specified");

  if (!ParseArchiveCommand(nonSwitchStrings[kCommandIndex], options.Command))
    throw CArcCmdLineException("Unsupported command:", nonSwitchStrings[kCommandIndex]);

  if (parser[NKey::kHash].ThereIs)
    options.HashMethods = parser[NKey::kHash].PostStrings;

  /*
  if (parser[NKey::kHashGenFile].ThereIs)
  {
    const UString &s = parser[NKey::kHashGenFile].PostStrings[0];
    for (unsigned i = 0 ; i < s.Len();)
    {
      const wchar_t c = s[i++];
      if (!options.HashOptions.ParseFlagCharOption(c, true))
      {
        if (c != '=')
          throw CArcCmdLineException("Unsupported hash mode switch:", s);
        options.HashOptions.HashFilePath = s.Ptr(i);
        break;
      }
    }
  }
  */

  if (parser[NKey::kHashDir].ThereIs)
    options.ExtractOptions.HashDir = parser[NKey::kHashDir].PostStrings[0];
  
  if (parser[NKey::kExtractMemLimit].ThereIs)
  {
    const UString &s = parser[NKey::kExtractMemLimit].PostStrings[0];
    if (!ParseSizeString(s, options.ExtractOptions.NtOptions.MemLimit))
      throw CArcCmdLineException("Unsupported -smemx:", s);
  }
  
  if (parser[NKey::kElimDup].ThereIs)
  {
    options.ExtractOptions.ElimDup.Def = true;
    options.ExtractOptions.ElimDup.Val = !parser[NKey::kElimDup].WithMinus;
  }
  
  NWildcard::ECensorPathMode censorPathMode = NWildcard::k_RelatPath;
  bool fullPathMode = parser[NKey::kFullPathMode].ThereIs;
  if (fullPathMode)
  {
    censorPathMode = NWildcard::k_AbsPath;
    const UString &s = parser[NKey::kFullPathMode].PostStrings[0];
    if (!s.IsEmpty())
    {
      if (s.IsEqualTo("2"))
        censorPathMode = NWildcard::k_FullPath;
      else
        throw CArcCmdLineException("Unsupported -spf:", s);
    }
  }

  if (parser[NKey::kNameTrailReplace].ThereIs)
    g_PathTrailReplaceMode = !parser[NKey::kNameTrailReplace].WithMinus;

  CNameOption nop;

  if (parser[NKey::kRecursed].ThereIs)
    nop.RecursedType = GetRecursedTypeFromIndex(parser[NKey::kRecursed].PostCharIndex);

  if (parser[NKey::kDisableWildcardParsing].ThereIs)
    nop.WildcardMatching = false;

  if (parser[NKey::kUseSlashMark].ThereIs)
  {
    const UString &s = parser[NKey::kUseSlashMark].PostStrings[0];
    if (s.IsEmpty())
      nop.MarkMode = NWildcard::kMark_StrictFile;
    else if (s.IsEqualTo_Ascii_NoCase("-"))
      nop.MarkMode = NWildcard::kMark_FileOrDir;
    else if (s.IsEqualTo_Ascii_NoCase("2"))
      nop.MarkMode = NWildcard::kMark_StrictFile_IfWildcard;
    else
      throw CArcCmdLineException("Unsupported -spm:", s);
  }


  options.ConsoleCodePage = FindCharset(parser, NKey::kConsoleCharSet, true, -1);

  const UInt32 codePage = (UInt32)FindCharset(parser, NKey::kListfileCharSet, false, CP_UTF8);

  bool thereAreSwitchIncludes = false;

  if (parser[NKey::kInclude].ThereIs)
  {
    thereAreSwitchIncludes = true;
    nop.Include = true;
    AddSwitchWildcardsToCensor(options.Censor,
        parser[NKey::kInclude].PostStrings, nop, codePage);
  }

  if (parser[NKey::kExclude].ThereIs)
  {
    nop.Include = false;
    AddSwitchWildcardsToCensor(options.Censor,
        parser[NKey::kExclude].PostStrings, nop, codePage);
  }
 
  unsigned curCommandIndex = kCommandIndex + 1;
  bool thereIsArchiveName = !parser[NKey::kNoArName].ThereIs &&
      options.Command.CommandType != NCommandType::kBenchmark &&
      options.Command.CommandType != NCommandType::kInfo &&
      options.Command.CommandType != NCommandType::kHash;

  const bool isExtractGroupCommand = options.Command.IsFromExtractGroup();
  const bool isExtractOrList = isExtractGroupCommand || options.Command.CommandType == NCommandType::kList;
  const bool isRename = options.Command.CommandType == NCommandType::kRename;
  options.UpdateOptions.RenameMode = isRename;

  if ((isExtractOrList || isRename) && options.StdInMode)
    thereIsArchiveName = false;

  if (parser[NKey::kArcNameMode].ThereIs)
    options.UpdateOptions.ArcNameMode = ParseArcNameMode(parser[NKey::kArcNameMode].PostCharIndex);

  if (thereIsArchiveName)
  {
    if (curCommandIndex >= numNonSwitchStrings)
      throw CArcCmdLineException("Cannot find archive name");
    options.ArchiveName = nonSwitchStrings[curCommandIndex++];
    if (options.ArchiveName.IsEmpty())
      throw CArcCmdLineException("Archive name cannot by empty");
    #ifdef _WIN32
    // options.ArchiveName.Replace(L'/', WCHAR_PATH_SEPARATOR);
    #endif
  }

  nop.Include = true;
  AddToCensorFromNonSwitchesStrings(isRename ? &options.UpdateOptions.RenamePairs : NULL,
      curCommandIndex, options.Censor,
      nonSwitchStrings, parser.StopSwitchIndex,
      nop,
      thereAreSwitchIncludes, codePage);

  #ifndef Z7_NO_CRYPTO
  options.PasswordEnabled = parser[NKey::kPassword].ThereIs;
  if (options.PasswordEnabled)
    options.Password = parser[NKey::kPassword].PostStrings[0];
  #endif

  options.ShowDialog = parser[NKey::kShowDialog].ThereIs;

  if (parser[NKey::kArchiveType].ThereIs)
    options.ArcType = parser[NKey::kArchiveType].PostStrings[0];

  options.ExcludedArcTypes = parser[NKey::kExcludedArcType].PostStrings;

  SetMethodOptions(parser, options.Properties);

  if (parser[NKey::kNtSecurity].ThereIs) options.NtSecurity.SetTrueTrue();

  SetBoolPair(parser, NKey::kAltStreams, options.AltStreams);
  SetBoolPair(parser, NKey::kHardLinks, options.HardLinks);
  SetBoolPair(parser, NKey::kSymLinks, options.SymLinks);
  
  SetBoolPair(parser, NKey::kStoreOwnerId, options.StoreOwnerId);
  SetBoolPair(parser, NKey::kStoreOwnerName, options.StoreOwnerName);
  /*
  bool supportSymLink = options.SymLinks.Val;
  if (!options.SymLinks.Def)
  {
    if (isExtractOrList)
      supportSymLink = true;
    else
      supportSymLink = false;
  }
  #ifdef ENV_HAVE_LSTAT
  if (supportSymLink)
    global_use_lstat = 1;
  else
    global_use_lstat = 0;
  #endif
  */

  if (isExtractOrList)
  {
    CExtractOptionsBase &eo = options.ExtractOptions;

    eo.ExcludeDirItems = options.Censor.ExcludeDirItems;
    eo.ExcludeFileItems = options.Censor.ExcludeFileItems;

    {
      CExtractNtOptions &nt = eo.NtOptions;
      nt.NtSecurity = options.NtSecurity;

      nt.AltStreams = options.AltStreams;
      if (!options.AltStreams.Def)
        nt.AltStreams.Val = true;

      nt.HardLinks = options.HardLinks;
      if (!options.HardLinks.Def)
        nt.HardLinks.Val = true;

      nt.SymLinks = options.SymLinks;
      if (!options.SymLinks.Def)
        nt.SymLinks.Val = true;

      if (parser[NKey::kSymLinks_AllowDangerous].ThereIs)
      {
        const UString &s = parser[NKey::kSymLinks_AllowDangerous].PostStrings[0];
        UInt32 v = 9; // default value for "-snld" instead of default = 5 without "-snld".
        if (!s.IsEmpty())
          if (!StringToUInt32(s, v))
            throw CArcCmdLineException("Unsupported switch postfix -snld", s);
        nt.SymLinks_DangerousLevel = (unsigned)v;
      }

      nt.ReplaceColonForAltStream = parser[NKey::kReplaceColonForAltStream].ThereIs;
      nt.WriteToAltStreamIfColon = parser[NKey::kWriteToAltStreamIfColon].ThereIs;

      nt.ExtractOwner = options.StoreOwnerId.Val; // StoreOwnerName

      if (parser[NKey::kPreserveATime].ThereIs)
        nt.PreserveATime = true;
      if (parser[NKey::kShareForWrite].ThereIs)
        nt.OpenShareForWrite = true;
    }

    if (parser[NKey::kZoneFile].ThereIs)
    {
      eo.ZoneMode = NExtract::NZoneIdMode::kAll;
      const UString &s = parser[NKey::kZoneFile].PostStrings[0];
      if (!s.IsEmpty())
      {
             if (s.IsEqualTo("0")) eo.ZoneMode = NExtract::NZoneIdMode::kNone;
        else if (s.IsEqualTo("1")) eo.ZoneMode = NExtract::NZoneIdMode::kAll;
        else if (s.IsEqualTo("2")) eo.ZoneMode = NExtract::NZoneIdMode::kOffice;
        else
          throw CArcCmdLineException("Unsupported -snz:", s);
      }
    }
    
    options.Censor.AddPathsToCensor(NWildcard::k_AbsPath);
    options.Censor.ExtendExclude();

    // are there paths that look as non-relative (!Prefix.IsEmpty())
    if (!options.Censor.AllAreRelative())
      throw CArcCmdLineException("Cannot use absolute pathnames for this command");

    NWildcard::CCensor &arcCensor = options.arcCensor;

    CNameOption nopArc;
    // nopArc.RecursedType = NRecursedType::kNonRecursed; // default:  we don't want recursing for archives, if -r specified
    // is it OK, external switches can disable WildcardMatching and MarcMode for arc.
    nopArc.WildcardMatching = nop.WildcardMatching;
    nopArc.MarkMode = nop.MarkMode;

    if (parser[NKey::kArInclude].ThereIs)
    {
      nopArc.Include = true;
      AddSwitchWildcardsToCensor(arcCensor, parser[NKey::kArInclude].PostStrings, nopArc, codePage);
    }
    if (parser[NKey::kArExclude].ThereIs)
    {
      nopArc.Include = false;
      AddSwitchWildcardsToCensor(arcCensor, parser[NKey::kArExclude].PostStrings, nopArc, codePage);
    }

    if (thereIsArchiveName)
    {
      nopArc.Include = true;
      AddNameToCensor(arcCensor, nopArc, options.ArchiveName);
    }

    arcCensor.AddPathsToCensor(NWildcard::k_RelatPath);

    #ifdef _WIN32
    ConvertToLongNames(arcCensor);
    #endif

    arcCensor.ExtendExclude();

    if (options.StdInMode)
      options.ArcName_for_StdInMode = parser[NKey::kStdIn].PostStrings.Front();
    
    if (isExtractGroupCommand)
    {
      if (options.StdOutMode)
      {
        if (
                  options.Number_for_Percents == k_OutStream_stdout
            // || options.Number_for_Out      == k_OutStream_stdout
            // || options.Number_for_Errors   == k_OutStream_stdout
            ||
            (
              (options.IsStdOutTerminal && options.IsStdErrTerminal)
              &&
              (
                      options.Number_for_Percents != k_OutStream_disabled
                // || options.Number_for_Out      != k_OutStream_disabled
                // || options.Number_for_Errors   != k_OutStream_disabled
              )
            )
           )
          throw CArcCmdLineException(kSameTerminalError);
      }
      
      if (parser[NKey::kOutputDir].ThereIs)
      {
        eo.OutputDir = us2fs(parser[NKey::kOutputDir].PostStrings[0]);
        #ifdef _WIN32
          NFile::NName::NormalizeDirSeparators(eo.OutputDir);
        #endif
        NFile::NName::NormalizeDirPathPrefix(eo.OutputDir);
      }
      if (parser[NKey::kOutDirMode].ThereIs)
      {
        const int index = parser[NKey::kOutDirMode].PostCharIndex;
        eo.OutDirMode =
          (index == 0) ? NExtractOutDirMode::k_Direct :
          (index == 1) ? NExtractOutDirMode::k_AddArcName :
                         NExtractOutDirMode::k_ReplaceAsterisk;
      }

      eo.OverwriteMode = NExtract::NOverwriteMode::kAsk;
      if (parser[NKey::kOverwrite].ThereIs)
      {
        eo.OverwriteMode = k_OverwriteModes[(unsigned)parser[NKey::kOverwrite].PostCharIndex];
        eo.OverwriteMode_Force = true;
      }
      else if (options.YesToAll)
      {
        eo.OverwriteMode = NExtract::NOverwriteMode::kOverwrite;
        eo.OverwriteMode_Force = true;
      }
    }

    eo.PathMode = options.Command.GetPathMode();
    if (censorPathMode == NWildcard::k_AbsPath)
    {
      eo.PathMode = NExtract::NPathMode::kAbsPaths;
      eo.PathMode_Force = true;
    }
    else if (censorPathMode == NWildcard::k_FullPath)
    {
      eo.PathMode = NExtract::NPathMode::kFullPaths;
      eo.PathMode_Force = true;
    }
  }
  else if (options.Command.IsFromUpdateGroup())
  {
    if (parser[NKey::kArInclude].ThereIs)
      throw CArcCmdLineException("-ai switch is not supported for this command");

    CUpdateOptions &updateOptions = options.UpdateOptions;

    SetAddCommandOptions(options.Command.CommandType, parser, updateOptions);
    
    updateOptions.MethodMode.Properties = options.Properties;

    if (parser[NKey::kPreserveATime].ThereIs)
      updateOptions.PreserveATime = true;
    if (parser[NKey::kShareForWrite].ThereIs)
      updateOptions.OpenShareForWrite = true;
    if (parser[NKey::kStopAfterOpenError].ThereIs)
      updateOptions.StopAfterOpenError = true;

    updateOptions.PathMode = censorPathMode;

    updateOptions.AltStreams = options.AltStreams;
    updateOptions.NtSecurity = options.NtSecurity;
    updateOptions.HardLinks = options.HardLinks;
    updateOptions.SymLinks = options.SymLinks;
    
    updateOptions.StoreOwnerId = options.StoreOwnerId;
    updateOptions.StoreOwnerName = options.StoreOwnerName;

    updateOptions.EMailMode = parser[NKey::kEmail].ThereIs;
    if (updateOptions.EMailMode)
    {
      updateOptions.EMailAddress = parser[NKey::kEmail].PostStrings.Front();
      if (updateOptions.EMailAddress.Len() > 0)
        if (updateOptions.EMailAddress[0] == L'.')
        {
          updateOptions.EMailRemoveAfter = true;
          updateOptions.EMailAddress.Delete(0);
        }
    }

    updateOptions.StdOutMode = options.StdOutMode;
    updateOptions.StdInMode = options.StdInMode;

    updateOptions.DeleteAfterCompressing = parser[NKey::kDeleteAfterCompressing].ThereIs;
    updateOptions.SetArcMTime = parser[NKey::kSetArcMTime].ThereIs;

    if (updateOptions.StdOutMode && updateOptions.EMailMode)
      throw CArcCmdLineException("stdout mode and email mode cannot be combined");
    
    if (updateOptions.StdOutMode)
    {
      if (options.IsStdOutTerminal)
        throw CArcCmdLineException(kTerminalOutError);
      
      if (options.Number_for_Percents == k_OutStream_stdout
          || options.Number_for_Out == k_OutStream_stdout
          || options.Number_for_Errors == k_OutStream_stdout)
        throw CArcCmdLineException(kSameTerminalError);
    }
    
    if (updateOptions.StdInMode)
      updateOptions.StdInFileName = parser[NKey::kStdIn].PostStrings.Front();

    if (options.Command.CommandType == NCommandType::kRename)
      if (updateOptions.Commands.Size() != 1)
        throw CArcCmdLineException("Only one archive can be created with rename command");
  }
  else if (options.Command.CommandType == NCommandType::kBenchmark)
  {
    options.NumIterations = 1;
    options.NumIterations_Defined = false;
    if (curCommandIndex < numNonSwitchStrings)
    {
      if (!StringToUInt32(nonSwitchStrings[curCommandIndex], options.NumIterations))
        throw CArcCmdLineException("Incorrect number of benchmark iterations", nonSwitchStrings[curCommandIndex]);
      curCommandIndex++;
      options.NumIterations_Defined = true;
    }
  }
  else if (options.Command.CommandType == NCommandType::kHash)
  {
    options.Censor.AddPathsToCensor(censorPathMode);
    options.Censor.ExtendExclude();

    CHashOptions &hashOptions = options.HashOptions;
    hashOptions.PathMode = censorPathMode;
    hashOptions.Methods = options.HashMethods;
    // hashOptions.HashFilePath = options.HashFilePath;
    if (parser[NKey::kPreserveATime].ThereIs)
      hashOptions.PreserveATime = true;
    if (parser[NKey::kShareForWrite].ThereIs)
      hashOptions.OpenShareForWrite = true;
    hashOptions.StdInMode = options.StdInMode;
    hashOptions.AltStreamsMode = options.AltStreams.Val;
    hashOptions.SymLinks = options.SymLinks;
  }
  else if (options.Command.CommandType == NCommandType::kInfo)
  {
  }
  else
    throw 20150919;
}



#ifndef _WIN32

static AString g_ModuleDirPrefix;

void Set_ModuleDirPrefix_From_ProgArg0(const char *s);
void Set_ModuleDirPrefix_From_ProgArg0(const char *s)
{
  AString a (s);
  int sep = a.ReverseFind_PathSepar();
  a.DeleteFrom((unsigned)(sep + 1));
  g_ModuleDirPrefix = a;
}

namespace NWindows {
namespace NDLL {

FString GetModuleDirPrefix();
FString GetModuleDirPrefix()
{
  FString s;

  s = fas2fs(g_ModuleDirPrefix);
  if (s.IsEmpty())
    s = FTEXT(".") FSTRING_PATH_SEPARATOR;
  return s;
  /*
  setenv("_7ZIP_HOME_DIR", "/test/", 0);
  const char *home = getenv("_7ZIP_HOME_DIR");
  if (home)
    s = home;
  else
    s = FTEXT(".") FSTRING_PATH_SEPARATOR;
  return s;
  */
}

}}

#endif // ! _WIN32
