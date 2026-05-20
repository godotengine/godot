// Main.cpp

#include "StdAfx.h"

#include "../../../../C/CpuArch.h"
#include "../../../../C/DllSecur.h"

#include "../../../Common/MyWindows.h"
#include "../../../Common/MyInitGuid.h"

#include "../../../Common/CommandLineParser.h"
#include "../../../Common/MyException.h"

#ifdef _WIN32
#include "../../../Windows/DLL.h"
#else
#include "../../../Common/StringConvert.h"
#endif
#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileName.h"

#include "../../UI/Common/ExitCode.h"
#include "../../UI/Common/Extract.h"

#include "../../UI/Console/ExtractCallbackConsole.h"
#include "../../UI/Console/List.h"
#include "../../UI/Console/OpenCallbackConsole.h"

#include "../../MyVersion.h"


using namespace NWindows;
using namespace NFile;
using namespace NDir;
using namespace NCommandLineParser;

#ifdef _WIN32
extern
HINSTANCE g_hInstance;
HINSTANCE g_hInstance = NULL;
#endif
extern
int g_CodePage;
int g_CodePage = -1;
extern CStdOutStream *g_StdStream;

static const char * const kCopyrightString =
"\n7-Zip SFX " MY_VERSION_CPU " : " MY_COPYRIGHT_DATE "\n";

static const int kNumSwitches = 6;

namespace NKey {
enum Enum
{
  kHelp1 = 0,
  kHelp2,
  kDisablePercents,
  kYes,
  kPassword,
  kOutputDir
};

}

namespace NRecursedType {
enum EEnum
{
  kRecursed,
  kWildcardOnlyRecursed,
  kNonRecursed
};
}
/*
static const char kRecursedIDChar = 'R';

namespace NRecursedPostCharIndex {
  enum EEnum
  {
    kWildcardRecursionOnly = 0,
    kNoRecursion = 1
  };
}

static const char kFileListID = '@';
static const char kImmediateNameID = '!';

static const char kSomeCludePostStringMinSize = 2; // at least <@|!><N>ame must be
static const char kSomeCludeAfterRecursedPostStringMinSize = 2; // at least <@|!><N>ame must be
*/

#define SWFRM_3(t, mu, mi) t, mu, mi, NULL
#define SWFRM_1(t)     SWFRM_3(t, false, 0)
#define SWFRM_SIMPLE   SWFRM_1(NSwitchType::kSimple)
#define SWFRM_STRING_SINGL(mi) SWFRM_3(NSwitchType::kString, false, mi)

static const CSwitchForm kSwitchForms[kNumSwitches] =
{
  { "?",  SWFRM_SIMPLE },
  { "H",  SWFRM_SIMPLE },
  { "BD", SWFRM_SIMPLE },
  { "Y",  SWFRM_SIMPLE },
  { "P",  SWFRM_STRING_SINGL(1) },
  { "O",  SWFRM_STRING_SINGL(1) },
};

static const int kNumCommandForms = 3;

static const NRecursedType::EEnum kCommandRecursedDefault[kNumCommandForms] =
{
  NRecursedType::kRecursed
};

// static const bool kTestExtractRecursedDefault = true;
// static const bool kAddRecursedDefault = false;

static const char * const kUniversalWildcard = "*";

static const char * const kHelpString =
    "\nUsage: 7zSFX [<command>] [<switches>...] [<file_name>...]\n"
    "\n"
    "<Commands>\n"
    // "  l: List contents of archive\n"
    "  t: Test integrity of archive\n"
    "  x: eXtract files with full pathname (default)\n"
    "<Switches>\n"
    // "  -bd Disable percentage indicator\n"
    "  -o{Directory}: set Output directory\n"
    "  -p{Password}: set Password\n"
    "  -y: assume Yes on all queries\n";


// ---------------------------
// exception messages

static const char * const kUserErrorMessage  = "Incorrect command line"; // NExitCode::kUserError
// static const char * const kIncorrectListFile = "Incorrect wildcard in listfile";
static const char * const kIncorrectWildcardInCommandLine  = "Incorrect wildcard in command line";

// static const CSysString kFileIsNotArchiveMessageBefore = "File \"";
// static const CSysString kFileIsNotArchiveMessageAfter = "\" is not archive";

// static const char * const kProcessArchiveMessage = " archive: ";

static const char * const kCantFindSFX = " cannot find sfx";

namespace NCommandType
{
  enum EEnum
  {
    kTest = 0,
    kFullExtract,
    kList
  };
}

static const char * const g_Commands = "txl";

struct CArchiveCommand
{
  NCommandType::EEnum CommandType;

  NRecursedType::EEnum DefaultRecursedType() const;
};

static bool ParseArchiveCommand(const UString &commandString, CArchiveCommand &command)
{
  UString s = commandString;
  s.MakeLower_Ascii();
  if (s.Len() != 1)
    return false;
  if (s[0] >= 0x80)
    return false;
  int index = FindCharPosInString(g_Commands, (char)s[0]);
  if (index < 0)
    return false;
  command.CommandType = (NCommandType::EEnum)index;
  return true;
}

NRecursedType::EEnum CArchiveCommand::DefaultRecursedType() const
{
  return kCommandRecursedDefault[CommandType];
}

static void PrintHelp(void)
{
  g_StdOut << kHelpString;
}

Z7_ATTR_NORETURN
static void ShowMessageAndThrowException(const char *message, NExitCode::EEnum code)
{
  g_StdOut << message << endl;
  throw code;
}

Z7_ATTR_NORETURN
static void PrintHelpAndExit() // yyy
{
  PrintHelp();
  ShowMessageAndThrowException(kUserErrorMessage, NExitCode::kUserError);
}

// ------------------------------------------------------------------
// filenames functions

static bool AddNameToCensor(NWildcard::CCensor &wildcardCensor,
    const UString &name, bool include, NRecursedType::EEnum type)
{
  /*
  if (!IsWildcardFilePathLegal(name))
    return false;
  */
  const bool isWildcard = DoesNameContainWildcard(name);
  bool recursed = false;

  switch (type)
  {
    case NRecursedType::kWildcardOnlyRecursed:
      recursed = isWildcard;
      break;
    case NRecursedType::kRecursed:
      recursed = true;
      break;
    case NRecursedType::kNonRecursed:
      recursed = false;
      break;
  }

  NWildcard::CCensorPathProps props;
  props.Recursive = recursed;
  wildcardCensor.AddPreItem(include, name, props);
  return true;
}

static void AddCommandLineWildcardToCensor(NWildcard::CCensor &wildcardCensor,
    const UString &name, bool include, NRecursedType::EEnum type)
{
  if (!AddNameToCensor(wildcardCensor, name, include, type))
    ShowMessageAndThrowException(kIncorrectWildcardInCommandLine, NExitCode::kUserError);
}


#ifndef _WIN32
static void GetArguments(int numArgs, char *args[], UStringVector &parts)
{
  parts.Clear();
  for (int i = 0; i < numArgs; i++)
  {
    UString s = MultiByteToUnicodeString(args[i]);
    parts.Add(s);
  }
}
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
  #ifdef _WIN32
  // do we need load Security DLLs for console program?
  LoadSecurityDlls();
  #endif

  #if defined(_WIN32) && !defined(UNDER_CE)
  SetFileApisToOEM();
  #endif
  
  #ifdef ENV_HAVE_LOCALE
  MY_SetLocale();
  #endif

  g_StdOut << kCopyrightString;

  UStringVector commandStrings;
  #ifdef _WIN32
  NCommandLineParser::SplitCommandLine(GetCommandLineW(), commandStrings);
  #else
  GetArguments(numArgs, args, commandStrings);
  #endif

  #ifdef _WIN32
  
  FString arcPath;
  {
    FString path;
    NDLL::MyGetModuleFileName(path);
    if (!MyGetFullPathName(path, arcPath))
    {
      g_StdOut << "GetFullPathName Error";
      return NExitCode::kFatalError;
    }
  }

  #else

  if (commandStrings.IsEmpty())
    return NExitCode::kFatalError;

  const FString arcPath = us2fs(commandStrings.Front());

  #endif

  #ifndef UNDER_CE
  if (commandStrings.Size() > 0)
    commandStrings.Delete(0);
  #endif

  NCommandLineParser::CParser parser;
  
  try
  {
    if (!parser.ParseStrings(kSwitchForms, kNumSwitches, commandStrings))
    {
      g_StdOut << "Command line error:" << endl
          << parser.ErrorMessage << endl
          << parser.ErrorLine << endl;
      return NExitCode::kUserError;
    }
  }
  catch(...)
  {
    PrintHelpAndExit();
  }

  if (parser[NKey::kHelp1].ThereIs || parser[NKey::kHelp2].ThereIs)
  {
    PrintHelp();
    return 0;
  }
  
  const UStringVector &nonSwitchStrings = parser.NonSwitchStrings;

  unsigned curCommandIndex = 0;

  CArchiveCommand command;
  if (nonSwitchStrings.IsEmpty())
    command.CommandType = NCommandType::kFullExtract;
  else
  {
    const UString &cmd = nonSwitchStrings[curCommandIndex];
    if (!ParseArchiveCommand(cmd, command))
    {
      g_StdOut << "ERROR: Unknown command:" << endl << cmd << endl;
      return NExitCode::kUserError;
    }
    curCommandIndex = 1;
  }


  NRecursedType::EEnum recursedType;
  recursedType = command.DefaultRecursedType();

  NWildcard::CCensor wildcardCensor;
  
  {
    if (nonSwitchStrings.Size() == curCommandIndex)
      AddCommandLineWildcardToCensor(wildcardCensor, (UString)kUniversalWildcard, true, recursedType);
    for (; curCommandIndex < nonSwitchStrings.Size(); curCommandIndex++)
    {
      const UString &s = nonSwitchStrings[curCommandIndex];
      if (s.IsEmpty())
        throw "Empty file path";
      AddCommandLineWildcardToCensor(wildcardCensor, s, true, recursedType);
    }
  }

  const bool yesToAll = parser[NKey::kYes].ThereIs;

  // NExtractMode::EEnum extractMode;
  // bool isExtractGroupCommand = command.IsFromExtractGroup(extractMode);

  const bool passwordEnabled = parser[NKey::kPassword].ThereIs;

  UString password;
  if (passwordEnabled)
    password = parser[NKey::kPassword].PostStrings[0];

  if (!NFind::DoesFileExist_FollowLink(arcPath))
    throw kCantFindSFX;
  
  FString outputDir;
  if (parser[NKey::kOutputDir].ThereIs)
  {
    outputDir = us2fs(parser[NKey::kOutputDir].PostStrings[0]);
    NName::NormalizeDirPathPrefix(outputDir);
  }


  wildcardCensor.AddPathsToCensor(NWildcard::k_RelatPath);
  
  {
    UStringVector v1, v2;
    v1.Add(fs2us(arcPath));
    v2.Add(fs2us(arcPath));
    const NWildcard::CCensorNode &wildcardCensorHead =
      wildcardCensor.Pairs.Front().Head;

    CCodecs *codecs = new CCodecs;
    CMyComPtr<
      #ifdef Z7_EXTERNAL_CODECS
      ICompressCodecsInfo
      #else
      IUnknown
      #endif
      > compressCodecsInfo = codecs;
    {
      HRESULT result = codecs->Load();
      if (result != S_OK)
        throw CSystemException(result);
    }

    if (command.CommandType != NCommandType::kList)
    {
      CExtractCallbackConsole *ecs = new CExtractCallbackConsole;
      CMyComPtr<IFolderArchiveExtractCallback> extractCallback = ecs;
      ecs->Init(g_StdStream, &g_StdErr, g_StdStream, false);

      #ifndef Z7_NO_CRYPTO
      ecs->PasswordIsDefined = passwordEnabled;
      ecs->Password = password;
      #endif

      /*
      COpenCallbackConsole openCallback;
      openCallback.Init(g_StdStream, g_StdStream);

      #ifndef Z7_NO_CRYPTO
      openCallback.PasswordIsDefined = passwordEnabled;
      openCallback.Password = password;
      #endif
      */

      CExtractOptions eo;
      eo.StdOutMode = false;
      eo.YesToAll = yesToAll;
      eo.TestMode = command.CommandType == NCommandType::kTest;
      eo.PathMode = NExtract::NPathMode::kFullPaths;
      eo.OverwriteMode = yesToAll ?
          NExtract::NOverwriteMode::kOverwrite :
          NExtract::NOverwriteMode::kAsk;
      eo.OutputDir = outputDir;

      UString errorMessage;
      CDecompressStat stat;
      HRESULT result = Extract(
          codecs, CObjectVector<COpenType>(), CIntVector(),
          v1, v2,
          wildcardCensorHead,
          eo,
          ecs, ecs, ecs,
          // NULL, // hash
          errorMessage, stat);

      ecs->ClosePercents();

      if (!errorMessage.IsEmpty())
      {
        (*g_StdStream) << endl << "Error: " << errorMessage;
        if (result == S_OK)
          result = E_FAIL;
      }

      if (   0 != ecs->NumCantOpenArcs
          || 0 != ecs->NumArcsWithError
          || 0 != ecs->NumFileErrors
          || 0 != ecs->NumOpenArcErrors)
      {
        if (ecs->NumCantOpenArcs != 0)
          (*g_StdStream) << endl << "Can't open as archive" << endl;
        if (ecs->NumArcsWithError != 0)
          (*g_StdStream) << endl << "Archive Errors" << endl;
        if (ecs->NumFileErrors != 0)
          (*g_StdStream) << endl << "Sub items Errors: " << ecs->NumFileErrors << endl;
        if (ecs->NumOpenArcErrors != 0)
          (*g_StdStream) << endl << "Open Errors: " << ecs->NumOpenArcErrors << endl;
        return NExitCode::kFatalError;
      }
      if (result != S_OK)
        throw CSystemException(result);
    }
    else
    {
      throw CSystemException(E_NOTIMPL);

      /*
      UInt64 numErrors = 0;
      UInt64 numWarnings = 0;
      HRESULT result = ListArchives(
          codecs, CObjectVector<COpenType>(), CIntVector(),
          false, // stdInMode
          v1, v2,
          true, // processAltStreams
          false, // showAltStreams
          wildcardCensorHead,
          true, // enableHeaders
          false, // techMode
          #ifndef Z7_NO_CRYPTO
          passwordEnabled, password,
          #endif
          numErrors, numWarnings);
      if (numErrors > 0)
      {
        g_StdOut << endl << "Errors: " << numErrors;
        return NExitCode::kFatalError;
      }
      if (result != S_OK)
        throw CSystemException(result);
      */
    }
  }
  return 0;
}
