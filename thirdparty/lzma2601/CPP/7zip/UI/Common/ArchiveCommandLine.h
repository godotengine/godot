// ArchiveCommandLine.h

#ifndef ZIP7_INC_ARCHIVE_COMMAND_LINE_H
#define ZIP7_INC_ARCHIVE_COMMAND_LINE_H

#include "../../../Common/CommandLineParser.h"
#include "../../../Common/Wildcard.h"

#include "EnumDirItems.h"

#include "Extract.h"
#include "HashCalc.h"
#include "Update.h"

typedef CMessagePathException CArcCmdLineException;

namespace NCommandType { enum EEnum
{
  kAdd = 0,
  kUpdate,
  kDelete,
  kTest,
  kExtract,
  kExtractFull,
  kList,
  kBenchmark,
  kInfo,
  kHash,
  kRename
};}

struct CArcCommand
{
  NCommandType::EEnum CommandType;

  bool IsFromExtractGroup() const;
  bool IsFromUpdateGroup() const;
  bool IsTestCommand() const { return CommandType == NCommandType::kTest; }
  NExtract::NPathMode::EEnum GetPathMode() const;
};

enum
{
  k_OutStream_disabled = 0,
  k_OutStream_stdout = 1,
  k_OutStream_stderr = 2
};

struct CArcCmdLineOptions
{
  bool HelpMode;

  // bool LargePages;
  bool CaseSensitive_Change;
  bool CaseSensitive;

  bool IsInTerminal;
  bool IsStdOutTerminal;
  bool IsStdErrTerminal;
  bool StdInMode;
  bool StdOutMode;
  bool EnableHeaders;
  bool DisablePercents;


  bool YesToAll;
  bool ShowDialog;
  bool TechMode;
  bool ShowTime;
  CBoolPair ListPathSeparatorSlash;

  CBoolPair NtSecurity;
  CBoolPair AltStreams;
  CBoolPair HardLinks;
  CBoolPair SymLinks;
  
  CBoolPair StoreOwnerId;
  CBoolPair StoreOwnerName;

  AString ListFields;

  int ConsoleCodePage;

  NWildcard::CCensor Censor;

  CArcCommand Command;
  UString ArchiveName;

  #ifndef Z7_NO_CRYPTO
  bool PasswordEnabled;
  UString Password;
  #endif

  UStringVector HashMethods;
  // UString HashFilePath;

  // bool AppendName;
  // UStringVector ArchivePathsSorted;
  // UStringVector ArchivePathsFullSorted;
  NWildcard::CCensor arcCensor;
  UString ArcName_for_StdInMode;

  CObjectVector<CProperty> Properties;

  CExtractOptionsBase ExtractOptions;

  CUpdateOptions UpdateOptions;
  CHashOptions HashOptions;
  UString ArcType;
  UStringVector ExcludedArcTypes;
  
  unsigned Number_for_Out;
  unsigned Number_for_Errors;
  unsigned Number_for_Percents;
  unsigned LogLevel;

  // bool IsOutAllowed() const { return Number_for_Out != k_OutStream_disabled; }

  // Benchmark
  UInt32 NumIterations;
  bool NumIterations_Defined;

  CArcCmdLineOptions():
      HelpMode(false),
      // LargePages(false),
      CaseSensitive_Change(false),
      CaseSensitive(false),

      IsInTerminal(false),
      IsStdOutTerminal(false),
      IsStdErrTerminal(false),

      StdInMode(false),
      StdOutMode(false),

      EnableHeaders(false),
      DisablePercents(false),
      
      YesToAll(false),
      ShowDialog(false),
      TechMode(false),
      ShowTime(false),

      ConsoleCodePage(-1),

      Number_for_Out(k_OutStream_stdout),
      Number_for_Errors(k_OutStream_stderr),
      Number_for_Percents(k_OutStream_stdout),

      LogLevel(0)
  {
    ListPathSeparatorSlash.Val =
#ifdef _WIN32
        false;
#else
        true;
#endif
  }
};

class CArcCmdLineParser
{
  NCommandLineParser::CParser parser;
public:
  UString Parse1Log;
  void Parse1(const UStringVector &commandStrings, CArcCmdLineOptions &options);
  void Parse2(CArcCmdLineOptions &options);
};

#endif
