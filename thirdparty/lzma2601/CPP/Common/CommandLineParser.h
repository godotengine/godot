// Common/CommandLineParser.h

#ifndef ZIP7_INC_COMMON_COMMAND_LINE_PARSER_H
#define ZIP7_INC_COMMON_COMMAND_LINE_PARSER_H

#include "MyString.h"

namespace NCommandLineParser {

bool SplitCommandLine(const UString &src, UString &dest1, UString &dest2);
void SplitCommandLine(const UString &s, UStringVector &parts);

namespace NSwitchType
{
  enum EEnum
  {
    kSimple,
    kMinus,
    kString,
    kChar
  };
}

struct CSwitchForm
{
  const char *Key;
  Byte Type;
  bool Multi;
  Byte MinLen;
  // int MaxLen;
  const char *PostCharSet;
};

struct CSwitchResult
{
  bool ThereIs;
  bool WithMinus;
  int PostCharIndex;
  UStringVector PostStrings;
  
  CSwitchResult(): ThereIs(false) {}
};
  
class CParser
{
  CSwitchResult *_switches;

  bool ParseString(const UString &s, const CSwitchForm *switchForms, unsigned numSwitches);
public:
  UStringVector NonSwitchStrings;
  int StopSwitchIndex;  // NonSwitchStrings[StopSwitchIndex+] are after "--"
  AString ErrorMessage;
  UString ErrorLine;
  
  CParser();
  ~CParser();
  bool ParseStrings(const CSwitchForm *switchForms, unsigned numSwitches, const UStringVector &commandStrings);
  const CSwitchResult& operator[](unsigned index) const { return _switches[index]; }
};

}

#endif
