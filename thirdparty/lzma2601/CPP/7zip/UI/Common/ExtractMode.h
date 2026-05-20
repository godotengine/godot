// ExtractMode.h

#ifndef ZIP7_INC_EXTRACT_MODE_H
#define ZIP7_INC_EXTRACT_MODE_H

namespace NExtract {
  
namespace NPathMode
{
  enum EEnum
  {
    kFullPaths,
    kCurPaths,
    kNoPaths,
    kAbsPaths,
    kNoPathsAlt // alt streams must be extracted without name of base file
  };
}

namespace NOverwriteMode
{
  enum EEnum
  {
    kAsk,
    kOverwrite,
    kSkip,
    kRename,
    kRenameExisting
  };
}

namespace NZoneIdMode
{
  enum EEnum
  {
    kNone,
    kAll,
    kOffice
  };
}

}

#endif
