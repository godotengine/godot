// ExtractingFilePath.h

#ifndef ZIP7_INC_EXTRACTING_FILE_PATH_H
#define ZIP7_INC_EXTRACTING_FILE_PATH_H

#include "../../../Common/MyString.h"

// #ifdef _WIN32
void Correct_AltStream_Name(UString &s);
// #endif

// replaces unsuported characters, and replaces "." , ".." and "" to "[]"
UString Get_Correct_FsFile_Name(const UString &name);

/*
  Correct_FsPath() corrects path parts to prepare it for File System operations.
  It also corrects empty path parts like "\\\\":
    - frontal empty path parts : it removes them or changes them to "_"
    - another empty path parts : it removes them
  if (absIsAllowed && path is absolute)  : it removes empty path parts after start absolute path prefix marker
  else
  {
    if (!keepAndReplaceEmptyPrefixes) : it removes empty path parts
    if ( keepAndReplaceEmptyPrefixes) : it changes each empty frontal path part to "_"
  }
*/
void Correct_FsPath(bool absIsAllowed, bool keepAndReplaceEmptyPrefixes, UStringVector &parts, bool isDir);

UString MakePathFromParts(const UStringVector &parts);

#endif
