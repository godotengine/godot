// DefaultName.cpp

#include "StdAfx.h"

#include "DefaultName.h"

static UString GetDefaultName3(const UString &fileName,
    const UString &extension, const UString &addSubExtension)
{
  const unsigned extLen = extension.Len();
  const unsigned fileNameLen = fileName.Len();
  
  if (fileNameLen > extLen + 1)
  {
    const unsigned dotPos = fileNameLen - (extLen + 1);
    if (fileName[dotPos] == '.')
      if (extension.IsEqualTo_NoCase(fileName.Ptr(dotPos + 1)))
        return fileName.Left(dotPos) + addSubExtension;
  }
  
  int dotPos = fileName.ReverseFind_Dot();
  if (dotPos > 0)
    return fileName.Left((unsigned)dotPos) + addSubExtension;

  if (addSubExtension.IsEmpty())
    return fileName + L'~';
  else
    return fileName + addSubExtension;
}

UString GetDefaultName2(const UString &fileName,
    const UString &extension, const UString &addSubExtension)
{
  UString name = GetDefaultName3(fileName, extension, addSubExtension);
  name.TrimRight();
  return name;
}
