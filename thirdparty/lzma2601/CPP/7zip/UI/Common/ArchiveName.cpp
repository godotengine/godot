// ArchiveName.cpp

#include "StdAfx.h"

#include "../../../../C/Sort.h"

#include "../../../Common/Wildcard.h"
#include "../../../Common/StringToInt.h"

#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileName.h"

#include "ArchiveName.h"
#include "ExtractingFilePath.h"

using namespace NWindows;
using namespace NFile;


static const char * const g_ArcExts =
        "7z"
  "\0"  "zip"
  "\0"  "tar"
  "\0"  "wim"
  "\0";

static const char * const g_HashExts =
  "sha256"
  "\0";


UString CreateArchiveName(
    const UStringVector &paths,
    bool isHash,
    const NFind::CFileInfo *fi,
    UString &baseName)
{
  bool keepName = isHash;
  /*
  if (paths.Size() == 1)
  {
    const UString &name = paths[0];
    if (name.Len() > 4)
      if (CompareFileNames(name.RightPtr(4), L".tar") == 0)
        keepName = true;
  }
  */

  UString name ("Archive");
  NFind::CFileInfo fi3;
  if (paths.Size() > 1)
    fi = NULL;
  if (!fi && paths.Size() != 0)
  {
    const UString &path = paths.Front();
    if (paths.Size() == 1)
    {
      if (fi3.Find(us2fs(path)))
        fi = &fi3;
    }
    else
    {
      // we try to use name of parent folder
      FString dirPrefix;
      if (NDir::GetOnlyDirPrefix(us2fs(path), dirPrefix))
      {
        if (!dirPrefix.IsEmpty() && IsPathSepar(dirPrefix.Back()))
        {
         #if defined(_WIN32) && !defined(UNDER_CE)
          if (NName::IsDriveRootPath_SuperAllowed(dirPrefix))
          {
            if (path != fs2us(dirPrefix))
              name = dirPrefix[dirPrefix.Len() - 3]; // only letter
          }
          else
         #endif
          {
            dirPrefix.DeleteBack();
            if (!dirPrefix.IsEmpty())
            {
              const int slash = dirPrefix.ReverseFind_PathSepar();
              if (slash >= 0 && slash != (int)dirPrefix.Len() - 1)
                name = dirPrefix.Ptr(slash + 1);
              else if (fi3.Find(dirPrefix))
                name = fs2us(fi3.Name);
            }
          }
        }
      }
    }
  }

  if (fi)
  {
    name = fs2us(fi->Name);
    if (!fi->IsDir() && !keepName)
    {
      const int dotPos = name.Find(L'.');
      if (dotPos > 0 && name.Find(L'.', (unsigned)dotPos + 1) < 0)
        name.DeleteFrom((unsigned)dotPos);
    }
  }
  name = Get_Correct_FsFile_Name(name);

  CRecordVector<UInt32> ids;
  bool simple_IsAllowed = true;
  // for (int y = 0; y < 10000; y++) // for debug
  {
    // ids.Clear();
    UString n;

    FOR_VECTOR (i, paths)
    {
      const UString &a = paths[i];
      const int slash = a.ReverseFind_PathSepar();
      // if (name.Len() >= a.Len() - slash + 1) continue;
      const wchar_t *s = a.Ptr(slash + 1);
      if (!IsPath1PrefixedByPath2(s, name))
        continue;
      s += name.Len();
      const char *exts = isHash ? g_HashExts : g_ArcExts;

      for (;;)
      {
        const char *ext = exts;
        const unsigned len = MyStringLen(ext);
        if (len == 0)
          break;
        exts += len + 1;
        n = s;
        if (n.Len() <= len)
          continue;
        if (!StringsAreEqualNoCase_Ascii(n.RightPtr(len), ext))
          continue;
        n.DeleteFrom(n.Len() - len);
        if (n.Back() != '.')
          continue;
        n.DeleteBack();
        if (n.IsEmpty())
        {
          simple_IsAllowed = false;
          break;
        }
        if (n.Len() < 2)
          continue;
        if (n[0] != '_')
          continue;
        const wchar_t *end;
        const UInt32 v = ConvertStringToUInt32(n.Ptr(1), &end);
        if (*end != 0)
          continue;
        ids.Add(v);
        break;
      }
    }
  }

  baseName = name;
  if (!simple_IsAllowed)
  {
    HeapSort(ids.NonConstData(), ids.Size());
    UInt32 v = 2;
    const unsigned num = ids.Size();
    for (unsigned i = 0; i < num; i++)
    {
      const UInt32 id = ids[i];
      if (id > v)
        break;
      if (id == v)
        v = id + 1;
    }
    name.Add_Char('_');
    name.Add_UInt32(v);
  }
  return name;
}
