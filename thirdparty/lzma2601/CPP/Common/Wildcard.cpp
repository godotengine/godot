// Common/Wildcard.cpp

#include "StdAfx.h"

#include "Wildcard.h"

extern
bool g_CaseSensitive;
bool g_CaseSensitive =
  #ifdef _WIN32
    false;
  #elif defined (__APPLE__)
    #ifdef TARGET_OS_IPHONE
      true;
    #else
      false;
    #endif
  #else
    true;
  #endif


bool IsPath1PrefixedByPath2(const wchar_t *s1, const wchar_t *s2)
{
  if (g_CaseSensitive)
    return IsString1PrefixedByString2(s1, s2);
  return IsString1PrefixedByString2_NoCase(s1, s2);
}

// #include <stdio.h>

/*
static int MyStringCompare_PathLinux(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    wchar_t c1 = *s1++;
    wchar_t c2 = *s2++;
    if (c1 != c2)
    {
      if (c1 == 0) return -1;
      if (c2 == 0) return 1;
      if (c1 == '/') c1 = 0;
      if (c2 == '/') c2 = 0;
      if (c1 < c2) return -1;
      if (c1 > c2) return 1;
      continue;
    }
    if (c1 == 0) return 0;
  }
}
*/

static int MyStringCompare_Path(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    wchar_t c1 = *s1++;
    wchar_t c2 = *s2++;
    if (c1 != c2)
    {
      if (c1 == 0) return -1;
      if (c2 == 0) return 1;
      if (IS_PATH_SEPAR(c1)) c1 = 0;
      if (IS_PATH_SEPAR(c2)) c2 = 0;
      if (c1 < c2) return -1;
      if (c1 > c2) return 1;
      continue;
    }
    if (c1 == 0) return 0;
  }
}

static int MyStringCompareNoCase_Path(const wchar_t *s1, const wchar_t *s2) throw()
{
  for (;;)
  {
    wchar_t c1 = *s1++;
    wchar_t c2 = *s2++;
    if (c1 != c2)
    {
      if (c1 == 0) return -1;
      if (c2 == 0) return 1;
      if (IS_PATH_SEPAR(c1)) c1 = 0;
      if (IS_PATH_SEPAR(c2)) c2 = 0;
      c1 = MyCharUpper(c1);
      c2 = MyCharUpper(c2);
      if (c1 < c2) return -1;
      if (c1 > c2) return 1;
      continue;
    }
    if (c1 == 0) return 0;
  }
}

int CompareFileNames(const wchar_t *s1, const wchar_t *s2) STRING_UNICODE_THROW
{
  /*
  printf("\nCompareFileNames");
  printf("\n S1: %ls", s1);
  printf("\n S2: %ls", s2);
  printf("\n");
  */
  // 21.07 : we parse PATH_SEPARATOR so: 0 < PATH_SEPARATOR < 1
  if (g_CaseSensitive)
    return MyStringCompare_Path(s1, s2);
  return MyStringCompareNoCase_Path(s1, s2);
}

#ifndef USE_UNICODE_FSTRING
int CompareFileNames(const char *s1, const char *s2)
{
  const UString u1 = fs2us(s1);
  const UString u2 = fs2us(s2);
  return CompareFileNames(u1, u2);
}
#endif

// -----------------------------------------
// this function compares name with mask
// ? - any char
// * - any char or empty

static bool EnhancedMaskTest(const wchar_t *mask, const wchar_t *name)
{
  for (;;)
  {
    const wchar_t m = *mask;
    const wchar_t c = *name;
    if (m == 0)
      return (c == 0);
    if (m == '*')
    {
      if (EnhancedMaskTest(mask + 1, name))
        return true;
      if (c == 0)
        return false;
    }
    else
    {
      if (m == '?')
      {
        if (c == 0)
          return false;
      }
      else if (m != c)
        if (g_CaseSensitive || MyCharUpper(m) != MyCharUpper(c))
          return false;
      mask++;
    }
    name++;
  }
}

// --------------------------------------------------
// Splits path to strings

void SplitPathToParts(const UString &path, UStringVector &pathParts)
{
  pathParts.Clear();
  unsigned len = path.Len();
  if (len == 0)
    return;
  UString name;
  unsigned prev = 0;
  for (unsigned i = 0; i < len; i++)
    if (IsPathSepar(path[i]))
    {
      name.SetFrom(path.Ptr(prev), i - prev);
      pathParts.Add(name);
      prev = i + 1;
    }
  name.SetFrom(path.Ptr(prev), len - prev);
  pathParts.Add(name);
}

void SplitPathToParts_2(const UString &path, UString &dirPrefix, UString &name)
{
  const wchar_t *start = path;
  const wchar_t *p = start + path.Len();
  for (; p != start; p--)
    if (IsPathSepar(*(p - 1)))
      break;
  dirPrefix.SetFrom(path, (unsigned)(p - start));
  name = p;
}

void SplitPathToParts_Smart(const UString &path, UString &dirPrefix, UString &name)
{
  const wchar_t *start = path;
  const wchar_t *p = start + path.Len();
  if (p != start)
  {
    if (IsPathSepar(*(p - 1)))
      p--;
    for (; p != start; p--)
      if (IsPathSepar(*(p - 1)))
        break;
  }
  dirPrefix.SetFrom(path, (unsigned)(p - start));
  name = p;
}

/*
UString ExtractDirPrefixFromPath(const UString &path)
{
  return path.Left(path.ReverseFind_PathSepar() + 1));
}
*/

UString ExtractFileNameFromPath(const UString &path)
{
  return UString(path.Ptr((unsigned)(path.ReverseFind_PathSepar() + 1)));
}


bool DoesWildcardMatchName(const UString &mask, const UString &name)
{
  return EnhancedMaskTest(mask, name);
}

bool DoesNameContainWildcard(const UString &path)
{
  for (unsigned i = 0; i < path.Len(); i++)
  {
    wchar_t c = path[i];
    if (c == '*' || c == '?')
      return true;
  }
  return false;
}


// ----------------------------------------------------------'
// NWildcard

namespace NWildcard {

/*

M = MaskParts.Size();
N = TestNameParts.Size();

                           File                          Dir
ForFile     rec   M<=N  [N-M, N)                          -
!ForDir  nonrec   M=N   [0, M)                            -
 
ForDir      rec   M<N   [0, M) ... [N-M-1, N-1)  same as ForBoth-File
!ForFile nonrec         [0, M)                   same as ForBoth-File

ForFile     rec   m<=N  [0, M) ... [N-M, N)      same as ForBoth-File
ForDir   nonrec         [0, M)                   same as ForBoth-File

*/

bool CItem::AreAllAllowed() const
{
  return ForFile && ForDir && WildcardMatching
      && PathParts.Size() == 1 && PathParts.Front().IsEqualTo("*");
}

bool CItem::CheckPath(const UStringVector &pathParts, bool isFile) const
{
  if (!isFile && !ForDir)
    return false;

  /*
  if (PathParts.IsEmpty())
  {
    // PathParts.IsEmpty() means all items (universal wildcard)
    if (!isFile)
      return true;
    if (pathParts.Size() <= 1)
      return ForFile;
    return (ForDir || Recursive && ForFile);
  }
  */

  int delta = (int)pathParts.Size() - (int)PathParts.Size();
  if (delta < 0)
    return false;
  int start = 0;
  int finish = 0;
  
  if (isFile)
  {
    if (!ForDir)
    {
      if (Recursive)
        start = delta;
      else if (delta !=0)
        return false;
    }
    if (!ForFile && delta == 0)
      return false;
  }
  
  if (Recursive)
  {
    finish = delta;
    if (isFile && !ForFile)
      finish = delta - 1;
  }
  
  for (int d = start; d <= finish; d++)
  {
    unsigned i;
    for (i = 0; i < PathParts.Size(); i++)
    {
      if (WildcardMatching)
      {
        if (!DoesWildcardMatchName(PathParts[i], pathParts[i + (unsigned)d]))
          break;
      }
      else
      {
        if (CompareFileNames(PathParts[i], pathParts[i + (unsigned)d]) != 0)
          break;
      }
    }
    if (i == PathParts.Size())
      return true;
  }
  return false;
}

bool CCensorNode::AreAllAllowed() const
{
  if (!Name.IsEmpty() ||
      !SubNodes.IsEmpty() ||
      !ExcludeItems.IsEmpty() ||
      IncludeItems.Size() != 1)
    return false;
  return IncludeItems.Front().AreAllAllowed();
}

int CCensorNode::FindSubNode(const UString &name) const
{
  FOR_VECTOR (i, SubNodes)
    if (CompareFileNames(SubNodes[i].Name, name) == 0)
      return (int)i;
  return -1;
}

void CCensorNode::AddItemSimple(bool include, CItem &item)
{
  CObjectVector<CItem> &items = include ? IncludeItems : ExcludeItems;
  items.Add(item);
}

void CCensorNode::AddItem(bool include, CItem &item, int ignoreWildcardIndex)
{
  if (item.PathParts.Size() <= 1)
  {
    if (item.PathParts.Size() != 0 && item.WildcardMatching)
    {
      if (!DoesNameContainWildcard(item.PathParts.Front()))
        item.WildcardMatching = false;
    }
    AddItemSimple(include, item);
    return;
  }

  const UString &front = item.PathParts.Front();
  
  // WIN32 doesn't support wildcards in file names
  if (item.WildcardMatching
      && ignoreWildcardIndex != 0
      && DoesNameContainWildcard(front))
  {
    AddItemSimple(include, item);
    return;
  }
  CCensorNode &subNode = Find_SubNode_Or_Add_New(front);
  item.PathParts.Delete(0);
  subNode.AddItem(include, item, ignoreWildcardIndex - 1);
}

/*
void CCensorNode::AddItem(bool include, const UString &path, const CCensorPathProps &props)
{
  CItem item;
  SplitPathToParts(path, item.PathParts);
  item.Recursive = props.Recursive;
  item.ForFile = props.ForFile;
  item.ForDir = props.ForDir;
  item.WildcardMatching = props.WildcardMatching;
  AddItem(include, item);
}
*/

bool CCensorNode::NeedCheckSubDirs() const
{
  FOR_VECTOR (i, IncludeItems)
  {
    const CItem &item = IncludeItems[i];
    if (item.Recursive || item.PathParts.Size() > 1)
      return true;
  }
  return false;
}

bool CCensorNode::AreThereIncludeItems() const
{
  if (IncludeItems.Size() > 0)
    return true;
  FOR_VECTOR (i, SubNodes)
    if (SubNodes[i].AreThereIncludeItems())
      return true;
  return false;
}

bool CCensorNode::CheckPathCurrent(bool include, const UStringVector &pathParts, bool isFile) const
{
  const CObjectVector<CItem> &items = include ? IncludeItems : ExcludeItems;
  FOR_VECTOR (i, items)
    if (items[i].CheckPath(pathParts, isFile))
      return true;
  return false;
}

bool CCensorNode::CheckPathVect(const UStringVector &pathParts, bool isFile, bool &include) const
{
  if (CheckPathCurrent(false, pathParts, isFile))
  {
    include = false;
    return true;
  }
  if (pathParts.Size() > 1)
  {
    int index = FindSubNode(pathParts.Front());
    if (index >= 0)
    {
      UStringVector pathParts2 = pathParts;
      pathParts2.Delete(0);
      if (SubNodes[(unsigned)index].CheckPathVect(pathParts2, isFile, include))
        return true;
    }
  }
  bool finded = CheckPathCurrent(true, pathParts, isFile);
  include = finded; // if (!finded), then (true) is allowed also
  return finded;
}

/*
bool CCensorNode::CheckPath2(bool isAltStream, const UString &path, bool isFile, bool &include) const
{
  UStringVector pathParts;
  SplitPathToParts(path, pathParts);
  if (CheckPathVect(pathParts, isFile, include))
  {
    if (!include || !isAltStream)
      return true;
  }
  if (isAltStream && !pathParts.IsEmpty())
  {
    UString &back = pathParts.Back();
    int pos = back.Find(L':');
    if (pos > 0)
    {
      back.DeleteFrom(pos);
      return CheckPathVect(pathParts, isFile, include);
    }
  }
  return false;
}

bool CCensorNode::CheckPath(bool isAltStream, const UString &path, bool isFile) const
{
  bool include;
  if (CheckPath2(isAltStream, path, isFile, include))
    return include;
  return false;
}
*/

bool CCensorNode::CheckPathToRoot_Change(bool include, UStringVector &pathParts, bool isFile) const
{
  if (CheckPathCurrent(include, pathParts, isFile))
    return true;
  if (!Parent)
    return false;
  pathParts.Insert(0, Name);
  return Parent->CheckPathToRoot_Change(include, pathParts, isFile);
}

bool CCensorNode::CheckPathToRoot(bool include, const UStringVector &pathParts, bool isFile) const
{
  if (CheckPathCurrent(include, pathParts, isFile))
    return true;
  if (!Parent)
    return false;
  UStringVector pathParts2;
  pathParts2.Add(Name);
  pathParts2 += pathParts;
  return Parent->CheckPathToRoot_Change(include, pathParts2, isFile);
}

/*
bool CCensorNode::CheckPathToRoot(bool include, const UString &path, bool isFile) const
{
  UStringVector pathParts;
  SplitPathToParts(path, pathParts);
  return CheckPathToRoot(include, pathParts, isFile);
}
*/

void CCensorNode::ExtendExclude(const CCensorNode &fromNodes)
{
  ExcludeItems += fromNodes.ExcludeItems;
  FOR_VECTOR (i, fromNodes.SubNodes)
  {
    const CCensorNode &node = fromNodes.SubNodes[i];
    Find_SubNode_Or_Add_New(node.Name).ExtendExclude(node);
  }
}

int CCensor::FindPairForPrefix(const UString &prefix) const
{
  FOR_VECTOR (i, Pairs)
    if (CompareFileNames(Pairs[i].Prefix, prefix) == 0)
      return (int)i;
  return -1;
}

#ifdef _WIN32

bool IsDriveColonName(const wchar_t *s)
{
  unsigned c = s[0];
  c |= 0x20;
  c -= 'a';
  return c <= (unsigned)('z' - 'a') && s[1] == ':' && s[2] == 0;
}

unsigned GetNumPrefixParts_if_DrivePath(UStringVector &pathParts)
{
  if (pathParts.IsEmpty())
    return 0;
  
  unsigned testIndex = 0;
  if (pathParts[0].IsEmpty())
  {
    if (pathParts.Size() < 4
        || !pathParts[1].IsEmpty()
        || !pathParts[2].IsEqualTo("?"))
      return 0;
    testIndex = 3;
  }
  if (NWildcard::IsDriveColonName(pathParts[testIndex]))
    return testIndex + 1;
  return 0;
}

#endif

static unsigned GetNumPrefixParts(const UStringVector &pathParts)
{
  if (pathParts.IsEmpty())
    return 0;

  /* empty last part could be removed already from (pathParts),
     if there was tail path separator (slash) in original full path string. */
  
  #ifdef _WIN32

  if (IsDriveColonName(pathParts[0]))
    return 1;
  if (!pathParts[0].IsEmpty())
    return 0;

  if (pathParts.Size() == 1)
    return 1;
  if (!pathParts[1].IsEmpty())
    return 1;
  if (pathParts.Size() == 2)
    return 2;
  if (pathParts[2].IsEqualTo("."))
    return 3;

  unsigned networkParts = 2;
  if (pathParts[2].IsEqualTo("?"))
  {
    if (pathParts.Size() == 3)
      return 3;
    if (IsDriveColonName(pathParts[3]))
      return 4;
    if (!pathParts[3].IsEqualTo_Ascii_NoCase("UNC"))
      return 3;
    networkParts = 4;
  }

  networkParts +=
      // 2; // server/share
      1; // server
  if (pathParts.Size() <= networkParts)
    return pathParts.Size();
  return networkParts;

  #else
  
  return pathParts[0].IsEmpty() ? 1 : 0;
 
  #endif
}

void CCensor::AddItem(ECensorPathMode pathMode, bool include, const UString &path,
    const CCensorPathProps &props)
{
  if (path.IsEmpty())
    throw "Empty file path";

  UStringVector pathParts;
  SplitPathToParts(path, pathParts);

  CCensorPathProps props2 = props;

  bool forFile = true;
  bool forDir = true;
  const UString &back = pathParts.Back();
  if (back.IsEmpty())
  {
    // we have tail path separator. So it's directory.
    // we delete tail path separator here even for "\" and "c:\"
    forFile = false;
    pathParts.DeleteBack();
  }
  else
  {
    if (props.MarkMode == kMark_StrictFile
        || (props.MarkMode == kMark_StrictFile_IfWildcard
            && DoesNameContainWildcard(back)))
      forDir = false;
  }

  
  UString prefix;
  
  int ignoreWildcardIndex = -1;

  // #ifdef _WIN32
  // we ignore "?" wildcard in "\\?\" prefix.
  if (pathParts.Size() >= 3
      && pathParts[0].IsEmpty()
      && pathParts[1].IsEmpty()
      && pathParts[2].IsEqualTo("?"))
    ignoreWildcardIndex = 2;
  // #endif

  if (pathMode != k_AbsPath)
  {
    // detection of the number of Skip Parts for prefix
    ignoreWildcardIndex = -1;

    const unsigned numPrefixParts = GetNumPrefixParts(pathParts);
    unsigned numSkipParts = numPrefixParts;

    if (pathMode != k_FullPath)
    {
      // if absolute path, then all parts before last part will be in prefix
      if (numPrefixParts != 0 && pathParts.Size() > numPrefixParts)
        numSkipParts = pathParts.Size() - 1;
    }
    {
      int dotsIndex = -1;
      for (unsigned i = numPrefixParts; i < pathParts.Size(); i++)
      {
        const UString &part = pathParts[i];
        if (part.IsEqualTo("..") || part.IsEqualTo("."))
          dotsIndex = (int)i;
      }

      if (dotsIndex >= 0)
      {
        if (dotsIndex == (int)pathParts.Size() - 1)
          numSkipParts = pathParts.Size();
        else
          numSkipParts = pathParts.Size() - 1;
      }
    }

    // we split (pathParts) to (prefix) and (pathParts).
    for (unsigned i = 0; i < numSkipParts; i++)
    {
      {
        const UString &front = pathParts.Front();
        // WIN32 doesn't support wildcards in file names
        if (props.WildcardMatching)
          if (i >= numPrefixParts && DoesNameContainWildcard(front))
            break;
        prefix += front;
        prefix.Add_PathSepar();
      }
      pathParts.Delete(0);
    }
  }

  int index = FindPairForPrefix(prefix);
  if (index < 0)
  {
    index = (int)Pairs.Size();
    Pairs.AddNew().Prefix = prefix;
  }

  if (pathMode != k_AbsPath)
  {
    if (pathParts.IsEmpty() || (pathParts.Size() == 1 && pathParts[0].IsEmpty()))
    {
      // we create universal item, if we skip all parts as prefix (like \ or L:\ )
      pathParts.Clear();
      pathParts.Add(UString("*"));
      forFile = true;
      forDir = true;
      props2.WildcardMatching = true;
      props2.Recursive = false;
    }
  }

  /*
  // not possible now
  if (!forDir && !forFile)
  {
    UString s ("file path was blocked for files and directories: ");
    s += path;
    throw s;
    // return; // for debug : ignore item (don't create Item)
  }
  */

  CItem item;
  item.PathParts = pathParts;
  item.ForDir = forDir;
  item.ForFile = forFile;
  item.Recursive = props2.Recursive;
  item.WildcardMatching = props2.WildcardMatching;
  Pairs[(unsigned)index].Head.AddItem(include, item, ignoreWildcardIndex);
}

/*
bool CCensor::CheckPath(bool isAltStream, const UString &path, bool isFile) const
{
  bool finded = false;
  FOR_VECTOR (i, Pairs)
  {
    bool include;
    if (Pairs[i].Head.CheckPath2(isAltStream, path, isFile, include))
    {
      if (!include)
        return false;
      finded = true;
    }
  }
  return finded;
}
*/

void CCensor::ExtendExclude()
{
  unsigned i;
  for (i = 0; i < Pairs.Size(); i++)
    if (Pairs[i].Prefix.IsEmpty())
      break;
  if (i == Pairs.Size())
    return;
  unsigned index = i;
  for (i = 0; i < Pairs.Size(); i++)
    if (index != i)
      Pairs[i].Head.ExtendExclude(Pairs[index].Head);
}

void CCensor::AddPathsToCensor(ECensorPathMode censorPathMode)
{
  FOR_VECTOR(i, CensorPaths)
  {
    const CCensorPath &cp = CensorPaths[i];
    AddItem(censorPathMode, cp.Include, cp.Path, cp.Props);
  }
  CensorPaths.Clear();
}

void CCensor::AddPreItem(bool include, const UString &path, const CCensorPathProps &props)
{
  CCensorPath &cp = CensorPaths.AddNew();
  cp.Path = path;
  cp.Include = include;
  cp.Props = props;
}

}
