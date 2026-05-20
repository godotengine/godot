// Common/Wildcard.h

#ifndef ZIP7_INC_COMMON_WILDCARD_H
#define ZIP7_INC_COMMON_WILDCARD_H

#include "MyString.h"

int CompareFileNames(const wchar_t *s1, const wchar_t *s2) STRING_UNICODE_THROW;
#ifndef USE_UNICODE_FSTRING
  int CompareFileNames(const char *s1, const char *s2);
#endif

bool IsPath1PrefixedByPath2(const wchar_t *s1, const wchar_t *s2);

void SplitPathToParts(const UString &path, UStringVector &pathParts);
void SplitPathToParts_2(const UString &path, UString &dirPrefix, UString &name);
void SplitPathToParts_Smart(const UString &path, UString &dirPrefix, UString &name); // ignores dir delimiter at the end of (path)

UString ExtractDirPrefixFromPath(const UString &path);
UString ExtractFileNameFromPath(const UString &path);

bool DoesNameContainWildcard(const UString &path);
bool DoesWildcardMatchName(const UString &mask, const UString &name);

namespace NWildcard {

#ifdef _WIN32
// returns true, if name is like "a:", "c:", ...
bool IsDriveColonName(const wchar_t *s);
unsigned GetNumPrefixParts_if_DrivePath(UStringVector &pathParts);
#endif

struct CItem
{
  UStringVector PathParts;
  bool Recursive;
  bool ForFile;
  bool ForDir;
  bool WildcardMatching;
  
  #ifdef _WIN32
  bool IsDriveItem() const
  {
    return PathParts.Size() == 1 && !ForFile && ForDir && IsDriveColonName(PathParts[0]);
  }
  #endif

  // CItem(): WildcardMatching(true) {}

  bool AreAllAllowed() const;
  bool CheckPath(const UStringVector &pathParts, bool isFile) const;
};



const Byte kMark_FileOrDir = 0;
const Byte kMark_StrictFile = 1;
const Byte kMark_StrictFile_IfWildcard = 2;

struct CCensorPathProps
{
  bool Recursive;
  bool WildcardMatching;
  Byte MarkMode;
  
  CCensorPathProps():
      Recursive(false),
      WildcardMatching(true),
      MarkMode(kMark_FileOrDir)
      {}
};


class CCensorNode  MY_UNCOPYABLE
{
  CCensorNode *Parent;
  
  bool CheckPathCurrent(bool include, const UStringVector &pathParts, bool isFile) const;
  void AddItemSimple(bool include, CItem &item);
public:
  // bool ExcludeDirItems;

  CCensorNode():
      Parent(NULL)
      // , ExcludeDirItems(false)
      {}

  CCensorNode(const UString &name, CCensorNode *parent):
      Parent(parent)
      // , ExcludeDirItems(false)
      , Name(name)
      {}

  UString Name; // WIN32 doesn't support wildcards in file names
  CObjectVector<CCensorNode> SubNodes;
  CObjectVector<CItem> IncludeItems;
  CObjectVector<CItem> ExcludeItems;

  CCensorNode &Find_SubNode_Or_Add_New(const UString &name)
  {
    int i = FindSubNode(name);
    if (i >= 0)
      return SubNodes[(unsigned)i];
    // return SubNodes.Add(CCensorNode(name, this));
    CCensorNode &node = SubNodes.AddNew();
    node.Parent = this;
    node.Name = name;
    return node;
  }

  bool AreAllAllowed() const;

  int FindSubNode(const UString &path) const;

  void AddItem(bool include, CItem &item, int ignoreWildcardIndex = -1);
  // void AddItem(bool include, const UString &path, const CCensorPathProps &props);
  void Add_Wildcard()
  {
    CItem item;
    item.PathParts.Add(L"*");
    item.Recursive = false;
    item.ForFile = true;
    item.ForDir = true;
    item.WildcardMatching = true;
    AddItem(
        true // include
        , item);
  }

  // NeedCheckSubDirs() returns true, if there are IncludeItems rules that affect items in subdirs
  bool NeedCheckSubDirs() const;
  bool AreThereIncludeItems() const;

  /*
  CheckPathVect() doesn't check path in Parent CCensorNode
  so use CheckPathVect() for root CCensorNode
  OUT:
    returns (true) && (include = false) - file in exlude list
    returns (true) && (include = true)  - file in include list and is not in exlude list
    returns (false)  - file is not in (include/exlude) list
  */
  bool CheckPathVect(const UStringVector &pathParts, bool isFile, bool &include) const;

  // bool CheckPath2(bool isAltStream, const UString &path, bool isFile, bool &include) const;
  // bool CheckPath(bool isAltStream, const UString &path, bool isFile) const;

  // CheckPathToRoot_Change() changes pathParts !!!
  bool CheckPathToRoot_Change(bool include, UStringVector &pathParts, bool isFile) const;
  bool CheckPathToRoot(bool include, const UStringVector &pathParts, bool isFile) const;

  // bool CheckPathToRoot(const UString &path, bool isFile, bool include) const;
  void ExtendExclude(const CCensorNode &fromNodes);
};


struct CPair  MY_UNCOPYABLE
{
  UString Prefix;
  CCensorNode Head;
  
  // CPair(const UString &prefix): Prefix(prefix) { };
};


enum ECensorPathMode
{
  k_RelatPath,  // absolute prefix as Prefix, remain path in Tree
  k_FullPath,   // drive prefix as Prefix, remain path in Tree
  k_AbsPath     // full path in Tree
};


struct CCensorPath
{
  UString Path;
  bool Include;
  CCensorPathProps Props;

  CCensorPath():
      Include(true)
      {}
};


class CCensor  MY_UNCOPYABLE
{
  int FindPairForPrefix(const UString &prefix) const;
public:
  CObjectVector<CPair> Pairs;

  bool ExcludeDirItems;
  bool ExcludeFileItems;

  CCensor():
      ExcludeDirItems(false),
      ExcludeFileItems(false)
      {}

  CObjectVector<NWildcard::CCensorPath> CensorPaths;
  
  bool AllAreRelative() const
    { return (Pairs.Size() == 1 && Pairs.Front().Prefix.IsEmpty()); }
  
  void AddItem(ECensorPathMode pathMode, bool include, const UString &path, const CCensorPathProps &props);
  // bool CheckPath(bool isAltStream, const UString &path, bool isFile) const;
  void ExtendExclude();

  void AddPathsToCensor(NWildcard::ECensorPathMode censorPathMode);
  void AddPreItem(bool include, const UString &path, const CCensorPathProps &props);

  void AddPreItem_NoWildcard(const UString &path)
  {
    CCensorPathProps props;
    props.WildcardMatching = false;
    AddPreItem(
        true,  // include
        path, props);
  }
  void AddPreItem_Wildcard()
  {
    CCensorPathProps props;
    // props.WildcardMatching = true;
    AddPreItem(
        true,  // include
        UString("*"), props);
  }
};

}

#endif
