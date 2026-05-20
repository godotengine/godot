// SysIconUtils.h

#ifndef ZIP7_INC_SYS_ICON_UTILS_H
#define ZIP7_INC_SYS_ICON_UTILS_H

#include "../../../Common/MyWindows.h"

#include <CommCtrl.h>

#include "../../../Common/MyString.h"

struct CExtIconPair
{
  UString Ext;
  int IconIndex;
  // UString TypeName;
  // int Compare(const CExtIconPair &a) const { return MyStringCompareNoCase(Ext, a.Ext); }
};

struct CAttribIconPair
{
  DWORD Attrib;
  int IconIndex;
  // UString TypeName;
  // int Compare(const CAttribIconPair &a) const { return Ext.Compare(a.Ext); }
};


struct CExtToIconMap
{
  CRecordVector<CAttribIconPair> _attribMap;
  CObjectVector<CExtIconPair> _extMap_Normal;
  CObjectVector<CExtIconPair> _extMap_Compressed;
  int SplitIconIndex;
  int SplitIconIndex_Defined;
  
  CExtToIconMap(): SplitIconIndex_Defined(false) {}

  void Clear()
  {
    SplitIconIndex_Defined = false;
    _extMap_Normal.Clear();
    _extMap_Compressed.Clear();
    _attribMap.Clear();
  }
  int GetIconIndex_DIR(DWORD attrib = FILE_ATTRIBUTE_DIRECTORY)
  {
    return GetIconIndex(attrib, L"__DIR__");
  }
  int GetIconIndex(DWORD attrib, const wchar_t *fileName /* , UString *typeName */);
};

extern CExtToIconMap g_Ext_to_Icon_Map;

DWORD_PTR Shell_GetFileInfo_SysIconIndex_for_Path_attrib_iconIndexRef(
    CFSTR path, DWORD attrib, int &iconIndex);
HRESULT Shell_GetFileInfo_SysIconIndex_for_Path_return_HRESULT(
    CFSTR path, DWORD attrib, Int32 *iconIndex);
int Shell_GetFileInfo_SysIconIndex_for_Path(CFSTR path, DWORD attrib);

int Shell_GetFileInfo_SysIconIndex_for_CSIDL(int csidl);

HIMAGELIST Shell_Get_SysImageList_smallIcons(bool smallIcons);

#endif
