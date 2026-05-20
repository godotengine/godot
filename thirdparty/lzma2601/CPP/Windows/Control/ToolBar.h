// Windows/Control/ToolBar.h
  
#ifndef ZIP7_INC_WINDOWS_CONTROL_TOOLBAR_H
#define ZIP7_INC_WINDOWS_CONTROL_TOOLBAR_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CToolBar: public NWindows::CWindow
{
public:
  void AutoSize() { SendMsg(TB_AUTOSIZE, 0, 0); }
  DWORD GetButtonSize() { return (DWORD)SendMsg(TB_GETBUTTONSIZE, 0, 0); }
  
  bool GetMaxSize(LPSIZE size)
  #ifdef UNDER_CE
  {
    // maybe it must be fixed for more than 1 buttons
    const DWORD val = GetButtonSize();
    size->cx = LOWORD(val);
    size->cy = HIWORD(val);
    return true;
  }
  #else
  {
    return LRESULTToBool(SendMsg(TB_GETMAXSIZE, 0, (LPARAM)size));
  }
  #endif

  bool EnableButton(UINT buttonID, bool enable) { return LRESULTToBool(SendMsg(TB_ENABLEBUTTON, buttonID, MAKELONG(BoolToBOOL(enable), 0))); }
  void ButtonStructSize() { SendMsg(TB_BUTTONSTRUCTSIZE, sizeof(TBBUTTON)); }
  HIMAGELIST SetImageList(UINT listIndex, HIMAGELIST imageList) { return HIMAGELIST(SendMsg(TB_SETIMAGELIST, listIndex, (LPARAM)imageList)); }
  bool AddButton(UINT numButtons, LPTBBUTTON buttons) { return LRESULTToBool(SendMsg(TB_ADDBUTTONS, numButtons, (LPARAM)buttons)); }
  #ifndef _UNICODE
  bool AddButtonW(UINT numButtons, LPTBBUTTON buttons) { return LRESULTToBool(SendMsg(TB_ADDBUTTONSW, numButtons, (LPARAM)buttons)); }
  #endif
};

}}

#endif
