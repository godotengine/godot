// Windows/Control/StatusBar.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_STATUSBAR_H
#define ZIP7_INC_WINDOWS_CONTROL_STATUSBAR_H

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CStatusBar: public NWindows::CWindow
{
public:
  bool Create(LONG style, LPCTSTR text, HWND hwndParent, UINT id)
    { return (_window = ::CreateStatusWindow(style, text, hwndParent, id)) != NULL; }
  bool SetText(LPCTSTR text)
    { return CWindow::SetText(text); }
  bool SetText(unsigned index, LPCTSTR text, UINT type)
    { return LRESULTToBool(SendMsg(SB_SETTEXT, index | type, (LPARAM)text)); }
  bool SetText(unsigned index, LPCTSTR text)
    { return SetText(index, text, 0); }

  #ifndef _UNICODE
  bool Create(LONG style, LPCWSTR text, HWND hwndParent, UINT id)
    { return (_window = ::CreateStatusWindowW(style, text, hwndParent, id)) != NULL; }
  bool SetText(LPCWSTR text)
    { return CWindow::SetText(text); }
  bool SetText(unsigned index, LPCWSTR text, UINT type)
    { return LRESULTToBool(SendMsg(SB_SETTEXTW, index | type, (LPARAM)text)); }
  bool SetText(unsigned index, LPCWSTR text)
    { return SetText(index, text, 0); }
  #endif

  bool SetParts(unsigned numParts, const int *edgePostions)
    { return LRESULTToBool(SendMsg(SB_SETPARTS, numParts, (LPARAM)edgePostions)); }
  void Simple(bool simple)
    { SendMsg(SB_SIMPLE, (WPARAM)BoolToBOOL(simple), 0); }
};

}}

#endif
