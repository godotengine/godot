// Windows/Control/CommandBar.h
  
#ifndef ZIP7_INC_WINDOWS_CONTROL_COMMANDBAR_H
#define ZIP7_INC_WINDOWS_CONTROL_COMMANDBAR_H

#ifdef UNDER_CE

#include "../../Common/MyWindows.h"

#include <commctrl.h>

#include "../Window.h"

namespace NWindows {
namespace NControl {

class CCommandBar: public NWindows::CWindow
{
public:
  bool Create(HINSTANCE hInst, HWND hwndParent, int idCmdBar)
  {
    _window = ::CommandBar_Create(hInst, hwndParent, idCmdBar);
    return (_window != NULL);
  }
  
  // Macros
  // void Destroy() { CommandBar_Destroy(_window); }
  // bool AddButtons(UINT numButtons, LPTBBUTTON buttons) { return BOOLToBool(SendMsg(TB_ADDBUTTONS, (WPARAM)numButtons, (LPARAM)buttons)); }
  // bool InsertButton(unsigned iButton, LPTBBUTTON button) { return BOOLToBool(SendMsg(TB_INSERTBUTTON, (WPARAM)iButton, (LPARAM)button)); }
  // BOOL AddToolTips(UINT numToolTips, LPTSTR toolTips) { return BOOLToBool(SendMsg(TB_SETTOOLTIPS, (WPARAM)numToolTips, (LPARAM)toolTips)); }
  void AutoSize() { SendMsg(TB_AUTOSIZE, 0, 0); }

  // bool AddAdornments(DWORD dwFlags) { return BOOLToBool(::CommandBar_AddAdornments(_window, dwFlags, 0)); }
  // int AddBitmap(HINSTANCE hInst, int idBitmap, int iNumImages, int iImageWidth, int iImageHeight) { return ::CommandBar_AddBitmap(_window, hInst, idBitmap, iNumImages, iImageWidth, iImageHeight); }
  bool DrawMenuBar(WORD iButton) { return BOOLToBool(::CommandBar_DrawMenuBar(_window, iButton)); }
  HMENU GetMenu(WORD iButton) { return ::CommandBar_GetMenu(_window, iButton); }
  int Height() { return CommandBar_Height(_window); }
  HWND InsertComboBox(HINSTANCE hInst, int iWidth, UINT dwStyle, WORD idComboBox, WORD iButton) { return ::CommandBar_InsertComboBox(_window, hInst, iWidth, dwStyle, idComboBox, iButton); }
  bool InsertMenubar(HINSTANCE hInst, WORD idMenu, WORD iButton) { return BOOLToBool(::CommandBar_InsertMenubar(_window, hInst, idMenu, iButton)); }
  bool InsertMenubarEx(HINSTANCE hInst, LPTSTR pszMenu, WORD iButton) { return BOOLToBool(::CommandBar_InsertMenubarEx(_window, hInst, pszMenu, iButton)); }
  bool Show(bool cmdShow) { return BOOLToBool(::CommandBar_Show(_window, BoolToBOOL(cmdShow))); }
  

  // CE 4.0
  void AlignAdornments() { CommandBar_AlignAdornments(_window); }
};

}}

#endif

#endif
