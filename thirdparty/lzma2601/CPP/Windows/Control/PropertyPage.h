// Windows/Control/PropertyPage.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_PROPERTYPAGE_H
#define ZIP7_INC_WINDOWS_CONTROL_PROPERTYPAGE_H

#include "../../Common/MyWindows.h"

#include <prsht.h>

#include "Dialog.h"

namespace NWindows {
namespace NControl {

INT_PTR APIENTRY ProperyPageProcedure(HWND dialogHWND, UINT message, WPARAM wParam, LPARAM lParam);

class CPropertyPage: public CDialog
{
public:
  CPropertyPage(HWND window = NULL): CDialog(window) {}
  
  void Changed() { PropSheet_Changed(GetParent(), (HWND)*this); }
  void UnChanged() { PropSheet_UnChanged(GetParent(), (HWND)*this); }

  virtual bool OnNotify(UINT controlID, LPNMHDR lParam) Z7_override;

  virtual bool OnKillActive() { return false; } // false = OK
  virtual bool OnKillActive2(const PSHNOTIFY *) { return OnKillActive(); }
  virtual LONG OnSetActive() { return false; } // false = OK
  virtual LONG OnSetActive2(const PSHNOTIFY *) { return OnSetActive(); }
  virtual LONG OnApply() { return PSNRET_NOERROR; }
  virtual LONG OnApply2(const PSHNOTIFY *) { return OnApply(); }
  virtual void OnNotifyHelp() {}
  virtual void OnNotifyHelp2(const PSHNOTIFY *) { OnNotifyHelp(); }
  virtual void OnReset() {}
  virtual void OnReset2(const PSHNOTIFY *) { OnReset(); }
};

struct CPageInfo
{
  CPropertyPage *Page;
  UString Title;
  UINT ID;
};

INT_PTR MyPropertySheet(const CObjectVector<CPageInfo> &pagesInfo, HWND hwndParent, const UString &title);

}}

#endif
