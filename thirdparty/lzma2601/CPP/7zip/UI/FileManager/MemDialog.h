// MemDialog.h

#ifndef ZIP7_INC_MEM_DIALOG_H
#define ZIP7_INC_MEM_DIALOG_H

#include "../../../Windows/Control/Dialog.h"
// #include "../../../Windows/Control/ComboBox.h"

#include "MemDialogRes.h"

class CMemDialog: public NWindows::NControl::CModalDialog
{
  // NWindows::NControl::CComboBox m_Action;
  // we can disable default OnOK() when we press Enter
  // virtual void OnOK() Z7_override { }
  virtual void OnContinue() Z7_override;
  virtual bool OnInit() Z7_override;
  virtual bool OnButtonClicked(unsigned buttonID, HWND buttonHWND) Z7_override;
  void EnableSpin(bool enable);
  // int AddAction(UINT id);
public:
  bool NeedSave;
  bool Remember;
  bool SkipArc;
  bool TestMode;
  bool ShowRemember;
  // bool ShowSkipFile;
  UInt32 Required_GB;
  UInt32 Limit_GB;
  UString ArcPath;
  UString FilePath;

  void AddInfoMessage_To_String(UString &s, const UInt32 *ramSize_GB = NULL);
  
  CMemDialog():
    NeedSave(false),
    Remember(false),
    SkipArc(false),
    TestMode(false),
    ShowRemember(true),
    // ShowSkipFile(true),
    Required_GB(4),
    Limit_GB(4)
    {}
  INT_PTR Create(HWND parentWindow = NULL) { return CModalDialog::Create(IDD_MEM, parentWindow); }
};

#endif
