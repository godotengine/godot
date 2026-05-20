// OverwriteDialog.h

#ifndef ZIP7_INC_OVERWRITE_DIALOG_H
#define ZIP7_INC_OVERWRITE_DIALOG_H

#include "../../../Windows/Control/Dialog.h"

#include "DialogSize.h"
#include "OverwriteDialogRes.h"

namespace NOverwriteDialog
{
  struct CFileInfo
  {
    bool Size_IsDefined;
    bool Time_IsDefined;
    bool Is_FileSystemFile;
    UInt64 Size;
    FILETIME Time;
    UString Path;

    void SetTime(const FILETIME &t)
    {
      Time = t;
      Time_IsDefined = true;
    }
    
    void SetTime2(const FILETIME *t)
    {
      if (!t)
        Time_IsDefined = false;
      else
        SetTime(*t);
    }

    void SetSize(UInt64 size)
    {
      Size = size;
      Size_IsDefined = true;
    }

    void SetSize2(const UInt64 *size)
    {
      if (!size)
        Size_IsDefined = false;
      else
        SetSize(*size);
    }

    CFileInfo():
      Size_IsDefined(false),
      Time_IsDefined(false),
      Is_FileSystemFile(false)
      {}
  };
}

class COverwriteDialog: public NWindows::NControl::CModalDialog
{
#ifdef UNDER_CE
  bool _isBig;
#endif

  void SetItemIcon(unsigned iconID, HICON hIcon);
  void SetFileInfoControl(const NOverwriteDialog::CFileInfo &fileInfo, unsigned textID, unsigned iconID, unsigned iconID_2);
  virtual bool OnInit() Z7_override;
  virtual bool OnDestroy() Z7_override;
  virtual bool OnButtonClicked(unsigned buttonID, HWND buttonHWND) Z7_override;
  void ReduceString(UString &s);

public:
  bool ShowExtraButtons;
  bool DefaultButton_is_NO;
  NOverwriteDialog::CFileInfo OldFileInfo;
  NOverwriteDialog::CFileInfo NewFileInfo;

  COverwriteDialog(): ShowExtraButtons(true), DefaultButton_is_NO(false) {}

  INT_PTR Create(HWND parent = NULL)
  {
#ifdef UNDER_CE
    BIG_DIALOG_SIZE(280, 200);
    _isBig = isBig;
#endif
    return CModalDialog::Create(SIZED_DIALOG(IDD_OVERWRITE), parent);
  }
};

#endif
