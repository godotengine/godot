// MemDialog.cpp

#include "StdAfx.h"

#include <CommCtrl.h>

#include "MemDialog.h"

#include "../../../Common/StringToInt.h"
#include "../../../Windows/System.h"
#include "../../../Windows/ErrorMsg.h"

#include "../Explorer/MyMessages.h"
#include "../GUI/ExtractRes.h"

#include "resourceGui.h"

#ifdef Z7_LANG
#include "LangUtils.h"
#endif

#ifdef Z7_LANG
static const UInt32 kLangIDs[] =
{
  IDX_MEM_SAVE_LIMIT,
  IDX_MEM_REMEMBER,
  IDG_MEM_ACTION,
  IDR_MEM_ACTION_ALLOW,
  IDR_MEM_ACTION_SKIP_ARC
  // , IDR_MEM_SKIP_FILE
};
#endif

static const unsigned k_Action_Buttons[] =
{
  IDR_MEM_ACTION_ALLOW,
  IDR_MEM_ACTION_SKIP_ARC
  // , IDR_MEM_SKIP_FILE
};


void CMemDialog::EnableSpin(bool enable)
{
  EnableItem(IDC_MEM_SPIN, enable);
  EnableItem(IDE_MEM_SPIN_EDIT, enable);
}


static void AddSize_GB(UString &s, UInt32 size_GB, UInt32 id)
{
  s.Add_LF();
  s += "    ";
  s.Add_UInt32(size_GB);
  s += " GB : ";
  AddLangString(s, id);
}

void CMemDialog::AddInfoMessage_To_String(UString &s, const UInt32 *ramSize_GB)
{
  AddLangString(s, IDS_MEM_REQUIRES_BIG_MEM);
  AddSize_GB(s, Required_GB, IDS_MEM_REQUIRED_MEM_SIZE);
  AddSize_GB(s, Limit_GB, IDS_MEM_CURRENT_MEM_LIMIT);
  if (ramSize_GB)
    AddSize_GB(s, *ramSize_GB, IDS_MEM_RAM_SIZE);
  if (!FilePath.IsEmpty())
  {
    s.Add_LF();
    s += "File: ";
    s += FilePath;
  }
}

/*
int CMemDialog::AddAction(UINT id)
{
  const int index = (int)m_Action.AddString(LangString(id));
  m_Action.SetItemData(index, (LPARAM)id);
  return index;
}
*/

bool CMemDialog::OnInit()
{
  #ifdef Z7_LANG
  LangSetWindowText(*this, IDD_MEM);
  LangSetDlgItems(*this, kLangIDs, Z7_ARRAY_SIZE(kLangIDs));
  #endif

  // m_Action.Attach(GetItem(IDC_MEM_ACTION));

  size_t ramSize = (size_t)sizeof(size_t) << 29;
  const bool ramSize_defined = NWindows::NSystem::GetRamSize(ramSize);
  // ramSize *= 10; // for debug

  UInt32 ramSize_GB = (UInt32)(((UInt64)ramSize + (1u << 29)) >> 30);
  if (ramSize_GB == 0)
    ramSize_GB = 1;

  const bool is_Allowed = (!ramSize_defined || ramSize > ((UInt64)Required_GB << 30));
  {
    UString s;
    if (!is_Allowed)
    {
      AddLangString(s, IDS_MEM_ERROR);
      s.Add_LF();
    }
    AddInfoMessage_To_String(s, is_Allowed ? NULL : &ramSize_GB);
    if (!ArcPath.IsEmpty())
    // for (int i = 0; i < 10; i++)
    {
      s.Add_LF();
      AddLangString(s, TestMode ?
          IDS_PROGRESS_TESTING :
          IDS_PROGRESS_EXTRACTING);
      s += ": ";
      s += ArcPath;
    }
    SetItemText(IDT_MEM_MESSAGE, s);

    s = "GB";
    if (ramSize_defined)
    {
      s += " / ";
      s.Add_UInt32(ramSize_GB);
      s += " GB (RAM)";
    }
    SetItemText(IDT_MEM_GB, s);
  }
  const UINT valMin = 1;
  UINT valMax = 64; // 64GB for RAR7
  if (ramSize_defined /* && ramSize_GB > valMax */)
  {
    const UINT k_max_val = 1u << 14;
    if (ramSize_GB >= k_max_val)
      valMax = k_max_val;
    else if (ramSize_GB > 1)
      valMax = (UINT)ramSize_GB - 1;
    else
      valMax = 1;
  }

  SendItemMessage(IDC_MEM_SPIN, UDM_SETRANGE, 0, MAKELPARAM(valMax, valMin));    // Sets the controls direction
  // UDM_SETPOS doesn't set value larger than max value (valMax) of range:
  SendItemMessage(IDC_MEM_SPIN, UDM_SETPOS, 0, Required_GB);
  {
    UString s;
    s.Add_UInt32(Required_GB);
    SetItemText(IDE_MEM_SPIN_EDIT, s);
  }

  EnableSpin(false);

  /*
  AddAction(IDB_ALLOW_OPERATION);
  m_Action.SetCurSel(0);
  AddAction(IDB_MEM_SKIP_ARC);
  AddAction(IDB_MEM_SKIP_FILE);
  */

  const UINT buttonId = is_Allowed ?
      IDR_MEM_ACTION_ALLOW :
      IDR_MEM_ACTION_SKIP_ARC;

  CheckRadioButton(
      k_Action_Buttons[0],
      k_Action_Buttons[Z7_ARRAY_SIZE(k_Action_Buttons) - 1],
      buttonId);
  /*
  if (!ShowSkipFile)
    HideItem(IDR_MEM_SKIP_FILE);
  */
  if (!ShowRemember)
    HideItem(IDX_MEM_REMEMBER);
  return CModalDialog::OnInit();
}


bool CMemDialog::OnButtonClicked(unsigned buttonID, HWND buttonHWND)
{
  if (buttonID == IDX_MEM_SAVE_LIMIT)
  {
    EnableSpin(IsButtonCheckedBool(IDX_MEM_SAVE_LIMIT));
    return true;
  }
  return CDialog::OnButtonClicked(buttonID, buttonHWND);
}


void CMemDialog::OnContinue()
{
  Remember = IsButtonCheckedBool(IDX_MEM_REMEMBER);
  NeedSave = IsButtonCheckedBool(IDX_MEM_SAVE_LIMIT);
  SkipArc  = IsButtonCheckedBool(IDR_MEM_ACTION_SKIP_ARC);
  if (NeedSave)
  {
#if 0
    // UDM_GETPOS doesn't support value outside of range that was set:
    LRESULT lresult = SendItemMessage(IDC_MEM_SPIN, UDM_GETPOS, 0, 0);
    const UInt32 val = LOWORD(lresult);
    if (HIWORD(lresult) != 0) // the value outside of allowed range
#else
    UString s;
    GetItemText(IDE_MEM_SPIN_EDIT, s);
    const wchar_t *end;
    const UInt32 val = ConvertStringToUInt32(s.Ptr(), &end);
    if (s.IsEmpty() || *end != 0 || val > (1u << 30))
#endif
    {
      ShowErrorMessage(*this,
          NWindows::NError::MyFormatMessage(E_INVALIDARG)
          // L"Incorrect value"
          );
      return;
    }
    Limit_GB = val;
  }
  CModalDialog::OnContinue();
}
