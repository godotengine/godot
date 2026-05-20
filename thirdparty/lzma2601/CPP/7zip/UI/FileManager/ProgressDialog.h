// ProgressDialog.h

#ifndef ZIP7_INC_PROGRESS_DIALOG_H
#define ZIP7_INC_PROGRESS_DIALOG_H

#include "../../../Windows/Synchronization.h"
#include "../../../Windows/Thread.h"

#include "../../../Windows/Control/Dialog.h"
#include "../../../Windows/Control/ProgressBar.h"

#include "ProgressDialogRes.h"

class CProgressSync
{
  NWindows::NSynchronization::CCriticalSection _cs;
  bool _stopped;
  bool _paused;
  UInt64 _total;
  UInt64 _completed;
public:
  CProgressSync(): _stopped(false), _paused(false), _total(1), _completed(0) {}

  HRESULT ProcessStopAndPause();
  bool GetStopped()
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    return _stopped;
  }
  void SetStopped(bool value)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _stopped = value;
  }
  bool GetPaused()
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    return _paused;
  }
  void SetPaused(bool value)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _paused = value;
  }
  void SetProgress(UInt64 total, UInt64 completed)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _total = total;
    _completed = completed;
  }
  void SetPos(UInt64 completed)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _completed = completed;
  }
  void GetProgress(UInt64 &total, UInt64 &completed)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    total = _total;
    completed = _completed;
  }
};

class CU64ToI32Converter
{
  UInt64 _numShiftBits;
public:
  void Init(UInt64 range)
  {
    // Windows CE doesn't like big number here.
    for (_numShiftBits = 0; range > (1 << 15); _numShiftBits++)
      range >>= 1;
  }
  int Count(UInt64 value) { return int(value >> _numShiftBits); }
};

class CProgressDialog: public NWindows::NControl::CModalDialog
{
private:
  UINT_PTR _timer;

  UString _title;
  CU64ToI32Converter _converter;
  UInt64 _peviousPos;
  UInt64 _range;
  NWindows::NControl::CProgressBar m_ProgressBar;

  UInt64 _prevPercentValue;

  bool _wasCreated;
  bool _needClose;
  bool _inCancelMessageBox;
  bool _externalCloseMessageWasReceived;

  virtual bool OnButtonClicked(unsigned buttonID, HWND buttonHWND) Z7_override;
  virtual bool OnTimer(WPARAM timerID, LPARAM callback) Z7_override;
  virtual bool OnInit() Z7_override;
  virtual void OnCancel() Z7_override;
  virtual void OnOK() Z7_override;
  virtual bool OnMessage(UINT message, WPARAM wParam, LPARAM lParam) Z7_override;

  void SetRange(UInt64 range);
  void SetPos(UInt64 pos);

  NWindows::NSynchronization::CManualResetEvent _dialogCreatedEvent;
  #ifndef Z7_SFX
  void AddToTitle(LPCWSTR string);
  #endif

  void WaitCreating() { _dialogCreatedEvent.Lock(); }
  void CheckNeedClose();
  bool OnExternalCloseMessage();
public:
  CProgressSync Sync;
  int IconID;

  #ifndef Z7_SFX
  HWND MainWindow;
  UString MainTitle;
  UString MainAddTitle;
  ~CProgressDialog();
  #endif

  CProgressDialog(): _timer(0)
    #ifndef Z7_SFX
    ,MainWindow(NULL)
    #endif
  {
    IconID = -1;
    _wasCreated = false;
    _needClose = false;
    _inCancelMessageBox = false;
    _externalCloseMessageWasReceived = false;

    if (_dialogCreatedEvent.Create() != S_OK)
      throw 1334987;
  }

  INT_PTR Create(const UString &title, NWindows::CThread &thread, HWND wndParent = NULL)
  {
    _title = title;
    INT_PTR res = CModalDialog::Create(IDD_PROGRESS, wndParent);
    thread.Wait_Close();
    return res;
  }

  enum
  {
    kCloseMessage = WM_APP + 1
  };

  void ProcessWasFinished()
  {
    WaitCreating();
    if (_wasCreated)
      PostMsg(kCloseMessage);
    else
      _needClose = true;
  }
};


class CProgressCloser
{
  CProgressDialog *_p;
public:
  CProgressCloser(CProgressDialog &p) : _p(&p) {}
  ~CProgressCloser() { _p->ProcessWasFinished(); }
};

#endif
