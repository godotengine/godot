// ProgressDialog2.h

#ifndef ZIP7_INC_PROGRESS_DIALOG_2_H
#define ZIP7_INC_PROGRESS_DIALOG_2_H

#include "../../../Common/MyCom.h"

#include "../../../Windows/ErrorMsg.h"
#include "../../../Windows/Synchronization.h"
#include "../../../Windows/Thread.h"

#include "../../../Windows/Control/Dialog.h"
#include "../../../Windows/Control/ListView.h"
#include "../../../Windows/Control/ProgressBar.h"

#include "MyWindowsNew.h"

struct CProgressMessageBoxPair
{
  UString Title;
  UString Message;
};

struct CProgressFinalMessage
{
  CProgressMessageBoxPair ErrorMessage;
  CProgressMessageBoxPair OkMessage;

  bool ThereIsMessage() const { return !ErrorMessage.Message.IsEmpty() || !OkMessage.Message.IsEmpty(); }
};

class CProgressSync
{
  bool _stopped;
  bool _paused;
public:
  bool _filesProgressMode;
  bool _isDir;
  UInt64 _totalBytes;
  UInt64 _completedBytes;
  UInt64 _totalFiles;
  UInt64 _curFiles;
  UInt64 _inSize;
  UInt64 _outSize;
  
  UString _titleFileName;
  UString _status;
  UString _filePath;

  UStringVector Messages;
  CProgressFinalMessage FinalMessage;

  NWindows::NSynchronization::CCriticalSection _cs;

  CProgressSync();

  bool Get_Stopped()
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    return _stopped;
  }
  void Set_Stopped(bool val)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _stopped = val;
  }
  
  bool Get_Paused();
  void Set_Paused(bool val)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _paused = val;
  }
  
  void Set_FilesProgressMode(bool filesProgressMode)
  {
    NWindows::NSynchronization::CCriticalSectionLock lock(_cs);
    _filesProgressMode = filesProgressMode;
  }
  
  HRESULT CheckStop();
  void Clear_Stop_Status();
  HRESULT ScanProgress(UInt64 numFiles, UInt64 totalSize, const FString &fileName, bool isDir = false);

  HRESULT Set_NumFilesTotal(UInt64 val);
  void Set_NumBytesTotal(UInt64 val);
  void Set_NumFilesCur(UInt64 val);
  HRESULT Set_NumBytesCur(const UInt64 *val);
  HRESULT Set_NumBytesCur(UInt64 val);
  void Set_Ratio(const UInt64 *inSize, const UInt64 *outSize);

  void Set_TitleFileName(const UString &fileName);
  void Set_Status(const UString &s);
  HRESULT Set_Status2(const UString &s, const wchar_t *path, bool isDir = false);
  void Set_FilePath(const wchar_t *path, bool isDir = false);

  void AddError_Message(const wchar_t *message);
  void AddError_Message_Name(const wchar_t *message, const wchar_t *name);
  // void AddError_Code_Name(DWORD systemError, const wchar_t *name);
  void AddError_Code_Name(HRESULT systemError, const wchar_t *name);

  bool ThereIsMessage() const { return !Messages.IsEmpty() || FinalMessage.ThereIsMessage(); }
};


class CProgressDialog: public NWindows::NControl::CModalDialog
{
  bool _isDir;
  bool _wasCreated;
  bool _needClose;
  bool _errorsWereDisplayed;
  bool _waitCloseByCancelButton;
  bool _cancelWasPressed;
  bool _inCancelMessageBox;
  bool _externalCloseMessageWasReceived;
  bool _background;
public:
  bool WaitMode;
  bool MessagesDisplayed; // = true if user pressed OK on all messages or there are no messages.
  bool CompressingMode;
  bool ShowCompressionInfo;

private:
  unsigned _numPostedMessages;
  unsigned _numAutoSizeMessages;
  unsigned _numMessages;

  UString _titleFileName;
  UString _filePath;
  UString _status;

  UString _background_String;
  UString _backgrounded_String;
  UString _foreground_String;
  UString _pause_String;
  UString _continue_String;
  UString _paused_String;

  int _buttonSizeX;
  int _buttonSizeY;

  UINT_PTR _timer;

  UString _title;

  class CU64ToI32Converter
  {
    unsigned _numShiftBits;
    UInt64 _range;
  public:
    CU64ToI32Converter(): _numShiftBits(0), _range(1) {}
    void Init(UInt64 range)
    {
      _range = range;
      // Windows CE doesn't like big number for ProgressBar.
      for (_numShiftBits = 0; range >= ((UInt32)1 << 15); _numShiftBits++)
        range >>= 1;
    }
    int Count(UInt64 val)
    {
      int res = (int)(val >> _numShiftBits);
      if (val == _range)
        res++;
      return res;
    }
  };
  
  CU64ToI32Converter _progressConv;
  UInt64 _progressBar_Pos;
  UInt64 _progressBar_Range;
  
  NWindows::NControl::CProgressBar m_ProgressBar;
  NWindows::NControl::CListView _messageList;
  
  UStringVector _messageStrings;

  // #ifdef __ITaskbarList3_INTERFACE_DEFINED__
  CMyComPtr<ITaskbarList3> _taskbarList;
  // #endif
  HWND _hwndForTaskbar;

  UInt32 _prevTime;
  UInt64 _elapsedTime;

  UInt64 _prevPercentValue;
  UInt64 _prevElapsedSec;
  UInt64 _prevRemainingSec;

  UInt64 _totalBytes_Prev;
  UInt64 _processed_Prev;
  UInt64 _packed_Prev;
  UInt64 _ratio_Prev;

  UString _filesStr_Prev;
  UString _filesTotStr_Prev;

  unsigned _numReduceSymbols;
  unsigned _prevSpeed_MoveBits;
  UInt64 _prevSpeed;

  // #ifdef __ITaskbarList3_INTERFACE_DEFINED__
  void SetTaskbarProgressState(TBPFLAG tbpFlags)
  {
    if (_taskbarList && _hwndForTaskbar)
      _taskbarList->SetProgressState(_hwndForTaskbar, tbpFlags);
  }
  // #endif
  void SetTaskbarProgressState();

  void UpdateStatInfo(bool showAll);
  void SetProgressRange(UInt64 range);
  void SetProgressPos(UInt64 pos);
  virtual bool OnTimer(WPARAM timerID, LPARAM callback) Z7_override;
  virtual bool OnInit() Z7_override;
  virtual bool OnSize(WPARAM wParam, int xSize, int ySize) Z7_override;
  virtual void OnCancel() Z7_override;
  virtual void OnOK() Z7_override;
  virtual bool OnNotify(UINT /* controlID */, LPNMHDR header) Z7_override;
  void CopyToClipboard();

  NWindows::NSynchronization::CManualResetEvent _createDialogEvent;
  NWindows::NSynchronization::CManualResetEvent _dialogCreatedEvent;
  #ifndef Z7_SFX
  void AddToTitle(LPCWSTR string);
  #endif

  void SetPauseText();
  void SetPriorityText();
  void OnPauseButton();
  void OnPriorityButton();
  bool OnButtonClicked(unsigned buttonID, HWND buttonHWND) Z7_override;
  bool OnMessage(UINT message, WPARAM wParam, LPARAM lParam) Z7_override;

  void SetTitleText();
  void ShowSize(unsigned id, UInt64 val, UInt64 &prev);

  void UpdateMessagesDialog();

  void AddMessageDirect(LPCWSTR message, bool needNumber);
  void AddMessage(LPCWSTR message);

  bool OnExternalCloseMessage();
  void EnableErrorsControls(bool enable);

  void ShowAfterMessages(HWND wndParent);

  void CheckNeedClose();

public:
  CProgressSync Sync;
  int IconID;
  HWND MainWindow;
  #ifndef Z7_SFX
  UString MainTitle;
  UString MainAddTitle;
  ~CProgressDialog() Z7_DESTRUCTOR_override;
  #endif

  CProgressDialog();
  void WaitCreating()
  {
    _createDialogEvent.Set();
    _dialogCreatedEvent.Lock();
  }

  INT_PTR Create(const UString &title, NWindows::CThread &thread, HWND wndParent = NULL);


  /* how it works:
     1) the working thread calls ProcessWasFinished()
        that sends kCloseMessage message to CProgressDialog (GUI) thread
     2) CProgressDialog (GUI) thread receives kCloseMessage message and
        calls ProcessWasFinished_GuiVirt();
        So we can implement ProcessWasFinished_GuiVirt() and show special
        results window in GUI thread with CProgressDialog as parent window
  */

  void ProcessWasFinished();
  virtual void ProcessWasFinished_GuiVirt() {}
};


class CProgressCloser
{
  CProgressDialog *_p;
public:
  CProgressCloser(CProgressDialog &p) : _p(&p) {}
  ~CProgressCloser() { _p->ProcessWasFinished(); }
};


class CProgressThreadVirt: public CProgressDialog
{
protected:
  FStringVector ErrorPaths;
  CProgressFinalMessage FinalMessage;

  // error if any of HRESULT, ErrorMessage, ErrorPath
  virtual HRESULT ProcessVirt() = 0;
public:
  HRESULT Result;
  bool ThreadFinishedOK; // if there is no fatal exception

  void Process();
  void AddErrorPath(const FString &path) { ErrorPaths.Add(path); }

  HRESULT Create(const UString &title, HWND parentWindow = NULL);
  CProgressThreadVirt(): Result(E_FAIL), ThreadFinishedOK(false) {}

  CProgressMessageBoxPair &GetMessagePair(bool isError) { return isError ? FinalMessage.ErrorMessage : FinalMessage.OkMessage; }
};

UString HResultToMessage(HRESULT errorCode);

/*
how it works:

client code inherits CProgressThreadVirt and calls
CProgressThreadVirt::Create()
{
  it creates new thread that calls CProgressThreadVirt::Process();
  it creates modal progress dialog window with ProgressDialog.Create()
}

CProgressThreadVirt::Process()
{
  {
    Result = ProcessVirt(); // virtual function that must implement real work
  }
  if (exceptions) or FinalMessage.ErrorMessage.Message
  {
    set message to ProgressDialog.Sync.FinalMessage.ErrorMessage.Message
  }
  else if (FinalMessage.OkMessage.Message)
  {
    set message to ProgressDialog.Sync.FinalMessage.OkMessage
  }

  PostMsg(kCloseMessage);
}


CProgressDialog::OnExternalCloseMessage()
{
  if (ProgressDialog.Sync.FinalMessage)
  {
    WorkWasFinishedVirt();
    Show (ProgressDialog.Sync.FinalMessage)
    MessagesDisplayed = true;
  }
}

*/

#endif
