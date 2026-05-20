// VirtThread.h

#ifndef ZIP7_INC_VIRT_THREAD_H
#define ZIP7_INC_VIRT_THREAD_H

#include "../../Windows/Synchronization.h"
#include "../../Windows/Thread.h"

struct CVirtThread
{
  NWindows::NSynchronization::CAutoResetEvent StartEvent;
  NWindows::NSynchronization::CAutoResetEvent FinishedEvent;
  NWindows::CThread Thread;
  bool Exit;

  virtual ~CVirtThread() { WaitThreadFinish(); }
  void WaitThreadFinish(); // call it in destructor of child class !
  WRes Create();
  WRes Start();
  virtual void Execute() = 0;
  WRes WaitExecuteFinish() { return FinishedEvent.Lock(); }
};

#endif
