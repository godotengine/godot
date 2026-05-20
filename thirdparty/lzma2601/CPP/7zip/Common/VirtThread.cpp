// VirtThread.cpp

#include "StdAfx.h"

#include "VirtThread.h"

static THREAD_FUNC_DECL CoderThread(void *p)
{
  for (;;)
  {
    CVirtThread *t = (CVirtThread *)p;
    t->StartEvent.Lock();
    if (t->Exit)
      return THREAD_FUNC_RET_ZERO;
    t->Execute();
    t->FinishedEvent.Set();
  }
}

WRes CVirtThread::Create()
{
  RINOK_WRes(StartEvent.CreateIfNotCreated_Reset())
  RINOK_WRes(FinishedEvent.CreateIfNotCreated_Reset())
  // StartEvent.Reset();
  // FinishedEvent.Reset();
  Exit = false;
  if (Thread.IsCreated())
    return S_OK;
  return Thread.Create(CoderThread, this);
}

WRes CVirtThread::Start()
{
  Exit = false;
  return StartEvent.Set();
}

void CVirtThread::WaitThreadFinish()
{
  Exit = true;
  if (StartEvent.IsCreated())
    StartEvent.Set();
  if (Thread.IsCreated())
  {
    Thread.Wait_Close();
  }
}
