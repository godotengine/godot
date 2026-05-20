// Windows/Synchronization.cpp

#include "StdAfx.h"

#ifndef _WIN32

#include "Synchronization.h"

namespace NWindows {
namespace NSynchronization {

/*
#define INFINITE  0xFFFFFFFF
#define MAXIMUM_WAIT_OBJECTS 64
#define STATUS_ABANDONED_WAIT_0 ((NTSTATUS)0x00000080L)
#define WAIT_ABANDONED   ((STATUS_ABANDONED_WAIT_0 ) + 0 )
#define WAIT_ABANDONED_0 ((STATUS_ABANDONED_WAIT_0 ) + 0 )
// WINAPI
DWORD WaitForMultipleObjects(DWORD count, const HANDLE *handles, BOOL wait_all, DWORD timeout);
*/

/* clang: we need to place some virtual functions in cpp file to rid off the warning:
   'CBaseHandle_WFMO' has no out-of-line virtual method definitions;
   its vtable will be emitted in every translation unit */
CBaseHandle_WFMO::~CBaseHandle_WFMO()
{
}

bool CBaseEvent_WFMO::IsSignaledAndUpdate()
{
  if (this->_state == false)
    return false;
  if (this->_manual_reset == false)
    this->_state = false;
  return true;
}

bool CSemaphore_WFMO::IsSignaledAndUpdate()
{
  if (this->_count == 0)
    return false;
  this->_count--;
  return true;
}

DWORD WINAPI WaitForMultiObj_Any_Infinite(DWORD count, const CHandle_WFMO *handles)
{
  if (count < 1)
  {
    // abort();
    SetLastError(EINVAL);
    return WAIT_FAILED;
  }

  CSynchro *synchro = handles[0]->_sync;
  synchro->Enter();
  
  // #ifdef DEBUG_SYNCHRO
  for (DWORD i = 1; i < count; i++)
  {
    if (synchro != handles[i]->_sync)
    {
      // abort();
      synchro->Leave();
      SetLastError(EINVAL);
      return WAIT_FAILED;
    }
  }
  // #endif

  for (;;)
  {
    for (DWORD i = 0; i < count; i++)
    {
      if (handles[i]->IsSignaledAndUpdate())
      {
        synchro->Leave();
        return WAIT_OBJECT_0 + i;
      }
    }
    synchro->WaitCond();
  }
}

}}

#endif
