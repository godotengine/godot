// Windows/MemoryLock.h

#ifndef ZIP7_INC_WINDOWS_MEMORY_LOCK_H
#define ZIP7_INC_WINDOWS_MEMORY_LOCK_H

#include "../Common/MyWindows.h"

namespace NWindows {
namespace NSecurity {

#ifndef UNDER_CE

bool EnablePrivilege(LPCTSTR privilegeName, bool enable = true);

inline bool EnablePrivilege_LockMemory(bool enable = true)
{
  return EnablePrivilege(SE_LOCK_MEMORY_NAME, enable);
}

inline void EnablePrivilege_SymLink()
{
  /* Probably we do not to set any Privilege for junction points.
     But we need them for Symbolic links */
  NSecurity::EnablePrivilege(SE_RESTORE_NAME);
  
  /* Probably we need only SE_RESTORE_NAME, but there is also
     SE_CREATE_SYMBOLIC_LINK_NAME. So we set it also. Do we need it? */

  NSecurity::EnablePrivilege(TEXT("SeCreateSymbolicLinkPrivilege")); // SE_CREATE_SYMBOLIC_LINK_NAME
  
  // Do we need to set SE_BACKUP_NAME ?
}

unsigned Get_LargePages_RiskLevel();

#endif

}}

#endif
