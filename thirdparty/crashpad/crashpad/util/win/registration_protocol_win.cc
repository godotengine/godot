// Copyright 2015 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/win/registration_protocol_win.h"

#include <stddef.h>
#include <windows.h>

#include "base/logging.h"
#include "base/macros.h"
#include "util/win/exception_handler_server.h"
#include "util/win/scoped_handle.h"

namespace crashpad {

bool SendToCrashHandlerServer(const base::string16& pipe_name,
                              const ClientToServerMessage& message,
                              ServerToClientMessage* response) {
  // Retry CreateFile() in a loop. If the handler isn’t actively waiting in
  // ConnectNamedPipe() on a pipe instance because it’s busy doing something
  // else, CreateFile() will fail with ERROR_PIPE_BUSY. WaitNamedPipe() waits
  // until a pipe instance is ready, but there’s no way to wait for this
  // condition and atomically open the client side of the pipe in a single
  // operation. CallNamedPipe() implements similar retry logic to this, also in
  // user-mode code.
  //
  // This loop is only intended to retry on ERROR_PIPE_BUSY. Notably, if the
  // handler is so lazy that it hasn’t even called CreateNamedPipe() yet,
  // CreateFile() will fail with ERROR_FILE_NOT_FOUND, and this function is
  // expected to fail without retrying anything. If the handler is started at
  // around the same time as its client, something external to this code must be
  // done to guarantee correct ordering. When the client starts the handler
  // itself, CrashpadClient::StartHandler() provides this synchronization.
  for (;;) {
    ScopedFileHANDLE pipe(
        CreateFile(pipe_name.c_str(),
                   GENERIC_READ | GENERIC_WRITE,
                   0,
                   nullptr,
                   OPEN_EXISTING,
                   SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION,
                   nullptr));
    if (!pipe.is_valid()) {
      if (GetLastError() != ERROR_PIPE_BUSY) {
        PLOG(ERROR) << "CreateFile";
        return false;
      }

      if (!WaitNamedPipe(pipe_name.c_str(), NMPWAIT_WAIT_FOREVER)) {
        PLOG(ERROR) << "WaitNamedPipe";
        return false;
      }

      continue;
    }

    DWORD mode = PIPE_READMODE_MESSAGE;
    if (!SetNamedPipeHandleState(pipe.get(), &mode, nullptr, nullptr)) {
      PLOG(ERROR) << "SetNamedPipeHandleState";
      return false;
    }
    DWORD bytes_read = 0;
    BOOL result = TransactNamedPipe(
        pipe.get(),
        // This is [in], but is incorrectly declared non-const.
        const_cast<ClientToServerMessage*>(&message),
        sizeof(message),
        response,
        sizeof(*response),
        &bytes_read,
        nullptr);
    if (!result) {
      PLOG(ERROR) << "TransactNamedPipe";
      return false;
    }
    if (bytes_read != sizeof(*response)) {
      LOG(ERROR) << "TransactNamedPipe: expected " << sizeof(*response)
                 << ", observed " << bytes_read;
      return false;
    }
    return true;
  }
}

HANDLE CreateNamedPipeInstance(const std::wstring& pipe_name,
                               bool first_instance) {
  SECURITY_ATTRIBUTES security_attributes;
  SECURITY_ATTRIBUTES* security_attributes_pointer = nullptr;

  if (first_instance) {
    // Pre-Vista does not have integrity levels.
    const DWORD version = GetVersion();
    const DWORD major_version = LOBYTE(LOWORD(version));
    const bool is_vista_or_later = major_version >= 6;
    if (is_vista_or_later) {
      memset(&security_attributes, 0, sizeof(security_attributes));
      security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
      security_attributes.lpSecurityDescriptor =
          const_cast<void*>(GetSecurityDescriptorForNamedPipeInstance(nullptr));
      security_attributes.bInheritHandle = TRUE;
      security_attributes_pointer = &security_attributes;
    }
  }

  return CreateNamedPipe(
      pipe_name.c_str(),
      PIPE_ACCESS_DUPLEX | (first_instance ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0),
      PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
      ExceptionHandlerServer::kPipeInstances,
      512,
      512,
      0,
      security_attributes_pointer);
}

const void* GetSecurityDescriptorForNamedPipeInstance(size_t* size) {
  // Mandatory Label, no ACE flags, no ObjectType, integrity level untrusted is
  // "S:(ML;;;;;S-1-16-0)". Typically
  // ConvertStringSecurityDescriptorToSecurityDescriptor() would be used to
  // convert from a string representation. However, that function cannot be used
  // because it is in advapi32.dll and CreateNamedPipeInstance() is called from
  // within DllMain() where the loader lock is held. advapi32.dll is delay
  // loaded in chrome_elf.dll because it must avoid loading user32.dll. If an
  // advapi32.dll function were used, it would cause a load of the DLL, which
  // would in turn cause deadlock.

#pragma pack(push, 1)
  static constexpr struct SecurityDescriptorBlob {
    // See https://msdn.microsoft.com/library/cc230366.aspx.
    SECURITY_DESCRIPTOR_RELATIVE sd_rel;
    struct {
      ACL acl;
      struct {
        // This is equivalent to SYSTEM_MANDATORY_LABEL_ACE, but there's no
        // DWORD offset to the SID, instead it's inline.
        ACE_HEADER header;
        ACCESS_MASK mask;
        SID sid;
      } ace[1];
    } sacl;
  } kSecDescBlob = {
      // sd_rel.
      {
          SECURITY_DESCRIPTOR_REVISION1,  // Revision.
          0x00,  // Sbz1.
          SE_SELF_RELATIVE | SE_SACL_PRESENT,  // Control.
          0,  // OffsetOwner.
          0,  // OffsetGroup.
          offsetof(SecurityDescriptorBlob, sacl),  // OffsetSacl.
          0,  // OffsetDacl.
      },

      // sacl.
      {
          // acl.
          {
              ACL_REVISION,  // AclRevision.
              0,  // Sbz1.
              sizeof(kSecDescBlob.sacl),  // AclSize.
              arraysize(kSecDescBlob.sacl.ace),  // AceCount.
              0,  // Sbz2.
          },

          // ace[0].
          {
              {
                  // header.
                  {
                      SYSTEM_MANDATORY_LABEL_ACE_TYPE,  // AceType.
                      0,  // AceFlags.
                      sizeof(kSecDescBlob.sacl.ace[0]),  // AceSize.
                  },

                  // mask.
                  0,

                  // sid.
                  {
                      SID_REVISION,  // Revision.
                      // SubAuthorityCount.
                      arraysize(kSecDescBlob.sacl.ace[0].sid.SubAuthority),
                      // IdentifierAuthority.
                      {SECURITY_MANDATORY_LABEL_AUTHORITY},
                      {SECURITY_MANDATORY_UNTRUSTED_RID},  // SubAuthority.
                  },
              },
          },
      },
  };
#pragma pack(pop)

  if (size)
    *size = sizeof(kSecDescBlob);
  return reinterpret_cast<const void*>(&kSecDescBlob);
}

}  // namespace crashpad
