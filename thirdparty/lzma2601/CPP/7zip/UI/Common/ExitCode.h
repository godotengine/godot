// ExitCode.h

#ifndef ZIP7_INC_EXIT_CODE_H
#define ZIP7_INC_EXIT_CODE_H

namespace NExitCode {

enum EEnum {

  kSuccess       = 0,     // Successful operation
  kWarning       = 1,     // Non fatal error(s) occurred
  kFatalError    = 2,     // A fatal error occurred
  // kCRCError      = 3,     // A CRC error occurred when unpacking
  // kLockedArchive = 4,     // Attempt to modify an archive previously locked
  // kWriteError    = 5,     // Write to disk error
  // kOpenError     = 6,     // Open file error
  kUserError     = 7,     // Command line option error
  kMemoryError   = 8,     // Not enough memory for operation
  // kCreateFileError = 9,     // Create file error
  
  kUserBreak     = 255   // User stopped the process

};

}

#endif
