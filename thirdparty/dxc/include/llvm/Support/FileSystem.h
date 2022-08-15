//===- llvm/Support/FileSystem.h - File System OS Concept -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::fs namespace. It is designed after
// TR2/boost filesystem (v3), but modified to remove exception handling and the
// path class.
//
// All functions return an error_code and their actual work via the last out
// argument. The out argument is defined if and only if errc::success is
// returned. A function may return any error code in the generic or system
// category. However, they shall be equivalent to any error conditions listed
// in each functions respective documentation if the condition applies. [ note:
// this does not guarantee that error_code will be in the set of explicitly
// listed codes, but it does guarantee that if any of the explicitly listed
// errors occur, the correct error_code will be used ]. All functions may
// return errc::not_enough_memory if there is not enough memory to complete the
// operation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILESYSTEM_H
#define LLVM_SUPPORT_FILESYSTEM_H

#include "dxc/Support/WinAdapter.h" // HLSL Change
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeValue.h"
#include <ctime>
#include <iterator>
#include <stack>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

namespace llvm {
namespace sys {
namespace fs {

// HLSL Change Start
class MSFileSystem;
typedef _Inout_ MSFileSystem* MSFileSystemRef;

std::error_code GetFileSystemTlsStatus() throw();

std::error_code SetupPerThreadFileSystem() throw();
void CleanupPerThreadFileSystem() throw();
struct AutoCleanupPerThreadFileSystem { ~AutoCleanupPerThreadFileSystem() { CleanupPerThreadFileSystem(); } };

/// <summary>Gets a reference to the file system installed for the current thread (possibly NULL).</summary>
/// <remarks>In practice, consumers of the library should always install a file system.</remarks>
MSFileSystemRef GetCurrentThreadFileSystem() throw();

/// <summary>
/// Sets the specified file system reference for the current thread.
/// </summary>
std::error_code SetCurrentThreadFileSystem(MSFileSystemRef value) throw();

int msf_read(int fd, void* buffer, unsigned int count) throw();
int msf_write(int fd, const void* buffer, unsigned int count) throw();
int msf_close(int fd) throw();
int msf_setmode(int fd, int mode) throw();
long msf_lseek(int fd, long offset, int origin);

class AutoPerThreadSystem
{
private:
  ::llvm::sys::fs::MSFileSystem* m_pOrigValue;
  std::error_code ec;
public:
  AutoPerThreadSystem(_In_ ::llvm::sys::fs::MSFileSystem *value)
      : m_pOrigValue(::llvm::sys::fs::GetCurrentThreadFileSystem()) {
    SetCurrentThreadFileSystem(nullptr);
    ec = ::llvm::sys::fs::SetCurrentThreadFileSystem(value);
  }

  ~AutoPerThreadSystem()
  {
    if (m_pOrigValue) {
      ::llvm::sys::fs::SetCurrentThreadFileSystem(nullptr);
      ::llvm::sys::fs::SetCurrentThreadFileSystem(m_pOrigValue);
    } else if (!ec) {
      ::llvm::sys::fs::SetCurrentThreadFileSystem(nullptr);
    }
  }

  const std::error_code& error_code() const { return ec; }
};

// HLSL Change Ends

/// An enumeration for the file system's view of the type.
enum class file_type {
  status_error,
  file_not_found,
  regular_file,
  directory_file,
  symlink_file,
  block_file,
  character_file,
  fifo_file,
  socket_file,
  type_unknown
};

/// space_info - Self explanatory.
struct space_info {
  uint64_t capacity;
  uint64_t free;
  uint64_t available;
};

enum perms {
  no_perms = 0,
  owner_read = 0400,
  owner_write = 0200,
  owner_exe = 0100,
  owner_all = owner_read | owner_write | owner_exe,
  group_read = 040,
  group_write = 020,
  group_exe = 010,
  group_all = group_read | group_write | group_exe,
  others_read = 04,
  others_write = 02,
  others_exe = 01,
  others_all = others_read | others_write | others_exe,
  all_read = owner_read | group_read | others_read,
  all_write = owner_write | group_write | others_write,
  all_exe = owner_exe | group_exe | others_exe,
  all_all = owner_all | group_all | others_all,
  set_uid_on_exe = 04000,
  set_gid_on_exe = 02000,
  sticky_bit = 01000,
  perms_not_known = 0xFFFF
};

// Helper functions so that you can use & and | to manipulate perms bits:
inline perms operator|(perms l , perms r) {
  return static_cast<perms>(
             static_cast<unsigned short>(l) | static_cast<unsigned short>(r)); 
}
inline perms operator&(perms l , perms r) {
  return static_cast<perms>(
             static_cast<unsigned short>(l) & static_cast<unsigned short>(r)); 
}
inline perms &operator|=(perms &l, perms r) {
  l = l | r; 
  return l; 
}
inline perms &operator&=(perms &l, perms r) {
  l = l & r; 
  return l; 
}
inline perms operator~(perms x) {
  return static_cast<perms>(~static_cast<unsigned short>(x));
}

class UniqueID {
  uint64_t Device;
  uint64_t File;

public:
  UniqueID() = default;
  UniqueID(uint64_t Device, uint64_t File) : Device(Device), File(File) {}
  bool operator==(const UniqueID &Other) const {
    return Device == Other.Device && File == Other.File;
  }
  bool operator!=(const UniqueID &Other) const { return !(*this == Other); }
  bool operator<(const UniqueID &Other) const {
    return std::tie(Device, File) < std::tie(Other.Device, Other.File);
  }
  uint64_t getDevice() const { return Device; }
  uint64_t getFile() const { return File; }
};

/// file_status - Represents the result of a call to stat and friends. It has
///               a platform-specific member to store the result.
class file_status
{
  #if defined(LLVM_ON_UNIX)
  dev_t fs_st_dev;
  ino_t fs_st_ino;
  time_t fs_st_mtime;
  uid_t fs_st_uid;
  gid_t fs_st_gid;
  off_t fs_st_size;
  #elif defined (LLVM_ON_WIN32)
  uint32_t LastWriteTimeHigh;
  uint32_t LastWriteTimeLow;
  uint32_t VolumeSerialNumber;
  uint32_t FileSizeHigh;
  uint32_t FileSizeLow;
  uint32_t FileIndexHigh;
  uint32_t FileIndexLow;
  #endif
  friend bool equivalent(file_status A, file_status B);
  file_type Type;
  perms Perms;
public:
  #if defined(LLVM_ON_UNIX)
    file_status() : fs_st_dev(0), fs_st_ino(0), fs_st_mtime(0),
        fs_st_uid(0), fs_st_gid(0), fs_st_size(0),
        Type(file_type::status_error), Perms(perms_not_known) {}

    file_status(file_type Type) : fs_st_dev(0), fs_st_ino(0), fs_st_mtime(0),
        fs_st_uid(0), fs_st_gid(0), fs_st_size(0), Type(Type),
        Perms(perms_not_known) {}

    file_status(file_type Type, perms Perms, dev_t Dev, ino_t Ino, time_t MTime,
                uid_t UID, gid_t GID, off_t Size)
        : fs_st_dev(Dev), fs_st_ino(Ino), fs_st_mtime(MTime), fs_st_uid(UID),
          fs_st_gid(GID), fs_st_size(Size), Type(Type), Perms(Perms) {}
  #elif defined(LLVM_ON_WIN32)
    file_status() : LastWriteTimeHigh(0), LastWriteTimeLow(0),
        VolumeSerialNumber(0), FileSizeHigh(0), FileSizeLow(0),
        FileIndexHigh(0), FileIndexLow(0), Type(file_type::status_error),
        Perms(perms_not_known) {}

    file_status(file_type Type) : LastWriteTimeHigh(0), LastWriteTimeLow(0),
        VolumeSerialNumber(0), FileSizeHigh(0), FileSizeLow(0),
        FileIndexHigh(0), FileIndexLow(0), Type(Type),
        Perms(perms_not_known) {}

    file_status(file_type Type, uint32_t LastWriteTimeHigh,
                uint32_t LastWriteTimeLow, uint32_t VolumeSerialNumber,
                uint32_t FileSizeHigh, uint32_t FileSizeLow,
                uint32_t FileIndexHigh, uint32_t FileIndexLow)
        : LastWriteTimeHigh(LastWriteTimeHigh),
          LastWriteTimeLow(LastWriteTimeLow),
          VolumeSerialNumber(VolumeSerialNumber), FileSizeHigh(FileSizeHigh),
          FileSizeLow(FileSizeLow), FileIndexHigh(FileIndexHigh),
          FileIndexLow(FileIndexLow), Type(Type), Perms(perms_not_known) {}
  #endif

  // getters
  file_type type() const { return Type; }
  perms permissions() const { return Perms; }
  TimeValue getLastModificationTime() const;
  UniqueID getUniqueID() const;

  #if defined(LLVM_ON_UNIX)
  uint32_t getUser() const { return fs_st_uid; }
  uint32_t getGroup() const { return fs_st_gid; }
  uint64_t getSize() const { return fs_st_size; }
  #elif defined (LLVM_ON_WIN32)
  uint32_t getUser() const {
    return 9999; // Not applicable to Windows, so...
  }
  uint32_t getGroup() const {
    return 9999; // Not applicable to Windows, so...
  }
  uint64_t getSize() const {
    return (uint64_t(FileSizeHigh) << 32) + FileSizeLow;
  }
  #endif

  // setters
  void type(file_type v) { Type = v; }
  void permissions(perms p) { Perms = p; }
};

/// file_magic - An "enum class" enumeration of file types based on magic (the first
///         N bytes of the file).
struct file_magic {
  enum Impl {
    unknown = 0,              ///< Unrecognized file
    bitcode,                  ///< Bitcode file
    archive,                  ///< ar style archive file
    elf,                      ///< ELF Unknown type
    elf_relocatable,          ///< ELF Relocatable object file
    elf_executable,           ///< ELF Executable image
    elf_shared_object,        ///< ELF dynamically linked shared lib
    elf_core,                 ///< ELF core image
    macho_object,             ///< Mach-O Object file
    macho_executable,         ///< Mach-O Executable
    macho_fixed_virtual_memory_shared_lib, ///< Mach-O Shared Lib, FVM
    macho_core,               ///< Mach-O Core File
    macho_preload_executable, ///< Mach-O Preloaded Executable
    macho_dynamically_linked_shared_lib, ///< Mach-O dynlinked shared lib
    macho_dynamic_linker,     ///< The Mach-O dynamic linker
    macho_bundle,             ///< Mach-O Bundle file
    macho_dynamically_linked_shared_lib_stub, ///< Mach-O Shared lib stub
    macho_dsym_companion,     ///< Mach-O dSYM companion file
    macho_kext_bundle,        ///< Mach-O kext bundle file
    macho_universal_binary,   ///< Mach-O universal binary
    coff_object,              ///< COFF object file
    coff_import_library,      ///< COFF import library
    pecoff_executable,        ///< PECOFF executable file
    windows_resource          ///< Windows compiled resource file (.rc)
  };

  bool is_object() const {
    return V == unknown ? false : true;
  }

  file_magic() : V(unknown) {}
  file_magic(Impl V) : V(V) {}
  operator Impl() const { return V; }

private:
  Impl V;
};

/// @}
/// @name Physical Operators
/// @{

/// @brief Make \a path an absolute path.
///
/// Makes \a path absolute using the current directory if it is not already. An
/// empty \a path will result in the current directory.
///
/// /absolute/path   => /absolute/path
/// relative/../path => <current-directory>/relative/../path
///
/// @param path A path that is modified to be an absolute path.
/// @returns errc::success if \a path has been made absolute, otherwise a
///          platform-specific error_code.
std::error_code make_absolute(SmallVectorImpl<char> &path);

/// @brief Create all the non-existent directories in path.
///
/// @param path Directories to create.
/// @returns errc::success if is_directory(path), otherwise a platform
///          specific error_code. If IgnoreExisting is false, also returns
///          error if the directory already existed.
std::error_code create_directories(const Twine &path,
                                   bool IgnoreExisting = true);

/// @brief Create the directory in path.
///
/// @param path Directory to create.
/// @returns errc::success if is_directory(path), otherwise a platform
///          specific error_code. If IgnoreExisting is false, also returns
///          error if the directory already existed.
std::error_code create_directory(const Twine &path, bool IgnoreExisting = true);

/// @brief Create a link from \a from to \a to.
///
/// The link may be a soft or a hard link, depending on the platform. The caller
/// may not assume which one. Currently on windows it creates a hard link since
/// soft links require extra privileges. On unix, it creates a soft link since
/// hard links don't work on SMB file systems.
///
/// @param to The path to hard link to.
/// @param from The path to hard link from. This is created.
/// @returns errc::success if the link was created, otherwise a platform
/// specific error_code.
std::error_code create_link(const Twine &to, const Twine &from);

/// @brief Get the current path.
///
/// @param result Holds the current path on return.
/// @returns errc::success if the current path has been stored in result,
///          otherwise a platform-specific error_code.
std::error_code current_path(SmallVectorImpl<char> &result);

/// @brief Remove path. Equivalent to POSIX remove().
///
/// @param path Input path.
/// @returns errc::success if path has been removed or didn't exist, otherwise a
///          platform-specific error code. If IgnoreNonExisting is false, also
///          returns error if the file didn't exist.
std::error_code remove(const Twine &path, bool IgnoreNonExisting = true);

/// @brief Rename \a from to \a to. Files are renamed as if by POSIX rename().
///
/// @param from The path to rename from.
/// @param to The path to rename to. This is created.
std::error_code rename(const Twine &from, const Twine &to);

/// @brief Copy the contents of \a From to \a To.
///
/// @param From The path to copy from.
/// @param To The path to copy to. This is created.
std::error_code copy_file(const Twine &From, const Twine &To);

/// @brief Resize path to size. File is resized as if by POSIX truncate().
///
/// @param FD Input file descriptor.
/// @param Size Size to resize to.
/// @returns errc::success if \a path has been resized to \a size, otherwise a
///          platform-specific error_code.
std::error_code resize_file(int FD, uint64_t Size);

/// @}
/// @name Physical Observers
/// @{

/// @brief Does file exist?
///
/// @param status A file_status previously returned from stat.
/// @returns True if the file represented by status exists, false if it does
///          not.
bool exists(file_status status);

enum class AccessMode { Exist, Write, Execute };

/// @brief Can the file be accessed?
///
/// @param Path Input path.
/// @returns errc::success if the path can be accessed, otherwise a
///          platform-specific error_code.
std::error_code access(const Twine &Path, AccessMode Mode);

/// @brief Does file exist?
///
/// @param Path Input path.
/// @returns True if it exists, false otherwise.
inline bool exists(const Twine &Path) {
  return !access(Path, AccessMode::Exist);
}

/// @brief Can we execute this file?
///
/// @param Path Input path.
/// @returns True if we can execute it, false otherwise.
inline bool can_execute(const Twine &Path) {
  return !access(Path, AccessMode::Execute);
}

/// @brief Can we write this file?
///
/// @param Path Input path.
/// @returns True if we can write to it, false otherwise.
inline bool can_write(const Twine &Path) {
  return !access(Path, AccessMode::Write);
}

/// @brief Do file_status's represent the same thing?
///
/// @param A Input file_status.
/// @param B Input file_status.
///
/// assert(status_known(A) || status_known(B));
///
/// @returns True if A and B both represent the same file system entity, false
///          otherwise.
bool equivalent(file_status A, file_status B);

/// @brief Do paths represent the same thing?
///
/// assert(status_known(A) || status_known(B));
///
/// @param A Input path A.
/// @param B Input path B.
/// @param result Set to true if stat(A) and stat(B) have the same device and
///               inode (or equivalent).
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code equivalent(const Twine &A, const Twine &B, bool &result);

/// @brief Simpler version of equivalent for clients that don't need to
///        differentiate between an error and false.
inline bool equivalent(const Twine &A, const Twine &B) {
  bool result;
  return !equivalent(A, B, result) && result;
}

/// @brief Does status represent a directory?
///
/// @param status A file_status previously returned from status.
/// @returns status.type() == file_type::directory_file.
bool is_directory(file_status status);

/// @brief Is path a directory?
///
/// @param path Input path.
/// @param result Set to true if \a path is a directory, false if it is not.
///               Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code is_directory(const Twine &path, bool &result);

/// @brief Simpler version of is_directory for clients that don't need to
///        differentiate between an error and false.
inline bool is_directory(const Twine &Path) {
  bool Result;
  return !is_directory(Path, Result) && Result;
}

/// @brief Does status represent a regular file?
///
/// @param status A file_status previously returned from status.
/// @returns status_known(status) && status.type() == file_type::regular_file.
bool is_regular_file(file_status status);

/// @brief Is path a regular file?
///
/// @param path Input path.
/// @param result Set to true if \a path is a regular file, false if it is not.
///               Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code is_regular_file(const Twine &path, bool &result);

/// @brief Simpler version of is_regular_file for clients that don't need to
///        differentiate between an error and false.
inline bool is_regular_file(const Twine &Path) {
  bool Result;
  if (is_regular_file(Path, Result))
    return false;
  return Result;
}

/// @brief Does this status represent something that exists but is not a
///        directory, regular file, or symlink?
///
/// @param status A file_status previously returned from status.
/// @returns exists(s) && !is_regular_file(s) && !is_directory(s)
bool is_other(file_status status);

/// @brief Is path something that exists but is not a directory,
///        regular file, or symlink?
///
/// @param path Input path.
/// @param result Set to true if \a path exists, but is not a directory, regular
///               file, or a symlink, false if it does not. Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code is_other(const Twine &path, bool &result);

/// @brief Get file status as if by POSIX stat().
///
/// @param path Input path.
/// @param result Set to the file status.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code status(const Twine &path, file_status &result);

/// @brief A version for when a file descriptor is already available.
std::error_code status(int FD, file_status &Result);

/// @brief Get file size.
///
/// @param Path Input path.
/// @param Result Set to the size of the file in \a Path.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
inline std::error_code file_size(const Twine &Path, uint64_t &Result) {
  file_status Status;
  std::error_code EC = status(Path, Status);
  if (EC)
    return EC;
  Result = Status.getSize();
  return std::error_code();
}

/// @brief Set the file modification and access time.
///
/// @returns errc::success if the file times were successfully set, otherwise a
///          platform-specific error_code or errc::function_not_supported on
///          platforms where the functionality isn't available.
std::error_code setLastModificationAndAccessTime(int FD, TimeValue Time);

/// @brief Is status available?
///
/// @param s Input file status.
/// @returns True if status() != status_error.
bool status_known(file_status s);

/// @brief Is status available?
///
/// @param path Input path.
/// @param result Set to true if status() != status_error.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code status_known(const Twine &path, bool &result);

/// @brief Create a uniquely named file.
///
/// Generates a unique path suitable for a temporary file and then opens it as a
/// file. The name is based on \a model with '%' replaced by a random char in
/// [0-9a-f]. If \a model is not an absolute path, a suitable temporary
/// directory will be prepended.
///
/// Example: clang-%%-%%-%%-%%-%%.s => clang-a0-b1-c2-d3-e4.s
///
/// This is an atomic operation. Either the file is created and opened, or the
/// file system is left untouched.
///
/// The intendend use is for files that are to be kept, possibly after
/// renaming them. For example, when running 'clang -c foo.o', the file can
/// be first created as foo-abc123.o and then renamed.
///
/// @param Model Name to base unique path off of.
/// @param ResultFD Set to the opened file's file descriptor.
/// @param ResultPath Set to the opened file's absolute path.
/// @returns errc::success if Result{FD,Path} have been successfully set,
///          otherwise a platform-specific error_code.
std::error_code createUniqueFile(const Twine &Model, int &ResultFD,
                                 SmallVectorImpl<char> &ResultPath,
                                 unsigned Mode = all_read | all_write);

/// @brief Simpler version for clients that don't want an open file.
std::error_code createUniqueFile(const Twine &Model,
                                 SmallVectorImpl<char> &ResultPath);

/// @brief Create a file in the system temporary directory.
///
/// The filename is of the form prefix-random_chars.suffix. Since the directory
/// is not know to the caller, Prefix and Suffix cannot have path separators.
/// The files are created with mode 0600.
///
/// This should be used for things like a temporary .s that is removed after
/// running the assembler.
std::error_code createTemporaryFile(const Twine &Prefix, StringRef Suffix,
                                    int &ResultFD,
                                    SmallVectorImpl<char> &ResultPath);

/// @brief Simpler version for clients that don't want an open file.
std::error_code createTemporaryFile(const Twine &Prefix, StringRef Suffix,
                                    SmallVectorImpl<char> &ResultPath);

std::error_code createUniqueDirectory(const Twine &Prefix,
                                      SmallVectorImpl<char> &ResultPath);

enum OpenFlags : unsigned {
  F_None = 0,

  /// F_Excl - When opening a file, this flag makes raw_fd_ostream
  /// report an error if the file already exists.
  F_Excl = 1,

  /// F_Append - When opening a file, if it already exists append to the
  /// existing file instead of returning an error.  This may not be specified
  /// with F_Excl.
  F_Append = 2,

  /// The file should be opened in text mode on platforms that make this
  /// distinction.
  F_Text = 4,

  /// Open the file for read and write.
  F_RW = 8
};

inline OpenFlags operator|(OpenFlags A, OpenFlags B) {
  return OpenFlags(unsigned(A) | unsigned(B));
}

inline OpenFlags &operator|=(OpenFlags &A, OpenFlags B) {
  A = A | B;
  return A;
}

std::error_code openFileForWrite(const Twine &Name, int &ResultFD,
                                 OpenFlags Flags, unsigned Mode = 0666);

std::error_code openFileForRead(const Twine &Name, int &ResultFD);

/// @brief Identify the type of a binary file based on how magical it is.
file_magic identify_magic(StringRef magic);

/// @brief Get and identify \a path's type based on its content.
///
/// @param path Input path.
/// @param result Set to the type of file, or file_magic::unknown.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform-specific error_code.
std::error_code identify_magic(const Twine &path, file_magic &result);

std::error_code getUniqueID(const Twine Path, UniqueID &Result);

/// This class represents a memory mapped file. It is based on
/// boost::iostreams::mapped_file.
class mapped_file_region {
  mapped_file_region() = delete;
  mapped_file_region(mapped_file_region&) = delete;
  mapped_file_region &operator =(mapped_file_region&) = delete;

public:
  enum mapmode {
    readonly, ///< May only access map via const_data as read only.
    readwrite, ///< May access map via data and modify it. Written to path.
    priv ///< May modify via data, but changes are lost on destruction.
  };

private:
  /// Platform-specific mapping state.
  uint64_t Size;
  void *Mapping;

  std::error_code init(int FD, uint64_t Offset, mapmode Mode);

public:
  /// \param fd An open file descriptor to map. mapped_file_region takes
  ///   ownership if closefd is true. It must have been opended in the correct
  ///   mode.
  mapped_file_region(int fd, mapmode mode, uint64_t length, uint64_t offset,
                     std::error_code &ec);

  ~mapped_file_region();

  uint64_t size() const;
  char *data() const;

  /// Get a const view of the data. Modifying this memory has undefined
  /// behavior.
  const char *const_data() const;

  /// \returns The minimum alignment offset must be.
  static int alignment();
};

/// Return the path to the main executable, given the value of argv[0] from
/// program startup and the address of main itself. In extremis, this function
/// may fail and return an empty path.
std::string getMainExecutable(const char *argv0, void *MainExecAddr);

/// @}
/// @name Iterators
/// @{

/// directory_entry - A single entry in a directory. Caches the status either
/// from the result of the iteration syscall, or the first time status is
/// called.
class directory_entry {
  std::string Path;
  mutable file_status Status;

public:
  explicit directory_entry(const Twine &path, file_status st = file_status())
    : Path(path.str())
    , Status(st) {}

  directory_entry() {}

  void assign(const Twine &path, file_status st = file_status()) {
    Path = path.str();
    Status = st;
  }

  void replace_filename(const Twine &filename, file_status st = file_status());

  const std::string &path() const { return Path; }
  std::error_code status(file_status &result) const;

  bool operator==(const directory_entry& rhs) const { return Path == rhs.Path; }
  bool operator!=(const directory_entry& rhs) const { return !(*this == rhs); }
  bool operator< (const directory_entry& rhs) const;
  bool operator<=(const directory_entry& rhs) const;
  bool operator> (const directory_entry& rhs) const;
  bool operator>=(const directory_entry& rhs) const;
};

namespace detail {
  struct DirIterState;

  std::error_code directory_iterator_construct(DirIterState &, StringRef);
  std::error_code directory_iterator_increment(DirIterState &);
  std::error_code directory_iterator_destruct(DirIterState &);

  /// DirIterState - Keeps state for the directory_iterator. It is reference
  /// counted in order to preserve InputIterator semantics on copy.
  struct DirIterState : public RefCountedBase<DirIterState> {
    DirIterState()
      : IterationHandle(0) {}

    ~DirIterState() {
      directory_iterator_destruct(*this);
    }

    intptr_t IterationHandle;
    directory_entry CurrentEntry;
  };
}

/// directory_iterator - Iterates through the entries in path. There is no
/// operator++ because we need an error_code. If it's really needed we can make
/// it call report_fatal_error on error.
class directory_iterator {
  IntrusiveRefCntPtr<detail::DirIterState> State;

public:
  explicit directory_iterator(const Twine &path, std::error_code &ec) {
    State = new detail::DirIterState;
    SmallString<128> path_storage;
    ec = detail::directory_iterator_construct(*State,
            path.toStringRef(path_storage));
  }

  explicit directory_iterator(const directory_entry &de, std::error_code &ec) {
    State = new detail::DirIterState;
    ec = detail::directory_iterator_construct(*State, de.path());
  }

  /// Construct end iterator.
  directory_iterator() : State(nullptr) {}

  // No operator++ because we need error_code.
  directory_iterator &increment(std::error_code &ec) {
    ec = directory_iterator_increment(*State);
    return *this;
  }

  const directory_entry &operator*() const { return State->CurrentEntry; }
  const directory_entry *operator->() const { return &State->CurrentEntry; }

  bool operator==(const directory_iterator &RHS) const {
    if (State == RHS.State)
      return true;
    if (!RHS.State)
      return State->CurrentEntry == directory_entry();
    if (!State)
      return RHS.State->CurrentEntry == directory_entry();
    return State->CurrentEntry == RHS.State->CurrentEntry;
  }

  bool operator!=(const directory_iterator &RHS) const {
    return !(*this == RHS);
  }
  // Other members as required by
  // C++ Std, 24.1.1 Input iterators [input.iterators]
};

namespace detail {
  /// RecDirIterState - Keeps state for the recursive_directory_iterator. It is
  /// reference counted in order to preserve InputIterator semantics on copy.
  struct RecDirIterState : public RefCountedBase<RecDirIterState> {
    RecDirIterState()
      : Level(0)
      , HasNoPushRequest(false) {}

    std::stack<directory_iterator, std::vector<directory_iterator> > Stack;
    uint16_t Level;
    bool HasNoPushRequest;
  };
}

/// recursive_directory_iterator - Same as directory_iterator except for it
/// recurses down into child directories.
class recursive_directory_iterator {
  IntrusiveRefCntPtr<detail::RecDirIterState> State;

public:
  recursive_directory_iterator() {}
  explicit recursive_directory_iterator(const Twine &path, std::error_code &ec)
      : State(new detail::RecDirIterState) {
    State->Stack.push(directory_iterator(path, ec));
    if (State->Stack.top() == directory_iterator())
      State.reset();
  }
  // No operator++ because we need error_code.
  recursive_directory_iterator &increment(std::error_code &ec) {
    const directory_iterator end_itr;

    if (State->HasNoPushRequest)
      State->HasNoPushRequest = false;
    else {
      file_status st;
      if ((ec = State->Stack.top()->status(st))) return *this;
      if (is_directory(st)) {
        State->Stack.push(directory_iterator(*State->Stack.top(), ec));
        if (ec) return *this;
        if (State->Stack.top() != end_itr) {
          ++State->Level;
          return *this;
        }
        State->Stack.pop();
      }
    }

    while (!State->Stack.empty()
           && State->Stack.top().increment(ec) == end_itr) {
      State->Stack.pop();
      --State->Level;
    }

    // Check if we are done. If so, create an end iterator.
    if (State->Stack.empty())
      State.reset();

    return *this;
  }

  const directory_entry &operator*() const { return *State->Stack.top(); }
  const directory_entry *operator->() const { return &*State->Stack.top(); }

  // observers
  /// Gets the current level. Starting path is at level 0.
  int level() const { return State->Level; }

  /// Returns true if no_push has been called for this directory_entry.
  bool no_push_request() const { return State->HasNoPushRequest; }

  // modifiers
  /// Goes up one level if Level > 0.
  void pop() {
    assert(State && "Cannot pop an end iterator!");
    assert(State->Level > 0 && "Cannot pop an iterator with level < 1");

    const directory_iterator end_itr;
    std::error_code ec;
    do {
      if (ec)
        report_fatal_error("Error incrementing directory iterator.");
      State->Stack.pop();
      --State->Level;
    } while (!State->Stack.empty()
             && State->Stack.top().increment(ec) == end_itr);

    // Check if we are done. If so, create an end iterator.
    if (State->Stack.empty())
      State.reset();
  }

  /// Does not go down into the current directory_entry.
  void no_push() { State->HasNoPushRequest = true; }

  bool operator==(const recursive_directory_iterator &RHS) const {
    return State == RHS.State;
  }

  bool operator!=(const recursive_directory_iterator &RHS) const {
    return !(*this == RHS);
  }
  // Other members as required by
  // C++ Std, 24.1.1 Input iterators [input.iterators]
};

/// @}

} // end namespace fs
} // end namespace sys
} // end namespace llvm

#endif
