//===- llvm/Support/Unix/Path.inc - Unix Path Implementation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific implementation of the Path API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include <limits.h>
#include <stdio.h>
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif
#if HAVE_DIRENT_H
# include <dirent.h>
# define NAMLEN(dirent) strlen((dirent)->d_name)
#else
# define dirent direct
# define NAMLEN(dirent) (dirent)->d_namlen
# if HAVE_SYS_NDIR_H
#  include <sys/ndir.h>
# endif
# if HAVE_SYS_DIR_H
#  include <sys/dir.h>
# endif
# if HAVE_NDIR_H
#  include <ndir.h>
# endif
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// Both stdio.h and cstdio are included via different pathes and
// stdcxx's cstdio doesn't include stdio.h, so it doesn't #undef the macros
// either.
#undef ferror
#undef feof

// For GNU Hurd
#if defined(__GNU__) && !defined(PATH_MAX)
# define PATH_MAX 4096
#endif

#include "llvm/Support/MSFileSystem.h" // HLSL change

using namespace llvm;

namespace llvm {
namespace sys  {
namespace fs {
#if defined(__FreeBSD__) || defined (__NetBSD__) || defined(__Bitrig__) || \
    defined(__OpenBSD__) || defined(__minix) || defined(__FreeBSD_kernel__) || \
    defined(__linux__) || defined(__CYGWIN__) || defined(__DragonFly__)
static int
test_dir(char ret[PATH_MAX], const char *dir, const char *bin)
{  
  struct stat sb;
  char fullpath[PATH_MAX];

  snprintf(fullpath, PATH_MAX, "%s/%s", dir, bin);
  if (realpath(fullpath, ret) == NULL)
    return (1);
  if (stat(fullpath, &sb) != 0)
    return (1);

  return (0);
}

static char *
getprogpath(char ret[PATH_MAX], const char *bin)
{
  char *pv, *s, *t;

  /* First approach: absolute path. */
  if (bin[0] == '/') {
    if (test_dir(ret, "/", bin) == 0)
      return (ret);
    return (NULL);
  }

  /* Second approach: relative path. */
  if (strchr(bin, '/') != NULL) {
    char cwd[PATH_MAX];
    if (getcwd(cwd, PATH_MAX) == NULL)
      return (NULL);
    if (test_dir(ret, cwd, bin) == 0)
      return (ret);
    return (NULL);
  }

  /* Third approach: $PATH */
  if ((pv = getenv("PATH")) == NULL)
    return (NULL);
  s = pv = strdup(pv);
  if (pv == NULL)
    return (NULL);
  while ((t = strsep(&s, ":")) != NULL) {
    if (test_dir(ret, t, bin) == 0) {
      free(pv);
      return (ret);
    }
  }
  free(pv);
  return (NULL);
}
#endif // __FreeBSD__ || __NetBSD__ || __FreeBSD_kernel__

/// GetMainExecutable - Return the path to the main executable, given the
/// value of argv[0] from program startup.
std::string getMainExecutable(const char *argv0, void *MainAddr) {
#if defined(__APPLE__)
  // On OS X the executable path is saved to the stack by dyld. Reading it
  // from there is much faster than calling dladdr, especially for large
  // binaries with symbols.
  char exe_path[MAXPATHLEN];
  uint32_t size = sizeof(exe_path);
  if (_NSGetExecutablePath(exe_path, &size) == 0) {
    char link_path[MAXPATHLEN];
    if (realpath(exe_path, link_path))
      return link_path;
  }
#elif defined(__FreeBSD__) || defined (__NetBSD__) || defined(__Bitrig__) || \
      defined(__OpenBSD__) || defined(__minix) || defined(__DragonFly__) || \
      defined(__FreeBSD_kernel__)
  char exe_path[PATH_MAX];

  if (getprogpath(exe_path, argv0) != NULL)
    return exe_path;
#elif defined(__linux__) || defined(__CYGWIN__)
  char exe_path[MAXPATHLEN];
  StringRef aPath("/proc/self/exe");
  if (sys::fs::exists(aPath)) {
      // /proc is not always mounted under Linux (chroot for example).
      ssize_t len = readlink(aPath.str().c_str(), exe_path, sizeof(exe_path));
      if (len >= 0)
          return std::string(exe_path, len);
  } else {
      // Fall back to the classical detection.
      if (getprogpath(exe_path, argv0) != NULL)
          return exe_path;
  }
#elif defined(HAVE_DLFCN_H)
  // Use dladdr to get executable path if available.
  Dl_info DLInfo;
  int err = dladdr(MainAddr, &DLInfo);
  if (err == 0)
    return "";

  // If the filename is a symlink, we need to resolve and return the location of
  // the actual executable.
  char link_path[MAXPATHLEN];
  if (realpath(DLInfo.dli_fname, link_path))
    return link_path;
#else
#error GetMainExecutable is not implemented on this host yet.
#endif
  return "";
}

TimeValue file_status::getLastModificationTime() const {
  TimeValue Ret;
  Ret.fromEpochTime(fs_st_mtime);
  return Ret;
}

UniqueID file_status::getUniqueID() const {
  return UniqueID(fs_st_dev, fs_st_ino);
}

std::error_code current_path(SmallVectorImpl<char> &result) {
  result.clear();

  const char *pwd = ::getenv("PWD");
  llvm::sys::fs::file_status PWDStatus, DotStatus;
  if (pwd && llvm::sys::path::is_absolute(pwd) &&
      !llvm::sys::fs::status(pwd, PWDStatus) &&
      !llvm::sys::fs::status(".", DotStatus) &&
      PWDStatus.getUniqueID() == DotStatus.getUniqueID()) {
    result.append(pwd, pwd + strlen(pwd));
    return std::error_code();
  }

#ifdef MAXPATHLEN
  result.reserve(MAXPATHLEN);
#else
// For GNU Hurd
  result.reserve(1024);
#endif

  while (true) {
    if (::getcwd(result.data(), result.capacity()) == nullptr) {
      // See if there was a real error.
      if (errno != ENOMEM)
        return std::error_code(errno, std::generic_category());
      // Otherwise there just wasn't enough space.
      result.reserve(result.capacity() * 2);
    } else
      break;
  }

  result.set_size(strlen(result.data()));
  return std::error_code();
}

std::error_code create_directory(const Twine &path, bool IgnoreExisting) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::mkdir(p.begin(), S_IRWXU | S_IRWXG) == -1) {
    if (errno != EEXIST || !IgnoreExisting)
      return std::error_code(errno, std::generic_category());
  }

  return std::error_code();
}

// Note that we are using symbolic link because hard links are not supported by
// all filesystems (SMB doesn't).
std::error_code create_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::symlink(t.begin(), f.begin()) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

std::error_code remove(const Twine &path, bool IgnoreNonExisting) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat buf;
  if (lstat(p.begin(), &buf) != 0) {
    if (errno != ENOENT || !IgnoreNonExisting)
      return std::error_code(errno, std::generic_category());
    return std::error_code();
  }

  // Note: this check catches strange situations. In all cases, LLVM should
  // only be involved in the creation and deletion of regular files.  This
  // check ensures that what we're trying to erase is a regular file. It
  // effectively prevents LLVM from erasing things like /dev/null, any block
  // special file, or other things that aren't "regular" files.
  if (!S_ISREG(buf.st_mode) && !S_ISDIR(buf.st_mode) && !S_ISLNK(buf.st_mode))
    return make_error_code(errc::operation_not_permitted);

  if (::remove(p.begin()) == -1) {
    if (errno != ENOENT || !IgnoreNonExisting)
      return std::error_code(errno, std::generic_category());
  }

  return std::error_code();
}

std::error_code rename(const Twine &from, const Twine &to) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::rename(f.begin(), t.begin()) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

std::error_code resize_file(int FD, uint64_t Size) {
  if (::ftruncate(FD, Size) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

static int convertAccessMode(AccessMode Mode) {
  switch (Mode) {
  case AccessMode::Exist:
    return F_OK;
  case AccessMode::Write:
    return W_OK;
  case AccessMode::Execute:
    return R_OK | X_OK; // scripts also need R_OK.
  }
  llvm_unreachable("invalid enum");
}

std::error_code access(const Twine &Path, AccessMode Mode) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  if (::access(P.begin(), convertAccessMode(Mode)) == -1)
    return std::error_code(errno, std::generic_category());

  if (Mode == AccessMode::Execute) {
    // Don't say that directories are executable.
    struct stat buf;
    if (0 != stat(P.begin(), &buf))
      return errc::permission_denied;
    if (!S_ISREG(buf.st_mode))
      return errc::permission_denied;
  }

  return std::error_code();
}

bool equivalent(file_status A, file_status B) {
  assert(status_known(A) && status_known(B));
  return A.fs_st_dev == B.fs_st_dev &&
         A.fs_st_ino == B.fs_st_ino;
}

std::error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  file_status fsA, fsB;
  if (std::error_code ec = status(A, fsA))
    return ec;
  if (std::error_code ec = status(B, fsB))
    return ec;
  result = equivalent(fsA, fsB);
  return std::error_code();
}

static std::error_code fillStatus(int StatRet, const struct stat &Status,
                             file_status &Result) {
  if (StatRet != 0) {
    std::error_code ec(errno, std::generic_category());
    if (ec == errc::no_such_file_or_directory)
      Result = file_status(file_type::file_not_found);
    else
      Result = file_status(file_type::status_error);
    return ec;
  }

  file_type Type = file_type::type_unknown;

  if (S_ISDIR(Status.st_mode))
    Type = file_type::directory_file;
  else if (S_ISREG(Status.st_mode))
    Type = file_type::regular_file;
  else if (S_ISBLK(Status.st_mode))
    Type = file_type::block_file;
  else if (S_ISCHR(Status.st_mode))
    Type = file_type::character_file;
  else if (S_ISFIFO(Status.st_mode))
    Type = file_type::fifo_file;
  else if (S_ISSOCK(Status.st_mode))
    Type = file_type::socket_file;

  perms Perms = static_cast<perms>(Status.st_mode);
  Result =
      file_status(Type, Perms, Status.st_dev, Status.st_ino, Status.st_mtime,
                  Status.st_uid, Status.st_gid, Status.st_size);

  return std::error_code();
}

std::error_code status(const Twine &Path, file_status &Result) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  struct stat Status;
  MSFileSystem* fsr = GetCurrentThreadFileSystem(); // HLSL Change
  int StatRet = fsr->Stat(P.begin(), &Status); // HLSL Change - FS abstraction
  return fillStatus(StatRet, Status, Result);
}

std::error_code status(int FD, file_status &Result) {
  struct stat Status;
  MSFileSystem* fsr = GetCurrentThreadFileSystem(); // HLSL Change
  int StatRet = fsr->Fstat(FD, &Status); // HLSL Change - FS abstraction
  return fillStatus(StatRet, Status, Result);
}

std::error_code setLastModificationAndAccessTime(int FD, TimeValue Time) {
#if defined(HAVE_FUTIMENS)
  timespec Times[2];
  Times[0].tv_sec = Time.toEpochTime();
  Times[0].tv_nsec = 0;
  Times[1] = Times[0];
  if (::futimens(FD, Times))
    return std::error_code(errno, std::generic_category());
  return std::error_code();
#elif defined(HAVE_FUTIMES)
  timeval Times[2];
  Times[0].tv_sec = Time.toEpochTime();
  Times[0].tv_usec = 0;
  Times[1] = Times[0];
  if (::futimes(FD, Times))
    return std::error_code(errno, std::generic_category());
  return std::error_code();
#else
#warning Missing futimes() and futimens()
  return make_error_code(errc::function_not_supported);
#endif
}

std::error_code mapped_file_region::init(int FD, uint64_t Offset,
                                         mapmode Mode) {
  assert(Size != 0);

  int flags = (Mode == readwrite) ? MAP_SHARED : MAP_PRIVATE;
  int prot = (Mode == readonly) ? PROT_READ : (PROT_READ | PROT_WRITE);
  Mapping = ::mmap(nullptr, Size, prot, flags, FD, Offset);
  if (Mapping == MAP_FAILED)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

mapped_file_region::mapped_file_region(int fd, mapmode mode, uint64_t length,
                                       uint64_t offset, std::error_code &ec)
    : Size(length), Mapping() {
  // Make sure that the requested size fits within SIZE_T.
  if (length > std::numeric_limits<size_t>::max()) {
    ec = make_error_code(errc::invalid_argument);
    return;
  }

  ec = init(fd, offset, mode);
  if (ec)
    Mapping = nullptr;
}

mapped_file_region::~mapped_file_region() {
  if (Mapping)
    ::munmap(Mapping, Size);
}

uint64_t mapped_file_region::size() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return Size;
}

char *mapped_file_region::data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return reinterpret_cast<char*>(Mapping);
}

const char *mapped_file_region::const_data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return reinterpret_cast<const char*>(Mapping);
}

int mapped_file_region::alignment() {
  return Process::getPageSize();
}

std::error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                StringRef path){
  SmallString<128> path_null(path);
  DIR *directory = ::opendir(path_null.c_str());
  if (!directory)
    return std::error_code(errno, std::generic_category());

  it.IterationHandle = reinterpret_cast<intptr_t>(directory);
  // Add something for replace_filename to replace.
  path::append(path_null, ".");
  it.CurrentEntry = directory_entry(path_null.str());
  return directory_iterator_increment(it);
}

std::error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle)
    ::closedir(reinterpret_cast<DIR *>(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return std::error_code();
}

std::error_code detail::directory_iterator_increment(detail::DirIterState &it) {
  errno = 0;
  dirent *cur_dir = ::readdir(reinterpret_cast<DIR *>(it.IterationHandle));
  if (cur_dir == nullptr && errno != 0) {
    return std::error_code(errno, std::generic_category());
  } else if (cur_dir != nullptr) {
    StringRef name(cur_dir->d_name, NAMLEN(cur_dir));
    if ((name.size() == 1 && name[0] == '.') ||
        (name.size() == 2 && name[0] == '.' && name[1] == '.'))
      return directory_iterator_increment(it);
    it.CurrentEntry.replace_filename(name);
  } else
    return directory_iterator_destruct(it);

  return std::error_code();
}

std::error_code openFileForRead(const Twine &Name, int &ResultFD) {
  SmallString<128> Storage;
  StringRef P = Name.toNullTerminatedStringRef(Storage);
  MSFileSystem* fsr = GetCurrentThreadFileSystem(); // HLSL Change
  while ((ResultFD = fsr->Open(P.begin(), O_RDONLY)) < 0) { // HLSL Change - FS abstraction
    if (errno != EINTR)
      return std::error_code(errno, std::generic_category());
  }
  return std::error_code();
}

std::error_code openFileForWrite(const Twine &Name, int &ResultFD,
                            sys::fs::OpenFlags Flags, unsigned Mode) {
  // Verify that we don't have both "append" and "excl".
  assert((!(Flags & sys::fs::F_Excl) || !(Flags & sys::fs::F_Append)) &&
         "Cannot specify both 'excl' and 'append' file creation flags!");

  int OpenFlags = O_CREAT;

  if (Flags & F_RW)
    OpenFlags |= O_RDWR;
  else
    OpenFlags |= O_WRONLY;

  if (Flags & F_Append)
    OpenFlags |= O_APPEND;
  else
    OpenFlags |= O_TRUNC;

  if (Flags & F_Excl)
    OpenFlags |= O_EXCL;

  SmallString<128> Storage;
  StringRef P = Name.toNullTerminatedStringRef(Storage);
  MSFileSystem* fsr = GetCurrentThreadFileSystem(); // HLSL Change
  while ((ResultFD = fsr->Open(P.begin(), OpenFlags, Mode)) < 0) { // HLSL Change - FS abstraction
    if (errno != EINTR)
      return std::error_code(errno, std::generic_category());
  }
  return std::error_code();
}

} // end namespace fs

namespace path {

bool home_directory(SmallVectorImpl<char> &result) {
  if (char *RequestedDir = getenv("HOME")) {
    result.clear();
    result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
    return true;
  }

  return false;
}

static const char *getEnvTempDir() {
  // Check whether the temporary directory is specified by an environment
  // variable.
  const char *EnvironmentVariables[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};
  for (const char *Env : EnvironmentVariables) {
    if (const char *Dir = std::getenv(Env))
      return Dir;
  }

  return nullptr;
}

static const char *getDefaultTempDir(bool ErasedOnReboot) {
#ifdef P_tmpdir
  if ((bool)P_tmpdir)
    return P_tmpdir;
#endif

  if (ErasedOnReboot)
    return "/tmp";
  return "/var/tmp";
}

void system_temp_directory(bool ErasedOnReboot, SmallVectorImpl<char> &Result) {
  Result.clear();

  if (ErasedOnReboot) {
    // There is no env variable for the cache directory.
    if (const char *RequestedDir = getEnvTempDir()) {
      Result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
      return;
    }
  }

#if defined(_CS_DARWIN_USER_TEMP_DIR) && defined(_CS_DARWIN_USER_CACHE_DIR)
  // On Darwin, use DARWIN_USER_TEMP_DIR or DARWIN_USER_CACHE_DIR.
  // macros defined in <unistd.h> on darwin >= 9
  int ConfName = ErasedOnReboot? _CS_DARWIN_USER_TEMP_DIR
                               : _CS_DARWIN_USER_CACHE_DIR;
  size_t ConfLen = confstr(ConfName, nullptr, 0);
  if (ConfLen > 0) {
    do {
      Result.resize(ConfLen);
      ConfLen = confstr(ConfName, Result.data(), Result.size());
    } while (ConfLen > 0 && ConfLen != Result.size());

    if (ConfLen > 0) {
      assert(Result.back() == 0);
      Result.pop_back();
      return;
    }

    Result.clear();
  }
#endif

  const char *RequestedDir = getDefaultTempDir(ErasedOnReboot);
  Result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
}

} // end namespace path

} // end namespace sys
} // end namespace llvm


// The following is implementation of MSFileSystem for Unix.
// This will make usages of MSFileSystem throughout DirectXShaderCompiler
// compatible with Unix.
namespace llvm {
namespace sys  {
namespace fs {

thread_local MSFileSystem* threadFS = nullptr;
std::atomic<bool> Initialized(false);

std::error_code SetupPerThreadFileSystem() throw() {
  // This method is now a no-op since we are using thread_local.
  Initialized = true;
  return std::error_code();
}
void CleanupPerThreadFileSystem() throw() {
  // This method is now a no-op since we are using thread_local.
}

MSFileSystem* GetCurrentThreadFileSystem() throw() {
#ifdef MS_IMPLICIT_DISK_FILESYSTEM
  if (!Initialized)
    getImplicitFilesystem();
#endif
  return threadFS;
}

std::error_code SetCurrentThreadFileSystem(MSFileSystem* value) throw() {
  // For now, disallow reentrancy in APIs (i.e., replace the current instance with another one).
  if (value != nullptr) {
    MSFileSystem* current = GetCurrentThreadFileSystem();
    if (current != nullptr) {
      return std::error_code(1131L, std::system_category());
    }
  }
  threadFS = value;
  return std::error_code();
}

int msf_read(int fd, void* buffer, unsigned int count) throw() {
  MSFileSystem* fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->Read(fd, buffer, count);
}

int msf_write(int fd, const void* buffer, unsigned int count) throw() {
  MSFileSystem* fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->Write(fd, buffer, count);
}

int msf_close(int fd) throw() {
  MSFileSystem* fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->close(fd);
}

long msf_lseek(int fd, long offset, int origin) {
  MSFileSystem* fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->lseek(fd, offset, origin);
}

int msf_setmode(int fd, int mode) throw() {
  MSFileSystem* fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->setmode(fd, mode);
}

} // namespace fs
} // namespace sys
} // namespace llvm
