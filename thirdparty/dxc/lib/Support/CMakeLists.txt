set(system_libs)
if( NOT MSVC )
  if( MINGW )
    set(system_libs ${system_libs} psapi shell32 ole32)
  elseif( CMAKE_HOST_UNIX )
    if( HAVE_LIBRT )
      set(system_libs ${system_libs} rt)
    endif()
    if( HAVE_LIBDL )
      set(system_libs ${system_libs} ${CMAKE_DL_LIBS})
    endif()
    if(LLVM_ENABLE_TERMINFO)
      if(HAVE_TERMINFO)
        set(system_libs ${system_libs} ${TERMINFO_LIBS})
      endif()
    endif()
    if( LLVM_ENABLE_THREADS AND HAVE_LIBATOMIC )
      set(system_libs ${system_libs} atomic)
    endif()
    if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
      set(system_libs ${system_libs} pthread)
    endif()
    if ( LLVM_ENABLE_ZLIB AND HAVE_LIBZ )
      set(system_libs ${system_libs} z)
    endif()
    set(system_libs ${system_libs} m)
  endif( MINGW )
endif( NOT MSVC )

# HLSL Change - add ignored sources
set(HLSL_IGNORE_SOURCES DynamicLibrary.cpp PluginLoader.cpp)

# HLSL Change Begin - Add implicit filesystem when tests are enabled
if (LLVM_INCLUDE_TESTS)
  set_source_files_properties(Path.cpp
    PROPERTIES COMPILE_DEFINITIONS "MS_IMPLICIT_DISK_FILESYSTEM")
endif()
# HLSL Change End

add_llvm_library(LLVMSupport
  APFloat.cpp
  APInt.cpp
  APSInt.cpp
  ARMBuildAttrs.cpp
  ARMWinEH.cpp
  Allocator.cpp
  BlockFrequency.cpp
  BranchProbability.cpp
  circular_raw_ostream.cpp
  COM.cpp
  CommandLine.cpp
  Compression.cpp
  ConvertUTF.c
  ConvertUTFWrapper.cpp
  CrashRecoveryContext.cpp
  DataExtractor.cpp
  DataStream.cpp
  Debug.cpp
  DeltaAlgorithm.cpp
  DAGDeltaAlgorithm.cpp
  Dwarf.cpp
  ErrorHandling.cpp
  FileUtilities.cpp
  FileOutputBuffer.cpp
  FoldingSet.cpp
  FormattedStream.cpp
  GraphWriter.cpp
  Hashing.cpp
  IntEqClasses.cpp
  IntervalMap.cpp
  IntrusiveRefCntPtr.cpp
  LEB128.cpp
  LineIterator.cpp
  Locale.cpp
  LockFileManager.cpp
  ManagedStatic.cpp
  MathExtras.cpp
  MemoryBuffer.cpp
  MemoryObject.cpp
  MSFileSystemBasic.cpp
  MD5.cpp
  Options.cpp
  # PluginLoader.cpp    # HLSL Change Starts - no support for plug-in loader
  PrettyStackTrace.cpp
  RandomNumberGenerator.cpp
  Regex.cpp
  ScaledNumber.cpp
  SmallPtrSet.cpp
  SmallVector.cpp
  SourceMgr.cpp
  SpecialCaseList.cpp
  Statistic.cpp
  StreamingMemoryObject.cpp
  StringExtras.cpp
  StringMap.cpp
  StringPool.cpp
  StringSaver.cpp
  StringRef.cpp
  SystemUtils.cpp
  TargetParser.cpp
  Timer.cpp
  ToolOutputFile.cpp
  Triple.cpp
  Twine.cpp
  Unicode.cpp
  YAMLParser.cpp
  YAMLTraits.cpp
  raw_os_ostream.cpp
  raw_ostream.cpp
  regcomp.c
  regerror.c
  regexec.c
  regfree.c
  regstrlcpy.c
  regmalloc.cpp # HLSL Change

# System
  assert.cpp      # HLSL Change
  Atomic.cpp
  # DynamicLibrary.cpp # HLSL Change - no support for interpreter or jit
  Errno.cpp
  Host.cpp
  Memory.cpp
  Mutex.cpp
  Path.cpp
  Process.cpp
  Program.cpp
  RWMutex.cpp
  SearchForAddressOfSpecialSymbol.cpp
  Signals.cpp
  TargetRegistry.cpp
  ThreadLocal.cpp
  Threading.cpp
  TimeValue.cpp
  Valgrind.cpp
  Watchdog.cpp

  ADDITIONAL_HEADER_DIRS
  Unix
  Windows
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support

  LINK_LIBS ${system_libs}
  )

set_property(TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS "${system_libs}")
if(WIN32)
set_property(TARGET LLVMSupport PROPERTY COMPILE_FLAGS /EHsc )
endif(WIN32)

target_link_libraries(LLVMSupport PUBLIC LLVMMSSupport) # HLSL Change
