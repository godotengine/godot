// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "Debug.h"
#include "Array.inl"
#include "StrLib.h" // StringBuilder

#include "StdStream.h" // fileOpen

#include <stdlib.h>

// Extern
#if NV_OS_WIN32 //&& NV_CC_MSVC
#   define WIN32_LEAN_AND_MEAN
#   define VC_EXTRALEAN
#   include <windows.h>
#   include <direct.h>
// -- GODOT start -
#   include <crtdbg.h>
#   if _MSC_VER < 1300
#       define DECLSPEC_DEPRECATED
// VC6: change this path to your Platform SDK headers
#       include <dbghelp.h> // must be XP version of file
//      include "M:\\dev7\\vs\\devtools\\common\\win32sdk\\include\\dbghelp.h"
#   else
// VC7: ships with updated headers
#       include <dbghelp.h>
#   endif
// -- GODOT end -
#   pragma comment(lib,"dbghelp.lib")
#endif

#if NV_OS_XBOX
#    include <Xtl.h>
#    ifdef _DEBUG
#        include <xbdm.h>
#    endif //_DEBUG
#endif //NV_OS_XBOX

#if !NV_OS_WIN32 && defined(NV_HAVE_SIGNAL_H)
#   include <signal.h>
#endif

#if NV_OS_UNIX
#   include <unistd.h> // getpid
#endif

#if NV_OS_LINUX && defined(NV_HAVE_EXECINFO_H)
#   include <execinfo.h> // backtrace
#   if NV_CC_GNUC // defined(NV_HAVE_CXXABI_H)
#       include <cxxabi.h>
#   endif
#endif

#if NV_OS_DARWIN || NV_OS_FREEBSD || NV_OS_OPENBSD
#   include <sys/types.h>
#   include <sys/param.h>
#   include <sys/sysctl.h> // sysctl
#   if !defined(NV_OS_OPENBSD)
#       include <sys/ucontext.h>
#   endif
#   if defined(NV_HAVE_EXECINFO_H) // only after OSX 10.5
#       include <execinfo.h> // backtrace
#       if NV_CC_GNUC // defined(NV_HAVE_CXXABI_H)
#           include <cxxabi.h>
#       endif
#   endif
#endif

#if NV_OS_ORBIS
#include <libdbg.h>
#endif

#if NV_OS_DURANGO
#include "Windows.h"
#include <winnt.h>
#include <crtdbg.h>
#include <dbghelp.h>
#include <errhandlingapi.h>
#define NV_USE_SEPARATE_THREAD 0
#else
#define NV_USE_SEPARATE_THREAD 1
#endif



using namespace nv;

namespace 
{

    static MessageHandler * s_message_handler = NULL;
    static AssertHandler * s_assert_handler = NULL;

    static bool s_sig_handler_enabled = false;
    static bool s_interactive = true;

#if (NV_OS_WIN32 && NV_CC_MSVC) || NV_OS_DURANGO

    // Old exception filter.
    static LPTOP_LEVEL_EXCEPTION_FILTER s_old_exception_filter = NULL;

#elif !NV_OS_WIN32 && defined(NV_HAVE_SIGNAL_H)

    // Old signal handlers.
    struct sigaction s_old_sigsegv;
    struct sigaction s_old_sigtrap;
    struct sigaction s_old_sigfpe;
    struct sigaction s_old_sigbus;

#endif

// -- GODOT start -
#if NV_OS_WIN32 || NV_OS_DURANGO
// -- GODOT end -

    // We should try to simplify the top level filter as much as possible.
    // http://www.nynaeve.net/?p=128

    // The critical section enforcing the requirement that only one exception be
    // handled by a handler at a time.
    static CRITICAL_SECTION s_handler_critical_section;

#if NV_USE_SEPARATE_THREAD
    // Semaphores used to move exception handling between the exception thread
    // and the handler thread.  handler_start_semaphore_ is signalled by the
    // exception thread to wake up the handler thread when an exception occurs.
    // handler_finish_semaphore_ is signalled by the handler thread to wake up
    // the exception thread when handling is complete.
    static HANDLE s_handler_start_semaphore = NULL;
    static HANDLE s_handler_finish_semaphore = NULL;

    // The exception handler thread.
    static HANDLE s_handler_thread = NULL;

    static DWORD s_requesting_thread_id = 0;
    static EXCEPTION_POINTERS * s_exception_info = NULL;

#endif // NV_USE_SEPARATE_THREAD


    struct MinidumpCallbackContext {
        ULONG64 memory_base;
        ULONG memory_size;
        bool finished;
    };

#if NV_OS_WIN32
    // static
    static BOOL CALLBACK miniDumpWriteDumpCallback(PVOID context, const PMINIDUMP_CALLBACK_INPUT callback_input, PMINIDUMP_CALLBACK_OUTPUT callback_output)
    {
        switch (callback_input->CallbackType)
        {
        case MemoryCallback: {
            MinidumpCallbackContext* callback_context = reinterpret_cast<MinidumpCallbackContext*>(context);
            if (callback_context->finished)
                return FALSE;

            // Include the specified memory region.
            callback_output->MemoryBase = callback_context->memory_base;
            callback_output->MemorySize = callback_context->memory_size;
            callback_context->finished = true;
            return TRUE;
        }

        // Include all modules.
        case IncludeModuleCallback:
        case ModuleCallback:
            return TRUE;

        // Include all threads.
        case IncludeThreadCallback:
        case ThreadCallback:
            return TRUE;

        // Stop receiving cancel callbacks.
        case CancelCallback:
            callback_output->CheckCancel = FALSE;
            callback_output->Cancel = FALSE;
            return TRUE;
        }

        // Ignore other callback types.
        return FALSE;
    }
#endif

    static bool writeMiniDump(EXCEPTION_POINTERS * pExceptionInfo)
    {
#if NV_OS_DURANGO
        // Get a handle to the minidump method.
        typedef BOOL(WINAPI* MiniDumpWriteDumpPfn) (
            _In_ HANDLE hProcess,
            _In_ DWORD ProcessId,
            _In_ HANDLE hFile,
            _In_ MINIDUMP_TYPE DumpType,
            _In_opt_ PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
            _In_opt_ PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
            _Reserved_ PVOID CallbackParam
            );
        MiniDumpWriteDumpPfn MiniDumpWriteDump = NULL;
        HMODULE hToolHelpModule = ::LoadLibraryW(L"toolhelpx.dll");
        if (hToolHelpModule != INVALID_HANDLE_VALUE) {
            MiniDumpWriteDump = reinterpret_cast<MiniDumpWriteDumpPfn>(::GetProcAddress(hToolHelpModule, "MiniDumpWriteDump"));
            if (!MiniDumpWriteDump) {
                FreeLibrary(hToolHelpModule);
                return false;
            }
        }
        else
            return false;

        // Generate a decent filename.
        nv::Path application_path(256);
        HINSTANCE hinstance = GetModuleHandle(NULL);
        GetModuleFileName(hinstance, application_path.str(), 256);
        application_path.stripExtension();
        const char * application_name = application_path.fileName();

        SYSTEMTIME local_time;
        GetLocalTime(&local_time);

        char dump_filename[MAX_PATH] = {};
        sprintf_s(dump_filename, "d:\\%s-%04d%02d%02d-%02d%02d%02d.dmp",
            application_name,
            local_time.wYear, local_time.wMonth, local_time.wDay,
            local_time.wHour, local_time.wMinute, local_time.wSecond );
#else
        const char* dump_filename = "crash.dmp";
#endif

        // create the file
        HANDLE hFile = CreateFileA(dump_filename, GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_WRITE | FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            //nvDebug("*** Failed to create dump file.\n");
#if NV_OS_DURANGO
            FreeLibrary(hToolHelpModule);
#endif
            return false;
        }

        MINIDUMP_EXCEPTION_INFORMATION * pExInfo = NULL;
#if NV_OS_WIN32
        MINIDUMP_CALLBACK_INFORMATION * pCallback = NULL;
#else
        void * pCallback = NULL;
#endif

        MINIDUMP_EXCEPTION_INFORMATION ExInfo;
        if (pExceptionInfo != NULL) {
            ExInfo.ThreadId = ::GetCurrentThreadId();
            ExInfo.ExceptionPointers = pExceptionInfo;
            ExInfo.ClientPointers = NULL;
            pExInfo = &ExInfo;

#if NV_OS_WIN32
            MINIDUMP_CALLBACK_INFORMATION callback;
            MinidumpCallbackContext context;

            // Find a memory region of 256 bytes centered on the
            // faulting instruction pointer.
            const ULONG64 instruction_pointer = 
            #if defined(_M_IX86)
                pExceptionInfo->ContextRecord->Eip;
            #elif defined(_M_AMD64)
                pExceptionInfo->ContextRecord->Rip;
            #else
                #error Unsupported platform
            #endif

            MEMORY_BASIC_INFORMATION info;
            
            if (VirtualQuery(reinterpret_cast<LPCVOID>(instruction_pointer), &info, sizeof(MEMORY_BASIC_INFORMATION)) != 0 && info.State == MEM_COMMIT)
            {
                // Attempt to get 128 bytes before and after the instruction
                // pointer, but settle for whatever's available up to the
                // boundaries of the memory region.
                const ULONG64 kIPMemorySize = 256;
                context.memory_base = max(reinterpret_cast<ULONG64>(info.BaseAddress), instruction_pointer - (kIPMemorySize / 2));
                ULONG64 end_of_range = min(instruction_pointer + (kIPMemorySize / 2), reinterpret_cast<ULONG64>(info.BaseAddress) + info.RegionSize);
                context.memory_size = static_cast<ULONG>(end_of_range - context.memory_base);
                context.finished = false;

                callback.CallbackRoutine = miniDumpWriteDumpCallback;
                callback.CallbackParam = reinterpret_cast<void*>(&context);
                pCallback = &callback;
            }
#endif
        }

        MINIDUMP_TYPE miniDumpType = (MINIDUMP_TYPE)(MiniDumpNormal|MiniDumpWithHandleData|MiniDumpWithThreadInfo);

        // write the dump
        BOOL ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, miniDumpType, pExInfo, NULL, pCallback) != 0;
        CloseHandle(hFile);
#if NV_OS_DURANGO
        FreeLibrary(hToolHelpModule);
#endif

        if (ok == FALSE) {
            //nvDebug("*** Failed to save dump file.\n");
            return false;
        }

        //nvDebug("\nDump file saved.\n");

        return true;
    }

#if NV_USE_SEPARATE_THREAD

    static DWORD WINAPI ExceptionHandlerThreadMain(void* lpParameter) {
        nvDebugCheck(s_handler_start_semaphore != NULL);
        nvDebugCheck(s_handler_finish_semaphore != NULL);

        while (true) {
            if (WaitForSingleObject(s_handler_start_semaphore, INFINITE) == WAIT_OBJECT_0) {
                writeMiniDump(s_exception_info);

                // Allow the requesting thread to proceed.
                ReleaseSemaphore(s_handler_finish_semaphore, 1, NULL);
            }
        }

        // This statement is not reached when the thread is unconditionally
        // terminated by the ExceptionHandler destructor.
        return 0;
    }

#endif // NV_USE_SEPARATE_THREAD

    static bool hasStackTrace() {
        return true;
    }

    /*static NV_NOINLINE int backtrace(void * trace[], int maxcount) {

        // In Windows XP and Windows Server 2003, the sum of the FramesToSkip and FramesToCapture parameters must be less than 63.
        int xp_maxcount = min(63-1, maxcount);

        int count = RtlCaptureStackBackTrace(1, xp_maxcount, trace, NULL);
        nvDebugCheck(count <= maxcount);

        return count;
    }*/

#if NV_OS_WIN32
    static NV_NOINLINE int backtraceWithSymbols(CONTEXT * ctx, void * trace[], int maxcount, int skip = 0) {
        
        // Init the stack frame for this function
        STACKFRAME64 stackFrame = { 0 };

    #if NV_CPU_X86_64
        DWORD dwMachineType = IMAGE_FILE_MACHINE_AMD64;
        stackFrame.AddrPC.Offset = ctx->Rip;
        stackFrame.AddrFrame.Offset = ctx->Rbp;
        stackFrame.AddrStack.Offset = ctx->Rsp;
    #elif NV_CPU_X86
        DWORD dwMachineType = IMAGE_FILE_MACHINE_I386;
        stackFrame.AddrPC.Offset = ctx->Eip;
        stackFrame.AddrFrame.Offset = ctx->Ebp;
        stackFrame.AddrStack.Offset = ctx->Esp;
    #else
        #error "Platform not supported!"
    #endif
        stackFrame.AddrPC.Mode = AddrModeFlat;
        stackFrame.AddrFrame.Mode = AddrModeFlat;
        stackFrame.AddrStack.Mode = AddrModeFlat;

        // Walk up the stack
        const HANDLE hThread = GetCurrentThread();
        const HANDLE hProcess = GetCurrentProcess();
        int i;
        for (i = 0; i < maxcount; i++)
        {
            // walking once first makes us skip self
            if (!StackWalk64(dwMachineType, hProcess, hThread, &stackFrame, ctx, NULL, &SymFunctionTableAccess64, &SymGetModuleBase64, NULL)) {
                break;
            }

            /*if (stackFrame.AddrPC.Offset == stackFrame.AddrReturn.Offset || stackFrame.AddrPC.Offset == 0) {
                break;
            }*/

            if (i >= skip) {
                trace[i - skip] = (PVOID)stackFrame.AddrPC.Offset;
            }
        }

        return i - skip;
    }

#pragma warning(push)
#pragma warning(disable:4748)
    static NV_NOINLINE int backtrace(void * trace[], int maxcount) {
        CONTEXT ctx = { 0 };
// -- GODOT start --
#if NV_CPU_X86 && !NV_CPU_X86_64
        ctx.ContextFlags = CONTEXT_CONTROL;
#if NV_CC_MSVC
        _asm {
             call x
          x: pop eax
             mov ctx.Eip, eax
             mov ctx.Ebp, ebp
             mov ctx.Esp, esp
        }
#else
        register long unsigned int ebp asm("ebp");
        ctx.Eip = (DWORD) __builtin_return_address(0);
        ctx.Ebp = ebp;
        ctx.Esp = (DWORD) __builtin_frame_address(0);
#endif
// -- GODOT end --
#else
        RtlCaptureContext(&ctx); // Not implemented correctly in x86.
#endif

        return backtraceWithSymbols(&ctx, trace, maxcount, 1);
    }
#pragma warning(pop)

    static NV_NOINLINE void writeStackTrace(void * trace[], int size, int start, Array<const char *> & lines)
    {
        StringBuilder builder(512);

        HANDLE hProcess = GetCurrentProcess();
        
        // Resolve PC to function names
        for (int i = start; i < size; i++)
        {
            // Check for end of stack walk
            DWORD64 ip = (DWORD64)trace[i];
            if (ip == NULL)
                break;

            // Get function name
            #define MAX_STRING_LEN  (512)
            unsigned char byBuffer[sizeof(IMAGEHLP_SYMBOL64) + MAX_STRING_LEN] = { 0 };
            IMAGEHLP_SYMBOL64 * pSymbol = (IMAGEHLP_SYMBOL64*)byBuffer;
            pSymbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL64);
            pSymbol->MaxNameLength = MAX_STRING_LEN;

            DWORD64 dwDisplacement;
            
            if (SymGetSymFromAddr64(hProcess, ip, &dwDisplacement, pSymbol))
            {
                pSymbol->Name[MAX_STRING_LEN-1] = 0;
                
                /*
                // Make the symbol readable for humans
                UnDecorateSymbolName( pSym->Name, lpszNonUnicodeUnDSymbol, BUFFERSIZE, 
                    UNDNAME_COMPLETE | 
                    UNDNAME_NO_THISTYPE |
                    UNDNAME_NO_SPECIAL_SYMS |
                    UNDNAME_NO_MEMBER_TYPE |
                    UNDNAME_NO_MS_KEYWORDS |
                    UNDNAME_NO_ACCESS_SPECIFIERS );
                */
                
                // pSymbol->Name
                const char * pFunc = pSymbol->Name;

                // Get file/line number
                IMAGEHLP_LINE64 theLine = { 0 };
                theLine.SizeOfStruct = sizeof(theLine);

                DWORD dwDisplacement;
                if (!SymGetLineFromAddr64(hProcess, ip, &dwDisplacement, &theLine))
                {
                    // Do not print unknown symbols anymore.
                    //break;
                    builder.format("unknown(%08X) : %s\n", (uint32)ip, pFunc);
                }
                else
                {
                    /*
                    const char* pFile = strrchr(theLine.FileName, '\\');
                    if ( pFile == NULL ) pFile = theLine.FileName;
                    else pFile++;
                    */
                    const char * pFile = theLine.FileName;
                    
                    int line = theLine.LineNumber;
                    
                    builder.format("%s(%d) : %s\n", pFile, line, pFunc);
                }

                lines.append(builder.release());

                if (pFunc != NULL && strcmp(pFunc, "WinMain") == 0) {
                    break;
                }
            }
        }
    }
#endif

    // Write mini dump and print stack trace.
    static LONG WINAPI handleException(EXCEPTION_POINTERS * pExceptionInfo)
    {
        EnterCriticalSection(&s_handler_critical_section);
#if NV_USE_SEPARATE_THREAD
        s_requesting_thread_id = GetCurrentThreadId();
        s_exception_info = pExceptionInfo;

        // This causes the handler thread to call writeMiniDump.
        ReleaseSemaphore(s_handler_start_semaphore, 1, NULL);

        // Wait until WriteMinidumpWithException is done and collect its return value.
        WaitForSingleObject(s_handler_finish_semaphore, INFINITE);
        //bool status = s_handler_return_value;

        // Clean up.
        s_requesting_thread_id = 0;
        s_exception_info = NULL;
#else
        // First of all, write mini dump.
        writeMiniDump(pExceptionInfo);
#endif
        LeaveCriticalSection(&s_handler_critical_section);

        nvDebug("\nDump file saved.\n");

        // Try to attach to debugger.
        if (s_interactive && debug::attachToDebugger()) {
            nvDebugBreak();
            return EXCEPTION_CONTINUE_EXECUTION;
        }

#if NV_OS_WIN32
        // If that fails, then try to pretty print a stack trace and terminate.
        void * trace[64];
        
        int size = backtraceWithSymbols(pExceptionInfo->ContextRecord, trace, 64);

        // @@ Use win32's CreateFile?
        FILE * fp = fileOpen("crash.txt", "wb");
        if (fp != NULL) {
            Array<const char *> lines;
            writeStackTrace(trace, size, 0, lines);

            for (uint i = 0; i < lines.count(); i++) {
                fputs(lines[i], fp);
                delete lines[i];
            }

            // @@ Add more info to crash.txt?

            fclose(fp);
        }
#endif

        // This should terminate the process and set the error exit code.
        TerminateProcess(GetCurrentProcess(), EXIT_FAILURE + 2);

        return EXCEPTION_EXECUTE_HANDLER;   // Terminate app. In case terminate process did not succeed.
    }

    static void handlePureVirtualCall() {
        nvDebugBreak();
        TerminateProcess(GetCurrentProcess(), EXIT_FAILURE + 8);
    }

    static void handleInvalidParameter(const wchar_t * wexpresion, const wchar_t * wfunction, const wchar_t * wfile, unsigned int line, uintptr_t reserved) {

        size_t convertedCharCount = 0;
        
        StringBuilder expresion;
        if (wexpresion != NULL) {
            uint size = U32(wcslen(wexpresion) + 1);
            expresion.reserve(size);
            wcstombs_s(&convertedCharCount, expresion.str(), size, wexpresion, _TRUNCATE);
        }

        StringBuilder file;
        if (wfile != NULL) {
            uint size = U32(wcslen(wfile) + 1);
            file.reserve(size);
            wcstombs_s(&convertedCharCount, file.str(), size, wfile, _TRUNCATE);
        }

        StringBuilder function;
        if (wfunction != NULL) {
            uint size = U32(wcslen(wfunction) + 1);
            function.reserve(size);
            wcstombs_s(&convertedCharCount, function.str(), size, wfunction, _TRUNCATE);
        }
        
        int result = nvAbort(expresion.str(), file.str(), line, function.str());
        if (result == NV_ABORT_DEBUG) {
            nvDebugBreak();
        } 
    }

#elif !NV_OS_WIN32 && defined(NV_HAVE_SIGNAL_H) // NV_OS_LINUX || NV_OS_DARWIN

#if defined(NV_HAVE_EXECINFO_H)

    static bool hasStackTrace() {
        return true;
    }


    static void writeStackTrace(void * trace[], int size, int start, Array<const char *> & lines) {
        StringBuilder builder(512);
        char ** string_array = backtrace_symbols(trace, size);

        for(int i = start; i < size-1; i++ ) {
            // IC: Just in case.
            if (string_array[i] == NULL || string_array[i][0] == '\0') break;

#       if NV_CC_GNUC // defined(NV_HAVE_CXXABI_H)
            // @@ Write a better parser for the possible formats.
            char * begin = strchr(string_array[i], '(');
            char * end = strrchr(string_array[i], '+');
            char * module = string_array[i];

            if (begin == 0 && end != 0) {
                *(end - 1) = '\0';
                begin = strrchr(string_array[i], ' ');
                module = NULL; // Ignore module.
            }

            if (begin != 0 && begin < end) {
                int stat;
                *end = '\0';
                *begin = '\0';
                char * name = abi::__cxa_demangle(begin+1, 0, 0, &stat);
                if (module == NULL) {
                    if (name == NULL || stat != 0) {
                        builder.format("  In: '%s'\n", begin+1);
                    }
                    else {
                        builder.format("  In: '%s'\n", name);
                    }
                }
                else {
                    if (name == NULL || stat != 0) {
                        builder.format("  In: [%s] '%s'\n", module, begin+1);
                    }
                    else {
                        builder.format("  In: [%s] '%s'\n", module, name);
                    }
                }
                free(name);
            }
            else {
                builder.format("  In: '%s'\n", string_array[i]);
            }
#       else
            builder.format("  In: '%s'\n", string_array[i]);
#       endif
            lines.append(builder.release());
        }

        free(string_array);
    }

    static void printStackTrace(void * trace[], int size, int start=0) {
        nvDebug( "\nDumping stacktrace:\n" );

        Array<const char *> lines;
        writeStackTrace(trace, size, 1, lines);

        for (uint i = 0; i < lines.count(); i++) {
            nvDebug("%s", lines[i]);
            delete lines[i];
        }

        nvDebug("\n");
    }

#endif // defined(NV_HAVE_EXECINFO_H)

    static void * callerAddress(void * secret)
    {
#if NV_OS_DARWIN
#  if defined(_STRUCT_MCONTEXT)
#    if NV_CPU_PPC
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext->__ss.__srr0;
#    elif NV_CPU_X86_64
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext->__ss.__rip;
#    elif NV_CPU_X86
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext->__ss.__eip;
#    elif NV_CPU_ARM
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext->__ss.__pc;
#    else
#      error "Unknown CPU"
#    endif
#  else
#    if NV_CPU_PPC
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext->ss.srr0;
#    elif NV_CPU_X86
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext->ss.eip;
#    else
#      error "Unknown CPU"
#    endif
#  endif
#elif NV_OS_FREEBSD
#  if NV_CPU_X86_64
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *)ucp->uc_mcontext.mc_rip;
#  elif NV_CPU_X86
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *)ucp->uc_mcontext.mc_eip;
#    else
#      error "Unknown CPU"
#    endif
#elif NV_OS_OPENBSD
#  if NV_CPU_X86_64
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *)ucp->sc_rip;
#  elif NV_CPU_X86
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *)ucp->sc_eip;
#  else
#       error "Unknown CPU"
#  endif        
#else
#  if NV_CPU_X86_64
        // #define REG_RIP REG_INDEX(rip) // seems to be 16
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *)ucp->uc_mcontext.gregs[REG_RIP];
#  elif NV_CPU_X86
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *)ucp->uc_mcontext.gregs[14/*REG_EIP*/];
#  elif NV_CPU_PPC
        ucontext_t * ucp = (ucontext_t *)secret;
        return (void *) ucp->uc_mcontext.regs->nip;
#    else
#      error "Unknown CPU"
#    endif
#endif

        // How to obtain the instruction pointers in different platforms, from mlton's source code.
        // http://mlton.org/
        // OpenBSD && NetBSD
        // ucp->sc_eip
        // FreeBSD:
        // ucp->uc_mcontext.mc_eip
        // HPUX:
        // ucp->uc_link
        // Solaris:
        // ucp->uc_mcontext.gregs[REG_PC]
        // Linux hppa:
        // uc->uc_mcontext.sc_iaoq[0] & ~0x3UL
        // Linux sparc:
        // ((struct sigcontext*) secret)->sigc_regs.tpc
        // Linux sparc64:
        // ((struct sigcontext*) secret)->si_regs.pc

        // potentially correct for other archs:
        // Linux alpha: ucp->m_context.sc_pc
        // Linux arm: ucp->m_context.ctx.arm_pc
        // Linux ia64: ucp->m_context.sc_ip & ~0x3UL
        // Linux mips: ucp->m_context.sc_pc
        // Linux s390: ucp->m_context.sregs->regs.psw.addr
    }

    static void nvSigHandler(int sig, siginfo_t *info, void *secret)
    {
        void * pnt = callerAddress(secret);

        // Do something useful with siginfo_t
        if (sig == SIGSEGV) {
            if (pnt != NULL) nvDebug("Got signal %d, faulty address is %p, from %p\n", sig, info->si_addr, pnt);
            else nvDebug("Got signal %d, faulty address is %p\n", sig, info->si_addr);
        }
        else if(sig == SIGTRAP) {
            nvDebug("Breakpoint hit.\n");
        }
        else {
            nvDebug("Got signal %d\n", sig);
        }

#if defined(NV_HAVE_EXECINFO_H)
        if (hasStackTrace()) // in case of weak linking
        {
            void * trace[64];
            int size = backtrace(trace, 64);

            if (pnt != NULL) {
                // Overwrite sigaction with caller's address.
                trace[1] = pnt;
            }

            printStackTrace(trace, size, 1);
        }
#endif // defined(NV_HAVE_EXECINFO_H)

        exit(0);
    }

#endif // defined(NV_HAVE_SIGNAL_H)



#if NV_OS_WIN32 //&& NV_CC_MSVC

    /** Win32 assert handler. */
    struct Win32AssertHandler : public AssertHandler 
    {
        // Flush the message queue. This is necessary for the message box to show up.
        static void flushMessageQueue()
        {
            MSG msg;
            while( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) ) {
                //if( msg.message == WM_QUIT ) break;
                TranslateMessage( &msg );
                DispatchMessage( &msg );
            }
        }

        // Assert handler method.
        virtual int assertion(const char * exp, const char * file, int line, const char * func, const char * msg, va_list arg)
        {
            int ret = NV_ABORT_EXIT;

            StringBuilder error_string;
            error_string.format("*** Assertion failed: %s\n    On file: %s\n    On line: %d\n", exp, file, line );
            if (func != NULL) {
                error_string.appendFormat("    On function: %s\n", func);
            }
            if (msg != NULL) {
                error_string.append("    Message: ");
                va_list tmp;
                va_copy(tmp, arg);
                error_string.appendFormatList(msg, tmp);
                va_end(tmp);
                error_string.append("\n");
            }
            nvDebug( error_string.str() );

            // Print stack trace:
            debug::dumpInfo();

            if (debug::isDebuggerPresent()) {
                return NV_ABORT_DEBUG;
            }

            if (s_interactive) {
                flushMessageQueue();
                int action = MessageBoxA(NULL, error_string.str(), "Assertion failed", MB_ABORTRETRYIGNORE | MB_ICONERROR | MB_TOPMOST);
                switch( action ) {
                case IDRETRY:
                    ret = NV_ABORT_DEBUG;
                    break;
                case IDIGNORE:
                    ret = NV_ABORT_IGNORE;
                    break;
                case IDABORT:
                default:
                    ret = NV_ABORT_EXIT;
                    break;
                }
                /*if( _CrtDbgReport( _CRT_ASSERT, file, line, module, exp ) == 1 ) {
                    return NV_ABORT_DEBUG;
                }*/
            }

            if (ret == NV_ABORT_EXIT) {
                // Exit cleanly.
                exit(EXIT_FAILURE + 1);
            }

            return ret;
        }
    };
#elif NV_OS_XBOX

    /** Xbox360 assert handler. */
    struct Xbox360AssertHandler : public AssertHandler 
    {
        // Assert handler method.
        virtual int assertion(const char * exp, const char * file, int line, const char * func, const char * msg, va_list arg)
        {
            int ret = NV_ABORT_EXIT;

            StringBuilder error_string;
            if( func != NULL ) {
                error_string.format( "*** Assertion failed: %s\n    On file: %s\n    On function: %s\n    On line: %d\n ", exp, file, func, line );
                nvDebug( error_string.str() );
            }
            else {
                error_string.format( "*** Assertion failed: %s\n    On file: %s\n    On line: %d\n ", exp, file, line );
                nvDebug( error_string.str() );
            }

            if (debug::isDebuggerPresent()) {
                return NV_ABORT_DEBUG;
            }

            if( ret == NV_ABORT_EXIT ) {
                 // Exit cleanly.
                exit(EXIT_FAILURE + 1);
            }

            return ret;
        }
    };
#elif NV_OS_ORBIS || NV_OS_DURANGO

    /** Console assert handler. */
    struct ConsoleAssertHandler : public AssertHandler
    {
        // Assert handler method.
        virtual int assertion(const char * exp, const char * file, int line, const char * func, const char * msg, va_list arg)
        {
            if( func != NULL ) {
                nvDebug( "*** Assertion failed: %s\n    On file: %s\n    On function: %s\n    On line: %d\n ", exp, file, func, line );
            }
            else {
                nvDebug( "*** Assertion failed: %s\n    On file: %s\n    On line: %d\n ", exp, file, line );
            }

            //SBtodoORBIS print stack trace
            /*if (hasStackTrace())
            {
                void * trace[64];
                int size = backtrace(trace, 64);
                printStackTrace(trace, size, 2);
            }*/
            
            if (debug::isDebuggerPresent())
                return NV_ABORT_DEBUG;

            return NV_ABORT_IGNORE;
        }
    };

#else

    /** Unix assert handler. */
    struct UnixAssertHandler : public AssertHandler
    {
        // Assert handler method.
        virtual int assertion(const char * exp, const char * file, int line, const char * func, const char * msg, va_list arg)
        {
            int ret = NV_ABORT_EXIT;            
            
            if( func != NULL ) {
                nvDebug( "*** Assertion failed: %s\n    On file: %s\n    On function: %s\n    On line: %d\n ", exp, file, func, line );
            }
            else {
                nvDebug( "*** Assertion failed: %s\n    On file: %s\n    On line: %d\n ", exp, file, line );
            }

#if _DEBUG
            if (debug::isDebuggerPresent()) {
                return NV_ABORT_DEBUG;
            }
#endif

#if defined(NV_HAVE_EXECINFO_H)
            if (hasStackTrace())
            {
                void * trace[64];
                int size = backtrace(trace, 64);
                printStackTrace(trace, size, 2);
            }
#endif

            if( ret == NV_ABORT_EXIT ) {
                // Exit cleanly.
                exit(EXIT_FAILURE + 1);
            }
            
            return ret;
        }
    };

#endif

} // namespace


/// Handle assertion through the assert handler.
int nvAbort(const char * exp, const char * file, int line, const char * func/*=NULL*/, const char * msg/*= NULL*/, ...)
{
#if NV_OS_WIN32 //&& NV_CC_MSVC
    static Win32AssertHandler s_default_assert_handler;
#elif NV_OS_XBOX
    static Xbox360AssertHandler s_default_assert_handler;
#elif NV_OS_ORBIS || NV_OS_DURANGO
    static ConsoleAssertHandler s_default_assert_handler;
#else
    static UnixAssertHandler s_default_assert_handler;
#endif

    va_list arg;
    va_start(arg,msg);

    AssertHandler * handler = s_assert_handler != NULL ? s_assert_handler : &s_default_assert_handler;
    int result = handler->assertion(exp, file, line, func, msg, arg);

    va_end(arg);

    return result;
}

// Abnormal termination. Create mini dump and output call stack.
void debug::terminate(int code)
{
#if NV_OS_WIN32 || NV_OS_DURANGO
    EnterCriticalSection(&s_handler_critical_section);

    writeMiniDump(NULL);

#if NV_OS_WIN32
    const int max_stack_size = 64;
    void * trace[max_stack_size];
    int size = backtrace(trace, max_stack_size);

    // @@ Use win32's CreateFile?
    FILE * fp = fileOpen("crash.txt", "wb");
    if (fp != NULL) {
        Array<const char *> lines;
        writeStackTrace(trace, size, 0, lines);

        for (uint i = 0; i < lines.count(); i++) {
            fputs(lines[i], fp);
            delete lines[i];
        }

        // @@ Add more info to crash.txt?

        fclose(fp);
    }
#endif

    LeaveCriticalSection(&s_handler_critical_section);
#endif

    exit(code);
}


/// Shows a message through the message handler.
void NV_CDECL nvDebugPrint(const char *msg, ...)
{
    va_list arg;
    va_start(arg,msg);
    if (s_message_handler != NULL) {
        s_message_handler->log( msg, arg );
    }
    else {
        vprintf(msg, arg);
    }
    va_end(arg);
}


/// Dump debug info.
void debug::dumpInfo()
{
#if (NV_OS_WIN32 && NV_CC_MSVC) || (defined(NV_HAVE_SIGNAL_H) && defined(NV_HAVE_EXECINFO_H))
    if (hasStackTrace())
    {
        void * trace[64];
        int size = backtrace(trace, 64);

        nvDebug( "\nDumping stacktrace:\n" );

        Array<const char *> lines;
        writeStackTrace(trace, size, 1, lines);

        for (uint i = 0; i < lines.count(); i++) {
            nvDebug("%s", lines[i]);
            delete lines[i];
        }
    }
#endif
}

/// Dump callstack using the specified handler.
void debug::dumpCallstack(MessageHandler *messageHandler, int callstackLevelsToSkip /*= 0*/)
{
#if (NV_OS_WIN32 && NV_CC_MSVC) || (defined(NV_HAVE_SIGNAL_H) && defined(NV_HAVE_EXECINFO_H))
    if (hasStackTrace())
    {
        void * trace[64];
        int size = backtrace(trace, 64);

        Array<const char *> lines;
        writeStackTrace(trace, size, callstackLevelsToSkip + 1, lines);     // + 1 to skip the call to dumpCallstack

        for (uint i = 0; i < lines.count(); i++) {
            messageHandler->log(lines[i], NULL);
            delete lines[i];
        }
    }
#endif
}


/// Set the debug message handler.
void debug::setMessageHandler(MessageHandler * message_handler)
{
    s_message_handler = message_handler;
}

/// Reset the debug message handler.
void debug::resetMessageHandler()
{
    s_message_handler = NULL;
}

/// Set the assert handler.
void debug::setAssertHandler(AssertHandler * assert_handler)
{
    s_assert_handler = assert_handler;
}

/// Reset the assert handler.
void debug::resetAssertHandler()
{
    s_assert_handler = NULL;
}

#if NV_OS_WIN32 || NV_OS_DURANGO
#if NV_USE_SEPARATE_THREAD

static void initHandlerThread()
{
    static const int kExceptionHandlerThreadInitialStackSize = 64 * 1024;

    // Set synchronization primitives and the handler thread.  Each
    // ExceptionHandler object gets its own handler thread because that's the
    // only way to reliably guarantee sufficient stack space in an exception,
    // and it allows an easy way to get a snapshot of the requesting thread's
    // context outside of an exception.
    InitializeCriticalSection(&s_handler_critical_section);
    
    s_handler_start_semaphore = CreateSemaphoreExW(NULL, 0, 1, NULL, 0,
        SEMAPHORE_MODIFY_STATE | DELETE | SYNCHRONIZE);
    nvDebugCheck(s_handler_start_semaphore != NULL);

    s_handler_finish_semaphore = CreateSemaphoreExW(NULL, 0, 1, NULL, 0,
        SEMAPHORE_MODIFY_STATE | DELETE | SYNCHRONIZE);
    nvDebugCheck(s_handler_finish_semaphore != NULL);

    // Don't attempt to create the thread if we could not create the semaphores.
    if (s_handler_finish_semaphore != NULL && s_handler_start_semaphore != NULL) {
        DWORD thread_id;
        s_handler_thread = CreateThread(NULL,         // lpThreadAttributes
                                        kExceptionHandlerThreadInitialStackSize,
                                        ExceptionHandlerThreadMain,
                                        NULL,         // lpParameter
                                        0,            // dwCreationFlags
                                        &thread_id);
        nvDebugCheck(s_handler_thread != NULL);
    }

    /* @@ We should avoid loading modules in the exception handler!
    dbghelp_module_ = LoadLibrary(L"dbghelp.dll");
    if (dbghelp_module_) {
        minidump_write_dump_ = reinterpret_cast<MiniDumpWriteDump_type>(GetProcAddress(dbghelp_module_, "MiniDumpWriteDump"));
    }
    */
}

static void shutHandlerThread() {
    // @@ Free stuff. Terminate thread.
}

#endif // NV_USE_SEPARATE_THREAD
#endif // NV_OS_WIN32


// Enable signal handler.
void debug::enableSigHandler(bool interactive)
{
    if (s_sig_handler_enabled) return;

    s_sig_handler_enabled = true;
    s_interactive = interactive;

#if (NV_OS_WIN32 && NV_CC_MSVC) || NV_OS_DURANGO
    if (interactive) {
#if NV_OS_WIN32
        // Do not display message boxes on error.
        // http://msdn.microsoft.com/en-us/library/windows/desktop/ms680621(v=vs.85).aspx
        SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
#endif

        // CRT reports errors to debug output only.
        // http://msdn.microsoft.com/en-us/library/1y71x448(v=vs.80).aspx
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
        _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    }


#if NV_USE_SEPARATE_THREAD
    initHandlerThread();
#else
    InitializeCriticalSection(&s_handler_critical_section);
#endif

    s_old_exception_filter = ::SetUnhandledExceptionFilter( handleException );

#if _MSC_VER >= 1400  // MSVC 2005/8
    _set_invalid_parameter_handler(handleInvalidParameter);
#endif  // _MSC_VER >= 1400

    _set_purecall_handler(handlePureVirtualCall);

#if NV_OS_WIN32
    // SYMOPT_DEFERRED_LOADS make us not take a ton of time unless we actual log traces
    SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_FAIL_CRITICAL_ERRORS|SYMOPT_LOAD_LINES|SYMOPT_UNDNAME);

    if (!SymInitialize(GetCurrentProcess(), NULL, TRUE)) {
        DWORD error = GetLastError();
        nvDebug("SymInitialize returned error : %d\n", error);
    }
#endif

#elif !NV_OS_WIN32 && defined(NV_HAVE_SIGNAL_H)

    // Install our signal handler
    struct sigaction sa;
    sa.sa_sigaction = nvSigHandler;
    sigemptyset (&sa.sa_mask);
    sa.sa_flags = SA_ONSTACK | SA_RESTART | SA_SIGINFO;

    sigaction(SIGSEGV, &sa, &s_old_sigsegv);
    sigaction(SIGTRAP, &sa, &s_old_sigtrap);
    sigaction(SIGFPE, &sa, &s_old_sigfpe);
    sigaction(SIGBUS, &sa, &s_old_sigbus);

#endif
}

/// Disable signal handler.
void debug::disableSigHandler()
{
    nvCheck(s_sig_handler_enabled == true);
    s_sig_handler_enabled = false;

#if (NV_OS_WIN32 && NV_CC_MSVC) || NV_OS_DURANGO

    ::SetUnhandledExceptionFilter( s_old_exception_filter );
    s_old_exception_filter = NULL;

#if NV_OS_WIN32
    SymCleanup(GetCurrentProcess());
#endif

#elif !NV_OS_WIN32 && defined(NV_HAVE_SIGNAL_H)

    sigaction(SIGSEGV, &s_old_sigsegv, NULL);
    sigaction(SIGTRAP, &s_old_sigtrap, NULL);
    sigaction(SIGFPE, &s_old_sigfpe, NULL);
    sigaction(SIGBUS, &s_old_sigbus, NULL);

#endif
}


bool debug::isDebuggerPresent()
{
#if NV_OS_WIN32
    HINSTANCE kernel32 = GetModuleHandleA("kernel32.dll");
    if (kernel32) {
        FARPROC IsDebuggerPresent = GetProcAddress(kernel32, "IsDebuggerPresent");
        if (IsDebuggerPresent != NULL && IsDebuggerPresent()) {
            return true;
        }
    }
    return false;
#elif NV_OS_XBOX
#ifdef _DEBUG
    return DmIsDebuggerPresent() == TRUE;
#else
    return false;
#endif
#elif NV_OS_ORBIS
  #if PS4_FINAL_REQUIREMENTS
    return false; 
  #else
    return sceDbgIsDebuggerAttached() == 1;
  #endif
#elif NV_OS_DURANGO
  #if XB1_FINAL_REQUIREMENTS
    return false;
  #else
    return IsDebuggerPresent() == TRUE;
  #endif
#elif NV_OS_DARWIN
    int mib[4];
    struct kinfo_proc info;
    size_t size;
    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = getpid();
    size = sizeof(info);
    info.kp_proc.p_flag = 0;
    sysctl(mib,4,&info,&size,NULL,0);
    return ((info.kp_proc.p_flag & P_TRACED) == P_TRACED);
#else
    // if ppid != sid, some process spawned our app, probably a debugger. 
    return getsid(getpid()) != getppid();
#endif
}

bool debug::attachToDebugger()
{
#if NV_OS_WIN32
    if (isDebuggerPresent() == FALSE) {
        Path process(1024);
        process.copy("\"");
        GetSystemDirectoryA(process.str() + 1, 1024 - 1);

        process.appendSeparator();

        process.appendFormat("VSJitDebugger.exe\" -p %lu", ::GetCurrentProcessId());

        STARTUPINFOA sSi;
        memset(&sSi, 0, sizeof(sSi));

        PROCESS_INFORMATION sPi;
        memset(&sPi, 0, sizeof(sPi));
        
        BOOL b = CreateProcessA(NULL, process.str(), NULL, NULL, FALSE, 0, NULL, NULL, &sSi, &sPi);
        if (b != FALSE) {
            ::WaitForSingleObject(sPi.hProcess, INFINITE);
            
            DWORD dwExitCode;
            ::GetExitCodeProcess(sPi.hProcess, &dwExitCode);
            if (dwExitCode != 0) //if exit code is zero, a debugger was selected
                b = FALSE;
        }

        if (sPi.hThread != NULL) ::CloseHandle(sPi.hThread);
        if (sPi.hProcess != NULL) ::CloseHandle(sPi.hProcess);

        if (b == FALSE)
            return false;

        for (int i = 0; i < 5*60; i++) {
            if (isDebuggerPresent())
                break;
            ::Sleep(200);
        }
    }
#endif // NV_OS_WIN32

    return true;
}
