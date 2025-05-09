/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include <SDL3/SDL_test.h>

#ifdef HAVE_LIBUNWIND_H
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#ifndef unw_get_proc_name_by_ip
#define SDLTEST_UNWIND_NO_PROC_NAME_BY_IP
static bool s_unwind_symbol_names = true;
#endif
#endif

#ifdef SDL_PLATFORM_WIN32
#include <windows.h>
#include <dbghelp.h>

static struct {
    SDL_SharedObject *module;
    BOOL (WINAPI *pSymInitialize)(HANDLE hProcess, PCSTR UserSearchPath, BOOL fInvadeProcess);
    BOOL (WINAPI *pSymFromAddr)(HANDLE hProcess, DWORD64 Address, PDWORD64 Displacement, PSYMBOL_INFO Symbol);
    BOOL (WINAPI *pSymGetLineFromAddr64)(HANDLE hProcess, DWORD64 qwAddr, PDWORD pdwDisplacement, PIMAGEHLP_LINE64 Line);
} dyn_dbghelp;

/* older SDKs might not have this: */
__declspec(dllimport) USHORT WINAPI RtlCaptureStackBackTrace(ULONG FramesToSkip, ULONG FramesToCapture, PVOID* BackTrace, PULONG BackTraceHash);
#define CaptureStackBackTrace RtlCaptureStackBackTrace

#endif

/* This is a simple tracking allocator to demonstrate the use of SDL's
   memory allocation replacement functionality.

   It gets slow with large numbers of allocations and shouldn't be used
   for production code.
*/

#define MAXIMUM_TRACKED_STACK_DEPTH 32

typedef struct SDL_tracked_allocation
{
    void *mem;
    size_t size;
    Uint64 stack[MAXIMUM_TRACKED_STACK_DEPTH];
    struct SDL_tracked_allocation *next;
#ifdef SDLTEST_UNWIND_NO_PROC_NAME_BY_IP
    char stack_names[MAXIMUM_TRACKED_STACK_DEPTH][256];
#endif
} SDL_tracked_allocation;

static SDLTest_Crc32Context s_crc32_context;
static SDL_malloc_func SDL_malloc_orig = NULL;
static SDL_calloc_func SDL_calloc_orig = NULL;
static SDL_realloc_func SDL_realloc_orig = NULL;
static SDL_free_func SDL_free_orig = NULL;
static int s_previous_allocations = 0;
static int s_unknown_frees = 0;
static SDL_tracked_allocation *s_tracked_allocations[256];
static bool s_randfill_allocations = false;
static SDL_AtomicInt s_lock;

#define LOCK_ALLOCATOR()                               \
    do {                                               \
        if (SDL_CompareAndSwapAtomicInt(&s_lock, 0, 1)) { \
            break;                                     \
        }                                              \
        SDL_CPUPauseInstruction();                     \
    } while (true)
#define UNLOCK_ALLOCATOR() do { SDL_SetAtomicInt(&s_lock, 0); } while (0)

static unsigned int get_allocation_bucket(void *mem)
{
    CrcUint32 crc_value;
    unsigned int index;
    SDLTest_Crc32Calc(&s_crc32_context, (CrcUint8 *)&mem, sizeof(mem), &crc_value);
    index = (crc_value & (SDL_arraysize(s_tracked_allocations) - 1));
    return index;
}

static SDL_tracked_allocation* SDL_GetTrackedAllocation(void *mem)
{
    SDL_tracked_allocation *entry;
    LOCK_ALLOCATOR();
    int index = get_allocation_bucket(mem);
    for (entry = s_tracked_allocations[index]; entry; entry = entry->next) {
        if (mem == entry->mem) {
            UNLOCK_ALLOCATOR();
            return entry;
        }
    }
    UNLOCK_ALLOCATOR();
    return NULL;
}

static size_t SDL_GetTrackedAllocationSize(void *mem)
{
    SDL_tracked_allocation *entry = SDL_GetTrackedAllocation(mem);

    return entry ? entry->size : SIZE_MAX;
}

static bool SDL_IsAllocationTracked(void *mem)
{
    return SDL_GetTrackedAllocation(mem) != NULL;
}

static void SDL_TrackAllocation(void *mem, size_t size)
{
    SDL_tracked_allocation *entry;
    int index = get_allocation_bucket(mem);

    if (SDL_IsAllocationTracked(mem)) {
        return;
    }
    entry = (SDL_tracked_allocation *)SDL_malloc_orig(sizeof(*entry));
    if (!entry) {
        return;
    }
    LOCK_ALLOCATOR();
    entry->mem = mem;
    entry->size = size;

    /* Generate the stack trace for the allocation */
    SDL_zeroa(entry->stack);
#ifdef HAVE_LIBUNWIND_H
    {
        int stack_index;
        unw_cursor_t cursor;
        unw_context_t context;

        unw_getcontext(&context);
        unw_init_local(&cursor, &context);

        stack_index = 0;
        while (unw_step(&cursor) > 0) {
            unw_word_t pc;
#ifdef SDLTEST_UNWIND_NO_PROC_NAME_BY_IP
            unw_word_t offset;
            char sym[236];
#endif

            unw_get_reg(&cursor, UNW_REG_IP, &pc);
            entry->stack[stack_index] = pc;

#ifdef SDLTEST_UNWIND_NO_PROC_NAME_BY_IP
            if (s_unwind_symbol_names && unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
                SDL_snprintf(entry->stack_names[stack_index], sizeof(entry->stack_names[stack_index]), "%s+0x%llx", sym, (unsigned long long)offset);
            }
#endif
            ++stack_index;

            if (stack_index == SDL_arraysize(entry->stack)) {
                break;
            }
        }
    }
#elif defined(SDL_PLATFORM_WIN32)
    {
        Uint32 count;
        PVOID frames[63];
        Uint32 i;

        count = CaptureStackBackTrace(1, SDL_arraysize(frames), frames, NULL);

        count = SDL_min(count, MAXIMUM_TRACKED_STACK_DEPTH);
        for (i = 0; i < count; i++) {
            entry->stack[i] = (Uint64)(uintptr_t)frames[i];
        }
    }
#endif /* HAVE_LIBUNWIND_H */

    entry->next = s_tracked_allocations[index];
    s_tracked_allocations[index] = entry;
    UNLOCK_ALLOCATOR();
}

static void SDL_UntrackAllocation(void *mem)
{
    SDL_tracked_allocation *entry, *prev;
    int index = get_allocation_bucket(mem);

    LOCK_ALLOCATOR();
    prev = NULL;
    for (entry = s_tracked_allocations[index]; entry; entry = entry->next) {
        if (mem == entry->mem) {
            if (prev) {
                prev->next = entry->next;
            } else {
                s_tracked_allocations[index] = entry->next;
            }
            SDL_free_orig(entry);
            UNLOCK_ALLOCATOR();
            return;
        }
        prev = entry;
    }
    s_unknown_frees += 1;
    UNLOCK_ALLOCATOR();
}

static void rand_fill_memory(void* ptr, size_t start, size_t end)
{
    Uint8* mem = (Uint8*) ptr;
    size_t i;

    if (!s_randfill_allocations)
        return;

    for (i = start; i < end; ++i) {
        mem[i] = SDLTest_RandomUint8();
    }
}

static void * SDLCALL SDLTest_TrackedMalloc(size_t size)
{
    void *mem;

    mem = SDL_malloc_orig(size);
    if (mem) {
        SDL_TrackAllocation(mem, size);
        rand_fill_memory(mem, 0, size);
    }
    return mem;
}

static void * SDLCALL SDLTest_TrackedCalloc(size_t nmemb, size_t size)
{
    void *mem;

    mem = SDL_calloc_orig(nmemb, size);
    if (mem) {
        SDL_TrackAllocation(mem, nmemb * size);
    }
    return mem;
}

static void * SDLCALL SDLTest_TrackedRealloc(void *ptr, size_t size)
{
    void *mem;
    size_t old_size = 0;
    if (ptr) {
         old_size = SDL_GetTrackedAllocationSize(ptr);
         SDL_assert(old_size != SIZE_MAX);
    }
    mem = SDL_realloc_orig(ptr, size);
    if (ptr) {
        SDL_UntrackAllocation(ptr);
    }
    if (mem) {
        SDL_TrackAllocation(mem, size);
        if (size > old_size) {
            rand_fill_memory(mem, old_size, size);
        }
    }
    return mem;
}

static void SDLCALL SDLTest_TrackedFree(void *ptr)
{
    if (!ptr) {
        return;
    }

    if (s_previous_allocations == 0) {
        SDL_assert(SDL_IsAllocationTracked(ptr));
    }
    SDL_UntrackAllocation(ptr);
    SDL_free_orig(ptr);
}

void SDLTest_TrackAllocations(void)
{
    if (SDL_malloc_orig) {
        return;
    }

    SDLTest_Crc32Init(&s_crc32_context);

    s_previous_allocations = SDL_GetNumAllocations();
    if (s_previous_allocations < 0) {
        SDL_Log("SDL was built without allocation count support, disabling free() validation");
    } else if (s_previous_allocations != 0) {
        SDL_Log("SDLTest_TrackAllocations(): There are %d previous allocations, disabling free() validation", s_previous_allocations);
    }
#ifdef SDLTEST_UNWIND_NO_PROC_NAME_BY_IP
    do {
        /* Don't use SDL_GetHint: SDL_malloc is off limits. */
        const char *env_trackmem = SDL_getenv_unsafe("SDL_TRACKMEM_SYMBOL_NAMES");
        if (env_trackmem) {
            if (SDL_strcasecmp(env_trackmem, "1") == 0 || SDL_strcasecmp(env_trackmem, "yes") == 0 || SDL_strcasecmp(env_trackmem, "true") == 0) {
                s_unwind_symbol_names = true;
            } else if (SDL_strcasecmp(env_trackmem, "0") == 0 || SDL_strcasecmp(env_trackmem, "no") == 0 || SDL_strcasecmp(env_trackmem, "false") == 0) {
                s_unwind_symbol_names = false;
            }
        }
    } while (0);

#elif defined(SDL_PLATFORM_WIN32)
    do {
        dyn_dbghelp.module = SDL_LoadObject("dbghelp.dll");
        if (!dyn_dbghelp.module) {
            goto dbghelp_failed;
        }
        dyn_dbghelp.pSymInitialize = (void *)SDL_LoadFunction(dyn_dbghelp.module, "SymInitialize");
        dyn_dbghelp.pSymFromAddr = (void *)SDL_LoadFunction(dyn_dbghelp.module, "SymFromAddr");
        dyn_dbghelp.pSymGetLineFromAddr64 = (void *)SDL_LoadFunction(dyn_dbghelp.module, "SymGetLineFromAddr64");
        if (!dyn_dbghelp.pSymInitialize || !dyn_dbghelp.pSymFromAddr || !dyn_dbghelp.pSymGetLineFromAddr64) {
            goto dbghelp_failed;
        }
        if (!dyn_dbghelp.pSymInitialize(GetCurrentProcess(), NULL, TRUE)) {
            goto dbghelp_failed;
        }
        break;
dbghelp_failed:
        if (dyn_dbghelp.module) {
            SDL_UnloadObject(dyn_dbghelp.module);
            dyn_dbghelp.module = NULL;
        }
    } while (0);
#endif

    SDL_GetMemoryFunctions(&SDL_malloc_orig,
                           &SDL_calloc_orig,
                           &SDL_realloc_orig,
                           &SDL_free_orig);

    SDL_SetMemoryFunctions(SDLTest_TrackedMalloc,
                           SDLTest_TrackedCalloc,
                           SDLTest_TrackedRealloc,
                           SDLTest_TrackedFree);
}

void SDLTest_RandFillAllocations(void)
{
    SDLTest_TrackAllocations();

    s_randfill_allocations = true;
}

void SDLTest_LogAllocations(void)
{
    char *message = NULL;
    size_t message_size = 0;
    char line[256], *tmp;
    SDL_tracked_allocation *entry;
    int index, count, stack_index;
    Uint64 total_allocated;

    if (!SDL_malloc_orig) {
        return;
    }

    message = SDL_realloc_orig(NULL, 1);
    if (!message) {
        return;
    }
    *message = 0;

#define ADD_LINE()                                         \
    message_size += (SDL_strlen(line) + 1);                \
    tmp = (char *)SDL_realloc_orig(message, message_size); \
    if (!tmp) {                                            \
        return;                                            \
    }                                                      \
    message = tmp;                                         \
    SDL_strlcat(message, line, message_size)

    SDL_strlcpy(line, "Memory allocations:\n", sizeof(line));
    ADD_LINE();

    count = 0;
    total_allocated = 0;
    for (index = 0; index < SDL_arraysize(s_tracked_allocations); ++index) {
        for (entry = s_tracked_allocations[index]; entry; entry = entry->next) {
            (void)SDL_snprintf(line, sizeof(line), "Allocation %d: %d bytes\n", count, (int)entry->size);
            ADD_LINE();
            /* Start at stack index 1 to skip our tracking functions */
            for (stack_index = 1; stack_index < SDL_arraysize(entry->stack); ++stack_index) {
                char stack_entry_description[256] = "???";

                if (!entry->stack[stack_index]) {
                    break;
                }
#ifdef HAVE_LIBUNWIND_H
                {
#ifdef SDLTEST_UNWIND_NO_PROC_NAME_BY_IP
                    if (s_unwind_symbol_names) {
                        (void)SDL_snprintf(stack_entry_description, sizeof(stack_entry_description), "%s", entry->stack_names[stack_index]);
                    }
#else
                    char name[256] = "???";
                    unw_word_t offset = 0;
                    unw_get_proc_name_by_ip(unw_local_addr_space, entry->stack[stack_index], name, sizeof(name), &offset, NULL);
                    (void)SDL_snprintf(stack_entry_description, sizeof(stack_entry_description), "%s+0x%llx", name, (long long unsigned int)offset);
#endif
                }
#elif defined(SDL_PLATFORM_WIN32)
                {
                    DWORD64 dwDisplacement = 0;
                    char symbol_buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
                    PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)symbol_buffer;
                    DWORD lineColumn = 0;
                    pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
                    pSymbol->MaxNameLen = MAX_SYM_NAME;
                    IMAGEHLP_LINE64 dbg_line;
                    dbg_line.SizeOfStruct = sizeof(dbg_line);
                    dbg_line.FileName = "";
                    dbg_line.LineNumber = 0;

                    if (dyn_dbghelp.module) {
                        if (!dyn_dbghelp.pSymFromAddr(GetCurrentProcess(), entry->stack[stack_index], &dwDisplacement, pSymbol)) {
                            SDL_strlcpy(pSymbol->Name, "???", MAX_SYM_NAME);
                            dwDisplacement = 0;
                        }
                        dyn_dbghelp.pSymGetLineFromAddr64(GetCurrentProcess(), (DWORD64)entry->stack[stack_index], &lineColumn, &dbg_line);
                    }
                    SDL_snprintf(stack_entry_description, sizeof(stack_entry_description), "%s+0x%I64x %s:%u", pSymbol->Name, dwDisplacement, dbg_line.FileName, (Uint32)dbg_line.LineNumber);
                }
#endif
                (void)SDL_snprintf(line, sizeof(line), "\t0x%" SDL_PRIx64 ": %s\n", entry->stack[stack_index], stack_entry_description);

                ADD_LINE();
            }
            total_allocated += entry->size;
            ++count;
        }
    }
    (void)SDL_snprintf(line, sizeof(line), "Total: %.2f Kb in %d allocations", total_allocated / 1024.0, count);
    ADD_LINE();
    if (s_unknown_frees != 0) {
        (void)SDL_snprintf(line, sizeof(line), ", %d unknown frees", s_unknown_frees);
        ADD_LINE();
    }
    (void)SDL_snprintf(line, sizeof(line), "\n");
    ADD_LINE();
#undef ADD_LINE

    SDL_Log("%s", message);
    SDL_free_orig(message);
}
