#pragma once
#include "libriscv_settings.h" // Build-system generated

#include <memory>
#include <type_traits>
#if __has_include(<span>)
# include <span>
# if defined(cpp_lib_span) || defined(__cpp_lib_concepts)
#  define RISCV_SPAN_AVAILABLE 1
# endif
#endif
#include <string>
#include <string_view>
#include <vector>
#include <variant>
#include "util/function.hpp"
#include "types.hpp"

#ifndef RISCV_SYSCALLS_MAX
#define RISCV_SYSCALLS_MAX   512
#endif

#ifndef RISCV_SYSCALL_EBREAK_NR
#define RISCV_SYSCALL_EBREAK_NR    (RISCV_SYSCALLS_MAX-1)
#endif

#ifndef RISCV_PAGE_SIZE
#define RISCV_PAGE_SIZE  4096UL
#endif

#ifndef RISCV_FORCE_ALIGN_MEMORY
#define RISCV_FORCE_ALIGN_MEMORY 1
#endif

#ifndef RISCV_MACHINE_ALIGNMENT
#define RISCV_MACHINE_ALIGNMENT 32
#endif

#ifndef RISCV_BRK_MEMORY_SIZE
#define RISCV_BRK_MEMORY_SIZE  (16ull << 20) // 16MB
#endif

#ifndef RISCV_MAX_EXECUTE_SEGS
#define RISCV_MAX_EXECUTE_SEGS  16
#endif

namespace riscv
{
	template <int W> struct Memory;

	struct MachineTranslationCrossOptions
	{
		/// @brief Provide a custom binary-translation compiler in order
		/// to produce a secondary binary that can be loaded on Windows machines.
		/// @example "x86_64-w64-mingw32-gcc"
		std::string cross_compiler = "x86_64-w64-mingw32-gcc";

		/// @brief Provide a custom prefix for the mingw PE-dll output.
		/// @example "rvbintr-"
		std::string cross_prefix = "rvbintr-";

		/// @brief Provide a custom suffix for the mingw PE-dll output.
		/// @example ".dll"
		std::string cross_suffix = ".dll";
	};
	/// @brief Options for generating embeddable C99 code into a C or C++ program.
	struct MachineTranslationEmbeddableCodeOptions
	{
		/// @brief Provide a filename prefix for the embedded code output.
		/// @example "mycode-"
		std::string prefix = "mycode-";

		/// @brief Provide a filename suffix for the embedded code output.
		/// @example ".c" or ".cpp"
		std::string suffix = ".cpp";

		/// @brief An optional std::string pointer to write the output code to,
		/// instead of writing to a file.
		/// @details Puts freestanding C99 code into the std::string pointer.
		std::string* result_c99 = nullptr;
	};
	using MachineTranslationOptions = std::variant<MachineTranslationCrossOptions, MachineTranslationEmbeddableCodeOptions>;

	/// @brief Options passed to Machine constructor
	/// @tparam W The RISC-V architecture
	template <int W>
	struct MachineOptions
	{
		/// @brief Maximum memory used by the machine, rounded down to
		/// the current page size (4kb).
		uint64_t memory_max = 64ull << 20; // 64MB

		/// @brief Virtual memory allocated for the main stack at construction.
		uint32_t stack_size = 1ul << 20; // 1MB default stack

		/// @brief Setting this option will load the binary at construction as if it
		/// was a RISC-V ELF binary. When disabled, no loading occurs.
		bool load_program = true;

		/// @brief Setting this option will apply page protections based on ELF segments
		/// from the program loaded at construction.
		bool protect_segments = true;

		/// @brief Enabling this will allow unsafe RWX segments (read-write-execute).
		bool allow_write_exec_segment = false;

		/// @brief Enabling this will enforce execute-only segments (X ^ R).
		bool enforce_exec_only = false;

		/// @brief Ignore .text section, as if not all executable code is in it.
		/// Instead, load all executable segments as normal. Some programs require using
		/// the .text section in order to get correctly aligned instructions.
		bool ignore_text_section = false;

		/// @brief Print some verbose loader information to stdout.
		/// @details If binary translation is enabled, this will also make the
		/// binary translation process print verbose information.
		bool verbose_loader = false;

		/// @brief Enabling this will skip assignment of copy-on-write pages
		/// to forked machines from the main machine, making fork operations faster,
		/// but requires the forks to fault in pages instead (slower).
		bool minimal_fork = false;

		/// @brief Create a linear memory arena for main memory, increasing memory
		/// locality and also enables read-write arena if the CMake option is ON.
		bool use_memory_arena = true;

		/// @brief Enable sharing of execute segments between machines.
		/// @details This will allow multiple machines to share the same execute
		/// segment, reducing memory usage and increasing performance.
		/// When binary translation is enabled, this will also share the dynamically
		/// translated code between machines. (Prevents some optimizations)
		bool use_shared_execute_segments = true;

		/// @brief Override a default-injected exit function with another function
		/// that is found by looking up the provided symbol name in the current program.
		/// Eg. if default_exit_function is "fast_exit", then the ELF binary must have
		/// that symbol visible in its .symbtab ELF section.
		std::string_view default_exit_function {};

		/// @brief Provide a custom page-fault handler at construction.
		riscv::Function<struct Page&(Memory<W>&, address_type<W>, bool)> page_fault_handler = nullptr;

		/// @brief Call ebreak for each of the addresses in the vector.
		/// @details This is useful for debugging and live-patching programs.
		std::vector<std::variant<address_type<W>, std::string>> ebreak_locations {};

#ifdef RISCV_BINARY_TRANSLATION
		/// @brief Enable the binary translator.
		bool translate_enabled = true;
		/// @brief Enable loading of embedded binary translated programs.
		/// @details This will allow the machine to load and execute *embedded*
		/// binary translated programs. They auto-register themselves.
		bool translate_enable_embedded = true;
		/// @brief Translate not just the initial execute segments of the ELF program,
		/// but also any future shared objects or JIT-produced segments.
		bool translate_future_segments = true;
		/// @brief Enable compiling execute segment on-demand during emulation.
		/// @details Not available on most Windows systems.
#if defined(_WIN32) && !defined(RISCV_LIBTCC)
		bool translate_invoke_compiler = false;
#else
		bool translate_invoke_compiler = true;
#endif
		/// @brief Enable tracing during emulation of the binary translated parts of the program.
		bool translate_trace  = false;
		/// @brief Enable verbose timing information for the binary translator.
		bool translate_timing = false;
		/// @brief Enable the translation cache for the binary translator.
		/// Translated shared objects will be stored in a file and can be re-used later.
		/// @details When TCC is enabled, the translation cache will be disabled.
		bool translation_cache = true;
		/// @brief Enable the use of the memory arena for the binary translator.
		/// @details If disabled, remote machines will be able to make remote
		/// calls to this machine. In most cases, this is not needed.
		bool translation_use_arena = true;
		/// @brief Allow the program to run forever, ignoring the instruction counter limit.
		/// @details This is useful when there are other ways of interrupting and cancelling the program.
		/// @note This option is only available when the binary translator is enabled. The main dispatch
		/// will always check the instruction counter limit.
		/// It is completely fine to enable this option when running from the command line,
		/// as a simple Ctrl+C will stop the program.
		bool translate_ignore_instruction_limit = false;
		/// @brief Enable the use of register caching for the binary translator. Always enabled
		/// when binary translation with libtcc is enabled.
		/// @details Enable this when compiling with -O0 or when using simple compilers like TCC.
#ifdef RISCV_LIBTCC
		bool translate_use_register_caching = true;
#else
		bool translate_use_register_caching = false;
#endif
		bool translate_use_syscall_clobbering_optimization = false;
		/// @brief Enable automatic n-bit address space for the binary translator by rounding down to the nearest power of 2.
		/// @details This will allow the binary translator to use and-masked addresses
		/// for all memory accesses, which can drastically improve performance.
		bool translate_automatic_nbit_address_space = false;
		/// @brief Enable unsafe removal of checks in the binary translator.
		/// @details This will remove checks that prevent the program from crashing, such
		/// as memory access checks, and other checks that sandboxes normally provide.
		bool translate_unsafe_remove_checks = false;
		/// @brief Enable recording of slowpaths to jump hints for the binary translator.
		/// @note This option is only available when RISCV_DEBUG and the binary translator is enabled.
		/// @details This will record slowpaths to the MachineOptions jump hints vector.
		/// From there the CLI can save the jump hints to a file after the program has run.
		bool record_slowpaths_to_jump_hints = false;
		/// @brief Enable live-patching a running instruction stream after background compilation.
		/// @details This will allow the binary translator to patch the running instruction stream
		/// with the newly compiled code, which allows it to switch to the newly compiled code
		/// without stopping the program.
		/// @warning This is an experimental feature and may not work correctly in all cases.
		/// @note Disabling this option will not disable background compilation, it will just
		/// not hot-reload the execution with the new binary translation when the emulator is running.
		bool translate_live_patching = true;
		/// @brief Prefix for the translation output file.
		std::string translation_prefix = "/tmp/rvbintr-";
		/// @brief Suffix for the translation output file. Eg. .dll or .so
		std::string translation_suffix = "";
		/// @brief Limits placed on the binary translator.
		/// @details The binary translator will stop translating after reaching
		/// either of these limits. The limits are per shared object.
		unsigned translate_blocks_max = 1024;
		unsigned translate_instr_max = 500'000;
		/// @brief Jump location hints for the binary translator.
		/// @details These hints can improve performance of the binary translation.
		std::vector<address_type<W>> translator_jump_hints {};
		/// @brief Enable background compilation of shared objects. The compilation step
		/// will be executed from a user-provided callback, and will be applied to the machine
		/// when ready. Applying the translation is thread-safe and will take effect on all
		/// machines using the translated execute segment, even while they are executing.
		/// For short-lived programs, this feature should be disabled, as it often takes more
		/// time to translate and compile than to execute the program.
		std::function<void(std::function<void()>& compilation_step)> translate_background_callback = nullptr;
		/// @brief Allow the production of a secondary dependency-free DLL that can be
		/// transferred to and loaded on Windows (or other) machines. It will be used
		/// to greatly accelerate the emulation of the RISC-V program.
		std::vector<MachineTranslationOptions> cross_compile {};

		/// @brief Produce the translation output filename from the prefix, hash and suffix.
		/// @param prefix A prefix for the filename.
		/// @param hash   A hash to include in the filename, retrieved from the execute segment.
		/// @param suffix A suffix for the filename.
		/// @return A filename string.
		/// @details The filename will be constructed as follows:
		/// @code
		/// const uint32_t hash = machine.current_execute_segment().translation_hash();
		/// char buffer[256];
		/// const int len = snprintf(buffer, sizeof(buffer), "%s%08x%s",
		/// 	prefix.c_str(), hash, suffix.c_str());
		/// return std::string(buffer, len);
		/// @endcode
		/// @note The hash is a CRC32-C of the execute segment + emulator settings.
		/// @note The hash can be found with machine.current_execute_segment().translation_hash()
		static std::string translation_filename(const std::string& prefix, uint32_t hash, const std::string& suffix);

#ifdef RISCV_LIBTCC
		/// @brief Provide a custom libtcc1 location for the binary translator.
		std::string libtcc1_location {};
#endif
#endif
	};

	static constexpr int SYSCALL_EBREAK = RISCV_SYSCALL_EBREAK_NR;

	static constexpr size_t PageSize = RISCV_PAGE_SIZE;
	static constexpr size_t PageMask = RISCV_PAGE_SIZE-1;

#ifdef RISCV_MEMORY_TRAPS
	static constexpr bool memory_traps_enabled = true;
#else
	static constexpr bool memory_traps_enabled = false;
#endif

#if RISCV_FORCE_ALIGN_MEMORY
	static constexpr bool force_align_memory = true;
#else
	static constexpr bool force_align_memory = false;
#endif

#ifdef RISCV_DEBUG
	static constexpr bool memory_alignment_check = true;
	static constexpr bool verbose_branches_enabled = false;
	static constexpr bool unaligned_memory_slowpaths = true;
	static constexpr bool nanboxing = true;
#else
	static constexpr bool memory_alignment_check = false;
	static constexpr bool verbose_branches_enabled = false;
	static constexpr bool unaligned_memory_slowpaths = false;
#ifdef RISCV_ALWAYS_NANBOXING // In order to override the default
	static constexpr bool nanboxing = true;
#else
	static constexpr bool nanboxing = false;
#endif
#endif

#ifdef RISCV_EXT_A
#define RISCV_EXT_ATOMICS
	static constexpr bool atomics_enabled = true;
#else
	static constexpr bool atomics_enabled = false;
#endif
#ifdef RISCV_EXT_C
#define RISCV_EXT_COMPRESSED
	static constexpr bool compressed_enabled = true;
#else
	static constexpr bool compressed_enabled = false;
#endif
#ifdef RISCV_EXT_V
#define RISCV_EXT_VECTOR 32
	static constexpr unsigned vector_extension = RISCV_EXT_VECTOR;
#else
	static constexpr unsigned vector_extension = 0;
#endif
#ifdef RISCV_128I
#define RISCV_128BIT_ISA
	static constexpr bool rv128i_enabled = true;
#else
	static constexpr bool rv128i_enabled = false;
#endif
#ifdef RISCV_FCSR
	static constexpr bool fcsr_emulation = true;
#else
	static constexpr bool fcsr_emulation = false;
#endif
#ifdef RISCV_BINARY_TRANSLATION
	static constexpr bool binary_translation_enabled = true;
#else
	static constexpr bool binary_translation_enabled = false;
#endif
#ifdef RISCV_FLAT_RW_ARENA
	static constexpr bool flat_readwrite_arena = true;
#else
	static constexpr bool flat_readwrite_arena = false;
#endif
#ifdef RISCV_ENCOMPASSING_ARENA_BITS
	static constexpr int encompassing_Nbit_arena = RISCV_ENCOMPASSING_ARENA_BITS;
	static constexpr uint64_t encompassing_arena_mask = (1ull << RISCV_ENCOMPASSING_ARENA_BITS) - 1;
#else
	static constexpr int encompassing_Nbit_arena = 0;
	static constexpr uint64_t encompassing_arena_mask = 0;
#endif
#ifdef RISCV_LIBTCC
	static constexpr bool libtcc_enabled = true;
#else
	static constexpr bool libtcc_enabled = false;
#endif


	template <int W> struct MultiThreading;
	template <int W> struct SerializedMachine;
	struct Arena;

	template <typename T>
	using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

	template <class...> constexpr std::false_type always_false {};

	template<typename T>
	struct is_string
		: public std::disjunction<
			std::is_same<char *, typename std::decay<T>::type>,
			std::is_same<const char *, typename std::decay<T>::type>
	> {};

	template<class T>
	struct is_stdstring : public std::is_same<T, std::basic_string<char>> {};

	template<class T>
	struct is_stdvector : public std::false_type {};

	template<class T>
	struct is_stdvector<std::vector<T>> : public std::true_type {};

	template<class T>
	struct is_stdarray_ptr : std::false_type {};

	template<class T, std::size_t N>
	struct is_stdarray_ptr<std::array<T, N>*> : std::true_type {};

	template<class T>
	constexpr bool is_stdarray_ptr_v = is_stdarray_ptr<T>::value;

	template <typename T>
	struct is_span : std::false_type{};
#ifdef RISCV_SPAN_AVAILABLE
	template <typename T>
	struct is_span<std::span<T>> : std::true_type{};
	template <typename T>
	constexpr bool is_span_v = is_span<T>::value;
#endif
} // riscv

#ifdef __GNUG__

#ifndef LIKELY
#define LIKELY(x) __builtin_expect((x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect((x), 0)
#endif
#ifndef RISCV_COLD_PATH
#define RISCV_COLD_PATH() __attribute__((cold))
#endif
#ifndef RISCV_HOT_PATH
#define RISCV_HOT_PATH() __attribute__((hot))
#endif
#define RISCV_ALWAYS_INLINE __attribute__((always_inline))
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#define RISCV_COLD_PATH() /* */
#define RISCV_HOT_PATH()  /* */
#define RISCV_ALWAYS_INLINE /* */
#endif

#ifdef _MSC_VER
#undef RISCV_ALWAYS_INLINE
#define RISCV_ALWAYS_INLINE __forceinline
#define RISCV_NOINLINE      __declspec(noinline)
#endif

#ifdef __HAVE_BUILTIN_SPECULATION_SAFE_VALUE
#define RISCV_SPECSAFE(x) __builtin_speculation_safe_value(x)
#else
#define RISCV_SPECSAFE(x) (x)
#endif

#ifndef RISCV_INTERNAL
#if defined(__GNUG__) && !defined(_WIN32)
#define RISCV_INTERNAL __attribute__((visibility("internal")))
#else
#define RISCV_INTERNAL /* */
#endif
#endif
