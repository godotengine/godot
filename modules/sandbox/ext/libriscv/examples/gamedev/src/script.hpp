#pragma once
#include <any>
#include <functional>
#include <libriscv/machine.hpp>
#include <optional>
#include <stdexcept>
#include <unordered_set>
template <typename T> struct GuestObjects;

struct Script
{
	static constexpr int MARCH = 8; // 64-bit RISC-V

	/// @brief A virtual address inside a Script program
	using gaddr_t		= riscv::address_type<MARCH>;
	using sgaddr_t      = riscv::signed_address_type<MARCH>;
	/// @brief A virtual machine running a program
	using machine_t		= riscv::Machine<MARCH>;

	/// @brief The total physical memory of the program
	static constexpr gaddr_t MAX_MEMORY	= 1024 * 1024 * 16ull;
	/// @brief A virtual memory area set aside for the initial stack
	static constexpr gaddr_t STACK_SIZE	= 1024 * 1024 * 2ull;
	/// @brief A virtual memory area set aside for the heap
	static constexpr gaddr_t MAX_HEAP	= 1024 * 1024 * 256ull;
	/// @brief The max number of instructions allowed during startup
	static constexpr uint64_t MAX_BOOT_INSTR = 32'000'000ull;
	/// @brief The max number of instructions allowed during calls
	static constexpr uint64_t MAX_CALL_INSTR = 32'000'000ull;
	/// @brief The max number of recursive calls into the Machine allowed
	static constexpr uint8_t  MAX_CALL_DEPTH = 8;

	/// @brief Make a function call into the script
	/// @param func The function to call. Must be a visible symbol in the program.
	/// @param args The arguments to pass to the function.
	/// @return The optional integral return value.
	template <typename... Args>
	std::optional<Script::sgaddr_t> call(const std::string& func, Args&&... args);

	/// @brief Make a function call into the script
	/// @param addr The functions direct address.
	/// @param args The arguments to pass to the function.
	/// @return The optional integral return value.
	template <typename... Args>
	std::optional<Script::sgaddr_t> call(gaddr_t addr, Args&&... args);

	/// @brief Make a preempted function call into the script, saving and
	/// restoring the current execution state.
	/// Preemption allows callers to temporarily interrupt the virtual machine,
	/// such that it can be resumed like normal later on.
	/// @param func The function to call. Must be a visible symbol in the program.
	/// @param args The arguments to the function call.
	/// @return The optional integral return value.
	template <typename... Args>
	std::optional<Script::sgaddr_t> preempt(const std::string& func, Args&&... args);

	/// @brief Make a preempted function call into the script, saving and
	/// restoring the current execution state.
	/// Preemption allows callers to temporarily interrupt the virtual machine,
	/// such that it can be resumed like normal later on.
	/// @param addr The functions address to call.
	/// @param args The arguments to the function call.
	/// @return The optional integral return value.
	template <typename... Args>
	std::optional<Script::sgaddr_t> preempt(gaddr_t addr, Args&&... args);

	/// @brief Resume execution of the script, until @param instruction_count has been reached,
	/// then stop execution and return. This function can be used to drive long-running tasks
	/// over time, by continually resuming them.
	/// @param instruction_count The max number of instructions to execute before returning.
	bool resume(uint64_t instruction_count);

	/// @brief Returns the pointer provided at instantiation of the Script instance.
	/// @tparam T The real type of the user-provided pointer.
	/// @return Returns the user-provided pointer.
	template <typename T> T* userptr() noexcept
	{
		return (T*)m_userptr;
	}

	/// @brief Returns the pointer provided at instantiation of the Script instance.
	/// @tparam T The real type of the user-provided pointer.
	/// @return Returns the user-provided pointer.
	template <typename T> const T* userptr() const noexcept
	{
		return (const T*)m_userptr;
	}

	/// @brief Tries to find the name of a symbol at the given virtual address.
	/// @param address The virtual address to find in the symbol table.
	/// @return The closest symbol to the given address.
	std::string symbol_name(gaddr_t address) const;

	/// @brief Look up the address of a name. Returns 0x0 if not found.
	/// Uses an unordered_map to remember lookups in order to speed up future lookups.
	/// @param name The name to find the virtual address for.
	/// @return The virtual address of name, or 0x0 if not found.
	gaddr_t address_of(const std::string& name) const;

	/// @brief Retrieve current argument registers, specifying each type.
	/// @tparam ...Args The types of arguments to retrieve.
	/// @return A tuple of arguments.
	template <typename... Args>
	auto args() const;

	/// @brief The virtual machine hosting the Scripts program.
	/// @return The underlying virtual machine.
	auto& machine()
	{
		return *m_machine;
	}

	/// @brief The virtual machine hosting the Scripts program.
	/// @return The underlying virtual machine.
	const auto& machine() const
	{
		return *m_machine;
	}

	/// @brief The name given to this Script instance during creation.
	/// @return The name of this Script instance.
	const auto& name() const noexcept
	{
		return m_name;
	}

	/// @brief The filename passed to this Script instance during creation.
	/// @return The filename of this Script instance.
	const auto& filename() const noexcept
	{
		return m_filename;
	}

	void print(std::string_view text);
	void print_backtrace(const gaddr_t addr);

	void stdout_enable(bool e) noexcept
	{
		m_stdout = e;
	}

	bool stdout_enabled() const noexcept
	{
		return m_stdout;
	}

	gaddr_t heap_area() const noexcept
	{
		return m_heap_area;
	}

	/* The guest heap is managed outside using system calls. */

	/// @brief Allocate bytes inside the program. All allocations are at least 8-byte aligned.
	/// @param bytes The number of 8-byte aligned bytes to allocate.
	/// @return The address of the allocated bytes.
	gaddr_t guest_alloc(gaddr_t bytes);

	/// @brief Allocate bytes inside the program. All allocations are at least 8-byte aligned.
	/// This allocation is sequentially accessible outside of the script. Using
	/// this function the host engine can view and use the allocation as a single
	/// consecutive array of bytes, allowing it to be used with many (if not most)
	/// APIs that do not handle fragmented memory.
	/// @param bytes The number of sequentially allocated 8-byte aligned bytes.
	/// @return The address of the sequentially allocated bytes.
	gaddr_t guest_alloc_sequential(gaddr_t bytes);
	bool guest_free(gaddr_t addr);

	/// @brief Create a wrapper object that manages an allocation of n objects of type T
	/// inside the program. All objects are default-constructed.
	/// All objects are accessible in the host and in the script. The script can read and write
	/// all objects at will, making the state of these objects fundamentally untrustable.
	/// When the wrapper destructs, the allocation in the program is also freed. Can be moved.
	/// @tparam T The type of the allocated objects.
	/// @param n The number of allocated objects in the array.
	/// @return A wrapper managing the program-hosted objects. Can be moved.
	template <typename T> GuestObjects<T> guest_alloc(size_t n = 1);

	/// @brief Retrieve the fork of this script instance.
	/// @return The fork of this instance.
	Script& get_fork();
	/// @brief Retrieve an instance of a script by its program name.
	/// @param  name The name of the script to find.
	/// @return The script instance with the given name.
	static Script& Find(const std::string& name);

	// Create new Script instance from file
	Script(
		const std::string& name, const std::string& filename,
		void* userptr = nullptr);
	// Create new Script instance from cloning another Script
	Script clone(const std::string& name, void* userptr = nullptr);
	~Script();

  private:
	// Create new Script instance from existing binary
	Script(
		std::shared_ptr<const std::vector<uint8_t>> binary, const std::string& name,
		const std::string& filename, void* userptr = nullptr);
	/// @brief Create a thread-local fork of this script instance.
	/// @return A new Script instance that is a fork of this instance.
	Script& create_fork();
	static void setup_syscall_interface();
	void reset(); // true if the reset was successful
	void initialize();
	void could_not_find(std::string_view);
	void handle_exception(gaddr_t);
	void handle_timeout(gaddr_t);
	void max_depth_exceeded(gaddr_t);
	void machine_setup();

	std::unique_ptr<machine_t> m_machine = nullptr;
	std::shared_ptr<const std::vector<uint8_t>> m_binary;
	void* m_userptr;
	gaddr_t m_heap_area		   = 0;
	std::string m_name;
	std::string m_filename;
	uint8_t  m_call_depth   = 0;
	bool m_stdout			= true;
	bool m_last_newline		= true;
	/// @brief Cached addresses for symbol lookups
	/// This could probably be improved by doing it per-binary instead
	/// of a separate cache per instance. But at least it's thread-safe.
	mutable std::unordered_map<std::string, gaddr_t> m_lookup_cache;
};

struct ScriptDepthMeter {
	ScriptDepthMeter(uint8_t& val) : m_val(++val) {}
	~ScriptDepthMeter() { m_val --; }

	uint8_t get() const noexcept { return m_val; }
	bool is_one() const noexcept { return m_val == 1; }

private:
	uint8_t& m_val;
};

template <typename... Args>
inline std::optional<Script::sgaddr_t> Script::call(gaddr_t address, Args&&... args)
{
	ScriptDepthMeter meter(this->m_call_depth);
	try
	{
		if (LIKELY(meter.is_one()))
			return {machine().vmcall<MAX_CALL_INSTR>(
				address, std::forward<Args>(args)...)};
		else if (LIKELY(meter.get() < MAX_CALL_DEPTH))
			return {machine().preempt(MAX_CALL_INSTR,
				address, std::forward<Args>(args)...)};
		else
			this->max_depth_exceeded(address);
	}
	catch (const std::exception& e)
	{
		this->handle_exception(address);
	}
	return std::nullopt;
}

template <typename... Args>
inline std::optional<Script::sgaddr_t> Script::call(const std::string& func, Args&&... args)
{
	const auto address = this->address_of(func.c_str());
	if (UNLIKELY(address == 0x0))
	{
		this->could_not_find(func);
		return std::nullopt;
	}
	return {this->call(address, std::forward<Args>(args)...)};
}

template <typename... Args>
inline std::optional<Script::sgaddr_t> Script::preempt(gaddr_t address, Args&&... args)
{
	try
	{
		return {machine().preempt(
			MAX_CALL_INSTR, address, std::forward<Args>(args)...)};
	}
	catch (const std::exception& e)
	{
		this->handle_exception(address);
	}
	return std::nullopt;
}

template <typename... Args>
inline std::optional<Script::sgaddr_t> Script::preempt(const std::string& func, Args&&... args)
{
	const auto address = this->address_of(func.c_str());
	if (UNLIKELY(address == 0x0))
	{
		this->could_not_find(func);
		return std::nullopt;
	}
	return {this->preempt(address, std::forward<Args>(args)...)};
}

inline bool Script::resume(uint64_t cycles)
{
	try
	{
		machine().resume<false>(cycles);
		return true;
	}
	catch (const std::exception& e)
	{
		this->handle_exception(machine().cpu.pc());
		return false;
	}
}

template <typename... Args>
inline auto Script::args() const
{
	return machine().sysargs<Args ...> ();
}

/**
 * This uses RAII to sequentially allocate a range
 * for the objects, which is freed on destruction.
 * This is a relatively inexpensive abstraction.
 *
 * Example:
 * myscript.guest_alloc<GameObject>(16) allocates one
 * single memory range on the managed heap of size:
 *  sizeof(GameObject) * 16
 * The range is properly aligned and holds all objects.
 * It is heap-allocated inside VM guest virtual memory.
 *
 * The returned GuestObjects<T> object manages all the
 * objects, and on destruction will free all objects.
 * It can be moved. The moved-from object manages nothing.
 *
 * All objects are potentially uninitialized, like all
 * heap allocations, and will need to be individually
 * initialized.
 **/
template <typename T> struct GuestObjects
{
	T& at(size_t n)
	{
		if (n < m_count)
			return m_object[n];
		throw riscv::MachineException(
			riscv::ILLEGAL_OPERATION, "at(): Object is out of range", n);
	}

	const T& at(size_t n) const
	{
		if (n < m_count)
			return m_object[n];
		throw riscv::MachineException(
			riscv::ILLEGAL_OPERATION, "at(): Object is out of range", n);
	}

	Script::gaddr_t address(size_t n) const
	{
		if (n < m_count)
			return m_address + sizeof(T) * n;
		throw riscv::MachineException(
			riscv::ILLEGAL_OPERATION, "address(): Object is out of range", n);
	}

	GuestObjects(Script& s, Script::gaddr_t a, T* o, size_t c)
	  : m_script(s), m_address(a), m_object(o), m_count(c)
	{
	}

	GuestObjects(GuestObjects&& other)
	  : m_script(other.m_script), m_address(other.m_address),
		m_object(other.m_object), m_count(other.m_count)
	{
		other.m_address = 0x0;
		other.m_count	= 0u;
	}

	~GuestObjects()
	{
		if (this->m_address != 0x0)
		{
			m_script.guest_free(this->m_address);
			this->m_address = 0x0;
		}
	}

	Script& m_script;
	Script::gaddr_t m_address;
	T* m_object;
	size_t m_count;
};

template <typename T> inline GuestObjects<T> Script::guest_alloc(size_t n)
{
	// XXX: If n is too large, it will always overflow a page,
	// and we will need another strategy in order to guarantee
	// sequential memory.
	auto addr = this->guest_alloc_sequential(sizeof(T) * n);
	if (addr != 0x0)
	{
		const auto pageno	= machine().memory.page_number(addr);
		const size_t offset = addr & (riscv::Page::size() - 1);
		// Lazily create zero-initialized page
		auto& page	 = machine().memory.create_writable_pageno(pageno, true);
		auto* object = (T*)&page.data()[offset];
		// Default-initialize all objects
		for (auto *o = object; o < object + n; o++)
			new (o) T{};
		// Note: this can fail and throw, but we don't care
		return {*this, addr, object, n};
	}
	throw std::runtime_error("Unable to allocate aligned sequential data");
}
