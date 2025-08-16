#pragma once
#include "machine.hpp"
#include <cstdarg>
#include <inttypes.h>
#include <unistd.h>

/**
  The ‘org.gnu.gdb.riscv.cpu’ feature is required
  for RISC-V targets. It should contain the registers
  ‘x0’ through ‘x31’, and ‘pc’. Either the
  architectural names (‘x0’, ‘x1’, etc) can be used,
  or the ABI names (‘zero’, ‘ra’, etc).

  The ‘org.gnu.gdb.riscv.fpu’ feature is optional.
  If present, it should contain registers ‘f0’ through
  ‘f31’, ‘fflags’, ‘frm’, and ‘fcsr’. As with the cpu
  feature, either the architectural register names,
  or the ABI names can be used.

  The ‘org.gnu.gdb.riscv.virtual’ feature is optional.
  If present, it should contain registers that are not
  backed by real registers on the target, but are
  instead virtual, where the register value is
  derived from other target state. In many ways these
  are like GDBs pseudo-registers, except implemented
  by the target. Currently the only register expected
  in this set is the one byte ‘priv’ register that
  contains the target’s privilege level in the least
  significant two bits.

  The ‘org.gnu.gdb.riscv.csr’ feature is optional.
  If present, it should contain all of the targets
  standard CSRs. Standard CSRs are those defined in
  the RISC-V specification documents. There is some
  overlap between this feature and the fpu feature;
  the ‘fflags’, ‘frm’, and ‘fcsr’ registers could
  be in either feature. The expectation is that these
  registers will be in the fpu feature if the target
  has floating point hardware, but can be moved into
  the csr feature if the target has the floating
  point control registers, but no other floating
  point hardware.
**/

namespace riscv {
#ifndef WIN32
    typedef int socket_fd_type;
#else
    typedef uint64_t socket_fd_type;
#endif
template <int W> struct RSPClient;

template <int W>
struct RSP
{
	// Wait for a connection for @timeout_secs
	std::unique_ptr<RSPClient<W>> accept(int timeout_secs = 30);
    socket_fd_type  fd() const noexcept { return server_fd; }

	RSP(riscv::Machine<W>&, uint16_t);
	~RSP();

private:
	riscv::Machine<W>& m_machine;
    socket_fd_type server_fd;
};
template <int W>
struct RSPClient
{
	using StopFunc = std::function<void(RSPClient<W>&)>;
	using PrinterFunc = void(*)(const Machine<W>&, const char*, size_t);
	bool is_closed() const noexcept { return m_closed; }

	bool process_one();
	bool send(const char* str);
	bool sendf(const char* fmt, ...);
	void reply_ack();
	void reply_ok();
	void interrupt();
	void kill();

	auto& machine() { return *m_machine; }
	void set_machine(Machine<W>& m) { m_machine = &m; }
	void set_instruction_limit(uint64_t limit) { m_ilimit = limit; }
	void set_verbose(bool v) { m_verbose = v; }
	void on_stopped(StopFunc f) { m_on_stopped = f; }

	// Debug printer (for printing exceptions)
	void debug_print(const char*, size_t) const;
	auto& get_debug_printer() const noexcept { return m_debug_printer; }
	void set_debug_printer(PrinterFunc pf) noexcept { m_debug_printer = pf; }

	RSPClient(riscv::Machine<W>& m, socket_fd_type fd);
	~RSPClient();

private:
	static constexpr char lut[] = "0123456789abcdef";
	static const int PACKET_SIZE = 1200;
	template <typename T>
	inline void putreg(char*& d, const char* end, const T& reg);
	int forge_packet(char* dst, size_t dstlen, const char*, int);
	int forge_packet(char* dst, size_t dstlen, const char*, va_list);
	void process_data();
	void handle_query();
	void handle_breakpoint();
	void handle_continue();
	void handle_step();
	void handle_exception(const std::exception&);
	void handle_executing();
	void handle_multithread();
	void handle_readmem();
	void handle_readreg();
	void handle_writereg();
	void handle_writemem();
	void report_gprs();
	void report_status();
	void close_now();
	riscv::Machine<W>* m_machine;
	uint64_t m_ilimit = 16'000'000UL;
    socket_fd_type  sockfd;
	bool m_closed  = false;
	bool m_verbose = false;
	std::string buffer;
	std::array<riscv::address_type<W>, 8> m_bp {};
	size_t m_bp_iterator = 0;
	StopFunc m_on_stopped = nullptr;
	mutable PrinterFunc m_debug_printer = [](const Machine<W>&, const char*, size_t) {};
};
} // riscv

// The entire RSP<W> must be implemented per OS
#ifndef WIN32
#include "linux/rsp_server.hpp"
#else
#include "win32/rsp_server.hpp"
#endif

namespace riscv {

template <int W> inline
RSPClient<W>::RSPClient(riscv::Machine<W>& m, socket_fd_type fd)
	: m_machine{&m}, sockfd(fd)
{
	m_machine->set_max_instructions(m_ilimit);
}

template <int W>
int RSPClient<W>::forge_packet(
	char* dst, size_t dstlen, const char* data, int datalen)
{
	char* d = dst;
	const char* maxd = &dst[dstlen];
	*d++ = '$';
	uint8_t csum = 0;
	for (int i = 0; i < datalen; i++) {
		uint8_t c = data[i];
		if (c == '$' || c == '#' || c == '*' || c == '}') {
			c ^= 0x20;
			csum += '}';
			*d++ = '}';
		}
		*d++ = c;
		csum += c;
		// Bounds-check the destination buffer
		if (UNLIKELY(d + 3 > maxd))
			break;
	}
	if (UNLIKELY(d + 3 > maxd))
		throw MachineException(OUT_OF_MEMORY, "Unable to forge RSP packet: Not enough space");
	*d++ = '#';
	*d++ = lut[(csum >> 4) & 0xF];
	*d++ = lut[(csum >> 0) & 0xF];
	return d - dst;
}
template <int W>
int RSPClient<W>::forge_packet(
	char* dst, size_t dstlen, const char* fmt, va_list args)
{
	char data[4 + 2*PACKET_SIZE];
	int datalen = vsnprintf(data, sizeof(data), fmt, args);
	return forge_packet(dst, dstlen, data, datalen);
}

template <int W>
void RSPClient<W>::process_data()
{
	switch (buffer[0]) {
	case 'q':
		handle_query();
		break;
	case 'c':
		handle_continue();
		break;
	case 's':
		handle_step();
		break;
	case 'g':
		report_gprs();
		break;
	case 'D':
	case 'k':
		kill();
		return;
	case 'H':
		handle_multithread();
		break;
	case 'm':
		handle_readmem();
		break;
	case 'p':
		handle_readreg();
		break;
	case 'P':
		handle_writereg();
		break;
	case 'v':
		handle_executing();
		break;
	case 'X':
		handle_writemem();
		break;
	case 'Z':
	case 'z':
		handle_breakpoint();
		break;
	case '?':
		report_status();
		break;
	default:
		if (UNLIKELY(m_verbose)) {
			fprintf(stderr, "Unhandled packet: %c\n",
				buffer[0]);
		}
	}
}
template <int W>
void RSPClient<W>::handle_query()
{
	if (strncmp("qSupported", buffer.data(), strlen("qSupported")) == 0)
	{
		sendf("PacketSize=%x;swbreak-;hwbreak+", PACKET_SIZE);
	}
	else if (strncmp("qAttached", buffer.data(), strlen("qC")) == 0)
	{
		send("1");
	}
	else if (strncmp("qC", buffer.data(), strlen("qC")) == 0)
	{
		// Current thread ID
		send("QC0");
	}
	else if (strncmp("qOffsets", buffer.data(), strlen("qOffsets")) == 0)
	{
		// Section relocation offsets
		send("Text=0;Data=0;Bss=0");
	}
	else if (strncmp("qfThreadInfo", buffer.data(), strlen("qfThreadInfo")) == 0)
	{
		// Start of threads list
		send("m0");
	}
	else if (strncmp("qsThreadInfo", buffer.data(), strlen("qfThreadInfo")) == 0)
	{
		// End of threads list
		send("l");
	}
	else if (strncmp("qSymbol::", buffer.data(), strlen("qSymbol::")) == 0)
	{
		send("OK");
	}
	else if (strncmp("qTStatus", buffer.data(), strlen("qTStatus")) == 0)
	{
		send("");
	}
	else {
		if (UNLIKELY(m_verbose)) {
			fprintf(stderr, "Unknown query: %s\n",
				buffer.data());
		}
		send("");
	}
}
template <int W>
void RSPClient<W>::handle_continue()
{
	try {
		for (auto bp : m_bp) {
			if (bp == m_machine->cpu.pc()) {
				send("S05");
				return;
			}
		}
		uint64_t n = m_ilimit;
		bool breakpoint_hit = false;

		while (n > 0) {
			// When stepping the machine will look stopped
			// simply by checking stopped(), however the stop()
			// function sets the max instruction counter to 0
			m_machine->cpu.step_one();
			// Breakpoint
			const auto pc = m_machine->cpu.pc();
			for (auto bp : m_bp) {
				if (bp == pc)
					breakpoint_hit = true;
			}
			// Stopped (usual way)
			if (m_machine->max_instructions() == 0 || breakpoint_hit)
				break;
			n--;
		}
		// Break reasons
		if (n == 0 || m_machine->stopped() || breakpoint_hit) {
			send("S05");
			return;
		}
	} catch (const std::exception& e) {
		handle_exception(e);
		return;
	}
	report_status();
}
template <int W>
void RSPClient<W>::handle_step()
{
	try {
		if (!m_machine->stopped()) {
			m_machine->cpu.step_one();
		} else {
			send("S01");
			return;
		}
	} catch (const std::exception& e) {
		handle_exception(e);
		return;
	}
	report_status();
}
template <int W>
void RSPClient<W>::handle_exception(const std::exception& e)
{
	char buffer[1024];
	int len = snprintf(buffer, sizeof(buffer), "Exception: %s\n", e.what());
	this->debug_print(buffer, len);
	// Is this the right thing to do?
	m_machine->stop();
	send("S01");
}
template <int W>
void RSPClient<W>::handle_breakpoint()
{
	uint32_t type = 0;
	uint64_t addr = 0;
	sscanf(&buffer[1], "%x,%" PRIx64, &type, &addr);
	if (buffer[0] == 'Z') {
		this->m_bp.at(m_bp_iterator) = addr;
		m_bp_iterator = (m_bp_iterator + 1) % m_bp.size();
	} else {
		for (auto& bp : this->m_bp) {
			if (bp == addr) bp = 0;
		}
	}
	reply_ok();
}
template <int W>
void RSPClient<W>::handle_executing()
{
	if (strncmp("vCont?", buffer.data(), strlen("vCont?")) == 0)
	{
		send("vCont;c;s");
	}
	else if (strncmp("vCont;c", buffer.data(), strlen("vCont;c")) == 0)
	{
		this->handle_continue();
	}
	else if (strncmp("vCont;s", buffer.data(), strlen("vCont;s")) == 0)
	{
		this->handle_step();
	}
	else if (strncmp("vKill", buffer.data(), strlen("vKill")) == 0)
	{
		this->kill();
	}
	else if (strncmp("vMustReplyEmpty", buffer.data(), strlen("vMustReplyEmpty")) == 0)
	{
		send("");
	}
	else {
		if (UNLIKELY(m_verbose)) {
			fprintf(stderr, "Unknown executor: %s\n",
				buffer.data());
		}
		send("");
	}
}
template <int W>
void RSPClient<W>::handle_multithread() {
	reply_ok();
}
template <int W>
void RSPClient<W>::handle_readmem()
{
	uint64_t addr = 0;
	uint32_t len = 0;
	sscanf(buffer.c_str(), "m%" PRIx64 ",%x", &addr, &len);
	if (len >= 500) {
		send("E01");
		return;
	}

	char data[1024];
	char* d = data;
	try {
		for (unsigned i = 0; i < len; i++) {
			uint8_t val =
			m_machine->memory.template read<uint8_t> (addr + i);
			*d++ = lut[(val >> 4) & 0xF];
			*d++ = lut[(val >> 0) & 0xF];
		}
	} catch (...) {
		send("E01");
		return;
	}
	*d++ = 0;
	send(data);
}
template <int W>
void RSPClient<W>::handle_writemem()
{
	uint64_t addr = 0;
	uint32_t len = 0;
	int ret = sscanf(buffer.c_str(), "X%" PRIx64 ",%x:", &addr, &len);
	if (ret <= 0) {
		send("E01");
		return;
	}
	char* bin = (char*)
		memchr(buffer.data(), ':', buffer.size());
	if (bin == nullptr) {
		send("E01");
		return;
	}
	bin += 1; // Move past colon
	const char* end = buffer.c_str() + buffer.size();
	uint32_t rlen = std::min(len, (uint32_t) (end - bin));
	try {
		for (auto i = 0u; i < rlen; i++) {
			char data = bin[i];
			if (data == '{' && i+1 < rlen) {
				data = bin[++i] ^ 0x20;
			}
			m_machine->memory.template write<uint8_t> (addr+i, data);
		}
		reply_ok();
	} catch (...) {
		send("E01");
	}
}
template <int W>
void RSPClient<W>::report_status()
{
	if (!m_machine->stopped())
		send("S05"); /* Just send TRAP */
	else {
		if (m_on_stopped != nullptr) {
			m_on_stopped(*this);
		} else {
			//send("vStopped");
			send("S05"); /* Just send TRAP */
		}
	}
}
template <int W>
template <typename T>
void RSPClient<W>::putreg(char*& d, const char* end, const T& reg)
{
	for (auto j = 0u; j < sizeof(reg) && d < end; j++) {
		*d++ = lut[(reg >> (j*8+4)) & 0xF];
		*d++ = lut[(reg >> (j*8+0)) & 0xF];
	}
}

template <int W>
void RSPClient<W>::handle_readreg()
{
	uint32_t idx = 0;
	sscanf(buffer.c_str(), "p%x", &idx);
	if (idx > 68) {
		send("E01");
		return;
	}

	char valdata[32];
	size_t vallen = 0;

	if (idx >= 33)
	{
		if (idx < 66) {
			const auto& fl = m_machine->cpu.registers().getfl(idx - 33);
			vallen = sizeof(fl.i64);
			std::memcpy(valdata, &fl.i64, vallen);
		} else {
			uint32_t reg = 0;
			switch (idx) {
			case 66: reg = m_machine->cpu.registers().fcsr().fflags; break;
			case 67: reg = m_machine->cpu.registers().fcsr().frm; break;
			case 68: reg = m_machine->cpu.registers().fcsr().whole; break;
			}
			vallen = sizeof(reg);
			std::memcpy(valdata, &reg, vallen);
		}
	}
	if (idx == 32)
	{
		const auto reg = m_machine->cpu.pc();
		vallen = sizeof(reg);
		std::memcpy(valdata, &reg, vallen);
	}
	else if (idx < 32)
	{
		const auto reg = m_machine->cpu.reg(idx);
		vallen = sizeof(reg);
		std::memcpy(valdata, &reg, vallen);
	}

	char data[32];
	char* d = data;
	try {
		for (unsigned i = 0; i < vallen; i++) {
			*d++ = lut[(valdata[i] >> 4) & 0xF];
			*d++ = lut[(valdata[i] >> 0) & 0xF];
		}
	} catch (...) {
		send("E01");
		return;
	}
	*d++ = 0;
	send(data);
}
template <int W>
void RSPClient<W>::handle_writereg()
{
	uint64_t value = 0;
	uint32_t idx = 0;
	sscanf(buffer.c_str(), "P%x=%" PRIx64, &idx, &value);
	value = __builtin_bswap64(value);

	if (idx < 32) {
		m_machine->cpu.reg(idx) = value;
		send("OK");
	} else if (idx == 32) {
		m_machine->cpu.jump(value);
		send("OK");
	} else if (idx >= 33 && idx <= 68) {
		switch (idx) {
		case 66: m_machine->cpu.registers().fcsr().fflags = value; break;
		case 67: m_machine->cpu.registers().fcsr().frm = value; break;
		case 68: m_machine->cpu.registers().fcsr().whole = value; break;
		default:
			auto& fl = m_machine->cpu.registers().getfl(idx - 33);
			fl.i64 = value;
		}
		send("OK");
	} else {
		send("E01");
	}
}

template <int W>
void RSPClient<W>::report_gprs()
{
	auto& regs = m_machine->cpu.registers();
	char data[1024];
	char* d = data;
	/* GPRs */
	for (int i = 0; i < 32; i++) {
		putreg(d, &data[sizeof(data)], regs.get(i));
	}
	/* PC */
	putreg(d, &data[sizeof(data)], regs.pc);
	*d++ = 0;
	send(data);
}

template <int W> inline
void RSPClient<W>::reply_ok() {
	send("OK");
}
template <int W>
void RSPClient<W>::interrupt() {
	send("S05");
}

template <int W>
inline void RSPClient<W>::debug_print(const char* buffer, size_t len) const
{
	this->m_debug_printer(*m_machine, buffer, len);
}

} // riscv
