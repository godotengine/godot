#include <libriscv/machine.hpp>

namespace riscv {

template <int W>
void add_socket_syscalls(Machine<W>& machine) {
}

#ifdef RISCV_32I
template void add_socket_syscalls<4>(Machine<4> &);
#endif
#ifdef RISCV_64I
template void add_socket_syscalls<8>(Machine<8> &);
#endif
} // namespace riscv
