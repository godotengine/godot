#include <sys/select.h>

template <int W>
static void syscall_pselect(Machine<W>&)
{
    throw MachineException(SYSTEM_CALL_FAILED, "pselect() not implemented");
}
