#include "os_3ds.h"

OS_3DS *OS_3DS::singleton = nullptr;

OS_3DS *OS_3DS::get_singleton() {
    return singleton;
}

void OS_3DS::initialize() {
    singleton = this;
}

void OS_3DS::finalize() {
    singleton = nullptr;
}
