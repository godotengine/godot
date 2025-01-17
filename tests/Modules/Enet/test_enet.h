#ifndef TEST_ENET_H
#define TEST_ENET_H

#include "tests/test_macros.h"

namespace TestEnet {

    void check_connection_creation();
    void check_packeds_received();
    void check_destroy();
    void check_flush();
}