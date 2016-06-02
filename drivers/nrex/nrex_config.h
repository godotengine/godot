// Godot-specific configuration
// To use this, replace nrex_config.h

#include "core/os/memory.h"

#define NREX_UNICODE
//#define NREX_THROW_ERROR

#define NREX_NEW(X) memnew(X)
#define NREX_NEW_ARRAY(X, N) memnew_arr(X, N)
#define NREX_DELETE(X) memdelete(X)
#define NREX_DELETE_ARRAY(X) memdelete_arr(X)
