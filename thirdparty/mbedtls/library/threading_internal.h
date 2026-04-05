/**
 * \file threading_internal.h
 *
 * \brief Threading interfaces used by the test framework
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_THREADING_INTERNAL_H
#define MBEDTLS_THREADING_INTERNAL_H

#include "common.h"

#include <mbedtls/threading.h>

/* A version number for the internal threading interface.
 * This is meant to allow the framework to remain compatible with
 * multiple versions, to facilitate transitions.
 *
 * Conventionally, this is the Mbed TLS version number when the
 * threading interface was last changed in a way that may impact the
 * test framework, with the lower byte incremented as necessary
 * if multiple changes happened between releases. */
#define MBEDTLS_THREADING_INTERNAL_VERSION 0x03060000

#endif /* MBEDTLS_THREADING_INTERNAL_H */
