/*
 *  Version information
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "common.h"

#if defined(MBEDTLS_VERSION_C)

#include "mbedtls/version.h"
#include <string.h>

unsigned int mbedtls_version_get_number(void)
{
    return MBEDTLS_VERSION_NUMBER;
}

void mbedtls_version_get_string(char *string)
{
    memcpy(string, MBEDTLS_VERSION_STRING,
           sizeof(MBEDTLS_VERSION_STRING));
}

void mbedtls_version_get_string_full(char *string)
{
    memcpy(string, MBEDTLS_VERSION_STRING_FULL,
           sizeof(MBEDTLS_VERSION_STRING_FULL));
}

#endif /* MBEDTLS_VERSION_C */
