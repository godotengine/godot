// Copyright 2015-2016 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _SSL3_H_
#define _SSL3_H_

#ifdef __cplusplus
 extern "C" {
#endif

# define SSL3_AD_CLOSE_NOTIFY             0
# define SSL3_AD_UNEXPECTED_MESSAGE      10/* fatal */
# define SSL3_AD_BAD_RECORD_MAC          20/* fatal */
# define SSL3_AD_DECOMPRESSION_FAILURE   30/* fatal */
# define SSL3_AD_HANDSHAKE_FAILURE       40/* fatal */
# define SSL3_AD_NO_CERTIFICATE          41
# define SSL3_AD_BAD_CERTIFICATE         42
# define SSL3_AD_UNSUPPORTED_CERTIFICATE 43
# define SSL3_AD_CERTIFICATE_REVOKED     44
# define SSL3_AD_CERTIFICATE_EXPIRED     45
# define SSL3_AD_CERTIFICATE_UNKNOWN     46
# define SSL3_AD_ILLEGAL_PARAMETER       47/* fatal */

# define SSL3_AL_WARNING                  1
# define SSL3_AL_FATAL                    2

#define SSL3_VERSION                 0x0300

#ifdef __cplusplus
}
#endif

#endif
