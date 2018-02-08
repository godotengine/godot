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

#ifndef _SSL_PORT_H_
#define _SSL_PORT_H_

#ifdef __cplusplus
 extern "C" {
#endif

/*
#include "esp_types.h"
#include "esp_log.h"
*/
#include "string.h"
#include "malloc.h"

void *ssl_mem_zalloc(size_t size);

#define ssl_mem_malloc malloc
#define ssl_mem_free   free

#define ssl_memcpy     memcpy
#define ssl_strlen     strlen

#define ssl_speed_up_enter()
#define ssl_speed_up_exit()

#define SSL_DEBUG_FL
#define SSL_DEBUG_LOG(fmt, ...) ESP_LOGI("openssl", fmt, ##__VA_ARGS__)

#ifdef __cplusplus
 }
#endif

#endif
