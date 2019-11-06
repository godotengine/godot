#
# Copyright (c) 2017 The Khronos Group Inc.
# Copyright (c) 2017 Valve Corporation
# Copyright (c) 2017 LunarG, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Lenny Komow <lenny@lunarg.com>
#

# This code is used to pass on device (including physical device) extensions through the call chain. It must do this without
# creating a stack frame, because the actual parameters of the call are not known. Since the first parameter is known to be a
# VkPhysicalDevice or a dispatchable object it can unwrap the object, possibly overwriting the wrapped physical device, and then
# jump to the next function in the call chain

#ifdef HAVE_CET_H
#include <cet.h>
#else
#define _CET_ENDBR
#endif

.intel_syntax noprefix
.include "gen_defines.asm"

.ifdef X86_64

.macro PhysDevExtTramp num
.global vkPhysDevExtTramp\num
vkPhysDevExtTramp\num:
    _CET_ENDBR
    mov     rax, [rdi]
    mov     rdi, [rdi + PHYS_DEV_OFFSET_PHYS_DEV_TRAMP]
    jmp     [rax + (PHYS_DEV_OFFSET_INST_DISPATCH + (PTR_SIZE * \num))]
.endm

.macro PhysDevExtTermin num
.global vkPhysDevExtTermin\num
vkPhysDevExtTermin\num:
    _CET_ENDBR
    mov     rax, [rdi + ICD_TERM_OFFSET_PHYS_DEV_TERM]                          # Store the loader_icd_term* in rax
    cmp     qword ptr [rax + (DISPATCH_OFFSET_ICD_TERM + (PTR_SIZE * \num))], 0 # Check if the next function in the chain is NULL
    je      terminError\num                                                     # Go to the error section if it is NULL
    mov     rdi, [rdi + PHYS_DEV_OFFSET_PHYS_DEV_TERM]                          # Load the unwrapped VkPhysicalDevice into the first arg
    jmp     [rax + (DISPATCH_OFFSET_ICD_TERM + (PTR_SIZE * \num))]              # Jump to the next function in the chain
terminError\num:
    sub     rsp, 56                                                             # Create the stack frame
    mov     rdi, [rax + INSTANCE_OFFSET_ICD_TERM]                               # Load the loader_instance into rdi (first arg)
    mov     r8, [rdi + (HASH_OFFSET_INSTANCE + (HASH_SIZE * \num) + FUNC_NAME_OFFSET_HASH)] # Load the func name into r8 (fifth arg)
    lea     rcx, termin_error_string@GOTPCREL                                   # Load the error string into rcx (fourth arg)
    xor     edx, edx                                                            # Set rdx to zero (third arg)
    lea     esi, [rdx + VK_DEBUG_REPORT_ERROR_BIT_EXT]                          # Write the error logging bit to rsi (second arg)
    call    loader_log                                                          # Log the error message before we crash
    add     rsp, 56                                                             # Clean up the stack frame
    mov     rax, 0
    jmp     rax                                                                 # Crash intentionally by jumping to address zero
.endm

.macro DevExtTramp num
.global vkdev_ext\num
vkdev_ext\num:
    _CET_ENDBR
    mov     rax, [rdi]                                                          # Dereference the handle to get the dispatch table
    jmp     [rax + (EXT_OFFSET_DEVICE_DISPATCH + (PTR_SIZE * \num))]            # Jump to the appropriate call chain
.endm

.else

.macro PhysDevExtTramp num
.global vkPhysDevExtTramp\num
vkPhysDevExtTramp\num:
    _CET_ENDBR
    mov     eax, [esp + 4]                              # Load the wrapped VkPhysicalDevice into eax
    mov     ecx, [eax + PHYS_DEV_OFFSET_PHYS_DEV_TRAMP] # Load the unwrapped VkPhysicalDevice into ecx
    mov     [esp + 4], ecx                              # Overwrite the wrapped VkPhysicalDevice with the unwrapped one (on the stack)
    mov     eax, [eax]                                  # Dereference the wrapped VkPhysicalDevice to get the dispatch table in eax
    jmp     [eax + (PHYS_DEV_OFFSET_INST_DISPATCH + (PTR_SIZE * \num))] # Dereference the wrapped VkPhysicalDevice to get the dispatch table in eax
.endm

.macro PhysDevExtTermin num
.global vkPhysDevExtTermin\num
vkPhysDevExtTermin\num:
    _CET_ENDBR
    mov     ecx, [esp + 4]                                                      # Move the wrapped VkPhysicalDevice into ecx
    mov     eax, [ecx + ICD_TERM_OFFSET_PHYS_DEV_TERM]                          # Store the loader_icd_term* in eax
    cmp     dword ptr [eax + (DISPATCH_OFFSET_ICD_TERM + (PTR_SIZE * \num))], 0 # Check if the next function in the chain is NULL
    je      terminError\num                                                     # Go to the error section if it is NULL
    mov     ecx, [ecx + PHYS_DEV_OFFSET_PHYS_DEV_TERM]                          # Unwrap the VkPhysicalDevice in ecx
    mov     [esp + 4], ecx                                                      # Copy the unwrapped VkPhysicalDevice into the first arg
    jmp     [eax + (DISPATCH_OFFSET_ICD_TERM + (PTR_SIZE * \num))]              # Jump to the next function in the chain
terminError\num:
    mov     eax, [eax + INSTANCE_OFFSET_ICD_TERM]                               # Load the loader_instance into eax
    push    [eax + (HASH_OFFSET_INSTANCE + (HASH_SIZE * \num) + FUNC_NAME_OFFSET_HASH)] # Push the func name (fifth arg)
    push    offset termin_error_string@GOT                                      # Push the error string (fourth arg)
    push    0                                                                   # Push zero (third arg)
    push    VK_DEBUG_REPORT_ERROR_BIT_EXT                                       # Push the error logging bit (second arg)
    push    eax                                                                 # Push the loader_instance (first arg)
    call    loader_log                                                          # Log the error message before we crash
    add     esp, 20                                                             # Clean up the args
    mov     eax, 0
    jmp     eax                                                                 # Crash intentionally by jumping to address zero
.endm

.macro DevExtTramp num
.global vkdev_ext\num
vkdev_ext\num:
    _CET_ENDBR
    mov     eax, [esp + 4]                                                      # Dereference the handle to get the dispatch table
    jmp     [eax + (EXT_OFFSET_DEVICE_DISPATCH + (PTR_SIZE * \num))]            # Jump to the appropriate call chain
.endm

.endif

#if defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif

.data

termin_error_string:
.string "Extension %s not supported for this physical device"

.text

    PhysDevExtTramp 0
    PhysDevExtTramp 1
    PhysDevExtTramp 2
    PhysDevExtTramp 3
    PhysDevExtTramp 4
    PhysDevExtTramp 5
    PhysDevExtTramp 6
    PhysDevExtTramp 7
    PhysDevExtTramp 8
    PhysDevExtTramp 9
    PhysDevExtTramp 10
    PhysDevExtTramp 11
    PhysDevExtTramp 12
    PhysDevExtTramp 13
    PhysDevExtTramp 14
    PhysDevExtTramp 15
    PhysDevExtTramp 16
    PhysDevExtTramp 17
    PhysDevExtTramp 18
    PhysDevExtTramp 19
    PhysDevExtTramp 20
    PhysDevExtTramp 21
    PhysDevExtTramp 22
    PhysDevExtTramp 23
    PhysDevExtTramp 24
    PhysDevExtTramp 25
    PhysDevExtTramp 26
    PhysDevExtTramp 27
    PhysDevExtTramp 28
    PhysDevExtTramp 29
    PhysDevExtTramp 30
    PhysDevExtTramp 31
    PhysDevExtTramp 32
    PhysDevExtTramp 33
    PhysDevExtTramp 34
    PhysDevExtTramp 35
    PhysDevExtTramp 36
    PhysDevExtTramp 37
    PhysDevExtTramp 38
    PhysDevExtTramp 39
    PhysDevExtTramp 40
    PhysDevExtTramp 41
    PhysDevExtTramp 42
    PhysDevExtTramp 43
    PhysDevExtTramp 44
    PhysDevExtTramp 45
    PhysDevExtTramp 46
    PhysDevExtTramp 47
    PhysDevExtTramp 48
    PhysDevExtTramp 49
    PhysDevExtTramp 50
    PhysDevExtTramp 51
    PhysDevExtTramp 52
    PhysDevExtTramp 53
    PhysDevExtTramp 54
    PhysDevExtTramp 55
    PhysDevExtTramp 56
    PhysDevExtTramp 57
    PhysDevExtTramp 58
    PhysDevExtTramp 59
    PhysDevExtTramp 60
    PhysDevExtTramp 61
    PhysDevExtTramp 62
    PhysDevExtTramp 63
    PhysDevExtTramp 64
    PhysDevExtTramp 65
    PhysDevExtTramp 66
    PhysDevExtTramp 67
    PhysDevExtTramp 68
    PhysDevExtTramp 69
    PhysDevExtTramp 70
    PhysDevExtTramp 71
    PhysDevExtTramp 72
    PhysDevExtTramp 73
    PhysDevExtTramp 74
    PhysDevExtTramp 75
    PhysDevExtTramp 76
    PhysDevExtTramp 77
    PhysDevExtTramp 78
    PhysDevExtTramp 79
    PhysDevExtTramp 80
    PhysDevExtTramp 81
    PhysDevExtTramp 82
    PhysDevExtTramp 83
    PhysDevExtTramp 84
    PhysDevExtTramp 85
    PhysDevExtTramp 86
    PhysDevExtTramp 87
    PhysDevExtTramp 88
    PhysDevExtTramp 89
    PhysDevExtTramp 90
    PhysDevExtTramp 91
    PhysDevExtTramp 92
    PhysDevExtTramp 93
    PhysDevExtTramp 94
    PhysDevExtTramp 95
    PhysDevExtTramp 96
    PhysDevExtTramp 97
    PhysDevExtTramp 98
    PhysDevExtTramp 99
    PhysDevExtTramp 100
    PhysDevExtTramp 101
    PhysDevExtTramp 102
    PhysDevExtTramp 103
    PhysDevExtTramp 104
    PhysDevExtTramp 105
    PhysDevExtTramp 106
    PhysDevExtTramp 107
    PhysDevExtTramp 108
    PhysDevExtTramp 109
    PhysDevExtTramp 110
    PhysDevExtTramp 111
    PhysDevExtTramp 112
    PhysDevExtTramp 113
    PhysDevExtTramp 114
    PhysDevExtTramp 115
    PhysDevExtTramp 116
    PhysDevExtTramp 117
    PhysDevExtTramp 118
    PhysDevExtTramp 119
    PhysDevExtTramp 120
    PhysDevExtTramp 121
    PhysDevExtTramp 122
    PhysDevExtTramp 123
    PhysDevExtTramp 124
    PhysDevExtTramp 125
    PhysDevExtTramp 126
    PhysDevExtTramp 127
    PhysDevExtTramp 128
    PhysDevExtTramp 129
    PhysDevExtTramp 130
    PhysDevExtTramp 131
    PhysDevExtTramp 132
    PhysDevExtTramp 133
    PhysDevExtTramp 134
    PhysDevExtTramp 135
    PhysDevExtTramp 136
    PhysDevExtTramp 137
    PhysDevExtTramp 138
    PhysDevExtTramp 139
    PhysDevExtTramp 140
    PhysDevExtTramp 141
    PhysDevExtTramp 142
    PhysDevExtTramp 143
    PhysDevExtTramp 144
    PhysDevExtTramp 145
    PhysDevExtTramp 146
    PhysDevExtTramp 147
    PhysDevExtTramp 148
    PhysDevExtTramp 149
    PhysDevExtTramp 150
    PhysDevExtTramp 151
    PhysDevExtTramp 152
    PhysDevExtTramp 153
    PhysDevExtTramp 154
    PhysDevExtTramp 155
    PhysDevExtTramp 156
    PhysDevExtTramp 157
    PhysDevExtTramp 158
    PhysDevExtTramp 159
    PhysDevExtTramp 160
    PhysDevExtTramp 161
    PhysDevExtTramp 162
    PhysDevExtTramp 163
    PhysDevExtTramp 164
    PhysDevExtTramp 165
    PhysDevExtTramp 166
    PhysDevExtTramp 167
    PhysDevExtTramp 168
    PhysDevExtTramp 169
    PhysDevExtTramp 170
    PhysDevExtTramp 171
    PhysDevExtTramp 172
    PhysDevExtTramp 173
    PhysDevExtTramp 174
    PhysDevExtTramp 175
    PhysDevExtTramp 176
    PhysDevExtTramp 177
    PhysDevExtTramp 178
    PhysDevExtTramp 179
    PhysDevExtTramp 180
    PhysDevExtTramp 181
    PhysDevExtTramp 182
    PhysDevExtTramp 183
    PhysDevExtTramp 184
    PhysDevExtTramp 185
    PhysDevExtTramp 186
    PhysDevExtTramp 187
    PhysDevExtTramp 188
    PhysDevExtTramp 189
    PhysDevExtTramp 190
    PhysDevExtTramp 191
    PhysDevExtTramp 192
    PhysDevExtTramp 193
    PhysDevExtTramp 194
    PhysDevExtTramp 195
    PhysDevExtTramp 196
    PhysDevExtTramp 197
    PhysDevExtTramp 198
    PhysDevExtTramp 199
    PhysDevExtTramp 200
    PhysDevExtTramp 201
    PhysDevExtTramp 202
    PhysDevExtTramp 203
    PhysDevExtTramp 204
    PhysDevExtTramp 205
    PhysDevExtTramp 206
    PhysDevExtTramp 207
    PhysDevExtTramp 208
    PhysDevExtTramp 209
    PhysDevExtTramp 210
    PhysDevExtTramp 211
    PhysDevExtTramp 212
    PhysDevExtTramp 213
    PhysDevExtTramp 214
    PhysDevExtTramp 215
    PhysDevExtTramp 216
    PhysDevExtTramp 217
    PhysDevExtTramp 218
    PhysDevExtTramp 219
    PhysDevExtTramp 220
    PhysDevExtTramp 221
    PhysDevExtTramp 222
    PhysDevExtTramp 223
    PhysDevExtTramp 224
    PhysDevExtTramp 225
    PhysDevExtTramp 226
    PhysDevExtTramp 227
    PhysDevExtTramp 228
    PhysDevExtTramp 229
    PhysDevExtTramp 230
    PhysDevExtTramp 231
    PhysDevExtTramp 232
    PhysDevExtTramp 233
    PhysDevExtTramp 234
    PhysDevExtTramp 235
    PhysDevExtTramp 236
    PhysDevExtTramp 237
    PhysDevExtTramp 238
    PhysDevExtTramp 239
    PhysDevExtTramp 240
    PhysDevExtTramp 241
    PhysDevExtTramp 242
    PhysDevExtTramp 243
    PhysDevExtTramp 244
    PhysDevExtTramp 245
    PhysDevExtTramp 246
    PhysDevExtTramp 247
    PhysDevExtTramp 248
    PhysDevExtTramp 249

    PhysDevExtTermin 0
    PhysDevExtTermin 1
    PhysDevExtTermin 2
    PhysDevExtTermin 3
    PhysDevExtTermin 4
    PhysDevExtTermin 5
    PhysDevExtTermin 6
    PhysDevExtTermin 7
    PhysDevExtTermin 8
    PhysDevExtTermin 9
    PhysDevExtTermin 10
    PhysDevExtTermin 11
    PhysDevExtTermin 12
    PhysDevExtTermin 13
    PhysDevExtTermin 14
    PhysDevExtTermin 15
    PhysDevExtTermin 16
    PhysDevExtTermin 17
    PhysDevExtTermin 18
    PhysDevExtTermin 19
    PhysDevExtTermin 20
    PhysDevExtTermin 21
    PhysDevExtTermin 22
    PhysDevExtTermin 23
    PhysDevExtTermin 24
    PhysDevExtTermin 25
    PhysDevExtTermin 26
    PhysDevExtTermin 27
    PhysDevExtTermin 28
    PhysDevExtTermin 29
    PhysDevExtTermin 30
    PhysDevExtTermin 31
    PhysDevExtTermin 32
    PhysDevExtTermin 33
    PhysDevExtTermin 34
    PhysDevExtTermin 35
    PhysDevExtTermin 36
    PhysDevExtTermin 37
    PhysDevExtTermin 38
    PhysDevExtTermin 39
    PhysDevExtTermin 40
    PhysDevExtTermin 41
    PhysDevExtTermin 42
    PhysDevExtTermin 43
    PhysDevExtTermin 44
    PhysDevExtTermin 45
    PhysDevExtTermin 46
    PhysDevExtTermin 47
    PhysDevExtTermin 48
    PhysDevExtTermin 49
    PhysDevExtTermin 50
    PhysDevExtTermin 51
    PhysDevExtTermin 52
    PhysDevExtTermin 53
    PhysDevExtTermin 54
    PhysDevExtTermin 55
    PhysDevExtTermin 56
    PhysDevExtTermin 57
    PhysDevExtTermin 58
    PhysDevExtTermin 59
    PhysDevExtTermin 60
    PhysDevExtTermin 61
    PhysDevExtTermin 62
    PhysDevExtTermin 63
    PhysDevExtTermin 64
    PhysDevExtTermin 65
    PhysDevExtTermin 66
    PhysDevExtTermin 67
    PhysDevExtTermin 68
    PhysDevExtTermin 69
    PhysDevExtTermin 70
    PhysDevExtTermin 71
    PhysDevExtTermin 72
    PhysDevExtTermin 73
    PhysDevExtTermin 74
    PhysDevExtTermin 75
    PhysDevExtTermin 76
    PhysDevExtTermin 77
    PhysDevExtTermin 78
    PhysDevExtTermin 79
    PhysDevExtTermin 80
    PhysDevExtTermin 81
    PhysDevExtTermin 82
    PhysDevExtTermin 83
    PhysDevExtTermin 84
    PhysDevExtTermin 85
    PhysDevExtTermin 86
    PhysDevExtTermin 87
    PhysDevExtTermin 88
    PhysDevExtTermin 89
    PhysDevExtTermin 90
    PhysDevExtTermin 91
    PhysDevExtTermin 92
    PhysDevExtTermin 93
    PhysDevExtTermin 94
    PhysDevExtTermin 95
    PhysDevExtTermin 96
    PhysDevExtTermin 97
    PhysDevExtTermin 98
    PhysDevExtTermin 99
    PhysDevExtTermin 100
    PhysDevExtTermin 101
    PhysDevExtTermin 102
    PhysDevExtTermin 103
    PhysDevExtTermin 104
    PhysDevExtTermin 105
    PhysDevExtTermin 106
    PhysDevExtTermin 107
    PhysDevExtTermin 108
    PhysDevExtTermin 109
    PhysDevExtTermin 110
    PhysDevExtTermin 111
    PhysDevExtTermin 112
    PhysDevExtTermin 113
    PhysDevExtTermin 114
    PhysDevExtTermin 115
    PhysDevExtTermin 116
    PhysDevExtTermin 117
    PhysDevExtTermin 118
    PhysDevExtTermin 119
    PhysDevExtTermin 120
    PhysDevExtTermin 121
    PhysDevExtTermin 122
    PhysDevExtTermin 123
    PhysDevExtTermin 124
    PhysDevExtTermin 125
    PhysDevExtTermin 126
    PhysDevExtTermin 127
    PhysDevExtTermin 128
    PhysDevExtTermin 129
    PhysDevExtTermin 130
    PhysDevExtTermin 131
    PhysDevExtTermin 132
    PhysDevExtTermin 133
    PhysDevExtTermin 134
    PhysDevExtTermin 135
    PhysDevExtTermin 136
    PhysDevExtTermin 137
    PhysDevExtTermin 138
    PhysDevExtTermin 139
    PhysDevExtTermin 140
    PhysDevExtTermin 141
    PhysDevExtTermin 142
    PhysDevExtTermin 143
    PhysDevExtTermin 144
    PhysDevExtTermin 145
    PhysDevExtTermin 146
    PhysDevExtTermin 147
    PhysDevExtTermin 148
    PhysDevExtTermin 149
    PhysDevExtTermin 150
    PhysDevExtTermin 151
    PhysDevExtTermin 152
    PhysDevExtTermin 153
    PhysDevExtTermin 154
    PhysDevExtTermin 155
    PhysDevExtTermin 156
    PhysDevExtTermin 157
    PhysDevExtTermin 158
    PhysDevExtTermin 159
    PhysDevExtTermin 160
    PhysDevExtTermin 161
    PhysDevExtTermin 162
    PhysDevExtTermin 163
    PhysDevExtTermin 164
    PhysDevExtTermin 165
    PhysDevExtTermin 166
    PhysDevExtTermin 167
    PhysDevExtTermin 168
    PhysDevExtTermin 169
    PhysDevExtTermin 170
    PhysDevExtTermin 171
    PhysDevExtTermin 172
    PhysDevExtTermin 173
    PhysDevExtTermin 174
    PhysDevExtTermin 175
    PhysDevExtTermin 176
    PhysDevExtTermin 177
    PhysDevExtTermin 178
    PhysDevExtTermin 179
    PhysDevExtTermin 180
    PhysDevExtTermin 181
    PhysDevExtTermin 182
    PhysDevExtTermin 183
    PhysDevExtTermin 184
    PhysDevExtTermin 185
    PhysDevExtTermin 186
    PhysDevExtTermin 187
    PhysDevExtTermin 188
    PhysDevExtTermin 189
    PhysDevExtTermin 190
    PhysDevExtTermin 191
    PhysDevExtTermin 192
    PhysDevExtTermin 193
    PhysDevExtTermin 194
    PhysDevExtTermin 195
    PhysDevExtTermin 196
    PhysDevExtTermin 197
    PhysDevExtTermin 198
    PhysDevExtTermin 199
    PhysDevExtTermin 200
    PhysDevExtTermin 201
    PhysDevExtTermin 202
    PhysDevExtTermin 203
    PhysDevExtTermin 204
    PhysDevExtTermin 205
    PhysDevExtTermin 206
    PhysDevExtTermin 207
    PhysDevExtTermin 208
    PhysDevExtTermin 209
    PhysDevExtTermin 210
    PhysDevExtTermin 211
    PhysDevExtTermin 212
    PhysDevExtTermin 213
    PhysDevExtTermin 214
    PhysDevExtTermin 215
    PhysDevExtTermin 216
    PhysDevExtTermin 217
    PhysDevExtTermin 218
    PhysDevExtTermin 219
    PhysDevExtTermin 220
    PhysDevExtTermin 221
    PhysDevExtTermin 222
    PhysDevExtTermin 223
    PhysDevExtTermin 224
    PhysDevExtTermin 225
    PhysDevExtTermin 226
    PhysDevExtTermin 227
    PhysDevExtTermin 228
    PhysDevExtTermin 229
    PhysDevExtTermin 230
    PhysDevExtTermin 231
    PhysDevExtTermin 232
    PhysDevExtTermin 233
    PhysDevExtTermin 234
    PhysDevExtTermin 235
    PhysDevExtTermin 236
    PhysDevExtTermin 237
    PhysDevExtTermin 238
    PhysDevExtTermin 239
    PhysDevExtTermin 240
    PhysDevExtTermin 241
    PhysDevExtTermin 242
    PhysDevExtTermin 243
    PhysDevExtTermin 244
    PhysDevExtTermin 245
    PhysDevExtTermin 246
    PhysDevExtTermin 247
    PhysDevExtTermin 248
    PhysDevExtTermin 249

    DevExtTramp 0
    DevExtTramp 1
    DevExtTramp 2
    DevExtTramp 3
    DevExtTramp 4
    DevExtTramp 5
    DevExtTramp 6
    DevExtTramp 7
    DevExtTramp 8
    DevExtTramp 9
    DevExtTramp 10
    DevExtTramp 11
    DevExtTramp 12
    DevExtTramp 13
    DevExtTramp 14
    DevExtTramp 15
    DevExtTramp 16
    DevExtTramp 17
    DevExtTramp 18
    DevExtTramp 19
    DevExtTramp 20
    DevExtTramp 21
    DevExtTramp 22
    DevExtTramp 23
    DevExtTramp 24
    DevExtTramp 25
    DevExtTramp 26
    DevExtTramp 27
    DevExtTramp 28
    DevExtTramp 29
    DevExtTramp 30
    DevExtTramp 31
    DevExtTramp 32
    DevExtTramp 33
    DevExtTramp 34
    DevExtTramp 35
    DevExtTramp 36
    DevExtTramp 37
    DevExtTramp 38
    DevExtTramp 39
    DevExtTramp 40
    DevExtTramp 41
    DevExtTramp 42
    DevExtTramp 43
    DevExtTramp 44
    DevExtTramp 45
    DevExtTramp 46
    DevExtTramp 47
    DevExtTramp 48
    DevExtTramp 49
    DevExtTramp 50
    DevExtTramp 51
    DevExtTramp 52
    DevExtTramp 53
    DevExtTramp 54
    DevExtTramp 55
    DevExtTramp 56
    DevExtTramp 57
    DevExtTramp 58
    DevExtTramp 59
    DevExtTramp 60
    DevExtTramp 61
    DevExtTramp 62
    DevExtTramp 63
    DevExtTramp 64
    DevExtTramp 65
    DevExtTramp 66
    DevExtTramp 67
    DevExtTramp 68
    DevExtTramp 69
    DevExtTramp 70
    DevExtTramp 71
    DevExtTramp 72
    DevExtTramp 73
    DevExtTramp 74
    DevExtTramp 75
    DevExtTramp 76
    DevExtTramp 77
    DevExtTramp 78
    DevExtTramp 79
    DevExtTramp 80
    DevExtTramp 81
    DevExtTramp 82
    DevExtTramp 83
    DevExtTramp 84
    DevExtTramp 85
    DevExtTramp 86
    DevExtTramp 87
    DevExtTramp 88
    DevExtTramp 89
    DevExtTramp 90
    DevExtTramp 91
    DevExtTramp 92
    DevExtTramp 93
    DevExtTramp 94
    DevExtTramp 95
    DevExtTramp 96
    DevExtTramp 97
    DevExtTramp 98
    DevExtTramp 99
    DevExtTramp 100
    DevExtTramp 101
    DevExtTramp 102
    DevExtTramp 103
    DevExtTramp 104
    DevExtTramp 105
    DevExtTramp 106
    DevExtTramp 107
    DevExtTramp 108
    DevExtTramp 109
    DevExtTramp 110
    DevExtTramp 111
    DevExtTramp 112
    DevExtTramp 113
    DevExtTramp 114
    DevExtTramp 115
    DevExtTramp 116
    DevExtTramp 117
    DevExtTramp 118
    DevExtTramp 119
    DevExtTramp 120
    DevExtTramp 121
    DevExtTramp 122
    DevExtTramp 123
    DevExtTramp 124
    DevExtTramp 125
    DevExtTramp 126
    DevExtTramp 127
    DevExtTramp 128
    DevExtTramp 129
    DevExtTramp 130
    DevExtTramp 131
    DevExtTramp 132
    DevExtTramp 133
    DevExtTramp 134
    DevExtTramp 135
    DevExtTramp 136
    DevExtTramp 137
    DevExtTramp 138
    DevExtTramp 139
    DevExtTramp 140
    DevExtTramp 141
    DevExtTramp 142
    DevExtTramp 143
    DevExtTramp 144
    DevExtTramp 145
    DevExtTramp 146
    DevExtTramp 147
    DevExtTramp 148
    DevExtTramp 149
    DevExtTramp 150
    DevExtTramp 151
    DevExtTramp 152
    DevExtTramp 153
    DevExtTramp 154
    DevExtTramp 155
    DevExtTramp 156
    DevExtTramp 157
    DevExtTramp 158
    DevExtTramp 159
    DevExtTramp 160
    DevExtTramp 161
    DevExtTramp 162
    DevExtTramp 163
    DevExtTramp 164
    DevExtTramp 165
    DevExtTramp 166
    DevExtTramp 167
    DevExtTramp 168
    DevExtTramp 169
    DevExtTramp 170
    DevExtTramp 171
    DevExtTramp 172
    DevExtTramp 173
    DevExtTramp 174
    DevExtTramp 175
    DevExtTramp 176
    DevExtTramp 177
    DevExtTramp 178
    DevExtTramp 179
    DevExtTramp 180
    DevExtTramp 181
    DevExtTramp 182
    DevExtTramp 183
    DevExtTramp 184
    DevExtTramp 185
    DevExtTramp 186
    DevExtTramp 187
    DevExtTramp 188
    DevExtTramp 189
    DevExtTramp 190
    DevExtTramp 191
    DevExtTramp 192
    DevExtTramp 193
    DevExtTramp 194
    DevExtTramp 195
    DevExtTramp 196
    DevExtTramp 197
    DevExtTramp 198
    DevExtTramp 199
    DevExtTramp 200
    DevExtTramp 201
    DevExtTramp 202
    DevExtTramp 203
    DevExtTramp 204
    DevExtTramp 205
    DevExtTramp 206
    DevExtTramp 207
    DevExtTramp 208
    DevExtTramp 209
    DevExtTramp 210
    DevExtTramp 211
    DevExtTramp 212
    DevExtTramp 213
    DevExtTramp 214
    DevExtTramp 215
    DevExtTramp 216
    DevExtTramp 217
    DevExtTramp 218
    DevExtTramp 219
    DevExtTramp 220
    DevExtTramp 221
    DevExtTramp 222
    DevExtTramp 223
    DevExtTramp 224
    DevExtTramp 225
    DevExtTramp 226
    DevExtTramp 227
    DevExtTramp 228
    DevExtTramp 229
    DevExtTramp 230
    DevExtTramp 231
    DevExtTramp 232
    DevExtTramp 233
    DevExtTramp 234
    DevExtTramp 235
    DevExtTramp 236
    DevExtTramp 237
    DevExtTramp 238
    DevExtTramp 239
    DevExtTramp 240
    DevExtTramp 241
    DevExtTramp 242
    DevExtTramp 243
    DevExtTramp 244
    DevExtTramp 245
    DevExtTramp 246
    DevExtTramp 247
    DevExtTramp 248
    DevExtTramp 249
