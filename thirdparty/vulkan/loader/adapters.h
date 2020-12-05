/*
* Copyright (c) 2019 The Khronos Group Inc.
* Copyright (c) 2019 Valve Corporation
* Copyright (c) 2019 LunarG, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Author: Lenny Komow <lenny@lunarg.com>
*/

typedef struct LoaderEnumAdapters2 {
    ULONG adapter_count;
    struct {
        UINT handle;
        LUID luid;
        ULONG source_count;
        BOOL present_move_regions_preferred;
    } * adapters;
} LoaderEnumAdapters2;

typedef _Check_return_ NTSTATUS(APIENTRY *PFN_LoaderEnumAdapters2)(const LoaderEnumAdapters2 *);

typedef enum AdapterInfoType {
    LOADER_QUERY_TYPE_REGISTRY = 48,
} AdapterInfoType;

typedef struct LoaderQueryAdapterInfo {
    UINT handle;
    AdapterInfoType type;
    VOID *private_data;
    UINT private_data_size;
} LoaderQueryAdapterInfo;

typedef _Check_return_ NTSTATUS(APIENTRY *PFN_LoaderQueryAdapterInfo)(const LoaderQueryAdapterInfo *);

typedef enum LoaderQueryRegistryType {
    LOADER_QUERY_REGISTRY_ADAPTER_KEY = 1,
} LoaderQueryRegistryType;

typedef enum LoaderQueryRegistryStatus {
    LOADER_QUERY_REGISTRY_STATUS_SUCCESS = 0,
    LOADER_QUERY_REGISTRY_STATUS_BUFFER_OVERFLOW = 1,
} LoaderQueryRegistryStatus;

typedef struct LoaderQueryRegistryFlags {
    union {
        struct {
            UINT translate_path : 1;
            UINT mutable_value : 1;
            UINT reserved : 30;
        };
        UINT value;
    };
} LoaderQueryRegistryFlags;

typedef struct LoaderQueryRegistryInfo {
    LoaderQueryRegistryType query_type;
    LoaderQueryRegistryFlags query_flags;
    WCHAR value_name[MAX_PATH];
    ULONG value_type;
    ULONG physical_adapter_index;
    ULONG output_value_size;
    LoaderQueryRegistryStatus status;
    union {
        DWORD output_dword;
        UINT64 output_qword;
        WCHAR output_string[1];
        BYTE output_binary[1];
    };
} LoaderQueryRegistryInfo;
