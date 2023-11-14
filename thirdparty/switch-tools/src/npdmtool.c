// Copyright 2018 SciresM
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include "cJSON.h"

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

#define MAGIC_META 0x4154454D
#define MAGIC_ACID 0x44494341
#define MAGIC_ACI0 0x30494341

/* FAC, FAH need to be tightly packed. */
#pragma pack(push, 1)
typedef struct {
    u8 Version;
    u8 CoiCount;
    u8 SdoiCount;
    u8 pad;
    u64 Perms;
    u64 CoiMin;
    u64 CoiMax;
    u64 SdoiMin;
    u64 SdoiMax;
} FilesystemAccessControl;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    u32 Version;
    u64 Perms;
    u32 CoiOffset;
    u32 CoiSize;
    u32 SdoiOffset;
    u32 SdoiSize;
} FilesystemAccessHeader;
#pragma pack(pop)

typedef struct {
    u32 Magic;
    u8 _0x4[0xC];
    u64 ProgramId;
    u64 _0x18;
    u32 FahOffset;
    u32 FahSize;
    u32 SacOffset;
    u32 SacSize;
    u32 KacOffset;
    u32 KacSize;
    u64 Padding;
} NpdmAci0;

typedef struct {
    u8 Signature[0x100];
    u8 Modulus[0x100];
    u32 Magic;
    u32 Size;
    u32 _0x208;
    u32 Flags;
    u64 ProgramIdRangeMin;
    u64 ProgramIdRangeMax;
    u32 FacOffset;
    u32 FacSize;
    u32 SacOffset;
    u32 SacSize;
    u32 KacOffset;
    u32 KacSize;
    u64 Padding;
} NpdmAcid;

typedef struct {
    u32 Magic;
    u32 SignatureKeyGeneration;
    u32 _0x8;
    u8 MmuFlags;
    u8 _0xD;
    u8 MainThreadPriority;
    u8 DefaultCpuId;
    u32 _0x10;
    u32 SystemResourceSize;
    u32 Version;
    u32 MainThreadStackSize;
    char Name[0x10];
    char ProductCode[0x10];
    u8 _0x40[0x30];
    u32 Aci0Offset;
    u32 Aci0Size;
    u32 AcidOffset;
    u32 AcidSize;
} NpdmHeader;


uint8_t* ReadEntireFile(const char* fn, size_t* len_out) {
    FILE* fd = fopen(fn, "rb");
    if (fd == NULL)
        return NULL;

    fseek(fd, 0, SEEK_END);
    size_t len = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    uint8_t* buf = malloc(len);
    if (buf == NULL) {
        fclose(fd);
        return NULL;
    }

    size_t rc = fread(buf, 1, len, fd);
    if (rc != len) {
        fclose(fd);
        free(buf);
        return NULL;
    }

    *len_out = len;
    return buf;
}

int cJSON_GetString(const cJSON *obj, const char *field, const char **out) {
  const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
  if (cJSON_IsString(config)) {
    *out = config->valuestring;
    return 1;
  } else {
    fprintf(stderr, "Failed to get %s (field not present).\n", field);
    return 0;
  }
}

int cJSON_GetU8(const cJSON *obj, const char *field, u8 *out) {
    const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
    if (cJSON_IsNumber(config)) {
        *out = (u8)config->valueint;
        return 1;
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", field);
        return 0;
    }
}

int cJSON_GetU16(const cJSON *obj, const char *field, u16 *out) {
    const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
    if (cJSON_IsNumber(config)) {
        *out = (u16)config->valueint;
        return 1;
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", field);
        return 0;
    }
}

int cJSON_GetU16FromObjectValue(const cJSON *config, u16 *out) {
    if (cJSON_IsNumber(config)) {
        *out = (u16)config->valueint;
        return 1;
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", config->string);
        return 0;
    }
}

int cJSON_GetBoolean(const cJSON *obj, const char *field, int *out) {
    const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
    if (cJSON_IsBool(config)) {
        if (cJSON_IsTrue(config)) {
            *out = 1;
        } else if (cJSON_IsFalse(config)) {
            *out = 0;
        } else {
            fprintf(stderr, "Unknown boolean value in %s.\n", field);
            return 0;
        }
        return 1;
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", field);
        return 0;
    }
}

int cJSON_GetBooleanOptional(const cJSON *obj, const char *field, int *out) {
    const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
    if (cJSON_IsBool(config)) {
        if (cJSON_IsTrue(config)) {
            *out = 1;
        } else if (cJSON_IsFalse(config)) {
            *out = 0;
        } else {
            fprintf(stderr, "Unknown boolean value in %s.\n", field);
            return 0;
        }
    } else {
        *out = 0;
    }
    return 1;
}

int cJSON_GetU64(const cJSON *obj, const char *field, u64 *out) {
    const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
    if (cJSON_IsString(config) && (config->valuestring != NULL)) {
        char *endptr = NULL;
        *out = strtoull(config->valuestring, &endptr, 16);
        if (config->valuestring == endptr) {
            fprintf(stderr, "Failed to get %s (empty string)\n", field);
            return 0;
        } else if (errno == ERANGE) {
            fprintf(stderr, "Failed to get %s (value out of range)\n", field);
            return 0;
        } else if (errno == EINVAL) {
            fprintf(stderr, "Failed to get %s (not base16 string)\n", field);
            return 0;
        } else if (errno) {
            fprintf(stderr, "Failed to get %s (unknown error)\n", field);
            return 0;
        } else {
            return 1;
        }
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", field);
        return 0;
    }
}

int cJSON_GetU32(const cJSON *obj, const char *field, u32 *out) {
    const cJSON *config = cJSON_GetObjectItemCaseSensitive(obj, field);
    if (cJSON_IsString(config) && (config->valuestring != NULL)) {
        char *endptr = NULL;
        *out = strtoul(config->valuestring, &endptr, 16);
        if (config->valuestring == endptr) {
            fprintf(stderr, "Failed to get %s (empty string)\n", field);
            return 0;
        } else if (errno == ERANGE) {
            fprintf(stderr, "Failed to get %s (value out of range)\n", field);
            return 0;
        } else if (errno == EINVAL) {
            fprintf(stderr, "Failed to get %s (not base16 string)\n", field);
            return 0;
        } else if (errno) {
            fprintf(stderr, "Failed to get %s (unknown error)\n", field);
            return 0;
        } else {
            return 1;
        }
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", field);
        return 0;
    }
}

int cJSON_GetU64FromObjectValue(const cJSON *config, u64 *out) {
    if (cJSON_IsString(config) && (config->valuestring != NULL)) {
        char *endptr = NULL;
        *out = strtoull(config->valuestring, &endptr, 16);
        if (config->valuestring == endptr) {
            fprintf(stderr, "Failed to get %s (empty string)\n", config->string);
            return 0;
        } else if (errno == ERANGE) {
            fprintf(stderr, "Failed to get %s (value out of range)\n", config->string);
            return 0;
        } else if (errno == EINVAL) {
            fprintf(stderr, "Failed to get %s (not base16 string)\n", config->string);
            return 0;
        } else if (errno) {
            fprintf(stderr, "Failed to get %s (unknown error)\n", config->string);
            return 0;
        } else {
            return 1;
        }
    } else {
        fprintf(stderr, "Failed to get %s (field not present).\n", config->string);
        return 0;
    }
}

int CreateNpdm(const char *json, void **dst, u32 *dst_size) {
    NpdmHeader header = {0};
    NpdmAci0 *aci0 = calloc(1, 0x100000);
    NpdmAcid *acid = calloc(1, 0x100000);
    if (aci0 == NULL || acid == NULL) {
        fprintf(stderr, "Failed to allocate NPDM resources!\n");
        exit(EXIT_FAILURE);
    }
    const cJSON *capability = NULL;
    const cJSON *capabilities = NULL;
    const cJSON *service = NULL;
    const cJSON *services = NULL;
    const cJSON *fsaccess = NULL;
    const cJSON *cois = NULL;
    const cJSON *coi = NULL;
    const cJSON *sdois = NULL;
    const cJSON *sdoi = NULL;

    int status = 0;
    cJSON *npdm_json = cJSON_Parse(json);
    if (npdm_json == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "JSON Parse Error: %s\n", error_ptr);
        }
        status = 0;
        goto NPDM_BUILD_END;
    }

    /* Initialize default NPDM values. */
    header.Magic = MAGIC_META; /* "META" */


    /* Parse name. */
    const cJSON *title_name = cJSON_GetObjectItemCaseSensitive(npdm_json, "name");
    if (cJSON_IsString(title_name) && (title_name->valuestring != NULL)) {
        strncpy(header.Name, title_name->valuestring, sizeof(header.Name) - 1);
    } else {
        fprintf(stderr, "Failed to get title name (name field not present).\n");
        status = 0;
        goto NPDM_BUILD_END;
    }

    /* Parse main_thread_stack_size. */
    u64 stack_size = 0;
    if (!cJSON_GetU64(npdm_json, "main_thread_stack_size", &stack_size)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    if (stack_size >> 32) {
        fprintf(stderr, "Error: Main thread stack size must be a u32!\n");
        status = 0;
        goto NPDM_BUILD_END;
    }
    header.MainThreadStackSize = (u32)(stack_size & 0xFFFFFFFF);

    /* Parse various config. */
    if (!cJSON_GetU8(npdm_json, "main_thread_priority", &header.MainThreadPriority)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    if (!cJSON_GetU8(npdm_json, "default_cpu_id", &header.DefaultCpuId)) {
        status = 0;
        goto NPDM_BUILD_END;
    }

    cJSON_GetU32(npdm_json, "system_resource_size", &header.SystemResourceSize); // optional

    /* Get version (deprecated name "process_category"). */
    if (!cJSON_GetU32(npdm_json, "version", &header.Version) && !cJSON_GetU32(npdm_json, "process_category", &header.Version)) { // optional
        header.Version = 0;
    }

    if (!cJSON_GetU8(npdm_json, "address_space_type", (u8 *)&header.MmuFlags)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    header.MmuFlags &= 3;
    header.MmuFlags <<= 1;
    int is_64_bit;
    if (!cJSON_GetBoolean(npdm_json, "is_64_bit", &is_64_bit)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    header.MmuFlags |= is_64_bit;

    int optimize_memory_allocation; // optional
    if (cJSON_GetBoolean(npdm_json, "optimize_memory_allocation", &optimize_memory_allocation)) {
        header.MmuFlags |= ((optimize_memory_allocation & 1) << 4);
    }

    int disable_device_address_space_merge; // optional
    if (cJSON_GetBoolean(npdm_json, "disable_device_address_space_merge", &disable_device_address_space_merge)) {
        header.MmuFlags |= ((disable_device_address_space_merge & 1) << 5);
    }

    u8 signature_key_generation; // optional
    if (cJSON_GetU8(npdm_json, "signature_key_generation", &signature_key_generation)) {
        header.SignatureKeyGeneration = signature_key_generation;
    } else {
        header.SignatureKeyGeneration = 0;
    }

    /* ACID. */
    memset(acid->Signature, 0, sizeof(acid->Signature));
    memset(acid->Modulus, 0, sizeof(acid->Modulus));
    acid->Magic = MAGIC_ACID; /* "ACID" */
    int is_retail;
    if (!cJSON_GetBoolean(npdm_json, "is_retail", &is_retail)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    acid->Flags |= is_retail;
    u8 pool_partition;
    if (!cJSON_GetU8(npdm_json, "pool_partition", &pool_partition)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    acid->Flags |= (pool_partition & 3) << 2;

    if (!cJSON_GetU64(npdm_json, "program_id_range_min", &acid->ProgramIdRangeMin) && !cJSON_GetU64(npdm_json, "title_id_range_min", &acid->ProgramIdRangeMin)) {
        status = 0;
        goto NPDM_BUILD_END;
    }
    if (!cJSON_GetU64(npdm_json, "program_id_range_max", &acid->ProgramIdRangeMax) && !cJSON_GetU64(npdm_json, "title_id_range_max", &acid->ProgramIdRangeMax)) {
        status = 0;
        goto NPDM_BUILD_END;
    }

    /* ACI0. */
    aci0->Magic = MAGIC_ACI0; /* "ACI0" */
    /* Parse program_id (or deprecated title_id). */
    if (!cJSON_GetU64(npdm_json, "program_id", &aci0->ProgramId) && !cJSON_GetU64(npdm_json, "title_id", &aci0->ProgramId)) {
        status = 0;
        goto NPDM_BUILD_END;
    }

    /* Fac. */
    fsaccess = cJSON_GetObjectItemCaseSensitive(npdm_json, "filesystem_access");
    if (!cJSON_IsObject(fsaccess)) {
        fprintf(stderr, "Filesystem Access must be an object!\n");
        status = 0;
        goto NPDM_BUILD_END;
    }

    FilesystemAccessControl *fac = (FilesystemAccessControl *)((u8 *)acid + sizeof(NpdmAcid));
    fac->Version = 1;
    if (!cJSON_GetU64(fsaccess, "permissions", &fac->Perms)) {
        status = 0;
        goto NPDM_BUILD_END;
    }

    fac->CoiMin    = 0;
    fac->CoiMax    = 0;
    fac->SdoiMin   = 0;
    fac->SdoiMax   = 0;
    fac->CoiCount  = 0;
    fac->SdoiCount = 0;

    acid->FacOffset = sizeof(NpdmAcid);
    acid->FacSize = sizeof(FilesystemAccessControl);
    acid->SacOffset = (acid->FacOffset + acid->FacSize + 0xF) & ~0xF;

    /* Fah. */
    FilesystemAccessHeader *fah = (FilesystemAccessHeader *)((u8 *)aci0 + sizeof(NpdmAci0));
    fah->Version = 1;
    fah->Perms = fac->Perms;
    fah->CoiOffset = sizeof(FilesystemAccessHeader);
    fah->CoiSize   = 0;

    cois = cJSON_GetObjectItemCaseSensitive(fsaccess, "content_owner_ids");
    if (cJSON_IsArray(cois)) {
        u32 *count = (u32 *)((u8 *)fah + fah->CoiOffset);
        u64 *id = (u64 *)((u8 *)count + sizeof(u32));
        cJSON_ArrayForEach(coi, cois) {
            if (!cJSON_GetU64FromObjectValue(coi, id)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            ++id;
            ++(*count);
        }

        if (*count > 0) {
            fah->CoiSize = sizeof(u32) + sizeof(u64) * (*count);
        }
    }

    fah->SdoiOffset = fah->CoiOffset + fah->CoiSize;
    fah->SdoiSize   = 0;

    sdois = cJSON_GetObjectItemCaseSensitive(fsaccess, "save_data_owner_ids");
    if (cJSON_IsArray(sdois)) {
        u32 *count = (u32 *)((u8 *)fah + fah->SdoiOffset);
        cJSON_ArrayForEach(sdoi, sdois) {
            if (!cJSON_IsObject(sdoi)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            ++(*count);
        }

        u8 *accessibility = (u8 *)count + sizeof(u32);
        u64 *id = (u64 *)(accessibility + (((*count) + 3ULL) & ~3ULL));

        cJSON_ArrayForEach(sdoi, sdois) {
            if (!cJSON_GetU8(sdoi, "accessibility", accessibility)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            if (!cJSON_GetU64(sdoi, "id", id)) {
                status = 0;
                goto NPDM_BUILD_END;
            }

            ++accessibility;
            ++id;
        }

        if (*count > 0) {
            fah->SdoiSize = sizeof(u32) + sizeof(u8) * ((((*count) + 3ULL) & ~3ULL)) + sizeof(u64) * (*count);
        }
    }

    aci0->FahOffset = sizeof(NpdmAci0);
    aci0->FahSize = sizeof(FilesystemAccessHeader) + fah->CoiSize + fah->SdoiSize;
    aci0->SacOffset = (aci0->FahOffset + aci0->FahSize + 0xF) & ~0xF;

    /* Sac. */
    u8 *sac = (u8*)aci0 + aci0->SacOffset;
    u32 sac_size = 0;

    services = cJSON_GetObjectItemCaseSensitive(npdm_json, "service_host");
    if (services != NULL && !cJSON_IsArray(services)) {
        fprintf(stderr, "Service Host must be an array!\n");
        status = 0;
        goto NPDM_BUILD_END;
    }

    cJSON_ArrayForEach(service, services) {
        int is_host = 1;
        char *service_name;

        if (!cJSON_IsString(service)) {
            fprintf(stderr, "service_access must be an array of string\n");
            status = 0;
            goto NPDM_BUILD_END;
        }
        service_name = service->valuestring;

        int cur_srv_len = strlen(service_name);
        if (cur_srv_len > 8 || cur_srv_len == 0) {
            fprintf(stderr, "Services must have name length 1 <= len <= 8!\n");
            status = 0;
            goto NPDM_BUILD_END;
        }
        u8 ctrl = (u8)(cur_srv_len - 1);
        if (is_host) {
            ctrl |= 0x80;
        }
        sac[sac_size++] = ctrl;
        memcpy(sac + sac_size, service_name, cur_srv_len);
        sac_size += cur_srv_len;
    }

    services = cJSON_GetObjectItemCaseSensitive(npdm_json, "service_access");
    if (!(services == NULL || cJSON_IsObject(services) || cJSON_IsArray(services))) {
      fprintf(stderr, "Service Access must be an array!\n");
      status = 0;
      goto NPDM_BUILD_END;
    }

    int sac_obj = 0;
    if (services != NULL && cJSON_IsObject(services)) {
      sac_obj = 1;
      fprintf(stderr, "Using deprecated service_access format. Please turn it into an array.\n");
    }

    cJSON_ArrayForEach(service, services) {
        int is_host = 0;
        char *service_name;

        if (sac_obj) {
            if (!cJSON_IsBool(service)) {
                fprintf(stderr, "Services must be of form service_name (str) : is_host (bool)\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            is_host = cJSON_IsTrue(service);
            service_name = service->string;
        } else {
            if (!cJSON_IsString(service)) {
                fprintf(stderr, "service_access must be an array of string\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            is_host = 0;
            service_name = service->valuestring;
        }

        int cur_srv_len = strlen(service_name);
        if (cur_srv_len > 8 || cur_srv_len == 0) {
            fprintf(stderr, "Services must have name length 1 <= len <= 8!\n");
            status = 0;
            goto NPDM_BUILD_END;
        }
        u8 ctrl = (u8)(cur_srv_len - 1);
        if (is_host) {
            ctrl |= 0x80;
        }
        sac[sac_size++] = ctrl;
        memcpy(sac + sac_size, service_name, cur_srv_len);
        sac_size += cur_srv_len;
    }


    memcpy((u8 *)acid + acid->SacOffset, sac, sac_size);
    aci0->SacSize = sac_size;
    acid->SacSize = sac_size;
    aci0->KacOffset = (aci0->SacOffset + aci0->SacSize + 0xF) & ~0xF;
    acid->KacOffset = (acid->SacOffset + acid->SacSize + 0xF) & ~0xF;

    /* Parse capabilities. */
    capabilities = cJSON_GetObjectItemCaseSensitive(npdm_json, "kernel_capabilities");
    if (!(cJSON_IsArray(capabilities) || cJSON_IsObject(capabilities))) {
        fprintf(stderr, "Kernel Capabilities must be an array!\n");
        status = 0;
        goto NPDM_BUILD_END;
    }

    int kac_obj = 0;
    if (cJSON_IsObject(capabilities)) {
        kac_obj = 1;
        fprintf(stderr, "Using deprecated kernel_capabilities format. Please turn it into an array.\n");
    }

    u32 *caps = (u32 *)((u8 *)aci0 + aci0->KacOffset);
    u32 cur_cap = 0;
    u32 desc;
    cJSON_ArrayForEach(capability, capabilities) {
        desc = 0;
        const char *type_str;
        const cJSON *value;

        if (kac_obj) {
          type_str = capability->string;
          value = capability;
        } else {
          if (!cJSON_GetString(capability, "type", &type_str)) {
            status = 0;
            goto NPDM_BUILD_END;
          }
          value = cJSON_GetObjectItemCaseSensitive(capability, "value");
        }

        if (!strcmp(type_str, "kernel_flags")) {
            if (!cJSON_IsObject(value)) {
                fprintf(stderr, "Kernel Flags Capability value must be object!\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            u8 highest_prio = 0, lowest_prio = 0, lowest_cpu = 0, highest_cpu = 0;
            if (!cJSON_GetU8(value, "highest_thread_priority", &highest_prio) ||
                !cJSON_GetU8(value, "lowest_thread_priority", &lowest_prio) ||
                !cJSON_GetU8(value, "highest_cpu_id", &highest_cpu) ||
                !cJSON_GetU8(value, "lowest_cpu_id", &lowest_cpu)) {
                status = 0;
                goto NPDM_BUILD_END;
            }

            u8 real_highest_prio = (lowest_prio < highest_prio) ? lowest_prio : highest_prio;
            u8 real_lowest_prio  = (lowest_prio > highest_prio) ? lowest_prio : highest_prio;

            desc = highest_cpu;
            desc <<= 8;
            desc |= lowest_cpu;
            desc <<= 6;
            desc |= (real_highest_prio & 0x3F);
            desc <<= 6;
            desc |= (real_lowest_prio & 0x3F);
            caps[cur_cap++] = (u32)((desc << 4) | (0x0007));
        } else if (!strcmp(type_str, "syscalls")) {
            if (!cJSON_IsObject(value)) {
                fprintf(stderr, "Syscalls Capability value must be object!\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            u32 num_descriptors;
            u32 descriptors[8] = {0}; /* alignup(0xC0/0x18); */
            char field_name[8] = {0};
            const cJSON *cur_syscall = NULL;
            u64 syscall_value = 0;
            cJSON_ArrayForEach(cur_syscall, value) {
                if (cJSON_IsNumber(cur_syscall)) {
                    syscall_value = (u64)cur_syscall->valueint;
                } else if (!cJSON_IsString(cur_syscall) || !cJSON_GetU64(value, cur_syscall->string, &syscall_value)) {
                    fprintf(stderr, "Error: Syscall entries must be integers or hex strings.\n");
                    status = 0;
                    goto NPDM_BUILD_END;
                }

                if (syscall_value >= 0xC0) {
                    fprintf(stderr, "Error: All syscall entries must be numbers in [0, 0xBF]\n");
                    status = 0;
                    goto NPDM_BUILD_END;
                }
                descriptors[syscall_value / 0x18] |= (1UL << (syscall_value % 0x18));
            }
            for (unsigned int i = 0; i < 8; i++) {
                if (descriptors[i]) {
                    desc = descriptors[i] | (i << 24);
                    caps[cur_cap++] = (u32)((desc << 5) | (0x000F));
                }
            }
        } else if (!strcmp(type_str, "map")) {
            if (!cJSON_IsObject(value)) {
                fprintf(stderr, "Map Capability value must be object!\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            u64 map_address = 0;
            u64 map_size = 0;
            int is_ro;
            int is_io;
            if (!cJSON_GetU64(value, "address", &map_address) ||
                !cJSON_GetU64(value, "size", &map_size) ||
                !cJSON_GetBoolean(value, "is_ro", &is_ro) ||
                !cJSON_GetBoolean(value, "is_io", &is_io)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            desc = (u32)((map_address >> 12) & 0x00FFFFFFULL);
            desc |= is_ro << 24;
            caps[cur_cap++] = (u32)((desc << 7) | (0x003F));

            desc = (u32)((map_size >> 12) & 0x000FFFFFULL);
            desc |= (u32)(((map_address >> 36) & 0xFULL) << 20);
            is_io ^= 1;
            desc |= is_io << 24;
            caps[cur_cap++] = (u32)((desc << 7) | (0x003F));
        } else if (!strcmp(type_str, "map_page")) {
            u64 page_address = 0;
            if (!cJSON_GetU64FromObjectValue(value, &page_address)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            desc = (u32)((page_address >> 12) & 0x00FFFFFFULL);
            caps[cur_cap++] = (u32)((desc << 8) | (0x007F));
        } else if (!strcmp(type_str, "map_region")) {
            if (cur_cap + 1 > 0x20) {
                fprintf(stderr, "Error: Too many capabilities!\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            if (!cJSON_IsArray(value)) {
                fprintf(stderr, "Map Region capability value must be array!\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            u8 regions[3] = {0};
            int is_ro[3] = {0};
            const cJSON *cur_region = NULL;
            int index = 0;
            cJSON_ArrayForEach(cur_region, value) {
                if (index >= 3) {
                    fprintf(stderr, "Too many region descriptors!\n");
                    status = 0;
                    goto NPDM_BUILD_END;
                }
                if (!cJSON_IsObject(cur_region)) {
                    fprintf(stderr, "Region descriptor value must be object!\n");
                    status = 0;
                    goto NPDM_BUILD_END;
                }

                if (!cJSON_GetU8(cur_region, "region_type", &regions[index]) ||
                    !cJSON_GetBoolean(cur_region, "is_ro", &is_ro[index])) {
                    status = 0;
                    goto NPDM_BUILD_END;
                }

                index++;
            }

            u32 capability = 0x3FF;
            for (int i = 0; i < 3; ++i) {
                capability |= ((regions[i] & 0x3F) | ((is_ro[i] & 1) << 6)) << (11 + 7 * i);
            }
            caps[cur_cap++] = capability;
        } else if (!strcmp(type_str, "irq_pair")) {
            if (!cJSON_IsArray(value) || cJSON_GetArraySize(value) != 2) {
                fprintf(stderr, "Error: IRQ Pairs must have size 2 array value.\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            const cJSON *irq = NULL;
            int desc_idx = 0;
            cJSON_ArrayForEach(irq, value) {
                if (cJSON_IsNull(irq)) {
                    desc |= 0x3FF << desc_idx;
                } else if (cJSON_IsNumber(irq)) {
                    desc |= (((u16)(irq->valueint)) & 0x3FF) << desc_idx;
                } else {
                    fprintf(stderr, "Failed to parse IRQ value.\n");
                    status = 0;
                    goto NPDM_BUILD_END;
                }
                desc_idx += 10;
            }
            caps[cur_cap++] = (u32)((desc << 12) | (0x07FF));
        } else if (!strcmp(type_str, "application_type")) {
            if (!cJSON_GetU16FromObjectValue(value, (u16 *)&desc)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            desc &= 7;
            caps[cur_cap++] = (u32)((desc << 14) | (0x1FFF));
        } else if (!strcmp(type_str, "min_kernel_version")) {
            u64 kern_ver = 0;
            if (cJSON_IsNumber(value)) {
                kern_ver = (u64)value->valueint;
            } else if (!cJSON_IsString(value) || !cJSON_GetU64FromObjectValue(value, &kern_ver)) {
                fprintf(stderr, "Error: Kernel version must be integer or hex strings.\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            desc = (kern_ver) & 0xFFFF;
            caps[cur_cap++] = (u32)((desc << 15) | (0x3FFF));
        } else if (!strcmp(type_str, "handle_table_size")) {
            if (!cJSON_GetU16FromObjectValue(value, (u16 *)&desc)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            caps[cur_cap++] = (u32)((desc << 16) | (0x7FFF));
        } else if (!strcmp(type_str, "debug_flags")) {
            if (!cJSON_IsObject(value)) {
                fprintf(stderr, "Debug Flag Capability value must be object!\n");
                status = 0;
                goto NPDM_BUILD_END;
            }
            int allow_debug = 0;
            int force_debug = 0;
            if (!cJSON_GetBoolean(value, "allow_debug", &allow_debug)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            if (!cJSON_GetBoolean(value, "force_debug", &force_debug)) {
                status = 0;
                goto NPDM_BUILD_END;
            }
            desc = (allow_debug & 1) | ((force_debug & 1) << 1);
            caps[cur_cap++] = (u32)((desc << 17) | (0xFFFF));
        }
    }
    aci0->KacSize = cur_cap * sizeof(u32);
    acid->KacSize = aci0->KacSize;
    memcpy((u8 *)acid + acid->KacOffset, caps, aci0->KacSize);

    header.AcidOffset = sizeof(header);
    header.AcidSize = acid->KacOffset + acid->KacSize;
    acid->Size = header.AcidSize - sizeof(acid->Signature);
    header.Aci0Offset = (header.AcidOffset + header.AcidSize + 0xF) & ~0xF;
    header.Aci0Size = aci0->KacOffset + aci0->KacSize;
    u32 total_size = header.Aci0Offset + header.Aci0Size;
    u8 *npdm = calloc(1, total_size);
    if (npdm == NULL) {
        fprintf(stderr, "Failed to allocate output!\n");
        exit(EXIT_FAILURE);
    }
    memcpy(npdm, &header, sizeof(header));
    memcpy(npdm + header.AcidOffset, acid, header.AcidSize);
    memcpy(npdm + header.Aci0Offset, aci0, header.Aci0Size);
    free(acid);
    free(aci0);
    *dst = npdm;
    *dst_size = total_size;

    status = 1;
    NPDM_BUILD_END:
    cJSON_Delete(npdm_json);
    return status;
}

// -- GODOT start --
int npdm_main(int argc, char* argv[]) {
// -- GODOT end --
    if (argc != 3) {
        fprintf(stderr, "%s <json-file> <npdm-file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    void *npdm;
    u32 npdm_size;

    if (sizeof(NpdmHeader) != 0x80 || sizeof(NpdmAcid) != 0x240 || sizeof(NpdmAci0) != 0x40) {
        fprintf(stderr, "Bad compile environment!\n");
        return EXIT_FAILURE;
    }

    size_t json_len;
    uint8_t* json = ReadEntireFile(argv[1], &json_len);
    if (json == NULL) {
        fprintf(stderr, "Failed to read descriptor json!\n");
        return EXIT_FAILURE;
    }

    if (!CreateNpdm(json, &npdm, &npdm_size)) {
        fprintf(stderr, "Failed to parse descriptor json!\n");
        return EXIT_FAILURE;
    }

    FILE *f_out = fopen(argv[2], "wb");
    if (f_out == NULL) {
        fprintf(stderr, "Failed to open %s for writing!\n", argv[2]);
        return EXIT_FAILURE;
    }
    if (fwrite(npdm, 1, npdm_size, f_out) != npdm_size) {
        fprintf(stderr, "Failed to write NPDM to %s!\n", argv[2]);
        return EXIT_FAILURE;
    }
    fclose(f_out);
    free(npdm);

    return EXIT_SUCCESS;
}
