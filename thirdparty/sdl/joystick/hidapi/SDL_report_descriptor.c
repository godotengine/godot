/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#include "SDL_report_descriptor.h"

// This is a very simple (and non-compliant!) report descriptor parser
// used to quickly parse Xbox Bluetooth reports

typedef enum
{
    DescriptorItemTypeMain = 0,
    DescriptorItemTypeGlobal = 1,
    DescriptorItemTypeLocal = 2,
    DescriptorItemTypeReserved = 3,
} ItemType;

typedef enum
{
    MainTagInput = 0x8,
    MainTagOutput = 0x9,
    MainTagFeature = 0xb,
    MainTagCollection = 0xa,
    MainTagEndCollection = 0xc,
} MainTag;

typedef enum
{
    MainFlagConstant        = 0x0001,
    MainFlagVariable        = 0x0002,
    MainFlagRelative        = 0x0004,
    MainFlagWrap            = 0x0008,
    MainFlagNonLinear        = 0x0010,
    MainFlagNoPreferred        = 0x0020,
    MainFlagNullState        = 0x0040,
    MainFlagVolatile        = 0x0080,
    MainFlagBufferedBytes    = 0x0100,
} MainFlag;

typedef enum
{
    GlobalTagUsagePage = 0x0,
    GlobalTagLogicalMinimum = 0x1,
    GlobalTagLogicalMaximum = 0x2,
    GlobalTagPhysicalMinimum = 0x3,
    GlobalTagPhysicalMaximum = 0x4,
    GlobalTagUnitExponent = 0x5,
    GlobalTagUnit = 0x6,
    GlobalTagReportSize = 0x7,
    GlobalTagReportID = 0x8,
    GlobalTagReportCount = 0x9,
    GlobalTagPush = 0xa,
    GlobalTagPop = 0xb,
} GlobalTag;

typedef enum
{
    LocalTagUsage = 0x0,
    LocalTagUsageMinimum = 0x1,
    LocalTagUsageMaximum = 0x2,
    LocalTagDesignatorIndex = 0x3,
    LocalTagDesignatorMinimum = 0x4,
    LocalTagDesignatorMaximum = 0x5,
    LocalTagStringIndex = 0x7,
    LocalTagStringMinimum = 0x8,
    LocalTagStringMaximum = 0x9,
    LocalTagDelimiter = 0xa,
} LocalTag;

typedef struct
{
    Uint32 usage_page;
    Uint32 report_size;
    Uint32 report_count;
    Uint32 report_id;
} DescriptorGlobalState;

typedef struct
{
    Uint32 usage_minimum;
    Uint32 usage_maximum;
    int usage_maxcount;
    int usage_count;
    Uint32 *usages;
} DescriptorLocalState;

typedef struct
{
    int collection_depth;
    DescriptorGlobalState global;
    DescriptorLocalState local;
    int field_maxcount;
    int field_count;
    int field_offset;
    DescriptorInputField *fields;
} DescriptorContext;

static void DebugDescriptor(DescriptorContext *ctx, const char *fmt, ...)
{
#ifdef DEBUG_DESCRIPTOR
    va_list ap;
    va_start(ap, fmt);
    char *message = NULL;
    SDL_vasprintf(&message, fmt, ap);
    va_end(ap);
    if (ctx->collection_depth > 0) {
        size_t len = 4 * ctx->collection_depth + SDL_strlen(message) + 1;
        char *output = (char *)SDL_malloc(len);
        if (output) {
            SDL_memset(output, ' ', 4 * ctx->collection_depth);
            output[4 * ctx->collection_depth] = '\0';
            SDL_strlcat(output, message, len);
            SDL_free(message);
            message = output;
        }
    }
    SDL_Log("%s", message);
    SDL_free(message);
#endif // DEBUG_DESCRIPTOR
}

static void DebugMainTag(DescriptorContext *ctx, const char *tag, Uint32 flags)
{
#ifdef DEBUG_DESCRIPTOR
    char message[1024] = { 0 };

    SDL_strlcat(message, tag, sizeof(message));
    SDL_strlcat(message, "(", sizeof(message));
    if (flags & MainFlagConstant) {
        SDL_strlcat(message, " Constant", sizeof(message));
    } else {
        SDL_strlcat(message, " Data", sizeof(message));
    }
    if (flags & MainFlagVariable) {
        SDL_strlcat(message, " Variable", sizeof(message));
    } else {
        SDL_strlcat(message, " Array", sizeof(message));
    }
    if (flags & MainFlagRelative) {
        SDL_strlcat(message, " Relative", sizeof(message));
    } else {
        SDL_strlcat(message, " Absolute", sizeof(message));
    }
    if (flags & MainFlagWrap) {
        SDL_strlcat(message, " Wrap", sizeof(message));
    } else {
        SDL_strlcat(message, " No Wrap", sizeof(message));
    }
    if (flags & MainFlagNonLinear) {
        SDL_strlcat(message, " Non Linear", sizeof(message));
    } else {
        SDL_strlcat(message, " Linear", sizeof(message));
    }
    if (flags & MainFlagNoPreferred) {
        SDL_strlcat(message, " No Preferred", sizeof(message));
    } else {
        SDL_strlcat(message, " Preferred State", sizeof(message));
    }
    if (flags & MainFlagNullState) {
        SDL_strlcat(message, " Null State", sizeof(message));
    } else {
        SDL_strlcat(message, " No Null Position", sizeof(message));
    }
    if (flags & MainFlagVolatile) {
        SDL_strlcat(message, " Volatile", sizeof(message));
    } else {
        SDL_strlcat(message, " Non Volatile", sizeof(message));
    }
    if (flags & MainFlagBufferedBytes) {
        SDL_strlcat(message, " Buffered Bytes", sizeof(message));
    } else {
        SDL_strlcat(message, " Bit Field", sizeof(message));
    }
    SDL_strlcat(message, " )", sizeof(message));

    DebugDescriptor(ctx, "%s", message);

#endif // DEBUG_DESCRIPTOR
}

static Uint32 ReadValue(const Uint8 *data, int size)
{
    Uint32 value = 0;

    int shift = 0;
    while (size--) {
        value |= ((Uint32)(*data++)) << shift;
        shift += 8;
    }
    return value;
}

static void ResetLocalState(DescriptorContext *ctx)
{
    ctx->local.usage_minimum = 0;
    ctx->local.usage_maximum = 0;
    ctx->local.usage_count = 0;
}

static bool AddUsage(DescriptorContext *ctx, Uint32 usage)
{
    if (ctx->local.usage_count == ctx->local.usage_maxcount) {
        int usage_maxcount = ctx->local.usage_maxcount + 4;
        Uint32 *usages = (Uint32 *)SDL_realloc(ctx->local.usages, usage_maxcount * sizeof(*usages));
        if (!usages) {
            return false;
        }
        ctx->local.usages = usages;
        ctx->local.usage_maxcount = usage_maxcount;
    }

    if (usage <= 0xFFFF) {
        usage |= (ctx->global.usage_page << 16);
    }
    ctx->local.usages[ctx->local.usage_count++] = usage;
    return true;
}

static bool AddInputField(DescriptorContext *ctx, Uint32 usage, int bit_size)
{
    if (ctx->field_count == ctx->field_maxcount) {
        int field_maxcount = ctx->field_maxcount + 4;
        DescriptorInputField *fields = (DescriptorInputField *)SDL_realloc(ctx->fields, field_maxcount * sizeof(*fields));
        if (!fields) {
            return false;
        }
        ctx->fields = fields;
        ctx->field_maxcount = field_maxcount;
    }

    DescriptorInputField *field = &ctx->fields[ctx->field_count++];
    field->report_id = (Uint8)ctx->global.report_id;
    field->usage = usage;
    field->bit_offset = ctx->field_offset;
    field->bit_size = bit_size;

    DebugDescriptor(ctx, "Adding report %d field 0x%.8x size %d bits at bit offset %d", field->report_id, field->usage, field->bit_size, field->bit_offset);
    return true;
}

static bool AddInputFields(DescriptorContext *ctx)
{
    Uint32 usage = 0;

    if (ctx->global.report_count == 0 || ctx->global.report_size == 0) {
        return true;
    }

    if (ctx->local.usage_count == 0 &&
        ctx->local.usage_minimum > 0 &&
        ctx->local.usage_maximum >= ctx->local.usage_minimum) {
        for (usage = ctx->local.usage_minimum; usage <= ctx->local.usage_maximum; ++usage) {
            if (!AddUsage(ctx, usage)) {
                return false;
            }
        }
    }

    int usage_index = 0;
    for (Uint32 i = 0; i < ctx->global.report_count; ++i) {
        if (usage_index < ctx->local.usage_count) {
            usage = ctx->local.usages[usage_index];
            if (usage_index < (ctx->local.usage_count - 1)) {
                ++usage_index;
            }
        }

        int size = (int)ctx->global.report_size;
        if (usage > 0) {
            if (!AddInputField(ctx, usage, size)) {
                return false;
            }
        }
        ctx->field_offset += size;
    }
    return true;
}

static bool ParseMainItem(DescriptorContext *ctx, int tag, int size, const Uint8 *data)
{
    Uint32 flags;

    switch (tag) {
    case MainTagInput:
        flags = ReadValue(data, size);
        DebugMainTag(ctx, "MainTagInput", flags);
        AddInputFields(ctx);
        break;
    case MainTagOutput:
        flags = ReadValue(data, size);
        DebugMainTag(ctx, "MainTagOutput", flags);
        break;
    case MainTagFeature:
        flags = ReadValue(data, size);
        DebugMainTag(ctx, "MainTagFeature", flags);
        break;
    case MainTagCollection:
        DebugDescriptor(ctx, "MainTagCollection");
        switch (*data) {
        case 0x00:
            DebugDescriptor(ctx, "Physical");
            break;
        case 0x01:
            DebugDescriptor(ctx, "Application");
            break;
        case 0x02:
            DebugDescriptor(ctx, "Logical");
            break;
        case 0x03:
            DebugDescriptor(ctx, "Report");
            break;
        case 0x04:
            DebugDescriptor(ctx, "Named Array");
            break;
        case 0x05:
            DebugDescriptor(ctx, "Usage Switch");
            break;
        case 0x06:
            DebugDescriptor(ctx, "Usage Modifier");
            break;
        default:
            break;
        }
        ++ctx->collection_depth;
        break;
    case MainTagEndCollection:
        if (ctx->collection_depth > 0) {
            --ctx->collection_depth;
        }
        DebugDescriptor(ctx, "MainTagEndCollection");
        break;
    default:
        DebugDescriptor(ctx, "Unknown main tag: %d", tag);
        break;
    }

    ResetLocalState(ctx);

    return true;
}

static bool ParseGlobalItem(DescriptorContext *ctx, int tag, int size, const Uint8 *data)
{
    Uint32 value;

    switch (tag) {
    case GlobalTagUsagePage:
        ctx->global.usage_page = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagUsagePage: 0x%.4x", ctx->global.usage_page);
        break;
    case GlobalTagLogicalMinimum:
        value = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagLogicalMinimum: %u", value);
        break;
    case GlobalTagLogicalMaximum:
        value = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagLogicalMaximum: %u", value);
        break;
    case GlobalTagPhysicalMinimum:
        value = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagPhysicalMinimum: %u", value);
        break;
    case GlobalTagPhysicalMaximum:
        value = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagPhysicalMaximum: %u", value);
        break;
    case GlobalTagUnitExponent:
        DebugDescriptor(ctx, "GlobalTagUnitExponent");
        break;
    case GlobalTagUnit:
        DebugDescriptor(ctx, "GlobalTagUnit");
        break;
    case GlobalTagReportSize:
        ctx->global.report_size = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagReportSize: %u", ctx->global.report_size);
        break;
    case GlobalTagReportID:
        ctx->global.report_id = ReadValue(data, size);
        ctx->field_offset = 0;
        DebugDescriptor(ctx, "GlobalTagReportID: %u", ctx->global.report_id);
        break;
    case GlobalTagReportCount:
        ctx->global.report_count = ReadValue(data, size);
        DebugDescriptor(ctx, "GlobalTagReportCount: %u", ctx->global.report_count);
        break;
    case GlobalTagPush:
        DebugDescriptor(ctx, "GlobalTagPush");
        break;
    case GlobalTagPop:
        DebugDescriptor(ctx, "GlobalTagPop");
        break;
    default:
        DebugDescriptor(ctx, "Unknown global tag");
        break;
    }
    return true;
}

static bool ParseLocalItem(DescriptorContext *ctx, int tag, int size, const Uint8 *data)
{
    Uint32 value;

    switch (tag) {
    case LocalTagUsage:
        value = ReadValue(data, size);
        AddUsage(ctx, value);
        DebugDescriptor(ctx, "LocalTagUsage: 0x%.4x", value);
        break;
    case LocalTagUsageMinimum:
        ctx->local.usage_minimum = ReadValue(data, size);
        DebugDescriptor(ctx, "LocalTagUsageMinimum: 0x%.4x", ctx->local.usage_minimum);
        break;
    case LocalTagUsageMaximum:
        ctx->local.usage_maximum = ReadValue(data, size);
        DebugDescriptor(ctx, "LocalTagUsageMaximum: 0x%.4x", ctx->local.usage_maximum);
        break;
    case LocalTagDesignatorIndex:
        DebugDescriptor(ctx, "LocalTagDesignatorIndex");
        break;
    case LocalTagDesignatorMinimum:
        DebugDescriptor(ctx, "LocalTagDesignatorMinimum");
        break;
    case LocalTagDesignatorMaximum:
        DebugDescriptor(ctx, "LocalTagDesignatorMaximum");
        break;
    case LocalTagStringIndex:
        DebugDescriptor(ctx, "LocalTagStringIndex");
        break;
    case LocalTagStringMinimum:
        DebugDescriptor(ctx, "LocalTagStringMinimum");
        break;
    case LocalTagStringMaximum:
        DebugDescriptor(ctx, "LocalTagStringMaximum");
        break;
    case LocalTagDelimiter:
        DebugDescriptor(ctx, "LocalTagDelimiter");
        break;
    default:
        DebugDescriptor(ctx, "Unknown local tag");
        break;
    }
    return true;
}

static bool ParseDescriptor(DescriptorContext *ctx, const Uint8 *descriptor, int descriptor_size)
{
    SDL_zerop(ctx);

    for (const Uint8 *here = descriptor; here < descriptor + descriptor_size; ) {
        static const int sizes[4] = { 0, 1, 2, 4 };
        Uint8 data = *here++;
        int size = sizes[(data & 0x3)];
        int type = ((data >> 2) & 0x3);
        int tag = (data >> 4);

        if ((here + size) > (descriptor + descriptor_size)) {
            return SDL_SetError("Invalid descriptor");
        }

#ifdef DEBUG_DESCRIPTOR
        SDL_Log("Data: 0x%.2x, size: %d, type: %d, tag: %d", data, size, type, tag);
#endif
        switch (type) {
        case DescriptorItemTypeMain:
            if (!ParseMainItem(ctx, tag, size, here)) {
                return false;
            }
            break;
        case DescriptorItemTypeGlobal:
            if (!ParseGlobalItem(ctx, tag, size, here)) {
                return false;
            }
            break;
        case DescriptorItemTypeLocal:
            if (!ParseLocalItem(ctx, tag, size, here)) {
                return false;
            }
            break;
        case DescriptorItemTypeReserved:
            // Long items are currently unsupported
            return SDL_Unsupported();
        }

        here += size;
    }
    return true;
}

static void CleanupContext(DescriptorContext *ctx)
{
    SDL_free(ctx->local.usages);
    SDL_free(ctx->fields);
}

SDL_ReportDescriptor *SDL_ParseReportDescriptor(const Uint8 *descriptor, int descriptor_size)
{
    SDL_ReportDescriptor *result = NULL;

    DescriptorContext ctx;
    if (ParseDescriptor(&ctx, descriptor, descriptor_size)) {
        result = (SDL_ReportDescriptor *)SDL_malloc(sizeof(*result));
        if (result) {
            result->field_count = ctx.field_count;
            result->fields = ctx.fields;
            ctx.fields = NULL;
        }
    }
    CleanupContext(&ctx);

    return result;
}

bool SDL_DescriptorHasUsage(SDL_ReportDescriptor *descriptor, Uint16 usage_page, Uint16 usage)
{
    if (!descriptor) {
        return false;
    }

    Uint32 full_usage = (((Uint32)usage_page << 16) | usage);
    for (int i = 0; i < descriptor->field_count; ++i) {
        if (descriptor->fields[i].usage == full_usage) {
            return true;
        }
    }
    return false;
}

void SDL_DestroyDescriptor(SDL_ReportDescriptor *descriptor)
{
    if (descriptor) {
        SDL_free(descriptor->fields);
        SDL_free(descriptor);
    }
}

bool SDL_ReadReportData(const Uint8 *data, int size, int bit_offset, int bit_size, Uint32 *value)
{
    int offset = (bit_offset / 8);
    if (offset >= size) {
        *value = 0;
        return SDL_SetError("Out of bounds reading report data");
    }

    *value = ReadValue(data + offset, (bit_size + 7) / 8);

    int shift = (bit_offset % 8);
    if (shift > 0) {
        *value >>= shift;
    }

    switch (bit_size) {
    case 1:
        *value &= 0x1;
        break;
    case 4:
        *value &= 0xf;
        break;
    case 10:
        *value &= 0x3ff;
        break;
    case 15:
        *value &= 0x7fff;
        break;
    default:
        SDL_assert((bit_size % 8) == 0);
        break;
    }
    return true;
}

#ifdef TEST_MAIN

#include <SDL3/SDL_main.h>

int main(int argc, char *argv[])
{
    const char *file = argv[1];
    if (argc < 2) {
        SDL_Log("Usage: %s file", argv[0]);
        return 1;
    }

    size_t descriptor_size = 0;
    Uint8 *descriptor = SDL_LoadFile(argv[1], &descriptor_size);
    if (!descriptor) {
        SDL_Log("Couldn't load %s: %s", argv[1], SDL_GetError());
        return 2;
    }

    DescriptorContext ctx;
    if (!ParseDescriptor(&ctx, descriptor, descriptor_size)) {
        SDL_Log("Couldn't parse %s: %s", argv[1], SDL_GetError());
        return 3;
    }
    return 0;
}

#endif // TEST_MAIN
