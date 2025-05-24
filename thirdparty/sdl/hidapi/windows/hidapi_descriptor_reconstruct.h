/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 libusb/hidapi Team

 Copyright 2022, All Rights Reserved.

 At the discretion of the user of this library,
 this software may be licensed under the terms of the
 GNU General Public License v3, a BSD-Style license, or the
 original HIDAPI license as outlined in the LICENSE.txt,
 LICENSE-gpl3.txt, LICENSE-bsd.txt, and LICENSE-orig.txt
 files located at the root of the source distribution.
 These files may also be found in the public source
 code repository located at:
        https://github.com/libusb/hidapi .
********************************************************/
#include "SDL_internal.h"

#ifndef HIDAPI_DESCRIPTOR_RECONSTRUCT_H__
#define HIDAPI_DESCRIPTOR_RECONSTRUCT_H__

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
/* Do not warn about wcsncpy usage.
   https://docs.microsoft.com/cpp/c-runtime-library/security-features-in-the-crt */
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "hidapi_winapi.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4200)
#pragma warning(disable: 4201)
#pragma warning(disable: 4214)
#endif

#include <windows.h>

#include "hidapi_hidsdi.h"
/*#include <assert.h>*/

#define NUM_OF_HIDP_REPORT_TYPES 3

typedef enum rd_items_ {
	rd_main_input               = 0x80, /* 1000 00 nn */
	rd_main_output              = 0x90, /* 1001 00 nn */
	rd_main_feature             = 0xB0, /* 1011 00 nn */
	rd_main_collection          = 0xA0, /* 1010 00 nn */
	rd_main_collection_end      = 0xC0, /* 1100 00 nn */
	rd_global_usage_page        = 0x04, /* 0000 01 nn */
	rd_global_logical_minimum   = 0x14, /* 0001 01 nn */
	rd_global_logical_maximum   = 0x24, /* 0010 01 nn */
	rd_global_physical_minimum  = 0x34, /* 0011 01 nn */
	rd_global_physical_maximum  = 0x44, /* 0100 01 nn */
	rd_global_unit_exponent     = 0x54, /* 0101 01 nn */
	rd_global_unit              = 0x64, /* 0110 01 nn */
	rd_global_report_size       = 0x74, /* 0111 01 nn */
	rd_global_report_id         = 0x84, /* 1000 01 nn */
	rd_global_report_count      = 0x94, /* 1001 01 nn */
	rd_global_push              = 0xA4, /* 1010 01 nn */
	rd_global_pop               = 0xB4, /* 1011 01 nn */
	rd_local_usage              = 0x08, /* 0000 10 nn */
	rd_local_usage_minimum      = 0x18, /* 0001 10 nn */
	rd_local_usage_maximum      = 0x28, /* 0010 10 nn */
	rd_local_designator_index   = 0x38, /* 0011 10 nn */
	rd_local_designator_minimum = 0x48, /* 0100 10 nn */
	rd_local_designator_maximum = 0x58, /* 0101 10 nn */
	rd_local_string             = 0x78, /* 0111 10 nn */
	rd_local_string_minimum     = 0x88, /* 1000 10 nn */
	rd_local_string_maximum     = 0x98, /* 1001 10 nn */
	rd_local_delimiter          = 0xA8  /* 1010 10 nn */
} rd_items;

typedef enum rd_main_items_ {
	rd_input = HidP_Input,
	rd_output = HidP_Output,
	rd_feature = HidP_Feature,
	rd_collection,
	rd_collection_end,
	rd_delimiter_open,
	rd_delimiter_usage,
	rd_delimiter_close,
} rd_main_items;

typedef struct rd_bit_range_ {
	int FirstBit;
	int LastBit;
} rd_bit_range;

typedef enum rd_item_node_type_ {
	rd_item_node_cap,
	rd_item_node_padding,
	rd_item_node_collection,
} rd_node_type;

struct rd_main_item_node {
	int FirstBit; /* Position of first bit in report (counting from 0) */
	int LastBit; /* Position of last bit in report (counting from 0) */
	rd_node_type TypeOfNode; /* Information if caps index refers to the array of button caps, value caps,
	                            or if the node is just a padding element to fill unused bit positions.
	                            The node can also be a collection node without any bits in the report. */
	int CapsIndex; /* Index in the array of caps */
	int CollectionIndex; /* Index in the array of link collections */
	rd_main_items MainItemType; /* Input, Output, Feature, Collection or Collection End */
	unsigned char ReportID;
	struct rd_main_item_node* next;
};

typedef struct hid_pp_caps_info_ {
	USHORT FirstCap;
	USHORT NumberOfCaps; // Includes empty caps after LastCap
	USHORT LastCap;
	USHORT ReportByteLength;
} hid_pp_caps_info, *phid_pp_caps_info;

typedef struct hid_pp_link_collection_node_ {
	USAGE  LinkUsage;
	USAGE  LinkUsagePage;
	USHORT Parent;
	USHORT NumberOfChildren;
	USHORT NextSibling;
	USHORT FirstChild;
	ULONG  CollectionType : 8;
	ULONG  IsAlias : 1;
	ULONG  Reserved : 23;
	// Same as the public API structure HIDP_LINK_COLLECTION_NODE, but without PVOID UserContext at the end
} hid_pp_link_collection_node, *phid_pp_link_collection_node;

// Note: This is risk-reduction-measure for this specific struct, as it has ULONG bit-field.
//       Although very unlikely, it might still be possible that the compiler creates a memory layout that is
//       not binary compatile.
//       Other structs are not checked at the time of writing.
//static_assert(sizeof(struct hid_pp_link_collection_node_) == 16,
//    "Size of struct hid_pp_link_collection_node_ not as expected. This might break binary compatibility");
SDL_COMPILE_TIME_ASSERT(hid_pp_link_collection_node_, sizeof(struct hid_pp_link_collection_node_) == 16);

typedef struct hidp_unknown_token_ {
	UCHAR Token; /* Specifies the one-byte prefix of a global item. */
	UCHAR Reserved[3];
	ULONG BitField; /* Specifies the data part of the global item. */
} hidp_unknown_token, * phidp_unknown_token;

typedef struct hid_pp_cap_ {
	USAGE   UsagePage;
	UCHAR   ReportID;
	UCHAR   BitPosition;
	USHORT  ReportSize; // WIN32 term for this is BitSize
	USHORT  ReportCount;
	USHORT  BytePosition;
	USHORT  BitCount;
	ULONG   BitField;
	USHORT  NextBytePosition;
	USHORT  LinkCollection;
	USAGE   LinkUsagePage;
	USAGE   LinkUsage;

	// Start of 8 Flags in one byte
	BOOLEAN IsMultipleItemsForArray:1;

	BOOLEAN IsPadding:1;
	BOOLEAN IsButtonCap:1;
	BOOLEAN IsAbsolute:1;
	BOOLEAN IsRange:1;
	BOOLEAN IsAlias:1; // IsAlias is set to TRUE in the first n-1 capability structures added to the capability array. IsAlias set to FALSE in the nth capability structure.
	BOOLEAN IsStringRange:1;
	BOOLEAN IsDesignatorRange:1;
	// End of 8 Flags in one byte
	BOOLEAN Reserved1[3];

	hidp_unknown_token UnknownTokens[4]; // 4 x 8 Byte

	union {
		struct {
			USAGE  UsageMin;
			USAGE  UsageMax;
			USHORT StringMin;
			USHORT StringMax;
			USHORT DesignatorMin;
			USHORT DesignatorMax;
			USHORT DataIndexMin;
			USHORT DataIndexMax;
		} Range;
		struct {
			USAGE  Usage;
			USAGE  Reserved1;
			USHORT StringIndex;
			USHORT Reserved2;
			USHORT DesignatorIndex;
			USHORT Reserved3;
			USHORT DataIndex;
			USHORT Reserved4;
		} NotRange;
	};
	union {
		struct {
			LONG    LogicalMin;
			LONG    LogicalMax;
		} Button;
		struct {
			BOOLEAN HasNull;
			UCHAR   Reserved4[3];
			LONG    LogicalMin;
			LONG    LogicalMax;
			LONG    PhysicalMin;
			LONG    PhysicalMax;
		} NotButton;
	};
	ULONG   Units;
	ULONG   UnitsExp;

} hid_pp_cap, *phid_pp_cap;

typedef struct hidp_preparsed_data_ {
	UCHAR MagicKey[8];
	USAGE Usage;
	USAGE UsagePage;
	USHORT Reserved[2];

	// CAPS structure for Input, Output and Feature
	hid_pp_caps_info caps_info[3];

	USHORT FirstByteOfLinkCollectionArray;
	USHORT NumberLinkCollectionNodes;

#ifndef _MSC_VER
	// MINGW fails with: Flexible array member in union not supported
	// Solution: https://gcc.gnu.org/onlinedocs/gcc/Zero-Length.html
	union {
#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
		hid_pp_cap caps[0];
		hid_pp_link_collection_node LinkCollectionArray[0];
#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic pop
#endif
	};
#else
	union {
		hid_pp_cap caps[];
		hid_pp_link_collection_node LinkCollectionArray[];
	};
#endif

} hidp_preparsed_data;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif
