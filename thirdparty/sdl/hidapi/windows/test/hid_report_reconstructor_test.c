#if defined(__MINGW32__)
	// Needed for %zu
	#define __USE_MINGW_ANSI_STDIO 1
#endif

#include "../hidapi_descriptor_reconstruct.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static hidp_preparsed_data * alloc_preparsed_data_from_file(char* filename)
{
	FILE* file;
	errno_t err = fopen_s(&file, filename, "r");

	if (err != 0) {
		fprintf(stderr, "ERROR: Couldn't open file '%s' for reading: %s\n", filename, strerror(err));
		return NULL;
	}

	char line[256];

	{
		unsigned short vendor_id = 0;
		unsigned short product_id = 0;
		unsigned short usage = 0;
		unsigned short usage_page = 0;
		unsigned short release_number = 0;
		int interface_number = -1;
		BOOLEAN header_read_success = FALSE;
		char manufacturer_string[128];
		manufacturer_string[0] = '\0';
		char product_string[128];
		product_string[0] = '\0';
		// char path[128];
		// path[0] = '\0';

		while (fgets(line, sizeof(line), file) != NULL) {
			if (line[0] == '\r' || line[0] == '\n') {
				line[0] = '\0';
			}
			if (line[0] == '\0') {
				// read the 'metadata' only until the first empty line
				header_read_success = TRUE;
				break;
			}
			if (sscanf(line, "dev->vendor_id           = 0x%04hX\n", &vendor_id)) continue;
			if (sscanf(line, "dev->product_id          = 0x%04hX\n", &product_id)) continue;
			if (sscanf(line, "dev->usage_page          = 0x%04hX\n", &usage_page)) continue;
			if (sscanf(line, "dev->usage               = 0x%04hX\n", &usage)) continue;
			if (sscanf(line, "dev->manufacturer_string = \"%127[^\"\n]", manufacturer_string)) continue;
			if (sscanf(line, "dev->product_string      = \"%127[^\"\n]", product_string)) continue;
			if (sscanf(line, "dev->release_number      = 0x%04hX\n", &release_number)) continue;
			if (sscanf(line, "dev->interface_number    = %d\n", &interface_number)) continue;
			// if (sscanf(line, "dev->path                = \"%127[^\"]\n", path)) continue;
		}
		if (!header_read_success) {
			fprintf(stderr, "ERROR: Couldn't read PP Data header (missing newline)\n");
			fclose(file);
			return  NULL;
		}
		printf("'Virtual' Device Read: %04hx %04hx\n", vendor_id, product_id);
		if (manufacturer_string[0] != '\0') {
			printf("  Manufacturer: %s\n", manufacturer_string);
		}
		if (product_string[0] != '\0') {
			printf("  Product:      %s\n", product_string);
		}
		printf("  Release:      %hx\n", release_number);
		printf("  Interface:    %d\n", interface_number);
		printf("  Usage (page): 0x%hx (0x%hx)\n", usage, usage_page);
	}

	hidp_preparsed_data static_pp_data;
	memset(&static_pp_data, 0, sizeof(static_pp_data));
	hidp_preparsed_data *pp_data = &static_pp_data;

	unsigned int rt_idx;
	unsigned int caps_idx;
	unsigned int token_idx;
	unsigned int coll_idx;
	USAGE temp_usage;
	BOOLEAN temp_boolean[3];
	UCHAR temp_uchar[3];
	USHORT temp_ushort;
	ULONG temp_ulong;
	LONG temp_long;

	USHORT FirstByteOfLinkCollectionArray = 0;
	USHORT NumberLinkCollectionNodes = 0;

	while (fgets(line, sizeof(line), file) != NULL) {
		if (line[0] == '#')
			continue;

		if (FirstByteOfLinkCollectionArray != 0 && NumberLinkCollectionNodes != 0) {
			size_t size_of_preparsed_data = offsetof(hidp_preparsed_data, caps) + FirstByteOfLinkCollectionArray + (NumberLinkCollectionNodes * sizeof(hid_pp_link_collection_node));
			pp_data->FirstByteOfLinkCollectionArray = FirstByteOfLinkCollectionArray;
			pp_data->NumberLinkCollectionNodes = NumberLinkCollectionNodes;
			FirstByteOfLinkCollectionArray = 0;
			NumberLinkCollectionNodes = 0;
			pp_data = malloc(size_of_preparsed_data);
			memcpy(pp_data, &static_pp_data, sizeof(static_pp_data));
		}

		if (sscanf(line, "pp_data->MagicKey                             = 0x%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX\n", &pp_data->MagicKey[0], &pp_data->MagicKey[1], &pp_data->MagicKey[2], &pp_data->MagicKey[3], &pp_data->MagicKey[4], &pp_data->MagicKey[5], &pp_data->MagicKey[6], &pp_data->MagicKey[7])) continue;
		if (sscanf(line, "pp_data->Usage                                = 0x%04hX\n", &pp_data->Usage)) continue;
		if (sscanf(line, "pp_data->UsagePage                            = 0x%04hX\n", &pp_data->UsagePage)) continue;
		if (sscanf(line, "pp_data->Reserved                             = 0x%04hX%04hX\n", &pp_data->Reserved[0], &pp_data->Reserved[1])) continue;

		if (sscanf(line, "pp_data->caps_info[%u]", &rt_idx) == 1) {
			const size_t caps_info_count = sizeof(pp_data->caps_info) / sizeof(pp_data->caps_info[0]);
			if (rt_idx >= caps_info_count) {
				fprintf(stderr, "Broken pp_data file, pp_data->caps_info[<idx>] can have at most %zu elements, accessing %ud, (%s)", caps_info_count, rt_idx, line);
				continue;
			}
			if (sscanf(line, "pp_data->caps_info[%u]->FirstCap           = %hu\n", &rt_idx, &temp_ushort) == 2) {
				pp_data->caps_info[rt_idx].FirstCap = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->caps_info[%u]->LastCap            = %hu\n", &rt_idx, &temp_ushort) == 2) {
				pp_data->caps_info[rt_idx].LastCap = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->caps_info[%u]->NumberOfCaps       = %hu\n", &rt_idx, &temp_ushort) == 2) {
				pp_data->caps_info[rt_idx].NumberOfCaps = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->caps_info[%u]->ReportByteLength   = %hu\n", &rt_idx, &temp_ushort) == 2) {
				pp_data->caps_info[rt_idx].ReportByteLength = temp_ushort;
				continue;
			}
			fprintf(stderr, "Ignoring unimplemented caps_info field: %s", line);
			continue;
		}

		if (sscanf(line, "pp_data->FirstByteOfLinkCollectionArray       = 0x%04hX\n", &FirstByteOfLinkCollectionArray)) {
			continue;
		}
		if (sscanf(line, "pp_data->NumberLinkCollectionNodes            = %hu\n", &NumberLinkCollectionNodes)) {
			continue;
		}

		if (sscanf(line, "pp_data->cap[%u]", &caps_idx) == 1) {
			if (pp_data->FirstByteOfLinkCollectionArray == 0) {
				fprintf(stderr, "Error reading pp_data file (%s): FirstByteOfLinkCollectionArray is 0 or not reported yet\n", line);
				continue;
			}
			if ((caps_idx + 1) * sizeof(hid_pp_cap) > pp_data->FirstByteOfLinkCollectionArray) {
				fprintf(stderr, "Error reading pp_data file (%s): the caps index (%u) is out of pp_data bytes boundary (%hu vs %hu)\n", line, caps_idx, (unsigned short) ((caps_idx + 1) * sizeof(hid_pp_cap)), pp_data->FirstByteOfLinkCollectionArray);
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->UsagePage                    = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].UsagePage = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->ReportID                     = 0x%02hhX\n", &caps_idx, &temp_uchar[0]) == 2) {
				pp_data->caps[caps_idx].ReportID = temp_uchar[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->BitPosition                  = %hhu\n", &caps_idx, &temp_uchar[0]) == 2) {
				pp_data->caps[caps_idx].BitPosition = temp_uchar[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->BitSize                      = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].ReportSize = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->ReportCount                  = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].ReportCount = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->BytePosition                 = 0x%04hX\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].BytePosition = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->BitCount                     = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].BitCount = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->BitField                     = 0x%02lX\n", &caps_idx, &temp_ulong) == 2) {
				pp_data->caps[caps_idx].BitField = temp_ulong;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NextBytePosition             = 0x%04hX\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NextBytePosition = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->LinkCollection               = 0x%04hX\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].LinkCollection = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->LinkUsagePage                = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].LinkUsagePage = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->LinkUsage                    = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].LinkUsage = temp_usage;
				continue;
			}

			// 8 Flags in one byte
			if (sscanf(line, "pp_data->cap[%u]->IsMultipleItemsForArray      = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsMultipleItemsForArray = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsButtonCap                  = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsButtonCap = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsPadding                    = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsPadding = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsAbsolute                   = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsAbsolute = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsRange                      = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsRange = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsAlias                      = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsAlias = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsStringRange                = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsStringRange = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->IsDesignatorRange            = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].IsDesignatorRange = temp_boolean[0];
				continue;
			}

			if (sscanf(line, "pp_data->cap[%u]->Reserved1                    = 0x%hhu%hhu%hhu\n", &caps_idx, &temp_uchar[0], &temp_uchar[1], &temp_uchar[2]) == 4) {
				pp_data->caps[caps_idx].Reserved1[0] = temp_uchar[0];
				pp_data->caps[caps_idx].Reserved1[1] = temp_uchar[1];
				pp_data->caps[caps_idx].Reserved1[2] = temp_uchar[2];
				continue;
			}

			if (sscanf(line, "pp_data->cap[%u]->pp_cap->UnknownTokens[%u]", &caps_idx, &token_idx) == 2) {
				const size_t unknown_tokens_count = sizeof(pp_data->caps[0].UnknownTokens) / sizeof(pp_data->caps[0].UnknownTokens[0]);
				if (token_idx >= unknown_tokens_count) {
					fprintf(stderr, "Broken pp_data file, pp_data->caps[<idx>].UnknownTokens[<idx>] can have at most %zu elements, accessing %ud, (%s)", unknown_tokens_count, token_idx, line);
					continue;
				}
				if (sscanf(line, "pp_data->cap[%u]->pp_cap->UnknownTokens[%u].Token    = 0x%02hhX\n", &caps_idx, &token_idx, &temp_uchar[0]) == 3) {
					pp_data->caps[caps_idx].UnknownTokens[token_idx].Token = temp_uchar[0];
					continue;
				}
				if (sscanf(line, "pp_data->cap[%u]->pp_cap->UnknownTokens[%u].Reserved = 0x%02hhX%02hhX%02hhX\n", &caps_idx, &token_idx, &temp_uchar[0], &temp_uchar[1], &temp_uchar[2]) == 5) {
					pp_data->caps[caps_idx].UnknownTokens[token_idx].Reserved[0] = temp_uchar[0];
					pp_data->caps[caps_idx].UnknownTokens[token_idx].Reserved[1] = temp_uchar[1];
					pp_data->caps[caps_idx].UnknownTokens[token_idx].Reserved[2] = temp_uchar[2];
					continue;
				}
				if (sscanf(line, "pp_data->cap[%u]->pp_cap->UnknownTokens[%u].BitField = 0x%08lX\n", &caps_idx, &token_idx, &temp_ulong) == 3) {
					pp_data->caps[caps_idx].UnknownTokens[token_idx].BitField = temp_ulong;
					continue;
				}
				fprintf(stderr, "Ignoring unimplemented pp_data->cap[]->pp_cap->UnknownTokens field: %s", line);
				continue;
			}

			// Range
			if (sscanf(line, "pp_data->cap[%u]->Range.UsageMin                     = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].Range.UsageMin = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.UsageMax                     = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].Range.UsageMax = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.StringMin                    = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].Range.StringMin = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.StringMax                    = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].Range.StringMax = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.DesignatorMin                = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].Range.DesignatorMin = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.DesignatorMax                = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].Range.DesignatorMax = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.DataIndexMin                 = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].Range.DataIndexMin = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Range.DataIndexMax                 = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].Range.DataIndexMax = temp_ushort;
				continue;
			}

			// NotRange
			if (sscanf(line, "pp_data->cap[%u]->NotRange.Usage                        = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].NotRange.Usage = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.Reserved1                    = 0x%04hX\n", &caps_idx, &temp_usage) == 2) {
				pp_data->caps[caps_idx].NotRange.Reserved1 = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.StringIndex                  = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NotRange.StringIndex = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.Reserved2                    = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NotRange.Reserved2 = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.DesignatorIndex              = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NotRange.DesignatorIndex = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.Reserved3                    = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NotRange.Reserved3 = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.DataIndex                    = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NotRange.DataIndex = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotRange.Reserved4                    = %hu\n", &caps_idx, &temp_ushort) == 2) {
				pp_data->caps[caps_idx].NotRange.Reserved4 = temp_ushort;
				continue;
			}

			// Button
			if (sscanf(line, "pp_data->cap[%u]->Button.LogicalMin                   = %ld\n", &caps_idx, &temp_long) == 2) {
				pp_data->caps[caps_idx].Button.LogicalMin = temp_long;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Button.LogicalMax                   = %ld\n", &caps_idx, &temp_long) == 2) {
				pp_data->caps[caps_idx].Button.LogicalMax = temp_long;
				continue;
			}

			// NotButton
			if (sscanf(line, "pp_data->cap[%u]->NotButton.HasNull                   = %hhu\n", &caps_idx, &temp_boolean[0]) == 2) {
				pp_data->caps[caps_idx].NotButton.HasNull = temp_boolean[0];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotButton.Reserved4                 = 0x%02hhX%02hhX%02hhX\n", &caps_idx, &temp_uchar[0], &temp_uchar[1], &temp_uchar[2]) == 4) {
				pp_data->caps[caps_idx].NotButton.Reserved4[0] = temp_uchar[0];
				pp_data->caps[caps_idx].NotButton.Reserved4[1] = temp_uchar[1];
				pp_data->caps[caps_idx].NotButton.Reserved4[2] = temp_uchar[2];
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotButton.LogicalMin                = %ld\n", &caps_idx, &temp_long) == 2) {
				pp_data->caps[caps_idx].NotButton.LogicalMin = temp_long;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotButton.LogicalMax                = %ld\n", &caps_idx, &temp_long) == 2) {
				pp_data->caps[caps_idx].NotButton.LogicalMax = temp_long;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotButton.PhysicalMin               = %ld\n", &caps_idx, &temp_long) == 2) {
				pp_data->caps[caps_idx].NotButton.PhysicalMin = temp_long;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->NotButton.PhysicalMax               = %ld\n", &caps_idx, &temp_long) == 2) {
				pp_data->caps[caps_idx].NotButton.PhysicalMax = temp_long;
				continue;
			}

			if (sscanf(line, "pp_data->cap[%u]->Units                    = %lu\n", &caps_idx, &temp_ulong) == 2) {
				pp_data->caps[caps_idx].Units = temp_ulong;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->UnitsExp                 = %lu\n", &caps_idx, &temp_ulong) == 2) {
				pp_data->caps[caps_idx].UnitsExp = temp_ulong;
				continue;
			}
			if (sscanf(line, "pp_data->cap[%u]->Reserved1                    = 0x%02hhu%02hhu%02hhu\n", &coll_idx, &temp_uchar[0], &temp_uchar[1], &temp_uchar[2]) == 4) {
				pp_data->caps[caps_idx].Reserved1[0] = temp_uchar[0];
				pp_data->caps[caps_idx].Reserved1[1] = temp_uchar[1];
				pp_data->caps[caps_idx].Reserved1[2] = temp_uchar[2];
				continue;
			}
			fprintf(stderr, "Ignoring unimplemented cap field: %s", line);
			continue;
		}

		if (sscanf(line, "pp_data->LinkCollectionArray[%u]", &coll_idx) == 1) {
			if (pp_data->FirstByteOfLinkCollectionArray == 0 || pp_data->NumberLinkCollectionNodes == 0) {
				fprintf(stderr, "Error reading pp_data file (%s): FirstByteOfLinkCollectionArray or NumberLinkCollectionNodes is 0 or not reported yet\n", line);
				continue;
			}
			if (coll_idx >= pp_data->NumberLinkCollectionNodes) {
				fprintf(stderr, "Error reading pp_data file (%s): the LinkCollection index (%u) is out of boundary (%hu)\n", line, coll_idx, pp_data->NumberLinkCollectionNodes);
				continue;
			}
			phid_pp_link_collection_node pcoll = (phid_pp_link_collection_node)(((unsigned char*)&pp_data->caps[0]) + pp_data->FirstByteOfLinkCollectionArray);
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->LinkUsage          = 0x%04hX\n", &coll_idx, &temp_usage) == 2) {
				pcoll[coll_idx].LinkUsage = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->LinkUsagePage      = 0x%04hX\n", &coll_idx, &temp_usage) == 2) {
				pcoll[coll_idx].LinkUsagePage = temp_usage;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->Parent             = %hu\n", &coll_idx, &temp_ushort) == 2) {
				pcoll[coll_idx].Parent = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->NumberOfChildren   = %hu\n", &coll_idx, &temp_ushort) == 2) {
				pcoll[coll_idx].NumberOfChildren = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->NextSibling        = %hu\n", &coll_idx, &temp_ushort) == 2) {
				pcoll[coll_idx].NextSibling = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->FirstChild         = %hu\n", &coll_idx, &temp_ushort) == 2) {
				pcoll[coll_idx].FirstChild = temp_ushort;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->CollectionType     = %lu\n", &coll_idx, &temp_ulong) == 2) {
				pcoll[coll_idx].CollectionType = temp_ulong;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->IsAlias            = %lu\n", &coll_idx, &temp_ulong) == 2) {
				pcoll[coll_idx].IsAlias = temp_ulong;
				continue;
			}
			if (sscanf(line, "pp_data->LinkCollectionArray[%u]->Reserved           = %lu\n", &coll_idx, &temp_ulong) == 2) {
				pcoll[coll_idx].Reserved = temp_ulong;
				continue;
			}
			fprintf(stderr, "Ignoring unimplemented LinkCollectionArray field: %s", line);
			continue;
		}
	}

//end:
	fclose(file);

	if (pp_data == &static_pp_data) {
		return NULL;
	}

	return pp_data;
}

static BOOLEAN read_hex_data_from_text_file(const char *filename, unsigned char *data_out, size_t data_size, size_t *actual_read)
{
	size_t read_index = 0;
	FILE* file = NULL;
	errno_t err = fopen_s(&file, filename, "r");
	if (err != 0) {
		fprintf(stderr, "ERROR: Couldn't open file '%s' for reading: %s\n", filename, strerror(err));
		return FALSE;
	}

	BOOLEAN result = TRUE;
	unsigned int val;
	char buf[16];
	while (fscanf(file, "%15s", buf) == 1) {
		if (sscanf(buf, "0x%X", &val) != 1) {
			fprintf(stderr, "Invalid HEX text ('%s') file, got %s\n", filename, buf);
			result = FALSE;
			goto end;
		}

		if (read_index >= data_size) {
			fprintf(stderr, "Buffer for file read is too small. Got only %zu bytes to read '%s'\n", data_size, filename);
			result = FALSE;
			goto end;
		}

		if (val > (unsigned char)-1) {
			fprintf(stderr, "Invalid HEX text ('%s') file, got a value of: %u\n", filename, val);
			result = FALSE;
			goto end;
		}

		data_out[read_index] = (unsigned char) val;

		read_index++;
	}

	if (!feof(file)) {
		fprintf(stderr, "Invalid HEX text ('%s') file - failed to read all values\n", filename);
		result = FALSE;
		goto end;
	}

	*actual_read = read_index;

end:
	fclose(file);
	return result;
}


int main(int argc, char* argv[])
{
	if (argc != 3) {
		fprintf(stderr, "Expected 2 arguments for the test ('<>.pp_data' and '<>_expected.rpt_desc'), got: %d\n", argc - 1);
		return EXIT_FAILURE;
	}

	printf("Checking: '%s' / '%s'\n", argv[1], argv[2]);

	hidp_preparsed_data *pp_data = alloc_preparsed_data_from_file(argv[1]);
	if (pp_data == NULL) {
		return EXIT_FAILURE;
	}

	int result = EXIT_SUCCESS;

	unsigned char report_descriptor[HID_API_MAX_REPORT_DESCRIPTOR_SIZE];

	int res = hid_winapi_descriptor_reconstruct_pp_data(pp_data, report_descriptor, sizeof(report_descriptor));

	if (res < 0) {
		result = EXIT_FAILURE;
		fprintf(stderr, "Failed to reconstruct descriptor");
		goto end;
	}
	size_t report_descriptor_size = (size_t) res;

	unsigned char expected_report_descriptor[HID_API_MAX_REPORT_DESCRIPTOR_SIZE];
	size_t expected_report_descriptor_size = 0;
	if (!read_hex_data_from_text_file(argv[2], expected_report_descriptor, sizeof(expected_report_descriptor), &expected_report_descriptor_size)) {
		result = EXIT_FAILURE;
		goto end;
	}

	if (report_descriptor_size == expected_report_descriptor_size) {
		if (memcmp(report_descriptor, expected_report_descriptor, report_descriptor_size) == 0) {
			printf("Reconstructed Report Descriptor matches the expected descriptor\n");
			goto end;
		}
		else {
			result = EXIT_FAILURE;
			fprintf(stderr, "Reconstructed Report Descriptor has different content than expected\n");
		}
	}
	else {
		result = EXIT_FAILURE;
		fprintf(stderr, "Reconstructed Report Descriptor has different size: %zu when expected %zu\n", report_descriptor_size, expected_report_descriptor_size);
	}

	printf("  Reconstructed Report Descriptor:\n");
	for (int i = 0; i < res; i++) {
		printf("0x%02X, ", report_descriptor[i]);
	}
	printf("\n");
	fflush(stdout);

end:
	free(pp_data);
	return result;
}
