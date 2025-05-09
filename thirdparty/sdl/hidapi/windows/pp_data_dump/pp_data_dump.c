#if defined(__MINGW32__)
	// Needed for %hh
	#define __USE_MINGW_ANSI_STDIO 1
#endif

#include <hid.c>
#include <../windows/hidapi_descriptor_reconstruct.h>

#include <hidapi.h>

void dump_hid_pp_cap(FILE* file, phid_pp_cap pp_cap, unsigned int cap_idx) {
	fprintf(file, "pp_data->cap[%u]->UsagePage                    = 0x%04hX\n", cap_idx, pp_cap->UsagePage);
	fprintf(file, "pp_data->cap[%u]->ReportID                     = 0x%02hhX\n", cap_idx, pp_cap->ReportID);
	fprintf(file, "pp_data->cap[%u]->BitPosition                  = %hhu\n", cap_idx, pp_cap->BitPosition);
	fprintf(file, "pp_data->cap[%u]->BitSize                      = %hu\n", cap_idx, pp_cap->ReportSize);
	fprintf(file, "pp_data->cap[%u]->ReportCount                  = %hu\n", cap_idx, pp_cap->ReportCount);
	fprintf(file, "pp_data->cap[%u]->BytePosition                 = 0x%04hX\n", cap_idx, pp_cap->BytePosition);
	fprintf(file, "pp_data->cap[%u]->BitCount                     = %hu\n", cap_idx, pp_cap->BitCount);
	fprintf(file, "pp_data->cap[%u]->BitField                     = 0x%02lX\n", cap_idx, pp_cap->BitField);
	fprintf(file, "pp_data->cap[%u]->NextBytePosition             = 0x%04hX\n", cap_idx, pp_cap->NextBytePosition);
	fprintf(file, "pp_data->cap[%u]->LinkCollection               = 0x%04hX\n", cap_idx, pp_cap->LinkCollection);
	fprintf(file, "pp_data->cap[%u]->LinkUsagePage                = 0x%04hX\n", cap_idx, pp_cap->LinkUsagePage);
	fprintf(file, "pp_data->cap[%u]->LinkUsage                    = 0x%04hX\n", cap_idx, pp_cap->LinkUsage);

	// 8 Flags in one byte
	fprintf(file, "pp_data->cap[%u]->IsMultipleItemsForArray      = %hhu\n", cap_idx, pp_cap->IsMultipleItemsForArray);
	fprintf(file, "pp_data->cap[%u]->IsButtonCap                  = %hhu\n", cap_idx, pp_cap->IsButtonCap);
	fprintf(file, "pp_data->cap[%u]->IsPadding                    = %hhu\n", cap_idx, pp_cap->IsPadding);
	fprintf(file, "pp_data->cap[%u]->IsAbsolute                   = %hhu\n", cap_idx, pp_cap->IsAbsolute);
	fprintf(file, "pp_data->cap[%u]->IsRange                      = %hhu\n", cap_idx, pp_cap->IsRange);
	fprintf(file, "pp_data->cap[%u]->IsAlias                      = %hhu\n", cap_idx, pp_cap->IsAlias);
	fprintf(file, "pp_data->cap[%u]->IsStringRange                = %hhu\n", cap_idx, pp_cap->IsStringRange);
	fprintf(file, "pp_data->cap[%u]->IsDesignatorRange            = %hhu\n", cap_idx, pp_cap->IsDesignatorRange);

	fprintf(file, "pp_data->cap[%u]->Reserved1                    = 0x%02hhX%02hhX%02hhX\n", cap_idx, pp_cap->Reserved1[0], pp_cap->Reserved1[1], pp_cap->Reserved1[2]);

	for (int token_idx = 0; token_idx < 4; token_idx++) {
		fprintf(file, "pp_data->cap[%u]->pp_cap->UnknownTokens[%d].Token    = 0x%02hhX\n", cap_idx, token_idx, pp_cap->UnknownTokens[token_idx].Token);
		fprintf(file, "pp_data->cap[%u]->pp_cap->UnknownTokens[%d].Reserved = 0x%02hhX%02hhX%02hhX\n", cap_idx, token_idx, pp_cap->UnknownTokens[token_idx].Reserved[0], pp_cap->UnknownTokens[token_idx].Reserved[1], pp_cap->UnknownTokens[token_idx].Reserved[2]);
		fprintf(file, "pp_data->cap[%u]->pp_cap->UnknownTokens[%d].BitField = 0x%08lX\n", cap_idx, token_idx, pp_cap->UnknownTokens[token_idx].BitField);
	}

	if (pp_cap->IsRange) {
		fprintf(file, "pp_data->cap[%u]->Range.UsageMin                     = 0x%04hX\n", cap_idx, pp_cap->Range.UsageMin);
		fprintf(file, "pp_data->cap[%u]->Range.UsageMax                     = 0x%04hX\n", cap_idx, pp_cap->Range.UsageMax);
		fprintf(file, "pp_data->cap[%u]->Range.StringMin                    = %hu\n", cap_idx, pp_cap->Range.StringMin);
		fprintf(file, "pp_data->cap[%u]->Range.StringMax                    = %hu\n", cap_idx, pp_cap->Range.StringMax);
		fprintf(file, "pp_data->cap[%u]->Range.DesignatorMin                = %hu\n", cap_idx, pp_cap->Range.DesignatorMin);
		fprintf(file, "pp_data->cap[%u]->Range.DesignatorMax                = %hu\n", cap_idx, pp_cap->Range.DesignatorMax);
		fprintf(file, "pp_data->cap[%u]->Range.DataIndexMin                 = %hu\n", cap_idx, pp_cap->Range.DataIndexMin);
		fprintf(file, "pp_data->cap[%u]->Range.DataIndexMax                 = %hu\n", cap_idx, pp_cap->Range.DataIndexMax);
	}
	else {
		fprintf(file, "pp_data->cap[%u]->NotRange.Usage                        = 0x%04hX\n", cap_idx, pp_cap->NotRange.Usage);
		fprintf(file, "pp_data->cap[%u]->NotRange.Reserved1                    = 0x%04hX\n", cap_idx, pp_cap->NotRange.Reserved1);
		fprintf(file, "pp_data->cap[%u]->NotRange.StringIndex                  = %hu\n", cap_idx, pp_cap->NotRange.StringIndex);
		fprintf(file, "pp_data->cap[%u]->NotRange.Reserved2                    = %hu\n", cap_idx, pp_cap->NotRange.Reserved2);
		fprintf(file, "pp_data->cap[%u]->NotRange.DesignatorIndex              = %hu\n", cap_idx, pp_cap->NotRange.DesignatorIndex);
		fprintf(file, "pp_data->cap[%u]->NotRange.Reserved3                    = %hu\n", cap_idx, pp_cap->NotRange.Reserved3);
		fprintf(file, "pp_data->cap[%u]->NotRange.DataIndex                    = %hu\n", cap_idx, pp_cap->NotRange.DataIndex);
		fprintf(file, "pp_data->cap[%u]->NotRange.Reserved4                    = %hu\n", cap_idx, pp_cap->NotRange.Reserved4);
	}

	if (pp_cap->IsButtonCap) {
		fprintf(file, "pp_data->cap[%u]->Button.LogicalMin                   = %ld\n", cap_idx, pp_cap->Button.LogicalMin);
		fprintf(file, "pp_data->cap[%u]->Button.LogicalMax                   = %ld\n", cap_idx, pp_cap->Button.LogicalMax);
	}
	else
	{
		fprintf(file, "pp_data->cap[%u]->NotButton.HasNull                   = %hhu\n", cap_idx, pp_cap->NotButton.HasNull);
		fprintf(file, "pp_data->cap[%u]->NotButton.Reserved4                 = 0x%02hhX%02hhX%02hhX\n", cap_idx, pp_cap->NotButton.Reserved4[0], pp_cap->NotButton.Reserved4[1], pp_cap->NotButton.Reserved4[2]);
		fprintf(file, "pp_data->cap[%u]->NotButton.LogicalMin                = %ld\n", cap_idx, pp_cap->NotButton.LogicalMin);
		fprintf(file, "pp_data->cap[%u]->NotButton.LogicalMax                = %ld\n", cap_idx, pp_cap->NotButton.LogicalMax);
		fprintf(file, "pp_data->cap[%u]->NotButton.PhysicalMin               = %ld\n", cap_idx, pp_cap->NotButton.PhysicalMin);
		fprintf(file, "pp_data->cap[%u]->NotButton.PhysicalMax               = %ld\n", cap_idx, pp_cap->NotButton.PhysicalMax);
	};
	fprintf(file, "pp_data->cap[%u]->Units                    = %lu\n", cap_idx, pp_cap->Units);
	fprintf(file, "pp_data->cap[%u]->UnitsExp                 = %lu\n", cap_idx, pp_cap->UnitsExp);
}

void dump_hidp_link_collection_node(FILE* file, phid_pp_link_collection_node pcoll, unsigned int coll_idx) {
	fprintf(file, "pp_data->LinkCollectionArray[%u]->LinkUsage          = 0x%04hX\n", coll_idx, pcoll->LinkUsage);
	fprintf(file, "pp_data->LinkCollectionArray[%u]->LinkUsagePage      = 0x%04hX\n", coll_idx, pcoll->LinkUsagePage);
	fprintf(file, "pp_data->LinkCollectionArray[%u]->Parent             = %hu\n", coll_idx, pcoll->Parent);
	fprintf(file, "pp_data->LinkCollectionArray[%u]->NumberOfChildren   = %hu\n", coll_idx, pcoll->NumberOfChildren);
	fprintf(file, "pp_data->LinkCollectionArray[%u]->NextSibling        = %hu\n", coll_idx, pcoll->NextSibling);
	fprintf(file, "pp_data->LinkCollectionArray[%u]->FirstChild         = %hu\n", coll_idx, pcoll->FirstChild);
	// The compilers are not consistent on ULONG-bit-fields: They lose the unsinged or define them as int.
	// Thus just always cast them to unsinged int, which should be fine, as the biggest bit-field is 28 bit
	fprintf(file, "pp_data->LinkCollectionArray[%u]->CollectionType     = %u\n", coll_idx, (unsigned int)(pcoll->CollectionType));
	fprintf(file, "pp_data->LinkCollectionArray[%u]->IsAlias            = %u\n", coll_idx, (unsigned int)(pcoll->IsAlias));
	fprintf(file, "pp_data->LinkCollectionArray[%u]->Reserved           = 0x%08X\n", coll_idx, (unsigned int)(pcoll->Reserved));
}

int dump_pp_data(FILE* file, hid_device* dev)
{
	BOOL res;
	hidp_preparsed_data* pp_data = NULL;

	res = HidD_GetPreparsedData(dev->device_handle, (PHIDP_PREPARSED_DATA*) &pp_data);
	if (!res) {
		printf("ERROR: HidD_GetPreparsedData failed!");
		return -1;
	}
	else {
		fprintf(file, "pp_data->MagicKey                             = 0x%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX\n", pp_data->MagicKey[0], pp_data->MagicKey[1], pp_data->MagicKey[2], pp_data->MagicKey[3], pp_data->MagicKey[4], pp_data->MagicKey[5], pp_data->MagicKey[6], pp_data->MagicKey[7]);
		fprintf(file, "pp_data->Usage                                = 0x%04hX\n", pp_data->Usage);
		fprintf(file, "pp_data->UsagePage                            = 0x%04hX\n", pp_data->UsagePage);
		fprintf(file, "pp_data->Reserved                             = 0x%04hX%04hX\n", pp_data->Reserved[0], pp_data->Reserved[1]);
		fprintf(file, "# Input caps_info struct:\n");
		fprintf(file, "pp_data->caps_info[0]->FirstCap           = %hu\n", pp_data->caps_info[0].FirstCap);
		fprintf(file, "pp_data->caps_info[0]->LastCap            = %hu\n", pp_data->caps_info[0].LastCap);
		fprintf(file, "pp_data->caps_info[0]->NumberOfCaps       = %hu\n", pp_data->caps_info[0].NumberOfCaps);
		fprintf(file, "pp_data->caps_info[0]->ReportByteLength   = %hu\n", pp_data->caps_info[0].ReportByteLength);
		fprintf(file, "# Output caps_info struct:\n");
		fprintf(file, "pp_data->caps_info[1]->FirstCap           = %hu\n", pp_data->caps_info[1].FirstCap);
		fprintf(file, "pp_data->caps_info[1]->LastCap            = %hu\n", pp_data->caps_info[1].LastCap);
		fprintf(file, "pp_data->caps_info[1]->NumberOfCaps       = %hu\n", pp_data->caps_info[1].NumberOfCaps);
		fprintf(file, "pp_data->caps_info[1]->ReportByteLength   = %hu\n", pp_data->caps_info[1].ReportByteLength);
		fprintf(file, "# Feature caps_info struct:\n");
		fprintf(file, "pp_data->caps_info[2]->FirstCap           = %hu\n", pp_data->caps_info[2].FirstCap);
		fprintf(file, "pp_data->caps_info[2]->LastCap            = %hu\n", pp_data->caps_info[2].LastCap);
		fprintf(file, "pp_data->caps_info[2]->NumberOfCaps       = %hu\n", pp_data->caps_info[2].NumberOfCaps);
		fprintf(file, "pp_data->caps_info[2]->ReportByteLength   = %hu\n", pp_data->caps_info[2].ReportByteLength);
		fprintf(file, "# LinkCollectionArray Offset & Size:\n");
		fprintf(file, "pp_data->FirstByteOfLinkCollectionArray       = 0x%04hX\n", pp_data->FirstByteOfLinkCollectionArray);
		fprintf(file, "pp_data->NumberLinkCollectionNodes            = %hu\n", pp_data->NumberLinkCollectionNodes);


		phid_pp_cap pcap = (phid_pp_cap)(((unsigned char*)pp_data) + offsetof(hidp_preparsed_data, caps));
		fprintf(file, "# Input hid_pp_cap struct:\n");
		for (int caps_idx = pp_data->caps_info[0].FirstCap; caps_idx < pp_data->caps_info[0].LastCap; caps_idx++) {
			dump_hid_pp_cap(file, pcap + caps_idx, caps_idx);
			fprintf(file, "\n");
		}
		fprintf(file, "# Output hid_pp_cap struct:\n");
		for (int caps_idx = pp_data->caps_info[1].FirstCap; caps_idx < pp_data->caps_info[1].LastCap; caps_idx++) {
			dump_hid_pp_cap(file, pcap + caps_idx, caps_idx);
			fprintf(file, "\n");
		}
		fprintf(file, "# Feature hid_pp_cap struct:\n");
		for (int caps_idx = pp_data->caps_info[2].FirstCap; caps_idx < pp_data->caps_info[2].LastCap; caps_idx++) {
			dump_hid_pp_cap(file, pcap + caps_idx, caps_idx);
			fprintf(file, "\n");
		}

		phid_pp_link_collection_node pcoll = (phid_pp_link_collection_node)(((unsigned char*)pcap) + pp_data->FirstByteOfLinkCollectionArray);
		fprintf(file, "# Link Collections:\n");
		for (int coll_idx = 0; coll_idx < pp_data->NumberLinkCollectionNodes; coll_idx++) {
			dump_hidp_link_collection_node(file, pcoll + coll_idx, coll_idx);
		}

		HidD_FreePreparsedData((PHIDP_PREPARSED_DATA) pp_data);
		return 0;
	}
}

int main(int argc, char* argv[])
{
	(void)argc;
	(void)argv;

	#define MAX_STR 255

	struct hid_device_info *devs, *cur_dev;

	printf("pp_data_dump tool. Compiled with hidapi version %s, runtime version %s.\n", HID_API_VERSION_STR, hid_version_str());
	if (hid_version()->major == HID_API_VERSION_MAJOR && hid_version()->minor == HID_API_VERSION_MINOR && hid_version()->patch == HID_API_VERSION_PATCH) {
		printf("Compile-time version matches runtime version of hidapi.\n\n");
	}
	else {
		printf("Compile-time version is different than runtime version of hidapi.\n]n");
	}

	if (hid_init())
		return -1;

	devs = hid_enumerate(0x0, 0x0);
	cur_dev = devs;
	while (cur_dev) {
		printf("Device Found\n  type: %04hx %04hx\n  path: %s\n  serial_number: %ls", cur_dev->vendor_id, cur_dev->product_id, cur_dev->path, cur_dev->serial_number);
		printf("\n");
		printf("  Manufacturer: %ls\n", cur_dev->manufacturer_string);
		printf("  Product:      %ls\n", cur_dev->product_string);
		printf("  Release:      %hX\n", cur_dev->release_number);
		printf("  Interface:    %d\n",  cur_dev->interface_number);
		printf("  Usage (page): %02X (%02X)\n", cur_dev->usage, cur_dev->usage_page);

		hid_device *device = hid_open_path(cur_dev->path);
		if (device) {
			char filename[MAX_STR];
			FILE* file;

			sprintf_s(filename, MAX_STR, "%04X_%04X_%04X_%04X.pp_data", cur_dev->vendor_id, cur_dev->product_id, cur_dev->usage, cur_dev->usage_page);
			errno_t err = fopen_s(&file, filename, "w");
			if (err == 0) {
				fprintf(file, "# HIDAPI device info struct:\n");
				fprintf(file, "dev->vendor_id           = 0x%04hX\n", cur_dev->vendor_id);
				fprintf(file, "dev->product_id          = 0x%04hX\n", cur_dev->product_id);
				fprintf(file, "dev->manufacturer_string = \"%ls\"\n", cur_dev->manufacturer_string);
				fprintf(file, "dev->product_string      = \"%ls\"\n", cur_dev->product_string);
				fprintf(file, "dev->release_number      = 0x%04hX\n", cur_dev->release_number);
				fprintf(file, "dev->interface_number    = %d\n", cur_dev->interface_number);
				fprintf(file, "dev->usage               = 0x%04hX\n", cur_dev->usage);
				fprintf(file, "dev->usage_page          = 0x%04hX\n", cur_dev->usage_page);
				fprintf(file, "dev->path                = \"%s\"\n", cur_dev->path);
				fprintf(file, "\n# Preparsed Data struct:\n");
				int res = dump_pp_data(file, device);

				if (res == 0) {
					printf("Dumped Preparsed Data to %s\n", filename);
				}
				else {
					printf("ERROR: Dump Preparsed Data to %s failed!\n", filename);
				}

				fclose(file);
			}

			hid_close(device);
		}
		else {
			printf("  Device: not available.\n");
		}

		printf("\n");
		cur_dev = cur_dev->next;
	}
	hid_free_enumeration(devs);


	/* Free static HIDAPI objects. */
	hid_exit();

	//system("pause");

	return 0;
}
