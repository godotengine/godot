## pp_data_dump.exe for Windows


pp_data_dump.exe is a small command line tool for Windows, which dumps the content of the [Preparsed Data](https://learn.microsoft.com/en-us/windows-hardware/drivers/hid/preparsed-data) structure, provided by the Windows HID subsystem, into a file.

The generated file is in a text format, which is readable for human, as by the hid_report_reconstructor_test.exe unit test executable of the HIDAPI project. This unit test allows it to test the HIDAPIs report descriptor reconstructor - offline, without the hardware device connected.

pp_data_dump.exe has no arguments, just connect you HID device and execute pp_data_dump.exe. It will generate one file with the name
```
<vendor_id>_<product_id>_<usage>_<usage_table>.pp_data
```
for each top-level collection, of each connected HID device.


## File content

The content of such a .pp_data file looks like the struct, which HIDAPI uses internally to represent the Preparsed Data.

*NOTE:
Windows parses HID report descriptors into opaque `_HIDP_PREPARSED_DATA` objects.
The internal structure of `_HIDP_PREPARSED_DATA` is reserved for internal system use.\
Microsoft doesn't document this structure. [hid_preparsed_data.cc](https://chromium.googlesource.com/chromium/src/+/73fdaaf605bb60caf34d5f30bb84a417688aa528/services/device/hid/hid_preparsed_data.cc) is taken as a reference for its parsing.*

```
# HIDAPI device info struct:
dev->vendor_id           = 0x046D
dev->product_id          = 0xB010
dev->manufacturer_string = "Logitech"
dev->product_string      = "Logitech Bluetooth Wireless Mouse"
dev->release_number      = 0x0000
dev->interface_number    = -1
dev->usage               = 0x0001
dev->usage_page          = 0x000C
dev->path                = "\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002046d_pid&b010&col02#8&1cf1c1b9&3&0001#{4d1e55b2-f16f-11cf-88cb-001111000030}"

# Preparsed Data struct:
pp_data->MagicKey                             = 0x48696450204B4452
pp_data->Usage                                = 0x0001
pp_data->UsagePage                            = 0x000C
pp_data->Reserved                             = 0x00000000
# Input caps_info struct:
pp_data->caps_info[0]->FirstCap           = 0
pp_data->caps_info[0]->LastCap            = 1
pp_data->caps_info[0]->NumberOfCaps       = 1
pp_data->caps_info[0]->ReportByteLength   = 2
# Output caps_info struct:
pp_data->caps_info[1]->FirstCap           = 1
pp_data->caps_info[1]->LastCap            = 1
pp_data->caps_info[1]->NumberOfCaps       = 0
pp_data->caps_info[1]->ReportByteLength   = 0
# Feature caps_info struct:
pp_data->caps_info[2]->FirstCap           = 1
pp_data->caps_info[2]->LastCap            = 1
pp_data->caps_info[2]->NumberOfCaps       = 0
pp_data->caps_info[2]->ReportByteLength   = 0
# LinkCollectionArray Offset & Size:
pp_data->FirstByteOfLinkCollectionArray       = 0x0068
pp_data->NumberLinkCollectionNodes            = 1
# Input hid_pp_cap struct:
pp_data->cap[0]->UsagePage                    = 0x0006
pp_data->cap[0]->ReportID                     = 0x03
pp_data->cap[0]->BitPosition                  = 0
pp_data->cap[0]->BitSize                      = 8
pp_data->cap[0]->ReportCount                  = 1
pp_data->cap[0]->BytePosition                 = 0x0001
pp_data->cap[0]->BitCount                     = 8
pp_data->cap[0]->BitField                     = 0x02
pp_data->cap[0]->NextBytePosition             = 0x0002
pp_data->cap[0]->LinkCollection               = 0x0000
pp_data->cap[0]->LinkUsagePage                = 0x000C
pp_data->cap[0]->LinkUsage                    = 0x0001
pp_data->cap[0]->IsMultipleItemsForArray      = 0
pp_data->cap[0]->IsButtonCap                  = 0
pp_data->cap[0]->IsPadding                    = 0
pp_data->cap[0]->IsAbsolute                   = 1
pp_data->cap[0]->IsRange                      = 0
pp_data->cap[0]->IsAlias                      = 0
pp_data->cap[0]->IsStringRange                = 0
pp_data->cap[0]->IsDesignatorRange            = 0
pp_data->cap[0]->Reserved1                    = 0x000000
pp_data->cap[0]->pp_cap->UnknownTokens[0].Token    = 0x00
pp_data->cap[0]->pp_cap->UnknownTokens[0].Reserved = 0x000000
pp_data->cap[0]->pp_cap->UnknownTokens[0].BitField = 0x00000000
pp_data->cap[0]->pp_cap->UnknownTokens[1].Token    = 0x00
pp_data->cap[0]->pp_cap->UnknownTokens[1].Reserved = 0x000000
pp_data->cap[0]->pp_cap->UnknownTokens[1].BitField = 0x00000000
pp_data->cap[0]->pp_cap->UnknownTokens[2].Token    = 0x00
pp_data->cap[0]->pp_cap->UnknownTokens[2].Reserved = 0x000000
pp_data->cap[0]->pp_cap->UnknownTokens[2].BitField = 0x00000000
pp_data->cap[0]->pp_cap->UnknownTokens[3].Token    = 0x00
pp_data->cap[0]->pp_cap->UnknownTokens[3].Reserved = 0x000000
pp_data->cap[0]->pp_cap->UnknownTokens[3].BitField = 0x00000000
pp_data->cap[0]->NotRange.Usage                        = 0x0020
pp_data->cap[0]->NotRange.Reserved1                    = 0x0020
pp_data->cap[0]->NotRange.StringIndex                  = 0
pp_data->cap[0]->NotRange.Reserved2                    = 0
pp_data->cap[0]->NotRange.DesignatorIndex              = 0
pp_data->cap[0]->NotRange.Reserved3                    = 0
pp_data->cap[0]->NotRange.DataIndex                    = 0
pp_data->cap[0]->NotRange.Reserved4                    = 0
pp_data->cap[0]->NotButton.HasNull                   = 0
pp_data->cap[0]->NotButton.Reserved4                 = 0x000000
pp_data->cap[0]->NotButton.LogicalMin                = 0
pp_data->cap[0]->NotButton.LogicalMax                = 100
pp_data->cap[0]->NotButton.PhysicalMin               = 0
pp_data->cap[0]->NotButton.PhysicalMax               = 0
pp_data->cap[0]->Units                    = 0
pp_data->cap[0]->UnitsExp                 = 0

# Output hid_pp_cap struct:
# Feature hid_pp_cap struct:
# Link Collections:
pp_data->LinkCollectionArray[0]->LinkUsage          = 0x0001
pp_data->LinkCollectionArray[0]->LinkUsagePage      = 0x000C
pp_data->LinkCollectionArray[0]->Parent             = 0
pp_data->LinkCollectionArray[0]->NumberOfChildren   = 0
pp_data->LinkCollectionArray[0]->NextSibling        = 0
pp_data->LinkCollectionArray[0]->FirstChild         = 0
pp_data->LinkCollectionArray[0]->CollectionType     = 1
pp_data->LinkCollectionArray[0]->IsAlias            = 0
pp_data->LinkCollectionArray[0]->Reserved           = 0x00000000
```