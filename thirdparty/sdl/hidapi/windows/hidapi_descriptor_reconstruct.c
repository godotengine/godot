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
#include "hidapi_descriptor_reconstruct.h"

/**
 * @brief References to report descriptor buffer.
 * 
 */
struct rd_buffer {
	unsigned char* buf; /* Pointer to the array which stores the reconstructed descriptor */
	size_t buf_size; /* Size of the buffer in bytes */
	size_t byte_idx; /* Index of the next report byte to write to buf array */
};

/**
 * @brief Function that appends a byte to encoded report descriptor buffer.
 *
 * @param[in]  byte     Single byte to append.
 * @param      rpt_desc Pointer to report descriptor buffer struct.
 */
static void rd_append_byte(unsigned char byte, struct rd_buffer* rpt_desc) {
	if (rpt_desc->byte_idx < rpt_desc->buf_size) {
		rpt_desc->buf[rpt_desc->byte_idx] = byte;
		rpt_desc->byte_idx++;
	}
}

/**
 * @brief Writes a short report descriptor item according USB HID spec 1.11 chapter 6.2.2.2.
 *
 * @param[in]  rd_item  Enumeration identifying type (Main, Global, Local) and function (e.g Usage or Report Count) of the item.
 * @param[in]  data     Data (Size depends on rd_item 0,1,2 or 4bytes).
 * @param      rpt_desc Pointer to report descriptor buffer struct.
 *
 * @return Returns 0 if successful, -1 for error.
 */
static int rd_write_short_item(rd_items rd_item, LONG64 data, struct rd_buffer* rpt_desc) {
	if (rd_item & 0x03) {
		// Invalid input data, last to bits are reserved for data size
		return -1;
	}

	if (rd_item == rd_main_collection_end) {
		// Item without data (1Byte prefix only)
		unsigned char oneBytePrefix = (unsigned char) rd_item + 0x00;
		rd_append_byte(oneBytePrefix, rpt_desc);
	}
	else if ((rd_item == rd_global_logical_minimum) ||
		(rd_item == rd_global_logical_maximum) ||
		(rd_item == rd_global_physical_minimum) ||
		(rd_item == rd_global_physical_maximum)) {
		// Item with signed integer data
		if ((data >= -128) && (data <= 127)) {
			// 1Byte prefix + 1Byte data
			unsigned char oneBytePrefix = (unsigned char) rd_item + 0x01;
			char localData = (char)data;
			rd_append_byte(oneBytePrefix, rpt_desc);
			rd_append_byte(localData & 0xFF, rpt_desc);
		}
		else if ((data >= -32768) && (data <= 32767)) {
			// 1Byte prefix + 2Byte data
			unsigned char oneBytePrefix = (unsigned char) rd_item + 0x02;
			INT16 localData = (INT16)data;
			rd_append_byte(oneBytePrefix, rpt_desc);
			rd_append_byte(localData & 0xFF, rpt_desc);
			rd_append_byte(localData >> 8 & 0xFF, rpt_desc);
		}
		else if ((data >= -2147483648LL) && (data <= 2147483647)) {
			// 1Byte prefix + 4Byte data
			unsigned char oneBytePrefix = (unsigned char) rd_item + 0x03;
			INT32 localData = (INT32)data;
			rd_append_byte(oneBytePrefix, rpt_desc);
			rd_append_byte(localData & 0xFF, rpt_desc);
			rd_append_byte(localData >> 8 & 0xFF, rpt_desc);
			rd_append_byte(localData >> 16 & 0xFF, rpt_desc);
			rd_append_byte(localData >> 24 & 0xFF, rpt_desc);
		}
		else {
			// Data out of 32 bit signed integer range
			return -1;
		}
	}
	else {
		// Item with unsigned integer data
		if ((data >= 0) && (data <= 0xFF)) {
			// 1Byte prefix + 1Byte data
			unsigned char oneBytePrefix = (unsigned char) rd_item + 0x01;
			unsigned char localData = (unsigned char)data;
			rd_append_byte(oneBytePrefix, rpt_desc);
			rd_append_byte(localData & 0xFF, rpt_desc);
		}
		else if ((data >= 0) && (data <= 0xFFFF)) {
			// 1Byte prefix + 2Byte data
			unsigned char oneBytePrefix = (unsigned char) rd_item + 0x02;
			UINT16 localData = (UINT16)data;
			rd_append_byte(oneBytePrefix, rpt_desc);
			rd_append_byte(localData & 0xFF, rpt_desc);
			rd_append_byte(localData >> 8 & 0xFF, rpt_desc);
		}
		else if ((data >= 0) && (data <= 0xFFFFFFFF)) {
			// 1Byte prefix + 4Byte data
			unsigned char oneBytePrefix = (unsigned char) rd_item + 0x03;
			UINT32 localData = (UINT32)data;
			rd_append_byte(oneBytePrefix, rpt_desc);
			rd_append_byte(localData & 0xFF, rpt_desc);
			rd_append_byte(localData >> 8 & 0xFF, rpt_desc);
			rd_append_byte(localData >> 16 & 0xFF, rpt_desc);
			rd_append_byte(localData >> 24 & 0xFF, rpt_desc);
		}
		else {
			// Data out of 32 bit unsigned integer range
			return -1;
		}
	}
	return 0;
}

static struct rd_main_item_node * rd_append_main_item_node(int first_bit, int last_bit, rd_node_type type_of_node, int caps_index, int collection_index, rd_main_items main_item_type, unsigned char report_id, struct rd_main_item_node **list) {
	struct rd_main_item_node *new_list_node;

	// Determine last node in the list
	while (*list != NULL)
	{
		list = &(*list)->next;
	}

	new_list_node = malloc(sizeof(*new_list_node)); // Create new list entry
	new_list_node->FirstBit = first_bit;
	new_list_node->LastBit = last_bit;
	new_list_node->TypeOfNode = type_of_node;
	new_list_node->CapsIndex = caps_index;
	new_list_node->CollectionIndex = collection_index;
	new_list_node->MainItemType = main_item_type;
	new_list_node->ReportID = report_id;
	new_list_node->next = NULL; // NULL marks last node in the list

	*list = new_list_node;
	return new_list_node;
}

static struct  rd_main_item_node * rd_insert_main_item_node(int first_bit, int last_bit, rd_node_type type_of_node, int caps_index, int collection_index, rd_main_items main_item_type, unsigned char report_id, struct rd_main_item_node **list) {
	// Insert item after the main item node referenced by list
	struct rd_main_item_node *next_item = (*list)->next;
	(*list)->next = NULL;
	rd_append_main_item_node(first_bit, last_bit, type_of_node, caps_index, collection_index, main_item_type, report_id, list);
	(*list)->next->next = next_item;
	return (*list)->next;
}

static struct rd_main_item_node * rd_search_main_item_list_for_bit_position(int search_bit, rd_main_items main_item_type, unsigned char report_id, struct rd_main_item_node **list) {
	// Determine first INPUT/OUTPUT/FEATURE main item, where the last bit position is equal or greater than the search bit position

	while (((*list)->next->MainItemType != rd_collection) &&
		((*list)->next->MainItemType != rd_collection_end) &&
		!(((*list)->next->LastBit >= search_bit) &&
			((*list)->next->ReportID == report_id) &&
			((*list)->next->MainItemType == main_item_type))
		)
	{
		list = &(*list)->next;
	}
	return *list;
}

int hid_winapi_descriptor_reconstruct_pp_data(void *preparsed_data, unsigned char *buf, size_t buf_size)
{
	hidp_preparsed_data *pp_data = (hidp_preparsed_data *) preparsed_data;

	// Check if MagicKey is correct, to ensure that pp_data points to an valid preparse data structure
	if (memcmp(pp_data->MagicKey, "HidP KDR", 8) != 0) {
		return -1;
	}

	struct rd_buffer rpt_desc;
	rpt_desc.buf = buf;
	rpt_desc.buf_size = buf_size;
	rpt_desc.byte_idx = 0;

	// Set pointer to the first node of link_collection_nodes
	phid_pp_link_collection_node link_collection_nodes = (phid_pp_link_collection_node)(((unsigned char*)&pp_data->caps[0]) + pp_data->FirstByteOfLinkCollectionArray);

	// ****************************************************************************************************************************
	// Create lookup tables for the bit range of each report per collection (position of first bit and last bit in each collection)
	// coll_bit_range[COLLECTION_INDEX][REPORT_ID][INPUT/OUTPUT/FEATURE]
	// ****************************************************************************************************************************
	
	// Allocate memory and initialize lookup table
	rd_bit_range ****coll_bit_range;
	coll_bit_range = malloc(pp_data->NumberLinkCollectionNodes * sizeof(*coll_bit_range));
	for (USHORT collection_node_idx = 0; collection_node_idx < pp_data->NumberLinkCollectionNodes; collection_node_idx++) {
		coll_bit_range[collection_node_idx] = malloc(256 * sizeof(*coll_bit_range[0])); // 256 possible report IDs (incl. 0x00)
		for (int reportid_idx = 0; reportid_idx < 256; reportid_idx++) {
			coll_bit_range[collection_node_idx][reportid_idx] = malloc(NUM_OF_HIDP_REPORT_TYPES * sizeof(*coll_bit_range[0][0]));
			for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
				coll_bit_range[collection_node_idx][reportid_idx][rt_idx] = malloc(sizeof(rd_bit_range));
				coll_bit_range[collection_node_idx][reportid_idx][rt_idx]->FirstBit = -1;
				coll_bit_range[collection_node_idx][reportid_idx][rt_idx]->LastBit = -1;
			}
		}
	}

	// Fill the lookup table where caps exist
	for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
		for (USHORT caps_idx = pp_data->caps_info[rt_idx].FirstCap; caps_idx < pp_data->caps_info[rt_idx].LastCap; caps_idx++) {
			int first_bit, last_bit;
			first_bit = (pp_data->caps[caps_idx].BytePosition - 1) * 8
			           + pp_data->caps[caps_idx].BitPosition;
			last_bit = first_bit + pp_data->caps[caps_idx].ReportSize
			                     * pp_data->caps[caps_idx].ReportCount - 1;
			if (coll_bit_range[pp_data->caps[caps_idx].LinkCollection][pp_data->caps[caps_idx].ReportID][rt_idx]->FirstBit == -1 ||
				coll_bit_range[pp_data->caps[caps_idx].LinkCollection][pp_data->caps[caps_idx].ReportID][rt_idx]->FirstBit > first_bit) {
				coll_bit_range[pp_data->caps[caps_idx].LinkCollection][pp_data->caps[caps_idx].ReportID][rt_idx]->FirstBit = first_bit;
			}
			if (coll_bit_range[pp_data->caps[caps_idx].LinkCollection][pp_data->caps[caps_idx].ReportID][rt_idx]->LastBit < last_bit) {
				coll_bit_range[pp_data->caps[caps_idx].LinkCollection][pp_data->caps[caps_idx].ReportID][rt_idx]->LastBit = last_bit;
			}
		}
	}

	// *************************************************************************
	// -Determine hierarchy levels of each collections and store it in:
	//  coll_levels[COLLECTION_INDEX]
	// -Determine number of direct childs of each collections and store it in:
	//  coll_number_of_direct_childs[COLLECTION_INDEX]
	// *************************************************************************
	int max_coll_level = 0;
	int *coll_levels = malloc(pp_data->NumberLinkCollectionNodes * sizeof(coll_levels[0]));
	int *coll_number_of_direct_childs = malloc(pp_data->NumberLinkCollectionNodes * sizeof(coll_number_of_direct_childs[0]));
	for (USHORT collection_node_idx = 0; collection_node_idx < pp_data->NumberLinkCollectionNodes; collection_node_idx++) {
		coll_levels[collection_node_idx] = -1;
		coll_number_of_direct_childs[collection_node_idx] = 0;
	}

	{
		int actual_coll_level = 0;
		USHORT collection_node_idx = 0;
		while (actual_coll_level >= 0) {
			coll_levels[collection_node_idx] = actual_coll_level;
			if ((link_collection_nodes[collection_node_idx].NumberOfChildren > 0) &&
				(coll_levels[link_collection_nodes[collection_node_idx].FirstChild] == -1)) {
				actual_coll_level++;
				coll_levels[collection_node_idx] = actual_coll_level;
				if (max_coll_level < actual_coll_level) {
					max_coll_level = actual_coll_level;
				}
				coll_number_of_direct_childs[collection_node_idx]++;
				collection_node_idx = link_collection_nodes[collection_node_idx].FirstChild;
			}
			else if (link_collection_nodes[collection_node_idx].NextSibling != 0) {
				coll_number_of_direct_childs[link_collection_nodes[collection_node_idx].Parent]++;
				collection_node_idx = link_collection_nodes[collection_node_idx].NextSibling;
			}
			else {
				actual_coll_level--;
				if (actual_coll_level >= 0) {
					collection_node_idx = link_collection_nodes[collection_node_idx].Parent;
				}
			}
		}
	}

	// *********************************************************************************
	// Propagate the bit range of each report from the child collections to their parent
	// and store the merged result for the parent
	// *********************************************************************************
	for (int actual_coll_level = max_coll_level - 1; actual_coll_level >= 0; actual_coll_level--) {
		for (USHORT collection_node_idx = 0; collection_node_idx < pp_data->NumberLinkCollectionNodes; collection_node_idx++) {
			if (coll_levels[collection_node_idx] == actual_coll_level) {
				USHORT child_idx = link_collection_nodes[collection_node_idx].FirstChild;
				while (child_idx) {
					for (int reportid_idx = 0; reportid_idx < 256; reportid_idx++) {
						for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
							// Merge bit range from childs
							if ((coll_bit_range[child_idx][reportid_idx][rt_idx]->FirstBit != -1) &&
								(coll_bit_range[collection_node_idx][reportid_idx][rt_idx]->FirstBit > coll_bit_range[child_idx][reportid_idx][rt_idx]->FirstBit)) {
								coll_bit_range[collection_node_idx][reportid_idx][rt_idx]->FirstBit = coll_bit_range[child_idx][reportid_idx][rt_idx]->FirstBit;
							}
							if (coll_bit_range[collection_node_idx][reportid_idx][rt_idx]->LastBit < coll_bit_range[child_idx][reportid_idx][rt_idx]->LastBit) {
								coll_bit_range[collection_node_idx][reportid_idx][rt_idx]->LastBit = coll_bit_range[child_idx][reportid_idx][rt_idx]->LastBit;
							}
							child_idx = link_collection_nodes[child_idx].NextSibling;
						}
					}
				}
			}
		}
	}

	// **************************************************************************************************
	// Determine child collection order of the whole hierarchy, based on previously determined bit ranges
	// and store it this index coll_child_order[COLLECTION_INDEX][DIRECT_CHILD_INDEX]
	// **************************************************************************************************
	USHORT **coll_child_order;
	coll_child_order = malloc(pp_data->NumberLinkCollectionNodes * sizeof(*coll_child_order));
	{
		BOOLEAN *coll_parsed_flag;
		coll_parsed_flag = malloc(pp_data->NumberLinkCollectionNodes * sizeof(coll_parsed_flag[0]));
		for (USHORT collection_node_idx = 0; collection_node_idx < pp_data->NumberLinkCollectionNodes; collection_node_idx++) {
			coll_parsed_flag[collection_node_idx] = FALSE;
		}
		int actual_coll_level = 0;
		USHORT collection_node_idx = 0;
		while (actual_coll_level >= 0) {
			if ((coll_number_of_direct_childs[collection_node_idx] != 0) &&
				(coll_parsed_flag[link_collection_nodes[collection_node_idx].FirstChild] == FALSE)) {
				coll_parsed_flag[link_collection_nodes[collection_node_idx].FirstChild] = TRUE;
				coll_child_order[collection_node_idx] = malloc((coll_number_of_direct_childs[collection_node_idx]) * sizeof(*coll_child_order[0]));

				{
					// Create list of child collection indices
					// sorted reverse to the order returned to HidP_GetLinkCollectionNodeschild
					// which seems to match the original order, as long as no bit position needs to be considered
					USHORT child_idx = link_collection_nodes[collection_node_idx].FirstChild;
					int child_count = coll_number_of_direct_childs[collection_node_idx] - 1;
					coll_child_order[collection_node_idx][child_count] = child_idx;
					while (link_collection_nodes[child_idx].NextSibling) {
						child_count--;
						child_idx = link_collection_nodes[child_idx].NextSibling;
						coll_child_order[collection_node_idx][child_count] = child_idx;
					}
				}

				if (coll_number_of_direct_childs[collection_node_idx] > 1) {
					// Sort child collections indices by bit positions
					for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
						for (int reportid_idx = 0; reportid_idx < 256; reportid_idx++) {
							for (int child_idx = 1; child_idx < coll_number_of_direct_childs[collection_node_idx]; child_idx++) {
								// since the coll_bit_range array is not sorted, we need to reference the collection index in 
								// our sorted coll_child_order array, and look up the corresponding bit ranges for comparing values to sort
								int prev_coll_idx = coll_child_order[collection_node_idx][child_idx - 1];
								int cur_coll_idx = coll_child_order[collection_node_idx][child_idx];
								if ((coll_bit_range[prev_coll_idx][reportid_idx][rt_idx]->FirstBit != -1) &&
									(coll_bit_range[cur_coll_idx][reportid_idx][rt_idx]->FirstBit != -1) &&
									(coll_bit_range[prev_coll_idx][reportid_idx][rt_idx]->FirstBit > coll_bit_range[cur_coll_idx][reportid_idx][rt_idx]->FirstBit)) {
									// Swap position indices of the two compared child collections
									USHORT idx_latch = coll_child_order[collection_node_idx][child_idx - 1];
									coll_child_order[collection_node_idx][child_idx - 1] = coll_child_order[collection_node_idx][child_idx];
									coll_child_order[collection_node_idx][child_idx] = idx_latch;
								}
							}
						}
					}
				}
				actual_coll_level++;
				collection_node_idx = link_collection_nodes[collection_node_idx].FirstChild;
			}
			else if (link_collection_nodes[collection_node_idx].NextSibling != 0) {
				collection_node_idx = link_collection_nodes[collection_node_idx].NextSibling;
			}
			else {
				actual_coll_level--;
				if (actual_coll_level >= 0) {
					collection_node_idx = link_collection_nodes[collection_node_idx].Parent;
				}
			}
		}
		free(coll_parsed_flag);
	}


	// ***************************************************************************************
	// Create sorted main_item_list containing all the Collection and CollectionEnd main items
	// ***************************************************************************************
	struct rd_main_item_node *main_item_list = NULL; // List root
	// Lookup table to find the Collection items in the list by index
	struct rd_main_item_node **coll_begin_lookup = malloc(pp_data->NumberLinkCollectionNodes * sizeof(*coll_begin_lookup));
	struct rd_main_item_node **coll_end_lookup = malloc(pp_data->NumberLinkCollectionNodes * sizeof(*coll_end_lookup));
	{
		int *coll_last_written_child = malloc(pp_data->NumberLinkCollectionNodes * sizeof(coll_last_written_child[0]));
		for (USHORT collection_node_idx = 0; collection_node_idx < pp_data->NumberLinkCollectionNodes; collection_node_idx++) {
			coll_last_written_child[collection_node_idx] = -1;
		}

		int actual_coll_level = 0;
		USHORT collection_node_idx = 0;
		struct rd_main_item_node *firstDelimiterNode = NULL;
		struct rd_main_item_node *delimiterCloseNode = NULL;
		coll_begin_lookup[0] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_collection, 0, &main_item_list);
		while (actual_coll_level >= 0) {
			if ((coll_number_of_direct_childs[collection_node_idx] != 0) &&
				(coll_last_written_child[collection_node_idx] == -1)) {
				// Collection has child collections, but none is written to the list yet

				coll_last_written_child[collection_node_idx] = coll_child_order[collection_node_idx][0];
				collection_node_idx = coll_child_order[collection_node_idx][0];

				// In a HID Report Descriptor, the first usage declared is the most preferred usage for the control.
				// While the order in the WIN32 capabiliy strutures is the opposite:
				// Here the preferred usage is the last aliased usage in the sequence.

				if (link_collection_nodes[collection_node_idx].IsAlias && (firstDelimiterNode == NULL)) {
					// Alliased Collection (First node in link_collection_nodes -> Last entry in report descriptor output)
					firstDelimiterNode = main_item_list;
					coll_begin_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_usage, 0, &main_item_list);
					coll_begin_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_close, 0, &main_item_list);
					delimiterCloseNode = main_item_list;
				}
				else {
					// Normal not aliased collection
					coll_begin_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_collection, 0, &main_item_list);
					actual_coll_level++;
				}


			}
			else if ((coll_number_of_direct_childs[collection_node_idx] > 1) &&
				(coll_last_written_child[collection_node_idx] != coll_child_order[collection_node_idx][coll_number_of_direct_childs[collection_node_idx] - 1])) {
				// Collection has child collections, and this is not the first child

				int nextChild = 1;
				while (coll_last_written_child[collection_node_idx] != coll_child_order[collection_node_idx][nextChild - 1]) {
					nextChild++;
				}
				coll_last_written_child[collection_node_idx] = coll_child_order[collection_node_idx][nextChild];
				collection_node_idx = coll_child_order[collection_node_idx][nextChild];
												
				if (link_collection_nodes[collection_node_idx].IsAlias && (firstDelimiterNode == NULL)) {
					// Alliased Collection (First node in link_collection_nodes -> Last entry in report descriptor output)
					firstDelimiterNode = main_item_list;
					coll_begin_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_usage, 0, &main_item_list);
					coll_begin_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_close, 0, &main_item_list);
					delimiterCloseNode = main_item_list;
				}
				else if (link_collection_nodes[collection_node_idx].IsAlias && (firstDelimiterNode != NULL)) {
					coll_begin_lookup[collection_node_idx] = rd_insert_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_usage, 0, &firstDelimiterNode);
				}
				else if (!link_collection_nodes[collection_node_idx].IsAlias && (firstDelimiterNode != NULL)) {
					coll_begin_lookup[collection_node_idx] = rd_insert_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_usage, 0, &firstDelimiterNode);
					coll_begin_lookup[collection_node_idx] = rd_insert_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_delimiter_open, 0, &firstDelimiterNode);
					firstDelimiterNode = NULL;
					main_item_list = delimiterCloseNode;
					delimiterCloseNode = NULL; // Last entry of alias has .IsAlias == FALSE
				}
				if (!link_collection_nodes[collection_node_idx].IsAlias) {
					coll_begin_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_collection, 0, &main_item_list);
					actual_coll_level++;
				}
			}
			else {
				actual_coll_level--;
				coll_end_lookup[collection_node_idx] = rd_append_main_item_node(0, 0, rd_item_node_collection, 0, collection_node_idx, rd_collection_end, 0, &main_item_list);
				collection_node_idx = link_collection_nodes[collection_node_idx].Parent;
			}
		}
		free(coll_last_written_child);
	}


	// ****************************************************************
	// Inserted Input/Output/Feature main items into the main_item_list
	// in order of reconstructed bit positions
	// ****************************************************************
	for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
		// Add all value caps to node list
		struct rd_main_item_node *firstDelimiterNode = NULL;
		struct rd_main_item_node *delimiterCloseNode = NULL;
		for (USHORT caps_idx = pp_data->caps_info[rt_idx].FirstCap; caps_idx < pp_data->caps_info[rt_idx].LastCap; caps_idx++) {
			struct rd_main_item_node *coll_begin = coll_begin_lookup[pp_data->caps[caps_idx].LinkCollection];
			int first_bit, last_bit;
			first_bit = (pp_data->caps[caps_idx].BytePosition - 1) * 8 +
				pp_data->caps[caps_idx].BitPosition;
			last_bit = first_bit + pp_data->caps[caps_idx].ReportSize *
				pp_data->caps[caps_idx].ReportCount - 1;

			for (int child_idx = 0; child_idx < coll_number_of_direct_childs[pp_data->caps[caps_idx].LinkCollection]; child_idx++) {
				// Determine in which section before/between/after child collection the item should be inserted
				if (first_bit < coll_bit_range[coll_child_order[pp_data->caps[caps_idx].LinkCollection][child_idx]][pp_data->caps[caps_idx].ReportID][rt_idx]->FirstBit)
				{
					// Note, that the default value for undefined coll_bit_range is -1, which can't be greater than the bit position
					break;
				}
				coll_begin = coll_end_lookup[coll_child_order[pp_data->caps[caps_idx].LinkCollection][child_idx]];
			}
			struct rd_main_item_node *list_node;
			list_node = rd_search_main_item_list_for_bit_position(first_bit, (rd_main_items) rt_idx, pp_data->caps[caps_idx].ReportID, &coll_begin);

			// In a HID Report Descriptor, the first usage declared is the most preferred usage for the control.
			// While the order in the WIN32 capabiliy strutures is the opposite:
			// Here the preferred usage is the last aliased usage in the sequence.

			if (pp_data->caps[caps_idx].IsAlias && (firstDelimiterNode == NULL)) {
				// Alliased Usage (First node in pp_data->caps -> Last entry in report descriptor output)
				firstDelimiterNode = list_node;
				rd_insert_main_item_node(first_bit, last_bit, rd_item_node_cap, caps_idx, pp_data->caps[caps_idx].LinkCollection, rd_delimiter_usage, pp_data->caps[caps_idx].ReportID, &list_node);
				rd_insert_main_item_node(first_bit, last_bit, rd_item_node_cap, caps_idx, pp_data->caps[caps_idx].LinkCollection, rd_delimiter_close, pp_data->caps[caps_idx].ReportID, &list_node);
				delimiterCloseNode = list_node;
			} else if (pp_data->caps[caps_idx].IsAlias && (firstDelimiterNode != NULL)) {
				rd_insert_main_item_node(first_bit, last_bit, rd_item_node_cap, caps_idx, pp_data->caps[caps_idx].LinkCollection, rd_delimiter_usage, pp_data->caps[caps_idx].ReportID, &list_node);
			}
			else if (!pp_data->caps[caps_idx].IsAlias && (firstDelimiterNode != NULL)) {
				// Alliased Collection (Last node in pp_data->caps -> First entry in report descriptor output)
				rd_insert_main_item_node(first_bit, last_bit, rd_item_node_cap, caps_idx, pp_data->caps[caps_idx].LinkCollection, rd_delimiter_usage, pp_data->caps[caps_idx].ReportID, &list_node);
				rd_insert_main_item_node(first_bit, last_bit, rd_item_node_cap, caps_idx, pp_data->caps[caps_idx].LinkCollection, rd_delimiter_open, pp_data->caps[caps_idx].ReportID, &list_node);
				firstDelimiterNode = NULL;
				list_node = delimiterCloseNode;
				delimiterCloseNode = NULL; // Last entry of alias has .IsAlias == FALSE
			}
			if (!pp_data->caps[caps_idx].IsAlias) {
				rd_insert_main_item_node(first_bit, last_bit, rd_item_node_cap, caps_idx, pp_data->caps[caps_idx].LinkCollection, (rd_main_items) rt_idx, pp_data->caps[caps_idx].ReportID, &list_node);
			}
		}
	}


	// ***********************************************************
	// Add const main items for padding to main_item_list
	// -To fill all bit gaps
	// -At each report end for 8bit padding
	//  Note that information about the padding at the report end,
	//  is not stored in the preparsed data, but in practice all
	//  report descriptors seem to have it, as assumed here.
	// ***********************************************************
	{
		int *last_bit_position[NUM_OF_HIDP_REPORT_TYPES];
		struct rd_main_item_node **last_report_item_lookup[NUM_OF_HIDP_REPORT_TYPES];
		for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
			last_bit_position[rt_idx] = malloc(256 * sizeof(*last_bit_position[rt_idx]));
			last_report_item_lookup[rt_idx] = malloc(256 * sizeof(*last_report_item_lookup[rt_idx]));
			for (int reportid_idx = 0; reportid_idx < 256; reportid_idx++) {
				last_bit_position[rt_idx][reportid_idx] = -1;
				last_report_item_lookup[rt_idx][reportid_idx] = NULL;
			}
		}

		struct rd_main_item_node *list = main_item_list; // List root;

		while (list->next != NULL)
		{
			if ((list->MainItemType >= rd_input) &&
				(list->MainItemType <= rd_feature)) {
				// INPUT, OUTPUT or FEATURE
				if (list->FirstBit != -1) {
					if ((last_bit_position[list->MainItemType][list->ReportID] + 1 != list->FirstBit) &&
						(last_report_item_lookup[list->MainItemType][list->ReportID] != NULL) &&
						(last_report_item_lookup[list->MainItemType][list->ReportID]->FirstBit != list->FirstBit) // Happens in case of IsMultipleItemsForArray for multiple dedicated usages for a multi-button array
						) {
						struct rd_main_item_node *list_node = rd_search_main_item_list_for_bit_position(last_bit_position[list->MainItemType][list->ReportID], list->MainItemType, list->ReportID, &last_report_item_lookup[list->MainItemType][list->ReportID]);
						rd_insert_main_item_node(last_bit_position[list->MainItemType][list->ReportID] + 1, list->FirstBit - 1, rd_item_node_padding, -1, 0, list->MainItemType, list->ReportID, &list_node);
					}
					last_bit_position[list->MainItemType][list->ReportID] = list->LastBit;
					last_report_item_lookup[list->MainItemType][list->ReportID] = list;
				}
			}
			list = list->next;
		}
		// Add 8 bit padding at each report end
		for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
			for (int reportid_idx = 0; reportid_idx < 256; reportid_idx++) {
				if (last_bit_position[rt_idx][reportid_idx] != -1) {
					int padding = 8 - ((last_bit_position[rt_idx][reportid_idx] + 1) % 8);
					if (padding < 8) {
						// Insert padding item after item referenced in last_report_item_lookup
						rd_insert_main_item_node(last_bit_position[rt_idx][reportid_idx] + 1, last_bit_position[rt_idx][reportid_idx] + padding, rd_item_node_padding, -1, 0, (rd_main_items) rt_idx, (unsigned char) reportid_idx, &last_report_item_lookup[rt_idx][reportid_idx]);
					}
				}
			}
			free(last_bit_position[rt_idx]);
			free(last_report_item_lookup[rt_idx]);
		}
	}


	// ***********************************
	// Encode the report descriptor output
	// ***********************************
	UCHAR last_report_id = 0;
	USAGE last_usage_page = 0;
	LONG last_physical_min = 0;// If both, Physical Minimum and Physical Maximum are 0, the logical limits should be taken as physical limits according USB HID spec 1.11 chapter 6.2.2.7
	LONG last_physical_max = 0;
	ULONG last_unit_exponent = 0; // If Unit Exponent is Undefined it should be considered as 0 according USB HID spec 1.11 chapter 6.2.2.7
	ULONG last_unit = 0; // If the first nibble is 7, or second nibble of Unit is 0, the unit is None according USB HID spec 1.11 chapter 6.2.2.7
	BOOLEAN inhibit_write_of_usage = FALSE; // Needed in case of delimited usage print, before the normal collection or cap
	int report_count = 0;
	while (main_item_list != NULL)
	{
		int rt_idx = main_item_list->MainItemType;
		int	caps_idx = main_item_list->CapsIndex;
		if (main_item_list->MainItemType == rd_collection) {
			if (last_usage_page != link_collection_nodes[main_item_list->CollectionIndex].LinkUsagePage) {
				// Write "Usage Page" at the begin of a collection - except it refers the same table as wrote last 
				rd_write_short_item(rd_global_usage_page, link_collection_nodes[main_item_list->CollectionIndex].LinkUsagePage, &rpt_desc);
				last_usage_page = link_collection_nodes[main_item_list->CollectionIndex].LinkUsagePage;
			}
			if (inhibit_write_of_usage) {
				// Inhibit only once after DELIMITER statement
				inhibit_write_of_usage = FALSE;
			}
			else {
				// Write "Usage" of collection
				rd_write_short_item(rd_local_usage, link_collection_nodes[main_item_list->CollectionIndex].LinkUsage, &rpt_desc);
			}
			// Write begin of "Collection" 
			rd_write_short_item(rd_main_collection, link_collection_nodes[main_item_list->CollectionIndex].CollectionType, &rpt_desc);
		}
		else if (main_item_list->MainItemType == rd_collection_end) {
			// Write "End Collection"
			rd_write_short_item(rd_main_collection_end, 0, &rpt_desc);
		}
		else if (main_item_list->MainItemType == rd_delimiter_open) {
			if (main_item_list->CollectionIndex != -1) {
				// Write "Usage Page" inside of a collection delmiter section
				if (last_usage_page != link_collection_nodes[main_item_list->CollectionIndex].LinkUsagePage) {
					rd_write_short_item(rd_global_usage_page, link_collection_nodes[main_item_list->CollectionIndex].LinkUsagePage, &rpt_desc);
					last_usage_page = link_collection_nodes[main_item_list->CollectionIndex].LinkUsagePage;
				}
			}
			else if (main_item_list->CapsIndex != 0) {
				// Write "Usage Page" inside of a main item delmiter section
				if (pp_data->caps[caps_idx].UsagePage != last_usage_page) {
					rd_write_short_item(rd_global_usage_page, pp_data->caps[caps_idx].UsagePage, &rpt_desc);
					last_usage_page = pp_data->caps[caps_idx].UsagePage;
				}
			}
			// Write "Delimiter Open"
			rd_write_short_item(rd_local_delimiter, 1, &rpt_desc); // 1 = open set of aliased usages
		}
		else if (main_item_list->MainItemType == rd_delimiter_usage) {
			if (main_item_list->CollectionIndex != -1) {
				// Write aliased collection "Usage"
				rd_write_short_item(rd_local_usage, link_collection_nodes[main_item_list->CollectionIndex].LinkUsage, &rpt_desc);
			}  if (main_item_list->CapsIndex != 0) {
				// Write aliased main item range from "Usage Minimum" to "Usage Maximum"
				if (pp_data->caps[caps_idx].IsRange) {
					rd_write_short_item(rd_local_usage_minimum, pp_data->caps[caps_idx].Range.UsageMin, &rpt_desc);
					rd_write_short_item(rd_local_usage_maximum, pp_data->caps[caps_idx].Range.UsageMax, &rpt_desc);
				}
				else {
					// Write single aliased main item "Usage"
					rd_write_short_item(rd_local_usage, pp_data->caps[caps_idx].NotRange.Usage, &rpt_desc);
				}
			}
		}
		else if (main_item_list->MainItemType == rd_delimiter_close) {
			// Write "Delimiter Close"
			rd_write_short_item(rd_local_delimiter, 0, &rpt_desc); // 0 = close set of aliased usages
			// Inhibit next usage write
			inhibit_write_of_usage = TRUE;
		}
		else if (main_item_list->TypeOfNode == rd_item_node_padding) {
			// Padding
			// The preparsed data doesn't contain any information about padding. Therefore all undefined gaps
			// in the reports are filled with the same style of constant padding. 

			// Write "Report Size" with number of padding bits
			rd_write_short_item(rd_global_report_size, (main_item_list->LastBit - main_item_list->FirstBit + 1), &rpt_desc);

			// Write "Report Count" for padding always as 1
			rd_write_short_item(rd_global_report_count, 1, &rpt_desc);

			if (rt_idx == HidP_Input) {
				// Write "Input" main item - We know it's Constant - We can only guess the other bits, but they don't matter in case of const
				rd_write_short_item(rd_main_input, 0x03, &rpt_desc); // Const / Abs
			}
			else if (rt_idx == HidP_Output) {
				// Write "Output" main item - We know it's Constant - We can only guess the other bits, but they don't matter in case of const
				rd_write_short_item(rd_main_output, 0x03, &rpt_desc); // Const / Abs
			}
			else if (rt_idx == HidP_Feature) {
				// Write "Feature" main item - We know it's Constant - We can only guess the other bits, but they don't matter in case of const
				rd_write_short_item(rd_main_feature, 0x03, &rpt_desc); // Const / Abs
			}
			report_count = 0;
		}
		else if (pp_data->caps[caps_idx].IsButtonCap) {
			// Button
			// (The preparsed data contain different data for 1 bit Button caps, than for parametric Value caps)

			if (last_report_id != pp_data->caps[caps_idx].ReportID) {
				// Write "Report ID" if changed
				rd_write_short_item(rd_global_report_id, pp_data->caps[caps_idx].ReportID, &rpt_desc);
				last_report_id = pp_data->caps[caps_idx].ReportID;
			}

			// Write "Usage Page" when changed
			if (pp_data->caps[caps_idx].UsagePage != last_usage_page) {
				rd_write_short_item(rd_global_usage_page, pp_data->caps[caps_idx].UsagePage, &rpt_desc);
				last_usage_page = pp_data->caps[caps_idx].UsagePage;
			}

			// Write only local report items for each cap, if ReportCount > 1
			if (pp_data->caps[caps_idx].IsRange) {
				report_count += (pp_data->caps[caps_idx].Range.DataIndexMax - pp_data->caps[caps_idx].Range.DataIndexMin);
			}

			if (inhibit_write_of_usage) {
				// Inhibit only once after Delimiter - Reset flag
				inhibit_write_of_usage = FALSE;
			}
			else {
				if (pp_data->caps[caps_idx].IsRange) {
					// Write range from "Usage Minimum" to "Usage Maximum"
					rd_write_short_item(rd_local_usage_minimum, pp_data->caps[caps_idx].Range.UsageMin, &rpt_desc);
					rd_write_short_item(rd_local_usage_maximum, pp_data->caps[caps_idx].Range.UsageMax, &rpt_desc);
				}
				else {
					// Write single "Usage"
					rd_write_short_item(rd_local_usage, pp_data->caps[caps_idx].NotRange.Usage, &rpt_desc);
				}
			}

			if (pp_data->caps[caps_idx].IsDesignatorRange) {
				// Write physical descriptor indices range from "Designator Minimum" to "Designator Maximum"
				rd_write_short_item(rd_local_designator_minimum, pp_data->caps[caps_idx].Range.DesignatorMin, &rpt_desc);
				rd_write_short_item(rd_local_designator_maximum, pp_data->caps[caps_idx].Range.DesignatorMax, &rpt_desc);
			}
			else if (pp_data->caps[caps_idx].NotRange.DesignatorIndex != 0) {
				// Designator set 0 is a special descriptor set (of the HID Physical Descriptor),
				// that specifies the number of additional descriptor sets.
				// Therefore Designator Index 0 can never be a useful reference for a control and we can inhibit it.
				// Write single "Designator Index"
				rd_write_short_item(rd_local_designator_index, pp_data->caps[caps_idx].NotRange.DesignatorIndex, &rpt_desc);
			}

			if (pp_data->caps[caps_idx].IsStringRange) {
				// Write range of indices of the USB string descriptor, from "String Minimum" to "String Maximum"
				rd_write_short_item(rd_local_string_minimum, pp_data->caps[caps_idx].Range.StringMin, &rpt_desc);
				rd_write_short_item(rd_local_string_maximum, pp_data->caps[caps_idx].Range.StringMax, &rpt_desc);
			}
			else if (pp_data->caps[caps_idx].NotRange.StringIndex != 0) {
				// String Index 0 is a special entry of the USB string descriptor, that contains a list of supported languages,
				// therefore Designator Index 0 can never be a useful reference for a control and we can inhibit it.
				// Write single "String Index"
				rd_write_short_item(rd_local_string, pp_data->caps[caps_idx].NotRange.StringIndex, &rpt_desc);
			}

			if ((main_item_list->next != NULL) &&
				((int)main_item_list->next->MainItemType == rt_idx) &&
				(main_item_list->next->TypeOfNode == rd_item_node_cap) &&
				(pp_data->caps[main_item_list->next->CapsIndex].IsButtonCap) &&
				(!pp_data->caps[caps_idx].IsRange) && // This node in list is no array
				(!pp_data->caps[main_item_list->next->CapsIndex].IsRange) && // Next node in list is no array
				(pp_data->caps[main_item_list->next->CapsIndex].UsagePage == pp_data->caps[caps_idx].UsagePage) &&
				(pp_data->caps[main_item_list->next->CapsIndex].ReportID == pp_data->caps[caps_idx].ReportID) &&
				(pp_data->caps[main_item_list->next->CapsIndex].BitField == pp_data->caps[caps_idx].BitField)
				) {
				if (main_item_list->next->FirstBit != main_item_list->FirstBit) {
					// In case of IsMultipleItemsForArray for multiple dedicated usages for a multi-button array, the report count should be incremented 
							
					// Skip global items until any of them changes, than use ReportCount item to write the count of identical report fields
					report_count++;
				}
			}
			else {

				if ((pp_data->caps[caps_idx].Button.LogicalMin == 0) &&
					(pp_data->caps[caps_idx].Button.LogicalMax == 0)) {
					// While a HID report descriptor must always contain LogicalMinimum and LogicalMaximum,
					// the preparsed data contain both fields set to zero, for the case of simple buttons
					// Write "Logical Minimum" set to 0 and "Logical Maximum" set to 1
					rd_write_short_item(rd_global_logical_minimum, 0, &rpt_desc);
					rd_write_short_item(rd_global_logical_maximum, 1, &rpt_desc);
				}
				else {
					// Write logical range from "Logical Minimum" to "Logical Maximum"
					rd_write_short_item(rd_global_logical_minimum, pp_data->caps[caps_idx].Button.LogicalMin, &rpt_desc);
					rd_write_short_item(rd_global_logical_maximum, pp_data->caps[caps_idx].Button.LogicalMax, &rpt_desc);
				}

				// Write "Report Size"
				rd_write_short_item(rd_global_report_size, pp_data->caps[caps_idx].ReportSize, &rpt_desc);

				// Write "Report Count"
				if (!pp_data->caps[caps_idx].IsRange) {
					// Variable bit field with one bit per button
					// In case of multiple usages with the same items, only "Usage" is written per cap, and "Report Count" is incremented
					rd_write_short_item(rd_global_report_count, pp_data->caps[caps_idx].ReportCount + report_count, &rpt_desc);
				}
				else {
					// Button array of "Report Size" x "Report Count
					rd_write_short_item(rd_global_report_count, pp_data->caps[caps_idx].ReportCount, &rpt_desc);
				}


				// Buttons have only 1 bit and therefore no physical limits/units -> Set to undefined state
				if (last_physical_min != 0) {
					// Write "Physical Minimum", but only if changed
					last_physical_min = 0;
					rd_write_short_item(rd_global_physical_minimum, last_physical_min, &rpt_desc);
				}
				if (last_physical_max != 0) {
					// Write "Physical Maximum", but only if changed
					last_physical_max = 0;
					rd_write_short_item(rd_global_physical_maximum, last_physical_max, &rpt_desc);
				}
				if (last_unit_exponent != 0) {
					// Write "Unit Exponent", but only if changed
					last_unit_exponent = 0;
					rd_write_short_item(rd_global_unit_exponent, last_unit_exponent, &rpt_desc);
				}
				if (last_unit != 0) {
					// Write "Unit",but only if changed
					last_unit = 0;
					rd_write_short_item(rd_global_unit, last_unit, &rpt_desc);
				}

				// Write "Input" main item
				if (rt_idx == HidP_Input) {
					rd_write_short_item(rd_main_input, pp_data->caps[caps_idx].BitField, &rpt_desc);
				}
				// Write "Output" main item
				else if (rt_idx == HidP_Output) {
					rd_write_short_item(rd_main_output, pp_data->caps[caps_idx].BitField, &rpt_desc);
				}
				// Write "Feature" main item
				else if (rt_idx == HidP_Feature) {
					rd_write_short_item(rd_main_feature, pp_data->caps[caps_idx].BitField, &rpt_desc);
				}
				report_count = 0;
			}
		}
		else {

			if (last_report_id != pp_data->caps[caps_idx].ReportID) {
				// Write "Report ID" if changed
				rd_write_short_item(rd_global_report_id, pp_data->caps[caps_idx].ReportID, &rpt_desc);
				last_report_id = pp_data->caps[caps_idx].ReportID;
			}

			// Write "Usage Page" if changed
			if (pp_data->caps[caps_idx].UsagePage != last_usage_page) {
				rd_write_short_item(rd_global_usage_page, pp_data->caps[caps_idx].UsagePage, &rpt_desc);
				last_usage_page = pp_data->caps[caps_idx].UsagePage;
			}

			if (inhibit_write_of_usage) {
				// Inhibit only once after Delimiter - Reset flag
				inhibit_write_of_usage = FALSE;
			}
			else {
				if (pp_data->caps[caps_idx].IsRange) {
					// Write usage range from "Usage Minimum" to "Usage Maximum"
					rd_write_short_item(rd_local_usage_minimum, pp_data->caps[caps_idx].Range.UsageMin, &rpt_desc);
					rd_write_short_item(rd_local_usage_maximum, pp_data->caps[caps_idx].Range.UsageMax, &rpt_desc);
				}
				else {
					// Write single "Usage"
					rd_write_short_item(rd_local_usage, pp_data->caps[caps_idx].NotRange.Usage, &rpt_desc);
				}
			}

			if (pp_data->caps[caps_idx].IsDesignatorRange) {
				// Write physical descriptor indices range from "Designator Minimum" to "Designator Maximum"
				rd_write_short_item(rd_local_designator_minimum, pp_data->caps[caps_idx].Range.DesignatorMin, &rpt_desc);
				rd_write_short_item(rd_local_designator_maximum, pp_data->caps[caps_idx].Range.DesignatorMax, &rpt_desc);
			}
			else if (pp_data->caps[caps_idx].NotRange.DesignatorIndex != 0) {
				// Designator set 0 is a special descriptor set (of the HID Physical Descriptor),
				// that specifies the number of additional descriptor sets.
				// Therefore Designator Index 0 can never be a useful reference for a control and we can inhibit it.
				// Write single "Designator Index"
				rd_write_short_item(rd_local_designator_index, pp_data->caps[caps_idx].NotRange.DesignatorIndex, &rpt_desc);
			}

			if (pp_data->caps[caps_idx].IsStringRange) {
				// Write range of indices of the USB string descriptor, from "String Minimum" to "String Maximum"
				rd_write_short_item(rd_local_string_minimum, pp_data->caps[caps_idx].Range.StringMin, &rpt_desc);
				rd_write_short_item(rd_local_string_maximum, pp_data->caps[caps_idx].Range.StringMax, &rpt_desc);
			}
			else if (pp_data->caps[caps_idx].NotRange.StringIndex != 0) {
				// String Index 0 is a special entry of the USB string descriptor, that contains a list of supported languages,
				// therefore Designator Index 0 can never be a useful reference for a control and we can inhibit it.
				// Write single "String Index"
				rd_write_short_item(rd_local_string, pp_data->caps[caps_idx].NotRange.StringIndex, &rpt_desc);
			}

			if ((pp_data->caps[caps_idx].BitField & 0x02) != 0x02) {
				// In case of an value array overwrite "Report Count"
				pp_data->caps[caps_idx].ReportCount = pp_data->caps[caps_idx].Range.DataIndexMax - pp_data->caps[caps_idx].Range.DataIndexMin + 1;
			}


			// Print only local report items for each cap, if ReportCount > 1
			if ((main_item_list->next != NULL) &&
				((int) main_item_list->next->MainItemType == rt_idx) &&
				(main_item_list->next->TypeOfNode == rd_item_node_cap) &&
				(!pp_data->caps[main_item_list->next->CapsIndex].IsButtonCap) &&
				(!pp_data->caps[caps_idx].IsRange) && // This node in list is no array
				(!pp_data->caps[main_item_list->next->CapsIndex].IsRange) && // Next node in list is no array
				(pp_data->caps[main_item_list->next->CapsIndex].UsagePage == pp_data->caps[caps_idx].UsagePage) &&
				(pp_data->caps[main_item_list->next->CapsIndex].NotButton.LogicalMin == pp_data->caps[caps_idx].NotButton.LogicalMin) &&
				(pp_data->caps[main_item_list->next->CapsIndex].NotButton.LogicalMax == pp_data->caps[caps_idx].NotButton.LogicalMax) &&
				(pp_data->caps[main_item_list->next->CapsIndex].NotButton.PhysicalMin == pp_data->caps[caps_idx].NotButton.PhysicalMin) &&
				(pp_data->caps[main_item_list->next->CapsIndex].NotButton.PhysicalMax == pp_data->caps[caps_idx].NotButton.PhysicalMax) &&
				(pp_data->caps[main_item_list->next->CapsIndex].UnitsExp == pp_data->caps[caps_idx].UnitsExp) &&
				(pp_data->caps[main_item_list->next->CapsIndex].Units == pp_data->caps[caps_idx].Units) &&
				(pp_data->caps[main_item_list->next->CapsIndex].ReportSize == pp_data->caps[caps_idx].ReportSize) &&
				(pp_data->caps[main_item_list->next->CapsIndex].ReportID == pp_data->caps[caps_idx].ReportID) &&
				(pp_data->caps[main_item_list->next->CapsIndex].BitField == pp_data->caps[caps_idx].BitField) &&
				(pp_data->caps[main_item_list->next->CapsIndex].ReportCount == 1) &&
				(pp_data->caps[caps_idx].ReportCount == 1)
				) {
				// Skip global items until any of them changes, than use ReportCount item to write the count of identical report fields
				report_count++;
			}
			else {
				// Value

				// Write logical range from "Logical Minimum" to "Logical Maximum"
				rd_write_short_item(rd_global_logical_minimum, pp_data->caps[caps_idx].NotButton.LogicalMin, &rpt_desc);
				rd_write_short_item(rd_global_logical_maximum, pp_data->caps[caps_idx].NotButton.LogicalMax, &rpt_desc);

				if ((last_physical_min != pp_data->caps[caps_idx].NotButton.PhysicalMin) ||
					(last_physical_max != pp_data->caps[caps_idx].NotButton.PhysicalMax)) {
					// Write range from "Physical Minimum" to " Physical Maximum", but only if one of them changed
					rd_write_short_item(rd_global_physical_minimum, pp_data->caps[caps_idx].NotButton.PhysicalMin, &rpt_desc);
					last_physical_min = pp_data->caps[caps_idx].NotButton.PhysicalMin;
					rd_write_short_item(rd_global_physical_maximum, pp_data->caps[caps_idx].NotButton.PhysicalMax, &rpt_desc);
					last_physical_max = pp_data->caps[caps_idx].NotButton.PhysicalMax;
				}


				if (last_unit_exponent != pp_data->caps[caps_idx].UnitsExp) {
					// Write "Unit Exponent", but only if changed
					rd_write_short_item(rd_global_unit_exponent, pp_data->caps[caps_idx].UnitsExp, &rpt_desc);
					last_unit_exponent = pp_data->caps[caps_idx].UnitsExp;
				}

				if (last_unit != pp_data->caps[caps_idx].Units) {
					// Write physical "Unit", but only if changed
					rd_write_short_item(rd_global_unit, pp_data->caps[caps_idx].Units, &rpt_desc);
					last_unit = pp_data->caps[caps_idx].Units;
				}

				// Write "Report Size"
				rd_write_short_item(rd_global_report_size, pp_data->caps[caps_idx].ReportSize, &rpt_desc);

				// Write "Report Count"
				rd_write_short_item(rd_global_report_count, pp_data->caps[caps_idx].ReportCount + report_count, &rpt_desc);

				if (rt_idx == HidP_Input) {
					// Write "Input" main item
					rd_write_short_item(rd_main_input, pp_data->caps[caps_idx].BitField, &rpt_desc);
				}
				else if (rt_idx == HidP_Output) {
					// Write "Output" main item
					rd_write_short_item(rd_main_output, pp_data->caps[caps_idx].BitField, &rpt_desc);
				}
				else if (rt_idx == HidP_Feature) {
					// Write "Feature" main item
					rd_write_short_item(rd_main_feature, pp_data->caps[caps_idx].BitField, &rpt_desc);
				}
				report_count = 0;
			}
		}

		// Go to next item in main_item_list and free the memory of the actual item
		struct rd_main_item_node *main_item_list_prev = main_item_list;
		main_item_list = main_item_list->next;
		free(main_item_list_prev);
	}

	// Free multidimensionable array: coll_bit_range[COLLECTION_INDEX][REPORT_ID][INPUT/OUTPUT/FEATURE]
	// Free multidimensionable array: coll_child_order[COLLECTION_INDEX][DIRECT_CHILD_INDEX]
	for (USHORT collection_node_idx = 0; collection_node_idx < pp_data->NumberLinkCollectionNodes; collection_node_idx++) {
		for (int reportid_idx = 0; reportid_idx < 256; reportid_idx++) {
			for (HIDP_REPORT_TYPE rt_idx = 0; rt_idx < NUM_OF_HIDP_REPORT_TYPES; rt_idx++) {
				free(coll_bit_range[collection_node_idx][reportid_idx][rt_idx]);
			}
			free(coll_bit_range[collection_node_idx][reportid_idx]);
		}
		free(coll_bit_range[collection_node_idx]);
		if (coll_number_of_direct_childs[collection_node_idx] != 0) free(coll_child_order[collection_node_idx]);
	}
	free(coll_bit_range);
	free(coll_child_order);

	// Free one dimensional arrays
	free(coll_begin_lookup);
	free(coll_end_lookup);
	free(coll_levels);
	free(coll_number_of_direct_childs);

	return (int) rpt_desc.byte_idx;
}
