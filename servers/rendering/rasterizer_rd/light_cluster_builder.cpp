/*************************************************************************/
/*  light_cluster_builder.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "light_cluster_builder.h"

void LightClusterBuilder::begin(const Transform &p_view_transform, const CameraMatrix &p_cam_projection) {
	view_xform = p_view_transform;
	projection = p_cam_projection;
	z_near = -projection.get_z_near();
	z_far = -projection.get_z_far();

	//reset counts
	light_count = 0;
	refprobe_count = 0;
	decal_count = 0;
	item_count = 0;
	sort_id_count = 0;
}

void LightClusterBuilder::bake_cluster() {
	float slice_depth = (z_near - z_far) / depth;

	uint8_t *cluster_dataw = cluster_data.ptrw();
	Cell *cluster_data_ptr = (Cell *)cluster_dataw;
	//clear the cluster
	zeromem(cluster_data_ptr, (width * height * depth * sizeof(Cell)));

	/* Step 1, create cell positions and count them */

	for (uint32_t i = 0; i < item_count; i++) {
		const Item &item = items[i];

		int from_slice = Math::floor((z_near - (item.aabb.position.z + item.aabb.size.z)) / slice_depth);
		int to_slice = Math::floor((z_near - item.aabb.position.z) / slice_depth);

		if (from_slice >= (int)depth || to_slice < 0) {
			continue; //sorry no go
		}

		from_slice = MAX(0, from_slice);
		to_slice = MIN((int)depth - 1, to_slice);

		for (int j = from_slice; j <= to_slice; j++) {
			Vector3 min = item.aabb.position;
			Vector3 max = item.aabb.position + item.aabb.size;

			float limit_near = MIN((z_near - slice_depth * j), max.z);
			float limit_far = MAX((z_near - slice_depth * (j + 1)), min.z);

			max.z = limit_near;
			min.z = limit_near;

			Vector3 proj_min = projection.xform(min);
			Vector3 proj_max = projection.xform(max);

			int near_from_x = int(Math::floor((proj_min.x * 0.5 + 0.5) * width));
			int near_from_y = int(Math::floor((-proj_max.y * 0.5 + 0.5) * height));
			int near_to_x = int(Math::floor((proj_max.x * 0.5 + 0.5) * width));
			int near_to_y = int(Math::floor((-proj_min.y * 0.5 + 0.5) * height));

			max.z = limit_far;
			min.z = limit_far;

			proj_min = projection.xform(min);
			proj_max = projection.xform(max);

			int far_from_x = int(Math::floor((proj_min.x * 0.5 + 0.5) * width));
			int far_from_y = int(Math::floor((-proj_max.y * 0.5 + 0.5) * height));
			int far_to_x = int(Math::floor((proj_max.x * 0.5 + 0.5) * width));
			int far_to_y = int(Math::floor((-proj_min.y * 0.5 + 0.5) * height));

			//print_line(itos(j) + " near - " + Vector2i(near_from_x, near_from_y) + " -> " + Vector2i(near_to_x, near_to_y));
			//print_line(itos(j) + " far - " + Vector2i(far_from_x, far_from_y) + " -> " + Vector2i(far_to_x, far_to_y));

			int from_x = MIN(near_from_x, far_from_x);
			int from_y = MIN(near_from_y, far_from_y);
			int to_x = MAX(near_to_x, far_to_x);
			int to_y = MAX(near_to_y, far_to_y);

			if (from_x >= (int)width || to_x < 0 || from_y >= (int)height || to_y < 0) {
				continue;
			}

			int sx = MAX(0, from_x);
			int sy = MAX(0, from_y);
			int dx = MIN((int)width - 1, to_x);
			int dy = MIN((int)height - 1, to_y);

			//print_line(itos(j) + " - " + Vector2i(sx, sy) + " -> " + Vector2i(dx, dy));

			for (int x = sx; x <= dx; x++) {
				for (int y = sy; y <= dy; y++) {
					uint32_t offset = j * (width * height) + y * width + x;

					if (unlikely(sort_id_count == sort_id_max)) {
						sort_id_max = nearest_power_of_2_templated(sort_id_max + 1);
						sort_ids = (SortID *)memrealloc(sort_ids, sizeof(SortID) * sort_id_max);
						if (ids.size()) {
							ids.resize(sort_id_max);
							RD::get_singleton()->free(items_buffer);
							items_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * sort_id_max);
						}
					}

					sort_ids[sort_id_count].cell_index = offset;
					sort_ids[sort_id_count].item_index = item.index;
					sort_ids[sort_id_count].item_type = item.type;

					sort_id_count++;

					//for now, only count
					cluster_data_ptr[offset].item_pointers[item.type]++;
					//print_line("at offset " + itos(offset) + " value: " + itos(cluster_data_ptr[offset].item_pointers[item.type]));
				}
			}
		}
	}

	/* Step 2, Assign pointers (and reset counters) */

	uint32_t offset = 0;
	for (uint32_t i = 0; i < (width * height * depth); i++) {
		for (int j = 0; j < ITEM_TYPE_MAX; j++) {
			uint32_t count = cluster_data_ptr[i].item_pointers[j]; //save count
			cluster_data_ptr[i].item_pointers[j] = offset; //replace count by pointer
			offset += count; //increase offset by count;
		}
	}

	//print_line("offset: " + itos(offset));
	/* Step 3, Place item lists */

	uint32_t *ids_ptr = ids.ptrw();

	for (uint32_t i = 0; i < sort_id_count; i++) {
		const SortID &id = sort_ids[i];
		Cell &cell = cluster_data_ptr[id.cell_index];
		uint32_t pointer = cell.item_pointers[id.item_type] & POINTER_MASK;
		uint32_t counter = cell.item_pointers[id.item_type] >> COUNTER_SHIFT;
		ids_ptr[pointer + counter] = id.item_index;

		cell.item_pointers[id.item_type] = pointer | ((counter + 1) << COUNTER_SHIFT);
	}

	RD::get_singleton()->texture_update(cluster_texture, 0, cluster_data, true);
	RD::get_singleton()->buffer_update(items_buffer, 0, offset * sizeof(uint32_t), ids_ptr, true);
}

void LightClusterBuilder::setup(uint32_t p_width, uint32_t p_height, uint32_t p_depth) {
	if (width == p_width && height == p_height && depth == p_depth) {
		return;
	}
	if (cluster_texture.is_valid()) {
		RD::get_singleton()->free(cluster_texture);
	}

	width = p_width;
	height = p_height;
	depth = p_depth;

	cluster_data.resize(width * height * depth * sizeof(Cell));

	{
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
		tf.type = RD::TEXTURE_TYPE_3D;
		tf.width = width;
		tf.height = height;
		tf.depth = depth;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;

		cluster_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}
}

RID LightClusterBuilder::get_cluster_texture() const {
	return cluster_texture;
}

RID LightClusterBuilder::get_cluster_indices_buffer() const {
	return items_buffer;
}

LightClusterBuilder::LightClusterBuilder() {
	//initialize accumulators to something
	lights = (LightData *)memalloc(sizeof(LightData) * 1024);
	light_max = 1024;

	refprobes = (OrientedBoxData *)memalloc(sizeof(OrientedBoxData) * 1024);
	refprobe_max = 1024;

	decals = (OrientedBoxData *)memalloc(sizeof(OrientedBoxData) * 1024);
	decal_max = 1024;

	items = (Item *)memalloc(sizeof(Item) * 1024);
	item_max = 1024;

	sort_ids = (SortID *)memalloc(sizeof(SortID) * 1024);
	ids.resize(2014);
	items_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 1024);
	item_max = 1024;
}

LightClusterBuilder::~LightClusterBuilder() {
	if (cluster_data.size()) {
		RD::get_singleton()->free(cluster_texture);
	}

	if (lights) {
		memfree(lights);
	}
	if (refprobes) {
		memfree(refprobes);
	}
	if (decals) {
		memfree(decals);
	}
	if (items) {
		memfree(items);
	}
	if (sort_ids) {
		memfree(sort_ids);
		RD::get_singleton()->free(items_buffer);
	}
}
