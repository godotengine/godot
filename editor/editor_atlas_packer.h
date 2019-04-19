#ifndef EDITOR_ATLAS_PACKER_H
#define EDITOR_ATLAS_PACKER_H

#include "core/math/vector2.h"

#include "core/vector.h"
#include "scene/resources/bit_map.h"

class EditorAtlasPacker {
public:
	struct Chart {
		Vector<Vector2> vertices;
		struct Face {
			int vertex[3];
		};
		Vector<Face> faces;
		bool can_transpose;

		Vector2 final_offset;
		bool transposed;
	};

private:
	struct PlottedBitmap {
		int chart_index;
		Vector2i offset;
		int area;
		Vector<int> top_heights;
		Vector<int> bottom_heights;
		bool transposed;

		Vector2 final_pos;

		bool operator<(const PlottedBitmap &p_bm) const {
			return area > p_bm.area;
		}
	};

	static void _plot_triangle(Ref<BitMap> p_bitmap, Vector2i *vertices);

public:
	static void chart_pack(Vector<Chart> &charts, int &r_width, int &r_height, int p_atlas_max_size = 2048, int p_cell_resolution = 4);
};

#endif // EDITOR_ATLAS_PACKER_H
