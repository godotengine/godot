#pragma once
#ifndef NV_MATH_PROXIMITYGRID_H
#define NV_MATH_PROXIMITYGRID_H

#include "Vector.h"
#include "ftoi.h"

#include "nvcore/Array.inl"


// A simple, dynamic proximity grid based on Jon's code.
// Instead of storing pointers here I store indices.

namespace nv {

    class Box;

    struct Cell {
        Array<uint> indexArray;
    };

    struct ProximityGrid {
        ProximityGrid();

        void reset();
        void init(const Array<Vector3> & pointArray);
        void init(const Box & box, uint count);

        int index_x(float x) const;
        int index_y(float y) const;
        int index_z(float z) const;
        int index(int x, int y, int z) const;
        int index(const Vector3 & pos) const;
        
        uint32 mortonCount() const;
        int mortonIndex(uint32 code) const;

        void add(const Vector3 & pos, uint key);
        bool remove(const Vector3 & pos, uint key);

        void gather(const Vector3 & pos, float radius, Array<uint> & indices);

        Array<Cell> cellArray;

        Vector3 corner;
        Vector3 invCellSize;
        int sx, sy, sz;
    };

    // For morton traversal, do:
    // for (int code = 0; code < mortonCount(); code++) {
    //   int idx = mortonIndex(code);
    //   if (idx < 0) continue;
    // }



    inline int ProximityGrid::index_x(float x) const {
        return clamp(ftoi_floor((x - corner.x) * invCellSize.x),  0, sx-1);
    }

    inline int ProximityGrid::index_y(float y) const {
        return clamp(ftoi_floor((y - corner.y) * invCellSize.y),  0, sy-1);
    }

    inline int ProximityGrid::index_z(float z) const {
        return clamp(ftoi_floor((z - corner.z) * invCellSize.z),  0, sz-1);
    }

    inline int ProximityGrid::index(int x, int y, int z) const {
        nvDebugCheck(x >= 0 && x < sx);
        nvDebugCheck(y >= 0 && y < sy);
        nvDebugCheck(z >= 0 && z < sz);
        int idx = (z * sy + y) * sx + x;
        nvDebugCheck(idx >= 0 && uint(idx) < cellArray.count());
        return idx;
    }

    inline int ProximityGrid::index(const Vector3 & pos) const {
        int x = index_x(pos.x);
        int y = index_y(pos.y);
        int z = index_z(pos.z);
        return index(x, y, z);
    }


    inline void ProximityGrid::add(const Vector3 & pos, uint key) {
        uint idx = index(pos);
        cellArray[idx].indexArray.append(key);
    }

    inline bool ProximityGrid::remove(const Vector3 & pos, uint key) {
        uint idx = index(pos);
        return cellArray[idx].indexArray.remove(key);
    }

} // nv namespace

#endif // NV_MATH_PROXIMITYGRID_H
