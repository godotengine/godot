/*
 *AABB bounding box
 *Bouding Volume Hierarchy
 */
class BoundingBox {
  float min_x, min_y, min_z, max_x, max_y, max_z;
  PVector center;
  BoundingBox() {
    min_x = Float.POSITIVE_INFINITY;
    min_y = Float.POSITIVE_INFINITY;
    min_z = Float.POSITIVE_INFINITY;
    max_x = Float.NEGATIVE_INFINITY;
    max_y = Float.NEGATIVE_INFINITY;
    max_z = Float.NEGATIVE_INFINITY;
    center = new PVector();
  }
  // build a bounding box for a triangle
  void create(Triangle t) {
    min_x = min(t.p1.x, min(t.p2.x, t.p3.x));
    max_x = max(t.p1.x, max(t.p2.x, t.p3.x));

    min_y = min(t.p1.y, min(t.p2.y, t.p3.y));
    max_y = max(t.p1.y, max(t.p2.y, t.p3.y));

    min_z = min(t.p1.z, min(t.p2.z, t.p3.z));
    max_z = max(t.p1.z, max(t.p2.z, t.p3.z));
    center.x = (max_x + min_x) / 2;
    center.y = (max_y + min_y) / 2;
    center.z = (max_z + min_z) / 2;
  }
  // merge two bounding boxes
  void add(BoundingBox bbx) {
    min_x = min(min_x, bbx.min_x);
    min_y = min(min_y, bbx.min_y);
    min_z = min(min_z, bbx.min_z);

    max_x = max(max_x, bbx.max_x);
    max_y = max(max_y, bbx.max_y);
    max_z = max(max_z, bbx.max_z);
    center.x = (max_x + min_x) / 2;
    center.y = (max_y + min_y) / 2;
    center.z = (max_z + min_z) / 2;
  }
  // get bounding box center axis value
  float getCenterAxisValue(int axis) {
    if (axis == 1) {
      return center.x;
    } else if (axis == 2) {
      return center.y;
    }
    // when axis == 3
    return center.z;
  }
  // check if a ray is intersected with the bounding box
  boolean intersect(Ray r) {
    float tmin, tmax;
    if (r.dir.x >= 0) {
      tmin = (min_x - r.ori.x) * (1.0f / r.dir.x);
      tmax = (max_x - r.ori.x) * (1.0f / r.dir.x);
    } else {
      tmin = (max_x - r.ori.x) * (1.0f / r.dir.x);
      tmax = (min_x - r.ori.x) * (1.0f / r.dir.x);
    }

    float tymin, tymax;
    if (r.dir.y >= 0) {
      tymin = (min_y - r.ori.y) * (1.0f / r.dir.y);
      tymax = (max_y - r.ori.y) * (1.0f / r.dir.y);
    } else {
      tymin = (max_y - r.ori.y) * (1.0f / r.dir.y);
      tymax = (min_y - r.ori.y) * (1.0f / r.dir.y);
    }

    if (tmax < tymin || tymax < tmin) {
      return false;
    }

    tmin = tmin < tymin ? tymin : tmin;
    tmax = tmax > tymax ? tymax : tmax;

    float tzmin, tzmax;
    if (r.dir.z >= 0) {
      tzmin = (min_z - r.ori.z) * (1.0f / r.dir.z);
      tzmax = (max_z - r.ori.z) * (1.0f / r.dir.z);
    } else {
      tzmin = (max_z - r.ori.z) * (1.0f / r.dir.z);
      tzmax = (min_z - r.ori.z) * (1.0f / r.dir.z);
    }
    if (tmax < tzmin || tmin > tzmax) {
      return false;
    }
    return true;
  }
}
// Bounding Volume Hierarchy
class BVH {
  // Binary Tree
  BVH left, right;
  BoundingBox overall_bbx;
  ArrayList<Triangle> mesh;
  BVH(ArrayList<Triangle> mesh) {
    this.mesh = mesh;
    overall_bbx = new BoundingBox();
    left = null;
    right = null;
    int mesh_size = this.mesh.size();
    if (mesh_size <= 1) {
      return;
    }
    // random select an axis
    int axis = int(random(100)) % 3 + 1;
    // build bounding box and save the selected center component
    float[] axis_values = new float[mesh_size];
    for (int i = 0; i < mesh_size; i++) {
      Triangle t = this.mesh.get(i);
      overall_bbx.add(t.bbx);
      axis_values[i] = t.bbx.getCenterAxisValue(axis);
    }
    // find the median value of selected center component as pivot
    axis_values = sort(axis_values);
    float pivot;
    if (mesh_size % 2 == 1) {
      pivot = axis_values[mesh_size / 2];
    } else {
      pivot =
          0.5f * (axis_values[mesh_size / 2 - 1] + axis_values[mesh_size / 2]);
    }
    // Build left node and right node by partitioning the mesh based on triangle
    // bounding box center component value
    ArrayList<Triangle> left_mesh = new ArrayList<Triangle>();
    ArrayList<Triangle> right_mesh = new ArrayList<Triangle>();
    for (int i = 0; i < mesh_size; i++) {
      Triangle t = this.mesh.get(i);
      if (t.bbx.getCenterAxisValue(axis) < pivot) {
        left_mesh.add(t);
      } else if (t.bbx.getCenterAxisValue(axis) > pivot) {
        right_mesh.add(t);
      } else if (left_mesh.size() < right_mesh.size()) {
        left_mesh.add(t);
      } else {
        right_mesh.add(t);
      }
    }
    left = new BVH(left_mesh);
    right = new BVH(right_mesh);
  }
  // check if a ray intersect with current volume
  boolean intersect(Ray r, float[] param) {
    if (mesh.size() == 0) {
      return false;
    }
    if (mesh.size() == 1) {
      Triangle t = mesh.get(0);
      return t.intersect(r, param);
    }
    if (!overall_bbx.intersect(r)) {
      return false;
    }
    boolean left_res = left.intersect(r, param);
    boolean right_res = right.intersect(r, param);
    return left_res || right_res;
  }
}
