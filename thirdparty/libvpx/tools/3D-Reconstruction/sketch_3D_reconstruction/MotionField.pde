class MotionField {
  int block_size;
  ArrayList<PVector> motion_field;
  MotionField(int block_size) {
    this.block_size = block_size;
    motion_field = new ArrayList<PVector>();
  }

  void update(Camera last_cam, Camera current_cam, PointCloud point_cloud,
              BVH bvh) {
    // clear motion field
    motion_field = new ArrayList<PVector>();
    int r_num = height / block_size, c_num = width / block_size;
    for (int i = 0; i < r_num * c_num; i++)
      motion_field.add(new PVector(0, 0, 0));
    // estimate motion vector of each point in point cloud
    for (int i = 0; i < point_cloud.size(); i++) {
      PVector p = point_cloud.getPosition(i);
      PVector p0 = current_cam.project(p);
      PVector p1 = last_cam.project(p);
      int row = int((p0.y + height / 2.0f) / block_size);
      int col = int((p0.x + width / 2.0f) / block_size);
      if (row >= 0 && row < r_num && col >= 0 && col < c_num) {
        PVector accu = motion_field.get(row * c_num + col);
        accu.x += p1.x - p0.x;
        accu.y += p1.y - p0.y;
        accu.z += 1;
      }
    }
    // if some blocks do not have point, then use ray tracing to see if they are
    // in triangles
    for (int i = 0; i < r_num; i++)
      for (int j = 0; j < c_num; j++) {
        PVector accu = motion_field.get(i * c_num + j);
        if (accu.z > 0) {
          continue;
        }
        // use the center of the block to generate view ray
        float cx = j * block_size + block_size / 2.0f - width / 2.0f;
        float cy = i * block_size + block_size / 2.0f - height / 2.0f;
        float cz = 0.5f * height / tan(current_cam.fov / 2.0f);
        PVector dir = new PVector(cx, cy, cz);
        float[] camMat = current_cam.getCameraMat();
        dir = MatxVec3(transpose3x3(camMat), dir);
        dir.normalize();
        Ray r = new Ray(current_cam.pos, dir);
        // ray tracing
        float[] param = new float[4];
        param[0] = Float.POSITIVE_INFINITY;
        if (bvh.intersect(r, param)) {
          PVector p = new PVector(param[1], param[2], param[3]);
          PVector p0 = current_cam.project(p);
          PVector p1 = last_cam.project(p);
          accu.x += p1.x - p0.x;
          accu.y += p1.y - p0.y;
          accu.z += 1;
        }
      }
    // estimate the motion vector of each block
    for (int i = 0; i < r_num * c_num; i++) {
      PVector mv = motion_field.get(i);
      if (mv.z > 0) {
        motion_field.set(i, new PVector(mv.x / mv.z, mv.y / mv.z, 0));
      } else  // there is nothing in the block, use -1 to mark it.
      {
        motion_field.set(i, new PVector(0.0, 0.0, -1));
      }
    }
  }

  void render() {
    int r_num = height / block_size, c_num = width / block_size;
    for (int i = 0; i < r_num; i++)
      for (int j = 0; j < c_num; j++) {
        PVector mv = motion_field.get(i * c_num + j);
        float ox = j * block_size + 0.5f * block_size;
        float oy = i * block_size + 0.5f * block_size;
        stroke(255, 0, 0);
        line(ox, oy, ox + mv.x, oy + mv.y);
      }
  }

  void save(String path) {
    int r_num = height / block_size;
    int c_num = width / block_size;
    String[] mvs = new String[r_num];
    for (int i = 0; i < r_num; i++) {
      mvs[i] = "";
      for (int j = 0; j < c_num; j++) {
        PVector mv = motion_field.get(i * c_num + j);
        if (mv.z != -1) {
          mvs[i] += str(mv.x) + "," + str(mv.y);
        } else  // there is nothing
        {
          mvs[i] += "-,-";
        }
        if (j != c_num - 1) mvs[i] += ";";
      }
    }
    saveStrings(path, mvs);
  }
}
