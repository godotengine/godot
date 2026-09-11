class Camera {
  // camera's field of view
  float fov;
  // camera's position, look at point and axis
  PVector pos, center, axis;
  PVector init_pos, init_center, init_axis;
  float move_speed;
  float rot_speed;
  Camera(float fov, PVector pos, PVector center, PVector axis) {
    this.fov = fov;
    this.pos = pos;
    this.center = center;
    this.axis = axis;
    this.axis.normalize();
    move_speed = 0.001;
    rot_speed = 0.01 * PI;
    init_pos = pos.copy();
    init_center = center.copy();
    init_axis = axis.copy();
  }

  Camera copy() {
    Camera cam = new Camera(fov, pos.copy(), center.copy(), axis.copy());
    return cam;
  }

  PVector project(PVector pos) {
    PVector proj = MatxVec3(getCameraMat(), PVector.sub(pos, this.pos));
    proj.x = (float)height / 2.0 * proj.x / proj.z / tan(fov / 2.0f);
    proj.y = (float)height / 2.0 * proj.y / proj.z / tan(fov / 2.0f);
    proj.z = proj.z;
    return proj;
  }

  float[] getCameraMat() {
    float[] mat = new float[9];
    PVector dir = PVector.sub(center, pos);
    dir.normalize();
    PVector left = dir.cross(axis);
    left.normalize();
    // processing camera system does not follow right hand rule
    mat[0] = -left.x;
    mat[1] = -left.y;
    mat[2] = -left.z;
    mat[3] = axis.x;
    mat[4] = axis.y;
    mat[5] = axis.z;
    mat[6] = dir.x;
    mat[7] = dir.y;
    mat[8] = dir.z;

    return mat;
  }

  void run() {
    PVector dir, left;
    if (mousePressed) {
      float angleX = (float)mouseX / width * PI - PI / 2;
      float angleY = (float)mouseY / height * PI - PI;
      PVector diff = PVector.sub(center, pos);
      float radius = diff.mag();
      pos.x = radius * sin(angleY) * sin(angleX) + center.x;
      pos.y = radius * cos(angleY) + center.y;
      pos.z = radius * sin(angleY) * cos(angleX) + center.z;
      dir = PVector.sub(center, pos);
      dir.normalize();
      PVector up = new PVector(0, 1, 0);
      left = up.cross(dir);
      left.normalize();
      axis = dir.cross(left);
      axis.normalize();
    }

    if (keyPressed) {
      switch (key) {
        case 'w':
          dir = PVector.sub(center, pos);
          dir.normalize();
          pos = PVector.add(pos, PVector.mult(dir, move_speed));
          center = PVector.add(center, PVector.mult(dir, move_speed));
          break;
        case 's':
          dir = PVector.sub(center, pos);
          dir.normalize();
          pos = PVector.sub(pos, PVector.mult(dir, move_speed));
          center = PVector.sub(center, PVector.mult(dir, move_speed));
          break;
        case 'a':
          dir = PVector.sub(center, pos);
          dir.normalize();
          left = axis.cross(dir);
          left.normalize();
          pos = PVector.add(pos, PVector.mult(left, move_speed));
          center = PVector.add(center, PVector.mult(left, move_speed));
          break;
        case 'd':
          dir = PVector.sub(center, pos);
          dir.normalize();
          left = axis.cross(dir);
          left.normalize();
          pos = PVector.sub(pos, PVector.mult(left, move_speed));
          center = PVector.sub(center, PVector.mult(left, move_speed));
          break;
        case 'r':
          dir = PVector.sub(center, pos);
          dir.normalize();
          float[] mat = getRotationMat3x3(rot_speed, dir.x, dir.y, dir.z);
          axis = MatxVec3(mat, axis);
          axis.normalize();
          break;
        case 'b':
          pos = init_pos.copy();
          center = init_center.copy();
          axis = init_axis.copy();
          break;
        case '+': move_speed *= 2.0f; break;
        case '-': move_speed /= 2.0; break;
        case CODED:
          if (keyCode == UP) {
            pos = PVector.add(pos, PVector.mult(axis, move_speed));
            center = PVector.add(center, PVector.mult(axis, move_speed));
          } else if (keyCode == DOWN) {
            pos = PVector.sub(pos, PVector.mult(axis, move_speed));
            center = PVector.sub(center, PVector.mult(axis, move_speed));
          }
      }
    }
  }
  void open() {
    perspective(fov, float(width) / height, 1e-6, 1e5);
    camera(pos.x, pos.y, pos.z, center.x, center.y, center.z, axis.x, axis.y,
           axis.z);
  }
  void close() {
    ortho(-width, 0, -height, 0);
    camera(0, 0, 0, 0, 0, 1, 0, 1, 0);
  }
}
