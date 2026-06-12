// Triangle
class Triangle {
  // position
  PVector p1, p2, p3;
  // color
  color c1, c2, c3;
  BoundingBox bbx;
  Triangle(PVector p1, PVector p2, PVector p3, color c1, color c2, color c3) {
    this.p1 = p1;
    this.p2 = p2;
    this.p3 = p3;
    this.c1 = c1;
    this.c2 = c2;
    this.c3 = c3;
    bbx = new BoundingBox();
    bbx.create(this);
  }
  // check to see if a ray intersects with the triangle
  boolean intersect(Ray r, float[] param) {
    PVector p21 = PVector.sub(p2, p1);
    PVector p31 = PVector.sub(p3, p1);
    PVector po1 = PVector.sub(r.ori, p1);

    PVector dxp31 = r.dir.cross(p31);
    PVector po1xp21 = po1.cross(p21);
    float denom = p21.dot(dxp31);
    float t = p31.dot(po1xp21) / denom;
    float alpha = po1.dot(dxp31) / denom;
    float beta = r.dir.dot(po1xp21) / denom;

    boolean res = t > 0 && alpha > 0 && alpha < 1 && beta > 0 && beta < 1 &&
                  alpha + beta < 1;
    // depth test
    if (res && t < param[0]) {
      param[0] = t;
      param[1] = alpha * p1.x + beta * p2.x + (1 - alpha - beta) * p3.x;
      param[2] = alpha * p1.y + beta * p2.y + (1 - alpha - beta) * p3.y;
      param[3] = alpha * p1.z + beta * p2.z + (1 - alpha - beta) * p3.z;
    }
    return res;
  }
  void render() {
    beginShape(TRIANGLES);
    fill(c1);
    vertex(p1.x, p1.y, p1.z);
    fill(c2);
    vertex(p2.x, p2.y, p2.z);
    fill(c3);
    vertex(p3.x, p3.y, p3.z);
    endShape();
  }
}
// Ray
class Ray {
  // origin and direction
  PVector ori, dir;
  Ray(PVector ori, PVector dir) {
    this.ori = ori;
    this.dir = dir;
  }
}
