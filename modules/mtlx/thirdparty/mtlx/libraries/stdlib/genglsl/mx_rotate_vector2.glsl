void mx_rotate_vector2(vec2 _in, float amount, out vec2 result)
{
    float rotationRadians = radians(amount);
    float sa = sin(rotationRadians);
    float ca = cos(rotationRadians);
    result = vec2(ca*_in.x + sa*_in.y, -sa*_in.x + ca*_in.y);
}
