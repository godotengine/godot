
#include "Vector2.h"

namespace msdfgen {

Vector2::Vector2(double val) : x(val), y(val) { }

Vector2::Vector2(double x, double y) : x(x), y(y) { }

void Vector2::reset() {
    x = 0, y = 0;
}

void Vector2::set(double x, double y) {
    Vector2::x = x, Vector2::y = y;
}

double Vector2::length() const {
    return sqrt(x*x+y*y);
}

double Vector2::direction() const {
    return atan2(y, x);
}

Vector2 Vector2::normalize(bool allowZero) const {
    double len = length();
    if (len == 0)
        return Vector2(0, !allowZero);
    return Vector2(x/len, y/len);
}

Vector2 Vector2::getOrthogonal(bool polarity) const {
    return polarity ? Vector2(-y, x) : Vector2(y, -x);
}

Vector2 Vector2::getOrthonormal(bool polarity, bool allowZero) const {
    double len = length();
    if (len == 0)
        return polarity ? Vector2(0, !allowZero) : Vector2(0, -!allowZero);
    return polarity ? Vector2(-y/len, x/len) : Vector2(y/len, -x/len);
}

Vector2 Vector2::project(const Vector2 &vector, bool positive) const {
    Vector2 n = normalize(true);
    double t = dotProduct(vector, n);
    if (positive && t <= 0)
        return Vector2();
    return t*n;
}

Vector2::operator const void*() const {
    return x || y ? this : NULL;
}

bool Vector2::operator!() const {
    return !x && !y;
}

bool Vector2::operator==(const Vector2 &other) const {
    return x == other.x && y == other.y;
}

bool Vector2::operator!=(const Vector2 &other) const {
    return x != other.x || y != other.y;
}

Vector2 Vector2::operator+() const {
    return *this;
}

Vector2 Vector2::operator-() const {
    return Vector2(-x, -y);
}

Vector2 Vector2::operator+(const Vector2 &other) const {
    return Vector2(x+other.x, y+other.y);
}

Vector2 Vector2::operator-(const Vector2 &other) const {
    return Vector2(x-other.x, y-other.y);
}

Vector2 Vector2::operator*(const Vector2 &other) const {
    return Vector2(x*other.x, y*other.y);
}

Vector2 Vector2::operator/(const Vector2 &other) const {
    return Vector2(x/other.x, y/other.y);
}

Vector2 Vector2::operator*(double value) const {
    return Vector2(x*value, y*value);
}

Vector2 Vector2::operator/(double value) const {
    return Vector2(x/value, y/value);
}

Vector2 & Vector2::operator+=(const Vector2 &other) {
    x += other.x, y += other.y;
    return *this;
}

Vector2 & Vector2::operator-=(const Vector2 &other) {
    x -= other.x, y -= other.y;
    return *this;
}

Vector2 & Vector2::operator*=(const Vector2 &other) {
    x *= other.x, y *= other.y;
    return *this;
}

Vector2 & Vector2::operator/=(const Vector2 &other) {
    x /= other.x, y /= other.y;
    return *this;
}

Vector2 & Vector2::operator*=(double value) {
    x *= value, y *= value;
    return *this;
}

Vector2 & Vector2::operator/=(double value) {
    x /= value, y /= value;
    return *this;
}

double dotProduct(const Vector2 &a, const Vector2 &b) {
    return a.x*b.x+a.y*b.y;
}

double crossProduct(const Vector2 &a, const Vector2 &b) {
    return a.x*b.y-a.y*b.x;
}

Vector2 operator*(double value, const Vector2 &vector) {
    return Vector2(value*vector.x, value*vector.y);
}

Vector2 operator/(double value, const Vector2 &vector) {
    return Vector2(value/vector.x, value/vector.y);
}

}
