/*
This mirrors C++ version
It defines the data structures needed
*/
using System;


public struct Vector2
{
    public float X, Y;
    public Vector2(float x, float y) { X = x; Y = y; }
}

public struct Vector2i
{
    public int X, Y;
    public Vector2i(int x, int y) { X = x; Y = y; }
}

public struct Vector3
{
    public float X, Y, Z;
    public Vector3(float x, float y, float z) { X = x; Y = y; Z = z; }
}

public struct Vector3i
{
    public int X, Y, Z;
    public Vector3i(int x, int y, int z) { X = x; Y = y; Z = z; }
}

public struct Vector4
{
    public float X, Y, Z, W;
    public Vector4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }
}

public struct Vector4i
{
    public int X, Y, Z, W;
    public Vector4i(int x, int y, int z, int w) { X = x; Y = y; Z = z; W = w; }
}

// color types

public struct Color
{
    public float R, G, B, A;
    public Color(float r, float g, float b, float a = 1.0f) { R = r; G = g; B = b; A = a; }
}

//2D axis‑aligned bounding box 
//layout is determined by the field order
public struct Rect2
{
    private Vector2 _position;
    private Vector2 _size;

    public Vector2 Position { readonly get => _position; set => _position = value; }
    public Vector2 Size     { readonly get => _size;     set => _size = value; }

    public Rect2(Vector2 position, Vector2 size)
    {
        _position = position;
        _size = size;
    }

    public Rect2(float x, float y, float width, float height)
        : this(new Vector2(x, y), new Vector2(width, height)) { }
}

public struct Rect2i
{
    private Vector2i _position;
    private Vector2i _size;

    public Vector2i Position { readonly get => _position; set => _position = value; }
    public Vector2i Size     { readonly get => _size;     set => _size = value; }

    public Rect2i(Vector2i position, Vector2i size)
    {
        _position = position;
        _size = size;
    }

    public Rect2i(int x, int y, int width, int height)
        : this(new Vector2i(x, y), new Vector2i(width, height)) { }
}

//2D transform (2×3 matrix: three Vector2 columns) 

public struct Transform2D
{
    public Vector2 X;
    public Vector2 Y;
    public Vector2 Origin;

    public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 origin)
    {
        X = xAxis;
        Y = yAxis;
        Origin = origin;
    }

    // Convenience: rotation (radians) + translation only
    public Transform2D(float rotation, Vector2 position)
    {
        X = new Vector2(MathF.Cos(rotation), MathF.Sin(rotation));
        Y = new Vector2(-MathF.Sin(rotation), MathF.Cos(rotation));
        Origin = position;
    }
}

// 3D transform (3×4 matrix: three Vector3 basis columns + origin) 
// Changed the naming
public struct Basis
{
    public Vector3 Column0;
    public Vector3 Column1;
    public Vector3 Column2;

    public Basis(Vector3 column0, Vector3 column1, Vector3 column2)
    {
        Column0 = column0;
        Column1 = column1;
        Column2 = column2;
    }
}

public struct Transform3D
{
    public Basis Basis;
    public Vector3 Origin;

    public Transform3D(Basis basis, Vector3 origin)
    {
        Basis = basis;
        Origin = origin;
    }
}

//Quaternion 

public struct Quaternion
{
    public float W, X, Y, Z;

    public Quaternion(float w, float x, float y, float z)
    {
        W = w; X = x; Y = y; Z = z;
    }
}

//3D axis‑aligned bounding box 

public struct AABB
{
    private Vector3 _position;
    private Vector3 _size;

    public Vector3 Position { readonly get => _position; set => _position = value; }
    public Vector3 Size     { readonly get => _size;     set => _size = value; }

    public AABB(Vector3 position, Vector3 size)
    {
        _position = position;
        _size = size;
    }
}

//Plane

public struct Plane
{
    public Vector3 Normal;
    public float D;

    public Plane(float d, Vector3 normal)
    {
        D = d;
        Normal = normal;
    }

    public Plane(Vector3 normal, float d)
    {
        Normal = normal;
        D = d;
    }
}

//Projection (4×4 matrix: four Vector4 columns)
public struct Projection
{
    public Vector4 X, Y, Z, W;

    public Projection(Vector4 x, Vector4 y, Vector4 z, Vector4 w)
    {
        X = x; Y = y; Z = z; W = w;
    }
}