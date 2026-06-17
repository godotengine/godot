/*
using System;
using System.Threading.Tasks;
using Array = System.Array;
using Godot;

public class DotRunner : Node
{
    public DotRunner(ulong id) : base(id)
    {
        _rngState = _rngSeed;
    }

    private const int ScreenWidth = 1024;
    private const int ScreenHeight = 768;

    private const float Gravity = 50.0f;
    private const float Damping = 0.99f;
    private const float BounceDamping = 0.85f;
    private const float TurbulenceStrength = 800.0f;

    private const int FloatsPerInstance = 8;

    private readonly uint _rngSeed = 0x12345678u;
    private uint _rngState;

    private int _dotCount;
    private Vector2[] _positions = Array.Empty<Vector2>();
    private Vector2[] _velocities = Array.Empty<Vector2>();
    private float[] _buffer = Array.Empty<float>();

    private MultiMesh? _multiMesh;

    public override void _Ready()
    {
        MultiMeshInstance2D dotsInstance = this.GetNode<MultiMeshInstance2D>("Dots");
        GodotObject multiMeshObj = dotsInstance.GetMultimesh();

        if (multiMeshObj == null || multiMeshObj.Id == 0)
            return;

        _multiMesh = new MultiMesh(multiMeshObj.Id);
        _multiMesh.SetTransformFormat(0);
        _multiMesh.SetUseColors(false);
        _multiMesh.SetUseCustomData(false);

        EnsureDotMesh(_multiMesh);
        InitializeDots(1000);
        _multiMesh.SetInstanceCount(_dotCount);
        PushBufferToMultiMesh();
    }

    public void Step(double delta)
    {
        StepSimulation((float)delta);
        PushBufferToMultiMesh();
    }

    private static void EnsureDotMesh(MultiMesh multiMesh)
    {
        GodotObject meshObj = multiMesh.GetMesh();
        if (meshObj != null && meshObj.Id != 0)
            return;

        QuadMesh quad = new QuadMesh();
        quad.SetSize(new Vector2(4.0f, 4.0f));
        multiMesh.SetMesh(quad);
    }

    private void InitializeDots(int count)
    {
        _dotCount = count;
        _positions = new Vector2[count];
        _velocities = new Vector2[count];
        _buffer = new float[count * FloatsPerInstance];

        _rngState = _rngSeed;

        for (int i = 0; i < count; i++)
        {
            float x = NextRandomFloat() * ScreenWidth;
            float y = NextRandomFloat() * ScreenHeight;

            float vx = (NextRandomFloat() - 0.5f) * 800.0f;
            float vy = (NextRandomFloat() - 0.5f) * 800.0f;

            _positions[i] = new Vector2(x, y);
            _velocities[i] = new Vector2(vx, vy);
        }
    }

    private void StepSimulation(float dt)
    {
        float w = ScreenWidth;
        float h = ScreenHeight;

        for (int i = 0; i < _dotCount; i++)
        {
            Vector2 p = _positions[i];
            Vector2 v = _velocities[i];

            float turbX = (NextRandomFloat() - 0.5f) * TurbulenceStrength;
            float turbY = (NextRandomFloat() - 0.5f) * TurbulenceStrength;

            v.X += turbX * dt;
            v.Y += (Gravity + turbY) * dt;
            v.X *= Damping;
            v.Y *= Damping;

            p.X += v.X * dt;
            p.Y += v.Y * dt;

            if (p.X < 0)       { p.X = 0; v.X = -v.X * BounceDamping; }
            else if (p.X > w)  { p.X = w; v.X = -v.X * BounceDamping; }

            if (p.Y < 0)       { p.Y = 0; v.Y = -v.Y * BounceDamping; }
            else if (p.Y > h)  { p.Y = h; v.Y = -v.Y * BounceDamping; }

            _positions[i] = p;
            _velocities[i] = v;
        }
    }

    private void PushBufferToMultiMesh()
    {
        if (_multiMesh == null) return;

        const float s = 4.0f;

        for (int i = 0; i < _dotCount; i++)
        {
            int b = i * FloatsPerInstance;
            Vector2 p = _positions[i];

            _buffer[b + 0] = s;    _buffer[b + 1] = 0.0f; _buffer[b + 2] = 0.0f; _buffer[b + 3] = p.X;
            _buffer[b + 4] = 0.0f; _buffer[b + 5] = s;    _buffer[b + 6] = 0.0f; _buffer[b + 7] = p.Y;
        }
        Console.WriteLine($"[PushBuffer] dotCount={_dotCount} buf[0..7] = {_buffer[0]}, {_buffer[1]}, {_buffer[2]}, {_buffer[3]}, {_buffer[4]}, {_buffer[5]}, {_buffer[6]}, {_buffer[7]}");

        _multiMesh.Buffer = _buffer;
    }

    private float NextRandomFloat()
    {
        _rngState ^= _rngState << 13;
        _rngState ^= _rngState >> 17;
        _rngState ^= _rngState << 5;
        return (_rngState & 0xFFFFFF) / 16777215.0f;
    }
}

 */
