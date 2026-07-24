/*
using System.Runtime.InteropServices.JavaScript;
using Godot;

// Entry point for the .NET WASM runtime. JS calls these [JSExport] methods
// directly; they drive the Godot scene graph from the .NET side via the
// __callGodot bridge.
public static partial class Interop
{
    static Ball?      _ball;
    static Paddle?    _left;
    static Paddle?    _right;
    static Node2D?    _pong;
    static SceneTree? _tree;
    static bool _initialized = false;

    // Called once by JS after the WASM module is ready.
    // Walks the live scene tree to grab node references; all subsequent
    // per-frame calls rely on these rather than re-querying the bridge.
        [JSExport]
    public static void RunTypeTests()
    {
        Console.WriteLine("[NET] Running type tests...");
        TypeTests.RunAll();
    }

    [JSExport]
    public static void InitInterop()
    {
        /*
        Console.WriteLine("[NET] Interop bridge ready");

        _tree = new SceneTree();
        _pong = _tree.Root.GetNode<Node2D>("Pong");
        if (_pong == null || _pong.Id == 0) return;

        _ball  = _pong.GetNode<Ball>("Ball");
        _left  = _pong.GetNode<Paddle>("Left");
        _right = _pong.GetNode<Paddle>("Right");


        var ceiling = _pong.GetNode<CeilingFloor>("Ceiling");
        var floor   = _pong.GetNode<CeilingFloor>("Floor");
        var lWall   = _pong.GetNode<Wall>("LeftWall");
        var rWall   = _pong.GetNode<Wall>("RightWall");

        _ball._Ready();
        _left._Ready();
        _right._Ready();

        Console.WriteLine("[NET] Pong initialized");
        _initialized = true;

    }

    // Called every frame by JS with the engine delta.
    // Mirrors Godot's _process() dispatch — each node gets exactly one
    // Step() call in scene order before JS returns control to the engine.
    [JSExport]
    public static void StepFrame(double delta)
    {
        /*
        if (!_initialized) return;
        _ball?.Step(delta);
        _left?.Step(delta);
        _right?.Step(delta);

    }
}
*/
