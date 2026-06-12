/*
This is C# simulation Runtime
- imports JS interop functions
- understands the memory Contract
- reads entity bytes in bulk
- simulates Physics
- writes the updated bytes back
*/
using System;
using System.Diagnostics;   //stopwatch
using System.Threading.Tasks; //or async initialization
using System.Runtime.InteropServices; //provides memory marshal to reinterpret bytes as floats
using System.Runtime.InteropServices.JavaScript; //dotnet to js interop laer

//interop bridge
public static partial class Host
{
    //imports Js function interop.bulkRead - copy bytes from shared WASM memory into managed array - has source offset, destination array and number of bytes - this replaces thousands of tiny reads so its much faster
    [JSImport("bulkRead", "interop")] public static partial void BulkRead(int srcOffset, byte[] destArray, int length);
    //imports interop.bulkWrite
    [JSImport("bulkWrite", "interop")] public static partial void BulkWrite(byte[] srcArray, int destOffset, int length);

    [JSExport]
    //loads js module dynamically so that later bulkRead and bulkWrite exist then logs
    public static async Task InitInterop()
    {
        await JSHost.ImportAsync("interop", "/cs/interop.js");
        //Console.WriteLine("C# interop imported (bulk version)");
    }
}

//memory schema
public static class Contract
{
    public const int EntityCount = 1000; //number of entities
    public const int EntitiesOffset = 0x1000; //start
    public const int EntityStride = 16; //each entity is 16 bytes
    public const int EntityFieldCount = 4; //contains 4 floats
    //field offsets
    public const int XOffset = 0;
    public const int YOffset = 4;
    public const int VXOffset = 8;
    public const int VYOffset = 12;

}


//contains the simulation
public static partial class GameLogic
{
    private static byte[] _entityBuffer; //managed bte buffer which stores copied entity memory, its reused in each frame
    private static int _frameCounter = 0; //tracks frames for wind calculation
    private static uint _randState = 123456789u; //RNG state

    //simulation bounds
    private const float ScreenWidth = 1024.0f; //
    private const float ScreenHeight = 768.0f;
    private const float Gravity = 500.0f; //acceleration downward
    private const float Damping = 0.995f; //velocity decay
    private const float BounceDamping = 0.9f; //lose energy on collision
    private const float WindStrength = 50.0f; //horizontal force
    private const float TurbulenceStrength = 30.0f; //random force
    private const float MaxSpeed = 800.0f; //velocity clamp

    
    //initializes simulation state
    [JSExport]
    public static void InitEntities()
    {
        uint rng = 0x12345678u; //seed
        int byteSize = Contract.EntityCount * Contract.EntityStride;//computes total bytes

        //ensures buffer exists
        if (_entityBuffer == null || _entityBuffer.Length < byteSize)
            _entityBuffer = new byte[byteSize];

        //reinterpret bytes as floats
        Span<float> fields = MemoryMarshal.Cast<byte, float>(_entityBuffer.AsSpan(0, byteSize));

        //loop entities
        for (int i = 0; i < Contract.EntityCount; i++)
        {
            uint NextU32() //same as the one at c++
            {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                return rng;
            }
            int baseIdx = i * Contract.EntityFieldCount; //computes float index which is then randomized
            fields[baseIdx + 0] = NextU32() % 1024u;                 // x
            fields[baseIdx + 1] = NextU32() % 768u;                  // y
            fields[baseIdx + 2] = (int)(NextU32() % 301u) - 150;    // vx
            fields[baseIdx + 3] = (int)(NextU32() % 301u) - 150;    // vy
        }

        // Write the whole buffer to shared memory
        Host.BulkWrite(_entityBuffer, Contract.EntitiesOffset, byteSize);
        //Console.WriteLine($"[C#] Initialised {Contract.EntityCount} entities.");
    }

    //per frame simulation
    [JSExport]
    public static void UpdateEntities(int entityBaseOffset, int entityCount, float dtSeconds)
    {
        float dt = dtSeconds > 0.033f ? 0.033f : dtSeconds; //clamps to 30 fps max timestep to  prevent unstable phsics if frame stalls
        int totalBytes = entityCount * Contract.EntityStride; //computes bytes

        //ensures bytes
        if (_entityBuffer == null || _entityBuffer.Length < totalBytes)
            _entityBuffer = new byte[totalBytes];

        //copiies bytes from shared memory
        Host.BulkRead(entityBaseOffset, _entityBuffer, totalBytes);
        Span<float> fields = MemoryMarshal.Cast<byte, float>(_entityBuffer.AsSpan(0, totalBytes)); //reinterpret as floats

        //reads fields and applies physics
        for (int i = 0; i < entityCount; i++)
        {
            int baseIdx = i * Contract.EntityFieldCount;
            float x  = fields[baseIdx + 0];
            float y  = fields[baseIdx + 1];
            float vx = fields[baseIdx + 2];
            float vy = fields[baseIdx + 3];

            // Physics
            float wind = (((_frameCounter + i) * 1103515245u) & 1023u) / 1023.0f;
            wind = (wind - 0.5f) * 2.0f * WindStrength;
            float turbX = (NextRandomFloat() - 0.5f) * TurbulenceStrength;
            float turbY = (NextRandomFloat() - 0.5f) * TurbulenceStrength;

            vx += (wind + turbX) * dt;
            vy += (Gravity + turbY) * dt;
            x += vx * dt;
            y += vy * dt;

            if (x < 0.0f) { x = -x; vx = -vx * BounceDamping; }
            else if (x > ScreenWidth) { x = 2.0f * ScreenWidth - x; vx = -vx * BounceDamping; }
            if (y < 0.0f) { y = -y; vy = -vy * BounceDamping; }
            else if (y > ScreenHeight) { y = 2.0f * ScreenHeight - y; vy = -vy * BounceDamping; }

            vx *= Damping; vy *= Damping;
            if (vx > MaxSpeed) vx = MaxSpeed; else if (vx < -MaxSpeed) vx = -MaxSpeed;
            if (vy > MaxSpeed) vy = MaxSpeed; else if (vy < -MaxSpeed) vy = -MaxSpeed;

            fields[baseIdx + 0] = x;
            fields[baseIdx + 1] = y;
            fields[baseIdx + 2] = vx;
            fields[baseIdx + 3] = vy;
        }
        //write updated bytes back, one op
        Host.BulkWrite(_entityBuffer, entityBaseOffset, totalBytes);


        _frameCounter++;
        
    }

    private static float NextRandomFloat()
    {
        uint x = _randState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        _randState = x;
        return (x & 0xFFFF) / 65535.0f;
    }
}