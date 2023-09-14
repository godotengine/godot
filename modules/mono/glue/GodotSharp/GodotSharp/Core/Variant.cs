using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot;

#nullable enable

/// <summary>
/// <para>The most important data type in Godot.</para>
/// <para>In computer programming, a Variant class is a class that is designed to store a variety of other types. Dynamic programming languages like PHP, Lua, JavaScript and GDScript like to use them to store variables' data on the backend. With these Variants, properties are able to change value types freely.</para>
/// <para><code>
/// // C# is statically typed. Once a variable has a type it cannot be changed. You can use the `var` keyword to let the compiler infer the type automatically.
/// var foo = 2; // Foo is a 32-bit integer (int). Be cautious, integers in GDScript are 64-bit and the direct C# equivalent is `long`.
/// // foo = "foo was and will always be an integer. It cannot be turned into a string!";
/// var boo = "Boo is a string!";
/// var ref = new RefCounted(); // var is especially useful when used together with a constructor.
///
/// // Godot also provides a Variant type that works like a union of all the Variant-compatible types.
/// Variant fooVar = 2; // fooVar is dynamically an integer (stored as a `long` in the Variant type).
/// fooVar = "Now fooVar is a string!";
/// fooVar = new RefCounted(); // fooVar is a GodotObject.
/// </code></para>
/// <para>Godot tracks all scripting API variables within Variants. Without even realizing it, you use Variants all the time. When a particular language enforces its own rules for keeping data typed, then that language is applying its own custom logic over the base Variant scripting API.</para>
/// <para>C# is statically typed, but uses its own implementation of the <see cref="Variant"/> type in place of Godot's Variant class when it needs to represent a dynamic value. A <see cref="Variant"/> can be assigned any compatible type implicitly but converting requires an explicit cast.</para>
/// The <see cref="VariantType"/> function returns the enumerated value of the Variant type stored in the current variable (see <see cref="Type"/>).
/// <para><code>
/// Variant foo = 2;
/// switch (foo.VariantType)
/// {
///     case Variant.Type.Nil:
///         GD.Print("foo is null");
///         break;
///     case Variant.Type.Int:
///         GD.Print("foo is an integer");
///         break;
///     case Variant.Type.Object:
///         // Note that Objects are their own special category.
///         // You can convert a Variant to a GodotObject and use reflection to get its name.
///         GD.Print($"foo is a(n) {foo.AsGodotObject().GetType().Name}");
///         break;
/// }
/// </code></para>
/// <para>A Variant takes up only 24 bytes (40 bytes in double precision builds) and can store almost any engine datatype inside of it. Variants are rarely used to hold information for long periods of time. Instead, they are used mainly for communication, editing, serialization and moving data around.</para>
/// <para>Godot has specifically invested in making its Variant class as flexible as possible; so much so that it is used for a multitude of operations to facilitate communication between all of Godot's systems.</para>
/// <para>A Variant:</para>
/// <list type="bullet">
/// <item>Can store almost any datatype.</item>
/// <item>Can perform operations between many variants.</item>
/// <item>Can be hashed, so it can be compared quickly to other variants.</item>
/// <item>Can be used to convert safely between datatypes.</item>
/// <item>Can be used to abstract calling methods and their arguments. Godot exports all its functions through variants.</item>
/// <item>Can be used to defer calls or move data between threads.</item>
/// <item>Can be serialized as binary and stored to disk, or transferred via network.</item>
/// <item>Can be serialized to text and use it for printing values and editable settings.</item>
/// <item>Can work as an exported property, so the editor can edit it universally.</item>
/// <item>Can be used for dictionaries, arrays, parsers, etc.</item>
/// </list>
/// <para><b>Containers (Array and Dictionary):</b> Both are implemented using variants. A <see cref="Collections.Dictionary"/> can match any datatype used as key to any other datatype. An <see cref="Collections.Array"/> just holds an array of Variants. Of course, a Variant can also hold a <see cref="Collections.Dictionary"/> and an <see cref="Collections.Array"/> inside, making it even more flexible.</para>
/// <para>Modifications to a container will modify all references to it. A <see cref="Mutex"/> should be created to lock it if multi-threaded access is desired.</para>
/// </summary>
[SuppressMessage("ReSharper", "RedundantNameQualifier")]
public partial struct Variant : IDisposable
{
    internal godot_variant.movable NativeVar;
    private object? _obj;
    private Disposer? _disposer;

    private sealed class Disposer : IDisposable
    {
        private godot_variant.movable _native;

        private WeakReference<IDisposable>? _weakReferenceToSelf;

        public Disposer(in godot_variant.movable nativeVar)
        {
            _native = nativeVar;
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
        }

        ~Disposer()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            _native.DangerousSelfRef.Dispose();

            if (_weakReferenceToSelf != null)
            {
                DisposablesTracker.UnregisterDisposable(_weakReferenceToSelf);
            }
        }
    }

    private Variant(in godot_variant nativeVar)
    {
        NativeVar = (godot_variant.movable)nativeVar;
        _obj = null;

        switch (nativeVar.Type)
        {
            case Type.Nil:
            case Type.Bool:
            case Type.Int:
            case Type.Float:
            case Type.Vector2:
            case Type.Vector2I:
            case Type.Rect2:
            case Type.Rect2I:
            case Type.Vector3:
            case Type.Vector3I:
            case Type.Vector4:
            case Type.Vector4I:
            case Type.Plane:
            case Type.Quaternion:
            case Type.Color:
            case Type.Rid:
                _disposer = null;
                break;
            default:
            {
                _disposer = new Disposer(NativeVar);
                break;
            }
        }
    }

    /// <summary>
    /// Converts a <see cref="godot_variant"/> to a Variant, taking ownership of the disposable value in the process.
    /// </summary>
    /// <param name="nativeValueToOwn">The godot_variant to convert.</param>
    /// <returns>A Variant representation of this godot_variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // Explicit name to make it very clear
    public static Variant CreateTakingOwnershipOfDisposableValue(in godot_variant nativeValueToOwn) =>
        new(nativeValueToOwn);

    /// <summary>
    /// Converts a <see cref="godot_variant"/> to a Variant, copying borrowed values in the process.
    /// </summary>
    /// <param name="nativeValueToOwn">The godot_variant to convert.</param>
    /// <returns>A Variant representation of this godot_variant.</returns>
    // Explicit name to make it very clear
    public static Variant CreateCopyingBorrowed(in godot_variant nativeValueToOwn) =>
        new(NativeFuncs.godotsharp_variant_new_copy(nativeValueToOwn));

    /// <summary>
    /// Constructs a new <see cref="godot_variant"/> from this instance.
    /// The caller is responsible of disposing the new instance to avoid memory leaks.
    /// </summary>
    public godot_variant CopyNativeVariant() =>
        NativeFuncs.godotsharp_variant_new_copy((godot_variant)NativeVar);

    /// <summary>
    /// Disposes of this <see cref="Variant"/>.
    /// </summary>
    public void Dispose()
    {
        _disposer?.Dispose();
        NativeVar = default;
        _obj = null;
    }

    /// <summary>
    /// The <see cref="Type"/> of this Variant.
    /// </summary>
    // TODO: Consider renaming Variant.Type to VariantType and this property to Type. VariantType would also avoid ambiguity with System.Type.
    public Type VariantType => NativeVar.DangerousSelfRef.Type;

    /// <summary>
    /// Converts this <see cref="Variant"/> to a string.
    /// </summary>
    /// <returns>A string representation of this Variant.</returns>
    public override string ToString() => AsString();

    /// <summary>
    /// The serialized internal <see cref="object"/> of this Variant.
    /// </summary>
    public object? Obj =>
        _obj ??= NativeVar.DangerousSelfRef.Type switch
        {
            Type.Bool => AsBool(),
            Type.Int => AsInt64(),
            Type.Float => AsDouble(),
            Type.String => AsString(),
            Type.Vector2 => AsVector2(),
            Type.Vector2I => AsVector2I(),
            Type.Rect2 => AsRect2(),
            Type.Rect2I => AsRect2I(),
            Type.Vector3 => AsVector3(),
            Type.Vector3I => AsVector3I(),
            Type.Transform2D => AsTransform2D(),
            Type.Vector4 => AsVector4(),
            Type.Vector4I => AsVector4I(),
            Type.Plane => AsPlane(),
            Type.Quaternion => AsQuaternion(),
            Type.Aabb => AsAabb(),
            Type.Basis => AsBasis(),
            Type.Transform3D => AsTransform3D(),
            Type.Projection => AsProjection(),
            Type.Color => AsColor(),
            Type.StringName => AsStringName(),
            Type.NodePath => AsNodePath(),
            Type.Rid => AsRid(),
            Type.Object => AsGodotObject(),
            Type.Callable => AsCallable(),
            Type.Signal => AsSignal(),
            Type.Dictionary => AsGodotDictionary(),
            Type.Array => AsGodotArray(),
            Type.PackedByteArray => AsByteArray(),
            Type.PackedInt32Array => AsInt32Array(),
            Type.PackedInt64Array => AsInt64Array(),
            Type.PackedFloat32Array => AsFloat32Array(),
            Type.PackedFloat64Array => AsFloat64Array(),
            Type.PackedStringArray => AsStringArray(),
            Type.PackedVector2Array => AsVector2Array(),
            Type.PackedVector3Array => AsVector3Array(),
            Type.PackedColorArray => AsColorArray(),
            Type.Nil => null,
            Type.Max or _ =>
                throw new InvalidOperationException($"Invalid Variant type: {NativeVar.DangerousSelfRef.Type}"),
        };

    /// <summary>
    /// Converts a <typeparamref name="T"/> to a Variant.
    /// </summary>
    /// <returns>The Variant representation of the provided <typeparamref name="T"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant From<[MustBeVariant] T>(in T from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFrom(from));

    /// <summary>
    /// Converts this <see cref="Variant"/> to a <typeparamref name="T"/>.
    /// </summary>
    /// <returns>A <typeparamref name="T"/> representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T As<[MustBeVariant] T>() =>
        VariantUtils.ConvertTo<T>(NativeVar.DangerousSelfRef);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a bool.
    /// </summary>
    /// <returns>A bool representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool AsBool() =>
        VariantUtils.ConvertToBool((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a char.
    /// </summary>
    /// <returns>A char representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public char AsChar() =>
        (char)VariantUtils.ConvertToUInt16((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an sbyte.
    /// </summary>
    /// <returns>An sbyte representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public sbyte AsSByte() =>
        VariantUtils.ConvertToInt8((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a short.
    /// </summary>
    /// <returns>A short representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public short AsInt16() =>
        VariantUtils.ConvertToInt16((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an int.
    /// </summary>
    /// <returns>An int representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AsInt32() =>
        VariantUtils.ConvertToInt32((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a long.
    /// </summary>
    /// <returns>A long representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AsInt64() =>
        VariantUtils.ConvertToInt64((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a byte.
    /// </summary>
    /// <returns>A byte representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte AsByte() =>
        VariantUtils.ConvertToUInt8((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a ushort.
    /// </summary>
    /// <returns>A ushort representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ushort AsUInt16() =>
        VariantUtils.ConvertToUInt16((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a uint.
    /// </summary>
    /// <returns>A uint representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint AsUInt32() =>
        VariantUtils.ConvertToUInt32((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a ulong.
    /// </summary>
    /// <returns>A ulong representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong AsUInt64() =>
        VariantUtils.ConvertToUInt64((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a float.
    /// </summary>
    /// <returns>A float representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float AsSingle() =>
        VariantUtils.ConvertToFloat32((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a double.
    /// </summary>
    /// <returns>A double representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double AsDouble() =>
        VariantUtils.ConvertToFloat64((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a string.
    /// </summary>
    /// <returns>A string representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string AsString() =>
        VariantUtils.ConvertToString((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector2.
    /// </summary>
    /// <returns>A Vector2 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2 AsVector2() =>
        VariantUtils.ConvertToVector2((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector2I.
    /// </summary>
    /// <returns>A Vector2I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2I AsVector2I() =>
        VariantUtils.ConvertToVector2I((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Rect2.
    /// </summary>
    /// <returns>A Rect2 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Rect2 AsRect2() =>
        VariantUtils.ConvertToRect2((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Rect2I.
    /// </summary>
    /// <returns>A Rect2I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Rect2I AsRect2I() =>
        VariantUtils.ConvertToRect2I((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Transform2D.
    /// </summary>
    /// <returns>A Transform2D representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Transform2D AsTransform2D() =>
        VariantUtils.ConvertToTransform2D((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector3.
    /// </summary>
    /// <returns>A Vector3 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3 AsVector3() =>
        VariantUtils.ConvertToVector3((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector3I.
    /// </summary>
    /// <returns>A Vector3I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3I AsVector3I() =>
        VariantUtils.ConvertToVector3I((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Basis.
    /// </summary>
    /// <returns>A Basis representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Basis AsBasis() =>
        VariantUtils.ConvertToBasis((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Quaternion.
    /// </summary>
    /// <returns>A Quaternion representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Quaternion AsQuaternion() =>
        VariantUtils.ConvertToQuaternion((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Transform3D.
    /// </summary>
    /// <returns>A Transform3D representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Transform3D AsTransform3D() =>
        VariantUtils.ConvertToTransform3D((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector4.
    /// </summary>
    /// <returns>A Vector4 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector4 AsVector4() =>
        VariantUtils.ConvertToVector4((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector4I.
    /// </summary>
    /// <returns>A Vector4I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector4I AsVector4I() =>
        VariantUtils.ConvertToVector4I((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Projection.
    /// </summary>
    /// <returns>A Projection representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Projection AsProjection() =>
        VariantUtils.ConvertToProjection((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an Aabb.
    /// </summary>
    /// <returns>An Aabb representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Aabb AsAabb() =>
        VariantUtils.ConvertToAabb((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Color.
    /// </summary>
    /// <returns>A Color representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Color AsColor() =>
        VariantUtils.ConvertToColor((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Plane.
    /// </summary>
    /// <returns>A Plane representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Plane AsPlane() =>
        VariantUtils.ConvertToPlane((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Callable.
    /// </summary>
    /// <returns>A Callable representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Callable AsCallable() =>
        VariantUtils.ConvertToCallable((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Signal.
    /// </summary>
    /// <returns>A Signal representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Signal AsSignal() =>
        VariantUtils.ConvertToSignal((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a byte[].
    /// </summary>
    /// <returns>A byte[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte[] AsByteArray() =>
        VariantUtils.ConvertAsPackedByteArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an int[].
    /// </summary>
    /// <returns>An int[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int[] AsInt32Array() =>
        VariantUtils.ConvertAsPackedInt32ArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a long[].
    /// </summary>
    /// <returns>A long[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long[] AsInt64Array() =>
        VariantUtils.ConvertAsPackedInt64ArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a float[].
    /// </summary>
    /// <returns>A float[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float[] AsFloat32Array() =>
        VariantUtils.ConvertAsPackedFloat32ArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a double[].
    /// </summary>
    /// <returns>A double[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double[] AsFloat64Array() =>
        VariantUtils.ConvertAsPackedFloat64ArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a string[].
    /// </summary>
    /// <returns>A string representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string[] AsStringArray() =>
        VariantUtils.ConvertAsPackedStringArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector2[].
    /// </summary>
    /// <returns>A Vector2[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2[] AsVector2Array() =>
        VariantUtils.ConvertAsPackedVector2ArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Vector3[].
    /// </summary>
    /// <returns>A Vector3[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3[] AsVector3Array() =>
        VariantUtils.ConvertAsPackedVector3ArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a Color[].
    /// </summary>
    /// <returns>A Color[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Color[] AsColorArray() =>
        VariantUtils.ConvertAsPackedColorArrayToSystemArray((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a GodotObject[].
    /// </summary>
    /// <returns>A GodotObject[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T[] AsGodotObjectArray<T>()
        where T : GodotObject =>
        VariantUtils.ConvertToSystemArrayOfGodotObject<T>((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a typed Godot Dictionary.
    /// </summary>
    /// <returns>A typed Godot Dictionary representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Collections.Dictionary<TKey, TValue> AsGodotDictionary<[MustBeVariant] TKey, [MustBeVariant] TValue>() =>
        VariantUtils.ConvertToDictionary<TKey, TValue>((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a typed Godot Array.
    /// </summary>
    /// <returns>A typed Godot Array representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Collections.Array<T> AsGodotArray<[MustBeVariant] T>() =>
        VariantUtils.ConvertToArray<T>((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a StringName[].
    /// </summary>
    /// <returns>A StringName[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public StringName[] AsSystemArrayOfStringName() =>
        VariantUtils.ConvertToSystemArrayOfStringName((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a NodePath[].
    /// </summary>
    /// <returns>A NodePath[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public NodePath[] AsSystemArrayOfNodePath() =>
        VariantUtils.ConvertToSystemArrayOfNodePath((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an Rid[].
    /// </summary>
    /// <returns>An Rid[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Rid[] AsSystemArrayOfRid() =>
        VariantUtils.ConvertToSystemArrayOfRid((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a GodotObject.
    /// </summary>
    /// <returns>A GodotObject representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GodotObject AsGodotObject() =>
        VariantUtils.ConvertToGodotObject((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a StringName.
    /// </summary>
    /// <returns>A StringName representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public StringName AsStringName() =>
        VariantUtils.ConvertToStringName((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to a NodePath.
    /// </summary>
    /// <returns>A NodePath representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public NodePath AsNodePath() =>
        VariantUtils.ConvertToNodePath((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an Rid.
    /// </summary>
    /// <returns>An Rid representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Rid AsRid() =>
        VariantUtils.ConvertToRid((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an untyped Godot Dictionary.
    /// </summary>
    /// <returns>An untyped Godot Dictionary representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Collections.Dictionary AsGodotDictionary() =>
        VariantUtils.ConvertToDictionary((godot_variant)NativeVar);

    /// <summary>
    /// Converts this <see cref="Variant"/> to an untyped Godot Array.
    /// </summary>
    /// <returns>An untyped Godot Array representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Collections.Array AsGodotArray() =>
        VariantUtils.ConvertToArray((godot_variant)NativeVar);

    // Explicit conversion operators to supported types

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a bool.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A bool representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator bool(Variant from) => from.AsBool();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a char.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A char representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator char(Variant from) => from.AsChar();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an sbyte.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An sbyte representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator sbyte(Variant from) => from.AsSByte();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a short.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A short representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator short(Variant from) => from.AsInt16();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an int.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An int representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator int(Variant from) => from.AsInt32();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a long.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A long representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator long(Variant from) => from.AsInt64();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a byte.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A byte representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator byte(Variant from) => from.AsByte();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a ushort.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A ushort representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator ushort(Variant from) => from.AsUInt16();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a uint.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A uint representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator uint(Variant from) => from.AsUInt32();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a ulong.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A ulong representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator ulong(Variant from) => from.AsUInt64();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a float.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A float representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator float(Variant from) => from.AsSingle();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a double.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A double representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator double(Variant from) => from.AsDouble();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a string.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A string representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator string(Variant from) => from.AsString();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector2.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector2 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector2(Variant from) => from.AsVector2();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector2I.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector2I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector2I(Variant from) => from.AsVector2I();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Rect2.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Rect2 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Rect2(Variant from) => from.AsRect2();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Rect2I.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Rect2I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Rect2I(Variant from) => from.AsRect2I();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Transform2D.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Transform2D representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Transform2D(Variant from) => from.AsTransform2D();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector3.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector3 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector3(Variant from) => from.AsVector3();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector3I.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector3I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector3I(Variant from) => from.AsVector3I();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Basis.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Basis representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Basis(Variant from) => from.AsBasis();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Quaternion.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Quaternion representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Quaternion(Variant from) => from.AsQuaternion();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Transform3D.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Transform3D representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Transform3D(Variant from) => from.AsTransform3D();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector4.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector4 representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector4(Variant from) => from.AsVector4();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector4I.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector4I representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector4I(Variant from) => from.AsVector4I();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Projection.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Projection representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Projection(Variant from) => from.AsProjection();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an Aabb.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An Aabb representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Aabb(Variant from) => from.AsAabb();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Color.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Color representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Color(Variant from) => from.AsColor();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Plane.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Plane representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Plane(Variant from) => from.AsPlane();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Callable.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Callable representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Callable(Variant from) => from.AsCallable();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Signal.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Signal representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Signal(Variant from) => from.AsSignal();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a byte[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A byte[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator byte[](Variant from) => from.AsByteArray();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an int[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An int[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator int[](Variant from) => from.AsInt32Array();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a long[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A long[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator long[](Variant from) => from.AsInt64Array();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a float[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A float[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator float[](Variant from) => from.AsFloat32Array();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a double[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A double[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator double[](Variant from) => from.AsFloat64Array();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a string[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A string[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator string[](Variant from) => from.AsStringArray();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector2[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector2[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector2[](Variant from) => from.AsVector2Array();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Vector3[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Vector3[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Vector3[](Variant from) => from.AsVector3Array();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a Color[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A Color[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Color[](Variant from) => from.AsColorArray();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a StringName[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A StringName[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator StringName[](Variant from) => from.AsSystemArrayOfStringName();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a NodePath[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A NodePath[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator NodePath[](Variant from) => from.AsSystemArrayOfNodePath();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an Rid[].
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An Rid[] representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Rid[](Variant from) => from.AsSystemArrayOfRid();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a GodotObject.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A GodotObject representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator GodotObject(Variant from) => from.AsGodotObject();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a StringName.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A StringName representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator StringName(Variant from) => from.AsStringName();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to a NodePath.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>A NodePath representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator NodePath(Variant from) => from.AsNodePath();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an Rid.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An Rid representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Rid(Variant from) => from.AsRid();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an untyped Godot Dictionary.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An untyped Godot Dictionary representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Collections.Dictionary(Variant from) => from.AsGodotDictionary();

    /// <summary>
    /// Converts the provided <see cref="Variant"/> to an untyped Godot Array.
    /// </summary>
    /// <param name="from">The Variant to convert.</param>
    /// <returns>An untyped Godot Array representation of this Variant.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static explicit operator Collections.Array(Variant from) => from.AsGodotArray();

    // While we provide implicit conversion operators, normal methods are still needed for
    // casts that are not done implicitly (e.g.: raw array to Span, enum to integer, etc).

    /// <summary>
    /// Converts the provided <see cref="bool"/> to a Variant.
    /// </summary>
    /// <param name="from">The bool to convert.</param>
    /// <returns>A Variant representation of this bool.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(bool from) => from;

    /// <summary>
    /// Converts the provided <see cref="char"/> to a Variant.
    /// </summary>
    /// <param name="from">The char to convert.</param>
    /// <returns>A Variant representation of this char.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(char from) => from;

    /// <summary>
    /// Converts the provided <see cref="sbyte"/> to a Variant.
    /// </summary>
    /// <param name="from">The sbyte to convert.</param>
    /// <returns>A Variant representation of this sbyte.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(sbyte from) => from;

    /// <summary>
    /// Converts the provided <see cref="short"/> to a Variant.
    /// </summary>
    /// <param name="from">The short to convert.</param>
    /// <returns>A Variant representation of this short.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(short from) => from;

    /// <summary>
    /// Converts the provided <see cref="int"/> to a Variant.
    /// </summary>
    /// <param name="from">The int to convert.</param>
    /// <returns>A Variant representation of this int.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(int from) => from;

    /// <summary>
    /// Converts the provided <see cref="long"/> to a Variant.
    /// </summary>
    /// <param name="from">The long to convert.</param>
    /// <returns>A Variant representation of this long.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(long from) => from;

    /// <summary>
    /// Converts the provided <see cref="byte"/> to a Variant.
    /// </summary>
    /// <param name="from">The byte to convert.</param>
    /// <returns>A Variant representation of this byte.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(byte from) => from;

    /// <summary>
    /// Converts the provided <see cref="ushort"/> to a Variant.
    /// </summary>
    /// <param name="from">The ushort to convert.</param>
    /// <returns>A Variant representation of this ushort.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(ushort from) => from;

    /// <summary>
    /// Converts the provided <see cref="uint"/> to a Variant.
    /// </summary>
    /// <param name="from">The uint to convert.</param>
    /// <returns>A Variant representation of this uint.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(uint from) => from;

    /// <summary>
    /// Converts the provided <see cref="ulong"/> to a Variant.
    /// </summary>
    /// <param name="from">The ulong to convert.</param>
    /// <returns>A Variant representation of this ulong.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(ulong from) => from;

    /// <summary>
    /// Converts the provided <see cref="float"/> to a Variant.
    /// </summary>
    /// <param name="from">The float to convert.</param>
    /// <returns>A Variant representation of this float.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(float from) => from;

    /// <summary>
    /// Converts the provided <see cref="double"/> to a Variant.
    /// </summary>
    /// <param name="from">The double to convert.</param>
    /// <returns>A Variant representation of this double.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(double from) => from;

    /// <summary>
    /// Converts the provided <see cref="string"/> to a Variant.
    /// </summary>
    /// <param name="from">The string to convert.</param>
    /// <returns>A Variant representation of this string.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(string from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector2"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector2 to convert.</param>
    /// <returns>A Variant representation of this Vector2.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Vector2 from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector2I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector2I to convert.</param>
    /// <returns>A Variant representation of this Vector2I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Vector2I from) => from;

    /// <summary>
    /// Converts the provided <see cref="Rect2"/> to a Variant.
    /// </summary>
    /// <param name="from">The Rect2 to convert.</param>
    /// <returns>A Variant representation of this Rect2.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Rect2 from) => from;

    /// <summary>
    /// Converts the provided <see cref="Rect2I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Rect2I to convert.</param>
    /// <returns>A Variant representation of this Rect2I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Rect2I from) => from;

    /// <summary>
    /// Converts the provided <see cref="Transform2D"/> to a Variant.
    /// </summary>
    /// <param name="from">The Transform2D to convert.</param>
    /// <returns>A Variant representation of this Transform2D.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Transform2D from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector3"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector3 to convert.</param>
    /// <returns>A Variant representation of this Vector3.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Vector3 from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector3I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector3I to convert.</param>
    /// <returns>A Variant representation of this Vector3I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Vector3I from) => from;

    /// <summary>
    /// Converts the provided <see cref="Basis"/> to a Variant.
    /// </summary>
    /// <param name="from">The Basis to convert.</param>
    /// <returns>A Variant representation of this Basis.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Basis from) => from;

    /// <summary>
    /// Converts the provided <see cref="Quaternion"/> to a Variant.
    /// </summary>
    /// <param name="from">The Quaternion to convert.</param>
    /// <returns>A Variant representation of this Quaternion.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Quaternion from) => from;

    /// <summary>
    /// Converts the provided <see cref="Transform3D"/> to a Variant.
    /// </summary>
    /// <param name="from">The Transform3D to convert.</param>
    /// <returns>A Variant representation of this Transform3D.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Transform3D from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector4"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector4 to convert.</param>
    /// <returns>A Variant representation of this Vector4.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Vector4 from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector4I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector4I to convert.</param>
    /// <returns>A Variant representation of this Vector4I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Vector4I from) => from;

    /// <summary>
    /// Converts the provided <see cref="Projection"/> to a Variant.
    /// </summary>
    /// <param name="from">The Projection to convert.</param>
    /// <returns>A Variant representation of this Projection.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Projection from) => from;

    /// <summary>
    /// Converts the provided <see cref="Aabb"/> to a Variant.
    /// </summary>
    /// <param name="from">The Aabb to convert.</param>
    /// <returns>A Variant representation of this Aabb.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Aabb from) => from;

    /// <summary>
    /// Converts the provided <see cref="Color"/> to a Variant.
    /// </summary>
    /// <param name="from">The Color to convert.</param>
    /// <returns>A Variant representation of this Color.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Color from) => from;

    /// <summary>
    /// Converts the provided <see cref="Plane"/> to a Variant.
    /// </summary>
    /// <param name="from">The Plane to convert.</param>
    /// <returns>A Variant representation of this Plane.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Plane from) => from;

    /// <summary>
    /// Converts the provided <see cref="Callable"/> to a Variant.
    /// </summary>
    /// <param name="from">The Callable to convert.</param>
    /// <returns>A Variant representation of this Callable.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Callable from) => from;

    /// <summary>
    /// Converts the provided <see cref="Signal"/> to a Variant.
    /// </summary>
    /// <param name="from">The Signal to convert.</param>
    /// <returns>A Variant representation of this Signal.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Signal from) => from;

    /// <summary>
    /// Converts the provided <see cref="byte"/> span to a Variant.
    /// </summary>
    /// <param name="from">The byte to convert.</param>
    /// <returns>A Variant representation of this byte span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<byte> from) => from;

    /// <summary>
    /// Converts the provided <see cref="int"/> span to a Variant.
    /// </summary>
    /// <param name="from">The int to convert.</param>
    /// <returns>A Variant representation of this int span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<int> from) => from;

    /// <summary>
    /// Converts the provided <see cref="long"/> span to a Variant.
    /// </summary>
    /// <param name="from">The long to convert.</param>
    /// <returns>A Variant representation of this long span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<long> from) => from;

    /// <summary>
    /// Converts the provided <see cref="float"/> span to a Variant.
    /// </summary>
    /// <param name="from">The float to convert.</param>
    /// <returns>A Variant representation of this float span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<float> from) => from;

    /// <summary>
    /// Converts the provided <see cref="double"/> span to a Variant.
    /// </summary>
    /// <param name="from">The double to convert.</param>
    /// <returns>A Variant representation of this double span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<double> from) => from;

    /// <summary>
    /// Converts the provided <see cref="string"/> span to a Variant.
    /// </summary>
    /// <param name="from">The string to convert.</param>
    /// <returns>A Variant representation of this string span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<string> from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector2"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Vector2 to convert.</param>
    /// <returns>A Variant representation of this Vector2 span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<Vector2> from) => from;

    /// <summary>
    /// Converts the provided <see cref="Vector3"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Vector3 to convert.</param>
    /// <returns>A Variant representation of this Vector3 span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<Vector3> from) => from;

    /// <summary>
    /// Converts the provided <see cref="Color"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Color to convert.</param>
    /// <returns>A Variant representation of this Color span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<Color> from) => from;

    /// <summary>
    /// Converts the provided <see cref="GodotObject"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The GodotObject to convert.</param>
    /// <returns>A Variant representation of this GodotObject[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(GodotObject[] from) => from;

    /// <summary>
    /// Converts the provided <see cref="Collections.Dictionary{TKey, TValue}"/> to a Variant.
    /// </summary>
    /// <param name="from">The typed Godot Dictionary to convert.</param>
    /// <returns>A Variant representation of this typed Godot Dictionary.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom<[MustBeVariant] TKey, [MustBeVariant] TValue>(Collections.Dictionary<TKey, TValue> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromDictionary(from));

    /// <summary>
    /// Converts the provided <see cref="Collections.Array{T}"/> to a Variant.
    /// </summary>
    /// <param name="from">The typed Godot Array to convert.</param>
    /// <returns>A Variant representation of this typed Godot Array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom<[MustBeVariant] T>(Collections.Array<T> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromArray(from));

    /// <summary>
    /// Converts the provided <see cref="StringName"/> span to a Variant.
    /// </summary>
    /// <param name="from">The StringName to convert.</param>
    /// <returns>A Variant representation of this StringName span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<StringName> from) => from;

    /// <summary>
    /// Converts the provided <see cref="NodePath"/> span to a Variant.
    /// </summary>
    /// <param name="from">The NodePath to convert.</param>
    /// <returns>A Variant representation of this NodePath span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<NodePath> from) => from;

    /// <summary>
    /// Converts the provided <see cref="Rid"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Rid to convert.</param>
    /// <returns>A Variant representation of this Rid span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Span<Rid> from) => from;

    /// <summary>
    /// Converts the provided <see cref="GodotObject"/> to a Variant.
    /// </summary>
    /// <param name="from">The GodotObject to convert.</param>
    /// <returns>A Variant representation of this GodotObject.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(GodotObject from) => from;

    /// <summary>
    /// Converts the provided <see cref="StringName"/> to a Variant.
    /// </summary>
    /// <param name="from">The StringName to convert.</param>
    /// <returns>A Variant representation of this StringName.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(StringName from) => from;

    /// <summary>
    /// Converts the provided <see cref="NodePath"/> to a Variant.
    /// </summary>
    /// <param name="from">The NodePath to convert.</param>
    /// <returns>A Variant representation of this NodePath.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(NodePath from) => from;

    /// <summary>
    /// Converts the provided <see cref="Rid"/> to a Variant.
    /// </summary>
    /// <param name="from">The Rid to convert.</param>
    /// <returns>A Variant representation of this Rid.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Rid from) => from;

    /// <summary>
    /// Converts the provided <see cref="Collections.Dictionary"/> to a Variant.
    /// </summary>
    /// <param name="from">The untyped Godot Dictionary to convert.</param>
    /// <returns>A Variant representation of this untyped Godot Dictionary.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Collections.Dictionary from) => from;

    /// <summary>
    /// Converts the provided <see cref="Collections.Array"/> to a Variant.
    /// </summary>
    /// <param name="from">The untyped Godot Array to convert.</param>
    /// <returns>A Variant representation of this untyped Godot Array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(Collections.Array from) => from;

    // Implicit conversion operators

    /// <summary>
    /// Converts the provided <see cref="bool"/> to a Variant.
    /// </summary>
    /// <param name="from">The bool to convert.</param>
    /// <returns>A Variant representation of this bool.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(bool from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromBool(from));

    /// <summary>
    /// Converts the provided <see cref="char"/> to a Variant.
    /// </summary>
    /// <param name="from">The char to convert.</param>
    /// <returns>A Variant representation of this char.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(char from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="sbyte"/> to a Variant.
    /// </summary>
    /// <param name="from">The sbyte to convert.</param>
    /// <returns>A Variant representation of this sbyte.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(sbyte from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="short"/> to a Variant.
    /// </summary>
    /// <param name="from">The short to convert.</param>
    /// <returns>A Variant representation of this short.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(short from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="int"/> to a Variant.
    /// </summary>
    /// <param name="from">The int to convert.</param>
    /// <returns>A Variant representation of this int.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(int from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="long"/> to a Variant.
    /// </summary>
    /// <param name="from">The long to convert.</param>
    /// <returns>A Variant representation of this long.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(long from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="byte"/> to a Variant.
    /// </summary>
    /// <param name="from">The byte to convert.</param>
    /// <returns>A Variant representation of this byte.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(byte from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="ushort"/> to a Variant.
    /// </summary>
    /// <param name="from">The ushort to convert.</param>
    /// <returns>A Variant representation of this ushort.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(ushort from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="uint"/> to a Variant.
    /// </summary>
    /// <param name="from">The uint to convert.</param>
    /// <returns>A Variant representation of this uint.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(uint from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="ulong"/> to a Variant.
    /// </summary>
    /// <param name="from">The ulong to convert.</param>
    /// <returns>A Variant representation of this ulong.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(ulong from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromInt(from));

    /// <summary>
    /// Converts the provided <see cref="float"/> to a Variant.
    /// </summary>
    /// <param name="from">The float to convert.</param>
    /// <returns>A Variant representation of this float.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(float from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromFloat(from));

    /// <summary>
    /// Converts the provided <see cref="double"/> to a Variant.
    /// </summary>
    /// <param name="from">The double to convert.</param>
    /// <returns>A Variant representation of this double.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(double from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromFloat(from));

    /// <summary>
    /// Converts the provided <see cref="string"/> to a Variant.
    /// </summary>
    /// <param name="from">The string to convert.</param>
    /// <returns>A Variant representation of this string.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(string from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromString(from));

    /// <summary>
    /// Converts the provided <see cref="Vector2"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector2 to convert.</param>
    /// <returns>A Variant representation of this Vector2.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector2 from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromVector2(from));

    /// <summary>
    /// Converts the provided <see cref="Vector2I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector2I to convert.</param>
    /// <returns>A Variant representation of this Vector2I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector2I from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromVector2I(from));

    /// <summary>
    /// Converts the provided <see cref="Rect2"/> to a Variant.
    /// </summary>
    /// <param name="from">The Rect2 to convert.</param>
    /// <returns>A Variant representation of this Rect2.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Rect2 from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromRect2(from));

    /// <summary>
    /// Converts the provided <see cref="Rect2I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Rect2I to convert.</param>
    /// <returns>A Variant representation of this Rect2I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Rect2I from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromRect2I(from));

    /// <summary>
    /// Converts the provided <see cref="Transform2D"/> to a Variant.
    /// </summary>
    /// <param name="from">The Transform2D to convert.</param>
    /// <returns>A Variant representation of this Transform2D.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Transform2D from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromTransform2D(from));

    /// <summary>
    /// Converts the provided <see cref="Vector3"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector3 to convert.</param>
    /// <returns>A Variant representation of this Vector3.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector3 from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromVector3(from));

    /// <summary>
    /// Converts the provided <see cref="Vector3I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector3I to convert.</param>
    /// <returns>A Variant representation of this Vector3I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector3I from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromVector3I(from));

    /// <summary>
    /// Converts the provided <see cref="Basis"/> to a Variant.
    /// </summary>
    /// <param name="from">The Basis to convert.</param>
    /// <returns>A Variant representation of this Basis.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Basis from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromBasis(from));

    /// <summary>
    /// Converts the provided <see cref="Quaternion"/> to a Variant.
    /// </summary>
    /// <param name="from">The Quaternion to convert.</param>
    /// <returns>A Variant representation of this Quaternion.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Quaternion from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromQuaternion(from));

    /// <summary>
    /// Converts the provided <see cref="Transform3D"/> to a Variant.
    /// </summary>
    /// <param name="from">The Transform3D to convert.</param>
    /// <returns>A Variant representation of this Transform3D.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Transform3D from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromTransform3D(from));

    /// <summary>
    /// Converts the provided <see cref="Vector4"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector4 to convert.</param>
    /// <returns>A Variant representation of this Vector4.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector4 from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromVector4(from));

    /// <summary>
    /// Converts the provided <see cref="Vector4I"/> to a Variant.
    /// </summary>
    /// <param name="from">The Vector4I to convert.</param>
    /// <returns>A Variant representation of this Vector4I.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector4I from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromVector4I(from));

    /// <summary>
    /// Converts the provided <see cref="Projection"/> to a Variant.
    /// </summary>
    /// <param name="from">The Projection to convert.</param>
    /// <returns>A Variant representation of this Projection.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Projection from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromProjection(from));

    /// <summary>
    /// Converts the provided <see cref="Aabb"/> to a Variant.
    /// </summary>
    /// <param name="from">The Aabb to convert.</param>
    /// <returns>A Variant representation of this Aabb.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Aabb from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromAabb(from));

    /// <summary>
    /// Converts the provided <see cref="Color"/> to a Variant.
    /// </summary>
    /// <param name="from">The Color to convert.</param>
    /// <returns>A Variant representation of this Color.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Color from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromColor(from));

    /// <summary>
    /// Converts the provided <see cref="Plane"/> to a Variant.
    /// </summary>
    /// <param name="from">The Plane to convert.</param>
    /// <returns>A Variant representation of this Plane.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Plane from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPlane(from));

    /// <summary>
    /// Converts the provided <see cref="Callable"/> to a Variant.
    /// </summary>
    /// <param name="from">The Callable to convert.</param>
    /// <returns>A Variant representation of this Callable.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Callable from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromCallable(from));

    /// <summary>
    /// Converts the provided <see cref="Signal"/> to a Variant.
    /// </summary>
    /// <param name="from">The Signal to convert.</param>
    /// <returns>A Variant representation of this Signal.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Signal from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromSignal(from));

    /// <summary>
    /// Converts the provided <see cref="byte"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The byte[] to convert.</param>
    /// <returns>A Variant representation of this byte[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(byte[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="int"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The int[] to convert.</param>
    /// <returns>A Variant representation of this int[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(int[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="long"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The long[] to convert.</param>
    /// <returns>A Variant representation of this long[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(long[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="float"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The float[] to convert.</param>
    /// <returns>A Variant representation of this float[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(float[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="double"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The double[] to convert.</param>
    /// <returns>A Variant representation of this double[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(double[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="string"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The string[] to convert.</param>
    /// <returns>A Variant representation of this string[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(string[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="Vector2"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The Vector2[] to convert.</param>
    /// <returns>A Variant representation of this Vector2[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector2[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="Vector3"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The Vector3[] to convert.</param>
    /// <returns>A Variant representation of this Vector3[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Vector3[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="Color"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The Color[] to convert.</param>
    /// <returns>A Variant representation of this Color[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Color[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="GodotObject"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The GodotObject[] to convert.</param>
    /// <returns>A Variant representation of this GodotObject[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(GodotObject[] from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromSystemArrayOfGodotObject(from));

    /// <summary>
    /// Converts the provided <see cref="StringName"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The StringName[] to convert.</param>
    /// <returns>A Variant representation of this StringName[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(StringName[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="NodePath"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The NodePath[] to convert.</param>
    /// <returns>A Variant representation of this NodePath[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(NodePath[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="Rid"/>[] to a Variant.
    /// </summary>
    /// <param name="from">The Rid[] to convert.</param>
    /// <returns>A Variant representation of this Rid[].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Rid[] from) =>
        (Variant)from.AsSpan();

    /// <summary>
    /// Converts the provided <see cref="byte"/> span to a Variant.
    /// </summary>
    /// <param name="from">The byte span to convert.</param>
    /// <returns>A Variant representation of this byte span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<byte> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedByteArray(from));

    /// <summary>
    /// Converts the provided <see cref="int"/> span to a Variant.
    /// </summary>
    /// <param name="from">The int span to convert.</param>
    /// <returns>A Variant representation of this int span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<int> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedInt32Array(from));

    /// <summary>
    /// Converts the provided <see cref="long"/> span to a Variant.
    /// </summary>
    /// <param name="from">The long span to convert.</param>
    /// <returns>A Variant representation of this long span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<long> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedInt64Array(from));

    /// <summary>
    /// Converts the provided <see cref="float"/> span to a Variant.
    /// </summary>
    /// <param name="from">The float span to convert.</param>
    /// <returns>A Variant representation of this float span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<float> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedFloat32Array(from));

    /// <summary>
    /// Converts the provided <see cref="double"/> span to a Variant.
    /// </summary>
    /// <param name="from">The double span to convert.</param>
    /// <returns>A Variant representation of this double span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<double> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedFloat64Array(from));

    /// <summary>
    /// Converts the provided <see cref="string"/> span to a Variant.
    /// </summary>
    /// <param name="from">The string span to convert.</param>
    /// <returns>A Variant representation of this string span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<string> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedStringArray(from));

    /// <summary>
    /// Converts the provided <see cref="Vector2"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Vector2 span to convert.</param>
    /// <returns>A Variant representation of this Vector2 span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<Vector2> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedVector2Array(from));

    /// <summary>
    /// Converts the provided <see cref="Vector3"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Vector3 span to convert.</param>
    /// <returns>A Variant representation of this Vector3 span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<Vector3> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedVector3Array(from));

    /// <summary>
    /// Converts the provided <see cref="Color"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Color span to convert.</param>
    /// <returns>A Variant representation of this Color span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<Color> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromPackedColorArray(from));

    /// <summary>
    /// Converts the provided <see cref="StringName"/> span to a Variant.
    /// </summary>
    /// <param name="from">The StringName span to convert.</param>
    /// <returns>A Variant representation of this StringName span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<StringName> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromSystemArrayOfStringName(from));

    /// <summary>
    /// Converts the provided <see cref="NodePath"/> span to a Variant.
    /// </summary>
    /// <param name="from">The NodePath span to convert.</param>
    /// <returns>A Variant representation of this NodePath span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<NodePath> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromSystemArrayOfNodePath(from));

    /// <summary>
    /// Converts the provided <see cref="Rid"/> span to a Variant.
    /// </summary>
    /// <param name="from">The Rid span to convert.</param>
    /// <returns>A Variant representation of this Rid span.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Span<Rid> from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromSystemArrayOfRid(from));

    /// <summary>
    /// Converts the provided <see cref="GodotObject"/> to a Variant.
    /// </summary>
    /// <param name="from">The GodotObject to convert.</param>
    /// <returns>A Variant representation of this GodotObject.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(GodotObject from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromGodotObject(from));

    /// <summary>
    /// Converts the provided <see cref="StringName"/> to a Variant.
    /// </summary>
    /// <param name="from">The StringName to convert.</param>
    /// <returns>A Variant representation of this StringName.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(StringName from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromStringName(from));

    /// <summary>
    /// Converts the provided <see cref="NodePath"/> to a Variant.
    /// </summary>
    /// <param name="from">The NodePath to convert.</param>
    /// <returns>A Variant representation of this NodePath.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(NodePath from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromNodePath(from));

    /// <summary>
    /// Converts the provided <see cref="Rid"/> to a Variant.
    /// </summary>
    /// <param name="from">The Rid to convert.</param>
    /// <returns>A Variant representation of this Rid.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Rid from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromRid(from));

    /// <summary>
    /// Converts the provided <see cref="Collections.Dictionary"/> to a Variant.
    /// </summary>
    /// <param name="from">The untyped Godot Dictionary to convert.</param>
    /// <returns>A Variant representation of this untyped Godot Dictionary.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Collections.Dictionary from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromDictionary(from));

    /// <summary>
    /// Converts the provided <see cref="Collections.Array"/> to a Variant.
    /// </summary>
    /// <param name="from">The untyped Godot Array to convert.</param>
    /// <returns>A Variant representation of this untyped Godot Array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Variant(Collections.Array from) =>
        CreateTakingOwnershipOfDisposableValue(VariantUtils.CreateFromArray(from));
}
