using System;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot;

#nullable enable

public readonly partial struct Callable
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ThrowIfArgCountMismatch(NativeVariantPtrArgs args, int countExpected,
        [CallerArgumentExpression("args")] string? paramName = null)
    {
        if (countExpected != args.Count)
            ThrowArgCountMismatch(countExpected, args.Count, paramName);

        static void ThrowArgCountMismatch(int countExpected, int countReceived, string? paramName)
        {
            throw new ArgumentException(
                "Invalid argument count for invoking callable." +
                $" Expected {countExpected} argument(s), received {countReceived}.",
                paramName);
        }
    }

    /// <summary>
    /// Constructs a new <see cref="Callable"/> for the given <paramref name="action"/>.
    /// </summary>
    /// <param name="action">Action method that will be called.</param>
    public static unsafe Callable From(
        Action action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 0);

            ((Action)delegateObj)();

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0>(
        Action<T0> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 1);

            ((Action<T0>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1>(
        Action<T0, T1> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 2);

            ((Action<T0, T1>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2>(
        Action<T0, T1, T2> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 3);

            ((Action<T0, T1, T2>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3>(
        Action<T0, T1, T2, T3> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 4);

            ((Action<T0, T1, T2, T3>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4>(
        Action<T0, T1, T2, T3, T4> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 5);

            ((Action<T0, T1, T2, T3, T4>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5>(
        Action<T0, T1, T2, T3, T4, T5> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 6);

            ((Action<T0, T1, T2, T3, T4, T5>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] T6>(
        Action<T0, T1, T2, T3, T4, T5, T6> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 7);

            ((Action<T0, T1, T2, T3, T4, T5, T6>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5]),
                VariantUtils.ConvertTo<T6>(args[6])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] T6, [MustBeVariant] T7>(
        Action<T0, T1, T2, T3, T4, T5, T6, T7> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 8);

            ((Action<T0, T1, T2, T3, T4, T5, T6, T7>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5]),
                VariantUtils.ConvertTo<T6>(args[6]),
                VariantUtils.ConvertTo<T7>(args[7])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] T6, [MustBeVariant] T7, [MustBeVariant] T8>(
        Action<T0, T1, T2, T3, T4, T5, T6, T7, T8> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 9);

            ((Action<T0, T1, T2, T3, T4, T5, T6, T7, T8>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5]),
                VariantUtils.ConvertTo<T6>(args[6]),
                VariantUtils.ConvertTo<T7>(args[7]),
                VariantUtils.ConvertTo<T8>(args[8])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <summary>
    /// Constructs a new <see cref="Callable"/> for the given <paramref name="func"/>.
    /// </summary>
    /// <param name="func">Action method that will be called.</param>
    public static unsafe Callable From<[MustBeVariant] TResult>(
        Func<TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 0);

            TResult res = ((Func<TResult>)delegateObj)();

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] TResult>(
        Func<T0, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 1);

            TResult res = ((Func<T0, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] TResult>(
        Func<T0, T1, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 2);

            TResult res = ((Func<T0, T1, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] TResult>(
        Func<T0, T1, T2, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 3);

            TResult res = ((Func<T0, T1, T2, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] TResult>(
        Func<T0, T1, T2, T3, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 4);

            TResult res = ((Func<T0, T1, T2, T3, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] TResult>(
        Func<T0, T1, T2, T3, T4, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 5);

            TResult res = ((Func<T0, T1, T2, T3, T4, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] TResult>(
        Func<T0, T1, T2, T3, T4, T5, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 6);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] T6, [MustBeVariant] TResult>(
        Func<T0, T1, T2, T3, T4, T5, T6, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 7);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, T6, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5]),
                VariantUtils.ConvertTo<T6>(args[6])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] T6, [MustBeVariant] T7, [MustBeVariant] TResult>(
        Func<T0, T1, T2, T3, T4, T5, T6, T7, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 8);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, T6, T7, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5]),
                VariantUtils.ConvertTo<T6>(args[6]),
                VariantUtils.ConvertTo<T7>(args[7])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<[MustBeVariant] T0, [MustBeVariant] T1, [MustBeVariant] T2, [MustBeVariant] T3, [MustBeVariant] T4, [MustBeVariant] T5, [MustBeVariant] T6, [MustBeVariant] T7, [MustBeVariant] T8, [MustBeVariant] TResult>(
        Func<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 9);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult>)delegateObj)(
                VariantUtils.ConvertTo<T0>(args[0]),
                VariantUtils.ConvertTo<T1>(args[1]),
                VariantUtils.ConvertTo<T2>(args[2]),
                VariantUtils.ConvertTo<T3>(args[3]),
                VariantUtils.ConvertTo<T4>(args[4]),
                VariantUtils.ConvertTo<T5>(args[5]),
                VariantUtils.ConvertTo<T6>(args[6]),
                VariantUtils.ConvertTo<T7>(args[7]),
                VariantUtils.ConvertTo<T8>(args[8])
            );

            ret = VariantUtils.CreateFrom(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }
}
