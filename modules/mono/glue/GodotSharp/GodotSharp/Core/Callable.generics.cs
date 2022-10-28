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
                $" Expected {countExpected} arguments, received {countReceived}.",
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
    public static unsafe Callable From<T0>(
        Action<T0> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 1);

            ((Action<T0>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1>(
        Action<T0, T1> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 2);

            ((Action<T0, T1>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2>(
        Action<T0, T1, T2> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 3);

            ((Action<T0, T1, T2>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2, T3>(
        Action<T0, T1, T2, T3> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 4);

            ((Action<T0, T1, T2, T3>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4>(
        Action<T0, T1, T2, T3, T4> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 5);

            ((Action<T0, T1, T2, T3, T4>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5>(
        Action<T0, T1, T2, T3, T4, T5> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 6);

            ((Action<T0, T1, T2, T3, T4, T5>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, T6>(
        Action<T0, T1, T2, T3, T4, T5, T6> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 7);

            ((Action<T0, T1, T2, T3, T4, T5, T6>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5]),
                VariantConversionCallbacks.GetToManagedCallback<T6>()(args[6])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, T6, T7>(
        Action<T0, T1, T2, T3, T4, T5, T6, T7> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 8);

            ((Action<T0, T1, T2, T3, T4, T5, T6, T7>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5]),
                VariantConversionCallbacks.GetToManagedCallback<T6>()(args[6]),
                VariantConversionCallbacks.GetToManagedCallback<T7>()(args[7])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <inheritdoc cref="From(Action)"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, T6, T7, T8>(
        Action<T0, T1, T2, T3, T4, T5, T6, T7, T8> action
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 9);

            ((Action<T0, T1, T2, T3, T4, T5, T6, T7, T8>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5]),
                VariantConversionCallbacks.GetToManagedCallback<T6>()(args[6]),
                VariantConversionCallbacks.GetToManagedCallback<T7>()(args[7]),
                VariantConversionCallbacks.GetToManagedCallback<T8>()(args[8])
            );

            ret = default;
        }

        return CreateWithUnsafeTrampoline(action, &Trampoline);
    }

    /// <summary>
    /// Constructs a new <see cref="Callable"/> for the given <paramref name="func"/>.
    /// </summary>
    /// <param name="func">Action method that will be called.</param>
    public static unsafe Callable From<TResult>(
        Func<TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 0);

            TResult res = ((Func<TResult>)delegateObj)();

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, TResult>(
        Func<T0, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 1);

            TResult res = ((Func<T0, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, TResult>(
        Func<T0, T1, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 2);

            TResult res = ((Func<T0, T1, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, TResult>(
        Func<T0, T1, T2, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 3);

            TResult res = ((Func<T0, T1, T2, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, T3, TResult>(
        Func<T0, T1, T2, T3, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 4);

            TResult res = ((Func<T0, T1, T2, T3, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, TResult>(
        Func<T0, T1, T2, T3, T4, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 5);

            TResult res = ((Func<T0, T1, T2, T3, T4, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, TResult>(
        Func<T0, T1, T2, T3, T4, T5, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 6);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, T6, TResult>(
        Func<T0, T1, T2, T3, T4, T5, T6, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 7);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, T6, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5]),
                VariantConversionCallbacks.GetToManagedCallback<T6>()(args[6])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, T6, T7, TResult>(
        Func<T0, T1, T2, T3, T4, T5, T6, T7, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 8);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, T6, T7, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5]),
                VariantConversionCallbacks.GetToManagedCallback<T6>()(args[6]),
                VariantConversionCallbacks.GetToManagedCallback<T7>()(args[7])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }

    /// <inheritdoc cref="From{TResult}(Func{TResult})"/>
    public static unsafe Callable From<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult>(
        Func<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult> func
    )
    {
        static void Trampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
        {
            ThrowIfArgCountMismatch(args, 9);

            TResult res = ((Func<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult>)delegateObj)(
                VariantConversionCallbacks.GetToManagedCallback<T0>()(args[0]),
                VariantConversionCallbacks.GetToManagedCallback<T1>()(args[1]),
                VariantConversionCallbacks.GetToManagedCallback<T2>()(args[2]),
                VariantConversionCallbacks.GetToManagedCallback<T3>()(args[3]),
                VariantConversionCallbacks.GetToManagedCallback<T4>()(args[4]),
                VariantConversionCallbacks.GetToManagedCallback<T5>()(args[5]),
                VariantConversionCallbacks.GetToManagedCallback<T6>()(args[6]),
                VariantConversionCallbacks.GetToManagedCallback<T7>()(args[7]),
                VariantConversionCallbacks.GetToManagedCallback<T8>()(args[8])
            );

            ret = VariantConversionCallbacks.GetToVariantCallback<TResult>()(res);
        }

        return CreateWithUnsafeTrampoline(func, &Trampoline);
    }
}
