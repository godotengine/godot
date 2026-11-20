using System;
using System.Runtime.InteropServices;
using System.Threading;
using Godot.NativeInterop;

#nullable enable

namespace Godot
{
    public class SignalAwaiter : IAwaiter<Variant[]>, IAwaitable<Variant[]>
    {
        private bool _completed;
        private bool _canceled;
        private Variant[] _result;
        private Action? _continuation;
        private readonly CancellationToken _cancellationToken;
        private readonly CancellationTokenRegistration _cancellationRegistration;
        private readonly GodotObject _source;
        private readonly StringName _signal;
        private readonly GodotObject _target;
        private readonly GCHandle _awaiterGCHandle;

        public SignalAwaiter(GodotObject source, StringName signal, GodotObject target) : this(source, signal, target, CancellationToken.None) { }

        public SignalAwaiter(GodotObject source, StringName signal, GodotObject target, CancellationToken cancellationToken)
        {
            _cancellationToken = cancellationToken;
            _result = [];

            _source = source;
            _signal = signal;
            _target = target;

            if (_cancellationToken.IsCancellationRequested)
            {
                _canceled = true;
                _completed = true;
                return;
            }

            _awaiterGCHandle = RegisterAwaiter(this, source, signal, target);

            if (_cancellationToken.CanBeCanceled)
            {
                _cancellationRegistration = _cancellationToken.Register(CancelCallback);
            }
        }

        private static GCHandle RegisterAwaiter(SignalAwaiter awaiter, GodotObject source, StringName signal, GodotObject target)
        {
            var awaiterGCHandle = CustomGCHandle.AllocStrong(awaiter);
            using godot_string_name signalSrc = NativeFuncs.godotsharp_string_name_new_copy(
                (godot_string_name)(signal?.NativeValue ?? default));
            NativeFuncs.godotsharp_internal_signal_awaiter_connect(
                GodotObject.GetPtr(source),
                in signalSrc,
                GodotObject.GetPtr(target),
                GCHandle.ToIntPtr(awaiterGCHandle));
            return awaiterGCHandle;
        }

        private static void UnregisterAwaiter(SignalAwaiter awaiter)
        {
            if (!awaiter._awaiterGCHandle.IsAllocated)
            {
                return;
            }

            using godot_string_name signalSrc = NativeFuncs.godotsharp_string_name_new_copy(
                (godot_string_name)(awaiter._signal?.NativeValue ?? default));
            NativeFuncs.godotsharp_internal_signal_awaiter_disconnect(
                GodotObject.GetPtr(awaiter._source),
                in signalSrc,
                GodotObject.GetPtr(awaiter._target),
                GCHandle.ToIntPtr(awaiter._awaiterGCHandle));

            CustomGCHandle.Free(awaiter._awaiterGCHandle);
        }

        public bool IsCompleted => _completed;

        public void OnCompleted(Action continuation)
        {
            _continuation = continuation;
        }

        public Variant[] GetResult()
        {
            _cancellationRegistration.Dispose();

            if (_canceled || _cancellationToken.IsCancellationRequested)
            {
                throw new OperationCanceledException(_cancellationToken);
            }

            return _result;
        }

        public IAwaiter<Variant[]> GetAwaiter() => this;

        [UnmanagedCallersOnly]
        internal static unsafe void SignalCallback(IntPtr awaiterGCHandlePtr, godot_variant** args, int argCount,
            godot_bool* outAwaiterIsNull)
        {
            try
            {
                var awaiter = (SignalAwaiter?)GCHandle.FromIntPtr(awaiterGCHandlePtr).Target;

                if (awaiter == null)
                {
                    *outAwaiterIsNull = godot_bool.True;
                    return;
                }

                *outAwaiterIsNull = godot_bool.False;

                if (Interlocked.Exchange(ref awaiter._completed, true))
                {
                    // The awaiter had already been completed.
                    return;
                }

                awaiter._cancellationRegistration.Dispose();

                if (awaiter._canceled)
                {
                    return;
                }

                if (argCount > 0)
                {
                    Variant[] signalArgs = new Variant[argCount];

                    for (int i = 0; i < argCount; i++)
                        signalArgs[i] = Variant.CreateCopyingBorrowed(*args[i]);

                    awaiter._result = signalArgs;
                }
                else
                {
                    awaiter._result = [];
                }

                awaiter._continuation?.Invoke();

                // When this method is called, the signal has already been disconnected because we connected
                // with Object.ConnectFlags.Oneshot. So we don't need to unregister the awaiter here.
                // But we still need to free the GCHandle.
                CustomGCHandle.Free(awaiter._awaiterGCHandle);
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outAwaiterIsNull = godot_bool.False;
            }
        }

        private void CancelCallback()
        {
            if (Interlocked.Exchange(ref _completed, true))
            {
                // The awaiter had already been completed.
                return;
            }

            _cancellationRegistration.Dispose();

            _canceled = true;

            // The signal must be disconnected on Godot's main thread.
            Dispatcher.SynchronizationContext.Post(_ =>
            {
                UnregisterAwaiter(this);
                _continuation?.Invoke();
            }, null);
        }
    }
}
