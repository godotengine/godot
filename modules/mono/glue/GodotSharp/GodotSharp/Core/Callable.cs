using System;

namespace Godot
{
    public struct Callable
    {
        private readonly Object _target;
        private readonly StringName _method;
        private readonly Delegate _delegate;

        public Object Target => _target;
        public StringName Method => _method;
        public Delegate Delegate => _delegate;

        public static implicit operator Callable(Delegate @delegate) => new Callable(@delegate);

        public Callable(Object target, StringName method)
        {
            _target = target;
            _method = method;
            _delegate = null;
        }

        public Callable(Delegate @delegate)
        {
            _target = null;
            _method = null;
            _delegate = @delegate;
        }
    }
}
