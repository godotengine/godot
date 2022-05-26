using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object : IDisposable
    {
        private bool _disposed = false;

        private const string nativeName = "Object";

        internal IntPtr ptr;
        internal bool memoryOwn;

        /// <summary>
        /// Constructs a new <see cref="Object"/>.
        /// </summary>
        public Object() : this(false)
        {
            if (ptr == IntPtr.Zero)
                ptr = godot_icall_Object_Ctor(this);
        }

        internal Object(bool memoryOwn)
        {
            this.memoryOwn = memoryOwn;
        }

        /// <summary>
        /// 指向此 <see cref="Object"/> 的本机实例的指针
        /// </summary>
        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        internal static IntPtr GetPtr(Object instance)
        {
            if (instance == null)
                return IntPtr.Zero;

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~Object()
        {
            Dispose(false);
        }

        /// <summary>
        /// 处理这个 <see cref="Object"/>。
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///处理此 <see cref="Object"/> 的实现。
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (ptr != IntPtr.Zero)
            {
                if (memoryOwn)
                {
                    memoryOwn = false;
                    godot_icall_Reference_Disposed(this, ptr, !disposing);
                }
                else
                {
                    godot_icall_Object_Disposed(this, ptr);
                }

                ptr = IntPtr.Zero;
            }

            _disposed = true;
        }

        /// <summary>
        /// 将此 <see cref="Object"/> 转换为字符串。
        /// </summary>
        /// <returns>此对象的字符串表示形式。</returns>
        public override string ToString()
        {
            return godot_icall_Object_ToString(GetPtr(this));
        }

        /// <summary>
        /// 返回一个新的 <see cref="SignalAwaiter"/> 等待实例配置完成时
        /// <paramref name="source"/> 发出由 <paramref name="signal"/> 参数指定的信号。
        /// </summary>
        /// <param name="source">
        /// 等待者将要监听的实例。
        /// </param>
        /// <param name="signal">
        /// 等待者将等待的信号。
        /// </param>
        /// <example>
        /// 此示例每帧打印一条消息，最多 100 次。
        /// <code>
        /// public override void _Ready()
        /// {
        ///     for (int i = 0; i &lt; 100; i++)
        ///     {
        ///         await ToSignal(GetTree(), "idle_frame");
        ///         GD.Print($"Frame {i}");
        ///     }
        /// }
        /// </code>
        /// </example>
        /// <returns>
        /// 一个完成的 <see cref="SignalAwaiter"/>
        /// <paramref name="source"/> 发出 <paramref name="signal"/>。
        /// </returns>
        public SignalAwaiter ToSignal(Object source, string signal)
        {
            return new SignalAwaiter(source, signal, this);
        }

        /// <summary>
        /// 获取与此实例关联的新 <see cref="DynamicGodotObject"/>。
        /// </summary>
        public dynamic DynamicObject => new DynamicGodotObject(this);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Object_Ctor(Object obj);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Object_Disposed(Object obj, IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Reference_Disposed(Object obj, IntPtr ptr, bool isFinalizer);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Object_ToString(IntPtr ptr);

        // Used by the generated API
        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Object_ClassDB_get_method(string type, string method);
    }
}
