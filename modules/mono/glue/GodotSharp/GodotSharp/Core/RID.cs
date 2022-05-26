using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// RID类型用于访问资源的唯一整数ID。
    /// 它们是不透明的，这意味着它们不授予对关联的访问权限
    /// 自己的资源。 它们由低级服务器使用并与低级服务器一起使用
    /// 类，例如 <see cref="VisualServer"/>。
    /// </summary>
    public sealed partial class RID : IDisposable
    {
        private bool _disposed = false;

        internal IntPtr ptr;

        internal static IntPtr GetPtr(RID instance)
        {
            if (instance == null)
                throw new NullReferenceException($"The instance of type {nameof(RID)} is null.");

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~RID()
        {
            Dispose(false);
        }

        /// <summary>
        /// 处理这个<see cref="RID"/>。
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (ptr != IntPtr.Zero)
            {
                godot_icall_RID_Dtor(ptr);
                ptr = IntPtr.Zero;
            }

            _disposed = true;
        }

        internal RID(IntPtr ptr)
        {
            this.ptr = ptr;
        }

        /// <summary>
        /// 指向此 <see cref="RID"/> 的本机实例的指针。
        /// </summary>
        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        internal RID()
        {
            this.ptr = IntPtr.Zero;
        }

        /// <summary>
        /// 为给定的 <see cref="Object"/> <paramref name="from"/> 构造一个新的 <see cref="RID"/>。
        /// </summary>
        public RID(Object from)
        {
            this.ptr = godot_icall_RID_Ctor(Object.GetPtr(from));
        }

        /// <summary>
        /// 返回引用资源的 ID。
        /// </summary>
        /// <returns>引用资源的ID。</returns>
        public int GetId()
        {
            return godot_icall_RID_get_id(GetPtr(this));
        }

        /// <summary>
        /// 将此 <see cref="RID"/> 转换为字符串。
        /// </summary>
        /// <returns>此 RID 的字符串表示形式。</returns>
        public override string ToString() => "[RID]";

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_RID_Ctor(IntPtr from);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_RID_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_RID_get_id(IntPtr ptr);
    }
}
