using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object
    {
        /// <summary>
        /// 返回  <paramref name="instance"/> 是否为有效对象
        /// （例如尚未从内存中删除）.
        /// </summary>
        /// <param name="instance">要检查的实例.</param>
        /// <returns>如果实例是有效对象.</returns>
        public static bool IsInstanceValid(Object instance)
        {
            return instance != null && instance.NativeInstance != IntPtr.Zero;
        }

        /// <summary>
        /// 返回对对象的弱引用, 或 <see langword="null"/>
        /// 如果参数无效.
        ///一个弱引用可以持有一个Reference，而不会对引用计数器产生影响。
        ///可以使用 @GDScript.weakref从一个Object创建一个弱引用。
        ///如果这个对象不是一个引用，弱引用仍然可以工作，但是，
        ///它对这个对象没有任何影响。在多个类有相互引用的变量的情况下，
        ///弱引用是很有用的。如果没有弱引用，使用这些类可能会导致内存泄漏，
        ///因为两个引用都会使对方不被释放。将变量的一部分变成弱引用可以防止这种循环依赖，
        ///并允许引用被释放。
        /// </summary>
        /// <param name="obj">对象.</param>
        /// <returns>
        /// The <see cref="WeakRef"/> 引用对象或 <see langword="null"/>.
        /// </returns>
        public static WeakRef WeakRef(Object obj)
        {
            return godot_icall_Object_weakref(GetPtr(obj));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern WeakRef godot_icall_Object_weakref(IntPtr obj);
    }
}
