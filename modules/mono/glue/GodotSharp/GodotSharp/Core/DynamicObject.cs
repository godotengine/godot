using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// 表示一个 <see cref="Object"/> ，其成员可以在运行时通过 Variant API 动态访问。
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="DynamicGodotObject"/> 类允许访问 Variant
    /// <see cref="Object"/> 实例在运行时的成员。
    /// </para>
    /// <para>
    /// 这允许使用它们在引擎中的原始名称以及来自
    /// 附加到 <see cref="Object"/> 的脚本，不管它是用什么脚本语言编写的。
    /// </para>
    /// </remarks>
    /// <example>
    /// 这个示例展示了如何使用 <see cref="DynamicGodotObject"/> 来动态访问 <see cref="Object"/> 的引擎成员。
    /// <code>
    /// dynamic sprite = GetNode("Sprite").DynamicGodotObject;
    /// sprite.add_child(this);
    ///
    /// if ((sprite.hframes * sprite.vframes) &gt; 0)
    ///     sprite.frame = 0;
    /// </code>
    /// </example>
    /// <example>
    /// 此示例显示如何使用 <see cref="DynamicGodotObject"/> 动态访问附加到 <see cref="Object"/> 的脚本成员。
    /// <code>
    /// dynamic childNode = GetNode("ChildNode").DynamicGodotObject;
    ///
    /// if (childNode.print_allowed)
    /// {
    ///     childNode.message = "Hello from C#";
    ///     childNode.print_message(3);
    /// }
    /// </code>
    /// <c>ChildNode</c> 节点附加了以下 GDScript 脚本：
    /// <code>
    /// // # ChildNode.gd
    /// // var print_allowed = true
    /// // var message = ""
    /// //
    /// // func print_message(times):
    /// //     for i in times:
    /// //         print(message)
    /// </code>
    /// </example>
    public class DynamicGodotObject : DynamicObject
    {
        /// <summary>
        /// 获取与此 <see cref="DynamicGodotObject"/> 关联的 <see cref="Object"/>。
        /// </summary>
        public Object Value { get; }

        /// <summary>
        /// 初始化 <see cref="DynamicGodotObject"/> 类的新实例。
        /// </summary>
        /// <param name="godotObject">
        /// 将与此 <see cref="DynamicGodotObject"/> 关联的 <see cref="Object"/>。
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when the <paramref name="godotObject"/> parameter is <see langword="null"/>.
        /// </exception>
        public DynamicGodotObject(Object godotObject)
        {
            if (godotObject == null)
                throw new ArgumentNullException(nameof(godotObject));

            Value = godotObject;
        }

        /// <inheritdoc/>
        public override IEnumerable<string> GetDynamicMemberNames()
        {
            return godot_icall_DynamicGodotObject_SetMemberList(Object.GetPtr(Value));
        }

        /// <inheritdoc/>
        public override bool TryBinaryOperation(BinaryOperationBinder binder, object arg, out object result)
        {
            switch (binder.Operation)
            {
                case ExpressionType.Equal:
                case ExpressionType.NotEqual:
                    if (binder.ReturnType == typeof(bool) || binder.ReturnType.IsAssignableFrom(typeof(bool)))
                    {
                        if (arg == null)
                        {
                            bool boolResult = Object.IsInstanceValid(Value);

                            if (binder.Operation == ExpressionType.Equal)
                                boolResult = !boolResult;

                            result = boolResult;
                            return true;
                        }

                        if (arg is Object other)
                        {
                            bool boolResult = (Value == other);

                            if (binder.Operation == ExpressionType.NotEqual)
                                boolResult = !boolResult;

                            result = boolResult;
                            return true;
                        }
                    }

                    break;
                default:
                    // We're not implementing operators <, <=, >, and >= (LessThan, LessThanOrEqual, GreaterThan, GreaterThanOrEqual).
                    // These are used on the actual pointers in variant_op.cpp. It's better to let the user do that explicitly.
                    break;
            }

            return base.TryBinaryOperation(binder, arg, out result);
        }

        /// <inheritdoc/>
        public override bool TryConvert(ConvertBinder binder, out object result)
        {
            if (binder.Type == typeof(Object))
            {
                result = Value;
                return true;
            }

            if (typeof(Object).IsAssignableFrom(binder.Type))
            {
                // Throws InvalidCastException when the cast fails
                result = Convert.ChangeType(Value, binder.Type);
                return true;
            }

            return base.TryConvert(binder, out result);
        }

        /// <inheritdoc/>
        public override bool TryGetIndex(GetIndexBinder binder, object[] indexes, out object result)
        {
            if (indexes.Length == 1)
            {
                if (indexes[0] is string name)
                {
                    return godot_icall_DynamicGodotObject_GetMember(Object.GetPtr(Value), name, out result);
                }
            }

            return base.TryGetIndex(binder, indexes, out result);
        }

        /// <inheritdoc/>
        public override bool TryGetMember(GetMemberBinder binder, out object result)
        {
            return godot_icall_DynamicGodotObject_GetMember(Object.GetPtr(Value), binder.Name, out result);
        }

        /// <inheritdoc/>
        public override bool TryInvokeMember(InvokeMemberBinder binder, object[] args, out object result)
        {
            return godot_icall_DynamicGodotObject_InvokeMember(Object.GetPtr(Value), binder.Name, args, out result);
        }

        /// <inheritdoc/>
        public override bool TrySetIndex(SetIndexBinder binder, object[] indexes, object value)
        {
            if (indexes.Length == 1)
            {
                if (indexes[0] is string name)
                {
                    return godot_icall_DynamicGodotObject_SetMember(Object.GetPtr(Value), name, value);
                }
            }

            return base.TrySetIndex(binder, indexes, value);
        }

        /// <inheritdoc/>
        public override bool TrySetMember(SetMemberBinder binder, object value)
        {
            return godot_icall_DynamicGodotObject_SetMember(Object.GetPtr(Value), binder.Name, value);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string[] godot_icall_DynamicGodotObject_SetMemberList(IntPtr godotObject);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_DynamicGodotObject_InvokeMember(IntPtr godotObject, string name, object[] args, out object result);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_DynamicGodotObject_GetMember(IntPtr godotObject, string name, out object result);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_DynamicGodotObject_SetMember(IntPtr godotObject, string name, object value);

        #region We don't override these methods

        // Looks like this is not usable from C#
        //public override bool TryCreateInstance(CreateInstanceBinder binder, object[] args, out object result);

        // Object members cannot be deleted
        //public override bool TryDeleteIndex(DeleteIndexBinder binder, object[] indexes);
        //public override bool TryDeleteMember(DeleteMemberBinder binder);

        // Invocation on the object itself, e.g.: obj(param)
        //public override bool TryInvoke(InvokeBinder binder, object[] args, out object result);

        // No unnary operations to handle
        //public override bool TryUnaryOperation(UnaryOperationBinder binder, out object result);

        #endregion
    }
}
