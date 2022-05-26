using System;

namespace Godot
{
    public partial class PackedScene
    {
        /// <summary>
        /// 实例化场景的节点层次结构，失败时出错.
        /// 触发子场景实例化。 触发一个
        /// <see cref="Node.NotificationInstanced"/> 根节点上的通知.
        /// </summary>
        /// <seealso cref="InstanceOrNull{T}(GenEditState)"/>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the instantiated node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>实例化的场景.</returns>
        public T Instance<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return (T)(object)Instance(editState);
        }

        /// <summary>
        /// 实例化场景的节点层次结构，失败时出错.
        /// 触发子场景实例化。 触发一个
        /// <see cref="Node.NotificationInstanced"/> 根节点上的通知.
        /// </summary>
        /// <seealso cref="Instance{T}(GenEditState)"/>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>实例化的场景.</returns>
        public T InstanceOrNull<T>(PackedScene.GenEditState editState = (PackedScene.GenEditState)0) where T : class
        {
            return Instance(editState) as T;
        }
    }
}
