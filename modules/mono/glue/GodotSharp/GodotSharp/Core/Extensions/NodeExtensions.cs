using System;

namespace Godot
{
    public partial class Node
    {
        /// <summary>
        ///获取一个节点。NodePath 可以是一个相对路径（从当前节点），
        ///也可以是一个绝对路径(在场景树中)。如果路径不存在，则返回
        ///null instance，并记录错误。尝试访问返回值上的方法将导致
        ///“Attempt to call(method) on a null instance.”错误。
        ///注意：获取绝对路径只在节点在场景树中时生效（请参阅 is_inside_tree()）
        /// </summary>
        /// <example>
        /// 示例：假设你当前的节点是 Character，并且有一下树结构:
        /// <code>
        /// /root
        /// /root/Character
        /// /root/Character/Sword
        /// /root/Character/Backpack/Dagger
        /// /root/MyGame
        /// /root/Swamp/Alligator
        /// /root/Swamp/Mosquito
        /// /root/Swamp/Goblin
        /// </code>
        /// 可能的路径会是下面这样：
        /// <code>
        /// GetNode("Sword");
        /// GetNode("Backpack/Dagger");
        /// GetNode("../Swamp/Alligator");
        /// GetNode("/root/MyGame");
        /// </code>
        /// </example>
        /// <seealso cref="GetNodeOrNull{T}(NodePath)"/>
        /// <param name="path">要获取的节点的路径.</param>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the fetched node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The <see cref="Node"/> at the given <paramref name="path"/>.
        /// </returns>
        public T GetNode<T>(NodePath path) where T : class
        {
            return (T)(object)GetNode(path);
        }

        /// <summary>
        ///获取一个节点。NodePath 可以是一个相对路径（从当前节点），
        ///也可以是一个绝对路径(在场景树中)。如果路径不存在，则返回
        ///null instance，并记录错误。尝试访问返回值上的方法将导致
        ///“Attempt to call(method) on a null instance.”错误。
        ///注意：获取绝对路径只在节点在场景树中时生效（请参阅 is_inside_tree()）
        /// </summary>
        /// <example>
        /// 示例：假设你当前的节点是 Character，并且有一下树结构:
        /// <code>
        /// /root
        /// /root/Character
        /// /root/Character/Sword
        /// /root/Character/Backpack/Dagger
        /// /root/MyGame
        /// /root/Swamp/Alligator
        /// /root/Swamp/Mosquito
        /// /root/Swamp/Goblin
        /// </code>
        /// 可能的路径会是下面这样：
        /// <code>
        /// GetNode("Sword");
        /// GetNode("Backpack/Dagger");
        /// GetNode("../Swamp/Alligator");
        /// GetNode("/root/MyGame");
        /// </code>
        /// </example>
        /// <seealso cref="GetNode{T}(NodePath)"/>
        /// <param name="path">要获取的节点的路径.</param>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The <see cref="Node"/> at the given <paramref name="path"/>, or <see langword="null"/> if not found.
        /// </returns>
        public T GetNodeOrNull<T>(NodePath path) where T : class
        {
            return GetNodeOrNull(path) as T;
        }

        /// <summary>
        /// 按索引返回一个子节点  <see cref="GetChildCount"/>.
        /// 这个方法经常被用于遍历一个节点的所有子节点。
        /// 要通过一个子节点的名字访问它，请使用 <见 cref="GetNode"/>.
        /// </summary>
        /// <seealso cref="GetChildOrNull{T}(int)"/>
        /// <param name="idx">子节点索引.</param>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the fetched node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The child <see cref="Node"/> at the given index <paramref name="idx"/>.
        /// </returns>
        public T GetChild<T>(int idx) where T : class
        {
            return (T)(object)GetChild(idx);
        }

        /// <summary>
        /// 按索引返回一个子节点 <see cref="GetChildCount"/>.
        /// 这个方法经常被用于遍历一个节点的所有子节点。
        /// 要通过一个子节点的名字访问它，请使用 <see cref="GetNode"/>.
        /// </summary>
        /// <seealso cref="GetChild{T}(int)"/>
        /// <param name="idx">子节点索引.</param>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The child <see cref="Node"/> at the given index <paramref name="idx"/>, or <see langword="null"/> if not found.
        /// </returns>
        public T GetChildOrNull<T>(int idx) where T : class
        {
            return GetChild(idx) as T;
        }

        /// <summary>
        /// The node owner. A node can have any other node as owner (as long as it is
        /// a valid parent, grandparent, etc. ascending in the tree). When saving a
        /// node (using <see cref="PackedScene"/>), all the nodes it owns will be saved
        /// with it. This allows for the creation of complex <see cref="SceneTree"/>s,
        /// with instancing and subinstancing.
        /// </summary>
        /// <seealso cref="GetOwnerOrNull{T}"/>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the fetched node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The owner <see cref="Node"/>.
        /// </returns>
        public T GetOwner<T>() where T : class
        {
            return (T)(object)Owner;
        }

        /// <summary>
        /// The node owner. A node can have any other node as owner (as long as it is
        /// a valid parent, grandparent, etc. ascending in the tree). When saving a
        /// node (using <see cref="PackedScene"/>), all the nodes it owns will be saved
        /// with it. This allows for the creation of complex <see cref="SceneTree"/>s,
        /// with instancing and subinstancing.
        /// </summary>
        /// <seealso cref="GetOwner{T}"/>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The owner <see cref="Node"/>, or <see langword="null"/> if there is no owner.
        /// </returns>
        public T GetOwnerOrNull<T>() where T : class
        {
            return Owner as T;
        }

        /// <summary>
        /// 返回当前节点的父节点，如果节点缺少父节点，则返回 <see langword="null"/> instance
        /// 如果节点缺少父节点.
        /// </summary>
        /// <seealso cref="GetParentOrNull{T}"/>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the fetched node can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The parent <see cref="Node"/>.
        /// </returns>
        public T GetParent<T>() where T : class
        {
            return (T)(object)GetParent();
        }

        /// <summary>
        /// 返回当前节点的父节点，如果节点缺少父节点，则返回 <see langword="null"/> instance
        /// 如果节点缺少父节点.
        /// </summary>
        /// <seealso cref="GetParent{T}"/>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        /// <returns>
        /// The parent <see cref="Node"/>, or <see langword="null"/> if the node has no parent.
        /// </returns>
        public T GetParentOrNull<T>() where T : class
        {
            return GetParent() as T;
        }
    }
}
