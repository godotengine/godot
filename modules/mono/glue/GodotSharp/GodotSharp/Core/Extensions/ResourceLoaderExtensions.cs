using System;

namespace Godot
{
    public static partial class ResourceLoader
    {
        /// <summary>
        /// 在给定的路径path上加载一个资源，缓存结果以便进一步访问。
        ///依次查询注册的 <see cref="ResourceFormatLoader"/> ，找到第一个可以处理该文件扩展名的加载器，
        ///然后尝试加载。如果加载失败，其余的 <see cref="ResourceFormatLoader"/> 也会被尝试。
        ///一个可选的 <paramref name="typeHint"/> 类型提示可以用来进一步指定 <see cref="ResourceFormatLoader"/> 应处理的 <see cref="Resource"/> 资源类型。
        ///任何继承自 <see cref="Resource"/> 的东西都可以被用作类型提示，例如图像 <see cref="Image"/>。
        ///如果no_cache是true，资源缓存将被绕过，资源将被重新加载。否则，如果缓存的资源存在，将被返回。
        ///如果没有 <see cref="ResourceFormatLoader"/> 可以处理该文件，则返回一个空资源。
        ///C#有一个简化的 GD.load()内置方法，可以在大多数情况下使用，把ResourceLoader的使用留给更高级的场景。
        /// </summary>
        /// <exception cref="InvalidCastException">
        /// Thrown when the given the loaded resource can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Resource"/>.</typeparam>
        public static T Load<T>(string path, string typeHint = null, bool noCache = false) where T : class
        {
            return (T)(object)Load(path, typeHint, noCache);
        }
    }
}
