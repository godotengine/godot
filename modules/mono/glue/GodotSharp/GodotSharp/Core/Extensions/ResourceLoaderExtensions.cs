using System;

namespace Godot
{
    public static partial class ResourceLoader
    {
        /// <summary>
        /// Loads a resource at the given <paramref name="path"/>, caching the result
        /// for further access.
        /// The registered <see cref="ResourceFormatLoader"/> instances are queried sequentially
        /// to find the first one which can handle the file's extension, and then attempt
        /// loading. If loading fails, the remaining ResourceFormatLoaders are also attempted.
        /// An optional <paramref name="typeHint"/> can be used to further specify the
        /// <see cref="Resource"/> type that should be handled by the <see cref="ResourceFormatLoader"/>.
        /// Anything that inherits from <see cref="Resource"/> can be used as a type hint,
        /// for example <see cref="Image"/>.
        /// The <paramref name="cacheMode"/> property defines whether and how the cache should
        /// be used or updated when loading the resource. See <see cref="CacheMode"/> for details.
        /// Returns an empty resource if no <see cref="ResourceFormatLoader"/> could handle the file.
        /// </summary>
        /// <exception cref="InvalidCastException">
        /// The loaded resource can't be casted to the given type <typeparamref name="T"/>.
        /// </exception>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Resource"/>.</typeparam>
        public static T Load<T>(string path, string typeHint = null, CacheMode cacheMode = CacheMode.Reuse) where T : class
        {
            return (T)(object)Load(path, typeHint, cacheMode);
        }
    }
}
