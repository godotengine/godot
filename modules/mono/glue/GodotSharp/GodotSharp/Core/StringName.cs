using System;
using Godot.NativeInterop;

namespace Godot
{
    /// <summary>
    /// StringNames are immutable strings designed for general-purpose representation of unique names.
    /// StringName ensures that only one instance of a given name exists (so two StringNames with the
    /// same value are the same object).
    /// Comparing them is much faster than with regular strings, because only the pointers are compared,
    /// not the whole strings.
    /// </summary>
    public sealed class StringName : IDisposable, IEquatable<StringName>
    {
        internal godot_string_name.movable NativeValue;

        private WeakReference<IDisposable> _weakReferenceToSelf;

        /// <summary>
        /// Deconstructs this <see cref="StringName"/>.
        /// </summary>
        ~StringName()
        {
            Dispose(false);
        }

        /// <summary>
        /// Disposes of this <see cref="StringName"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes implementation of this <see cref="StringName"/>.
        /// </summary>
        public void Dispose(bool disposing)
        {
            // Always dispose `NativeValue` even if disposing is true
            NativeValue.DangerousSelfRef.Dispose();

            if (_weakReferenceToSelf != null)
            {
                DisposablesTracker.UnregisterDisposable(_weakReferenceToSelf);
            }
        }

        private StringName(godot_string_name nativeValueToOwn)
        {
            NativeValue = (godot_string_name.movable)nativeValueToOwn;
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
        }

        // Explicit name to make it very clear
        internal static StringName CreateTakingOwnershipOfDisposableValue(godot_string_name nativeValueToOwn)
            => new StringName(nativeValueToOwn);

        /// <summary>
        /// Constructs an empty <see cref="StringName"/>.
        /// </summary>
        public StringName()
        {
        }

        /// <summary>
        /// Constructs a <see cref="StringName"/> from the given <paramref name="name"/> string.
        /// </summary>
        /// <param name="name">String to construct the <see cref="StringName"/> from.</param>
        public StringName(string name)
        {
            if (!string.IsNullOrEmpty(name))
            {
                NativeValue = (godot_string_name.movable)NativeFuncs.godotsharp_string_name_new_from_string(name);
                _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
            }
        }

        /// <summary>
        /// Converts a string to a <see cref="StringName"/>.
        /// </summary>
        /// <param name="from">The string to convert.</param>
        public static implicit operator StringName(string from) => new StringName(from);

        /// <summary>
        /// Converts a <see cref="StringName"/> to a string.
        /// </summary>
        /// <param name="from">The <see cref="StringName"/> to convert.</param>
        public static implicit operator string(StringName from) => from?.ToString();

        /// <summary>
        /// Converts this <see cref="StringName"/> to a string.
        /// </summary>
        /// <returns>A string representation of this <see cref="StringName"/>.</returns>
        public override string ToString()
        {
            if (IsEmpty)
                return string.Empty;

            var src = (godot_string_name)NativeValue;
            NativeFuncs.godotsharp_string_name_as_string(out godot_string dest, src);
            using (dest)
                return Marshaling.ConvertStringToManaged(dest);
        }

        /// <summary>
        /// Check whether this <see cref="StringName"/> is empty.
        /// </summary>
        /// <returns>If the <see cref="StringName"/> is empty.</returns>
        public bool IsEmpty => NativeValue.DangerousSelfRef.IsEmpty;

        /// <summary>
        /// Evaluates if the <see cref="StringName"/> instances are exactly equal.
        /// </summary>
        /// <param name="left">The left <see cref="StringName"/>.</param>
        /// <param name="right">The right <see cref="StringName"/>.</param>
        /// <returns><see langword="true"/> if these <see cref="StringName"/> are
        /// exactly equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator ==(StringName left, StringName right)
        {
            if (left is null)
                return right is null;
            return left.Equals(right);
        }

        /// <summary>
        /// Evaluates if the <see cref="StringName"/> instances are not equal.
        /// </summary>
        /// <param name="left">The left <see cref="StringName"/>.</param>
        /// <param name="right">The right <see cref="StringName"/>.</param>
        /// <returns><see langword="true"/> if these <see cref="StringName"/> are
        /// not equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator !=(StringName left, StringName right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Evaluates if the <see cref="StringName"/> instances are exactly equal.
        /// </summary>
        /// <param name="other">The <see cref="StringName"/> to compare with.</param>
        /// <returns><see langword="true"/> if these <see cref="StringName"/> are
        /// exactly equal; otherwise, <see langword="false"/>.</returns>
        public bool Equals(StringName other)
        {
            if (other is null)
                return false;
            return NativeValue.DangerousSelfRef == other.NativeValue.DangerousSelfRef;
        }

        /// <summary>
        /// Evaluates if this <see cref="StringName"/> is exactly equal to the
        /// given <see cref="godot_string_name"/>
        /// </summary>
        /// <param name="left">The left <see cref="StringName"/>.</param>
        /// <param name="right">The right <see cref="godot_string_name"/>.</param>
        /// <returns><see langword="true"/> if this <see cref="StringName"/>and the
        /// <see cref="godot_string_name"/> are equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator ==(StringName left, in godot_string_name right)
        {
            if (left is null)
                return right.IsEmpty;
            return left.Equals(right);
        }

        /// <summary>
        /// Evaluates if this <see cref="StringName"/> is not equal to the
        /// given <see cref="godot_string_name"/>
        /// </summary>
        /// <param name="left">The left <see cref="StringName"/>.</param>
        /// <param name="right">The right <see cref="godot_string_name"/>.</param>
        /// <returns><see langword="true"/> if this <see cref="StringName"/>and the
        /// <see cref="godot_string_name"/> are equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator !=(StringName left, in godot_string_name right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_string_name"/> is exactly equal to the
        /// given <see cref="StringName"/>
        /// </summary>
        /// <param name="left">The left <see cref="godot_string_name"/>.</param>
        /// <param name="right">The right <see cref="StringName"/>.</param>
        /// <returns><see langword="true"/> if this <see cref="godot_string_name"/>and the
        /// <see cref="StringName"/> are equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator ==(in godot_string_name left, StringName right)
        {
            return right == left;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_string_name"/> is not equal to the
        /// given <see cref="StringName"/>
        /// </summary>
        /// <param name="left">The left <see cref="godot_string_name"/>.</param>
        /// <param name="right">The right <see cref="StringName"/>.</param>
        /// <returns><see langword="true"/> if this <see cref="godot_string_name"/>and the
        /// <see cref="StringName"/> are equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator !=(in godot_string_name left, StringName right)
        {
            return !(right == left);
        }

        /// <summary>
        /// Evaluates if this <see cref="StringName"/> is exactly equal to the
        /// given <see cref="godot_string_name"/>
        /// </summary>
        /// <param name="other">The <see cref="godot_string_name"/> to compare with.</param>
        /// <returns><see langword="true"/> if this <see cref="StringName"/>and the
        /// <see cref="godot_string_name"/> are equal; otherwise, <see langword="false"/>.</returns>
        public bool Equals(in godot_string_name other)
        {
            return NativeValue.DangerousSelfRef == other;
        }

        /// <summary>
        /// Evaluates if this <see cref="StringName"/> is exactly equal to the
        /// given object (<paramref name="obj"/>).
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns><see langword="true"/> if this <see cref="StringName"/>and
        /// the object are equal; otherwise, <see langword="false"/>.</returns>
        public override bool Equals(object obj)
        {
            return ReferenceEquals(this, obj) || (obj is StringName other && Equals(other));
        }

        /// <summary>
        /// Serves as the hash function for <see cref="StringName"/>.
        /// </summary>
        /// <returns>A hash code for this <see cref="StringName"/>.</returns>
        public override int GetHashCode()
        {
            return NativeValue.GetHashCode();
        }
    }
}
