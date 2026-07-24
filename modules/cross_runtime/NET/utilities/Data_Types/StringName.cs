using System;
using System.Diagnostics.CodeAnalysis;

#nullable enable
namespace Godot
{
	public sealed class StringName : IEquatable<StringName?>
	{
		private readonly string _value;

		public StringName()
		{
			_value = string.Empty;
		}

		public StringName(string name)
		{
			_value = string.IsNullOrEmpty(name) ? string.Empty : string.Intern(name);
		}

		public bool IsEmpty => _value.Length == 0;

		public string Value => _value;

		public static implicit operator StringName(string from) => new StringName(from);

		public static implicit operator string?(StringName? from) => from?._value;

		public static bool operator ==(StringName? left, StringName? right)
		{
			if (left is null)
				return right is null;
			return left.Equals(right);
		}

		public static bool operator !=(StringName? left, StringName? right)
		{
			return !(left == right);
		}

		public bool Equals([NotNullWhen(true)] StringName? other)
		{
			if (other is null)
				return false;

			return string.Equals(_value, other._value, StringComparison.Ordinal);
		}

		public override bool Equals([NotNullWhen(true)] object? obj)
		{
			return ReferenceEquals(this, obj) || (obj is StringName other && Equals(other));
		}

		public override int GetHashCode()
		{
			return StringComparer.Ordinal.GetHashCode(_value);
		}

		public override string ToString()
		{
			return _value;
		}
	}
}
