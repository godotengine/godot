// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2024 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Math/BVec16.h>

JPH_NAMESPACE_BEGIN

/// Helper class for implementing an UnorderedSet or UnorderedMap
/// Based on CppCon 2017: Matt Kulukundis "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"
/// See: https://www.youtube.com/watch?v=ncHmEUmJZf4
template <class Key, class KeyValue, class HashTableDetail, class Hash, class KeyEqual>
class HashTable
{
public:
	/// Properties
	using value_type = KeyValue;
	using size_type = uint32;
	using difference_type = ptrdiff_t;

private:
	/// Base class for iterators
	template <class Table, class Iterator>
	class IteratorBase
	{
	public:
        /// Properties
		using difference_type = typename Table::difference_type;
        using value_type = typename Table::value_type;
        using iterator_category = std::forward_iterator_tag;

		/// Copy constructor
							IteratorBase(const IteratorBase &inRHS) = default;

		/// Assignment operator
		IteratorBase &		operator = (const IteratorBase &inRHS) = default;

		/// Iterator at start of table
		explicit			IteratorBase(Table *inTable) :
			mTable(inTable),
			mIndex(0)
		{
			while (mIndex < mTable->mMaxSize && (mTable->mControl[mIndex] & cBucketUsed) == 0)
				++mIndex;
		}

		/// Iterator at specific index
							IteratorBase(Table *inTable, size_type inIndex) :
			mTable(inTable),
			mIndex(inIndex)
		{
		}

		/// Prefix increment
		Iterator &			operator ++ ()
		{
			JPH_ASSERT(IsValid());

			do
			{
				++mIndex;
			}
			while (mIndex < mTable->mMaxSize && (mTable->mControl[mIndex] & cBucketUsed) == 0);

			return static_cast<Iterator &>(*this);
		}

		/// Postfix increment
		Iterator			operator ++ (int)
		{
			Iterator result(mTable, mIndex);
			++(*this);
			return result;
		}

		/// Access to key value pair
		const KeyValue &	operator * () const
		{
			JPH_ASSERT(IsValid());
			return mTable->mData[mIndex];
		}

		/// Access to key value pair
		const KeyValue *	operator -> () const
		{
			JPH_ASSERT(IsValid());
			return mTable->mData + mIndex;
		}

		/// Equality operator
		bool				operator == (const Iterator &inRHS) const
		{
			return mIndex == inRHS.mIndex && mTable == inRHS.mTable;
		}

		/// Inequality operator
		bool				operator != (const Iterator &inRHS) const
		{
			return !(*this == inRHS);
		}

		/// Check that the iterator is valid
		bool				IsValid() const
		{
			return mIndex < mTable->mMaxSize
				&& (mTable->mControl[mIndex] & cBucketUsed) != 0;
		}

		Table *				mTable;
		size_type			mIndex;
	};

	/// Allocate space for the hash table
	void					AllocateTable(size_type inMaxSize)
	{
		JPH_ASSERT(mData == nullptr);

		mMaxSize = inMaxSize;
		mMaxLoad = uint32((cMaxLoadFactorNumerator * inMaxSize) / cMaxLoadFactorDenominator);
		size_type required_size = mMaxSize * (sizeof(KeyValue) + 1) + 15; // Add 15 bytes to mirror the first 15 bytes of the control values
		if constexpr (cNeedsAlignedAllocate)
			mData = reinterpret_cast<KeyValue *>(AlignedAllocate(required_size, alignof(KeyValue)));
		else
			mData = reinterpret_cast<KeyValue *>(Allocate(required_size));
		mControl = reinterpret_cast<uint8 *>(mData + mMaxSize);
	}

	/// Copy the contents of another hash table
	void					CopyTable(const HashTable &inRHS)
	{
		if (inRHS.empty())
			return;

		AllocateTable(inRHS.mMaxSize);

		// Copy control bytes
		memcpy(mControl, inRHS.mControl, mMaxSize + 15);

		// Copy elements
		uint index = 0;
		for (const uint8 *control = mControl, *control_end = mControl + mMaxSize; control != control_end; ++control, ++index)
			if (*control & cBucketUsed)
				::new (mData + index) KeyValue(inRHS.mData[index]);
		mSize = inRHS.mSize;
	}

	/// Grow the table to the next power of 2
	void					GrowTable()
	{
		// Calculate new size
		size_type new_max_size = max<size_type>(mMaxSize << 1, 16);
		if (new_max_size < mMaxSize)
		{
			JPH_ASSERT(false, "Overflow in hash table size, can't grow!");
			return;
		}

		// Move the old table to a temporary structure
		size_type old_max_size = mMaxSize;
		KeyValue *old_data = mData;
		const uint8 *old_control = mControl;
		mData = nullptr;
		mControl = nullptr;
		mSize = 0;
		mMaxSize = 0;
		mMaxLoad = 0;

		// Allocate new table
		AllocateTable(new_max_size);

		// Reset all control bytes
		memset(mControl, cBucketEmpty, mMaxSize + 15);

		if (old_data != nullptr)
		{
			// Copy all elements from the old table
			for (size_type i = 0; i < old_max_size; ++i)
				if (old_control[i] & cBucketUsed)
				{
					size_type index;
					KeyValue *element = old_data + i;
					JPH_IF_ENABLE_ASSERTS(bool inserted =) InsertKey</* AllowDeleted= */ false>(HashTableDetail::sGetKey(*element), index);
					JPH_ASSERT(inserted);
					::new (mData + index) KeyValue(std::move(*element));
					element->~KeyValue();
				}

			// Free memory
			if constexpr (cNeedsAlignedAllocate)
				AlignedFree(old_data);
			else
				Free(old_data);
		}
	}

protected:
	/// Get an element by index
	KeyValue &				GetElement(size_type inIndex) const
	{
		return mData[inIndex];
	}

	/// Insert a key into the map, returns true if the element was inserted, false if it already existed.
	/// outIndex is the index at which the element should be constructed / where it is located.
	template <bool AllowDeleted = true>
	bool					InsertKey(const Key &inKey, size_type &outIndex)
	{
		// Ensure we have enough space
		if (mSize + 1 >= mMaxLoad)
			GrowTable();

		// Calculate hash
		uint64 hash_value = Hash { } (inKey);

		// Split hash into control byte and index
		uint8 control = cBucketUsed | uint8(hash_value);
		size_type bucket_mask = mMaxSize - 1;
		size_type index = size_type(hash_value >> 7) & bucket_mask;

		BVec16 control16 = BVec16::sReplicate(control);
		BVec16 bucket_empty = BVec16::sZero();
		BVec16 bucket_deleted = BVec16::sReplicate(cBucketDeleted);

		// Keeps track of the index of the first deleted bucket we found
		constexpr size_type cNoDeleted = ~size_type(0);
		size_type first_deleted_index = cNoDeleted;

		// Linear probing
		KeyEqual equal;
		for (;;)
		{
			// Read 16 control values (note that we added 15 bytes at the end of the control values that mirror the first 15 bytes)
			BVec16 control_bytes = BVec16::sLoadByte16(mControl + index);

			// Check for the control value we're looking for
			uint32 control_equal = uint32(BVec16::sEquals(control_bytes, control16).GetTrues());

			// Check for empty buckets
			uint32 control_empty = uint32(BVec16::sEquals(control_bytes, bucket_empty).GetTrues());

			// Check if we're still scanning for deleted buckets
			if constexpr (AllowDeleted)
				if (first_deleted_index == cNoDeleted)
				{
					// Check if any buckets have been deleted, if so store the first one
					uint32 control_deleted = uint32(BVec16::sEquals(control_bytes, bucket_deleted).GetTrues());
					if (control_deleted != 0)
						first_deleted_index = index + CountTrailingZeros(control_deleted);
				}

			// Index within the 16 buckets
			size_type local_index = index;

			// Loop while there's still buckets to process
			while ((control_equal | control_empty) != 0)
			{
				// Get the index of the first bucket that is either equal or empty
				uint first_equal = CountTrailingZeros(control_equal);
				uint first_empty = CountTrailingZeros(control_empty);

				// Check if we first found a bucket with equal control value before an empty bucket
				if (first_equal < first_empty)
				{
					// Skip to the bucket
					local_index += first_equal;

					// Make sure that our index is not beyond the end of the table
					local_index &= bucket_mask;

					// We found a bucket with same control value
					if (equal(HashTableDetail::sGetKey(mData[local_index]), inKey))
					{
						// Element already exists
						outIndex = local_index;
						return false;
					}

					// Skip past this bucket
					local_index++;
					uint shift = first_equal + 1;
					control_equal >>= shift;
					control_empty >>= shift;
				}
				else
				{
					// An empty bucket was found, we can insert a new item
					JPH_ASSERT(control_empty != 0);

					// Get the location of the first empty or deleted bucket
					local_index += first_empty;
					if constexpr (AllowDeleted)
						if (first_deleted_index < local_index)
							local_index = first_deleted_index;

					// Make sure that our index is not beyond the end of the table
					local_index &= bucket_mask;

					// Update control byte
					mControl[local_index] = control;
					if (local_index < 15)
						mControl[mMaxSize + local_index] = control; // Mirror the first 15 bytes at the end of the control values
					++mSize;

					// Return index to newly allocated bucket
					outIndex = local_index;
					return true;
				}
			}

			// Move to next batch of 16 buckets
			index = (index + 16) & bucket_mask;
		}
	}

public:
	/// Non-const iterator
	class iterator : public IteratorBase<HashTable, iterator>
	{
		using Base = IteratorBase<HashTable, iterator>;

	public:
        /// Properties
        using reference = typename Base::value_type &;
        using pointer = typename Base::value_type *;

		/// Constructors
		explicit			iterator(HashTable *inTable) : Base(inTable) { }
							iterator(HashTable *inTable, size_type inIndex) : Base(inTable, inIndex) { }
							iterator(const iterator &inIterator) : Base(inIterator) { }

		/// Assignment
		iterator &			operator = (const iterator &inRHS) { Base::operator = (inRHS); return *this; }

		using Base::operator *;

		/// Non-const access to key value pair
		KeyValue &			operator * ()
		{
			JPH_ASSERT(this->IsValid());
			return this->mTable->mData[this->mIndex];
		}

		using Base::operator ->;

		/// Non-const access to key value pair
		KeyValue *			operator -> ()
		{
			JPH_ASSERT(this->IsValid());
			return this->mTable->mData + this->mIndex;
		}
	};

	/// Const iterator
	class const_iterator : public IteratorBase<const HashTable, const_iterator>
	{
		using Base = IteratorBase<const HashTable, const_iterator>;

	public:
        /// Properties
        using reference = const typename Base::value_type &;
        using pointer = const typename Base::value_type *;

		/// Constructors
		explicit			const_iterator(const HashTable *inTable) : Base(inTable) { }
							const_iterator(const HashTable *inTable, size_type inIndex) : Base(inTable, inIndex) { }
							const_iterator(const const_iterator &inRHS) : Base(inRHS) { }
							const_iterator(const iterator &inIterator) : Base(inIterator.mTable, inIterator.mIndex) { }

		/// Assignment
		const_iterator &	operator = (const iterator &inRHS) { this->mTable = inRHS.mTable; this->mIndex = inRHS.mIndex; return *this; }
		const_iterator &	operator = (const const_iterator &inRHS) { Base::operator = (inRHS); return *this; }
	};

	/// Default constructor
							HashTable() = default;

	/// Copy constructor
							HashTable(const HashTable &inRHS)
	{
		CopyTable(inRHS);
	}

	/// Move constructor
							HashTable(HashTable &&ioRHS) noexcept :
		mData(ioRHS.mData),
		mControl(ioRHS.mControl),
		mSize(ioRHS.mSize),
		mMaxSize(ioRHS.mMaxSize),
		mMaxLoad(ioRHS.mMaxLoad)
	{
		ioRHS.mData = nullptr;
		ioRHS.mControl = nullptr;
		ioRHS.mSize = 0;
		ioRHS.mMaxSize = 0;
		ioRHS.mMaxLoad = 0;
	}

	/// Assignment operator
	HashTable &				operator = (const HashTable &inRHS)
	{
		if (this != &inRHS)
		{
			clear();

			CopyTable(inRHS);
		}

		return *this;
	}

	/// Destructor
							~HashTable()
	{
		clear();
	}

	/// Reserve memory for a certain number of elements
	void					reserve(size_type inMaxSize)
	{
		// Calculate max size based on load factor
		size_type max_size = GetNextPowerOf2(max<uint32>((cMaxLoadFactorDenominator * inMaxSize) / cMaxLoadFactorNumerator, 16));
		if (max_size <= mMaxSize)
			return;

		// Allocate buffers
		AllocateTable(max_size);

		// Reset all control bytes
		memset(mControl, cBucketEmpty, mMaxSize + 15);
	}

	/// Destroy the entire hash table
	void					clear()
	{
		// Delete all elements
		if constexpr (!is_trivially_destructible<KeyValue>())
			if (!empty())
				for (size_type i = 0; i < mMaxSize; ++i)
					if (mControl[i] & cBucketUsed)
						mData[i].~KeyValue();

		if (mData != nullptr)
		{
			// Free memory
			if constexpr (cNeedsAlignedAllocate)
				AlignedFree(mData);
			else
				Free(mData);

			// Reset members
			mData = nullptr;
			mControl = nullptr;
			mSize = 0;
			mMaxSize = 0;
			mMaxLoad = 0;
		}
	}

	/// Iterator to first element
	iterator				begin()
	{
		return iterator(this);
	}

	/// Iterator to one beyond last element
	iterator				end()
	{
		return iterator(this, mMaxSize);
	}

	/// Iterator to first element
	const_iterator			begin() const
	{
		return const_iterator(this);
	}

	/// Iterator to one beyond last element
	const_iterator			end() const
	{
		return const_iterator(this, mMaxSize);
	}

	/// Iterator to first element
	const_iterator			cbegin() const
	{
		return const_iterator(this);
	}

	/// Iterator to one beyond last element
	const_iterator			cend() const
	{
		return const_iterator(this, mMaxSize);
	}

	/// Check if there are no elements in the table
	bool					empty() const
	{
		return mSize == 0;
	}

	/// Number of elements in the table
	size_type				size() const
	{
		return mSize;
	}

	/// Insert a new element, returns iterator and if the element was inserted
	std::pair<iterator, bool> insert(const value_type &inValue)
	{
		size_type index;
		bool inserted = InsertKey(HashTableDetail::sGetKey(inValue), index);
		if (inserted)
			::new (mData + index) KeyValue(inValue);
		return std::make_pair(iterator(this, index), inserted);
	}

	/// Find an element, returns iterator to element or end() if not found
	const_iterator			find(const Key &inKey) const
	{
		// Check if we have any data
		if (empty())
			return cend();

		// Calculate hash
		uint64 hash_value = Hash { } (inKey);

		// Split hash into control byte and index
		uint8 control = cBucketUsed | uint8(hash_value);
		size_type bucket_mask = mMaxSize - 1;
		size_type index = size_type(hash_value >> 7) & bucket_mask;

		BVec16 control16 = BVec16::sReplicate(control);
		BVec16 bucket_empty = BVec16::sZero();

		// Linear probing
		KeyEqual equal;
		for (;;)
		{
			// Read 16 control values (note that we added 15 bytes at the end of the control values that mirror the first 15 bytes)
			BVec16 control_bytes = BVec16::sLoadByte16(mControl + index);

			// Check for the control value we're looking for
			uint32 control_equal = uint32(BVec16::sEquals(control_bytes, control16).GetTrues());

			// Check for empty buckets
			uint32 control_empty = uint32(BVec16::sEquals(control_bytes, bucket_empty).GetTrues());

			// Index within the 16 buckets
			size_type local_index = index;

			// Loop while there's still buckets to process
			while ((control_equal | control_empty) != 0)
			{
				// Get the index of the first bucket that is either equal or empty
				uint first_equal = CountTrailingZeros(control_equal);
				uint first_empty = CountTrailingZeros(control_empty);

				// Check if we first found a bucket with equal control value before an empty bucket
				if (first_equal < first_empty)
				{
					// Skip to the bucket
					local_index += first_equal;

					// Make sure that our index is not beyond the end of the table
					local_index &= bucket_mask;

					// We found a bucket with same control value
					if (equal(HashTableDetail::sGetKey(mData[local_index]), inKey))
					{
						// Element found
						return const_iterator(this, local_index);
					}

					// Skip past this bucket
					local_index++;
					uint shift = first_equal + 1;
					control_equal >>= shift;
					control_empty >>= shift;
				}
				else
				{
					// An empty bucket was found, we didn't find the element
					JPH_ASSERT(control_empty != 0);
					return cend();
				}
			}

			// Move to next batch of 16 buckets
			index = (index + 16) & bucket_mask;
		}
	}

	/// @brief Erase an element by iterator
	void					erase(const const_iterator &inIterator)
	{
		JPH_ASSERT(inIterator.IsValid());

		// Mark the bucket as deleted
		mControl[inIterator.mIndex] = cBucketDeleted;
		if (inIterator.mIndex < 15)
			mControl[inIterator.mIndex + mMaxSize] = cBucketDeleted;

		// Destruct the element
		mData[inIterator.mIndex].~KeyValue();

		// Decrease size
		--mSize;
	}

	/// @brief Erase an element by key
	size_type				erase(const Key &inKey)
	{
		const_iterator it = find(inKey);
		if (it == cend())
			return 0;

		erase(it);
		return 1;
	}

	/// Swap the contents of two hash tables
	void					swap(HashTable &ioRHS) noexcept
	{
		std::swap(mData, ioRHS.mData);
		std::swap(mControl, ioRHS.mControl);
		std::swap(mSize, ioRHS.mSize);
		std::swap(mMaxSize, ioRHS.mMaxSize);
		std::swap(mMaxLoad, ioRHS.mMaxLoad);
	}

private:
	/// If this allocator needs to fall back to aligned allocations because the type requires it
	static constexpr bool	cNeedsAlignedAllocate = alignof(KeyValue) > (JPH_CPU_ADDRESS_BITS == 32? 8 : 16);

	/// Max load factor is cMaxLoadFactorNumerator / cMaxLoadFactorDenominator
	static constexpr uint64	cMaxLoadFactorNumerator = 7;
	static constexpr uint64	cMaxLoadFactorDenominator = 8;

	/// Values that the control bytes can have
	static constexpr uint8	cBucketEmpty = 0;
	static constexpr uint8	cBucketDeleted = 0x7f;
	static constexpr uint8	cBucketUsed = 0x80;	// Lowest 7 bits are lowest 7 bits of the hash value

	/// The buckets, an array of size mMaxSize
	KeyValue *				mData = nullptr;

	/// Control bytes, an array of size mMaxSize + 15
	uint8 *					mControl = nullptr;

	/// Number of elements in the table
	size_type				mSize = 0;

	/// Max number of elements that can be stored in the table
	size_type				mMaxSize = 0;

	/// Max number of elements in the table before it should grow
	size_type				mMaxLoad = 0;
};

JPH_NAMESPACE_END
