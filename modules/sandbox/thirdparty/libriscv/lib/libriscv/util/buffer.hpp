#pragma once
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

/**
 * Container that is designed to hold pointers to guest data, which can
 * be sequentialized in various ways.
**/

namespace riscv
{
	struct Buffer
	{
		bool is_sequential() const noexcept { return m_overflow.empty(); }
		const std::string_view& strview() const noexcept { return m_data; }
		const char* c_str() const noexcept { return strview().data(); }
		const char* data() const noexcept { return strview().data(); }
		size_t      size() const noexcept { return m_len; }

		size_t copy_to(char* dst, size_t dstlen) const;
		void   copy_to(std::vector<uint8_t>&) const;
		void   foreach(std::function<void(const char*, size_t)> cb);
		std::string to_string() const;

		Buffer() = default;
		void append_page(const char* data, size_t len);

	private:
		std::string_view m_data;
		std::vector<std::pair<const char*, size_t>> m_overflow;
		size_t m_len  = 0; /* Total length */
	};

	inline size_t Buffer::copy_to(char* dst, size_t maxlen) const
	{
		if (UNLIKELY(m_data.size() > maxlen))
			return 0;
		size_t len = m_data.size();
		std::copy(m_data.begin(), m_data.end(), dst);

		for (const auto& entry : m_overflow) {
			if (UNLIKELY(len + entry.second > maxlen)) break;
			std::copy(entry.first, entry.first + entry.second, &dst[len]);
			len += entry.second;
		}

		return len;
	}
	inline void Buffer::copy_to(std::vector<uint8_t>& vec) const
	{
		vec.insert(vec.end(), m_data.begin(), m_data.end());
		for (const auto& entry : m_overflow) {
			vec.insert(vec.end(), entry.first, entry.first + entry.second);
		}
	}

	inline void Buffer::foreach(std::function<void(const char*, size_t)> cb)
	{
		cb(m_data.data(), m_data.size());
		for (const auto& entry : m_overflow) {
			cb(entry.first, entry.second);
		}
	}

	inline void Buffer::append_page(const char* buffer, size_t len)
	{
		if (m_data.empty())
		{
			m_data = {buffer, len};
			m_len = len;
			return;
		}
		else if (&*m_data.end() == buffer)
		{
			// In some cases we can continue the last entry
			m_len  += len;
			m_data = {m_data.data(), m_len};
			return;
		}

		// In some cases we can continue the last entry
		if (!m_overflow.empty()) {
			auto& last = m_overflow.back();
			if (last.first + last.second == buffer) {
				last.second += len;
				m_len += len;
				return;
			}
		}
		// Otherwise, append new entry
		m_len += len;
		m_overflow.emplace_back(buffer, len);
	}

	inline std::string Buffer::to_string() const
	{
		if (is_sequential()) {
			return std::string(m_data);
		}

		std::string result;
		result.reserve(this->m_len);
		result.append(m_data);
		for (const auto& entry : m_overflow) {
			result.append(entry.first, entry.first + entry.second);
		}
		return result;
	}
}
