#pragma once

#ifdef GDJ_CONFIG_EDITOR

class JoltStreamOutWrapper final : public JPH::StreamOut {
public:
	explicit JoltStreamOutWrapper(const Ref<FileAccess>& p_file_access)
		: file_access(p_file_access) { }

	void WriteBytes(const void* p_data, size_t p_bytes) override {
		file_access->store_buffer(
			static_cast<const uint8_t*>(p_data),
			static_cast<uint64_t>(p_bytes)
		);
	}

	bool IsFailed() const override { return file_access->get_error() != OK; }

private:
	Ref<FileAccess> file_access;
};

class JoltStreamInWrapper final : public JPH::StreamIn {
public:
	explicit JoltStreamInWrapper(const Ref<FileAccess>& p_file_access)
		: file_access(p_file_access) { }

	void ReadBytes(void* p_data, size_t p_bytes) override {
		file_access->get_buffer(static_cast<uint8_t*>(p_data), static_cast<uint64_t>(p_bytes));
	}

	bool IsEOF() const override { return file_access->eof_reached(); }

	bool IsFailed() const override { return file_access->get_error() != OK; }

private:
	Ref<FileAccess> file_access;
};

#endif // GDJ_CONFIG_EDITOR
