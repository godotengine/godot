#include "machine.hpp"
#include "internal_common.hpp"

#include <inttypes.h>

namespace riscv
{
	template <int W>
	address_type<W> Memory<W>::elf_base_address(address_t offset) const {
		if (this->m_is_dynamic) {
			const address_t vaddr_base = DYLINK_BASE;
			if (UNLIKELY(vaddr_base + offset < vaddr_base))
				throw MachineException(INVALID_PROGRAM, "Bogus virtual address + offset");
			return vaddr_base + offset;
		} else {
			return offset;
		}
	}

	template <int W>
	const typename Elf<W>::Sym* Memory<W>::elf_sym_index(const typename Elf::SectionHeader* shdr, uint32_t symidx) const
	{
		if (symidx >= shdr->sh_size / sizeof(typename Elf::Sym))
#ifdef __EXCEPTIONS
			throw MachineException(INVALID_PROGRAM, "ELF Symtab section index overflow");
#else
			std::abort();
#endif
		auto* symtab = this->elf_offset<typename Elf::Sym>(shdr->sh_offset);
		return &symtab[symidx];
	}

	template <int W>
	const typename Elf<W>::SectionHeader* Memory<W>::section_by_name(const std::string& name) const
	{
		auto& elf = *elf_header();
		const auto sh_end_offset = elf.e_shoff + elf.e_shnum * sizeof(typename Elf::SectionHeader);

		if (elf.e_shoff > m_binary.size())
			throw MachineException(INVALID_PROGRAM, "Invalid section header offset", elf.e_shoff);
		if (sh_end_offset < elf.e_shoff || sh_end_offset > m_binary.size())
			throw MachineException(INVALID_PROGRAM, "Invalid section header offset", sh_end_offset);
		if (elf.e_shnum == 0 || elf.e_shnum > 64)
			throw MachineException(INVALID_PROGRAM, "Invalid section header count", elf.e_shnum);
		const auto* shdr = elf_offset<typename Elf::SectionHeader> (elf.e_shoff);

		if (elf.e_shstrndx >= elf.e_shnum)
			throw MachineException(INVALID_PROGRAM, "Invalid section header strtab index");

		const auto& shstrtab = shdr[elf.e_shstrndx];
		const char* strings = elf_offset<char>(shstrtab.sh_offset);
		const char* endptr = m_binary.data() + m_binary.size();

		for (auto i = 0; i < elf.e_shnum; i++)
		{
			// Bounds-check and overflow-check on sh_name from strtab sh_offset
			const auto name_offset = shstrtab.sh_offset + shdr[i].sh_name;
			if (name_offset < shstrtab.sh_offset || name_offset >= m_binary.size())
				throw MachineException(INVALID_PROGRAM, "Invalid ELF string offset");

			const char* shname = &strings[shdr[i].sh_name];
			const size_t len = strnlen(shname, endptr - shname);
			if (len != name.size())
				continue;

			if (strncmp(shname, name.c_str(), len) == 0) {
				return &shdr[i];
			}
		}
		return nullptr;
	}

	template <int W>
	const typename Elf<W>::Sym* Memory<W>::resolve_symbol(std::string_view name) const
	{
		if (UNLIKELY(m_binary.empty())) return nullptr;
		const auto* sym_hdr = section_by_name(".symtab");
		if (UNLIKELY(sym_hdr == nullptr)) return nullptr;
		const auto* str_hdr = section_by_name(".strtab");
		if (UNLIKELY(str_hdr == nullptr)) return nullptr;
		// ELF with no symbols
		if (UNLIKELY(sym_hdr->sh_size == 0)) return nullptr;

		const auto* symtab = elf_sym_index(sym_hdr, 0);
		const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
		const char* strtab = elf_offset<char>(str_hdr->sh_offset);

		for (size_t i = 0; i < symtab_ents; i++)
		{
			const char* symname = &strtab[symtab[i].st_name];
			if (name.compare(symname) == 0) {
				return &symtab[i];
			}
		}
		return nullptr;
	}

	template <int W>
	std::vector<const char*> Memory<W>::all_symbols() const
	{
		std::vector<const char*> symbols;
		if (UNLIKELY(m_binary.empty())) return symbols;
		const auto* sym_hdr = section_by_name(".symtab");
		const auto* str_hdr = section_by_name(".strtab");
		if (UNLIKELY(sym_hdr == nullptr || str_hdr == nullptr)) return symbols;
		// ELF with no symbols
		if (UNLIKELY(sym_hdr->sh_size == 0)) return symbols;

		const auto* symtab = elf_sym_index(sym_hdr, 0);
		const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
		const char* strtab = elf_offset<char>(str_hdr->sh_offset);
		symbols.reserve(symtab_ents);

		for (size_t i = 0; i < symtab_ents; i++)
		{
			const char* symname = &strtab[symtab[i].st_name];
			symbols.push_back(symname);
		}
		return symbols;
	}

	template <int W>
	std::vector<std::string_view> Memory<W>::all_unmangled_function_symbols(const std::string& prefix) const
	{
		std::vector<std::string_view> symbols;
		if (UNLIKELY(m_binary.empty())) return symbols;
		const auto* sym_hdr = section_by_name(".symtab");
		const auto* str_hdr = section_by_name(".strtab");
		if (UNLIKELY(sym_hdr == nullptr || str_hdr == nullptr)) return symbols;
		// ELF with no symbols
		if (UNLIKELY(sym_hdr->sh_size == 0)) return symbols;

		const auto* symtab = elf_sym_index(sym_hdr, 0);
		const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
		const char* strtab = elf_offset<char>(str_hdr->sh_offset);
		symbols.reserve(symtab_ents);

		for (size_t i = 0; i < symtab_ents; i++)
		{
			const char* symname = &strtab[symtab[i].st_name];
			if (Elf::SymbolType(symtab[i].st_info) == Elf::STT_FUNC && Elf::SymbolBind(symtab[i].st_info) != Elf::STB_WEAK) {
				std::string_view symview(symname);
				// Detect if the symbol is unmangled (no _Z prefix)
				if (symview.size() > 2 && !(symview[0] == '_' && symview[1] == 'Z')) {
					if (prefix.empty() || symview.compare(0, prefix.size(), prefix) == 0)
						symbols.push_back(symview);
				}
			}
		}
		return symbols;
	}

	template <int W>
	std::vector<std::string_view> Memory<W>::elf_comments() const
	{
		std::vector<std::string_view> comments;
		if (UNLIKELY(m_binary.empty())) return comments;
		const auto* hdr = elf_header();
		if (UNLIKELY(hdr == nullptr)) return comments;
		const auto* shdr = section_by_name(".comment");
		if (UNLIKELY(shdr == nullptr)) return comments;
		// ELF with no comments
		if (UNLIKELY(shdr->sh_size == 0)) return comments;

		const char* strtab = elf_offset<char>(shdr->sh_offset);
		const char* end = strtab + shdr->sh_size;
		const char* binary_end = m_binary.data() + m_binary.size(); // MSVC doesn't like m_binary.end()

		if (end < strtab || end > binary_end)
			throw MachineException(INVALID_PROGRAM, "Invalid ELF comment section");
		// Check if the comment section is null-terminated at the end
		if (UNLIKELY(end[-1] != '\0'))
			throw MachineException(INVALID_PROGRAM, "Invalid ELF comment section");
		// Use string_view to find each null-terminated comment
		while (strtab < end) {
			std::string_view comment(strtab);
			if (comment.empty()) {
				strtab++;
				continue;
			}
			comments.push_back(comment);
			strtab += comment.size() + 1;
			if (strtab >= binary_end)
				break;
		}
		return comments;
	}

	template <int W>
	static void elf_print_sym(const typename Elf<W>::Sym* sym)
	{
		if constexpr (W == 4) {
			printf("-> Sym is at 0x%" PRIX32 " with size %" PRIu32 ", type %u name %u\n",
				sym->st_value, sym->st_size,
				Elf<W>::SymbolType(sym->st_info), sym->st_name);
		} else {
			printf("-> Sym is at 0x%" PRIX64 " with size %" PRIu64 ", type %u name %u\n",
				(uint64_t)sym->st_value, sym->st_size,
				Elf<W>::SymbolType(sym->st_info), sym->st_name);
		}
	}

	template <int W> RISCV_INTERNAL
	void Memory<W>::relocate_section(const char* section_name, const char* sym_section)
	{
		using ElfRela = typename Elf::Rela;

		const auto* rela = section_by_name(section_name);
		if (rela == nullptr) return;
		const auto* dyn_hdr = section_by_name(sym_section);
		if (dyn_hdr == nullptr) return;
		const size_t rela_ents = rela->sh_size / sizeof(ElfRela);

		const auto rela_ents_offset = rela->sh_offset + rela_ents * sizeof(ElfRela);
		if (rela_ents_offset < rela->sh_offset || m_binary.size() < rela_ents_offset)
			throw MachineException(INVALID_PROGRAM, "Invalid ELF relocations");

		auto* rela_addr = elf_offset<ElfRela>(rela->sh_offset);
		for (size_t i = 0; i < rela_ents; i++)
		{
			size_t symidx;
			if constexpr (W == 4)
				symidx = Elf::RelaSym(rela_addr[i].r_info);
			else
				symidx = Elf::RelaSym(rela_addr[i].r_info);
			auto* sym = elf_sym_index(dyn_hdr, symidx);

			const uint8_t type = Elf::SymbolType(sym->st_info);
			if (true || type == Elf::STT_FUNC || type == Elf::STT_OBJECT)
			{
				if constexpr (false)
				{
					printf("Relocating rela %zu with sym idx %ld where 0x%lX -> 0x%lX\n",
							i, (long)symidx, (long)rela_addr[i].r_offset, (long)sym->st_value);
					elf_print_sym<W>(sym);
				}
				const auto rtype = Elf::RelaType(rela_addr[i].r_info);
				static constexpr int R_RISCV_64 = 0x2;
				static constexpr int R_RISCV_RELATIVE = 0x3;
				static constexpr int R_RISCV_JUMPSLOT = 0x5;
				if (rtype == 0) {
					// Do nothing
				}
				else if (rtype == R_RISCV_64) {
					this->write<address_t>(elf_base_address(rela_addr[i].r_offset), elf_base_address(sym->st_value));
				}
				else if (rtype == R_RISCV_RELATIVE) {
					this->write<address_t>(elf_base_address(rela_addr[i].r_offset), sym->st_value);
				}
				else if (rtype == R_RISCV_JUMPSLOT) {
					//typedef struct {
					//	address_t r_offset;
					//	address_t r_info;
					//} Elf64_Rel;
					//printf("Relocating jumpslot %zu with sym idx %ld where 0x%lX -> 0x%lX\n",
					//		i, (long)symidx, (long)rela_addr[i].r_offset, (long)sym->st_value);
					//const auto* plt = section_by_name(".plt");
					//if (plt == nullptr)
					//	throw MachineException(INVALID_PROGRAM, "Missing .plt section for jumpslot relocation");
					//const auto* plt_addr = elf_offset<Elf64_Rel>(plt->sh_offset);
					//const Elf64_Rel& plt_entry = plt_addr[sym->st_value / sizeof(Elf64_Rel)];
					//const auto plt_address = elf_base_address(plt_entry.r_offset);
					// Write the PLT address to the jumpslot
					//this->write<address_t>(elf_base_address(rela_addr[i].r_offset), plt_address);
				}
				else {
					throw MachineException(INVALID_PROGRAM, "Unknown relocation type", rtype);
				}
			}
		}
	}

	template <int W> RISCV_INTERNAL
	void Memory<W>::dynamic_linking(const typename Elf::Header& hdr)
	{
		(void)hdr;
		this->relocate_section(".rela.dyn", ".dynsym");
		this->relocate_section(".rela.plt", ".symtab");
	}

	INSTANTIATE_32_IF_ENABLED(Memory);
	INSTANTIATE_64_IF_ENABLED(Memory);
	INSTANTIATE_128_IF_ENABLED(Memory);
} // riscv
