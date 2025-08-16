#include <libriscv/machine.hpp>

#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include "SDL.h"

static inline std::vector<uint8_t> load_file(const std::string &);
static constexpr auto ARCH = riscv::RISCV32;
using Machine = riscv::Machine<ARCH>;

/**
 * SDL Doom loosely based on port by @lcq2
 */
static constexpr size_t GAME_W = 320;
static constexpr size_t GAME_H = 200;
static SDL_Window* window = nullptr;
static SDL_Renderer* renderer = nullptr;
static SDL_Surface* screen_surface = nullptr;
static SDL_Texture* screen_tex = nullptr;
static struct {
	SDL_Surface* surface = nullptr;
	uint32_t g_pixels = 0x0;
	std::vector<uint8_t> buffer;
	std::vector<SDL_Color> palette;
} pixels;

struct DoomEvent {
	uint32_t type;
	uint32_t ts;

	union {
		struct {
			uint32_t scancode;
			int32_t  keycode;
		} k;
		struct {
			uint8_t  button;
			uint8_t  state;
			uint8_t  clicks;
			uint8_t  padding;
			int32_t  x;
			int32_t  y;
		} mb;
		struct {
			uint32_t state;
			int32_t  x;
			int32_t  y;
			int32_t  xrel;
			int32_t  yrel;
		} mm;
	};

	static const int KDOWN = 0;
	static const int KUP   = 1;
	static const int MUP   = 2;
	static const int MDOWN = 3;
	static const int MMOVE = 4;
	static const int QUIT = 5;
};

struct {
	uint32_t ts = 0;
	uint32_t last_fps_ts = 0;
	bool     restarting = false;
	uint16_t frames;
	uint64_t last_counter = 0;
	double   fps = 60.0;
	int64_t  adjust = 16700000L;
} stats;

static void do_rendering(Machine& machine)
{
	SDL_BlitSurface(pixels.surface, nullptr, screen_surface, nullptr);
	SDL_UpdateTexture(screen_tex, nullptr, screen_surface->pixels, screen_surface->pitch);
	SDL_RenderCopy(renderer, screen_tex, nullptr, nullptr);
	SDL_RenderPresent(renderer);

	stats.frames ++;

	// Every 100ms, calculate FPS
	const auto now = SDL_GetTicks();
	if (now >= stats.ts + 100) {
		machine.stop();
		// Perform stats outside of simulation in
		// order to get accurate instruction counter.
		stats.restarting = true;
	}

	// This greatly reduces CPU usage
	// Lower it if you have less than 60 fps.
	const struct timespec ts {
		.tv_sec = 0,
		.tv_nsec = stats.adjust
	};
	nanosleep(&ts, nullptr);
}

static void do_sdl_events(SDL_Event& event, DoomEvent& doomev, Machine& machine)
{
	doomev.ts = SDL_GetTicks();
	switch (event.type) {
		case SDL_KEYDOWN:
		case SDL_KEYUP:
			doomev.type =
				(event.type == SDL_KEYDOWN) ? DoomEvent::KDOWN : DoomEvent::KUP;
			doomev.k.scancode = event.key.keysym.scancode;
			doomev.k.keycode  = event.key.keysym.sym;
			break;
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
			doomev.type =
				(event.type == SDL_MOUSEBUTTONDOWN) ? DoomEvent::MDOWN : DoomEvent::MUP;
			doomev.mb.clicks = event.button.clicks;
			doomev.mb.state = event.button.state;
			doomev.mb.button = event.button.button;
			doomev.mb.x = event.button.x;
			doomev.mb.y = event.button.y;
			break;
		case SDL_MOUSEMOTION:
			doomev.type = DoomEvent::MMOVE;
			doomev.mm.state = event.motion.state;
			doomev.mm.x = event.motion.x;
			doomev.mm.y = event.motion.y;
			doomev.mm.xrel = event.motion.xrel;
			doomev.mm.yrel = event.motion.yrel;
			break;
		case SDL_QUIT:
			doomev.type = DoomEvent::QUIT;
			machine.stop();
			break;
	}
}

static void doom_system_calls(Machine& machine, size_t num)
{
	switch (num) {
	case 1024: { // newlib open
		auto [path, flags, mode] = machine.sysargs <std::string, int, int> ();
		if ((flags & O_ACCMODE) == O_WRONLY || (flags & O_ACCMODE) == O_RDWR) {
			// We only accept direct filenames, and savegame extension
			if (path.find("/") != std::string::npos) {
				machine.set_result_or_error(-1);
				return;
			}
			if (path.find(".dsg") == std::string::npos) {
				machine.set_result_or_error(-1);
				return;
			}
		}
		// Newlib doesn't think O_CREAT is required for new files :)
		if (flags & O_WRONLY) flags |= O_CREAT;

		int real_fd = open(path.c_str(), flags, mode);
		if (real_fd > 0)
		{
			const int vfd = machine.fds().assign_file(real_fd);
			machine.set_result(vfd);
		} else {
			machine.set_result_or_error(real_fd);
		}
		return;
	}
	case 2048: { // AV init
		const auto [width, height] = machine.sysargs<int, int> ();
		assert(width == GAME_W && height == GAME_H);
		machine.set_result(0);
		return;
	}
	case 2049: { // AV set_framebuffer
		const auto pixel_addr = machine.sysarg (0);
		pixels.g_pixels = pixel_addr;
		machine.set_result(0);
		return;
	}
	case 2050: { // AV update
		machine.copy_from_guest(
			pixels.buffer.data(), pixels.g_pixels, pixels.buffer.size());
		do_rendering(machine);
		machine.set_result(0);
		return;
	}
	case 2051: { // AV set_palette
		const auto [g_pal, count] = machine.sysargs<uint32_t, unsigned> ();
		if (count > 65536)
			riscv::MachineException(riscv::INVALID_PROGRAM, "Invalid number of palette colors");
		pixels.palette.resize(count);
		machine.copy_from_guest(pixels.palette.data(), g_pal, sizeof(SDL_Color) * count);
		SDL_SetPaletteColors(pixels.surface->format->palette, pixels.palette.data(), 0, count);
		machine.set_result(0);
		return;
	}
	case 2052: { // AV delay
		const auto millis = machine.sysarg(0);
		SDL_Delay(millis);
		return;
	}
	case 2053: { // AV poll_events
		auto [doomev] = machine.sysargs<DoomEvent> ();
		SDL_Event event;
		if (SDL_PollEvent(&event)) {
			do_sdl_events(event, doomev, machine);
			machine.copy_to_guest(machine.sysarg(0), &doomev, sizeof(doomev));
			machine.set_result(1);
		} else {
			machine.set_result(0);
		}
		return;
	}
	case 2054: // AV get_ticks
		machine.set_result<uint32_t>(SDL_GetTicks());
		return;
	case 2055: { // AV get_mousestate
		const auto [g_x, g_y] = machine.sysargs<uint32_t, uint32_t> ();
		int x, y;
		const uint32_t buttons = SDL_GetMouseState(&x, &y);
		if (g_x != 0x0)
			machine.copy_to_guest(g_x, &x, sizeof(x));
		if (g_y != 0x0)
			machine.copy_to_guest(g_y, &y, sizeof(y));
		machine.set_result<uint32_t>(buttons);
		return;
	}
	case 2056: { // AV warp_mouse
		const auto [x, y] = machine.sysargs<int, int> ();
		SDL_WarpMouseInWindow(window, x, y);
		machine.set_result(0);
		return;
	}
	case 2057: // AV shutdown
		machine.stop();
		return;
	default:
		printf("Unhandled system call: %zu\n", num);
	}
}

int main(int argc, char *argv[])
{
	const auto binary = load_file("doom-rv32g_b");
	Machine machine { binary };

	machine.setup_linux(
		{"rvdoom"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	machine.setup_linux_syscalls();
	machine.fds().proxy_mode = true;
	// Doom communicates intentions via some system calls
	machine.on_unhandled_syscall = doom_system_calls;

	SDL_Init(SDL_INIT_VIDEO);

	SDL_CreateWindowAndRenderer(
		GAME_W * 4,
		GAME_H * 4,
		SDL_WINDOW_RESIZABLE, &window, &renderer);
	SDL_SetWindowTitle(window, "RISC-V D00M");
	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "nearest");
	SDL_SetRelativeMouseMode(SDL_TRUE);

	pixels.buffer.resize(GAME_W * GAME_H * 4);
	pixels.surface = SDL_CreateRGBSurfaceFrom(
		pixels.buffer.data(),
		GAME_W, GAME_H, 8, GAME_W, 0, 0, 0, 0);

	screen_surface = SDL_CreateRGBSurface(0, GAME_W, GAME_H, 32, 0, 0, 0, 0);

	screen_tex = SDL_CreateTexture(renderer,
		SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, GAME_W, GAME_H);

	try {
		do {
			// Resume simulation:
			// Keep the instruction counter value without resetting it,
			// by instead setting the max counter relative to the
			// current instruction counter.
			machine.resume(50'000'000ull);

			// Statistics performed outside of simulation
			if (stats.restarting) {
				stats.restarting = false;
				const auto ts_now = SDL_GetTicks();
				const double secs = (ts_now - stats.ts) / 1000.;
				stats.fps  = stats.frames / secs;

				if (stats.fps < 59.0)
					stats.adjust -= 100000L;
				else if (stats.fps > 61.0)
					stats.adjust += 100000L;

				if (ts_now >= stats.last_fps_ts + 3000) {
					const auto mips = (machine.instruction_counter() - stats.last_counter) * 1e-6 / secs;
					stats.last_counter = machine.instruction_counter();

					stats.last_fps_ts = ts_now;
					printf("> Millions in/sec: %.2f  FPS: %.2f\n", mips, stats.fps);
				}

				stats.ts = ts_now;
				stats.frames = 0;
				continue;
			}
			break;
		} while (true);
	}
	catch (riscv::MachineException& me)
	{
		printf("%s\n", machine.cpu.current_instruction_to_string().c_str());
		printf(">>> Machine exception %d: %s (data: 0x%" PRIX64 ")\n",
				me.type(), me.what(), me.data());
		printf("%s\n", machine.cpu.registers().to_string().c_str());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
	}
	catch (const std::exception& e)
	{
		printf("Program exception: %s\n", e.what());
	}

	SDL_DestroyWindow(window);
	SDL_Quit();
}

#include <stdexcept>
#include <unistd.h>
std::vector<uint8_t> load_file(const std::string& filename)
{
	size_t size = 0;
	FILE* f = fopen(filename.c_str(), "rb");
	if (f == NULL) throw std::runtime_error("Could not open file: " + filename);

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);

	std::vector<uint8_t> result(size);
	if (size != fread(result.data(), 1, size, f))
	{
		fclose(f);
		throw std::runtime_error("Error when reading from file: " + filename);
	}
	fclose(f);
	return result;
}
