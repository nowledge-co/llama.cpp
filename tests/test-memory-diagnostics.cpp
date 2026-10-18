#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#include "../ggml/src/ggml-metal/ggml-metal-device.h"
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>

struct allocation { uint64_t bytes; uint32_t role; uint32_t flags; uintptr_t owner; };
static std::map<std::pair<uint32_t, uintptr_t>, allocation> live;
static unsigned events = 0;
static uint32_t last_cpu_release_role = 0;
static void require(bool ok) { if (!ok) { std::fprintf(stderr, "memory diagnostic lifecycle failed\n"); std::abort(); } }
static void observer(uint32_t event, uint32_t domain, uintptr_t id, uintptr_t owner,
                     uint64_t bytes, uint32_t role, uint32_t flags) {
    ++events;
    auto key = std::make_pair(domain, id);
    if (event == 1) { require(live.count(key) == 0); live[key] = {bytes, role, flags, owner}; }
    if (event == 2 && live.count(key)) { live[key].role = role; }
    if (event == 3) {
        if (live.count(key)) {
            require(owner == live.at(key).owner);
            if (domain == 1) { require(bytes == live.at(key).bytes); require(role == live.at(key).role); last_cpu_release_role = role; }
        }
        live.erase(key);
    }
}

int main() {
    require(!ggml_backend_memory_diagnostics_enabled());
    auto off = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 65536);
    require(off != nullptr);
    ggml_backend_buffer_free(off);
    require(events == 0);
    ggml_backend_memory_diagnostics_set_callback(observer);
    auto cpu = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 65536);
    require(cpu != nullptr);
    require(live.at({1, (uintptr_t) cpu}).bytes >= 65536);
    ggml_backend_buffer_diagnostic_role(cpu, 3);
    require(live.at({1, (uintptr_t) cpu}).role == 3);
    ggml_backend_buffer_free(cpu);
    require(live.empty());
    require(last_cpu_release_role == 3);
    ggml_backend_buffer_t children[] = {
        ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 256),
        ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 256),
    };
    auto multi = ggml_backend_multi_buffer_alloc_buffer(children, 2);
    require(live.at({1, (uintptr_t) multi}).flags == 1);
    require(live.at({1, (uintptr_t) children[0]}).flags == 0);
    ggml_backend_buffer_diagnostic_role(multi, 3);
    require(live.at({1, (uintptr_t) children[0]}).role == 3);
    require(live.at({1, (uintptr_t) children[1]}).role == 3);
    ggml_backend_buffer_free(multi);
    require(live.empty());
#ifdef GGML_USE_METAL
    auto reg = ggml_backend_metal_reg();
    require(ggml_backend_reg_dev_count(reg) > 0);
    auto dev = ggml_backend_reg_dev_get(reg, 0);
    auto gpu = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 65536);
    require(gpu != nullptr);
    unsigned metal = 0;
    const uint32_t expected_storage = ggml_metal_buffer_is_shared((ggml_metal_buffer_t) gpu->context) ? 0 : 2;
    for (const auto & entry : live) {
        if (entry.first.first == 2) {
            require(entry.second.bytes >= 65536);
            require(entry.second.owner == live.at({1, (uintptr_t) gpu}).owner);
            require((entry.second.flags & 255) == expected_storage);
            ++metal;
        }
    }
    require(metal > 0);
    ggml_backend_buffer_set_usage(gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    require(live.at({1, (uintptr_t) gpu}).role == 1);
    // Exercise private-buffer staging references with page-aligned inputs.
    // On unified memory the same terminal data contract uses the shared path.
    struct ggml_init_params params = { 4 * ggml_tensor_overhead(), nullptr, true };
    auto ctx = ggml_init(params);
    auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);
    tensor->buffer = gpu;
    tensor->data = ggml_backend_buffer_get_base(gpu);
    void * input = nullptr;
    void * output = nullptr;
    require(posix_memalign(&input, 16384, 16384) == 0);
    require(posix_memalign(&output, 16384, 16384) == 0);
    std::memset(input, 42, 16384);
    ggml_backend_tensor_set(tensor, input, 0, 16384);
    ggml_backend_tensor_get(tensor, output, 0, 16384);
    require(std::memcmp(input, output, 16384) == 0);
    unsigned remaining_metal = 0;
    for (const auto & entry : live) { if (entry.first.first == 2) { ++remaining_metal; } }
    require(remaining_metal == metal);
    std::free(input); std::free(output);
    ggml_free(ctx);
    ggml_backend_buffer_free(gpu);
    require(live.empty());
#endif
    ggml_backend_memory_diagnostics_set_callback(nullptr);
    require(!ggml_backend_memory_diagnostics_enabled());
#if defined(GGML_USE_METAL)
    std::puts("native and Metal lifecycle: PASS");
#else
    std::puts("native lifecycle: PASS; Metal: NOT COMPILED");
#endif
}
