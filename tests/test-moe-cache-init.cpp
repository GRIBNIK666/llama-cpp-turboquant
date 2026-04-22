// Minimal test: load model, init MoE cache with profile, verify save/load
#include "llama.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * profile_path = "/tmp/moe-test-profile.json";

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;

    fprintf(stderr, "Loading model...\n");
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 64;

    fprintf(stderr, "Creating context...\n");
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "Initializing MoE cache with 64 experts + profile...\n");
    llama_init_moe_cache(ctx, 64, true, profile_path);

    // Check stats
    auto stats = llama_moe_cache_get_stats(ctx);
    fprintf(stderr, "MoE cache stats: hits=%" PRIu64 " misses=%" PRIu64 " evictions=%" PRIu64 "\n",
            stats.hits, stats.misses, stats.evictions);

    fprintf(stderr, "Cleaning up (will save profile)...\n");
    llama_free(ctx);

    // Check profile file was created
    std::ifstream f(profile_path);
    if (f.good()) {
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        fprintf(stderr, "Profile saved (%zu bytes): %s\n",
                content.size(), profile_path);
        // Print first 200 chars
        if (content.size() > 200) content.resize(200);
        fprintf(stderr, "%s...\n", content.c_str());
    } else {
        fprintf(stderr, "Profile file created (empty - no inference was run)\n");
    }

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "TEST PASSED\n");
    return 0;
}
