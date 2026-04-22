#include "llama.h"
#include <cinttypes>
#include <cstdio>
int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model>\n", argv[0]); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.use_mmap = true;
    mp.vocab_only = true;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "failed\n"); return 1; }
    fprintf(stderr, "n_expert:  %d\n", llama_model_n_expert(m));
    fprintf(stderr, "n_layer:   %d\n", llama_model_n_layer(m));
    fprintf(stderr, "n_embd:    %d\n", llama_model_n_embd(m));
    fprintf(stderr, "n_params:  %" PRIu64 " (%.1fB)\n", llama_model_n_params(m), llama_model_n_params(m)/1e9);
    llama_model_free(m);
    llama_backend_free();
}
