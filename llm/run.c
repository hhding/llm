#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>

typedef struct {
    int dim;    // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // num of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;   // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight;  // (layer, dim)
    float* rms_ffn_weight;  // (layer, dim)
    // weights for matmuls, note dim == n_heads * head_size
    float* wq;  // (layer, dim, n_heads * head_size)
    float* wk;  // (layer, dim, n_kv_heads * head_size)
    float* wv;  // (layer, dim, n_kv_heads * head_size)
    float* wo;  // (layer, n_heads * head_size, dim)
    // weight for ffn
    float* w1;  // (layer, hidden_dim, dim)
    float* w2;  // (layer, dim, hidden_dim)
    float* w3;  // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight;    // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    float* x;   // activation at current timestamp (dim,)
    float* xb;  // same, but inside a residual branch (dim,)
    float* xb2;    // an additional buffer just for convenience (dim,)
    float* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q;  // query (dim,)
    float* k;  // key (dim,)
    float* v;  // value (dim,)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits;  // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    if(!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
      || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
      }
}

void memory_map_weight(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layer = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += p->n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoints(char * checkpoint, Config* config, TransformerWeights* w,
                        int* fd, float** data, ssize_t* file_size) {
    *fd = open(checkpoint, O_RDONLY);
    if (read(*fd, config, sizeof(Config)) != 1) {exit(EXIT_FAILURE);}
    *file_size = lseek(*fd, 0, SEEK_END);
    data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if(*data == MAP_FAILED) {fprintf(stderr, "mmap failed!\n"), exit(EXIT_FAILURE);}
    float* weight_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weight(w, config, weight_ptr, config->vocab_size >0? 1:0);
}

void build_tranformer(Transformer *t, char* checkpoint_path) {
    read_checkpoints(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

#ifndef TESTING
void error_usage() {
    fprintf(stderr, "Usage:     run <checkpoint> [options]\n");
}
int main(int argc, char** argv) {
    char *checkpoint_path = NULL;
    char * tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;
    if (argc >= 2) {checkpoint_path = argv[1];} else {error_usage();}

    Transformer transformer;
    build_tranformer(&transformer, checkpoint_path);
}

#endif