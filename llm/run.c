#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
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
        perror("malloc failed!\n");
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
    FILE* fp = fopen(checkpoint, "rb");
    if (fread(config, sizeof(Config), 1, fp) != 1) {perror("error read checkpoint"); exit(EXIT_FAILURE);}
    if (fseek(fp, 0, SEEK_END) == -1) {perror("error fseek checkpoint"), exit(EXIT_FAILURE);}
    *file_size = ftell(fp);
    *fd = fileno(fp);
    data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if(*data == MAP_FAILED) {perror("mmap failed!\n"), exit(EXIT_FAILURE);}
    float* weight_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weight(w, config, weight_ptr, config->vocab_size >0? 1:0);
}

void build_tranformer(Transformer *t, char* checkpoint_path) {
    read_checkpoints(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

// output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 
// return output * self.weights
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// softmax = exp(x[i]) / sum(exp[i])
void softmax(float* x, int size) {
    // 引入 max_val 是为了直接计算出来的 expf(x[i]) 太大溢出
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W(d,n) @ x(n,) -> xout (d,)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * w[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->n_kv_heads * p->dim) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;    // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = p->dim / p->n_heads;

    // 将 token 对应的 embedding 向量拷贝到 x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // 开始对所有层进行 forward 计算
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm; save to: s->xb as input of q, k, v
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        int loff = l * p->seq_len * kv_dim; // 当前层的偏移
        s->k = s->key_cache + loff + pos * kv_dim;      // 再加上当前 pos 的偏移
        s->v = s->value_cache + loff + pos * kv_dim;    // 会存当前 token 产生的 k，v

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE; 加上位置信息
        for (int i = 0; i < dim; i++) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k; // 超过 kvm_dim 后，只对 q 加位置信息
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // score = q @ k
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t < pos; t++) {
                float * k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos+1);
            // score @ v
            float* xb = s->xb + h * head_size;  // 结果存这里
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t < pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h/kv_dim) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        // s->xb2: output = self.wo(output)
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        // s->xb2 = self.attention.forward(self.attention_norm(x))
        // x = x + s->xb2
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        // h = self.ffn_norm(h)
        // h = self.w2( F.silu(self.w1(h)) * self.w3(h) )
        // x = x + h
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        // self.w2(h)
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

typedef struct {
    float prob;
    int index;
} ProbIndex;    // struct used when sorting probabilities during top-p sampleing

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rnd_state;
} Sampler;

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rnd_state = rng_seed;
    sampler->probindex = malloc(vocab_size * sizeof(ProbIndex));
}

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex * sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings;
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE* fp = fopen(tokenizer_path, "rb");
    if (fp == NULL) {perror("error open token file"); exit(EXIT_FAILURE);}
    if (fread(&t->max_token_length, sizeof(int), 1, fp) != 1) {perror("error read max token length"); exit(EXIT_FAILURE);}
    for (int i = 0, len = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, fp) != 1) {perror("error read vocab scores"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, fp) != 1) {perror("error read token size"); exit(EXIT_FAILURE);}
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, fp) != 1) {perror("error read token"); exit(EXIT_FAILURE);}
        t->vocab[i][len] = '\0';
    }
    fclose(fp);
}

int str_lookup(char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) {fprintf(stderr, "Can't encode NULL text"); exit(EXIT_FAILURE);}
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i<t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;

    // always add dummy prefix when text is not '\0'
    if (text[0] != '\0') {
        tokens[(*n_tokens)++] = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    }
    // merge tow tokens (*2), '\0'(+1), +2 for UTF-8??
    char* str_buffer = malloc(t->max_token_length * 2 + 1 + 2);
    // UTF-8
    for (char* c = text; *c != '\0'; c++) {
        // ASCII or start of UTF-8
        if ( (*c & 0xC0) != 0x80 ) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        // max length of utf-8 is 4
        if((*(c+1) & 0xC0) == 0x80 & str_len < 4) continue;
        // got one charactor or one utf-8
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // fallback to byte by byte mode
            for (int i=0; i < str_len; i++) {
                // 3: <unk>, <s>, </s>
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }
    // merge token[i] and token[i+1]
    while(1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        // 在这个递归里面，相邻两个 token 两两进行合并后查询 id
        // 如果发现有多个匹配的，那么仅合并分数最高的那个
        // 如果发现合并后都不匹配，那么就说明全部到达最大长度了，结束这个流程
        for (int i = 0; i < *n_tokens; i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 ) continue;
            if (t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;   // 合并成哪个 ID
                best_idx = i;   // 匹配到哪个位置
            }
        }
        // 合并后的都是不是合法 token，那么该结束了
        if (best_idx == -1) break;
        // 合并两个 token
        tokens[best_idx] = best_id;
        // 将 best_idx 后面的全部向前平移一位
        for (int i = best_idx+1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }
    if(eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps) {
    char* empty_prompt = "";
    if (prompt == NULL) {prompt = empty_prompt;}
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3)*sizeof(int));  // +3 for '\0', BOS, EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
}

void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, 
            char* cli_user_prompt, char* cli_system_prompt, int steps) {
    fprintf(stderr, "chat\n");
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
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
    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_tranformer(&transformer, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    fprintf(stderr, "run!\n");
    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        exit(EXIT_FAILURE);
    }
    return 0;
}
