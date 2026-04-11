/*
 * rampart-clip.c — Duktape module wrapping monatis/clip.cpp
 *
 * Provides CLIP image and text embedding for the Rampart JavaScript runtime.
 *
 * Usage:
 *   var clip = require("rampart-clip");
 *
 *   var model = clip.load("model.gguf", { nThreads: 4 });
 *
 *   var v = model.embedImageToFp16Buf("/path/to/image.jpg");
 *   var v = model.embedImageToFp32Buf("/path/to/image.jpg");
 *   var v = model.embedImageToNumbers("/path/to/image.jpg");
 *
 *   var v = model.embedTextToFp16Buf("a photo of a cat");
 *   var v = model.embedTextToFp32Buf("a photo of a cat");
 *   var v = model.embedTextToNumbers("a photo of a cat");
 *
 *   var score = model.similarity(vec1, vec2);
 *
 *   model.destroy();
 */

#include <ctype.h>
#include <errno.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include "clip.h"
#include "rampart.h"

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

#define NOPACK 0
#define PACK16 1
#define PACK32 2

/*
 * Check that the calling thread is the one that created the clip_ctx.
 * clip.cpp has no model/context split, so the entire clip_ctx
 * (weights + compute buffer) is one object and cannot be safely
 * shared across threads (which share the same address space).
 *
 * Fork is fine — the child gets its own copy of everything.
 */
static void check_thread(duk_context *ctx)
{
    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_thread"));
    int thrno = duk_get_int(ctx, -1);
    duk_pop(ctx);

    duk_pop(ctx); /* this */

    int curthr = get_thread_num();

    if (curthr != thrno)
        RP_THROW(ctx,
            "rampart-clip: this model object was created in thread %d "
            "but is being used in thread %d. "
            "clip.cpp does not support sharing a model across threads — "
            "each thread must call clip.load() separately.",
            thrno, curthr);
}

/* Retrieve the clip_ctx pointer stored as a hidden property on `this` */
static struct clip_ctx *get_clip_ctx(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("clip_ctx"));
    struct clip_ctx *cctx = duk_get_pointer(ctx, -1);
    duk_pop_2(ctx); /* pointer + this */
    return cctx;
}

/* Retrieve projection_dim stored as hidden int on `this` */
static int get_vec_dim(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("vec_dim"));
    int d = duk_get_int(ctx, -1);
    duk_pop_2(ctx);
    return d;
}

/* Retrieve nthreads stored as hidden int on `this` */
static int get_nthreads(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("nthreads"));
    int n = duk_get_int(ctx, -1);
    duk_pop_2(ctx);
    return n;
}

/* ------------------------------------------------------------------ */
/*  embed_image_to_ — core image embedding, returns packed or array   */
/* ------------------------------------------------------------------ */

static duk_ret_t embed_image_to_(duk_context *ctx, int pack)
{
    check_thread(ctx);

    const char *fname = REQUIRE_STRING(ctx, 0,
        "rampart-clip: embedImage*() argument must be a String (image file path)");

    struct clip_ctx *cctx = get_clip_ctx(ctx);
    int vec_dim = get_vec_dim(ctx);
    int n_threads = get_nthreads(ctx);

    if (!cctx)
        RP_THROW(ctx, "rampart-clip: model has been destroyed");

    /* load image from file */
    struct clip_image_u8 *img_u8 = clip_image_u8_make();
    if (!img_u8)
        RP_THROW(ctx, "rampart-clip: failed to allocate image");

    if (!clip_image_load_from_file(fname, img_u8))
    {
        clip_image_u8_free(img_u8);
        RP_THROW(ctx, "rampart-clip: failed to load image '%s'", fname);
    }

    /* preprocess */
    struct clip_image_f32 *img_f32 = clip_image_f32_make();
    if (!img_f32)
    {
        clip_image_u8_free(img_u8);
        RP_THROW(ctx, "rampart-clip: failed to allocate preprocessed image");
    }

    if (!clip_image_preprocess(cctx, img_u8, img_f32))
    {
        clip_image_f32_free(img_f32);
        clip_image_u8_free(img_u8);
        RP_THROW(ctx, "rampart-clip: failed to preprocess image '%s'", fname);
    }

    /* encode */
    float *vec = malloc(vec_dim * sizeof(float));
    if (!vec)
    {
        clip_image_f32_free(img_f32);
        clip_image_u8_free(img_u8);
        RP_THROW(ctx, "rampart-clip: out of memory allocating embedding vector");
    }

    if (!clip_image_encode(cctx, n_threads, img_f32, vec, true))
    {
        free(vec);
        clip_image_f32_free(img_f32);
        clip_image_u8_free(img_u8);
        RP_THROW(ctx, "rampart-clip: failed to encode image '%s'", fname);
    }

    clip_image_f32_free(img_f32);
    clip_image_u8_free(img_u8);

    /* return result */
    if (pack == PACK16)
    {
        /* fp16 buffer */
        void *buf = duk_push_fixed_buffer(ctx, vec_dim * sizeof(uint16_t));
        rpvec_f32_to_f16(vec, (uint16_t *)buf, vec_dim);
    }
    else if (pack == PACK32)
    {
        /* fp32 buffer */
        void *buf = duk_push_fixed_buffer(ctx, vec_dim * sizeof(float));
        memcpy(buf, vec, vec_dim * sizeof(float));
    }
    else
    {
        /* array of numbers */
        duk_push_array(ctx);
        for (int i = 0; i < vec_dim; i++)
        {
            duk_push_number(ctx, (double)vec[i]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
    }

    free(vec);
    return 1;
}

static duk_ret_t embed_image_to_buf16(duk_context *ctx) { return embed_image_to_(ctx, PACK16); }
static duk_ret_t embed_image_to_buf32(duk_context *ctx) { return embed_image_to_(ctx, PACK32); }
static duk_ret_t embed_image_to_numbers(duk_context *ctx) { return embed_image_to_(ctx, NOPACK); }

/* ------------------------------------------------------------------ */
/*  embed_text_to_ — core text embedding, returns packed or array     */
/* ------------------------------------------------------------------ */

static duk_ret_t embed_text_to_(duk_context *ctx, int pack)
{
    check_thread(ctx);

    if (duk_is_buffer_data(ctx, 0))
        duk_buffer_to_string(ctx, 0);

    const char *text = REQUIRE_STRING(ctx, 0,
        "rampart-clip: embedText*() argument must be a String");

    struct clip_ctx *cctx = get_clip_ctx(ctx);
    int vec_dim = get_vec_dim(ctx);
    int n_threads = get_nthreads(ctx);

    if (!cctx)
        RP_THROW(ctx, "rampart-clip: model has been destroyed");

    /* tokenize */
    struct clip_tokens tokens = {0};
    if (!clip_tokenize(cctx, text, &tokens))
        RP_THROW(ctx, "rampart-clip: failed to tokenize text");

    /* encode */
    float *vec = malloc(vec_dim * sizeof(float));
    if (!vec)
    {
        free(tokens.data);
        RP_THROW(ctx, "rampart-clip: out of memory allocating embedding vector");
    }

    if (!clip_text_encode(cctx, n_threads, &tokens, vec, true))
    {
        free(vec);
        free(tokens.data);
        RP_THROW(ctx, "rampart-clip: failed to encode text");
    }

    free(tokens.data);

    /* return result */
    if (pack == PACK16)
    {
        void *buf = duk_push_fixed_buffer(ctx, vec_dim * sizeof(uint16_t));
        rpvec_f32_to_f16(vec, (uint16_t *)buf, vec_dim);
    }
    else if (pack == PACK32)
    {
        void *buf = duk_push_fixed_buffer(ctx, vec_dim * sizeof(float));
        memcpy(buf, vec, vec_dim * sizeof(float));
    }
    else
    {
        duk_push_array(ctx);
        for (int i = 0; i < vec_dim; i++)
        {
            duk_push_number(ctx, (double)vec[i]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
    }

    free(vec);
    return 1;
}

static duk_ret_t embed_text_to_buf16(duk_context *ctx) { return embed_text_to_(ctx, PACK16); }
static duk_ret_t embed_text_to_buf32(duk_context *ctx) { return embed_text_to_(ctx, PACK32); }
static duk_ret_t embed_text_to_numbers(duk_context *ctx) { return embed_text_to_(ctx, NOPACK); }

/* ------------------------------------------------------------------ */
/*  similarity — cosine similarity between two fp16 or fp32 buffers   */
/* ------------------------------------------------------------------ */

static duk_ret_t clip_similarity(duk_context *ctx)
{
    int vec_dim = get_vec_dim(ctx);
    duk_size_t sz1 = 0, sz2 = 0;

    void *buf1 = REQUIRE_BUFFER_DATA(ctx, 0, &sz1,
        "rampart-clip: similarity() arguments must be Buffers");
    void *buf2 = REQUIRE_BUFFER_DATA(ctx, 1, &sz2,
        "rampart-clip: similarity() arguments must be Buffers");

    if (sz1 != sz2)
        RP_THROW(ctx, "rampart-clip: similarity() buffers must be the same size");

    float *v1 = NULL, *v2 = NULL;
    int need_free = 0;

    if ((int)(sz1 / sizeof(float)) == vec_dim)
    {
        /* fp32 buffers */
        v1 = (float *)buf1;
        v2 = (float *)buf2;
    }
    else if ((int)(sz1 / sizeof(uint16_t)) == vec_dim)
    {
        /* fp16 buffers — convert to fp32 */
        v1 = malloc(vec_dim * sizeof(float));
        v2 = malloc(vec_dim * sizeof(float));
        if (!v1 || !v2)
        {
            free(v1);
            free(v2);
            RP_THROW(ctx, "rampart-clip: out of memory in similarity()");
        }
        rpvec_f16_to_f32((uint16_t *)buf1, v1, vec_dim);
        rpvec_f16_to_f32((uint16_t *)buf2, v2, vec_dim);
        need_free = 1;
    }
    else
    {
        RP_THROW(ctx, "rampart-clip: similarity() buffer size does not match model dimension (%d)", vec_dim);
    }

    float score = clip_similarity_score(v1, v2, vec_dim);

    if (need_free)
    {
        free(v1);
        free(v2);
    }

    duk_push_number(ctx, (double)score);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  destroy — free the clip context                                   */
/* ------------------------------------------------------------------ */

static duk_ret_t clip_destroy(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("clip_ctx"));
    struct clip_ctx *cctx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (cctx)
    {
        clip_free(cctx);
        duk_push_pointer(ctx, NULL);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("clip_ctx"));
    }

    duk_pop(ctx); /* this */
    return 0;
}

/* Destructor / GC finalizer */
static duk_ret_t clip_finalizer(duk_context *ctx)
{
    duk_get_prop_string(ctx, 0, DUK_HIDDEN_SYMBOL("clip_ctx"));
    struct clip_ctx *cctx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (cctx)
    {
        clip_free(cctx);
        duk_push_pointer(ctx, NULL);
        duk_put_prop_string(ctx, 0, DUK_HIDDEN_SYMBOL("clip_ctx"));
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  load — load a CLIP GGUF model and return a model object           */
/* ------------------------------------------------------------------ */

static duk_ret_t clip_load(duk_context *ctx)
{
    const char *model_path = REQUIRE_STRING(ctx, 0,
        "rampart-clip: load() argument 1 must be a String (model path)");

    int n_threads = 4;
    int verbosity = 0;

    if (duk_is_object(ctx, 1))
    {
        if (duk_get_prop_string(ctx, 1, "nThreads"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "rampart-clip: option nThreads must be a Number");
            n_threads = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, 1, "verbosity"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "rampart-clip: option verbosity must be a Number");
            verbosity = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);
    }

    /* check file exists before calling clip_model_load, which
       will segfault on a missing/invalid file rather than returning NULL */
    struct stat st;
    if (stat(model_path, &st) != 0)
        RP_THROW(ctx, "rampart-clip: cannot open model file '%s': %s", model_path, strerror(errno));

    struct clip_ctx *cctx = clip_model_load(model_path, verbosity);
    if (!cctx)
        RP_THROW(ctx, "rampart-clip: failed to load model '%s'", model_path);

    /* determine embedding dimension */
    struct clip_vision_hparams *vhp = clip_get_vision_hparams(cctx);
    int vec_dim = vhp ? vhp->projection_dim : 0;

    if (vec_dim <= 0)
    {
        /* try text hparams */
        struct clip_text_hparams *thp = clip_get_text_hparams(cctx);
        vec_dim = thp ? thp->projection_dim : 0;
    }

    if (vec_dim <= 0)
    {
        clip_free(cctx);
        RP_THROW(ctx, "rampart-clip: failed to determine embedding dimension from model");
    }

    /* build return object */
    duk_push_object(ctx);

    /* hidden state */
    duk_push_pointer(ctx, cctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("clip_ctx"));

    duk_push_int(ctx, vec_dim);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("vec_dim"));

    duk_push_int(ctx, n_threads);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("nthreads"));

    /* track creating thread for cross-thread detection */
    duk_push_int(ctx, (int)get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));

    /* read-only properties */
    duk_push_int(ctx, vec_dim);
    duk_rp_put_prop_string_ro(ctx, -2, "dimension");

    /* image embedding methods */
    duk_push_c_function(ctx, embed_image_to_buf16, 1);
    duk_put_prop_string(ctx, -2, "embedImageToFp16Buf");

    duk_push_c_function(ctx, embed_image_to_buf32, 1);
    duk_put_prop_string(ctx, -2, "embedImageToFp32Buf");

    duk_push_c_function(ctx, embed_image_to_numbers, 1);
    duk_put_prop_string(ctx, -2, "embedImageToNumbers");

    /* text embedding methods */
    duk_push_c_function(ctx, embed_text_to_buf16, 1);
    duk_put_prop_string(ctx, -2, "embedTextToFp16Buf");

    duk_push_c_function(ctx, embed_text_to_buf32, 1);
    duk_put_prop_string(ctx, -2, "embedTextToFp32Buf");

    duk_push_c_function(ctx, embed_text_to_numbers, 1);
    duk_put_prop_string(ctx, -2, "embedTextToNumbers");

    /* similarity */
    duk_push_c_function(ctx, clip_similarity, 2);
    duk_put_prop_string(ctx, -2, "similarity");

    /* destroy */
    duk_push_c_function(ctx, clip_destroy, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    /* finalizer for GC */
    duk_push_c_function(ctx, clip_finalizer, 1);
    duk_set_finalizer(ctx, -2);

    return 1;
}

/* ------------------------------------------------------------------ */
/*  Module entry point                                                */
/* ------------------------------------------------------------------ */

duk_ret_t duk_open_module(duk_context *ctx)
{
    duk_push_object(ctx);

    duk_push_c_function(ctx, clip_load, DUK_VARARGS);
    duk_put_prop_string(ctx, -2, "load");

    return 1;
}
