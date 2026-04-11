# rampart-clip

A native [Rampart](https://github.com/aflin/rampart) module for CLIP
(Contrastive Language-Image Pre-training) inference.  It wraps
[monatis/clip.cpp](https://github.com/monatis/clip.cpp) to provide image
and text embedding from JavaScript, enabling semantic image search,
zero-shot classification, and cross-modal similarity scoring.

CLIP maps both images and text into a shared vector space.  Similar
concepts cluster together regardless of modality, so you can search a
collection of images using a text query or find images similar to a
reference image.

## Requirements

- [Rampart](https://github.com/aflin/rampart) JavaScript runtime
- CMake 3.15+
- C/C++ compiler (GCC, Clang, or Apple Clang)
- A CLIP model in GGUF format (see [Models](#models) below)

## Building

```bash
git clone https://github.com/aflin/rampart-clip.git
cd rampart-clip
mkdir build && cd build
cmake ..
make -j$(nproc)
```

CMake auto-detects CPU features (SSE, AVX, AVX2, AVX512) on x86 and
enables appropriate optimizations.  On Apple Silicon it enables
`-march=native` and Metal acceleration via the ggml backend.

No additional flags are needed.

### Installing

```bash
sudo make install
```

This copies `rampart-clip.so` into the Rampart modules directory.
Alternatively, place the `.so` in the same directory as your script and
`require('rampart-clip')` will find it there.

## Models

rampart-clip uses GGUF model files compatible with
[monatis/clip.cpp](https://github.com/monatis/clip.cpp).  These are
standard OpenAI and LAION CLIP models converted to GGUF format.

Pre-converted models are available on HuggingFace:

| Model | Params | Dim | HuggingFace |
|-------|--------|-----|-------------|
| ViT-B/32 (LAION) | 0.2B | 512 | [mys/ggml\_CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K) |
| ViT-B/32 (OpenAI) | 0.2B | 512 | [mys/ggml\_clip-vit-base-patch32](https://huggingface.co/mys/ggml_clip-vit-base-patch32) |
| ViT-L/14 (LAION) | 0.4B | 768 | [mys/ggml\_CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/mys/ggml_CLIP-ViT-L-14-laion2B-s32B-b82K) |
| ViT-L/14 (OpenAI) | 0.4B | 768 | [mys/ggml\_clip-vit-large-patch14](https://huggingface.co/mys/ggml_clip-vit-large-patch14) |
| ViT-H/14 (LAION) | 1.0B | 1024 | [mys/ggml\_CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/mys/ggml_CLIP-ViT-H-14-laion2B-s32B-b79K) |

Each repository offers multiple quantizations: `f32`, `f16`, `q8_0`,
`q5_1`, `q5_0`, `q4_1`, `q4_0`, as well as vision-only and text-only
variants.  The `q8_0` quantization is recommended as it is virtually
identical to `f16` in quality at roughly half the size.

LAION-trained models generally outperform their OpenAI counterparts on
retrieval benchmarks.

### Downloading a model

```bash
# ViT-B/32 q8_0 (164 MB) — fast, good for prototyping
curl -L -o CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf \
  https://huggingface.co/mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf

# ViT-L/14 q8_0 (457 MB) — better quality, recommended for production
curl -L -o CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q8_0.gguf \
  https://huggingface.co/mys/ggml_CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q8_0.gguf
```

### Converting your own models

The included `extern/clip.cpp/models/convert_hf_to_gguf.py` script can
convert any CLIP model in HuggingFace Transformers format (`CLIPModel`)
to GGUF.  Models using OpenCLIP or other architectures (SigLIP, EVA-CLIP,
etc.) are not supported.

## API

### clip.load(modelPath [, options])

Load a CLIP model and return a model object.

```javascript
var clip = require("rampart-clip");

var model = clip.load("model.gguf");

// or with options:
var model = clip.load("model.gguf", {
    nThreads:  4,   // CPU threads for inference (default: 4)
    verbosity: 0    // 0 = silent, 1 = model info, 2+ = debug (default: 0)
});
```

### model.dimension

Read-only property giving the embedding dimension of the loaded model
(e.g., 512 for ViT-B/32, 768 for ViT-L/14).

### model.embedImageToFp16Buf(imagePath)

Embed an image file and return the vector as a `Buffer` of fp16 values.

```javascript
var vec = model.embedImageToFp16Buf("/path/to/image.jpg");
// vec.length === model.dimension * 2 (bytes)
```

### model.embedImageToFp32Buf(imagePath)

Same as above but returns fp32 values.

```javascript
var vec = model.embedImageToFp32Buf("/path/to/image.jpg");
// vec.length === model.dimension * 4 (bytes)
```

### model.embedImageToNumbers(imagePath)

Same as above but returns a JavaScript `Array` of `Number` values.

```javascript
var vec = model.embedImageToNumbers("/path/to/image.jpg");
// vec.length === model.dimension
```

### model.embedTextToFp16Buf(text)

Embed a text string and return the vector as a `Buffer` of fp16 values.
The text and image vectors occupy the same vector space.

```javascript
var vec = model.embedTextToFp16Buf("a photo of a cat");
```

### model.embedTextToFp32Buf(text)

Same as above but returns fp32 values.

### model.embedTextToNumbers(text)

Same as above but returns a JavaScript `Array` of `Number` values.

### model.similarity(vec1, vec2)

Compute cosine similarity between two vectors.  Accepts fp16 or fp32
buffers (both must be the same type).  Returns a `Number` between -1 and 1.

```javascript
var score = model.similarity(imageVec, textVec);
```

### model.destroy()

Free the model and its resources.  The model object should not be used
after calling this.

## Supported Image Formats

Image loading uses stb\_image, which supports: JPEG, PNG, BMP, TGA, PSD,
GIF (first frame), HDR, PIC, and PNM.

Images are automatically resized and center-cropped to the model's
expected input size (typically 224x224) using bicubic interpolation.

## Threading

Each model object is bound to the thread that created it.  Attempting to
use a model from a different thread will throw an error.  Each thread
must call `clip.load()` separately.

Forked processes can safely use a model loaded before the fork, since
each process gets its own copy of the memory via copy-on-write.  This is
the recommended approach for parallel processing, as the read-only model
weights are shared across processes without additional memory cost.

## Demo

```javascript
var clip = require("rampart-clip");
var printf = rampart.utils.printf;

// Load model
var model = clip.load("CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf");
printf("Model loaded, dimension: %d\n\n", model.dimension);

// Embed some images
var images = [
    { name: "cat.jpg",    vec: model.embedImageToFp16Buf("cat.jpg") },
    { name: "dog.jpg",    vec: model.embedImageToFp16Buf("dog.jpg") },
    { name: "flower.jpg", vec: model.embedImageToFp16Buf("flower.jpg") }
];

// Search by text
var queries = ["a cute cat", "a dog playing", "flowers in a garden"];

queries.forEach(function(q) {
    var qvec = model.embedTextToFp16Buf(q);

    // Score each image against the query
    var scores = images.map(function(img) {
        return {
            name: img.name,
            score: model.similarity(img.vec, qvec)
        };
    });
    scores.sort(function(a, b) { return b.score - a.score; });

    printf("Query: \"%s\"\n", q);
    scores.forEach(function(s) {
        printf("  %-20s %.4f\n", s.name, s.score);
    });
    printf("\n");
});

// Image-to-image similarity
printf("Image similarity:\n");
for (var i = 0; i < images.length; i++) {
    for (var j = i + 1; j < images.length; j++) {
        var score = model.similarity(images[i].vec, images[j].vec);
        printf("  %-12s <-> %-12s  %.4f\n",
            images[i].name, images[j].name, score);
    }
}

model.destroy();
```

Vectors returned by this module are compatible with
[rampart-langtools](https://github.com/aflin/rampart-langtools) for FAISS
indexing and with Rampart's built-in
[vector functions](https://rampart.dev/docs/rampart-vector.html)
for distance calculations and type conversions.

## License

MIT
