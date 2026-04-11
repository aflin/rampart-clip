rampart.globalize(rampart.utils);

var modelFile = 'CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q4_0.gguf';
var modelUrl  = 'https://huggingface.co/mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/' + modelFile;

/* ------------------------------------------------------------------ */
/*  Download model if missing                                         */
/* ------------------------------------------------------------------ */

if (!stat(modelFile)) {
    printf("Model '%s' not found.\n", modelFile);
    printf("Download it? (%d MB) [y/n] ", 86);
    fflush(stdout);

    var resp = fgets(stdin, 10).trim().toLowerCase();
    if (resp !== 'y' && resp !== 'yes') {
        printf("Exiting.\n");
        process.exit(0);
    }

    load.curl;

    var f = fopen(modelFile, 'w+');
    var nchunks = 0;

    curl.fetchAsync(modelUrl, {
        location: true,
        returnText: false,
        skipFinalRes: true,

        chunkCallback: function(res) {
            f.fprintf('%s', res.body);
        },

        progressCallback: function(res) {
            nchunks++;
            if (nchunks % 30) return;

            var tot = res.progress;
            var rate = tot / (res.totalTime * 1024);
            var unit = "KB/s";
            if (rate > 10000) {
                rate /= 1024;
                unit = "MB/s";
            }
            var out;
            if (res.expectedTotal != -1) {
                var perc = 100 * tot / res.expectedTotal;
                out = sprintf("    %Ad%%, %d of %d bytes (%.1f %s)",
                    'green', perc, tot, res.expectedTotal, rate, unit);
            } else {
                out = sprintf("    %d bytes", tot);
            }
            printf('%M', ['Downloading ' + modelFile, out]);
        },

        callback: function() {
            f.fclose();
            printf("\nDownload complete.\n\n");
            runTest();
        }
    });
} else {
    runTest();
}

/* ------------------------------------------------------------------ */
/*  Test                                                              */
/* ------------------------------------------------------------------ */

function runTest() {
    var clip = require('rampart-clip');

    // Load model
    var t0 = new Date().getTime();
    var model = clip.load(modelFile);
    var loadTime = new Date().getTime() - t0;
    printf("Model loaded in %d ms  (dimension: %d)\n\n", loadTime, model.dimension);

    // Embed all images with timing
    function embedDir(dir) {
        var results = [];
        var files = readDir(dir);
        files.sort();
        files.forEach(function(f) {
            if (f.match(/\.jpg$/)) {
                var path = dir + '/' + f;
                var start = new Date().getTime();
                var vec = model.embedImageToFp16Buf(path);
                var elapsed = new Date().getTime() - start;
                results.push({ name: f, vec: vec, ms: elapsed });
            }
        });
        return results;
    }

    printf("--- Embedding Timings ---\n\n");

    printf("Horses:\n");
    var horses = embedDir('test_images/horses');
    horses.forEach(function(r) {
        printf("  %-45s %4d ms\n", r.name, r.ms);
    });
    var horseAvgMs = horses.reduce(function(s,r){return s+r.ms;}, 0) / horses.length;
    printf("  Average: %d ms\n\n", Math.round(horseAvgMs));

    printf("Flowers:\n");
    var flowers = embedDir('test_images/flowers');
    flowers.forEach(function(r) {
        printf("  %-45s %4d ms\n", r.name, r.ms);
    });
    var flowerAvgMs = flowers.reduce(function(s,r){return s+r.ms;}, 0) / flowers.length;
    printf("  Average: %d ms\n\n", Math.round(flowerAvgMs));

    // Pairwise similarity within a group
    function avgPairwise(group) {
        var sum = 0, count = 0;
        for (var i = 0; i < group.length; i++) {
            for (var j = i + 1; j < group.length; j++) {
                sum += rampart.vector.raw.distance(group[i].vec, group[j].vec, 'dot', 'f16');
                count++;
            }
        }
        return sum / count;
    }

    // Average similarity between two groups
    function avgCross(g1, g2) {
        var sum = 0, count = 0;
        for (var i = 0; i < g1.length; i++) {
            for (var j = 0; j < g2.length; j++) {
                sum += rampart.vector.raw.distance(g1[i].vec, g2[j].vec, 'dot', 'f16');
                count++;
            }
        }
        return sum / count;
    }

    printf("--- Image-to-Image Similarity (dot product) ---\n\n");
    printf("  Horses  <-> Horses:  %.4f  (within-group)\n", avgPairwise(horses));
    printf("  Flowers <-> Flowers: %.4f  (within-group)\n", avgPairwise(flowers));
    printf("  Horses  <-> Flowers: %.4f  (cross-group)\n\n", avgCross(horses, flowers));

    // Text query search
    printf("--- Text-to-Image Search ---\n\n");

    var all = horses.concat(flowers);
    var queries = [
        'horses running on a beach',
        'spring flowers blooming',
        'ocean waves and sand',
        'a horse at sunset',
        'pink cherry blossoms'
    ];

    queries.forEach(function(q) {
        var start = new Date().getTime();
        var qvec = model.embedTextToFp16Buf(q);
        var encTime = new Date().getTime() - start;
        var scores = all.map(function(img) {
            return {
                name: img.name,
                score: rampart.vector.raw.distance(img.vec, qvec, 'dot', 'f16')
            };
        });
        scores.sort(function(a,b) { return b.score - a.score; });
        printf("Query: \"%s\"  (text embed: %d ms)\n", q, encTime);
        printf("  Top 3:\n");
        scores.slice(0,3).forEach(function(s, i) {
            printf("    %d. %-45s (%.4f)\n", i+1, s.name, s.score);
        });
        printf("\n");
    });

    model.destroy();
    printf("Done.\n");
}
