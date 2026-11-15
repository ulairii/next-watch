// Browser port of recommender/service.py for the static site (GitHub Pages). It scores with the
// model files exported by app/build_static.py and answers the same /api/* paths as app/app.py,
// so main.js runs unchanged. Arrays are float32 like the NumPy code, so rankings match it.
(function (root) {
  const LIKED_STARS = 4;
  const VALID_STARS = new Set([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]);
  const f32 = Math.fround;

  function stableArgsortDesc(values) {
    const idx = Array.from(values, (_, i) => i);
    return idx.sort((a, b) => (values[b] - values[a]) || (a - b));
  }

  function splitTitle(t) {
    // MovieLens appends alternate titles in parentheses.
    if (t.endsWith(')') && t.includes(' (')) {
      const i = t.indexOf(' (');
      return [t.slice(0, i), t.slice(i + 2, -1)];
    }
    return [t, null];
  }

  // Standardize scores over unrated movies (zscore_rows in models.py).
  function zscore(S, seen) {
    let sum = 0, cnt = 0;
    for (let i = 0; i < S.length; i++) if (!seen.has(i)) { sum += S[i]; cnt++; }
    const mu = f32(sum / cnt);
    let ss = 0;
    for (let i = 0; i < S.length; i++) if (!seen.has(i)) { const d = f32(S[i] - mu); ss += d * d; }
    let sd = f32(Math.sqrt(f32(ss / cnt)));
    if (sd === 0) sd = 1;
    const Z = new Float32Array(S.length);
    for (let i = 0; i < S.length; i++) Z[i] = seen.has(i) ? 0 : f32(f32(S[i] - mu) / sd);
    return Z;
  }

  class Recommender {
    constructor(meta, bin) {
      const mv = meta.movies;
      this.mv = mv;
      this.n = mv.movieId.length;
      this.config = meta.config;
      this.genres = meta.genres;
      this.vocab = new Map(meta.vocab.map((g, i) => [g, i]));
      this.nVocab = meta.vocab.length;
      this.svdK = meta.svd_k;
      this.X = new Float32Array(bin.content_x);          // movies x genres, rows L2-normalized
      this.V = new Float32Array(bin.svd_v);              // factors x movies
      this.knnPtr = new Int32Array(bin.knn_indptr);      // S[j, i] by row j (neighbor -> target)
      this.knnIdx = new Uint16Array(bin.knn_indices);
      this.knnVal = new Float32Array(bin.knn_values);
      this.col = new Map(mv.movieId.map((id, j) => [id, j]));
      this.counts = Float32Array.from(mv.n_ratings);
      this.byPopularity = stableArgsortDesc(this.counts);
      this.genreSets = mv.genres.map((g) => new Set(g));
      this.displayLower = mv.title.map((t, j) => (mv.aka[j] ? `${t} (${mv.aka[j]})` : t).toLowerCase());
    }

    static async load(get) {
      const names = ['content_x', 'svd_v', 'knn_indptr', 'knn_indices', 'knn_values'];
      const [meta, ...bufs] = await Promise.all([get('meta.json', 'json'), ...names.map((n) => get(`${n}.bin`, 'bin'))]);
      return new Recommender(meta, Object.fromEntries(names.map((n, i) => [n, bufs[i]])));
    }

    // ---- helpers ------------------------------------------------------------
    card(j, reason = null) {
      const m = this.mv;
      return {
        movieId: m.movieId[j],
        title: m.title[j],
        aka: m.aka[j],
        year: m.year[j],
        genres: m.genres[j],
        avg_rating: m.avg_rating[j],
        n_ratings: m.n_ratings[j],
        tmdb_url: m.tmdbId[j] == null ? null : `https://www.themoviedb.org/movie/${m.tmdbId[j]}`,
        imdb_url: `https://www.imdb.com/title/tt${String(m.imdbId[j]).padStart(7, '0')}/`,
        reason,
      };
    }

    cleanInput(genres, ratings) {
      genres = (genres || []).filter((g) => this.genres.includes(g));
      const clean = new Map();                           // column -> stars, in request order
      for (const [key, value] of Object.entries(ratings || {})) {
        const id = Number(key), stars = Number(value);
        if (Number.isInteger(id) && this.col.has(id) && VALID_STARS.has(stars)) clean.set(this.col.get(id), stars);
      }
      return [genres, clean];
    }

    centered(ratings) {
      let sum = 0;
      for (const s of ratings.values()) sum += s;
      const mean = f32(sum / ratings.size);
      return [...ratings].map(([j, s]) => [j, f32(s - mean)]);
    }

    knnRow(j) {
      const row = new Map();
      for (let p = this.knnPtr[j]; p < this.knnPtr[j + 1]; p++) row.set(this.knnIdx[p], this.knnVal[p]);
      return row;
    }

    // ---- component models (models.py) ------------------------------------------
    popularity() {
      return this.counts;
    }

    content(ratings, genres) {
      const G = this.nVocab, P = new Float32Array(G);
      if (ratings.size) {
        for (const [j, r] of this.centered(ratings)) for (let g = 0; g < G; g++) P[g] = P[g] + f32(r * this.X[j * G + g]);
      }
      const chosen = [...new Set(genres)].filter((g) => this.vocab.has(g));
      if (chosen.length) {
        const w = f32(this.config.genre_weight * f32(1 / f32(Math.sqrt(chosen.length))));
        for (const g of chosen) P[this.vocab.get(g)] = P[this.vocab.get(g)] + w;
      }
      let norm = 0;
      for (let g = 0; g < G; g++) norm += P[g] * P[g];
      norm = f32(Math.sqrt(norm));
      if (norm > 0) for (let g = 0; g < G; g++) P[g] = f32(P[g] / norm);
      const S = new Float32Array(this.n);
      for (let i = 0; i < this.n; i++) {
        let s = 0;
        for (let g = 0; g < G; g++) s += P[g] * this.X[i * G + g];
        S[i] = s;
      }
      return S;
    }

    svd(ratings) {
      const k = this.svdK, n = this.n, u = new Float64Array(k);
      for (const [j, r] of ratings) for (let c = 0; c < k; c++) u[c] += r * this.V[c * n + j];
      for (let c = 0; c < k; c++) u[c] = f32(u[c]);
      const S = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < k; c++) s += u[c] * this.V[c * n + i];
        S[i] = s;
      }
      return S;
    }

    itemKnn(ratings) {
      const S = new Float64Array(this.n);
      if (ratings.size) {
        for (const [j, r] of this.centered(ratings)) {
          for (let p = this.knnPtr[j]; p < this.knnPtr[j + 1]; p++) S[this.knnIdx[p]] += r * this.knnVal[p];
        }
      }
      return Float32Array.from(S);
    }

    score(mode, ratings, genres) {
      const w = this.config.weights[mode];
      const parts = {
        ContentTFIDF: () => this.content(ratings, genres),
        Popularity: () => this.popularity(),
        PureSVD: () => this.svd(ratings),
        ItemKNN: () => this.itemKnn(ratings),
      };
      const seen = new Set(ratings.keys());
      const total = new Float32Array(this.n);
      for (const [name, weight] of Object.entries(w)) {
        if (!weight) continue;
        const Z = zscore(parts[name](), seen), wf = f32(weight);
        for (let i = 0; i < this.n; i++) total[i] = total[i] + f32(wf * Z[i]);
      }
      return total;
    }

    modeFor(ratings) {
      if (ratings.size >= this.config.hybrid_min_ratings) return 'hybrid';
      return ratings.size ? 'cold_start' : 'genres';
    }

    reason(j, genres, liked) {
      if (liked.length) {
        let best = -1, bestSim = 0;
        liked.forEach(([l, row], idx) => {
          const s = row.get(j) || 0;
          if (best < 0 || s > bestSim) { best = idx; bestSim = s; }
        });
        if (bestSim > 0) return `Because you liked ${this.mv.title[liked[best][0]]}`;
      }
      const shared = this.mv.genres[j].filter((g) => genres.includes(g));
      if (shared.length) return 'Matches your genres: ' + shared.slice(0, 3).join(', ');
      const avg = this.mv.avg_rating[j];
      return `Rated ${avg == null ? 'nan' : avg.toFixed(1)} stars on average by ${this.mv.n_ratings[j]} MovieLens viewers`;
    }

    // ---- public API (same results as RecommenderService) ---------------------------
    rateCandidates(genres, n = 15, exclude = []) {
      [genres] = this.cleanInput(genres, {});
      const ex = new Set(exclude.filter((m) => this.col.has(m)).map((m) => this.col.get(m)));
      let picks = [];
      if (genres.length) {
        const inGenre = this.byPopularity.slice(0, 600).filter((j) => genres.some((g) => this.genreSets[j].has(g)));
        picks = inGenre.filter((j) => !ex.has(j)).slice(0, n - Math.floor(n / 3));
      }
      for (const j of this.byPopularity) {
        if (picks.length >= n) break;
        if (!ex.has(j) && !picks.includes(j)) picks.push(j);
      }
      return picks.map((j) => this.card(j));
    }

    search(query, limit = 8) {
      const q = (query || '').trim().toLowerCase();
      if (q.length < 2) return [];
      const hits = [];
      this.displayLower.forEach((t, j) => { if (t.includes(q)) hits.push(j); });
      hits.sort((a, b) => (this.mv.n_ratings[b] - this.mv.n_ratings[a]) || (a - b));
      return hits.slice(0, limit).map((j) => this.card(j));
    }

    recommend(genres, ratings, exclude = [], k = 10, genreFilter = true) {
      const t0 = (root.performance || Date).now();
      let clean;
      [genres, clean] = this.cleanInput(genres, ratings);
      const mode = this.modeFor(clean);
      const scores = Float64Array.from(this.score(mode, clean, genres));
      for (const j of clean.keys()) scores[j] = -Infinity;
      for (const m of exclude) if (this.col.has(m)) scores[this.col.get(m)] = -Infinity;
      if (genreFilter && genres.length) {
        for (let j = 0; j < this.n; j++) if (!genres.some((g) => this.genreSets[j].has(g))) scores[j] = -Infinity;
      }
      const top = stableArgsortDesc(scores).slice(0, k).filter((j) => Number.isFinite(scores[j]));
      const liked = [...clean].filter(([, s]) => s >= LIKED_STARS).map(([j]) => [j, this.knnRow(j)]);
      const items = top.map((j) => this.card(j, this.reason(j, genres, liked)));
      return {
        mode, n_ratings: clean.size, hybrid_min_ratings: this.config.hybrid_min_ratings,
        genre_filter: Boolean(genreFilter && genres.length), items,
        latency_ms: Math.round(((root.performance || Date).now() - t0) * 10) / 10,
      };
    }
  }

  function intList(values) {
    const out = [];
    for (const v of values || []) {
      if (typeof v === 'number' && Number.isFinite(v)) out.push(Math.trunc(v));
      else if (typeof v === 'string' && /^\s*[+-]?\d+\s*$/.test(v)) out.push(parseInt(v, 10));
    }
    return out;
  }

  // Same paths and JSON as app/app.py; errors reject like a failed fetch.
  function makeApi(ready) {
    return async (path, opts = {}) => {
      const rec = await ready;
      const url = new URL(path, 'http://local/');
      const arg = (name) => url.searchParams.get(name) || '';
      switch (url.pathname) {
        case '/api/health': return { status: 'ok' };
        case '/api/genres': return { genres: rec.genres };
        case '/api/rate-candidates':
          return { movies: rec.rateCandidates(arg('genres').split(',').filter(Boolean), 15, intList(arg('exclude').split(','))) };
        case '/api/search': return { movies: rec.search(arg('q')) };
        case '/api/recommend': {
          let body;
          try { body = JSON.parse(opts.body); } catch (e) { body = null; }
          if (!body || typeof body !== 'object' || Array.isArray(body)) body = {};
          const ratings = body.ratings || {};
          const genres = body.genres || [];
          if (typeof ratings !== 'object' || Array.isArray(ratings) && ratings.length) throw new Error(`${path} returned 400`);
          if (!Array.isArray(genres)) throw new Error(`${path} returned 400`);
          const filter = 'genre_filter' in body ? Boolean(body.genre_filter) : true;
          return rec.recommend(genres, Array.isArray(ratings) ? {} : ratings, intList(body.exclude), 10, filter);
        }
        default: throw new Error(`${path} returned 404`);
      }
    };
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Recommender, makeApi };
  } else {
    const base = new URL('model/', document.baseURI);
    const ready = Recommender.load((name, kind) => fetch(new URL(name, base)).then((r) => {
      if (!r.ok) throw new Error(`${name} returned ${r.status}`);
      return kind === 'json' ? r.json() : r.arrayBuffer();
    }));
    root.localApi = makeApi(ready);
  }
})(typeof window !== 'undefined' ? window : globalThis);
