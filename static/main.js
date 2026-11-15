// State lives only in this tab: selected genres, star ratings, skipped and seen movies.
const state = {
  genres: new Set(),
  ratings: {},          // movieId -> stars (1-5)
  skipped: new Set(),   // haven't seen / don't remember: never sent as a rating
  seen: new Set(),      // "Seen it" on a suggestion: excluded, not rated
  toRate: [],
  hybridMin: 15,
};

const $ = (sel) => document.querySelector(sel);
// The static build (GitHub Pages) defines window.localApi, which answers the same paths in the browser.
const api = window.localApi || ((path, opts) => fetch(path, opts).then((r) => {
  if (!r.ok) throw new Error(`${path} returned ${r.status}`);
  return r.json();
}));

function genresParam() {
  return [...state.genres].map(encodeURIComponent).join(',');
}

// ---- Step 1: genres ---------------------------------------------------------
async function loadGenres() {
  const { genres } = await api('/api/genres');
  const box = $('#genres');
  for (const g of genres) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = g;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => {
      const on = !state.genres.has(g);
      on ? state.genres.add(g) : state.genres.delete(g);
      b.setAttribute('aria-pressed', String(on));
      refreshRecommendations();
      if (Object.keys(state.ratings).length === 0) loadToRate();
    });
    box.appendChild(b);
  }
}

// ---- Step 2: optional ratings ---------------------------------------------------
function rateRow(movie) {
  const row = $('#rate-row').content.firstElementChild.cloneNode(true);
  row.dataset.id = movie.movieId;
  row.querySelector('.title').textContent = movie.title;
  row.querySelector('.meta').textContent = [movie.year, movie.genres.slice(0, 3).join(', ')].filter(Boolean).join(' · ');
  const stars = row.querySelector('.stars');
  stars.setAttribute('aria-label', `Rate ${movie.title}`);
  for (let s = 1; s <= 5; s++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'star';
    b.textContent = '★';
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-label', `${s} star${s > 1 ? 's' : ''}`);
    b.addEventListener('click', () => setRating(movie.movieId, s, row));
    stars.appendChild(b);
  }
  row.querySelector('.skip').addEventListener('click', () => skip(movie.movieId, row));
  paintStars(row, state.ratings[movie.movieId] || 0);
  return row;
}

function paintStars(row, value) {
  row.querySelectorAll('.star').forEach((b, i) => {
    b.classList.toggle('on', i < value);
    b.setAttribute('aria-checked', String(i + 1 === value));
  });
  row.classList.toggle('rated', value > 0);
}

function setRating(id, stars, row) {
  if (state.ratings[id] === stars) {
    delete state.ratings[id];       // clicking the same star again clears it
    stars = 0;
  } else {
    state.ratings[id] = stars;
    state.skipped.delete(id);
  }
  paintStars(row, stars);
  updateMeter();
  refreshRecommendations();
}

function skip(id, row) {
  delete state.ratings[id];
  state.skipped.add(id);
  row.remove();
  updateMeter();
  refreshRecommendations();
  if (!$('#to-rate').children.length) loadToRate();
}

async function loadToRate() {
  const exclude = [...state.skipped, ...Object.keys(state.ratings)].join(',');
  const { movies } = await api(`/api/rate-candidates?genres=${genresParam()}&exclude=${exclude}`);
  const list = $('#to-rate');
  list.replaceChildren(...movies.map(rateRow));
}

function updateMeter() {
  const n = Object.keys(state.ratings).length;
  const pct = Math.min(100, (n / state.hybridMin) * 100);
  $('#meter-fill').style.width = `${pct}%`;
  $('#meter-text').textContent = n >= state.hybridMin
    ? `${n} rated · full model on`
    : `${n} rated · ${state.hybridMin - n} more for the full model`;
}

let searchTimer;
$('#search').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  const box = $('#search-results');
  if (q.length < 2) { box.hidden = true; box.replaceChildren(); return; }
  searchTimer = setTimeout(async () => {
    const { movies } = await api(`/api/search?q=${encodeURIComponent(q)}`);
    box.replaceChildren(...movies.map((m) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = `${m.title}${m.year ? ` (${m.year})` : ''}`;
      b.addEventListener('click', () => {
        const list = $('#to-rate');
        const existing = list.querySelector(`[data-id="${m.movieId}"]`);
        const row = existing || rateRow(m);
        if (!existing) list.prepend(row);
        row.querySelector('.star').focus();
        box.hidden = true;
        $('#search').value = '';
      });
      li.appendChild(b);
      return li;
    }));
    box.hidden = movies.length === 0;
  }, 200);
});

$('#more').addEventListener('click', loadToRate);

// ---- Results ----------------------------------------------------------------------
const MODE_TEXT = {
  genres: () => 'Based on <strong>your genres</strong> and what MovieLens viewers rate highly.',
  cold_start: (n) => `Based on <strong>${n} rating${n > 1 ? 's' : ''}</strong> and your genres.`,
  hybrid: (n) => `Using the <strong>full model</strong> on your ${n} ratings.`,
};

let refreshTimer;
let requestId = 0;
function refreshRecommendations() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(fetchRecommendations, 150);
}

async function fetchRecommendations() {
  const id = ++requestId;
  const nRated = Object.keys(state.ratings).length;
  if (!state.genres.size && !nRated) {
    $('#recs').replaceChildren();
    $('#mode').textContent = '';
    $('#empty').hidden = false;
    return;
  }
  $('#empty').hidden = true;
  $('#recs').classList.add('loading');
  try {
    const data = await api('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        genres: [...state.genres],
        ratings: state.ratings,
        exclude: [...state.seen],
        genre_filter: $('#genre-filter').checked,
      }),
    });
    if (id !== requestId) return;   // a newer request is on its way
    state.hybridMin = data.hybrid_min_ratings;
    $('#mode').innerHTML = MODE_TEXT[data.mode](data.n_ratings);
    $('#recs').replaceChildren(...data.items.map(recCard));
    updateJump();
    $('#empty').hidden = data.items.length > 0;
    if (!data.items.length) $('#empty').textContent = 'No matches left. Turn off "Only my genres" or choose another genre.';
  } catch (err) {
    $('#mode').textContent = 'Could not reach the recommendation server. Please try again.';
  } finally {
    if (id === requestId) $('#recs').classList.remove('loading');
  }
}

function recCard(m) {
  const card = $('#rec-card').content.firstElementChild.cloneNode(true);
  card.querySelector('.title').textContent = m.title;
  card.querySelector('.year').textContent = m.year ? `(${m.year})` : '';
  const aka = card.querySelector('.aka');
  aka.textContent = m.aka || '';
  aka.hidden = !m.aka;
  const avg = m.avg_rating ? `${m.avg_rating.toFixed(1)} ★ from ${m.n_ratings} ratings` : 'not yet rated';
  card.querySelector('.meta').textContent = `${m.genres.join(', ') || 'No genre listed'} · ${avg}`;
  card.querySelector('.reason').textContent = m.reason;
  const tmdb = card.querySelector('.tmdb');
  if (m.tmdb_url) tmdb.href = m.tmdb_url; else tmdb.hidden = true;
  card.querySelector('.imdb').href = m.imdb_url;
  const seen = card.querySelector('.seen');
  seen.setAttribute('aria-label', `I've seen ${m.title}; show something else`);
  seen.addEventListener('click', () => {
    state.seen.add(m.movieId);
    refreshRecommendations();
  });
  return card;
}

$('#genre-filter').addEventListener('change', refreshRecommendations);

// On narrow screens the suggestions sit below the rating list; offer a jump while they are out of view.
const jump = $('#jump');
let resultsVisible = false;
function updateJump() {
  jump.classList.toggle('show', !resultsVisible && $('#recs').children.length > 0);
  jump.textContent = `See your ${$('#recs').children.length} suggestions \u2193`;
}
new IntersectionObserver(([entry]) => { resultsVisible = entry.isIntersecting; updateJump(); })
  .observe($('.results'));
jump.addEventListener('click', () => {
  $('.results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('#results-title').focus({ preventScroll: true });
});
$('#results-title').tabIndex = -1;

loadGenres().then(loadToRate).then(() => { updateMeter(); fetchRecommendations(); });
