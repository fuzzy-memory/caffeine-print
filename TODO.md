# Refactor TODO

Goal: make the pipeline more robust and performant, and make it flexible
enough to send different types of news (not just politics), without a full
rewrite. Sections match the call order in `src/main.py`.

## 0. Config layer (blocks everything else)

- [ ] Define a `NewsProfile` Pydantic model (`tags: dict[str, int]`,
      `gpt_category_multipliers: dict[str, float]`, `sentiment_threshold`,
      `dbscan_epsilon`, `chatgpt_model`, exclusion rules, etc.)
- [ ] Move current `properties.py` politics-specific values into a
      `profiles/politics.yaml` (or `.json`) loaded into `NewsProfile` at
      startup
- [ ] Load `assets/control.json` once at startup into a typed model; pass it
      down instead of re-reading in `get_news.py`, `similarity_clustering.py`,
      and `ranking.py`
- [ ] Add a shared `get_asset_path(name, test_mode)` helper to replace the
      duplicated `"assets/" + ("test/" if test_mode else "")` string-building
      across `get_news.py`, `similarity_clustering.py`, `ranking.py`, `htmls.py`
- [ ] Rename `src/db/data_models.py` or `src/data_models.py` so the two
      same-named, unrelated modules (Pydantic vs. SQLAlchemy) aren't confusable
- [ ] Unify "test mode" naming (`testing_flag`, `test_mode`,
      `testing_gpt_threshold`) to one convention

## 1. `get_news_from_api` (get_news.py)

- [ ] Fix `news_items_per_call` bug (line ~135) — uses
      `total_items - received` instead of a correct "remaining items" calc
      matching the branch above it
- [ ] Replace `exit(1)` with a raised exception, caught in `main.py`
- [ ] Drive query params (sources, tags, date range) from `NewsProfile`
      instead of hardcoded `permitted_callers`/tags
- [ ] Add retry/backoff around paginated API calls
- [ ] Remove dead commented-out `# "news-sources": news_sources,` line
- [ ] Investigate whether pagination can be parallelized (only if call counts
      are actually high — verify before committing)

## 2. `deduplicate_articles` (similarity_clustering.py)

- [ ] Fix mutate-while-iterating bug (`news_items.pop(news_items.index(...))`
      inside a `for article in news_items` loop) — build a filtered list
      instead
- [ ] Check whether SBERT encoding is batched; batch it if not
- [ ] Read source-score filtering from the shared config object instead of a
      fresh `control.json` load

## 3. `rank_articles` (ranking.py)

- [ ] Split into `score_articles(df, profile)` (pure scoring) and
      `persist_results(df, test_mode)` (I/O), so scoring is testable without
      touching disk
- [ ] Replace `assert set(metric_dict.keys()) == ...` and other
      production-affecting asserts with explicit `if/raise`
- [ ] Move hardcoded entertainment filter (lines ~174-178) into
      `NewsProfile` as a configurable exclusion rule
- [ ] Add retry/backoff around OpenAI scoring calls
- [ ] Check whether GPT scoring is one-call-per-article; batch if so
      (biggest perf lever in the pipeline, also relevant to §6)

## 4. `render_news` (htmls.py)

- [ ] Move inline CSS/HTML string building to a Jinja2 template file
- [ ] Extract `mean_report_count` / "BREAKING" labeling logic into a pure
      function, separate from HTML string-building
- [ ] Confirm template approach can support per-profile layouts later

## 5. `send_email` (main.py → mailer.py)

- [ ] Extract `send_email` out of `main.py` into `mailer.py`
      (`send_digest(html, recipients, subject)`)
- [ ] Make per-recipient send failures non-fatal — collect failures, send to
      all valid recipients, report failures at the end

## 6. LLM evaluation rubric (new)

- [ ] Build `assets/eval/golden_set.json` — small fixed batch of real
      articles with human-labeled expected `GPTArticleEvaluationMetrics`
- [ ] Write `eval/rubric.py` — scoring rubric (relevance, category accuracy,
      sentiment accuracy, entertainment-classification accuracy)
- [ ] Write `eval/run_eval.py` — runs the golden set through the existing
      GPT-scoring function (reused from `ranking.py`) across multiple model
      names, reports per-model accuracy/latency/cost
- [ ] Use eval results as a regression check before changing
      `chatgpt_model` in any `NewsProfile`

## Deferred / lower priority

- [ ] Add `pydantic`, `tqdm`, `python-dateutil` as direct deps in
      `pyproject.toml` (currently only transitive)
- [ ] Replace manual `to_json()`/`__str__` methods in `data_models.py` with
      Pydantic's `model_dump()`
- [ ] Remove unused `assets/incomplete_news.json` recovery path (nothing
      reads it)
- [ ] Add a `pytest` smoke-test suite (currently 0% coverage) — at minimum
      cover `ranking.py` scoring and `data_models.py` validators
