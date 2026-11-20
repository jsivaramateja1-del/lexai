# Deploying LexAI

Everything below is free. The app is a Streamlit process with local embeddings and a local vector
store, which shapes what deployment can and cannot do.

## What deploys, and what has to change

| Piece | Local | On Streamlit Community Cloud |
| --- | --- | --- |
| Embedding model | cached in your user folder | downloaded on first run, ~470 MB, then cached |
| Vector store | `chroma_db/` on disk | **must be built at boot** — the filesystem is ephemeral |
| API key | `.env` | `st.secrets`, bridged into the environment by `src/config.py` |
| Query log | `analytics/queries.csv` | lost on restart (ephemeral disk) |

That third row is already handled: `src/config.py` copies Streamlit secrets into the environment,
because every module reads `os.environ`. Without that bridge a deployed app silently runs in quote
mode with no key — the failure looks like "the model is down" when it is really "the key is
invisible".

## Steps

1. **Push everything.** `git push origin main`. Confirm `data/documents/*.pdf` is in the repo so the
   index can be rebuilt from source.
2. **Create the app.** Go to share.streamlit.io → *New app* → pick the repository
   `jsivaramateja1-del/lexai`, branch `main`, main file `app.py`.
3. **Add secrets.** In *Settings → Secrets*, paste:

   ```toml
   SARVAM_API_KEY = "your-key-here"
   SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"
   SARVAM_MODEL = "sarvam-105b-conversations"
   LLM_ENABLED = "true"
   MAX_DISTANCE = "0.40"
   STRONG_DISTANCE = "0.28"
   ```

4. **Build the index at boot.** The deployed filesystem is wiped on redeploy, so the app must be able
   to rebuild `chroma_db/` itself. Add this to `app.py`, above `main()`:

   ```python
   @st.cache_resource(show_spinner="Building the index from the bundled documents…")
   def ensure_index() -> int:
       from src.config import DATA_DIR
       from src.ingestor import ingest_folder
       from src import retriever

       if retriever.stats()["chunks"] == 0:
           retriever.add_chunks(ingest_folder(DATA_DIR))
       return retriever.stats()["chunks"]
   ```

   Call `ensure_index()` next to `warm_embeddings()` in `main()`. First boot takes one to two
   minutes; later boots reuse the cache.

5. **Check three things after it deploys:** the sidebar shows 629 chunks, a Patents question answers
   with a page citation, and an off-topic question still refuses. If chunks show 0, the index rebuild
   did not run — check the logs for a missing document path.

## Limits you should state, not hide

- **Ephemeral disk.** Query logs do not survive a restart, so the dashboard must be built from a local
  run, not from the deployed app.
- **Cold starts.** The embedding model reloads after the app sleeps; the first question after idle is
  slow.
- **Memory.** The free tier gives about 1 GB. Sentence-transformers plus ChromaDB fits, but barely;
  if it crashes, switch `EMBEDDING_MODEL` to `sentence-transformers/all-MiniLM-L6-v2` (English-only,
  much smaller) and re-ingest.
- **No authentication.** Anyone with the URL can use your API credits. For a demo, set
  `LLM_ENABLED = "false"` in secrets to serve quote mode for free, and turn it on only while you are
  presenting.
- **Concurrency.** ChromaDB local mode is single-process; two visitors at once are fine, a crowd is
  not. Say so rather than pretending otherwise.

## Alternative if the free tier is tight

Deploy the **landing page** to GitHub Pages (static, free, no limits) and keep the app local for your
demo. The landing page then links to a short screen recording instead of a live app. That is a
legitimate choice and worth one line in your report.
