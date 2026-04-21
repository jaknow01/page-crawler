# SEO Semantic Competitor Analysis

System do semantycznej analizy konkurencji w SEO dla placowki medycznej. Automatycznie crawluje wlasna strone i strony konkurencji, zapisuje embeddingi do bazy wektorowej, klastruje tematy oraz przygotowuje raport luk tresciowych. Wyniki sa widoczne w interaktywnym UI.

## Jak to dziala

1. Crawlowanie wlasnej strony (Firecrawl).
2. Lista zapytan z queries.yaml (Google search).
3. Crawlowanie stron konkurencji z wynikow top-k.
4. Zapis stron do Qdrant + embeddingi (OpenAI).
5. Klastrowanie HDBSCAN + UMAP do 2D.
6. Gap analysis: porownanie wlasnej strony do centroidow klastrow.
7. UI w Streamlit.

## Wymagania

- Docker i docker compose
- Dostepy API (OpenAI, Firecrawl, Claude Code)
- Python 3.11+ i uv (opcjonalnie, tylko do uruchamiania lokalnego)

## Konfiguracja

Pliki:
- config.yaml (parametry crawlowania, klastrowania, model)
- queries.yaml (lista zapytan)
- .env (klucze API)

Zmienne srodowiskowe (przyklad):
- OPENAI_API_KEY
- FIRECRAWL_API_KEY
- ANTHROPIC_API_KEY lub CLAUDE_CODE_OAUTH_TOKEN
- QDRANT_URL (domyslnie http://localhost:6333)

### Uwaga
Dla wielu zadanych queries działanie tego programu prawdopodobnie wykorzysta wszystkie tokeny sesji Claude Code (nawet dla słabszego modelu Haiku). W przypadku osiągnięcia limitu 5-godzinnej sesji program zatrzyma się i będzie oczekiwał na zresetowanie limitu, po czym kontynuuje pracę. W dowolnym momencie możesz przerwać pracę systemu - wyniki crawlowania są na bieżąco zapisywane w bazie, a zadania w kolejce zostaną zapamiętane.

Dodatkowo upewnij się że posiadasz wystarczającą liczbę kredytów w Firecrawl API.

## Szybki start (docker)

1) Infrastruktura:

```
docker compose up -d qdrant
```

2) Crawlowanie:

```
docker compose run --rm crawler crawl-own
docker compose run --rm crawler crawl-competitors --query-limit 5
docker compose run --rm crawler crawl-competitors
```

3) Analiza:

```
docker compose run --rm crawler analyze
```

4) UI:

```
docker compose up -d ui
```

UI jest domyslnie pod adresem http://localhost:8502

## Wyniki

Pliki wynikowe trafiaja do data/reports:
- clusters.json
- reduction.parquet
- gap_report.json

Baza Qdrant jest w data/qdrant.

## Komendy CLI

```
docker compose run --rm crawler crawl-own
docker compose run --rm crawler crawl-competitors
docker compose run --rm crawler analyze
docker compose run --rm crawler run-all
docker compose run --rm crawler stats
```
