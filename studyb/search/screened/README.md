# Screened results

One CSV file per search and per citation list, written by hand by the searcher (see ../README.md):

* `<source>__<query_id>.csv` for each row of ../search_log.csv, e.g. `google_scholar__Q1-base.csv`
  (sources: `google_scholar`, `semantic_scholar`, `arxiv`, `openreview`; query ids from
  `python -m studyb.search queries`);
* `citations__<list_id>.csv` for each row of ../citations_log.csv.

Columns: `rank,citation,url_or_doi,stage1_decision,reason`, and `duplicate_of` in a file that holds a
duplicate. One row per result screened at title and abstract, ranks 1, 2, ... in the order the source
listed them, as many rows as the log says were screened. `stage1_decision` (the title-and-abstract
stage) is `exclude`, `full_text` or `duplicate`. `url_or_doi` and `reason` are filled on every row
(`url_or_doi` is `none` when the source gives neither): an excluded result's reason names the element
of the inclusion criterion it fails or the Table C.1 "Exclusion" clause it falls under; a duplicate's
reason names its first screened occurrence, whose exact citation text is its `duplicate_of` (empty on
every other row). Every `full_text` result gets a row of ../search_record.csv with the same citation
text, and its reference list is screened too (a row of ../citations_log.csv).
