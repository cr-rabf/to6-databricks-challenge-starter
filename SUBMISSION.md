# My dataset submission

Complete this file in Databricks. Keep README.md as the course navigation page.

## Inputs and outputs

**Inputs** (read-only, synthetic, tax years 2022–2024):

| Role | Full table name | Delta version checked |
| --- | --- | --- |
| Initial batch (rejected) | `training.to6_challenge3_inputs.returns_initial` | 2 |
| Corrected batch (accepted) | `training.to6_challenge3_inputs.returns_corrected` | 2 |
| Lineage view | `training.to6_challenge3_inputs.source_lineage` | n/a (view) |
| Upstream source named by the lineage view | `training.to6_challenge3_inputs.parsed_source` | not read |

**What one row represents:** one tax return.
* `return_id` identifies the return and must be unique within a batch.
* `taxpayer_id` identifies the filer, who can have returns in several years (e.g. F0 has R01 in 2022 and R09 in 2024).
* Required fields: `return_id`, `tax_year`, `income`, `deduction`.
* Amounts are nonnegative, and zero income is valid.

**Outputs** (individual output schema `training.to6_c3_48f0f0239a0e`; every table is replaced on each run):

| Full table name | Contents | Rows |
| --- | --- | --- |
| `training.to6_c3_48f0f0239a0e.return_features` | every accepted return | 12 |
| `training.to6_c3_48f0f0239a0e.features_train` | `return_features` rows with `tax_year` = 2022 | 4 |
| `training.to6_c3_48f0f0239a0e.features_validation` | `return_features` rows with `tax_year` = 2023 | 4 |
| `training.to6_c3_48f0f0239a0e.features_test` | `return_features` rows with `tax_year` = 2024 | 4 |

All four tables have exactly three columns: `return_id` (string), `tax_year` (int) and `deduction_to_income` (double). Under the need-to-know rule, `taxpayer_id`, `income` and `deduction` are excluded.

**Ratio definition:** `deduction_to_income = deduction / income`, and **NULL when income = 0**. The zero-income return stays in the table.
* The zero-income rule is written explicitly (`when(income == 0, NULL)`).
* Serverless runs in ANSI mode, so a plain division raises `DIVIDE_BY_ZERO` on R03 and R09.

## Checks and decisions

**Source references** (notebook cells 5–7):
* **Catalog descriptions:**
  * `returns_initial`: "Initial synthetic delivery from parsed_source; known delivery defects. Validate before feature output."
  * `returns_corrected`: "Corrected synthetic delivery from parsed_source; one row per return. Validate required values and return_id uniqueness."
* **Lineage view:** both delivered tables come from `training.to6_challenge3_inputs.parsed_source`. The view states that it does not claim a Catalog UI lineage check.
* **Sample rows:** three per batch.
* **Full-batch profile:**

| Summary | returns_initial | returns_corrected |
| --- | --- | --- |
| Rows (2022 / 2023 / 2024) | 13 (5 / 4 / 4) | 12 (4 / 4 / 4) |
| Missing required values | 1 (R02 `income`) | 0 |
| Duplicate return IDs | 1 (R01, 2 identical rows) | 0 |
| Zero-income returns | 2 (R03, R09) | 2 (R03, R09) |
| Negative amounts | 0 | 0 |

**Rules:** notebook cell 9 applies one function, `check_batch`, unchanged to both batches. Every row is checked.
* **Completeness:** `return_id`, `tax_year`, `income` and `deduction` are not null.
* **Uniqueness:** each `return_id` appears exactly once in the batch.
* **Nonnegative:** `income` and `deduction` are >= 0.
* A batch is accepted only with zero violations.
* Feature output is gated on `ACCEPTED_TABLE`, which is set only when a batch passes.

**Initial rejection:** `returns_initial` (13 rows) is rejected before any feature output is written.
* **Uniqueness:** R01 (2022, income 100, deduction 20) appears twice. Both copies are kept as affected records.
* **Completeness:** R02 (2022, deduction 33) has a null `income`.
* **Not done, because none of these is a correction:**
  * dropping an R01 copy
  * setting R02's income to 0
  * copying values from the corrected batch
  * relaxing or removing a check
* **Next action:** report the affected records to the data owner, then assess `returns_corrected` with the same rules.

**Rejection evidence:**
* **Rejection reference:** `outputs/runs/<run_id>/initial_rejection.json`, written by every run. It records the table, Delta version, rules, counts per rule, affected IDs, decision and next action.
* **Affected-record export:** `outputs/initial_rejected_records.csv`, with a per-run copy in each run folder. It holds 3 rows (R01 ×2 for uniqueness, R02 for completeness), each with its rule and reason. `taxpayer_id` is redacted to `**`, and R02's unknown income is left empty rather than set to 0.

**Replacement checks:** `returns_corrected` passed the same rules with 0 violations (12 rows, 12 distinct `return_id`s).
* A reconciliation (cell 9, step 2e) shows that the replacement changes only the reported defects: one R01 copy is removed and R02's income is supplied (200).
* All other rows are identical.

**Independently derived nonzero expected value** (cell 12, step 3e):
* **R01:** income 100 and deduction 20, so the hand-derived ratio is 20 / 100 = **0.2**. It was also recomputed in Python from the source row. The saved value read back from the table is 0.2 (tolerance 1e-12).
* **Second example:** R02 = 33 / 200 = 0.165.

**Saved-table readback, including zero income** (cell 12, step 3e). `return_features` is read back from the catalog and checked for:
* exactly the three contract columns and types
* 12 rows and 12 distinct IDs
* the same ID set as the source
* 4 rows per year
* R03 (2022) and R09 (2024) present with a **null** ratio
* every ratio matching an independent Python recompute from the source, with 0 mismatches

The pre-write checks (step 3c) compare row count, uniqueness and the ID set against the accepted source.

## Two successful runs

Both runs are interactive notebook runs, using **Run all** on serverless compute. No Jobs run IDs are needed.

| | Run 1 | Run 2 |
| --- | --- | --- |
| Run ID / evidence folder | `outputs/runs/20261120T171851Z/` | `outputs/runs/20261120T172041Z/` |
| Notebook URL | https://dbc-48a16403-f66f.cloud.databricks.com/?o=7474648218609329#notebook/127024866558081 | https://dbc-48a16403-f66f.cloud.databricks.com/?o=7474648218609329#notebook/127024866558081 |
| Execution start (UTC) | 2026-11-20T17:18:51.751646Z | 2026-11-20T17:20:41.463169Z |
| Execution completed (UTC) | 2026-11-20T17:19:36.917686Z | 2026-11-20T17:21:21.542668Z |
| Status in `run.json` | succeeded | succeeded |
| Checks passed (`checks.csv`) | 34 / 34 | 34 / 34 |
| `return_features` Delta version after run | 3 | 5 |
| Year tables Delta version after run | 0 (first write) | 2 |

Each run folder holds:
* `run.json`: run identity, input Delta versions, decisions, output tables and versions, and snapshot SHA-256s.
* `checks.csv`: every 3c, 3e and 4b check, with expected value, actual value and result.
* `initial_rejection.json` and `initial_rejected_records.csv`.
* Sorted readback snapshots of all four output tables: the actual return IDs and values.

**IDs-and-values comparison:** `outputs/run_comparison.json` (written by run 2, cell 15, step 4e) compares the two runs from their saved snapshot files, not the live tables. For each table it records the byte-level SHA-256, the return IDs, and a row-by-row multiset diff in both directions. Result: **match = true** for all four tables, with 0 differing rows.

| Snapshot | Rows (run 1 / run 2) | SHA-256 (identical in both runs) |
| --- | --- | --- |
| `return_features.csv` | 12 / 12 | `382f97d69814056f81e58f6166b4b5761e46a17cfeb682046fd22b785827b2fb` |
| `features_train.csv` | 4 / 4 | `85edc0fe84a534383e8d32e28e5dd0665ed615664095004ac3cd2bc64151f11b` |
| `features_validation.csv` | 4 / 4 | `4f997086def1a1c98d298d5ddf6679dd357cf8ca02e318e8d1f8b76f09077ba8` |
| `features_test.csv` | 4 / 4 | `f133daffb5d300f2c7fe1e8418ac9aea59a06f7345b2468c6710ddc2a8e10052` |

The comparison can be reproduced by hashing or diffing the committed CSVs in the two run folders. The two runs' `checks.csv` files differ only in the Delta version numbers recorded by the "latest write replaced the table" checks, which is expected because each run writes new versions. All 34 checks passed in both runs.

**How the key properties were checked** (cell 15, step 4b, read back from the catalog in each run):
* **Year assignment:**
  * Each year table has 0 rows with any other `tax_year`.
  * Each table's row count equals the `return_features` rows for its year (4 / 4 / 4).
* **One copy of every accepted return:**
  * The three year tables total 12 rows, equal to the accepted returns.
  * No `return_id` appears in more than one table.
  * The union equals `return_features` on IDs and values in both directions, with 0 missing and 0 extra.
  * Each table has one row per `return_id`.
* **No appended copies:**
  * Every table is written with overwrite mode.
  * In each table's Delta history, the latest data write is `CREATE OR REPLACE TABLE AS SELECT`, not an append.
  * After run 2, row counts are still 12 / 4 / 4 / 4 with no duplicate IDs, and the live tables still hash to the values above.

**Partial run (not one of the two successful runs):** `outputs/runs/20261120T161138Z/` is kept and flagged as a partial run.
* It holds Outcome 2 from 16:11 UTC and Outcome 3 from 16:29 UTC. These were separate cell runs in one session during development, not one end-to-end notebook run.
* Its `run.json` has no `status: succeeded`, so `run_comparison.json` lists it under `skipped_runs`, with the reason.
* Its `return_features.csv` has the same SHA-256 as the successful runs (`382f97d6…27b2fb`).

## Final files and limitations

**Final exports under `outputs/`** (synthetic, small):
* `return_features.csv`, `features_train.csv`, `features_validation.csv`, `features_test.csv`: sorted readbacks of the saved tables, the stable final copies from run 2.
* `initial_rejected_records.csv`: the affected initial records with rule and reason (`taxpayer_id` redacted to `**`).
* `run_comparison.json`: the two-run IDs-and-values comparison.
* `runs/20261120T171851Z/` and `runs/20261120T172041Z/`: the per-run evidence for the two successful runs, as listed above.
* `runs/20261120T161138Z/`: the partial development run (see above), kept for reference only.

**Practice file (not part of the task):** `outputs/connection-practice.json`, from the ungraded connection check.

**Not verified, or known limitations:**
* **Model leakage and labels:** these checks establish this task's data contract only. They do not show that the ratio is free of leakage or available at prediction time. Labels and that assessment are left to later modelling work.
* **GitHub "Starter boundary tests" check:** pushing to `main` runs `tests/test_platform.py`. One test executes the whole `notebook.py` with a fake `spark` object. The practice cell and the task code call things the fake doesn't provide (`.asDict()`, `display`, `dbutils`, table reads), so this check is expected to fail. The notebook was not changed to make it pass, because CHALLENGE.md does not list it as part of the assessment. The evidence for the task is the two Databricks runs and the exports above.
* **Notebook results are not in `notebook.py`:** the Git folder commits the notebook as source, without cell outputs. The results of the inspection, check and readback steps are therefore kept as files under `outputs/` and as summaries in this file and the notebook markdown. The Outcome 1 profile is in the notebook's displayed results and summarised here; it is not exported as a separate file.
* **Notebook edit between the runs:** the workspace records the notebook as last modified at 2026-11-20T17:20:33Z, between run 1 (completed 17:19:36Z) and run 2 (started 17:20:41Z). What, if anything, changed could not be confirmed from the saved files. Both runs produced identical outputs and passed the same 34 checks. The notebook has not been modified since run 2.
* **Same notebook URL for both runs:** both runs are interactive runs of the same notebook, so the URL is the same; the execution timestamps and run IDs distinguish them.
* **`parsed_source` contents:** not read. The upstream source is taken from the supplied lineage view and Catalog descriptions only.
* **Commit SHA:** reported in Classroom, not in this file.
