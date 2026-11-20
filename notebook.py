# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Prepare the next team's dataset
# MAGIC Follow CHALLENGE.md. These headings are a starting point; implement and verify each stage yourself.

# COMMAND ----------

import hashlib

INPUT_SCHEMA = "training.to6_challenge3_inputs"
INITIAL_TABLE = f"{INPUT_SCHEMA}.returns_initial"
CORRECTED_TABLE = f"{INPUT_SCHEMA}.returns_corrected"
LINEAGE_VIEW = f"{INPUT_SCHEMA}.source_lineage"
workspace_user = spark.sql("SELECT current_user() AS username").first()["username"]
OUTPUT_SCHEMA = "training.to6_c3_" + hashlib.sha256(workspace_user.encode("utf-8")).hexdigest()[:12]
print({"input_schema": INPUT_SCHEMA, "output_schema": OUTPUT_SCHEMA})

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 1 AS connection_check;

# COMMAND ----------

from pathlib import Path
import json

# Copy this Git folder's absolute path from its Workspace menu; include /Workspace/.
repo_root = Path("/Workspace/Users/steven.c.rose.irs@catalyte.ai/to6-databricks-challenge-starter")
assert (repo_root / "README.md").is_file(), "Check the Git folder path"
practice_result = spark.sql("SELECT 1 AS connection_check").first().asDict()
assert practice_result == {"connection_check": 1}
(repo_root / "outputs").mkdir(exist_ok=True)
(repo_root / "outputs/connection-practice.json").write_text(
    json.dumps(practice_result) + "\n", encoding="utf-8"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understand the input
# MAGIC Record Catalog descriptions, sample observations and the supplied lineage source.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understand the input
# MAGIC
# MAGIC **Source reference**
# MAGIC * Inputs: `training.to6_challenge3_inputs.returns_initial` and `training.to6_challenge3_inputs.returns_corrected` (managed Delta tables, synthetic, tax years 2022–2024).
# MAGIC * Catalog descriptions:
# MAGIC   * `returns_initial`: "Initial synthetic delivery from parsed_source; known delivery defects. Validate before feature output."
# MAGIC   * `returns_corrected`: "Corrected synthetic delivery from parsed_source; one row per return. Validate required values and return_id uniqueness."
# MAGIC * Lineage (`training.to6_challenge3_inputs.source_lineage`): both tables were created by SQL from the upstream table `training.to6_challenge3_inputs.parsed_source`. The view itself notes that it does not claim a Catalog UI lineage check.
# MAGIC
# MAGIC **Row meaning and fields**
# MAGIC * One row = one tax return.
# MAGIC * `return_id` (string): identifies the return and must be unique within a batch.
# MAGIC * `taxpayer_id` (string): identifies the filer, who can have returns in several years (e.g. F0 has R01 in 2022 and R09 in 2024). Not unique; excluded from outputs (need-to-know).
# MAGIC * `tax_year` (int), `income` (int), `deduction` (int): amounts are nonnegative, and zero income is valid.
# MAGIC * Required fields: `return_id`, `tax_year`, `income`, `deduction`.
# MAGIC
# MAGIC The next cells show three sample rows per batch, then summarise all rows of both batches (year counts, missing required values, duplicate return_ids, zero income).

# COMMAND ----------

# DBTITLE 1,Inspect inputs: descriptions, lineage, samples and full-batch profile
# Outcome 1: understand the input (read-only; nothing is written here).
from pyspark.sql import functions as F

BATCHES = {"returns_initial": INITIAL_TABLE, "returns_corrected": CORRECTED_TABLE}
REQUIRED = ["return_id", "tax_year", "income", "deduction"]

# 1a. Catalog descriptions (table comments) for both batches.
descriptions = None
for name, table in BATCHES.items():
    d = (spark.sql(f"DESCRIBE TABLE EXTENDED {table}")
         .filter(F.col("col_name") == "Comment")
         .select(F.lit(name).alias("batch"), F.lit(table).alias("table"),
                 F.col("data_type").alias("catalog_description")))
    descriptions = d if descriptions is None else descriptions.unionByName(d)
print("1a. Catalog descriptions (table comments) for both batches")
display(descriptions)

# 1b. Supplied lineage view: delivered table -> upstream source.
print(f"1b. Supplied lineage view ({LINEAGE_VIEW}): delivered table -> upstream source")
display(spark.table(LINEAGE_VIEW))

# 1c. At least three sample rows from each batch (row meaning only, not quality evidence).
for name, table in BATCHES.items():
    print(f"1c. Sample rows: first 3 rows of {name} by return_id (row meaning only, not quality evidence)")
    display(spark.table(table).orderBy("return_id").limit(3)
            .select(F.lit(name).alias("batch"), "*"))

# Full batches: every row of both tables, labelled by batch. No LIMIT or sampling below.
all_rows = None
for name, table in BATCHES.items():
    t = spark.table(table).withColumn("batch", F.lit(name))
    all_rows = t if all_rows is None else all_rows.unionByName(t)

# 1d. Returns by tax year, per batch.
print("1d. Returns by tax year, per batch (all rows)")
display(all_rows.groupBy("batch", "tax_year").count().orderBy("batch", "tax_year"))

# 1e. One profile row per batch: row count, missing required values,
#     duplicate return IDs (counted within each batch) and zero-income returns.
id_counts = all_rows.groupBy("batch", "return_id").agg(F.count("*").alias("n"))
dup_summary = id_counts.groupBy("batch").agg(
    F.countDistinct("return_id").alias("distinct_return_ids"),
    F.sum(F.when(F.col("n") > 1, 1).otherwise(0)).alias("duplicated_return_ids"),
    F.sum(F.when(F.col("n") > 1, F.col("n")).otherwise(0)).alias("rows_in_duplicates"),
)
profile = all_rows.groupBy("batch").agg(
    F.count("*").alias("rows"),
    *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"missing_{c}") for c in REQUIRED],
    F.sum(F.when(F.col("income") == 0, 1).otherwise(0)).alias("zero_income"),
    F.sum(F.when((F.col("income") < 0) | (F.col("deduction") < 0), 1).otherwise(0)).alias("negative_amounts"),
)
print("1e. Full-batch profile: row count, missing required values, zero income, negative amounts, duplicate return IDs")
display(profile.join(dup_summary, "batch").orderBy("batch"))

# 1f. The actual records behind the findings: missing required values,
#     duplicate return IDs and zero-income returns, with the finding named.
flagged = (all_rows.join(id_counts, ["batch", "return_id"], "left")
    .withColumn("finding", F.concat_ws("; ",
        *[F.when(F.col(c).isNull(), F.lit(f"missing {c}")) for c in REQUIRED],
        F.when(F.col("n") > 1, F.lit("duplicate return_id")),
        F.when(F.col("income") == 0, F.lit("zero income (valid)"))))
    .filter(F.col("finding") != "")
    .select("batch", "return_id", "taxpayer_id", "tax_year", "income", "deduction", "finding")
    .orderBy("batch", "return_id"))
print("1f. Flagged records: missing required values, duplicate return IDs and zero-income returns")
display(flagged)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Decide which batch is usable
# MAGIC Keep the initial rejection evidence and apply the same checks to the replacement.
# MAGIC
# MAGIC **Data-contract rules** (one function, `check_batch`, applied unchanged to both batches; every row is checked):
# MAGIC * **Completeness:** `return_id`, `tax_year`, `income` and `deduction` are not null.
# MAGIC * **Uniqueness:** each `return_id` appears exactly once in the batch.
# MAGIC * **Nonnegative:** `income` and `deduction` are >= 0 (zero income is valid).
# MAGIC
# MAGIC A batch is accepted only with zero violations. Feature output (Outcome 3) is gated on `ACCEPTED_TABLE`, which is set only when a batch passes.
# MAGIC
# MAGIC **Per-run evidence** in `outputs/runs/<run_id>/`: `run.json` (run ID, notebook URL, execution timestamp, input Delta versions, decisions), `initial_rejection.json` (rejection reference) and `initial_rejected_records.csv` (affected records with rule and reason). `outputs/initial_rejected_records.csv` is the stable final copy. `taxpayer_id` is redacted to `**` in all exports.

# COMMAND ----------

# DBTITLE 1,Assess both batches, reject initial, save per-run evidence
# Outcome 2: decide which batch is usable. Same rules for both batches; feature output is gated on the result.
import datetime
import json
from pyspark.sql import Window
from pyspark.sql import functions as F

# 2a. Run identity, reused by Outcomes 3 and 4 for this run's evidence.
RUN_STARTED_UTC = datetime.datetime.now(datetime.timezone.utc)
RUN_ID = RUN_STARTED_UTC.strftime("%Y%m%dT%H%M%SZ")
try:
    _attrs = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())["attributes"]
    NOTEBOOK_URL = f"https://{_attrs['browserHostName']}/?o={_attrs['orgId']}#notebook/{_attrs['notebook_id']}"
    NOTEBOOK_PATH = _attrs.get("notebook_path")
    del _attrs  # the context also holds an API token; keep only the fields above
except Exception as e:
    NOTEBOOK_URL, NOTEBOOK_PATH = None, None
    print(f"Notebook URL not available from context ({type(e).__name__}); record it manually for this run.")
RUN_DIR = repo_root / "outputs" / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_COLS = ["return_id", "taxpayer_id", "tax_year", "income", "deduction"]
print("2a. Run identity")
print({"run_id": RUN_ID, "run_started_utc": RUN_STARTED_UTC.isoformat(), "notebook_url": NOTEBOOK_URL, "run_dir": str(RUN_DIR)})

def table_version(table):
    """Latest Delta version of a table, so the evidence names exactly which data was checked."""
    return spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"]

def check_batch(table):
    """Apply the data-contract rules to every row; return one violation row per affected record and rule."""
    df = (spark.table(table).select(*SOURCE_COLS)
          .withColumn("id_rows", F.count(F.lit(1)).over(Window.partitionBy("return_id"))))
    checks = [(F.col(c).isNull(), "completeness", F.lit(f"required field {c} is null")) for c in REQUIRED]
    checks.append((F.col("return_id").isNotNull() & (F.col("id_rows") > 1), "uniqueness",
                   F.concat(F.lit("return_id appears "), F.col("id_rows").cast("string"), F.lit(" times in batch"))))
    checks.append(((F.col("income") < 0) | (F.col("deduction") < 0), "nonnegative",
                   F.lit("income or deduction is negative")))
    violations = None
    for condition, rule, reason in checks:
        v = df.filter(condition).select(F.lit(table).alias("source_table"), *SOURCE_COLS,
                                        F.lit(rule).alias("rule"), reason.alias("reason"))
        violations = v if violations is None else violations.unionByName(v)
    violations = violations.orderBy("return_id", "rule", "reason")
    by_rule = {r["rule"]: r["count"] for r in violations.groupBy("rule").count().collect()}
    summary = {
        "table": table,
        "delta_version": table_version(table),
        "rows_checked": df.count(),
        "rules": ["completeness: return_id, tax_year, income, deduction are not null",
                  "uniqueness: each return_id appears exactly once",
                  "nonnegative: income and deduction are >= 0"],
        "violations_by_rule": {r: by_rule.get(r, 0) for r in ["completeness", "uniqueness", "nonnegative"]},
        "affected_return_ids": sorted({r["return_id"] for r in violations.select("return_id").collect()}, key=str),
    }
    summary["decision"] = "accepted" if sum(summary["violations_by_rule"].values()) == 0 else "rejected"
    return violations, summary

def export_redacted(violations, path):
    """Write the affected records with taxpayer_id redacted to "**"; values kept as text so nulls stay empty."""
    pdf = (violations.withColumn("taxpayer_id", F.lit("**"))
           .select(*[F.col(c).cast("string") for c in violations.columns]).toPandas())
    pdf.to_csv(path, index=False)
    return len(pdf)

# 2b. Assess the initial batch. It must be rejected before any feature output is written.
initial_violations, initial_summary = check_batch(INITIAL_TABLE)
print("2b. Initial batch: rule violations (one row per affected record and rule)")
display(initial_violations)
assert initial_summary["decision"] == "rejected", "returns_initial unexpectedly passed; investigate before continuing"
initial_summary["next_action"] = ("Do not repair or drop records. Report the affected records to the data owner "
                                  "and assess returns_corrected with the same rules.")

# 2c. Preserve the affected records and reasons: this run's folder plus the stable top-level copy.
for path in [RUN_DIR / "initial_rejected_records.csv", repo_root / "outputs" / "initial_rejected_records.csv"]:
    n_exported = export_redacted(initial_violations, path)
rejection = {"run_id": RUN_ID, "run_started_utc": RUN_STARTED_UTC.isoformat(), "notebook_url": NOTEBOOK_URL,
             "notebook_path": NOTEBOOK_PATH, **initial_summary,
             "affected_records_file": "initial_rejected_records.csv", "affected_record_rows": n_exported}
(RUN_DIR / "initial_rejection.json").write_text(json.dumps(rejection, indent=2) + "\n", encoding="utf-8")
print("2c. Initial rejection reference saved to", RUN_DIR / "initial_rejection.json")
print(json.dumps(rejection, indent=2))

# 2d. Assess the replacement with the same function and rules. Stop the run if it fails.
corrected_violations, corrected_summary = check_batch(CORRECTED_TABLE)
print("2d. Corrected batch: rule violations (expected: none)")
display(corrected_violations)
if corrected_summary["decision"] != "accepted":
    raise ValueError(f"returns_corrected failed the data contract: {corrected_summary['violations_by_rule']}")
ACCEPTED_TABLE = CORRECTED_TABLE  # Outcome 3 must assert this before writing features.
print(json.dumps(corrected_summary, indent=2))

# 2e. Reconcile the batches: rows that differ between them (exact row comparison, both directions).
initial_rows = spark.table(INITIAL_TABLE).select(*SOURCE_COLS)
corrected_rows = spark.table(CORRECTED_TABLE).select(*SOURCE_COLS)
reconciliation = (initial_rows.exceptAll(corrected_rows).withColumn("present_only_in", F.lit("returns_initial"))
    .unionByName(corrected_rows.exceptAll(initial_rows).withColumn("present_only_in", F.lit("returns_corrected")))
    .orderBy("return_id", "present_only_in"))
print("2e. Reconciliation: rows present in only one batch (all other rows are identical)")
display(reconciliation)

# 2f. Start this run's manifest; Outcomes 3 and 4 add outputs and snapshots to it.
run_manifest = {"run_id": RUN_ID, "run_started_utc": RUN_STARTED_UTC.isoformat(), "notebook_url": NOTEBOOK_URL,
                "notebook_path": NOTEBOOK_PATH,
                "batch_assessments": {"returns_initial": initial_summary, "returns_corrected": corrected_summary},
                "accepted_table": ACCEPTED_TABLE}
(RUN_DIR / "run.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
print("2f. Run manifest saved to", RUN_DIR / "run.json")

# COMMAND ----------

# DBTITLE 1,Outcome 2 decision and next action
# MAGIC %md
# MAGIC ### Decision and next action
# MAGIC
# MAGIC **`returns_initial` is rejected** (13 rows checked):
# MAGIC * **Uniqueness:** `R01` (2022, income 100, deduction 20) appears twice. Both copies are kept as affected records.
# MAGIC * **Completeness:** `R02` (2022, deduction 33) has a null `income`.
# MAGIC
# MAGIC No feature output is written from this batch. The affected records, rule and reason are saved in `outputs/runs/<run_id>/initial_rejected_records.csv`, with the rejection reference in `initial_rejection.json`.
# MAGIC
# MAGIC **Not done, because none of these is a fix:** dropping one `R01` copy, setting `R02` income to 0, copying values from the corrected batch, or relaxing or removing the checks. An unknown income can't be inferred.
# MAGIC
# MAGIC **Next action:** report the affected records to the data owner, then assess `returns_corrected` with the same `check_batch` rules. It passed with no violations (12 rows, 12 distinct `return_id`s), so it is the `ACCEPTED_TABLE` for Outcome 3. The reconciliation (2e) shows the replacement changes only the reported defects: one `R01` copy removed and `R02` income supplied (200). All other rows are identical.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deliver a checked feature table
# MAGIC Implement the task's data contract and verify the saved result by reading it back.
# MAGIC
# MAGIC **Contract for `return_features`** (saved to the individual output schema):
# MAGIC * Exactly three columns: `return_id` (string), `tax_year` (int), `deduction_to_income` (double).
# MAGIC * `deduction_to_income = deduction / income`, and **NULL when income = 0**. A zero-income return stays in the table. The rule is explicit because ANSI mode raises `DIVIDE_BY_ZERO` on a plain division.
# MAGIC * Every accepted return appears exactly once. `taxpayer_id`, `income` and `deduction` are excluded (need-to-know).
# MAGIC * Built only from `ACCEPTED_TABLE` (gate 3a). The table is overwritten on each run, so there are no appended copies.
# MAGIC
# MAGIC **Checks:**
# MAGIC * **Pre-write (3c):** row count, `return_id` uniqueness and the ID set against the accepted source.
# MAGIC * **Readback (3e)** of the saved table from the catalog:
# MAGIC   * exact schema
# MAGIC   * row count and uniqueness
# MAGIC   * the same ID set as the source
# MAGIC   * rows per year
# MAGIC   * a nonzero case: R01 = 20 / 100 = 0.2, hand-derived and recomputed from the source
# MAGIC   * the zero-income cases R03 and R09, which must have a null ratio
# MAGIC   * a full Python recompute of every ratio
# MAGIC
# MAGIC **Evidence (3f):** `outputs/runs/<run_id>/return_features.csv` (sorted readback snapshot), `outputs/return_features.csv` (stable final copy) and a `return_features` entry in `run.json` with the Delta version and the snapshot SHA-256.

# COMMAND ----------

# DBTITLE 1,Build, save and verify return_features
# Outcome 3: build, save and verify return_features from the accepted batch only.
import hashlib
import json
from pyspark.sql import functions as F

FEATURES_TABLE = f"{OUTPUT_SCHEMA}.return_features"
EXPECTED_SCHEMA = [("return_id", "string"), ("tax_year", "int"), ("deduction_to_income", "double")]
TOL = 1e-12

CHECK_RESULTS = []  # every check in this run; saved to outputs/runs/<run_id>/checks.csv in Outcome 4

def run_checks(title, checks):
    """Show each check with expected and actual values, record it for the run evidence, stop if any failed."""
    rows = [(title.split(".")[0], c, str(e), str(a), bool(p)) for c, e, a, p in checks]
    CHECK_RESULTS.extend(rows)
    print(title)
    display(spark.createDataFrame(rows, "stage string, check string, expected string, actual string, passed boolean"))
    failed = [r[1] for r in rows if not r[-1]]
    if failed:
        raise AssertionError(f"{title} failed: {failed}")

# 3a. Gate: features may only come from the batch that passed Outcome 2.
if not (ACCEPTED_TABLE == CORRECTED_TABLE and corrected_summary["decision"] == "accepted"):
    raise AssertionError("No accepted batch; run Outcome 2 first. Feature output not written.")
print(f"3a. Gate passed: building features from {ACCEPTED_TABLE} ({corrected_summary['decision']})")

# 3b. Build the features. Explicit zero-income rule: ANSI mode would raise DIVIDE_BY_ZERO on a plain division.
source = spark.table(ACCEPTED_TABLE)
features = source.select(
    "return_id",
    "tax_year",
    F.when(F.col("income") == 0, F.lit(None).cast("double"))
     .otherwise(F.col("deduction").cast("double") / F.col("income"))
     .alias("deduction_to_income"),
)  # taxpayer_id, income and deduction are not selected (need-to-know).

# 3c. Pre-write checks against the accepted source.
src_ids, feat_ids = source.select("return_id"), features.select("return_id")
src_n, feat_n = source.count(), features.count()
run_checks("3c. Pre-write checks", [
    ("row count equals accepted source", src_n, feat_n, feat_n == src_n),
    ("return_id unique", feat_n, feat_ids.distinct().count(), feat_ids.distinct().count() == feat_n),
    ("source IDs missing from features", 0, src_ids.exceptAll(feat_ids).count(), src_ids.exceptAll(feat_ids).count() == 0),
    ("feature IDs not in source", 0, feat_ids.exceptAll(src_ids).count(), feat_ids.exceptAll(src_ids).count() == 0),
])

# 3d. Save, replacing any earlier version (no appended copies on rerun).
(features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FEATURES_TABLE))
spark.sql(f"COMMENT ON TABLE {FEATURES_TABLE} IS 'One row per accepted return from {ACCEPTED_TABLE}. "
          "deduction_to_income = deduction / income; NULL when income = 0. No filer ID or raw amounts.'")
print(f"3d. Saved {FEATURES_TABLE}")

# 3e. Read the saved table back (from the catalog, not the in-memory DataFrame) and verify it.
saved = spark.table(FEATURES_TABLE)
print(f"3e. Readback of {FEATURES_TABLE} (all rows)")
display(saved.orderBy("return_id"))
saved_rows = {r["return_id"]: r for r in saved.collect()}
source_rows = {r["return_id"]: r for r in source.collect()}
actual_schema = [(f.name, f.dataType.simpleString()) for f in saved.schema]
by_year = {r["tax_year"]: r["count"] for r in saved.groupBy("tax_year").count().collect()}
src_by_year = {r["tax_year"]: r["count"] for r in source.groupBy("tax_year").count().collect()}

# Nonzero case: R01 hand-derived as 20 / 100 = 0.2, and recomputed in Python from its source row.
r01_src, r01_saved = source_rows["R01"], saved_rows.get("R01")
r01_expected = r01_src["deduction"] / r01_src["income"]
r01_actual = r01_saved["deduction_to_income"] if r01_saved else None

# Zero-income cases: every source return with income 0 must be present with a null ratio.
zero_ids = sorted(k for k, r in source_rows.items() if r["income"] == 0)
zero_ok = len(zero_ids) > 0 and all(k in saved_rows and saved_rows[k]["deduction_to_income"] is None for k in zero_ids)

# Full recompute in Python, independent of the Spark expression used to build the table.
def expected_ratio(r):
    return None if r["income"] == 0 else r["deduction"] / r["income"]
mismatches = sorted(
    k for k, r in source_rows.items()
    if k not in saved_rows
    or (expected_ratio(r) is None) != (saved_rows[k]["deduction_to_income"] is None)
    or (expected_ratio(r) is not None and abs(saved_rows[k]["deduction_to_income"] - expected_ratio(r)) > TOL)
    or saved_rows[k]["tax_year"] != r["tax_year"]
)

run_checks("3e. Readback checks on the saved table", [
    ("schema is exactly the three contract columns", EXPECTED_SCHEMA, actual_schema, actual_schema == EXPECTED_SCHEMA),
    ("row count equals accepted source", src_n, saved.count(), saved.count() == src_n),
    ("return_id unique", saved.count(), len(saved_rows), len(saved_rows) == saved.count()),
    ("same return_id set as source", sorted(source_rows), sorted(saved_rows), set(saved_rows) == set(source_rows)),
    ("no null return_id or tax_year", 0, saved.filter(F.col("return_id").isNull() | F.col("tax_year").isNull()).count(),
     saved.filter(F.col("return_id").isNull() | F.col("tax_year").isNull()).count() == 0),
    ("rows per tax_year match source", dict(sorted(src_by_year.items())), dict(sorted(by_year.items())), by_year == src_by_year),
    ("R01 expected value is hand-derived 20/100", 0.2, r01_expected, abs(r01_expected - 0.2) <= TOL),
    ("R01 saved ratio equals expected", r01_expected, r01_actual, r01_actual is not None and abs(r01_actual - r01_expected) <= TOL),
    ("zero-income returns present with null ratio", f"{zero_ids}: null",
     {k: (saved_rows[k]["deduction_to_income"] if k in saved_rows else "missing") for k in zero_ids}, zero_ok),
    ("all ratios match Python recompute from source", [], mismatches, not mismatches),
])

# 3f. Evidence: sorted readback snapshot for this run plus the stable final copy, and the manifest entry.
snapshot_csv = saved.orderBy("return_id").toPandas().to_csv(index=False)
for path in [RUN_DIR / "return_features.csv", repo_root / "outputs" / "return_features.csv"]:
    path.write_text(snapshot_csv, encoding="utf-8")
run_manifest["return_features"] = {
    "table": FEATURES_TABLE,
    "delta_version": table_version(FEATURES_TABLE),
    "rows": len(saved_rows),
    "ratio_definition": "deduction / income; NULL when income = 0",
    "nonzero_check": {"return_id": "R01", "expected": r01_expected, "saved": r01_actual},
    "zero_income_null_ratio": zero_ids,
    "snapshot_file": "return_features.csv",
    "snapshot_sha256": hashlib.sha256(snapshot_csv.encode("utf-8")).hexdigest(),
}
(RUN_DIR / "run.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
print("3f. Snapshot and manifest saved:", RUN_DIR / "return_features.csv", RUN_DIR / "run.json")
print(json.dumps(run_manifest["return_features"], indent=2))

# COMMAND ----------

# DBTITLE 1,Outcome 3 results
# MAGIC %md
# MAGIC ### Feature table results
# MAGIC
# MAGIC * `return_features` holds all **12 accepted returns** (4 per year for 2022, 2023 and 2024). There is one row per `return_id`, and the columns are exactly `return_id`, `tax_year` and `deduction_to_income`.
# MAGIC * **Nonzero check:** R01 has income 100 and deduction 20, so the expected ratio is 20 / 100 = **0.2**. The saved value read back from the table is 0.2 (tolerance 1e-12). Another example: R02 = 33 / 200 = 0.165.
# MAGIC * **Zero-income check:** R03 (2022) and R09 (2024) have income 0. Both are present in the saved table with a **null** ratio.
# MAGIC * **Full recompute:** every saved ratio matches a Python recompute from `returns_corrected`, with 0 mismatches.
# MAGIC * **Not verified here:** whether the ratio is a useful or leakage-free model feature. That is left to later modelling work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Make the handoff usable
# MAGIC Save the required year tables. Keep references and IDs-and-values evidence for two successful runs.
# MAGIC Write SUBMISSION.md and the small exports inside this Databricks Git folder.
# MAGIC
# MAGIC **Year tables** (built from the saved `return_features`, overwritten on each run):
# MAGIC
# MAGIC | Table | tax_year |
# MAGIC | --- | --- |
# MAGIC | `features_train` | 2022 |
# MAGIC | `features_validation` | 2023 |
# MAGIC | `features_test` | 2024 |
# MAGIC
# MAGIC **Checks (4b), read back from the catalog:**
# MAGIC * **Each table:** exact schema, no rows from another year, the same row count as `return_features` for that year, one row per `return_id`, and a latest Delta write that replaced the table rather than appending.
# MAGIC * **Across tables:** the total equals the 12 accepted returns, no `return_id` is in two tables, and the union equals `return_features` on IDs and values in both directions.
# MAGIC
# MAGIC **Run evidence:**
# MAGIC * **4c:** a snapshot CSV of each year table in `outputs/runs/<run_id>/`, plus stable copies in `outputs/`.
# MAGIC * **4d:** `checks.csv`, holding every 3c, 3e and 4b check. `run.json` is marked `status: succeeded`, with a completion time, only at the very end. A run that stops earlier stays partial.
# MAGIC * **4e:** `outputs/run_comparison.json` compares the **two latest successful runs**, using the saved snapshot files rather than the live tables. For each of the four tables it records the byte-level SHA-256 and a row-by-row diff (IDs and values) in both directions. Partial runs are listed as skipped, with the reason. A mismatch stops the run.

# COMMAND ----------

# DBTITLE 1,Save year tables, record run evidence, compare two runs
# Outcome 4: year tables from the saved features, this run's evidence, and the two-run comparison.
import csv
import datetime
import hashlib
import io
import json
from collections import Counter
import pandas as pd
from pyspark.sql import functions as F

SPLITS = {"features_train": 2022, "features_validation": 2023, "features_test": 2024}
SPLIT_TABLES = {name: f"{OUTPUT_SCHEMA}.{name}" for name in SPLITS}
REPLACE_OPS = {"CREATE OR REPLACE TABLE AS SELECT", "CREATE TABLE AS SELECT"}
RUNS_ROOT = repo_root / "outputs" / "runs"
SNAPSHOT_FILES = ["return_features.csv"] + [f"{name}.csv" for name in SPLITS]

def save_table(df, table, comment):
    """Replace the table (never append) and describe it in the Catalog."""
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    spark.sql(f"COMMENT ON TABLE {table} IS '{comment}'")

def last_write(table):
    """Latest data-writing operation in the table's Delta history (comment-only changes skipped)."""
    for r in spark.sql(f"DESCRIBE HISTORY {table}").orderBy(F.col("version").desc()).collect():
        if r["operation"] != "SET TBLPROPERTIES":
            return {"version": r["version"], "operation": r["operation"]}
    return None

# 4a. Build each year table from the saved, verified return_features (not from the raw source).
features_saved = spark.table(FEATURES_TABLE)
for name, year in SPLITS.items():
    save_table(features_saved.filter(F.col("tax_year") == year), SPLIT_TABLES[name],
               f"{name}: return_features rows with tax_year = {year}. Overwritten on each run.")
print("4a. Saved year tables:", list(SPLIT_TABLES.values()))

# 4b. Read the year tables back and check year assignment, one copy per return and no appended copies.
saved_splits = {name: spark.table(table) for name, table in SPLIT_TABLES.items()}
all_splits = None
for df in saved_splits.values():
    all_splits = df if all_splits is None else all_splits.unionByName(df)
split_checks = []
for name, year in SPLITS.items():
    df = saved_splits[name]
    schema = [(f.name, f.dataType.simpleString()) for f in df.schema]
    n, n_ids = df.count(), df.select("return_id").distinct().count()
    other_year = df.filter(F.col("tax_year").isNull() | (F.col("tax_year") != year)).count()
    expected_n = features_saved.filter(F.col("tax_year") == year).count()
    lw = last_write(SPLIT_TABLES[name])
    split_checks += [
        (f"{name}: schema is the three contract columns", EXPECTED_SCHEMA, schema, schema == EXPECTED_SCHEMA),
        (f"{name}: rows with a tax_year other than {year}", 0, other_year, other_year == 0),
        (f"{name}: rows equal return_features rows for {year}", expected_n, n, n == expected_n),
        (f"{name}: one row per return_id (no appended copies)", n, n_ids, n_ids == n),
        (f"{name}: latest write replaced the table", sorted(REPLACE_OPS), lw, lw is not None and lw["operation"] in REPLACE_OPS),
    ]
union_n, union_ids = all_splits.count(), all_splits.select("return_id").distinct().count()
missing = features_saved.exceptAll(all_splits).count()
extra = all_splits.exceptAll(features_saved).count()
lw_features = last_write(FEATURES_TABLE)
split_checks += [
    ("all splits: total rows equal accepted returns", len(source_rows), union_n, union_n == len(source_rows)),
    ("all splits: no return_id in more than one table", union_n, union_ids, union_ids == union_n),
    ("return_features rows missing from the splits (IDs and values)", 0, missing, missing == 0),
    ("split rows not in return_features (IDs and values)", 0, extra, extra == 0),
    ("return_features: latest write replaced the table", sorted(REPLACE_OPS), lw_features,
     lw_features is not None and lw_features["operation"] in REPLACE_OPS),
]
run_checks("4b. Year-table readback checks", split_checks)

# 4c. Snapshots: sorted readback of each year table for this run, plus the stable final copies.
for name, df in saved_splits.items():
    text = df.orderBy("return_id").toPandas().to_csv(index=False)
    for path in [RUN_DIR / f"{name}.csv", repo_root / "outputs" / f"{name}.csv"]:
        path.write_text(text, encoding="utf-8")

# 4d. Save every check of this run and mark the run complete. Only now does it count as successful.
pd.DataFrame(CHECK_RESULTS, columns=["stage", "check", "expected", "actual", "passed"]).to_csv(
    RUN_DIR / "checks.csv", index=False)
run_manifest["year_tables"] = {
    name: {"table": SPLIT_TABLES[name], "tax_year": year, "rows": saved_splits[name].count(),
           "delta_version": last_write(SPLIT_TABLES[name])["version"]}
    for name, year in SPLITS.items()}
run_manifest["snapshots_sha256"] = {f: hashlib.sha256((RUN_DIR / f).read_bytes()).hexdigest() for f in SNAPSHOT_FILES}
run_manifest["checks"] = {"file": "checks.csv", "total": len(CHECK_RESULTS),
                          "passed": sum(1 for r in CHECK_RESULTS if r[-1])}
run_manifest["run_completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
run_manifest["status"] = "succeeded"
(RUN_DIR / "run.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
print(f"4d. Run {RUN_ID} marked succeeded: {RUN_DIR / 'run.json'}")

# 4e. Compare the two latest successful runs from their saved snapshot files (reproducible from outputs/).
def read_snapshot(path):
    data = path.read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    return data, rows[0], rows[1:]

succeeded, skipped = [], []
for run_dir in sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir()):
    manifest_path = run_dir / "run.json"
    info = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if info.get("status") == "succeeded":
        succeeded.append((run_dir, info))
    else:
        skipped.append({"run_id": run_dir.name, "run_started_utc": info.get("run_started_utc"),
                        "reason": "partial run: run.json has no status 'succeeded' (stopped before Outcome 4 "
                                  "completed); not one of the two successful runs"})

comparison = {"generated_by_run": RUN_ID, "method": "byte-level SHA-256 and row-by-row multiset diff of the sorted "
              "snapshot CSVs saved in each run folder", "skipped_runs": skipped}
if len(succeeded) < 2:
    comparison.update(compared=False, note="Fewer than two successful runs; run the whole notebook again.")
else:
    (dir_a, run_a), (dir_b, run_b) = succeeded[-2:]
    tables = {}
    for f in SNAPSHOT_FILES:
        bytes_a, head_a, rows_a = read_snapshot(dir_a / f)
        bytes_b, head_b, rows_b = read_snapshot(dir_b / f)
        only_a = sorted((Counter(map(tuple, rows_a)) - Counter(map(tuple, rows_b))).elements())
        only_b = sorted((Counter(map(tuple, rows_b)) - Counter(map(tuple, rows_a))).elements())
        tables[f] = {"sha256_run_a": hashlib.sha256(bytes_a).hexdigest(), "sha256_run_b": hashlib.sha256(bytes_b).hexdigest(),
                     "rows_run_a": len(rows_a), "rows_run_b": len(rows_b), "header_match": head_a == head_b,
                     "return_ids": sorted(r[0] for r in rows_b),
                     "rows_only_in_run_a": [list(r) for r in only_a], "rows_only_in_run_b": [list(r) for r in only_b]}
        tables[f]["match"] = (tables[f]["sha256_run_a"] == tables[f]["sha256_run_b"] and head_a == head_b
                              and not only_a and not only_b)
    comparison.update(
        compared=True,
        runs={label: {k: run.get(k) for k in ("run_id", "run_started_utc", "run_completed_utc", "notebook_url")}
              for label, run in (("run_a", run_a), ("run_b", run_b))},
        tables=tables, match=all(t["match"] for t in tables.values()))
(repo_root / "outputs" / "run_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

print("4e. Two-run comparison saved to", repo_root / "outputs" / "run_comparison.json")
if comparison["compared"]:
    print({label: r["run_id"] for label, r in comparison["runs"].items()}, "| skipped:", [s["run_id"] for s in skipped])
    display(spark.createDataFrame(
        [(f, t["rows_run_a"], t["rows_run_b"], t["sha256_run_a"][:12], t["sha256_run_b"][:12],
          len(t["rows_only_in_run_a"]) + len(t["rows_only_in_run_b"]), t["match"]) for f, t in comparison["tables"].items()],
        "snapshot string, rows_run_a long, rows_run_b long, sha256_run_a string, sha256_run_b string, differing_rows long, match boolean"))
    if not comparison["match"]:
        raise AssertionError("The two latest successful runs differ; see outputs/run_comparison.json")
else:
    print(comparison["note"], "| skipped:", [s["run_id"] for s in skipped])

# COMMAND ----------

# DBTITLE 1,Handoff summary
# MAGIC %md
# MAGIC ### Handoff
# MAGIC
# MAGIC **Inputs:**
# MAGIC * `training.to6_challenge3_inputs.returns_corrected`: the accepted batch.
# MAGIC * `training.to6_challenge3_inputs.returns_initial`: rejected; R01 is duplicated and R02 has a null income.
# MAGIC * Lineage: `training.to6_challenge3_inputs.source_lineage`, which traces both batches to `parsed_source`.
# MAGIC
# MAGIC **Outputs**, all in `training.to6_c3_48f0f0239a0e`:
# MAGIC * `return_features`: 12 rows.
# MAGIC * `features_train` (2022), `features_validation` (2023) and `features_test` (2024): 4 rows each.
# MAGIC
# MAGIC **Ratio:** `deduction_to_income = deduction / income`, and NULL when income = 0 (R03, R09). The filer ID and raw amounts are excluded.
# MAGIC
# MAGIC **Checks:**
# MAGIC * completeness, uniqueness and nonnegative rules on both batches
# MAGIC * readback of every saved table
# MAGIC * nonzero expected value: R01 = 0.2
# MAGIC * zero-income cases with a null ratio
# MAGIC * year assignment, one copy per return and no appended copies
# MAGIC * an IDs-and-values comparison of the two latest successful runs (`outputs/run_comparison.json`)
# MAGIC
# MAGIC **Not verified:**
# MAGIC * labels and the prediction-time availability of features (model leakage), which are left to later modelling work
# MAGIC * the starter GitHub test check, which executes `notebook.py` with a fake `spark` and is expected to fail
# MAGIC
# MAGIC The full handoff is in `SUBMISSION.md`.