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

# COMMAND ----------

# Your implementation and verification here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Make the handoff usable
# MAGIC Save the required year tables. Keep references and IDs-and-values evidence for two successful runs.
# MAGIC Write SUBMISSION.md and the small exports inside this Databricks Git folder.

# COMMAND ----------

# Your implementation, rerun comparison and exports here.