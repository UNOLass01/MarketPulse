"""DAG integrity (Phase 4 plan: "a broken DAG silently disables scheduling;
this is a genuinely nasty failure mode").

Parses every file in ``airflow/dags/`` directly rather than going through
Airflow's ``DagBag`` — ``DagBag`` lives in the scheduler/dag-processor side
of ``apache-airflow-core``, but parsing a DAG file is exactly "exec the
file, find the DAG object(s) it defines", which needs nothing beyond the
lightweight ``apache-airflow-task-sdk`` surface (``from airflow.sdk import
DAG``) that DAG files themselves import. Keeping this test's own dependency
surface that small is deliberate, not a shortcut.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from airflow.sdk import DAG

pytestmark = pytest.mark.unit

DAGS_DIR = Path(__file__).resolve().parents[2] / "airflow" / "dags"


def _dag_files() -> list[Path]:
    return sorted(p for p in DAGS_DIR.glob("*.py") if not p.name.startswith("_"))


def _load_dag(path: Path) -> DAG:
    """Import ``path`` as a standalone module and return its one DAG object.

    Airflow's DAG file processor adds the dags folder to ``sys.path`` before
    parsing each file so sibling helper modules (``_dag_common``) import as
    plain top-level modules — replicated here so the DAG files under test
    don't need any test-only accommodation.
    """
    if str(DAGS_DIR) not in sys.path:
        sys.path.insert(0, str(DAGS_DIR))

    module_name = f"_dag_under_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    dags = [v for v in vars(module).values() if isinstance(v, DAG)]
    assert len(dags) == 1, f"{path.name} must define exactly one DAG object, found {len(dags)}"
    return dags[0]


DAG_FILES = _dag_files()

EXPECTED_TASK_COUNTS = {
    "dag_data_quality": 4,
    "dag_model_retraining": 3,
    "dag_feature_backfill": 1,
    "dag_data_archival": 1,
    "dag_partition_maintenance": 1,
}


def test_at_least_one_dag_file_exists() -> None:
    # A guard against the glob silently matching nothing (e.g. a path typo)
    # and every parametrized test below vacuously "passing" on zero cases.
    assert DAG_FILES
    assert {p.stem for p in DAG_FILES} == set(EXPECTED_TASK_COUNTS)


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.stem)
def test_dag_file_imports_without_error(path: Path) -> None:
    _load_dag(path)


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.stem)
def test_dag_catchup_is_false(path: Path) -> None:
    assert _load_dag(path).catchup is False


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.stem)
def test_dag_has_owner_tags_and_doc(path: Path) -> None:
    dag = _load_dag(path)
    assert dag.owner
    assert dag.tags
    assert dag.doc_md


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.stem)
def test_every_task_has_execution_timeout(path: Path) -> None:
    dag = _load_dag(path)
    for dag_task in dag.tasks:
        assert (
            dag_task.execution_timeout is not None
        ), f"{dag.dag_id}.{dag_task.task_id} has no execution_timeout"


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.stem)
def test_every_task_retries_at_least_once(path: Path) -> None:
    # CLAUDE.md rule #10 + phase-4 plan: every task retries -- a single
    # transient blip must not fail a whole run.
    dag = _load_dag(path)
    for dag_task in dag.tasks:
        assert (
            dag_task.retries and dag_task.retries >= 1
        ), f"{dag.dag_id}.{dag_task.task_id} has no retries configured"


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.stem)
def test_dag_has_no_cycles(path: Path) -> None:
    # DAG construction itself raises AirflowDagCycleException on a cycle
    # (topological sort happens at add_task/dependency-set time) -- every
    # import test above already proves this implicitly. This test exists so
    # "no cycles" has its own explicit, named assertion per the phase-4
    # plan's test list, and to catch a cycle even if some future task ever
    # bypasses eager validation.
    dag = _load_dag(path)
    assert dag.topological_sort()


def test_expected_task_counts_and_dag_ids() -> None:
    found = {_load_dag(path).dag_id: len(_load_dag(path).tasks) for path in DAG_FILES}
    assert found == EXPECTED_TASK_COUNTS


def test_dag_ids_match_their_filenames() -> None:
    # dag_{domain}_{action} naming convention (CLAUDE.md) -- a mismatched
    # dag_id is confusing in the UI and easy to introduce via copy-paste.
    for path in DAG_FILES:
        assert _load_dag(path).dag_id == path.stem


def test_dag_data_quality_branches_to_alert_on_failure() -> None:
    dag = _load_dag(DAGS_DIR / "dag_data_quality.py")
    branch_task = dag.task_dict["branch_on_failure"]
    assert set(branch_task.downstream_task_ids) == {"checks_passed", "alert_on_failure"}


def test_dag_model_retraining_gates_training_on_quality() -> None:
    dag = _load_dag(DAGS_DIR / "dag_model_retraining.py")
    gate_task = dag.task_dict["quality_gate"]
    assert "train_and_evaluate" in gate_task.downstream_task_ids


def test_dag_feature_backfill_is_manual_trigger_only() -> None:
    assert _load_dag(DAGS_DIR / "dag_feature_backfill.py").schedule is None
