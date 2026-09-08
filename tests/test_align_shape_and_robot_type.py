"""Tests for the two gaps that blocked a real UR + Franka merge.

`postprocess_align_datasets.py` computed its shared column set from parquet
column NAMES. That strips a column missing on one side (extra.ext_torque, which
only a Franka has) but keeps one whose name matches and whose SHAPE does not
(extra.joints: (6,) on a 6-DOF UR, (7,) on a 7-DOF Franka). LeRobot's
features_equal_for_merge compares whole feature dicts, so the merge still
failed after aligning. Separately, validate_all_metadata refuses two datasets
whose robot_type differs — before it reads any feature at all — and nothing
here could set it.

The fixture is the real case: a 6-DOF UR and a 7-DOF Franka recorded to the
same TCP-space contract, so every policy-facing feature already matches and
only the joint-space extras conflict.

Run:  python -m pytest tests/test_align_shape_and_robot_type.py
"""

import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

align_mod = SourceFileLoader(
    "postprocess_align_datasets",
    str(REPO / "crisp_gym" / "scripts" / "postprocess_align_datasets.py"),
).load_module()


# Shared by both arms — this is what makes the merge meaningful at all.
SHARED_FEATURES = {
    "action": ("float32", (10,)),
    "observation.state": ("float32", (10,)),
    "observation.state.cartesian": ("float32", (9,)),
    "observation.state.gripper": ("float32", (1,)),
    "extra.gripper_raw": ("float32", (1,)),
    "extra.target_cartesian": ("float32", (9,)),
}
BOOKKEEPING = {
    "timestamp": ("float32", (1,)),
    "frame_index": ("int64", (1,)),
    "episode_index": ("int64", (1,)),
    "index": ("int64", (1,)),
    "task_index": ("int64", (1,)),
}


def _make_dataset(root: Path, robot_type: str, dof: int, *, ext_torque: bool) -> Path:
    """Write a minimal but structurally real LeRobot dataset."""
    features = {
        **{k: {"dtype": d, "shape": list(s)} for k, (d, s) in SHARED_FEATURES.items()},
        **{k: {"dtype": d, "shape": list(s)} for k, (d, s) in BOOKKEEPING.items()},
        "observation.images.oakw_cam": {"dtype": "video", "shape": [800, 1280, 3]},
        # DOF-dependent: same names on both arms, different widths.
        "extra.joints": {"dtype": "float32", "shape": [dof]},
        "extra.joint_efforts": {"dtype": "float32", "shape": [dof]},
        "extra.joint_velocities": {"dtype": "float32", "shape": [dof]},
        "extra.target_joints": {"dtype": "float32", "shape": [dof]},
    }
    if ext_torque:
        features["extra.ext_torque"] = {"dtype": "float32", "shape": [dof]}

    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"robot_type": robot_type, "fps": 15, "features": features}, indent=4)
    )
    # Only include_in_state observations take part in contract comparison, so
    # the joint-space extras are declared as non-state on both sides.
    (root / "meta" / "record_config.json").write_text(
        json.dumps(
            {
                "record_config_name": "umi_robot_full",
                "rate_hz": 15.0,
                "action": {"definition": "next_tcp_pose", "lookahead": 1},
                "observations": [
                    {"key": "observation.state.cartesian", "source": "robot.tcp_pose",
                     "include_in_state": True},
                    {"key": "observation.state.gripper", "source": "gripper.value",
                     "include_in_state": True},
                    {"key": "extra.joints", "source": "robot.joints",
                     "include_in_state": False},
                ],
            },
            indent=4,
        )
    )

    columns = {k: [np.zeros(s, np.float32)] for k, (_, s) in SHARED_FEATURES.items()}
    columns.update({k: [0] for k in BOOKKEEPING})
    for key in ("extra.joints", "extra.joint_efforts", "extra.joint_velocities",
                "extra.target_joints"):
        columns[key] = [np.zeros(dof, np.float32)]
    if ext_torque:
        columns["extra.ext_torque"] = [np.zeros(dof, np.float32)]

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pd.DataFrame(columns).to_parquet(data_dir / "file-000.parquet", index=False)
    return root


@pytest.fixture
def ur_and_franka(tmp_path):
    """A 6-DOF UR and a 7-DOF Franka, as in the real datasets."""
    return [
        _make_dataset(tmp_path / "ur", "ur", 6, ext_torque=False),
        _make_dataset(tmp_path / "franka", "franka", 7, ext_torque=True),
    ]


def _aligned_info(root: Path, suffix: str = "_aligned") -> dict:
    return json.loads((root.parent / (root.name + suffix) / "meta" / "info.json").read_text())


# ── shape-aware stripping ────────────────────────────────────────────────────


def test_signatures_read_dtype_and_shape(ur_and_franka):
    ur, franka = ur_and_franka
    assert align_mod.feature_signatures(ur)["extra.joints"] == ("float32", (6,))
    assert align_mod.feature_signatures(franka)["extra.joints"] == ("float32", (7,))


def test_signatures_of_a_dataset_without_info_json_are_empty(tmp_path):
    """No info.json must mean 'conflicts with nothing', not a crash."""
    assert align_mod.feature_signatures(tmp_path) == {}


def test_same_name_different_shape_columns_are_dropped(ur_and_franka):
    """The bug: these survived a name-only intersection and blocked the merge."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)

    for root in (ur, franka):
        features = _aligned_info(root)["features"]
        for key in ("extra.joints", "extra.joint_efforts",
                    "extra.joint_velocities", "extra.target_joints"):
            assert key not in features, f"{key} still present in {root.name}"


def test_column_missing_on_one_side_is_still_dropped(ur_and_franka):
    """The pre-existing behaviour must not regress."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)
    assert "extra.ext_torque" not in _aligned_info(franka)["features"]


def test_matching_features_survive(ur_and_franka):
    """Everything the policy consumes is arm-agnostic and must be kept."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)
    for root in (ur, franka):
        features = _aligned_info(root)["features"]
        for key in SHARED_FEATURES:
            assert key in features, f"{key} was wrongly stripped from {root.name}"
        assert "observation.images.oakw_cam" in features


def test_aligned_datasets_have_identical_feature_dicts(ur_and_franka):
    """The whole point: what lerobot's features_equal_for_merge compares."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)
    assert _aligned_info(ur)["features"] == _aligned_info(franka)["features"]


def test_dropped_columns_leave_the_parquet_too(ur_and_franka):
    """Metadata and data must not disagree."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)
    frame = pd.read_parquet(ur.parent / "ur_aligned" / "data" / "chunk-000" / "file-000.parquet")
    assert "extra.joints" not in frame.columns
    assert "observation.state.cartesian" in frame.columns


def test_promoting_a_shape_conflicted_column_is_refused(ur_and_franka):
    """A policy input cannot be 6 wide in one dataset and 7 in another."""
    ur, franka = ur_and_franka
    with pytest.raises(SystemExit, match="different shapes"):
        align_mod.align([ur, franka], "_aligned", None, ["extra.joints"], dry_run=False)


# ── robot_type override ──────────────────────────────────────────────────────


def test_robot_type_is_left_alone_by_default(ur_and_franka):
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)
    assert _aligned_info(ur)["robot_type"] == "ur"
    assert _aligned_info(franka)["robot_type"] == "franka"


def test_robot_type_override_makes_both_sides_agree(ur_and_franka):
    """validate_all_metadata refuses differing robot_type before any feature."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False, robot_type="ur+franka")
    assert _aligned_info(ur)["robot_type"] == "ur+franka"
    assert _aligned_info(franka)["robot_type"] == "ur+franka"


def test_original_robot_type_is_kept_as_provenance(ur_and_franka):
    """info.json can no longer say which arm; record_config.json still can."""
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=False, robot_type="ur+franka")
    for root, expected in ((ur, "ur"), (franka, "franka")):
        contract = json.loads(
            (root.parent / (root.name + "_aligned") / "meta" / "record_config.json").read_text()
        )
        assert contract["source_robot_type"] == expected
        assert contract["robot_type"] == "ur+franka"


# ── labels must distinguish the datasets ────────────────────────────────────


def test_labels_widen_when_directory_names_collide():
    """Datasets laid out as <name>/lerobot all have d.name == 'lerobot'.

    Reporting them by bare name produced two identical log lines whose only
    difference was which columns they dropped.
    """
    dirs = [Path("/data/ur_electricbox/lerobot"), Path("/data/franka_electricbox/lerobot")]
    labels = align_mod.dataset_labels(dirs)
    assert labels[dirs[0]] == "ur_electricbox/lerobot"
    assert labels[dirs[1]] == "franka_electricbox/lerobot"


def test_labels_stay_short_when_names_are_already_distinct():
    dirs = [Path("/data/ur"), Path("/data/franka")]
    labels = align_mod.dataset_labels(dirs)
    assert set(labels.values()) == {"ur", "franka"}


def test_labels_are_unique_even_when_deeply_similar():
    dirs = [Path("/a/x/lerobot"), Path("/b/x/lerobot")]
    labels = align_mod.dataset_labels(dirs)
    assert len(set(labels.values())) == 2


def test_dry_run_names_each_dataset_distinctly(tmp_path, caplog):
    """The end-to-end symptom: two <name>/lerobot datasets in one run."""
    ur = _make_dataset(tmp_path / "ur_electricbox" / "lerobot", "ur", 6, ext_torque=False)
    franka = _make_dataset(
        tmp_path / "franka_electricbox" / "lerobot", "franka", 7, ext_torque=True
    )
    with caplog.at_level("INFO"):
        align_mod.align([ur, franka], "_aligned", None, [], dry_run=True,
                        robot_type="ur+franka")
    text = caplog.text
    assert "ur_electricbox/lerobot" in text
    assert "franka_electricbox/lerobot" in text
    # and the per-dataset robot_type must not be conflated
    assert "robot_type 'ur' -> 'ur+franka'" in text
    assert "robot_type 'franka' -> 'ur+franka'" in text


# ── datasets whose contract was stripped by a lerobot merge ─────────────────


def _strip_contract(root: Path) -> Path:
    """Reproduce what aggregate_datasets leaves behind.

    It writes only info.json, tasks.parquet, stats.json and the episodes
    parquet, so a merged dataset has no crisp_gym record_config.json even
    though every source had one.
    """
    (root / "meta" / "record_config.json").unlink()
    return root


def test_missing_contract_is_refused_by_default(ur_and_franka):
    """The refusal is the point: unverified action semantics corrupt training."""
    ur, franka = ur_and_franka
    _strip_contract(franka)
    with pytest.raises(FileNotFoundError, match="aggregate_datasets"):
        align_mod.align([ur, franka], "_aligned", None, [], dry_run=False)


def test_missing_contract_is_allowed_with_the_flag(ur_and_franka):
    ur, franka = ur_and_franka
    _strip_contract(franka)
    align_mod.align(
        [ur, franka], "_aligned", None, [], dry_run=False,
        robot_type="ur+franka", skip_contract_check=True,
    )
    assert _aligned_info(ur)["features"] == _aligned_info(franka)["features"]
    assert _aligned_info(franka)["robot_type"] == "ur+franka"


def test_contractless_output_simply_has_no_record_config(ur_and_franka):
    """Nothing to rewrite must not mean a crash or a fabricated contract."""
    ur, franka = ur_and_franka
    _strip_contract(franka)
    align_mod.align(
        [ur, franka], "_aligned", None, [], dry_run=False, skip_contract_check=True
    )
    assert not (franka.parent / "franka_aligned" / "meta" / "record_config.json").exists()
    # the dataset that HAD one still gets it rewritten
    assert (ur.parent / "ur_aligned" / "meta" / "record_config.json").exists()


def test_the_flag_does_not_abandon_checks_it_can_still_make(ur_and_franka, tmp_path):
    """Two datasets that DO carry contracts are still compared to each other."""
    ur, franka = ur_and_franka
    third = _make_dataset(tmp_path / "other", "ur", 6, ext_torque=False)
    contract = json.loads((third / "meta" / "record_config.json").read_text())
    contract["rate_hz"] = 30.0  # a real, unfixable contract difference
    (third / "meta" / "record_config.json").write_text(json.dumps(contract, indent=4))
    _strip_contract(franka)

    with pytest.raises(SystemExit, match="not mixable"):
        align_mod.align(
            [ur, franka, third], "_aligned", None, [], dry_run=False,
            skip_contract_check=True,
        )


def test_dry_run_writes_nothing(ur_and_franka):
    ur, franka = ur_and_franka
    align_mod.align([ur, franka], "_aligned", None, [], dry_run=True, robot_type="ur+franka")
    assert not (ur.parent / "ur_aligned").exists()
    assert _aligned_info.__name__  # sanity: originals untouched
    assert json.loads((ur / "meta" / "info.json").read_text())["robot_type"] == "ur"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
