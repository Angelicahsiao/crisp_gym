"""Test the record config's `robot_type` and how it reaches info.json.

`robot_type` used to come only from a recording script's --robot-type flag,
which defaulted to "franka". Recording on the UR with the flag forgotten
produced a dataset labelled franka, and nothing downstream could tell — the
label is not derivable from the env, because crisp_py POPS robot_type when
building a RobotConfig (it only selects the config class, then discards the
string). So the data contract is where the label now lives.

Precedence is CLI > record config > the script's own fallback, and the field
is deliberately kept out of CONTRACT_FIELDS: a UR and a Franka recorded to the
same TCP-space contract must still compare as mixable (see
scripts/postprocess_align_datasets.py, which owns robot_type after a merge).

Run:  python -m pytest tests/test_record_config_robot_type.py
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_record_config():
    return SourceFileLoader(
        "rc_robot_type", str(REPO / "crisp_gym" / "record" / "record_config.py")
    ).load_module()


_MINIMAL = """
name: "t"
rate_hz: 15.0
observations:
  - key: observation.state.cartesian
    source: robot.tcp_pose
    representation: rotation_6d
action:
  definition: next_tcp_pose
  lookahead: 1
  representation: rotation_6d
  include_gripper: false
"""


def _write(tmp_path, extra: str = "") -> Path:
    p = tmp_path / "rc.yaml"
    p.write_text(_MINIMAL + extra)
    return p


# ── parsing ──────────────────────────────────────────────────────────────────

def test_robot_type_is_none_when_absent(tmp_path):
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path))
    assert cfg.robot_type is None


def test_robot_type_is_read_from_yaml(tmp_path):
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path, '\nrobot_type: "ur"\n'))
    assert cfg.robot_type == "ur"


# ── precedence ───────────────────────────────────────────────────────────────

def test_cli_wins_over_the_contract(tmp_path):
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path, '\nrobot_type: "ur"\n'))
    assert rc.resolve_robot_type("kinova", cfg, "franka") == "kinova"


def test_contract_wins_over_the_fallback(tmp_path):
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path, '\nrobot_type: "ur"\n'))
    assert rc.resolve_robot_type(None, cfg, "franka") == "ur"


def test_fallback_when_neither_is_set(tmp_path):
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path))
    assert rc.resolve_robot_type(None, cfg, "franka") == "franka"


def test_missing_record_config_still_resolves():
    """The leader/follower script's --record-config is optional."""
    rc = _load_record_config()
    assert rc.resolve_robot_type(None, None, "franka") == "franka"
    assert rc.resolve_robot_type("ur", None, "franka") == "ur"


def test_empty_strings_do_not_win(tmp_path):
    """An empty --robot-type or `robot_type: ""` must not blank the label."""
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path, '\nrobot_type: ""\n'))
    assert rc.resolve_robot_type("", cfg, "franka") == "franka"


# ── stamping ─────────────────────────────────────────────────────────────────

def test_metadata_omits_robot_type_when_unset(tmp_path):
    """A dataset recorded without it stays byte-identical to a pre-field one."""
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path))
    assert "robot_type" not in cfg.to_metadata()


def test_metadata_stamps_robot_type_when_set(tmp_path):
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(_write(tmp_path, '\nrobot_type: "ur"\n'))
    assert cfg.to_metadata()["robot_type"] == "ur"


# ── mixability must be unaffected ────────────────────────────────────────────

def test_robot_type_does_not_break_contract_compatibility(tmp_path):
    """A UR and a Franka on the same contract still train together."""
    rc = _load_record_config()
    ur = rc.RecordConfig.from_yaml(_write(tmp_path, '\nrobot_type: "ur"\n')).to_metadata()
    fr_path = tmp_path / "rc2.yaml"
    fr_path.write_text(_MINIMAL + '\nrobot_type: "franka"\n')
    fr = rc.RecordConfig.from_yaml(fr_path).to_metadata()

    assert ur["robot_type"] != fr["robot_type"]
    assert rc.RecordConfig.contracts_compatible(ur, fr)
    assert "robot_type" not in rc.RecordConfig.CONTRACT_FIELDS
