from __future__ import annotations

import pytest

from intervenebench.qualification import freeze_audit_batch


STRICT_CANDIDATES = """
326nv 345ms 3bzxg 3rvgz 3xy9j 4w9pz 4zscf 52ymc 5hqan 5mt6r 5vm8g
6wbd7 84vy3 9263n 9fehm 9n7hj 9tv2b a2nbf a42yg ac9jm bsd7j c956y
d3agv de5hx e2pyb egmxd ervm8 es4xw fct42 ftwqy fxcn4 gx6hp gzdnf
h84nt hgmu6 j38gd j6xgs jf46x jtgyq k9bwj kryns m52pd mkgvp mzm26
ncs7k nhgxf nj5dx nk9jd pb2rr r9v2d s43kb sffyb tcg8p ux8qt v6kqy
v6nhw vemrp vz5r4 w72cz waq4m xc4yq xfmrn xtvu5 xy8jw y9nb7 yg958
yh2ef yn7mx yp736 yuazs z358z zrwjp ztwqy zx5b8
""".split()

ALREADY_AUDITED = """
5vm8g 9263n bsd7j fxcn4 j6xgs jf46x ncs7k nhgxf nk9jd v6nhw vemrp
vz5r4 xc4yq xfmrn xy8jw y9nb7 yg958
""".split()


def test_frozen_viability_batch_is_deterministic_and_matches_lock() -> None:
    first = freeze_audit_batch(
        STRICT_CANDIDATES,
        excluded_ids=ALREADY_AUDITED,
        batch_size=40,
        seed_label="phase2-viability-v1:2102026",
    )
    second = freeze_audit_batch(
        reversed(STRICT_CANDIDATES),
        excluded_ids=reversed(ALREADY_AUDITED),
        batch_size=40,
        seed_label="phase2-viability-v1:2102026",
    )

    assert first == second
    assert len(first) == 40
    assert [entry.experiment_id for entry in first[:3]] == ["c956y", "mzm26", "d3agv"]
    assert [entry.experiment_id for entry in first[-3:]] == ["5hqan", "yn7mx", "s43kb"]
    assert first[0].selection_hash == (
        "029868c9e65043bbf3d89df5889bfd45fbd6e785176b417a7f00de1f74e2ef26"
    )
    assert not set(ALREADY_AUDITED) & {entry.experiment_id for entry in first}


def test_freeze_audit_batch_fails_closed_on_duplicates_or_short_pool() -> None:
    with pytest.raises(ValueError, match="unique"):
        freeze_audit_batch(["a", "a", "b"], batch_size=2, seed_label="seed")
    with pytest.raises(ValueError, match="only 1 candidates"):
        freeze_audit_batch(["a", "b"], excluded_ids=["a"], batch_size=2, seed_label="seed")


def test_freeze_audit_batch_rejects_blank_identifiers_and_seed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        freeze_audit_batch(["a", ""], batch_size=1, seed_label="seed")
    with pytest.raises(ValueError, match="seed_label"):
        freeze_audit_batch(["a"], batch_size=1, seed_label="")
