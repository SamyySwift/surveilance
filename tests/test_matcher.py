import numpy as np

import database
import matcher
from conftest import FakeEngine, blend, unit

MODEL = FakeEngine.model_id


def gallery_of(*people):
    ids, names, vecs = [], [], []
    for sid, name, vectors in people:
        for v in vectors:
            ids.append(sid)
            names.append(name)
            vecs.append(v)
    return matcher.Gallery.from_arrays(MODEL, ids, names, np.stack(vecs))


def test_identify_accepts_a_clear_match(fake_engine):
    alice, bob = unit(1), unit(2)
    g = gallery_of((1, "Alice", [alice]), (2, "Bob", [bob]))
    r = matcher.identify(blend(alice, unit(9), 0.2), g)
    assert r.accepted and r.staff_id == 1 and r.score > r.runner_up


def test_identify_refuses_when_two_people_are_too_close(fake_engine):
    alice = unit(1)
    twin = blend(alice, unit(3), 0.08)  # nearly identical enrolled person
    g = gallery_of((1, "Alice", [alice]), (2, "Twin", [twin]))
    r = matcher.identify(alice, g)
    assert r.score >= 0.5 and not r.accepted  # margin check


def test_identify_rejects_strangers(fake_engine):
    g = gallery_of((1, "Alice", [unit(1)]), (2, "Bob", [unit(2)]))
    r = matcher.identify(unit(50), g)
    assert not r.accepted and r.score < 0.3


def test_verify_checks_only_the_badge_holder(fake_engine):
    alice, bob = unit(1), unit(2)
    g = gallery_of((1, "Alice", [alice]), (2, "Bob", [bob]))
    assert matcher.verify(alice, 1, g)[1]
    assert not matcher.verify(alice, 2, g)[1]
    assert matcher.verify(alice, 99, g) == (-1.0, False)


def test_aggregate_embeddings_weights_frames():
    a, b = unit(1), unit(2)
    agg = matcher.aggregate_embeddings([a, b], [0.9, 0.1])
    assert np.isclose(np.linalg.norm(agg), 1.0)
    assert float(agg @ a) > float(agg @ b)


def test_gallery_cache_reloads_after_any_enrollment_change(fake_engine):
    sid = database.add_staff("Alice")
    database.add_embedding(sid, unit(1), model_id=MODEL)
    first = matcher.get_gallery(MODEL)
    assert matcher.get_gallery(MODEL) is first          # cached
    database.add_embedding(sid, unit(2), model_id=MODEL)
    assert len(matcher.get_gallery(MODEL)) == 2         # reloaded
    database.set_staff_active(sid, False)
    assert len(matcher.get_gallery(MODEL)) == 0         # suspended staff never match
    assert len(matcher.get_gallery(MODEL, include_inactive=True)) == 2


def test_models_are_never_mixed(fake_engine):
    sid = database.add_staff("Alice")
    database.add_embedding(sid, unit(1), model_id=MODEL)
    database.add_embedding(sid, unit(2), model_id="other:model")
    assert len(matcher.get_gallery(MODEL)) == 1
