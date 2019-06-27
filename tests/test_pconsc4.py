import os
import pconsc4

base_path = os.path.dirname(__file__)
small_alignment = os.path.join(base_path, 'data/small.a3m')
large_alignment = os.path.join(base_path, 'data/large.a3m')

model = pconsc4.get_pconsc4()


def test_small_all():
    results = pconsc4.predict_all(model, small_alignment)
    assert 'cmap' in results['contacts']
    assert 'features' in results
    assert 'eff_seq' in results


def test_small_contact():
    results = pconsc4.predict_contacts(model, small_alignment)
    assert 'cmap' in results
    assert 'features' in results
    assert 'eff_seq' in results


def test_large_all():
    results = pconsc4.predict_all(model, large_alignment)
    assert 'cmap' in results['contacts']
    assert 'features' in results
    assert 'eff_seq' in results


def test_large_contact():
    results = pconsc4.predict_contacts(model, large_alignment)
    assert 'cmap' in results
    assert 'features' in results
    assert 'eff_seq' in results


def test_reuse():
    results = pconsc4.predict_all(model, small_alignment)
    results_large = pconsc4.predict_all(model, large_alignment)

    assert 'cmap' in results['contacts']
    assert 'features' in results
    assert 'eff_seq' in results

    assert 'cmap' in results_large['contacts']
    assert 'features' in results_large
    assert 'eff_seq' in results_large


def test_format_cameo():
    from pconsc4.utils import format_contacts_cameo

    results = pconsc4.predict_contacts(model, small_alignment)
    f = open(small_alignment)
    f.readline()
    seq = f.readline().strip()

    text = format_contacts_cameo(results['cmap'], seq, min_sep=5)
    assert len(text.splitlines()) > 100


def test_format_casp():
    from pconsc4.utils import format_contacts_casp

    results = pconsc4.predict_contacts(model, small_alignment)
    f = open(small_alignment)
    f.readline()
    seq = f.readline().strip()

    text = format_contacts_casp(results['cmap'], seq, min_sep=5)
    assert len(text.splitlines()) > 100
