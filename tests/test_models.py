import numpy as np

from src.models import APDropout, build_mlp


def scans(n=2000, d=520, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, d), dtype=np.float32)
    detected = rng.random((n, d)) < 0.05
    x[detected] = rng.uniform(0.1, 1.0, detected.sum())
    return x


def test_identity_at_inference():
    x = scans()
    out = np.asarray(APDropout(0.5, seed=1)(x, training=False))
    assert np.array_equal(out, x)


def test_training_only_zeroes_and_never_rescales():
    x = scans()
    out = np.asarray(APDropout(0.5, seed=1)(x, training=True))
    changed = out != x
    assert np.all(out[changed] == 0)          # failed APs become "not detected"
    assert np.all(x[changed] > 0)             # only detected APs can fail
    kept = (out != 0)
    assert np.array_equal(out[kept], x[kept])  # survivors keep their RSSI


def test_mean_drop_rate_matches_uniform_p():
    x = scans(n=4000)
    out = np.asarray(APDropout(0.6, seed=2)(x, training=True))
    det = x > 0
    frac = ((out == 0) & det).sum() / det.sum()
    assert abs(frac - 0.3) < 0.02  # E[p] = max_rate / 2


def test_model_roundtrip_keeps_layer(tmp_path):
    import keras
    m = build_mlp(hidden=(16,), ap_dropout=0.4, seed=0)
    path = tmp_path / "m.keras"
    m.save(path)
    m2 = keras.models.load_model(path)
    assert m2.get_layer("ap_dropout").max_rate == 0.4
    x = scans(n=8)
    assert np.allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-6)


def test_ae_and_cnn_build_and_train_step():
    from src.models import build_ae_classifier, build_autoencoder, build_cnn
    x = scans(n=64)
    y = np.arange(64) % 13
    ae, enc = build_autoencoder(encoder=(32, 16), ap_dropout=0.5, seed=0)
    ae.fit(x, x, epochs=1, verbose=0)
    clf = build_ae_classifier(enc, head_units=16, seed=0)
    # Classifier reuses the pretrained encoder weights and keeps the APDropout layer.
    assert clf.get_layer("enc_0") is enc.get_layer("enc_0")
    assert clf.get_layer("ap_dropout").max_rate == 0.5
    clf.fit(x, y, epochs=1, verbose=0)
    cnn = build_cnn(proj=32, filters=(4, 8), seed=0)
    cnn.fit(x, y, epochs=1, verbose=0)
    for m in (clf, cnn):
        p = m.predict(x, verbose=0)
        assert p.shape == (64, 13) and np.allclose(p.sum(1), 1, atol=1e-5)
