from heart_disease_pca.data_loader import load_data, make_binary_target


def test_load_data_has_expected_shape():
    df = load_data()
    assert df.shape[0] == 303
    assert "target" in df.columns


def test_make_binary_target():
    df = make_binary_target(load_data())
    assert set(df["target"].unique()).issubset({0, 1})
