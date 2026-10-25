"""Core (non-GUI) tests for data preparation, evaluation and result persistence.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from physo_ui import data as D
from physo_ui import engine as E
from physo_ui.mathrender import latex_from_infix
from physo_ui.units import format_units


class DataTests(unittest.TestCase):
    def test_sanitize_names(self):
        mapping = D.sanitize_names(["Temperature (K)", "2nd value", "lambda", "x"])
        names = list(mapping.values())
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertNotIn(" ", name)
            self.assertFalse(name[0].isdigit())
            self.assertNotIn(name, D.RESERVED_NAMES)

    def test_prepare_and_scaling(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, 120)
        y = 3.1 * x - 0.7
        raw = D.RawDataset("test", pd.DataFrame({"x": x, "y": y}))
        prepared = D.prepare_data(raw, ["x"], "y", split_enabled=True, train_fraction=0.8, normalize=True)
        self.assertEqual(prepared.n_samples, 120)
        self.assertEqual(prepared.n_train + prepared.n_eval, 120)
        X_train, y_train, _, _ = prepared.model_arrays()
        self.assertEqual(X_train.shape, (1, prepared.n_train))
        self.assertAlmostEqual(float(np.mean(y_train)), 0.0, places=6)
        self.assertAlmostEqual(float(np.std(y_train)), 1.0, places=6)
        predicted = D.predict_expression("2.0*x + 1.0", ["x"], X_train)
        self.assertEqual(predicted.shape, (prepared.n_train,))
        restored = prepared.inverse_y(predicted)
        self.assertEqual(restored.shape, (prepared.n_train,))

    def test_reward_to_r2(self):
        self.assertAlmostEqual(D.reward_to_r2(1.0), 1.0)
        self.assertAlmostEqual(D.reward_to_r2(0.5), 0.0)

    def test_units_are_preserved(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0.5, 2.0, 80)
        y = 2 * np.pi * np.sqrt(x / 9.81)
        raw = D.RawDataset("units", pd.DataFrame({"L": x, "T": y}))
        prepared = D.prepare_data(
            raw,
            ["L"],
            "T",
            x_units=[[1, 0, 0, 0, 0, 0, 0]],
            y_units=[0, 0, 1, 0, 0, 0, 0],
            use_units=True,
        )
        self.assertTrue(prepared.has_units)
        self.assertEqual(prepared.summary_dict()["x_units"][0][0], 1.0)
        self.assertEqual(format_units(prepared.x_units[0]), "L")
        self.assertEqual(format_units(prepared.y_units), "T")
        self.assertEqual(format_units([1, 0, -1, 0, 0, 0, 0]), "L·T⁻¹")
        self.assertEqual(format_units([0] * 7), "无量纲")

    def test_complex_latex_rendering(self):
        latex = latex_from_infix("n2(x) + sqrt(x) + log(x) + sin(x)", ["x"])
        self.assertIn("x^{2}", latex)
        self.assertIn("\sqrt", latex)
        self.assertIn("\log", latex)
        self.assertIn("\sin", latex)

    def test_load_bundled(self):
        raw = D.load_bundled("damped_oscillator.csv")
        self.assertIn("t", raw.frame.columns)
        self.assertIn("amplitude", raw.frame.columns)
        self.assertGreater(len(raw.frame), 100)

    def test_all_bundled_datasets(self):
        library = D.dataset_library()
        self.assertEqual(len(library), 9)
        for filename, meta in library.items():
            raw = D.load_bundled(filename)
            self.assertEqual(raw.n_rows, 320, filename)
            self.assertEqual(raw.n_cols, len(meta["x_columns"]) + 1, filename)
            for col in [*meta["x_columns"], meta["y_column"]]:
                self.assertIn(col, raw.frame.columns, filename)
            subset = raw.frame[[*meta["x_columns"], meta["y_column"]]]
            self.assertEqual(int(subset.isna().sum().sum()), 0, filename)


class ExpressionTests(unittest.TestCase):
    def test_custom_ops(self):
        x = np.array([0.5, 1.0, 2.0])
        pred = D.predict_expression("n2(x) + sqrt(x) + log(x)", ["x"], x.reshape(1, -1))
        expected = x**2 + np.sqrt(x) + np.log(x)
        np.testing.assert_allclose(pred, expected)

    def test_single_argument_shapes(self):
        X = np.array([[1.0, 2.0, 3.0]])  # one variable, three samples
        pred = D.predict_expression("2*x + 1", ["x"], X)
        np.testing.assert_allclose(pred, [3.0, 5.0, 7.0])


class ResultPersistenceTests(unittest.TestCase):
    def _result(self) -> E.RunResult:
        rng = np.random.default_rng(1)
        x = rng.uniform(-1, 1, 60)
        y = 2.0 * x + 1.5
        raw = D.RawDataset("unit-test", pd.DataFrame({"x": x, "y": y}))
        prepared = D.prepare_data(raw, ["x"], "y", split_enabled=False, normalize=False)
        infix = "2.0*x + 1.5"
        candidates = [
            E.Candidate(
                expression=infix,
                display_expression=infix,
                pretty="2.0*x + 1.5",
                latex="2.0 x + 1.5",
                complexity=5,
                length=5,
                reward=1.0,
                r2_train=1.0,
                rmse_train=0.0,
                r2_eval=float("nan"),
                rmse_eval=float("nan"),
                source="best",
            )
        ]
        return E.RunResult(
            summary=prepared.summary_dict(),
            config=E.TrainConfig().to_dict(),
            candidates=candidates,
            active_index=0,
            history=[{"epoch": 0, "best_r2": 0.9, "mean_r2": 0.8}],
            created_at="2026-01-01T00:00:00",
            duration=1.0,
            epochs_done=1,
            data=prepared,
            X_train=prepared.X_raw[:, prepared.train_idx],
            y_train=prepared.y_raw[prepared.train_idx],
            X_eval=np.empty((1, 0)),
            y_eval=np.empty((0,)),
        )

    def test_save_load_predict(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            path = result.save_to_dir(Path(tmp) / "run")
            self.assertTrue((path / "result.json").exists())
            self.assertTrue((path / "predictions.csv").exists())
            loaded = E.RunResult.load(path)
            self.assertIsNotNone(loaded.data)
            self.assertEqual(len(loaded.candidates), 1)
            X = np.array([[0.0, 1.0, 2.0]])
            pred = loaded.predict_array(X)
            np.testing.assert_allclose(pred, [1.5, 3.5, 5.5])

    def test_infer_frame(self):
        result = self._result()
        frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.5, 3.5]})
        out = result.infer_frame(frame)
        self.assertIn("prediction", out.columns)
        self.assertIn("residual", out.columns)
        np.testing.assert_allclose(out["prediction"].to_numpy(), [1.5, 3.5], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
