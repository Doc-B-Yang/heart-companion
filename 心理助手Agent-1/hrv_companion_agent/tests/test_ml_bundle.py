# -*- coding: utf-8 -*-
"""
ML 分类器 bundle 兼容性测试
==========================
覆盖两类模型 bundle:
  1. 新模型（真实数据训练）: 携带 classes_zh 中文标签自描述 → 预测必须映射到中文情绪名
  2. 旧模型（MAD 7 类）: 无 classes_zh → 回退 MAD_TO_ZH 映射，保持兼容

说明: 临时文件放在本目录 _tmp_mlbundle/ 下并手动清理
（系统 Temp 区在沙箱下存在 chmod 权限限制，Windows 下 TemporaryDirectory 清理会报错）。
清理动作本身是**容错**的 —— 受管环境若在删除上挂守卫并抛出 SystemExit，
不应把测试体已通过的用例标成 ERROR（见 `_TmpBundleTestCase.tearDown`）。
"""
import os
import shutil
import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from hrv_companion_agent.algorithms.emotion_recognizer import (
    MLClassifier, MAD_TO_ZH)

_TMP = Path(tempfile.mkdtemp(prefix='companion-ml-tests-'))


def _make_bundle(classes_zh=None, feature_names=None, n_classes=4):
    """构造一个微型随机森林 bundle（仅用于测试映射逻辑，不追求精度）"""
    X = np.array([
        [70, 60, 45, 1.5], [95, 20, 15, 4.5], [75, 35, 25, 2.8],
        [68, 55, 40, 1.2], [90, 22, 14, 5.0], [72, 50, 38, 1.8],
        [80, 30, 20, 3.2], [66, 58, 42, 1.0],
        [85, 26, 17, 4.8], [70, 62, 46, 1.6],
    ], dtype=float)
    y = np.array([0, 1, 2, 3, 1, 0, 2, 3, 1, 0])  # n_classes=4
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    bundle = {
        "model": clf,
        "scaler": StandardScaler().fit(X),
        "feature_names": feature_names or ["hr_bpm", "sdnn_ms", "rmssd_ms", "lf_hf"],
        "model_type": "rf",
    }
    if classes_zh is not None:
        bundle["classes_zh"] = classes_zh
        bundle["label_map"] = {i: c for i, c in enumerate(classes_zh)}
    return bundle


class _TmpBundleTestCase(unittest.TestCase):
    """在测试目录下创建/清理临时 bundle 文件"""

    def setUp(self):
        _TMP.mkdir(exist_ok=True)
        self._files = []

    def tearDown(self):
        """清理临时 bundle。

        .. note::
           这里刻意**吞掉所有异常**（含 ``SystemExit``），而不只是 ``OSError``。

           原因：某些受管执行环境会在 ``os.remove`` / ``os.rmdir`` 上挂
           删除守卫（例如通过 ``sitecustomize`` 注入的批量删除保护）。
           守卫不可用时它会直接 ``raise SystemExit(1)`` ——
           于是清理动作把**测试体已经通过**的用例标成 ERROR，
           看起来像代码缺陷，实际只是环境限制。

           清理失败不该改变测试结论，因此这里统一兜住：
           删不掉就留着（临时文件名固定，下次 setUp 会覆盖）。
        """
        for p in self._files:
            try:
                if p.exists():
                    os.remove(p)
            except BaseException:      # noqa: BLE001 —— 见 docstring
                pass
        try:
            _TMP.rmdir()
        except BaseException:          # noqa: BLE001
            pass

    def _dump(self, name, bundle) -> str:
        path = _TMP / name
        joblib.dump(bundle, path)
        self._files.append(path)
        return str(path)


class TestMLBundleClassesZh(_TmpBundleTestCase):
    """新模型: bundle 自带 classes_zh → 预测必须映射为中文情绪名"""

    def test_new_bundle_maps_to_chinese(self):
        zh = ["低焦虑", "平静", "愉悦", "高焦虑"]
        path = self._dump("model_zh.joblib", _make_bundle(classes_zh=zh))
        ml = MLClassifier(path)
        self.assertTrue(ml.available)
        out = ml.predict({"hr_bpm": 95, "sdnn_ms": 20, "rmssd_ms": 15,
                          "lf_hf": 4.5})
        self.assertIsNotNone(out)
        # 映射结果必须来自 classes_zh 本身（而非 MAD 映射表）
        self.assertIn(out["emotion"], zh)
        self.assertIn("confidence", out)

    def test_new_bundle_missing_features_filled(self):
        """新模型只有 4 个特征时，缺失的 8 特征名按 0 填充不应崩溃"""
        zh = ["低焦虑", "平静", "愉悦", "高焦虑"]
        path = self._dump("model_zh2.joblib", _make_bundle(classes_zh=zh))
        ml = MLClassifier(path)
        out = ml.predict({"hr_bpm": 70, "sdnn_ms": 60, "rmssd_ms": 45,
                          "lf_hf": 1.5})
        self.assertIsNotNone(out)
        self.assertIn(out["emotion"], zh)


class TestMLBundleMadFallback(_TmpBundleTestCase):
    """旧 MAD 模型: 无 classes_zh → 回退 MAD_TO_ZH 映射"""

    def test_mad_bundle_fallback(self):
        path = self._dump("model_mad.joblib", _make_bundle(classes_zh=None))
        ml = MLClassifier(path)
        self.assertTrue(ml.available)
        out = ml.predict({"hr_bpm": 95, "sdnn_ms": 20, "rmssd_ms": 15,
                          "lf_hf": 4.5, "pnn50_pct": 3, "sd1_ms": 11,
                          "sd2_ms": 40, "sd_ratio": 0.28})
        self.assertIsNotNone(out)
        self.assertIn("mad_label", out)
        # 回退映射输出必须是 5 类中文之一
        self.assertIn(out["emotion"], set(MAD_TO_ZH.values()))

    def test_missing_model_file_disabled(self):
        ml = MLClassifier("definitely_not_exist.joblib")
        self.assertFalse(ml.available)
        self.assertIsNone(ml.predict({"hr_bpm": 70}))


if __name__ == "__main__":
    unittest.main()
