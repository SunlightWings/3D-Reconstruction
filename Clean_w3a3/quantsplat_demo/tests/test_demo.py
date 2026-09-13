"""Run: python -m unittest discover -s tests -v (no GPU or research repo)."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demo_data import DemoData
import ui
spec=importlib.util.spec_from_file_location('pack_results',ROOT/'tools/pack_results.py')
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)


class PackagedDataTests(unittest.TestCase):
    def setUp(self):self.d=DemoData(ROOT/'data')
    def test_every_packaged_image_exists_and_views_match(self):
        for s in self.d.scenes:
            self.assertEqual(len(s['frames']),9)
            self.assertEqual(len(set(s['frames'])),9)
            for arm in ['gt']+list(s['renders']):
                for i in range(9):self.assertTrue(Path(self.d.image(s['id'],arm,i)).is_file())
    def test_absent_arm_does_not_substitute_another(self):
        for s in self.d.scenes:
            if 'w3a3_conf' not in s['renders']:
                self.assertIsNone(self.d.image(s['id'],'w3a3_conf',0))
    def test_all_results_selections(self):
        for r in ['foreground','content']:
            for c in ['Raw','Exposure-corrected']:
                self.assertIsInstance(ui.summary_table(self.d,r,c),str)
                self.assertIsInstance(ui.pairs_table(self.d,r,c),str)
                for m in ['psnr','ssim','lpips']:self.assertIsInstance(ui.delta_plot(self.d,r,m,c),str)
    def test_scene_metrics_not_per_view(self):
        for s in self.d.scenes:
            self.assertIn('all nine',ui.scene_scores(self.d,s['id'],['full','w3a3']))
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):self.d.path('../app.py')


class PairingTests(unittest.TestCase):
    def test_camera_names_and_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);sparse=root/'sources/w3a3/heldout/sparse/0';sparse.mkdir(parents=True)
            names=[f'frame{i:06d}.png' for i in [2,24,52,75,102,125,153,175,202]]
            text='# Image records\n'+''.join(f'{i+1} 1 0 0 0 0 0 0 {i+1} {name}\n\n' for i,name in enumerate(names))
            (sparse/'images.txt').write_text(text)
            self.assertEqual(pack.read_text_names(sparse/'images.txt'),names)
            self.assertIn('checked',pack.validate_local_order(root,'w3a3',names))
            with self.assertRaises(ValueError):pack.validate_local_order(root,'w3a3',['frame999999.png']*9)
    def test_missing_camera_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):pack.validate_local_order(Path(t),'w3a3',[])
            self.assertIn('UNVERIFIED',pack.validate_local_order(Path(t),'w3a3',[],trust=True))
    def test_different_metric_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'result.json';p.write_text(json.dumps({'per_scene':[{'scene':'wrong/scene'}]}))
            with self.assertRaises(ValueError):pack.copy_metrics(p,root,root,['apple/110_13051_23361'])


if __name__=='__main__':unittest.main()
