"""Cleanup must cover unregistered outputs while protecting source files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clean_outputs


class CleanupTests(unittest.TestCase):
    def test_all_preview_delete_and_repeat_preserve_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'main.py'
            source.write_text('print("keep")\n')
            for name, filename in [('direct_search/out_old', 'database.json'),
                                   ('reduction/clucr/fig5', 'plot_summary.json'),
                                   ('direct_search/__pycache__', 'main.pyc')]:
                directory = root / name
                directory.mkdir(parents=True)
                (directory / filename).write_text('{}')
            (root / 'VALIDATION.json').write_text('{}')
            registry = root / '.output_registry.json'
            registry.write_text('{}')
            cache = root / '.venv/lib/python3.11/site-packages/__pycache__'
            cache.mkdir(parents=True)
            (cache / 'dependency.pyc').touch()
            (root / '.venv/pyvenv.cfg').touch()
            with patch.object(clean_outputs, 'BASE', root), \
                 patch.object(clean_outputs, 'REGISTRY', registry), \
                 patch.object(clean_outputs, 'registry_lock', contextlib.nullcontext), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', ['clean_outputs.py', '--all']):
                    clean_outputs.main()
                self.assertTrue((root / 'reduction/clucr/fig5').exists())
                with patch('sys.argv', ['clean_outputs.py', '--all', '--yes']):
                    clean_outputs.main()
                    clean_outputs.main()
            self.assertEqual(clean_outputs.discover_outputs(root), set())
            self.assertEqual(source.read_text(), 'print("keep")\n')
            self.assertTrue((cache / 'dependency.pyc').is_file())
            self.assertEqual(json.loads(registry.read_text()), {})

    def test_source_code_inside_output_name_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'out_source'
            output.mkdir()
            (output / 'important.py').write_text('')
            found = clean_outputs.discover_outputs(root)
            with self.assertRaisesRegex(ValueError, 'Python source'):
                clean_outputs.validate_output(output, {}, found, root)
            with self.assertRaises(ValueError):
                clean_outputs.validate_output(root, {}, {root}, root)

    def test_symlink_to_external_dataset_is_not_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'
            root.mkdir()
            external = Path(tmp) / 'external'
            external.mkdir()
            (external / 'database.json').write_text('{}')
            link = root / 'out_link'
            link.symlink_to(external, target_is_directory=True)
            self.assertEqual(clean_outputs.discover_outputs(root), set())
            with self.assertRaisesRegex(ValueError, 'symlink'):
                clean_outputs.validate_output(link, {}, {link}, root)


if __name__ == '__main__':
    unittest.main()
