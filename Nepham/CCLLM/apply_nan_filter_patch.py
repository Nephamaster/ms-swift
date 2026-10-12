"""Backport only EncodePreprocessor.preprocess into the active Swift installation."""
import ast
import inspect
import shutil
from pathlib import Path


def method_node(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EncodePreprocessor')
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'preprocess')


def patched_source(installed, checkout):
    replacement = method_node(checkout)
    # The older installation may not import the newer method's typing names.
    replacement.returns = None
    for argument in replacement.args.args:
        argument.annotation = None
    replacement_text = '\n'.join('    ' + line if line else '' for line in ast.unparse(replacement).splitlines()) + '\n'
    current = method_node(installed)
    lines = installed.splitlines(keepends=True)
    updated = ''.join(lines[:current.lineno - 1]) + replacement_text + ''.join(lines[current.end_lineno:])
    compile(updated, '<patched swift.dataset.utils>', 'exec')
    return updated


def main():
    from swift.dataset.utils import EncodePreprocessor

    checkout = Path(__file__).resolve().parents[2] / 'swift/dataset/utils.py'
    installed = Path(inspect.getsourcefile(EncodePreprocessor)).resolve()
    if installed == checkout.resolve():
        print(f'Already using the checkout: {installed}')
        return
    original = installed.read_text(encoding='utf-8')
    updated = patched_source(original, checkout.read_text(encoding='utf-8'))
    if updated == original:
        print(f'Patch already installed: {installed}')
        return
    backup = installed.with_name(installed.name + '.before-nan-filter')
    if not backup.exists():
        shutil.copy2(installed, backup)
    installed.write_text(updated, encoding='utf-8')
    print(f'Patched: {installed}\nBackup: {backup}')
    print('Restart Swift with --load_from_cache_file false. Re-export any prebuilt cached datasets.')


if __name__ == '__main__':
    main()
