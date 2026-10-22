import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from infrastructure import runtime


class RuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.local = self.root / ".local"
        self.root_patch = patch.object(runtime, "ROOT", self.root)
        self.local_patch = patch.object(runtime, "LOCAL", self.local)
        self.root_patch.start()
        self.local_patch.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.local_patch.stop)

    def asset(self, content=b"verified bytes"):
        return runtime.Asset("example.bin", "https://example.invalid/example.bin", "sha256", hashlib.sha256(content).hexdigest(), len(content))

    def test_verified_cache_is_reused_without_network(self):
        directory = self.local / "downloads"
        directory.mkdir(parents=True)
        destination = directory / "example.bin"
        destination.write_bytes(b"verified bytes")
        with patch.object(runtime.urllib.request, "urlopen") as request:
            self.assertEqual(runtime.download_verified(self.asset(), directory, lambda _: None), destination)
        request.assert_not_called()

    def test_corrupt_download_is_never_published(self):
        directory = self.local / "downloads"
        with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(b"corruption")):
            with self.assertRaisesRegex(runtime.RuntimeSetupError, "verification failed"):
                runtime.download_verified(self.asset(), directory, lambda _: None)
        self.assertEqual(list(directory.iterdir()), [])

    def test_corrupt_cache_can_be_replaced_by_verified_download(self):
        directory = self.local / "downloads"
        directory.mkdir(parents=True)
        (directory / "example.bin").write_bytes(b"partial old file")
        with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(b"verified bytes")):
            destination = runtime.download_verified(self.asset(), directory, lambda _: None)
        self.assertEqual(destination.read_bytes(), b"verified bytes")
        self.assertEqual(len(list(directory.iterdir())), 1)

    def test_failed_redownload_preserves_existing_cache_for_diagnostics(self):
        directory = self.local / "downloads"
        directory.mkdir(parents=True)
        destination = directory / "example.bin"
        destination.write_bytes(b"old bad cache")
        with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(b"bad response")):
            with self.assertRaises(runtime.RuntimeSetupError):
                runtime.download_verified(self.asset(), directory, lambda _: None)
        self.assertEqual(destination.read_bytes(), b"old bad cache")
        self.assertEqual(len(list(directory.iterdir())), 1)

    def test_refuses_output_outside_local_directory(self):
        with self.assertRaisesRegex(runtime.RuntimeSetupError, "outside"):
            runtime.download_verified(self.asset(), self.root / "elsewhere", lambda _: None)

    def test_zip_traversal_and_windows_special_paths_are_rejected(self):
        for index, name in enumerate(["../escape.txt", "/absolute.txt", "C:/escape.txt", "folder\\escape.txt", "folder/NUL.txt", "folder/name. "]):
            with self.subTest(name=name):
                archive = self.root / f"bad-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    entry = zipfile.ZipInfo("placeholder")
                    entry.filename = name  # Preserve backslashes; ZipInfo's constructor normalizes them on Windows.
                    bundle.writestr(entry, b"bad")
                with self.assertRaisesRegex(runtime.RuntimeSetupError, "unsafe archive path"):
                    runtime.extract_archive(archive, self.local / f"extract-{index}")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_zip_symbolic_link_is_rejected(self):
        archive = self.root / "symlink.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            entry = zipfile.ZipInfo("root/link")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(entry, "../../outside")
        with self.assertRaisesRegex(runtime.RuntimeSetupError, "link or special file"):
            runtime.extract_archive(archive, self.local / "extract")

    def test_tar_hard_link_is_rejected(self):
        archive = self.root / "hardlink.tgz"
        with tarfile.open(archive, "w:gz") as bundle:
            entry = tarfile.TarInfo("package/link")
            entry.type = tarfile.LNKTYPE
            entry.linkname = "../../outside"
            bundle.addfile(entry)
        with self.assertRaisesRegex(runtime.RuntimeSetupError, "link or special file"):
            runtime.extract_archive(archive, self.local / "extract")

    def test_install_archive_is_idempotent_and_preserves_other_files(self):
        archive = self.root / "runtime.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("runtime-release/bin/tool.exe", b"fake executable")
            bundle.writestr("runtime-release/LICENSE", b"license")
        asset = runtime.Asset("runtime.zip", "https://example.invalid/runtime.zip", "sha256", runtime._hash_file(archive, "sha256"))
        destination = self.local / "runtime" / "example"
        with patch.object(runtime, "download_verified", return_value=archive) as download:
            runtime._install_archive(asset, destination, "bin/tool.exe", lambda _: None)
            runtime._install_archive(asset, destination, "bin/tool.exe", lambda _: None)
        download.assert_called_once()
        self.assertEqual((destination / "LICENSE").read_bytes(), b"license")
        self.assertEqual(json.loads((destination / ".runtime-manifest.json").read_text())["digest"], asset.digest)

    def test_existing_unmanaged_runtime_is_not_deleted(self):
        destination = self.local / "runtime" / "example"
        destination.mkdir(parents=True)
        precious = destination / "notes.txt"
        precious.write_text("keep me")
        with self.assertRaisesRegex(runtime.RuntimeSetupError, "will not erase"):
            runtime._install_archive(self.asset(), destination, "bin/tool.exe", lambda _: None)
        self.assertEqual(precious.read_text(), "keep me")

    def test_missing_runtime_diagnostics_are_explicit(self):
        self.assertEqual(runtime.runtime_paths(), {"node": None, "java": None, "pnpm": None, "server": None})
        for entry in runtime.describe_runtime().values():
            self.assertFalse(entry["exists"])
            self.assertEqual(entry["error"], "Missing. Run setup.")


if __name__ == "__main__":
    unittest.main()
