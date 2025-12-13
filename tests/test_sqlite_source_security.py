import sqlite3
import tempfile
import unittest
from pathlib import Path

from sievio.sources.sqlite_source import SQLiteSource


class SQLiteSourceSecurityTest(unittest.TestCase):
    def test_reserved_keyword_column_is_quoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute('CREATE TABLE "items" ("id" INTEGER PRIMARY KEY, "order" TEXT, "body" TEXT)')
                conn.execute('INSERT INTO "items" ("order", "body") VALUES (?, ?)', ("first", "payload"))
                conn.commit()
            finally:
                conn.close()

            source = SQLiteSource(
                db_path=db_path, table="items", text_columns=("order", "body"), id_column="id"
            )

            items = list(source.iter_files())

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].path, "items:1")
        self.assertIn(b"first", items[0].data)
        self.assertIn(b"payload", items[0].data)
