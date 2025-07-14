
import sqlite3
from sqlite3 import Connection
import os
from time import strftime
from typing import Optional, List
import logging

DATABASE_FOLDER = "database"
PMID_PAPER_DB = "pmid_paper"
pmid_paper_create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {PMID_PAPER_DB} (
    pmid TEXT PRIMARY KEY,
    html_content TEXT DEFAULT NULL,
    title TEXT DEFAULT NULL,
    abstract TEXT DEFAULT NULL,
    is_preprint BOOLEAN DEFAULT FALSE,
    datetime TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    UNIQUE(pmid)
);
"""
pmid_paper_insert_html_sql = f"""
INSERT INTO {PMID_PAPER_DB} (pmid, html_content, datetime)
VALUES (?, ?, strftime('%Y-%m-%d %H:%M:%S', 'now'))
ON CONFLICT(pmid) DO UPDATE SET
    html_content = excluded.html_content,
    datetime=strftime('%Y-%m-%d %H:%M:%S', 'now');
"""
pmid_paper_insert_title_abstract_sql = f"""
INSERT INTO {PMID_PAPER_DB} (pmid, title, abstract, is_preprint, datetime)
VALUES (?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%S', 'now'))
ON CONFLICT(pmid) DO UPDATE SET
    title = excluded.title,
    abstract = excluded.abstract,
    is_preprint = excluded.is_preprint,
    datetime = strftime('%Y-%m-%d %H:%M:%S', 'now');
"""
pmid_paper_select_html_sql = f"""
SELECT html_content FROM {PMID_PAPER_DB} WHERE pmid = ?;
"""
pmid_paper_select_title_abstract_sql = f"""
SELECT title, abstract, is_preprint FROM {PMID_PAPER_DB} WHERE pmid = ?;
"""

class PMIDPaperDB:
    def __init__(self):
        self.db_path = os.path.join(os.getenv("DATA_FOLDER", "."), DATABASE_FOLDER)
        self.connection: Optional[Connection] = None
    
    def _ensure_tables(self):
        if self.connection is None:
            raise ValueError("Database connection is not initialized.")
        cursor = self.connection.cursor()
        cursor.execute(pmid_paper_create_table_sql)
        self.connection.commit()
    
    def _connect_db(self):
        if self.connection is not None:
            return
        db_path = self.db_path
        # Ensure the local path exists
        try:
            os.makedirs(db_path, exist_ok=True)
        except Exception as e:
            logging.error(e)
            return False
        db_path = os.path.join(db_path, PMID_PAPER_DB + ".db")
        if not os.path.exists(db_path):
            try:
                with open(db_path, "w"):
                    pass
            except Exception as e:
                logging.error(e)
                return False
        self.connection = sqlite3.connect(db_path)
        return True
    
    def insert_paper_html_content(self, pmid: str, html_content: str) -> bool:
        if not self._connect_db():
            return False
        self._ensure_tables()
        cursor = self.connection.cursor()
        try:
            cursor.execute(pmid_paper_insert_html_sql, (pmid, html_content))
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            logging.error(f"Error inserting paper with PMID {pmid}: {e}")
            return False
        finally:
            self.connection.close()
            self.connection = None
        
    def insert_paper_title_abstract(self, pmid: str, title: str, abstract: str, is_preprint: bool) -> bool:
        if not self._connect_db():
            return False
        self._ensure_tables()
        cursor = self.connection.cursor()
        try:
            cursor.execute(pmid_paper_insert_title_abstract_sql, (pmid, title, abstract, is_preprint))
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            logging.error(f"Error inserting paper with PMID {pmid}: {e}")
            return False
        finally:
            self.connection.close()
            self.connection = None
        
    def select_paper_html_content(self, pmid: str) -> Optional[str]:
        if not self._connect_db():
            return None
        self._ensure_tables()
        cursor = self.connection.cursor()
        try:
            cursor.execute(pmid_paper_select_html_sql, (pmid,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None
        except sqlite3.Error as e:
            logging.error(f"Error selecting paper with PMID {pmid}: {e}")
            return None
        finally:
            self.connection.close()
            self.connection = None

    def select_paper_title_abstract(self, pmid: str) -> tuple[Optional[str], Optional[str], bool]:
        
        if not self._connect_db():
            return None, None, False
        self._ensure_tables()
        cursor = self.connection.cursor()
        try:
            cursor.execute(pmid_paper_select_title_abstract_sql, (pmid,))
            row = cursor.fetchone()
            if row:
                return row[0], row[1], bool(row[2])
            return None, None, False
        except sqlite3.Error as e:
            logging.error(f"Error selecting paper with PMID {pmid}: {e}")
            return None, None, False
        finally:
            self.connection.close()
            self.connection = None