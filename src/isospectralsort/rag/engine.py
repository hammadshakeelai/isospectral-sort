"""
Lightweight Semantic & Lexical RAG Engine for Research Papers.

Parses, chunks, indexes, and searches foundational papers on continuous dynamical,
Lie-algebraic, Hamiltonian, and optimal transport sorting systems.
"""

from dataclasses import dataclass, asdict
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class PaperChunk:
    chunk_id: str
    paper_title: str
    author: str
    section_title: str
    content: str
    equations: List[str]
    keywords: List[str]


class PaperRAG:
    """Retrieval-Augmented Generation / Search Engine over foundational research papers."""

    def __init__(self, papers_dir: Optional[str] = None):
        if papers_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            papers_dir = os.path.join(base_dir, "papers")
            if not os.path.exists(papers_dir):
                papers_dir = os.path.abspath("papers")
        else:
            papers_dir = os.path.abspath(papers_dir)
            if not os.path.isdir(papers_dir):
                raise ValueError(f"Invalid papers directory: {papers_dir}")
                
        self.papers_dir = papers_dir
        self.chunks: List[PaperChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[Dict[str, float]] = []
        
        self._load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust word tokenizer for scientific & mathematical text."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in clean.split() if len(t) > 2]
        return tokens

    def _extract_equations(self, text: str) -> List[str]:
        """Extract LaTeX display and inline equations."""
        display = re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)
        return [eq.strip() for eq in display if eq.strip()]

    def _chunk_markdown(self, filename: str, content: str) -> List[PaperChunk]:
        """Split a research paper markdown file into semantic chunks by headers."""
        lines = content.splitlines()
        paper_title = "Unknown Paper"
        author = "Unknown Author"
        
        for line in lines[:10]:
            if line.startswith("# "):
                paper_title = line[2:].strip()
            elif line.startswith("**Author:**") or line.startswith("**Authors:**"):
                author = line.split(":**", 1)[1].strip()

        sections = re.split(r"\n(?=## )", content)
        chunks = []
        
        for idx, sec in enumerate(sections):
            if not sec.strip():
                continue
            sec_lines = sec.strip().splitlines()
            sec_title = sec_lines[0].replace("#", "").strip() if sec_lines else f"Section {idx}"
            
            eqs = self._extract_equations(sec)
            tokens = self._tokenize(sec)
            
            chunk = PaperChunk(
                chunk_id=f"{os.path.basename(filename).replace('.md', '')}_chunk_{idx}",
                paper_title=paper_title,
                author=author,
                section_title=sec_title,
                content=sec.strip(),
                equations=eqs,
                keywords=list(set(tokens[:20]))
            )
            chunks.append(chunk)
            
        return chunks

    def _load_and_index(self):
        """Index all markdown papers in papers_dir safely."""
        if not os.path.exists(self.papers_dir):
            return

        for fname in sorted(os.listdir(self.papers_dir)):
            if fname.endswith(".md") and not fname.startswith("."):
                fpath = os.path.join(self.papers_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    chunks = self._chunk_markdown(fname, content)
                    self.chunks.extend(chunks)

        if not self.chunks:
            return

        doc_count = len(self.chunks)
        df: Dict[str, int] = {}
        
        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk.content))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        for t, count in df.items():
            self.idf[t] = math.log((1 + doc_count) / (1 + count)) + 1.0

        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
            
            total = len(tokens) if tokens else 1
            tfidf = {t: (cnt / total) * self.idf.get(t, 1.0) for t, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in tfidf.values())) or 1.0
            self.tfidf_matrix.append({t: v / norm for t, v in tfidf.items()})

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[PaperChunk, float]]:
        """
        Retrieve the most relevant paper chunks for a given natural language query.
        """
        if not self.chunks or not query_text:
            return []

        safe_query = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(query_text)).strip()
        if not safe_query:
            return []

        q_tokens = self._tokenize(safe_query)
        if not q_tokens:
            return []

        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0.0) + 1.0
            
        q_tfidf = {t: (cnt / len(q_tokens)) * self.idf.get(t, 1.0) for t, cnt in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_tfidf.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_tfidf.items()}

        scores = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            sim = sum(q_vec.get(t, 0.0) * val for t, val in doc_vec.items())
            scores.append((self.chunks[idx], float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def export_index(self, output_path: str):
        """Export indexed chunks to JSON for persistent storage."""
        data = [asdict(c) for c in self.chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
