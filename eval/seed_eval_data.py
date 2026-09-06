"""
seed_eval_data.py — Build the evaluation corpora
=================================================
Creates two collections used only by the eval harness, via the project's
EXISTING ingestion paths (rag/ingest.py and rag/doc_ingest.py) — nothing here
reimplements ingestion:

  * eval-movies     — 18 structured movie records (metadata filtering)
  * eval-documents  — a generated PDF, chunked (document mode)

The demo collection (`collection_name` in config.yaml) is left untouched.

A richer corpus than the 5-document demo set matters: with only 5 documents and
a default k of 4, precision@k is degenerate — almost everything is retrieved no
matter what the filter does.

Usage:
    python -m eval.seed_eval_data              # both corpora
    python -m eval.seed_eval_data --skip-docs  # structured only
"""

import argparse
import io
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from langchain.chains.query_constructor.base import AttributeInfo

from rag.doc_ingest import ingest_documents
from rag.ingest import ingest_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_MOVIES_COLLECTION = "eval-movies"
EVAL_DOCS_COLLECTION = "eval-documents"

# ──────────────────────────────────────────────────────────────────────────────
# Structured corpus — 18 movies with deliberately varied genre/date/rating
# ──────────────────────────────────────────────────────────────────────────────
MOVIES = [
    # title,                     summary,                                                          genre,                      release_date, rating, director
    ("Paprika", "A dream detective enters people's dreams to catch a psychic terrorist.", "anime,thriller,scifi", "2006-11-25", 8.6, "Satoshi Kon"),
    ("Perfect Blue", "A retired pop idol is stalked and loses her grip on reality.", "anime,thriller", "1997-08-05", 8.0, "Satoshi Kon"),
    ("Akira", "A biker gang member gains psychic powers in a rebuilt Neo-Tokyo.", "anime,action,scifi", "1988-07-16", 8.3, "Katsuhiro Otomo"),
    ("My Neighbor Totoro", "Two sisters befriend a gentle forest spirit in rural Japan.", "anime,fantasy", "1988-04-16", 8.1, "Hayao Miyazaki"),
    ("Spirited Away", "A girl works in a bathhouse for spirits to free her parents.", "anime,fantasy", "2001-07-20", 8.6, "Hayao Miyazaki"),
    ("Grave of the Fireflies", "Two siblings struggle to survive in wartime Japan.", "anime,drama", "1988-04-16", 8.5, "Isao Takahata"),
    ("Inception", "A thief steals secrets from within dreams and plants an idea.", "action,thriller,scifi", "2010-07-16", 8.8, "Christopher Nolan"),
    ("The Prestige", "Two rival magicians destroy each other chasing the perfect trick.", "thriller,drama", "2006-10-20", 8.5, "Christopher Nolan"),
    ("Memento", "A man with no short-term memory hunts his wife's killer.", "thriller,mystery", "2000-10-11", 8.4, "Christopher Nolan"),
    ("Interstellar", "Explorers travel through a wormhole to find humanity a new home.", "scifi,drama", "2014-11-07", 8.6, "Christopher Nolan"),
    ("Little Women", "Four sisters come of age in the aftermath of the Civil War.", "romance,drama", "2019-12-25", 7.8, "Greta Gerwig"),
    ("Lady Bird", "A headstrong teenager clashes with her mother in Sacramento.", "comedy,drama", "2017-11-03", 7.4, "Greta Gerwig"),
    ("Toy Story", "Toys come alive and have a blast when people are not looking.", "animation,comedy,fantasy", "1995-11-22", 8.3, "John Lasseter"),
    ("Jurassic Park", "Scientists bring back dinosaurs and mayhem breaks loose.", "action,adventure,scifi", "1993-06-11", 8.2, "Steven Spielberg"),
    ("Groundhog Day", "A weatherman relives the same day until he changes himself.", "comedy,romance,fantasy", "1993-02-12", 8.0, "Harold Ramis"),
    ("The Grand Budapest Hotel", "A concierge and a lobby boy chase a stolen painting.", "comedy,adventure", "2014-03-28", 8.1, "Wes Anderson"),
    ("Parasite", "A poor family schemes their way into a wealthy household.", "thriller,drama", "2019-05-30", 8.5, "Bong Joon-ho"),
    ("Whiplash", "A young drummer is pushed to his limit by a brutal instructor.", "drama,music", "2014-10-10", 8.5, "Damien Chazelle"),
]

EVAL_METADATA_FIELDS = [
    AttributeInfo(
        name="genre",
        description="Keywords for filtering: ['anime', 'action', 'comedy', 'romance', "
                    "'thriller', 'scifi', 'drama', 'fantasy', 'adventure', 'mystery', "
                    "'animation', 'music']",
        type="[string]",
    ),
    AttributeInfo(
        name="release_date",
        description="The date the movie was released on, format YYYY-MM-DD",
        type="string",
    ),
    AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
    AttributeInfo(
        name="director",
        description="The director of the movie, e.g. 'Christopher Nolan', 'Satoshi Kon', "
                    "'Hayao Miyazaki', 'Greta Gerwig'",
        type="string",
    ),
]

EVAL_CONTENT_DESCRIPTION = "Brief summary of a movie"


def movies_dataframe() -> pd.DataFrame:
    """The eval corpus as a DataFrame the existing ingester understands.

    page_content leads with the title so the harness can identify a retrieved
    document by title without adding an id field that would otherwise leak into
    the filter-generation prompt.
    """
    return pd.DataFrame(
        [
            {
                "summary": f"{title}. {summary}",
                "genre": genre,
                "release_date": release_date,
                "rating": rating,
                "director": director,
            }
            for title, summary, genre, release_date, rating, director in MOVIES
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Document corpus — a small generated PDF with checkable facts
# ──────────────────────────────────────────────────────────────────────────────
PDF_PAGES = [
    (
        "Section 1: Company Overview",
        [
            "Northwind Analytics was founded in 2016 and is headquartered in Bristol.",
            "The company employs 412 people across four offices.",
            "Its primary product is a telemetry platform for industrial equipment.",
        ],
    ),
    (
        "Section 2: Financial Performance",
        [
            "Revenue for fiscal year 2024 was 58.3 million pounds.",
            "Revenue grew 14 percent year over year, driven by cloud subscriptions.",
            "Operating margin expanded to 22 percent in the same period.",
            "Research and development spending was 9.1 million pounds.",
        ],
    ),
    (
        "Section 3: Risks",
        [
            "The principal risk identified is customer concentration.",
            "The three largest customers account for 41 percent of total revenue.",
            "A secondary risk is exposure to semiconductor supply shortages.",
            "The company holds no material foreign exchange hedges.",
        ],
    ),
]


def build_eval_pdf() -> bytes:
    """Generate the eval PDF in memory (reportlab)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        raise RuntimeError(
            "reportlab is required to generate the eval PDF. "
            "Install it (pip install reportlab) or run with --skip-docs."
        )

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for heading, lines in PDF_PAGES:
        y = 720
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, heading)
        y -= 30
        c.setFont("Helvetica", 11)
        for line in lines:
            c.drawString(72, y, line)
            y -= 20
        c.showPage()
    c.save()
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Seed the evaluation corpora.")
    parser.add_argument("--skip-structured", action="store_true")
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--movies-collection", default=EVAL_MOVIES_COLLECTION)
    parser.add_argument("--docs-collection", default=EVAL_DOCS_COLLECTION)
    args = parser.parse_args()

    if not args.skip_structured:
        df = movies_dataframe()
        print(f"Ingesting {len(df)} movies into '{args.movies_collection}'…")
        count = ingest_dataset(
            df=df,
            metadata_field_info=EVAL_METADATA_FIELDS,
            content_column="summary",
            collection_name=args.movies_collection,
            progress_callback=lambda step, pct: print(f"  [{pct:.0%}] {step}"),
        )
        print(f"  -> {count} documents ingested.\n")

    if not args.skip_docs:
        print(f"Ingesting the eval PDF into '{args.docs_collection}'…")
        summary = ingest_documents(
            files=[("northwind_report.pdf", build_eval_pdf())],
            collection_name=args.docs_collection,
            replace=True,
            progress_callback=lambda step, pct: print(f"  [{pct:.0%}] {step}"),
        )
        print(f"  -> {summary.chunks_inserted} chunks from "
              f"{summary.pages_extracted} pages.")
        for w in summary.warnings:
            print("  warning:", w)
        for s in summary.skipped:
            print("  skipped:", s)

    print("\nDone. Atlas search indexes take ~30s to become queryable.")


if __name__ == "__main__":
    main()
