#!/usr/bin/env python3
"""Seed script to populate the database with sample claims."""

import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import Claim, Verdict, create_claim
from src.graph import ClaimGraph
from src.storage import InMemoryClaimRepository
from src.preprocessing import EmbeddingEncoder


SAMPLE_CLAIMS = [
    ("Scientists discover new cure for common cold using vitamin C", "news", "trusted"),
    ("Government planning to ban all social media next month", "twitter", "false"),
    ("Breaking: Celebrity involved in major scandal", "facebook", "unverified"),
    ("Miracle supplement reverses aging in new study", "reddit", "misleading"),
    ("Bank accounts will be frozen next week - urgent alert", "twitter", "false"),
    ("Scientists confirm Earth is actually flat", "facebook", "false"),
    ("Major earthquake expected in next 48 hours", "twitter", "misleading"),
    ("Police warn of new scam targeting bank customers", "news", "trusted"),
    ("Leaked document reveals tech company scandal", "reddit", "unverified"),
    ("Emergency: tap water contaminated in your city", "twitter", "misleading"),
    ("Celebrity announces surprise retirement from public life", "news", "trusted"),
    ("New law passed - all homeowners must register", "facebook", "false"),
    ("Study shows coffee doubles cancer risk", "news", "misleading"),
    ("Breaking: World leader announces unexpected resignation", "twitter", "trusted"),
    ("Miracle food additive cures disease - doctors warn", "facebook", "misleading"),
]


async def seed_database(num_claims: int = 50):
    """Seed the database with sample claims."""
    print(f"Seeding database with {num_claims} claims...")

    repo = InMemoryClaimRepository()
    graph = ClaimGraph()

    for i in range(num_claims):
        # Pick random sample claim
        content, platform, verdict = random.choice(SAMPLE_CLAIMS)

        # Add variation to content
        content = f"[{i}] {content} (Source: {random.choice(['A', 'B', 'C'])})"

        claim = create_claim(
            content=content,
            source_url=f"https://{platform}.com/post/{random.randint(100000, 999999)}",
            source_platform=platform,
            author_id=f"user_{random.randint(1000, 9999)}",
        )

        # Set verdict
        claim.initial_verdict = Verdict(verdict)

        # Save to repository
        repo.save(claim)

        # Add to graph
        graph.add_claim(claim)

        # Generate some propagation
        if random.random() < 0.7:
            graph.generate_propagation_tree(
                claim.claim_id,
                num_nodes=random.randint(5, 20),
                max_depth=random.randint(2, 5),
            )

    print(f"Created {repo.count()} claims")
    print(f"Graph has {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")

    # Print stats
    stats = graph.get_graph_stats()
    print(f"Graph density: {stats['density']:.4f}")

    return repo, graph


def main():
    """Main entry point."""
    asyncio.run(seed_database())
    print("Seeding complete!")


if __name__ == "__main__":
    main()
