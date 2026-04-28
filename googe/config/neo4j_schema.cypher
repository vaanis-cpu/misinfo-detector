// Neo4j Schema for Misinformation Graph Detector
// Migration: 001_initial_schema.cypher

// Create constraints
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE;
CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE;
CREATE CONSTRAINT platform_id IF NOT EXISTS FOR (p:Platform) REQUIRE p.platform_id IS UNIQUE;

// Node types
CREATE (c:Claim {
    claim_id: String(),
    content: String(),
    source_url: String(),
    source_platform: String(),
    author_id: String(),
    timestamp: DateTime(),
    embedding: List<Float>(),
    initial_verdict: String(),
    created_at: DateTime(),
    updated_at: DateTime()
})
CREATE INDEX claim_content IF NOT EXISTS FOR (c:Claim) ON (c.content);
CREATE INDEX claim_timestamp IF NOT EXISTS FOR (c:Claim) ON (c.timestamp);
CREATE INDEX claim_verdict IF NOT EXISTS FOR (c:Claim) ON (c.initial_verdict);

CREATE (a:Author {
    author_id: String(),
    username: String(),
    display_name: String(),
    followers_count: Integer(),
    following_count: Integer(),
    account_age_days: Integer(),
    is_verified: Boolean(),
    risk_score: Float(),
    created_at: DateTime()
})
CREATE INDEX author_risk IF NOT EXISTS FOR (a:Author) ON (a.risk_score);

CREATE (p:Platform {
    platform_id: String(),
    name: String(),
    domain: String()
});

CREATE (rs:RiskScore {
    claim_id: String(),
    risk_score: Float(),
    confidence: Float(),
    veracity_prediction: String(),
    propagation_depth: Integer(),
    velocity: Float(),
    graph_centrality: Float(),
    temporal_trend: String(),
    timestamp: DateTime()
})
CREATE INDEX risk_timestamp IF NOT EXISTS FOR (rs:RiskScore) ON (rs.timestamp);

// Relationship types
CREATE (c1:Claim)-[:SHARES {
    share_id: String(),
    share_type: String(),  // retweet, quote, reply
    timestamp: DateTime(),
    weight: Float()
}]->(c2:Claim);

CREATE (a:Author)-[:POSTED {
    post_id: String(),
    timestamp: DateTime()
}]->(c:Claim);

CREATE (c:Claim)-[:RECEIVED_ASSESSMENT {
    assessment_id: String(),
    timestamp: DateTime()
}]->(rs:RiskScore);

CREATE (c1:Claim)-[:REPLY_TO {
    reply_id: String(),
    timestamp: DateTime()
}]->(c2:Claim);

// Temporal graph for snapshots
CREATE (s:Snapshot {
    snapshot_id: String(),
    claim_id: String(),
    window_start: DateTime(),
    window_end: DateTime(),
    node_count: Integer(),
    edge_count: Integer()
});
