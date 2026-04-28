"""Streamlit dashboard for misinformation detection visualization."""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Optional
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/assessments"


def init_session_state():
    """Initialize session state variables."""
    if "claims" not in st.session_state:
        st.session_state.claims = []
    if "assessments" not in st.session_state:
        st.session_state.assessments = {}
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 5


def fetch_claims() -> list:
    """Fetch claims from API."""
    try:
        response = requests.get(f"{API_URL}/claims", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        st.warning("Could not connect to API. Is the server running?")
    return []


def create_claim(content: str) -> Optional[dict]:
    """Create a new claim via API."""
    try:
        response = requests.post(
            f"{API_URL}/claims",
            json={
                "content": content,
                "source_platform": "dashboard",
                "author_id": "dashboard_user",
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error creating claim: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {e}")
    return None


def get_risk_color(score: float) -> str:
    """Get color for risk score."""
    if score < 0.3:
        return "green"
    elif score < 0.6:
        return "yellow"
    elif score < 0.8:
        return "orange"
    else:
        return "red"


def render_risk_gauge(score: float, confidence: float):
    """Render a risk gauge visualization."""
    color = get_risk_color(score)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #{'ffdddd' if score > 0.6 else 'ddffdd'}, #{'ffdddd' if score > 0.6 else 'ddffdd'}); border-radius: 10px; margin: 10px 0;">
                <h1 style="color: {color}; margin: 0;">{score:.2f}</h1>
                <p style="color: #666; margin: 5px 0;">Risk Score</p>
                <p style="color: #888; margin: 0;">Confidence: {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_claim_card(claim: dict):
    """Render a single claim card."""
    assessment = st.session_state.assessments.get(claim["claim_id"], {})

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**{claim['content'][:200]}...**" if len(claim["content"]) > 200 else f"**{claim['content']}**")
            st.caption(f"Platform: {claim['source_platform']} | Author: {claim['author_id']}")
            st.caption(f"Time: {claim.get('timestamp', 'N/A')}")

        with col2:
            if assessment:
                score = assessment.get("risk_score", 0)
                st.markdown(f"### {score:.2f}")
                verdict = assessment.get("veracity_prediction", "unknown")
                st.markdown(f"**{verdict.upper()}**")
            else:
                st.info("Processing...")

        st.divider()


def render_overview_page():
    """Render the overview dashboard page."""
    st.header("Overview")

    # Stats row
    stats = fetch_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Claims", stats.get("total_claims", 0))
        with col2:
            st.metric("Total Edges", stats.get("total_edges", 0))
        with col3:
            st.metric("Graph Density", f"{stats.get('density', 0):.4f}")
        with col4:
            st.metric("High Risk", stats.get("high_risk_count", 0))

    st.divider()

    # New claim form
    st.subheader("Submit New Claim")
    with st.form("claim_form", clear_on_submit=True):
        content = st.text_area(
            "Claim content",
            placeholder="Enter the claim to analyze...",
            height=100,
        )
        submitted = st.form_submit_button("Analyze Claim")

        if submitted and content:
            if len(content) >= 10:
                with st.spinner("Analyzing claim..."):
                    result = create_claim(content)
                    if result:
                        st.success("Claim submitted for analysis!")
                        time.sleep(1)
                        st.rerun()
            else:
                st.warning("Claim must be at least 10 characters.")

    st.divider()

    # Recent claims
    st.subheader("Recent Claims")
    claims = fetch_claims()

    if claims:
        for claim in claims[:10]:
            render_claim_card(claim)
    else:
        st.info("No claims yet. Submit one above!")


def fetch_stats() -> dict:
    """Fetch graph stats from API."""
    try:
        response = requests.get(f"{API_URL}/graphs/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def render_investigation_page():
    """Render claim investigation page."""
    st.header("Claim Investigation")

    claim_id = st.text_input("Enter Claim ID to investigate")

    if claim_id:
        try:
            response = requests.get(f"{API_URL}/claims/{claim_id}", timeout=5)
            if response.status_code == 200:
                claim = response.json()
                st.json(claim)

                # Fetch assessment
                ass_response = requests.get(f"{API_URL}/assessments/{claim_id}", timeout=5)
                if ass_response.status_code == 200:
                    assessment = ass_response.json()
                    st.subheader("Risk Assessment")
                    render_risk_gauge(
                        assessment.get("risk_score", 0),
                        assessment.get("confidence", 0),
                    )

                    st.json(assessment)
            else:
                st.error("Claim not found")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")


def render_graph_explorer_page():
    """Render graph explorer page."""
    st.header("Graph Explorer")

    claim_id = st.text_input("Enter Claim ID to explore its subgraph")

    if claim_id:
        depth = st.slider("Graph depth", 1, 5, 3)

        try:
            response = requests.get(
                f"{API_URL}/graphs/subgraph/{claim_id}",
                params={"depth": depth},
                timeout=10,
            )
            if response.status_code == 200:
                subgraph = response.json()
                st.json(subgraph)

                # Simple visualization
                st.subheader("Graph Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nodes", subgraph.get("node_count", 0))
                with col2:
                    st.metric("Edges", subgraph.get("edge_count", 0))
            else:
                st.error("Could not fetch subgraph")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")


def render_settings_page():
    """Render settings page."""
    st.header("Settings")

    st.subheader("Refresh Settings")
    st.session_state.refresh_interval = st.slider(
        "Auto-refresh interval (seconds)",
        1,
        30,
        st.session_state.refresh_interval,
    )

    st.divider()

    st.subheader("API Connection")
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("Connected to API!")
            else:
                st.error(f"API returned status {response.status_code}")
        except:
            st.error("Could not connect to API")


def main():
    """Main dashboard entry point."""
    init_session_state()

    st.title("🛡️ Misinformation Graph Detector")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Claim Investigation", "Graph Explorer", "Settings"],
    )

    # Route to appropriate page
    if page == "Overview":
        render_overview_page()
    elif page == "Claim Investigation":
        render_investigation_page()
    elif page == "Graph Explorer":
        render_graph_explorer_page()
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
