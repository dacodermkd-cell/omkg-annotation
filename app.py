import streamlit as st
import pandas as pd
import json
import os
import gdown
from datetime import datetime

st.set_page_config(
    page_title="OMKG Annotation Tool",
    page_icon="🧬",
    layout="wide"
)

DATASET_URL = "1H8A-jQZ64uPNgsXFurG6iY8f1MjIRrby"
ANNOTATIONS_FILE = "annotations.csv"
DATASET_FILE = "annotation_dataset.json"

ANNOTATORS = [
    "Mitul Kumar",
    "Guide 1",
    "Guide 2"
]

LABELS = {
    "✅ Correct": "correct",
    "❌ Incorrect": "incorrect",
    "⚠️ Partial": "partial"
}

@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_FILE):
        with st.spinner(
                "Downloading annotation dataset..."):
            gdown.download(
                f"https://drive.google.com/uc?"
                f"id={DATASET_URL}",
                DATASET_FILE, quiet=False)
    with open(DATASET_FILE, "r") as f:
        return json.load(f)

def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        return pd.read_csv(ANNOTATIONS_FILE)
    return pd.DataFrame(columns=[
        "annotator", "chunk_id", "disease",
        "subject", "predicate", "object",
        "label", "comment", "timestamp"
    ])

def save_annotation(annotator, chunk_id,
                    disease, triple,
                    label, comment):
    df = load_annotations()
    new_row = {
        "annotator": annotator,
        "chunk_id": chunk_id,
        "disease": disease,
        "subject": triple["subject"],
        "predicate": triple["predicate"],
        "object": triple["object"],
        "label": label,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    }
    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True)
    df.to_csv(ANNOTATIONS_FILE, index=False)
    return df

def get_annotated_chunks(annotator, df):
    if len(df) == 0:
        return set()
    return set(
        df[df["annotator"] == annotator][
            "chunk_id"].unique())

def main():
    st.title("🧬 OMKG Annotation Tool")
    st.caption(
        "Biomedical Knowledge Graph — "
        "Triple Annotation Interface")
    st.divider()

    with st.sidebar:
        st.markdown("## 👤 Annotator")
        annotator = st.selectbox(
            "Select your name:", ANNOTATORS)
        st.divider()
        st.markdown("## 🗂️ Navigation")
        page = st.radio("", [
            "📋 Annotate",
            "📊 Progress",
            "📥 Download"])
        st.divider()
        st.markdown("""
## 📖 Instructions

For each triple shown:

**Read** the chunk text on the left.

Then judge each triple:

✅ **Correct** — The triple states a
fact that is supported by the text
or is medically accurate.

⚠️ **Partial** — The triple is
partially correct or imprecise.

❌ **Incorrect** — The triple is
wrong, hallucinated, or not supported.

Click **Submit** when done with
all 15 triples.
        """)

    try:
        dataset = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    annotations_df = load_annotations()
    annotated = get_annotated_chunks(
        annotator, annotations_df)

    # ── ANNOTATE PAGE ──────────────────────────
    if page == "📋 Annotate":
        st.markdown(
            f"### Annotating as: **{annotator}**")

        diseases = sorted(set(
            c["disease"] for c in dataset))
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            sel_disease = st.selectbox(
                "Filter by disease:",
                ["All diseases"] + diseases)
        with col_f2:
            show_done = st.checkbox(
                "Show completed chunks", False)

        if sel_disease == "All diseases":
            filtered = dataset
        else:
            filtered = [
                c for c in dataset
                if c["disease"] == sel_disease]

        if show_done:
            pool = filtered
        else:
            pool = [c for c in filtered
                    if c["chunk_id"]
                    not in annotated]

        done_count = len(annotated)
        st.progress(done_count / 300,
                    text=f"{done_count}/300 chunks "
                         f"annotated")

        if len(pool) == 0:
            st.success(
                "🎉 All chunks annotated!")
            return

        chunk_idx = st.number_input(
            f"Chunk number (1–{len(pool)})",
            min_value=1,
            max_value=len(pool),
            value=1) - 1

        chunk = pool[chunk_idx]
        triples = chunk["triples"]

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            st.markdown(
                f"**Chunk:** `{chunk['chunk_id']}`")
            st.markdown(
                f"**Disease:** {chunk['disease']}")
            st.markdown(
                f"**Triples:** {len(triples)}")
            st.markdown("**Text:**")
            # Use st.text_area for reliable display
            st.text_area(
                label="chunk_text_display",
                value=chunk["text"],
                height=500,
                disabled=True,
                label_visibility="collapsed")

        with right:
            st.markdown(
                f"**Triples ({len(triples)} total)**")

            with st.form(
                    key=f"f_{chunk['chunk_id']}"):
                labels = {}
                comments = {}

                for i, t in enumerate(triples):
                    conf = float(
                        t.get("avg_confidence",
                              0.5))
                    cons = int(
                        t.get("consensus_score",
                              1))
                    hall = bool(
                        t.get("hallucination_flag",
                              False))

                    with st.container(border=True):
                        st.markdown(
                            f"**Triple {i+1}**")
                        st.markdown(
                            f"**Subject:** "
                            f"`{t['subject']}`")
                        st.markdown(
                            f"**Predicate:** "
                            f"`{t['predicate']}`")
                        st.markdown(
                            f"**Object:** "
                            f"`{t['object']}`")

                        badge_cols = st.columns(3)
                        badge_cols[0].metric(
                            "Confidence",
                            f"{conf:.2f}")
                        badge_cols[1].metric(
                            "Consensus",
                            f"{cons}/5")
                        if hall:
                            badge_cols[2].warning(
                                "⚠️ Hallucination "
                                "Risk")

                        labels[i] = st.radio(
                            "Label:",
                            list(LABELS.keys()),
                            key=f"l_{chunk['chunk_id']}"
                                f"_{i}",
                            horizontal=True)
                        comments[i] = st.text_input(
                            "Comment:",
                            key=f"c_{chunk['chunk_id']}"
                                f"_{i}",
                            label_visibility=
                            "collapsed",
                            placeholder=
                            "Optional comment...")

                submitted = st.form_submit_button(
                    "✅ Submit All Annotations",
                    use_container_width=True,
                    type="primary")

                if submitted:
                    for i, t in enumerate(triples):
                        save_annotation(
                            annotator,
                            chunk["chunk_id"],
                            chunk["disease"],
                            t,
                            LABELS[labels[i]],
                            comments[i])
                    st.success(
                        f"Saved {len(triples)} "
                        f"annotations!")
                    st.rerun()

    # ── PROGRESS PAGE ──────────────────────────
    elif page == "📊 Progress":
        st.markdown("### 📊 Progress")

        if len(annotations_df) == 0:
            st.info("No annotations yet.")
            return

        c1, c2, c3, c4 = st.columns(4)
        total_chunks = len(
            annotations_df["chunk_id"].unique())
        total_t = len(annotations_df)
        correct = (
            annotations_df["label"]
            == "correct").sum()
        incorrect = (
            annotations_df["label"]
            == "incorrect").sum()

        c1.metric("Chunks Done",
                  f"{total_chunks}/300")
        c2.metric("Triples Done", f"{total_t:,}")
        c3.metric("✅ Correct",
                  f"{correct/total_t*100:.1f}%")
        c4.metric("❌ Incorrect",
                  f"{incorrect/total_t*100:.1f}%")

        st.divider()
        st.markdown("#### Per Annotator")
        for ann in ANNOTATORS:
            adf = annotations_df[
                annotations_df["annotator"] == ann]
            n = len(adf["chunk_id"].unique())
            st.markdown(
                f"**{ann}:** {n}/300 chunks | "
                f"{len(adf):,} triples")
            st.progress(n / 300)

        st.divider()
        st.markdown("#### Per Disease")
        dp = annotations_df.groupby("disease")[
            "chunk_id"].nunique()
        for d in sorted(dp.index):
            n = dp[d]
            st.markdown(f"**{d}:** {n}/30")
            st.progress(n / 30)

        st.divider()
        st.markdown("#### Label Distribution")
        lc = annotations_df["label"].value_counts()
        st.bar_chart(lc)

    # ── DOWNLOAD PAGE ───────────────────────────
    elif page == "📥 Download":
        st.markdown("### 📥 Download Results")

        if len(annotations_df) == 0:
            st.info("No annotations yet.")
            return

        st.markdown(
            f"**{len(annotations_df):,} total "
            f"annotations**")

        st.download_button(
            "⬇️ Download All (CSV)",
            annotations_df.to_csv(index=False),
            "omkg_annotations.csv",
            "text/csv",
            use_container_width=True)

        st.divider()
        for ann in ANNOTATORS:
            adf = annotations_df[
                annotations_df["annotator"] == ann]
            if len(adf) > 0:
                st.download_button(
                    f"⬇️ {ann} ({len(adf):,} rows)",
                    adf.to_csv(index=False),
                    f"annotations_"
                    f"{ann.replace(' ','_')}.csv",
                    "text/csv")

        st.divider()
        st.markdown("#### Preview (last 20)")
        st.dataframe(annotations_df.tail(20))

if __name__ == "__main__":
    main()
