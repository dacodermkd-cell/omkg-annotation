import streamlit as st
import pandas as pd
import json
import os
import gdown
from datetime import datetime

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="OMKG Annotation Tool",
    page_icon="🧬",
    layout="wide"
)

# ── Constants ──────────────────────────────────────────
DATASET_URL = "https://drive.google.com/file/d/1H8A-jQZ64uPNgsXFurG6iY8f1MjIRrby/view?usp=sharing"
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

# ── Load dataset ───────────────────────────────────────
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

# ── Load/save annotations ──────────────────────────────
def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        return pd.read_csv(ANNOTATIONS_FILE)
    return pd.DataFrame(columns=[
        "annotator", "chunk_id", "disease",
        "subject", "predicate", "object",
        "label", "comment", "timestamp"
    ])

def save_annotation(annotator, chunk_id, disease,
                    triple, label, comment):
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

def get_annotated_chunks(annotator, annotations_df):
    if len(annotations_df) == 0:
        return set()
    annotator_df = annotations_df[
        annotations_df["annotator"] == annotator]
    return set(annotator_df["chunk_id"].unique())

# ── Styling ────────────────────────────────────────────
st.markdown("""
<style>
.chunk-text {
    background-color: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 15px;
    border-radius: 5px;
    font-size: 14px;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
}
.triple-card {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
.metric-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}
.high-conf { background-color: #d4edda; color: #155724; }
.low-conf { background-color: #f8d7da; color: #721c24; }
.hall-flag { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# ── Main app ───────────────────────────────────────────
def main():
    st.title("🧬 OMKG Biomedical Knowledge Graph")
    st.subheader("Triple Annotation Interface")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/"
            "000000/dna-helix.png",
            width=80)
        st.markdown("## Annotator")
        annotator = st.selectbox(
            "Select your name:",
            ANNOTATORS)

        st.markdown("---")
        st.markdown("## Navigation")
        page = st.radio(
            "Go to:",
            ["📋 Annotate", "📊 Progress",
             "📥 Download Results"])

        st.markdown("---")
        st.markdown("## Instructions")
        st.markdown("""
**For each triple:**
- Read the chunk text carefully
- Decide if the triple is supported
- ✅ **Correct** — fully supported by text
- ⚠️ **Partial** — partially supported
- ❌ **Incorrect** — not supported or wrong
- Add a comment if unsure
        """)

    # Load data
    try:
        dataset = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.info(
            "Make sure the Google Drive file "
            "ID is set correctly in the app.")
        return

    annotations_df = load_annotations()
    annotated_chunks = get_annotated_chunks(
        annotator, annotations_df)

    # ── Page: Annotate ─────────────────────────────────
    if page == "📋 Annotate":
        st.markdown(f"### Annotating as: "
                    f"**{annotator}**")

        # Disease filter
        diseases = sorted(set(
            c["disease"] for c in dataset))
        selected_disease = st.selectbox(
            "Filter by disease:",
            ["All diseases"] + diseases)

        # Filter chunks
        if selected_disease == "All diseases":
            filtered = dataset
        else:
            filtered = [
                c for c in dataset
                if c["disease"] == selected_disease]

        # Find next unannotated chunk
        unannotated = [
            c for c in filtered
            if c["chunk_id"] not in annotated_chunks]

        st.markdown(
            f"**Progress:** "
            f"{len(annotated_chunks)}/300 chunks done "
            f"| {len(unannotated)} remaining")

        if len(unannotated) == 0:
            st.success(
                "🎉 All chunks annotated! "
                "Check the Progress tab.")
            return

        # Show chunk navigator
        chunk_idx = st.number_input(
            f"Chunk (1 to {len(unannotated)})",
            min_value=1,
            max_value=len(unannotated),
            value=1) - 1

        chunk = unannotated[chunk_idx]

        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                f"#### 📄 Chunk: `{chunk['chunk_id']}`")
            st.markdown(
                f"**Disease:** {chunk['disease']}")
            st.markdown(
                f"**Triples to annotate:** "
                f"{len(chunk['triples'])}")
            st.markdown(
                "<div class='chunk-text'>"
                f"{chunk['text']}"
                "</div>",
                unsafe_allow_html=True)

        with col2:
            st.markdown(
                f"#### 🔗 Triples "
                f"({len(chunk['triples'])} total)")

            with st.form(
                    key=f"form_{chunk['chunk_id']}"):
                labels = {}
                comments = {}

                for i, triple in enumerate(
                        chunk["triples"]):
                    conf = triple.get(
                        "avg_confidence", 0)
                    cons = triple.get(
                        "consensus_score", 1)
                    hall = triple.get(
                        "hallucination_flag", False)
                    epi = triple.get(
                        "epistemic_uncertainty", 0.8)

                    conf_class = "high-conf" \
                        if conf >= 0.8 else "low-conf"
                    hall_str = (
                        "<span class='metric-badge "
                        "hall-flag'>⚠️ Hallucination "
                        "Risk</span>"
                        if hall else "")

                    st.markdown(
                        f"<div class='triple-card'>"
                        f"<b>{i+1}.</b> "
                        f"<code>{triple['subject']}</code>"
                        f" → <b>{triple['predicate']}"
                        f"</b> → "
                        f"<code>{triple['object']}"
                        f"</code><br>"
                        f"<span class='metric-badge "
                        f"{conf_class}'>"
                        f"conf={conf:.2f}</span> "
                        f"<span class='metric-badge "
                        f"high-conf'>"
                        f"consensus={cons}/5</span> "
                        f"{hall_str}"
                        f"</div>",
                        unsafe_allow_html=True)

                    labels[i] = st.radio(
                        f"Label triple {i+1}:",
                        list(LABELS.keys()),
                        key=f"label_{chunk['chunk_id']}"
                            f"_{i}",
                        horizontal=True)
                    comments[i] = st.text_input(
                        f"Comment (optional):",
                        key=f"comment_"
                            f"{chunk['chunk_id']}"
                            f"_{i}")

                submitted = st.form_submit_button(
                    "✅ Submit All Annotations",
                    use_container_width=True,
                    type="primary")

                if submitted:
                    for i, triple in enumerate(
                            chunk["triples"]):
                        save_annotation(
                            annotator,
                            chunk["chunk_id"],
                            chunk["disease"],
                            triple,
                            LABELS[labels[i]],
                            comments[i])
                    st.success(
                        f"✅ Saved {len(chunk['triples'])}"
                        f" annotations for "
                        f"`{chunk['chunk_id']}`!")
                    st.rerun()

    # ── Page: Progress ─────────────────────────────────
    elif page == "📊 Progress":
        st.markdown("### 📊 Annotation Progress")

        if len(annotations_df) == 0:
            st.info("No annotations yet.")
            return

        col1, col2, col3, col4 = st.columns(4)

        total_annotated = len(
            annotations_df["chunk_id"].unique())
        total_triples = len(annotations_df)
        correct = (
            annotations_df["label"] == "correct"
            ).sum()
        incorrect = (
            annotations_df["label"] == "incorrect"
            ).sum()
        partial = (
            annotations_df["label"] == "partial"
            ).sum()

        col1.metric("Chunks Done",
                    f"{total_annotated}/300")
        col2.metric("Triples Annotated",
                    f"{total_triples:,}")
        col3.metric("Correct",
                    f"{correct:,}",
                    f"{correct/total_triples*100:.1f}%"
                    if total_triples > 0 else "0%")
        col4.metric("Incorrect",
                    f"{incorrect:,}",
                    f"{incorrect/total_triples*100:.1f}%"
                    if total_triples > 0 else "0%")

        st.markdown("---")

        # Per annotator progress
        st.markdown("#### Per Annotator")
        for ann in ANNOTATORS:
            ann_df = annotations_df[
                annotations_df["annotator"] == ann]
            chunks_done = len(
                ann_df["chunk_id"].unique())
            triples_done = len(ann_df)
            st.markdown(
                f"**{ann}:** {chunks_done}/300 chunks "
                f"| {triples_done:,} triples")
            st.progress(chunks_done / 300)

        st.markdown("---")

        # Per disease progress
        st.markdown("#### Per Disease")
        disease_progress = annotations_df.groupby(
            "disease")["chunk_id"].nunique()
        for disease in sorted(
                annotations_df["disease"].unique()):
            done = disease_progress.get(disease, 0)
            st.markdown(
                f"**{disease}:** {done}/30 chunks")
            st.progress(done / 30)

        st.markdown("---")

        # Label distribution
        st.markdown("#### Label Distribution")
        label_counts = annotations_df[
            "label"].value_counts()
        st.bar_chart(label_counts)

        # Disease breakdown
        st.markdown("#### Annotations by Disease")
        disease_labels = annotations_df.groupby(
            ["disease", "label"]).size().unstack(
            fill_value=0)
        st.dataframe(disease_labels)

    # ── Page: Download ──────────────────────────────────
    elif page == "📥 Download Results":
        st.markdown("### 📥 Download Annotations")

        if len(annotations_df) == 0:
            st.info("No annotations to download yet.")
            return

        st.markdown(
            f"**Total annotations:** "
            f"{len(annotations_df):,}")

        # Full CSV download
        csv = annotations_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download All Annotations (CSV)",
            data=csv,
            file_name="omkg_annotations.csv",
            mime="text/csv",
            use_container_width=True)

        # Per annotator download
        st.markdown("#### Per Annotator")
        for ann in ANNOTATORS:
            ann_df = annotations_df[
                annotations_df["annotator"] == ann]
            if len(ann_df) > 0:
                csv_ann = ann_df.to_csv(index=False)
                st.download_button(
                    label=f"⬇️ {ann} annotations "
                          f"({len(ann_df):,} rows)",
                    data=csv_ann,
                    file_name=f"annotations_"
                              f"{ann.replace(' ','_')}"
                              f".csv",
                    mime="text/csv")

        st.markdown("---")
        st.markdown("#### Preview")
        st.dataframe(annotations_df.tail(20))

if __name__ == "__main__":
    main()
