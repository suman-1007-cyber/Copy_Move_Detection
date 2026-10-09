
import io
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="PixelTrace — Image Forensics",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# DESIGN SYSTEM — warm white, black, subtle gray
# --------------------------------------------------
st.markdown("""
<style>
:root {
    --paper: #f8f8f6;
    --white: #ffffff;
    --ink: #191919;
    --muted: #777772;
    --line: #e4e4df;
    --soft: #efefeb;
    --accent: #242424;
}
.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}
.block-container {
    max-width: 1440px;
    padding: 1.5rem 2.5rem 3rem;
}
h1, h2, h3, h4 {
    color: var(--ink) !important;
    letter-spacing: -0.045em !important;
}
h1 {
    font-size: clamp(2.2rem, 4vw, 3.7rem) !important;
    font-weight: 650 !important;
    line-height: 1.06 !important;
}
h2 { font-size: 1.65rem !important; font-weight: 620 !important; }
h3 { font-size: 1.1rem !important; font-weight: 620 !important; }
p, label { line-height: 1.65; }
[data-testid="stCaptionContainer"] {
    color: var(--muted);
}
[data-testid="stSidebar"] {
    background: #f2f2ee;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
.stButton > button, .stDownloadButton > button {
    min-height: 43px;
    border: 1px solid #d6d6d0;
    border-radius: 5px;
    background: white;
    color: #222;
    font-size: 0.9rem;
    font-weight: 550;
    box-shadow: none;
    transition: background .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #efefeb;
    border-color: #aaa9a2;
    color: #111;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #202020;
    border-color: #202020;
    color: white;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #414141;
    border-color: #414141;
    color: white;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1.1rem;
    box-shadow: none;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .8rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-weight: 620 !important;
    letter-spacing: -.04em;
}
[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #c9c9c2;
    border-radius: 5px;
    padding: .75rem;
}
[data-testid="stImage"] img {
    border-radius: 3px;
    border: 1px solid var(--line);
}
button[data-baseweb="tab"] {
    font-weight: 550;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #111;
}
[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 5px;
}
[data-testid="stAlert"] {
    border-radius: 5px;
}
hr {
    border-color: var(--line);
}
footer { visibility: hidden; }

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0 18px;
    margin-bottom: 54px;
    border-bottom: 1px solid var(--line);
}
.wordmark {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -.04em;
    color: #181818;
}
.wordmark span {
    font-weight: 450;
    color: #777772;
}
.topbar-meta {
    color: #777772;
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.hero-label {
    margin-bottom: 17px;
    color: #777772;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: .13em;
    text-transform: uppercase;
}
.hero-copy {
    max-width: 670px;
    margin: 18px 0 26px;
    color: #666660;
    font-size: 16px;
    line-height: 1.8;
}
.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 30px;
}
.hero-meta span {
    padding: 7px 10px;
    border: 1px solid #deded8;
    background: transparent;
    color: #555550;
    font-size: 10px;
    letter-spacing: .055em;
}
.section-label {
    margin: 34px 0 8px;
    color: #777772;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.panel {
    padding: 17px;
    margin-bottom: 10px;
    background: white;
    border: 1px solid var(--line);
    border-radius: 5px;
}
.panel-title {
    color: #202020;
    font-size: 14px;
    font-weight: 650;
}
.panel-copy {
    margin-top: 5px;
    color: #777772;
    font-size: 12px;
    line-height: 1.65;
}
.notice {
    margin: 28px 0;
    padding: 14px 16px;
    border-left: 2px solid #777772;
    background: #f0f0ec;
    color: #5f5f59;
    font-size: 12px;
    line-height: 1.7;
}
.result-banner {
    padding: 15px 17px;
    margin: 18px 0;
    background: white;
    border: 1px solid var(--line);
    border-left: 3px solid #363636;
    color: #393939;
    font-size: 13px;
    line-height: 1.7;
}
.footer-note {
    padding-top: 14px;
    border-top: 1px solid var(--line);
    color: #888880;
    font-size: 10px;
    letter-spacing: .035em;
}
@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .topbar { margin-bottom: 32px; }
    .topbar-meta { display: none; }
    h1 { font-size: 2.35rem !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="topbar">
    <div class="wordmark">PIXELTRACE <span>/ IMAGE FORENSICS</span></div>
    <div class="topbar-meta">Digital Image Processing · Research Tool</div>
</div>
<div class="hero-label">Image integrity / 001</div>
<h1>Look closer.<br>Find what was copied.</h1>
<div class="hero-copy">
    An image-forensics tool for identifying potentially duplicated
    regions in digital images using traditional image processing
    and Discrete Cosine Transform (DCT) block matching.
</div>
<div class="hero-meta">
    <span>TRADITIONAL DIP</span>
    <span>DCT BLOCK MATCHING</span>
    <span>NO AI / ML / DL / CNN</span>
</div>
<div class="notice">
    <strong>About the results.</strong>
    Highlighted areas indicate potential matching regions, not
    definitive proof of manipulation. Natural repetition and textures
    can produce false positives. Results require human interpretation.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ORIGINAL DEMO IMAGE GENERATOR — UNCHANGED
# --------------------------------------------------
def make_demo_image():
    """Create a deterministic synthetic scene with a copied patch for repeatable testing."""
    rng = np.random.default_rng(17)
    h, w = 360, 520
    noise = rng.normal(0, 10, (h, w, 3)).astype(np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), dtype=np.int16)
    base[:, :, 0] = 95 + (yy * 35 // h)
    base[:, :, 1] = 145 + (xx * 30 // w)
    base[:, :, 2] = 75 + (yy * 20 // h)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(
        (55, 90, 175, 210), radius=8,
        fill=(225, 185, 65), outline=(35, 35, 35), width=4
    )
    d.rectangle(
        (68, 103, 162, 197),
        fill=(40, 90, 155), outline=(245, 245, 245), width=3
    )
    for i in range(5):
        d.line(
            (73, 112 + i * 15, 157, 112 + i * 15),
            fill=(240, 220, 120), width=4
        )
    d.ellipse(
        (91, 130, 139, 178),
        fill=(220, 65, 55), outline=(255, 255, 255), width=3
    )
    d.line((115, 134, 126, 120), fill=(35, 100, 40), width=5)
    for cx, cy, r, color in [
        (260, 100, 24, (235, 220, 120)),
        (350, 260, 32, (70, 120, 60)),
        (450, 100, 18, (190, 90, 60)),
    ]:
        d.ellipse(
            (cx-r, cy-r, cx+r, cy+r),
            fill=color, outline=(35, 35, 35), width=3
        )
    d.rectangle((220, 285, 500, 305), fill=(95, 70, 45))
    patch = im.crop((55, 90, 175, 210))
    im.paste(patch, (311, 114))
    return im


# --------------------------------------------------
# ORIGINAL DETECTION ALGORITHM — UNCHANGED
# --------------------------------------------------
def detect_copy_move(
    rgb, block_size=16, stride=8, min_distance=40, threshold=0.06
):
    """
    Traditional block-based DCT matching.
    Classroom prototype; repeated textures can create false positives.
    """
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, 512.0 / max(h0, w0))
    if scale < 1:
        resized = cv2.resize(
            rgb,
            (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if h < block_size or w < block_size:
        raise ValueError(
            "Image is too small. Please upload an image at least 16 × 16 pixels."
        )

    buckets = defaultdict(list)
    descriptors = []
    coords = []

    for y in range(0, h - block_size + 1, stride):
        for x in range(0, w - block_size + 1, stride):
            block = gray[y:y + block_size, x:x + block_size].astype(np.float32)
            if float(block.std()) < 8.0:
                continue
            dct = cv2.dct(block)
            feat = dct[:8, :8].flatten()[1:]
            feat = feat / (float(np.linalg.norm(feat)) + 1e-8)
            key = tuple(np.round(feat / threshold).astype(np.int16).tolist())
            idx = len(coords)
            coords.append((x, y))
            descriptors.append(feat)
            buckets[key].append(idx)

    pairs = []
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        ids = ids[:80]
        for a_pos in range(len(ids)):
            ia = ids[a_pos]
            xa, ya = coords[ia]
            for b_pos in range(a_pos + 1, len(ids)):
                ib = ids[b_pos]
                xb, yb = coords[ib]
                if (xa - xb) ** 2 + (ya - yb) ** 2 < min_distance ** 2:
                    continue
                sim = float(np.dot(descriptors[ia], descriptors[ib]))
                if sim > 0.985:
                    pairs.append((ia, ib, sim))

    offset_counts = defaultdict(list)
    for ia, ib, sim in pairs:
        xa, ya = coords[ia]
        xb, yb = coords[ib]
        dx, dy = xb - xa, yb - ya
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
            ia, ib = ib, ia
        offset_counts[(dx, dy)].append((ia, ib, sim))

    selected = []
    for group in offset_counts.values():
        if len(group) >= 3:
            selected.extend(group)
    if not selected and pairs:
        selected = sorted(pairs, key=lambda p: p[2], reverse=True)[:30]

    mask = np.zeros((h, w), dtype=np.uint8)
    for ia, ib, _ in selected:
        for idx in (ia, ib):
            x, y = coords[idx]
            mask[y:y + block_size, x:x + block_size] = 255

    if scale < 1:
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    marked = rgb.copy()
    if mask.any():
        red = np.zeros_like(marked)
        red[:, :, 0] = 255
        m = mask > 0
        marked[m] = (0.55 * marked[m] + 0.45 * red[m]).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(marked, contours, -1, (255, 30, 30), 2)

    return marked, mask, len(coords), len(pairs), len(selected)


# --------------------------------------------------
# SIDEBAR — SAME ANALYSIS CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.markdown("### PixelTrace")
    st.caption("Analysis settings")
    st.divider()

    st.markdown("**01 / Block analysis**")
    block_size = st.select_slider(
        "Block size (pixels)",
        options=[8, 12, 16, 20, 24],
        value=16
    )
    stride = st.select_slider(
        "Block stride (pixels)",
        options=[4, 8, 12, 16],
        value=8
    )
    min_distance = st.slider(
        "Minimum match distance",
        min_value=16,
        max_value=100,
        value=40,
        step=8
    )

    st.divider()
    st.markdown("**02 / Method**")
    st.caption(
        "Grayscale → DCT descriptors → block matching → "
        "spatial localization."
    )
    st.caption(
        "Best suited to copied patches pasted without rotation or resizing."
    )


# --------------------------------------------------
# IMAGE SOURCE
# --------------------------------------------------
st.markdown(
    '<div class="section-label">01 — IMAGE SOURCE</div>',
    unsafe_allow_html=True
)
st.subheader("Choose an image to inspect")

demo_col, upload_col = st.columns([1, 1.5], gap="large")

if "demo_image" not in st.session_state:
    st.session_state.demo_image = None

with demo_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Synthetic test image</div>
        <div class="panel-copy">
            A repeatable sample containing a deliberately copied region.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Generate test image",
        type="primary",
        use_container_width=True
    ):
        st.session_state.demo_image = make_demo_image()

with upload_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Upload an image</div>
        <div class="panel-copy">
            Choose a JPG, JPEG or PNG image from your computer.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


# --------------------------------------------------
# LOAD SELECTED IMAGE
# --------------------------------------------------
rgb = None
source_name = None

if uploaded is not None:
    try:
        rgb = np.array(Image.open(uploaded).convert("RGB"))
        source_name = uploaded.name
    except Exception as e:
        st.error(f"Could not read this image: {e}")
elif st.session_state.demo_image is not None:
    rgb = np.array(st.session_state.demo_image.convert("RGB"))
    source_name = "Synthetic copy-move test image"


# --------------------------------------------------
# FORENSIC WORKSPACE — SAME RESULTS AND FEATURES
# --------------------------------------------------
if rgb is not None:
    st.divider()
    st.markdown(
        '<div class="section-label">02 — FORENSIC WORKSPACE</div>',
        unsafe_allow_html=True
    )
    st.subheader("Image analysis")
    st.caption(
        f"Source: {source_name}  ·  "
        f"Resolution: {rgb.shape[1]:,} × {rgb.shape[0]:,} pixels"
    )

    with st.spinner("Computing DCT descriptors and matching blocks..."):
        try:
            marked, mask, block_count, pair_count, selected_count = (
                detect_copy_move(
                    rgb,
                    block_size=block_size,
                    stride=stride,
                    min_distance=min_distance
                )
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**01 / Original image**")
        st.caption("Unmodified source image")
        st.image(rgb, use_container_width=True)

    with right:
        st.markdown("**02 / Detection overlay**")
        st.caption("Red indicates regions flagged by the detector")
        st.image(marked, use_container_width=True)

    st.markdown(
        '<div class="section-label">03 — ANALYSIS METRICS</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("Blocks analyzed", f"{block_count:,}")
    m2.metric("Candidate matches", f"{pair_count:,}")
    m3.metric("Selected matching pairs", f"{selected_count:,}")

    if mask.any():
        st.markdown("""
        <div class="result-banner">
            <strong>Potential matching regions identified.</strong><br>
            Review the highlighted areas. This result is not definitive
            proof of image forgery.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(
            "No matching regions were highlighted. Try another image or "
            "adjust the block size, stride, or minimum distance."
        )

    with st.expander("Inspect binary localization mask"):
        st.caption(
            "White pixels represent regions selected by the algorithm; "
            "black pixels are not flagged."
        )
        st.image(mask, clamp=True, use_container_width=True)

    output_buffer = io.BytesIO()
    Image.fromarray(marked).save(output_buffer, format="PNG")

    st.download_button(
        "Download annotated image (PNG)",
        data=output_buffer.getvalue(),
        file_name="copy_move_detection_result.png",
        mime="image/png",
        type="primary",
        use_container_width=True
    )

    st.markdown(
        '<div class="section-label">04 — TECHNICAL REPORT</div>',
        unsafe_allow_html=True
    )

    with st.expander("Algorithm workflow and interpretation"):
        st.markdown("""
**Detection pipeline**

1. **Preprocessing:** Convert RGB pixels to grayscale; downscale very large images.
2. **Block extraction:** Traverse overlapping image blocks using the configured stride.
3. **DCT feature extraction:** Compute the Discrete Cosine Transform and retain low-frequency coefficients.
4. **Descriptor matching:** Normalize and quantize block descriptors to find similar candidates.
5. **Spatial filtering:** Exclude nearby matches and group matches by displacement vectors.
6. **Localization:** Build a binary mask and overlay selected blocks in red.

**Interpretation and limitations**

- This is a traditional image-processing prototype, not an AI or machine-learning system.
- Repeated natural textures can generate false positives.
- Rotation, resizing, compression, or other transformations can reduce detection reliability.
- A highlighted area is a candidate for human review, not conclusive proof of manipulation.
""")

else:
    st.divider()
    st.markdown(
        '<div class="section-label">02 — AWAITING INPUT</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Ready to start an analysis</div>
        <div class="panel-copy">
            Generate the synthetic test image or upload a JPG, JPEG or PNG.
            Start with the synthetic image for a repeatable classroom demonstration.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div class="footer-note">
    PIXELTRACE &nbsp;·&nbsp; ACADEMIC DIGITAL IMAGE PROCESSING
    &nbsp;·&nbsp; DCT BLOCK MATCHING
    &nbsp;·&nbsp; RESULTS REQUIRE HUMAN INTERPRETATION
</div>
""", unsafe_allow_html=True)
