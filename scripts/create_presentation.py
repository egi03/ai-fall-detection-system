"""Generate faculty presentation PPTX for the fall detection project."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.util as util
from pathlib import Path


# ── Color palette (dark navy + accent) ──────────────────────────────────────
NAVY   = RGBColor(0x1A, 0x2F, 0x5E)   # slide background / title bar
BLUE   = RGBColor(0x1F, 0x6F, 0xC8)   # accent / headers
TEAL   = RGBColor(0x00, 0xB0, 0xA0)   # highlight
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT  = RGBColor(0xF0, 0xF4, 0xFF)   # body background
GRAY   = RGBColor(0x55, 0x65, 0x75)
GREEN  = RGBColor(0x28, 0xA7, 0x45)
RED    = RGBColor(0xDC, 0x35, 0x45)
ORANGE = RGBColor(0xFD, 0x7E, 0x14)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── Helpers ──────────────────────────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs: Presentation):
    blank = prs.slide_layouts[6]   # truly blank
    return prs.slides.add_slide(blank)


def fill_bg(slide, color: RGBColor):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, fill_color: RGBColor = None,
             line_color: RGBColor = None, line_width: int = 0):
    shape = slide.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE_TYPE.AUTO_SHAPE if False else 1,   # 1 = rectangle
        l, t, w, h)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color and line_width:
        shape.line.color.rgb = line_color
        shape.line.width = util.Pt(line_width)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text, l, t, w, h,
                font_size=18, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_para(tf, text, font_size=16, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, space_before=Pt(4), indent=False):
    p = tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = space_before
    if indent:
        p.level = 1
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p


def header_bar(slide, title_text, subtitle=None):
    """Navy header bar at top of slide."""
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.15), fill_color=NAVY)
    add_textbox(slide, title_text,
                Inches(0.3), Inches(0.08), Inches(12), Inches(0.65),
                font_size=28, bold=True, color=WHITE)
    if subtitle:
        add_textbox(slide, subtitle,
                    Inches(0.3), Inches(0.72), Inches(12), Inches(0.38),
                    font_size=15, color=TEAL)


def slide_number(slide, n, total):
    add_textbox(slide, f"{n} / {total}",
                Inches(12.0), Inches(7.15), Inches(1.2), Inches(0.3),
                font_size=11, color=GRAY, align=PP_ALIGN.RIGHT)


def metric_box(slide, label, value, l, t, w=Inches(2.4), h=Inches(1.2),
               bg=BLUE, val_size=32, lbl_size=13):
    add_rect(slide, l, t, w, h, fill_color=bg)
    add_textbox(slide, value, l, t + Inches(0.08), w, Inches(0.65),
                font_size=val_size, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, label, l, t + Inches(0.72), w, Inches(0.42),
                font_size=lbl_size, color=LIGHT, align=PP_ALIGN.CENTER)


def bullet_box(slide, title, bullets, l, t, w, h,
               bg=LIGHT, title_color=NAVY, bullet_color=GRAY,
               title_size=17, bullet_size=14):
    add_rect(slide, l, t, w, h, fill_color=bg)
    y = t + Inches(0.12)
    add_textbox(slide, title, l + Inches(0.15), y, w - Inches(0.3), Inches(0.38),
                font_size=title_size, bold=True, color=title_color)
    y += Inches(0.42)
    for b in bullets:
        add_textbox(slide, f"• {b}", l + Inches(0.15), y,
                    w - Inches(0.3), Inches(0.35),
                    font_size=bullet_size, color=bullet_color)
        y += Inches(0.33)


# ── Slides ───────────────────────────────────────────────────────────────────

def slide_title(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, NAVY)
    # Decorative side stripe
    add_rect(sl, Inches(0), Inches(0), Inches(0.5), SLIDE_H, fill_color=TEAL)
    add_rect(sl, Inches(0.5), Inches(0), Inches(0.08), SLIDE_H, fill_color=BLUE)

    # Main title
    add_textbox(sl, "Detekcija Pada Osobe iz Video Zapisa",
                Inches(1.0), Inches(1.4), Inches(11.5), Inches(1.4),
                font_size=40, bold=True, color=WHITE)
    add_textbox(sl, "Real-Time Skeleton-Based Human Fall Detection",
                Inches(1.0), Inches(2.7), Inches(11.5), Inches(0.7),
                font_size=24, color=TEAL)

    # Divider
    add_rect(sl, Inches(1.0), Inches(3.45), Inches(10.5), Inches(0.05), fill_color=TEAL)

    # Author + meta
    add_textbox(sl, "Eugen Sedlar",
                Inches(1.0), Inches(3.65), Inches(6), Inches(0.5),
                font_size=20, bold=True, color=WHITE)
    add_textbox(sl, "Završni rad — 4. godina, Softversko inženjerstvo  |  SST 2026",
                Inches(1.0), Inches(4.15), Inches(10), Inches(0.4),
                font_size=15, color=LIGHT)
    add_textbox(sl, "Svibanj 2026.",
                Inches(1.0), Inches(4.55), Inches(4), Inches(0.4),
                font_size=15, color=GRAY)

    # Key metric badges
    for i, (val, lbl, col) in enumerate([
        ("AUC 0.897", "Ensemble model", BLUE),
        ("93.9%", "UP-Fall sensitivity", TEAL),
        ("198", "Unit tests passing", GREEN),
    ]):
        metric_box(sl, lbl, val, Inches(1.0 + i * 3.5), Inches(5.5),
                   w=Inches(3.2), h=Inches(1.2), bg=col, val_size=26)

    slide_number(sl, n, total)


def slide_motivation(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Motivacija", "Zašto je detekcija pada važna?")

    # Left column — problem
    add_rect(sl, Inches(0.3), Inches(1.3), Inches(5.8), Inches(5.7), fill_color=WHITE)
    add_textbox(sl, "Problem", Inches(0.5), Inches(1.4), Inches(5.4), Inches(0.5),
                font_size=20, bold=True, color=NAVY)

    facts = [
        "Padovi su vodeći uzrok ozljeda u starijih od 65 g.",
        "~30% osoba starijih od 65 g. doživi pad godišnje",
        "Dugotrajno ležanje na podu → sekundarne ozljede",
        "Kašnjenje reakcije povećava smrtnost za 50%+",
        "Ručni nadzor je skup i neizvediv 24/7",
        "Automatska detekcija → brža intervencija",
    ]
    for i, f in enumerate(facts):
        add_textbox(sl, f"• {f}", Inches(0.5), Inches(1.95 + i * 0.52),
                    Inches(5.4), Inches(0.48),
                    font_size=14, color=GRAY)

    # Right column — solution overview
    add_rect(sl, Inches(6.5), Inches(1.3), Inches(6.5), Inches(5.7), fill_color=NAVY)
    add_textbox(sl, "Naše rješenje", Inches(6.7), Inches(1.4), Inches(6.1), Inches(0.5),
                font_size=20, bold=True, color=TEAL)

    steps = [
        ("Vizija stroja", "YOLOv8 detekcija osobe"),
        ("Procjena poze", "MediaPipe — 33 ključne točke"),
        ("Ekstrakcija značajki", "15 biomehaničkih značajki / frame"),
        ("Temporalni model", "Bidirectional LSTM, prozor 30 okvira"),
        ("Klasifikacija", "Pada vs. ne-pada, 3 radna praga"),
        ("Alarm i log", "FSM s EMA, SQLite zapis događaja"),
    ]
    for i, (title, desc) in enumerate(steps):
        add_textbox(sl, f"  {i+1}.  {title}",
                    Inches(6.7), Inches(2.0 + i * 0.65), Inches(5.8), Inches(0.32),
                    font_size=14, bold=True, color=WHITE)
        add_textbox(sl, f"         {desc}",
                    Inches(6.7), Inches(2.32 + i * 0.65), Inches(5.8), Inches(0.28),
                    font_size=12, color=LIGHT)

    slide_number(sl, n, total)


def slide_pipeline(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Arhitektura sustava", "End-to-end pipeline u realnom vremenu")

    boxes = [
        ("Video\nulaz",          "Webcam ili\nvideo datoteka",  NAVY),
        ("Detekcija\nosobe",     "YOLOv8n\n(COCO weights)",     BLUE),
        ("Procjena\npoze",       "MediaPipe Pose\n33 točke, 3D", TEAL),
        ("Ekstrakcija\nznačajki","15 značajki\npo okviru",       BLUE),
        ("LSTM\nklasifikator",   "BiLSTM\nWindow=30, stride=2",  NAVY),
        ("Alarm\nFSM",           "EMA + persistencija\nSQLite",  GREEN),
    ]

    box_w = Inches(1.8)
    box_h = Inches(1.4)
    total_w = len(boxes) * box_w + (len(boxes) - 1) * Inches(0.3)
    start_x = (SLIDE_W - total_w) / 2
    y = Inches(2.2)

    for i, (title, sub, color) in enumerate(boxes):
        x = start_x + i * (box_w + Inches(0.3))
        add_rect(sl, x, y, box_w, box_h, fill_color=color)
        add_textbox(sl, title, x, y + Inches(0.1), box_w, Inches(0.65),
                    font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(sl, sub, x, y + Inches(0.72), box_w, Inches(0.6),
                    font_size=11, color=LIGHT, align=PP_ALIGN.CENTER)
        # Arrow
        if i < len(boxes) - 1:
            ax = x + box_w + Inches(0.05)
            add_textbox(sl, "→", ax, y + Inches(0.5), Inches(0.22), Inches(0.4),
                        font_size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # Bottom notes
    notes = [
        "• CPU-only inference — AMD Ryzen 7 7700, ~12.3 FPS pipeline",
        "• Skeleton-only mode: raw video never stored (privatnost)",
        "• Modularna arhitektura: svaki blok zamjenjiv bez refaktoriranja ostatka",
    ]
    for i, note in enumerate(notes):
        add_textbox(sl, note, Inches(0.5), Inches(4.0 + i * 0.45),
                    Inches(12.3), Inches(0.4), font_size=13, color=GRAY)

    slide_number(sl, n, total)


def slide_datasets(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Skupovi podataka", "URFD · Le2i · UP-Fall — tri različite domene")

    datasets = [
        {
            "name": "URFD",
            "color": BLUE,
            "rows": [
                ("Subjekti", "5"),
                ("Sekvence", "70 pad / 130 ADL = 200"),
                ("Rezolucija", "640×480, 30 FPS"),
                ("Format", "MP4 side-by-side (RGB desno)"),
                ("Uloga", "Primarni trening + LOSO eval"),
                ("Napomena", "Samo 5 subjekata → ograničena raznolikost"),
            ],
        },
        {
            "name": "Le2i",
            "color": NAVY,
            "rows": [
                ("Subjekti", "6 prostorija"),
                ("Sekvence", "~191 pad + ADL"),
                ("Rezolucija", "320×240, 25 FPS"),
                ("Format", "AVI, različiti kutovi"),
                ("Uloga", "Cross-dataset eval (URFD→Le2i)"),
                ("Napomena", "3/6 prostorija bez pada → LOSO nije izvediv"),
            ],
        },
        {
            "name": "UP-Fall",
            "color": TEAL,
            "rows": [
                ("Subjekti", "17 subjekata, 5 kamera"),
                ("Sekvence", "82 sekvence pada (A1–A5)"),
                ("Format", "3D skeleton (Zenodo DOI: 10.5281/zenodo.12773013)"),
                ("Uloga", "Neovisna generalizacijska provjera"),
                ("Napomena", "Samo padovi — mjeri se samo osjetljivost"),
                ("Posebnost", "Kamera 2 (prednja) = 100% detekcija"),
            ],
        },
    ]

    col_w = Inches(4.0)
    for ci, ds in enumerate(datasets):
        x = Inches(0.2) + ci * (col_w + Inches(0.25))
        add_rect(sl, x, Inches(1.2), col_w, Inches(0.5), fill_color=ds["color"])
        add_textbox(sl, ds["name"], x, Inches(1.22), col_w, Inches(0.45),
                    font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        for ri, (lbl, val) in enumerate(ds["rows"]):
            ry = Inches(1.75) + ri * Inches(0.75)
            add_rect(sl, x, ry, col_w, Inches(0.72),
                     fill_color=WHITE if ri % 2 == 0 else LIGHT)
            add_textbox(sl, lbl, x + Inches(0.1), ry + Inches(0.04),
                        Inches(1.5), Inches(0.3), font_size=11, bold=True, color=NAVY)
            add_textbox(sl, val, x + Inches(1.55), ry + Inches(0.04),
                        col_w - Inches(1.65), Inches(0.62), font_size=11, color=GRAY)

    slide_number(sl, n, total)


def slide_pose(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Detekcija i procjena poze",
               "YOLOv8n → MediaPipe Pose (33 ključne točke, 3D)")

    # Left: detection
    add_rect(sl, Inches(0.3), Inches(1.3), Inches(5.9), Inches(5.7), fill_color=WHITE)
    add_textbox(sl, "Detekcija osobe — YOLOv8n",
                Inches(0.5), Inches(1.4), Inches(5.5), Inches(0.45),
                font_size=17, bold=True, color=NAVY)
    det_items = [
        "COCO pretrained, fine-tune nije potreban",
        "Pouzdan za nestandardne poze (ležanje)",
        "Confidence threshold: 0.50",
        "Fallback: proširivanje bounding boxa",
        "Praćenje: centroid + IoU tracker",
        "URFD: potrebno cropati desnu polovicu (RGB)",
    ]
    for i, item in enumerate(det_items):
        add_textbox(sl, f"• {item}", Inches(0.5), Inches(1.93 + i * 0.55),
                    Inches(5.5), Inches(0.5), font_size=13, color=GRAY)

    # Right: pose
    add_rect(sl, Inches(6.6), Inches(1.3), Inches(6.4), Inches(5.7), fill_color=NAVY)
    add_textbox(sl, "Procjena poze — MediaPipe",
                Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.45),
                font_size=17, bold=True, color=TEAL)
    pose_items = [
        ("33 ključne točke u 3D (x, y, z, visibility)", WHITE),
        ("Detectiran standardni skeleton s licem", LIGHT),
        ("min_detection_conf = 0.50", LIGHT),
        ("min_tracking_conf = 0.50", LIGHT),
        ("Nedostajuće točke → NaN, ne ignoriraju se", ORANGE),
        ("Osobno-centrirana normalizacija: NE (uništava", ORANGE),
        ("  brzinų i akceleraciju — bug ispravljen u R3)", ORANGE),
        ("Alternativa: YOLO-Pose, RTMPose (2.4× sporiji,", LIGHT),
        ("  ali 100% vs 66-83% detekcija na Le2i)", LIGHT),
    ]
    for i, (item, col) in enumerate(pose_items):
        add_textbox(sl, f"  {item}" if not item.startswith("  ") else item,
                    Inches(6.8), Inches(1.93 + i * 0.49),
                    Inches(6.0), Inches(0.46), font_size=13, color=col)

    slide_number(sl, n, total)


def slide_features(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Inženjering značajki",
               "15 biomehaničkih značajki, ablacijom svedeno na 12 optimalnih")

    features = [
        ("0", "torso_inclination",        "Kut trupa od vertikale",          "Kritična"),
        ("1", "hip_shoulder_angle",       "Kut kuk–rame od vertikale",       "★ Najkritičnija (−0.025)"),
        ("2", "bbox_aspect_ratio",        "Omjer širine i visine bbox-a",     "Korisna"),
        ("3", "com_velocity",             "Brzina centra mase",               "Korisna"),
        ("4", "com_acceleration",         "Akceleracija centra mase",         "★ Skidanje +0.018"),
        ("5", "head_to_toe_distance",     "Udaljenost glave od nožnih prstiju","Korisna (r=0.993 s #6)"),
        ("6", "shoulder_ankle_distance",  "Udaljenost rame–gležanj",          "★ Skidanje +0.016"),
        ("7", "left_knee_angle",          "Kut lijevog koljena",              "Kritična (−0.021)"),
        ("8", "right_knee_angle",         "Kut desnog koljena",               "Korisna"),
        ("9", "left_hip_angle",           "Kut lijevog kuka",                 "Korisna"),
        ("10","right_hip_angle",          "Kut desnog kuka",                  "Korisna"),
        ("11","wrist_hip_distance",       "Udaljenost zapešće–kuk",           "★ Skidanje +0.023"),
        ("12","body_spread",              "Prostorna rasprostranjenost tijela","Korisna"),
        ("13","mean_visibility",          "Srednja vidljivost ključnih točaka","Kritična (−0.011)"),
        ("14","min_core_visibility",      "Minimalna vidljivost torzo točaka", "Korisna"),
    ]

    col_colors = [NAVY, BLUE, GRAY, TEAL]
    # Table header
    headers = ["#", "Naziv značajke", "Opis", "Ablacija"]
    col_xs   = [Inches(0.15), Inches(0.55), Inches(3.8), Inches(9.5)]
    col_ws   = [Inches(0.4),  Inches(3.2),  Inches(5.6), Inches(3.6)]

    add_rect(sl, Inches(0.1), Inches(1.2), Inches(13.1), Inches(0.38), fill_color=NAVY)
    for j, hdr in enumerate(headers):
        add_textbox(sl, hdr, col_xs[j], Inches(1.22), col_ws[j], Inches(0.32),
                    font_size=12, bold=True, color=WHITE)

    row_h = Inches(0.36)
    for i, (idx, name, desc, abl) in enumerate(features):
        ry = Inches(1.6) + i * row_h
        row_bg = WHITE if i % 2 == 0 else LIGHT
        # Grey out removed features
        removed = "★ Skidanje" in abl
        if removed:
            row_bg = RGBColor(0xFF, 0xF0, 0xF0)
        add_rect(sl, Inches(0.1), ry, Inches(13.1), row_h, fill_color=row_bg)
        texts = [idx, name, desc, abl]
        colors = [NAVY, NAVY if not removed else RED, GRAY, RED if removed else (TEAL if "Kritična" in abl or "★ Naj" in abl else GRAY)]
        for j, (txt, clr) in enumerate(zip(texts, colors)):
            add_textbox(sl, txt, col_xs[j], ry + Inches(0.03), col_ws[j], row_h - Inches(0.06),
                        font_size=10, color=clr, bold=(j == 1))

    add_textbox(sl, "★ Crvene retke (11, 4, 6) — ablacijom utvrđeno da štete AUC-u; uklonjene u optimalnom skupu.",
                Inches(0.15), Inches(7.05), Inches(12.5), Inches(0.35),
                font_size=11, color=RED, italic=True)

    slide_number(sl, n, total)


def slide_model(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Model: Bidirectionalni LSTM",
               "Klizni prozor 30 okvira, stride=2, augmentacija sekvenci")

    # Left: arch
    add_rect(sl, Inches(0.3), Inches(1.3), Inches(6.0), Inches(5.7), fill_color=WHITE)
    add_textbox(sl, "Arhitektura", Inches(0.5), Inches(1.4), Inches(5.6), Inches(0.4),
                font_size=17, bold=True, color=NAVY)
    arch_items = [
        "Ulaz: (batch, 30, 15) → (batch, 30, 12) s opt. skupom",
        "BiLSTM, 2 sloja, hidden=128, dropout=0.3",
        "Izlaz posljednjeg vremena → Linear(256→1) → Sigmoid",
        "Adam, lr=1e-3, batch=32, max 100 epoha",
        "Early stopping, patience=15",
        "class_weight_auto = True (balansiranje pada/ADL)",
        "Augmentacija: Gaussian šum σ=0.01, temporal jitter ±2",
        "Seed=42 za reproducibilnost",
    ]
    for i, item in enumerate(arch_items):
        add_textbox(sl, f"• {item}", Inches(0.5), Inches(1.88 + i * 0.56),
                    Inches(5.6), Inches(0.52), font_size=12, color=GRAY)

    # Right: design decisions
    add_rect(sl, Inches(6.6), Inches(1.3), Inches(6.4), Inches(5.7), fill_color=NAVY)
    add_textbox(sl, "Ključne odluke", Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.4),
                font_size=17, bold=True, color=TEAL)
    decisions = [
        ("Stride=2 (ne 1 ili 5)", "97% overlap → over-fitting; prestandrija → premalo prozora"),
        ("Bidirektionalan", "Marginalan dobitak, podnošljiva latencija"),
        ("Prozor=30 okvira", "Eksperiment 15–50: 30 optimalno (AUC 0.880)"),
        ("Hidden=128", "h=64 daje niži AUC (0.875 vs 0.888)"),
        ("Dropout=0.5", "Robusnost (spread 0.024 AUC)"),
        ("LOSO CV", "5 podjela po subjektu — nema data leakagea"),
    ]
    for i, (title, exp) in enumerate(decisions):
        y = Inches(1.93 + i * 0.75)
        add_textbox(sl, title, Inches(6.8), y, Inches(6.0), Inches(0.32),
                    font_size=13, bold=True, color=WHITE)
        add_textbox(sl, exp, Inches(6.8), y + Inches(0.3), Inches(6.0), Inches(0.38),
                    font_size=11, color=LIGHT)

    slide_number(sl, n, total)


def slide_results(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Rezultati — URFD LOSO evaluacija",
               "5-fold Leave-One-Subject-Out · Ensemble (Run3 + Run5)")

    # Big metrics row
    metrics = [
        ("AUC-ROC\n(prozor)", "0.897", NAVY),
        ("Osjetljivost\nt=0.30", "90.1%", GREEN),
        ("Specifičnost\nt=0.54", "85.1%", TEAL),
        ("AUC-ROC\n(sekvence)", "0.840", BLUE),
    ]
    for i, (lbl, val, col) in enumerate(metrics):
        metric_box(sl, lbl, val,
                   Inches(0.5 + i * 3.1), Inches(1.25),
                   w=Inches(2.9), h=Inches(1.35), bg=col, val_size=34, lbl_size=13)

    # Operating points table
    add_textbox(sl, "Tri radna praga — ensemble model:",
                Inches(0.5), Inches(2.75), Inches(10), Inches(0.4),
                font_size=15, bold=True, color=NAVY)
    headers2 = ["Prag", "Osjetljivost", "Specifičnost", "Primjena"]
    col_xs2  = [Inches(0.5), Inches(2.6), Inches(5.2), Inches(7.8)]
    col_ws2  = [Inches(2.0), Inches(2.5), Inches(2.5), Inches(4.8)]
    add_rect(sl, Inches(0.4), Inches(3.15), Inches(12.5), Inches(0.38), fill_color=NAVY)
    for j, h in enumerate(headers2):
        add_textbox(sl, h, col_xs2[j], Inches(3.17), col_ws2[j], Inches(0.32),
                    font_size=13, bold=True, color=WHITE)
    rows2 = [
        ("t = 0.30", "0.901 ✓", "0.755", "Kritična sigurnost (dom/bolnica)"),
        ("t = 0.50", "0.820", "0.838", "Uravnoteženi default"),
        ("t = 0.54", "0.803", "0.851 ✓", "Niska lažna dojava"),
    ]
    for i, row in enumerate(rows2):
        ry = Inches(3.55) + i * Inches(0.5)
        add_rect(sl, Inches(0.4), ry, Inches(12.5), Inches(0.48),
                 fill_color=WHITE if i % 2 == 0 else LIGHT)
        colors3 = [NAVY, GREEN if "✓" in row[1] else GRAY,
                   TEAL if "✓" in row[2] else GRAY, GRAY]
        for j, (txt, clr) in enumerate(zip(row, colors3)):
            add_textbox(sl, txt, col_xs2[j], ry + Inches(0.05),
                        col_ws2[j], Inches(0.38), font_size=13, color=clr)

    # Bottom note
    add_textbox(sl,
        "Napomena: ni jedan prag ne zadovoljava ISTOVREMENO sens≥0.90 I spec≥0.85 "
        "— ograničenje od samo 5 subjekata u URFD datasetu.",
        Inches(0.4), Inches(5.15), Inches(12.5), Inches(0.5),
        font_size=12, color=ORANGE, italic=True)

    # Single model comparison
    add_textbox(sl, "Usporedba modela (prozorski AUC):",
                Inches(0.4), Inches(5.75), Inches(12.5), Inches(0.38),
                font_size=14, bold=True, color=NAVY)
    models_row = [
        ("Run 5 (BiLSTM, stride=2)", "0.888"),
        ("Run 3 (BiLSTM, stride=5)", "0.878"),
        ("Ensemble R3+R5",           "0.897 ★"),
        ("Optimal 12-feat BiLSTM",   "0.884"),
    ]
    for i, (m, a) in enumerate(models_row):
        x = Inches(0.4 + i * 3.2)
        add_rect(sl, x, Inches(6.18), Inches(3.0), Inches(0.65),
                 fill_color=NAVY if "★" in a else LIGHT)
        add_textbox(sl, m, x + Inches(0.1), Inches(6.2), Inches(2.8), Inches(0.3),
                    font_size=11, color=WHITE if "★" in a else NAVY)
        add_textbox(sl, f"AUC {a}", x + Inches(0.1), Inches(6.5), Inches(2.8), Inches(0.28),
                    font_size=12, bold=True, color=TEAL if "★" in a else BLUE)

    slide_number(sl, n, total)


def slide_cross_dataset(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Međuskupna generalizacija",
               "Train na jednom skupu, test na potpuno različitom skupu")

    # URFD → UP-Fall
    add_rect(sl, Inches(0.3), Inches(1.3), Inches(6.2), Inches(5.7), fill_color=WHITE)
    add_textbox(sl, "URFD → UP-Fall (5 vrsta pada, 82 seq.)",
                Inches(0.5), Inches(1.4), Inches(5.8), Inches(0.45),
                font_size=15, bold=True, color=NAVY)
    upfall_data = [
        ("Obje kamere (ukupno)", "93.9%", "97.6%", GREEN),
        ("Kamera 1 (bočni pogled)", "81.5%", "92.6%", ORANGE),
        ("Kamera 2 (prednji pogled)", "100.0%", "100.0%", TEAL),
    ]
    add_rect(sl, Inches(0.4), Inches(1.9), Inches(6.0), Inches(0.35), fill_color=NAVY)
    add_textbox(sl, "  Konfiguracija                 t=0.50   t=0.30",
                Inches(0.4), Inches(1.92), Inches(6.0), Inches(0.3),
                font_size=12, bold=True, color=WHITE)
    for i, (lbl, s50, s30, col) in enumerate(upfall_data):
        ry = Inches(2.27) + i * Inches(0.58)
        add_rect(sl, Inches(0.4), ry, Inches(6.0), Inches(0.54),
                 fill_color=LIGHT if i % 2 == 0 else WHITE)
        add_textbox(sl, lbl, Inches(0.5), ry + Inches(0.06), Inches(3.2), Inches(0.38),
                    font_size=12, color=NAVY)
        add_textbox(sl, s50, Inches(3.7), ry + Inches(0.06), Inches(1.0), Inches(0.38),
                    font_size=12, bold=True, color=col)
        add_textbox(sl, s30, Inches(5.0), ry + Inches(0.06), Inches(1.0), Inches(0.38),
                    font_size=12, bold=True, color=col)

    # Per activity
    add_textbox(sl, "Osjetljivost po vrsti pada (t=0.50):",
                Inches(0.5), Inches(3.62), Inches(5.8), Inches(0.35),
                font_size=13, bold=True, color=NAVY)
    act_data = [
        ("A1 Pad naprijed (rukama)", "100%", GREEN),
        ("A2 Pad naprijed (koljenima)", "75%", ORANGE),
        ("A3 Pad unazad", "100%", GREEN),
        ("A4 Pad u stranu", "100%", GREEN),
        ("A5 Sjedanje u prazan stolac", "100%", GREEN),
    ]
    for i, (act, s, col) in enumerate(act_data):
        ry = Inches(4.0) + i * Inches(0.46)
        add_textbox(sl, f"  {act}",
                    Inches(0.5), ry, Inches(4.0), Inches(0.38),
                    font_size=12, color=GRAY)
        add_textbox(sl, s, Inches(4.8), ry, Inches(1.5), Inches(0.38),
                    font_size=13, bold=True, color=col, align=PP_ALIGN.CENTER)

    add_textbox(sl, "A2 je najteži — sporo spuštanje podsjeća na čučanj.",
                Inches(0.5), Inches(6.35), Inches(5.8), Inches(0.35),
                font_size=11, color=ORANGE, italic=True)

    # URFD ↔ Le2i
    add_rect(sl, Inches(6.8), Inches(1.3), Inches(6.2), Inches(5.7), fill_color=NAVY)
    add_textbox(sl, "URFD ↔ Le2i (asimetrični domain shift)",
                Inches(7.0), Inches(1.4), Inches(5.8), Inches(0.45),
                font_size=15, bold=True, color=TEAL)

    le2i_rows = [
        ("URFD → Le2i", "0.645", "−0.252 od LOSO", RED),
        ("Le2i → URFD", "0.827", "−0.070 od LOSO", ORANGE),
    ]
    for i, (direction, auc, gap, col) in enumerate(le2i_rows):
        ry = Inches(1.95) + i * Inches(1.3)
        add_rect(sl, Inches(7.0), ry, Inches(5.8), Inches(1.15),
                 fill_color=RGBColor(0x0D, 0x1E, 0x40))
        add_textbox(sl, direction, Inches(7.1), ry + Inches(0.08), Inches(5.6), Inches(0.38),
                    font_size=14, bold=True, color=WHITE)
        add_textbox(sl, f"AUC = {auc}", Inches(7.1), ry + Inches(0.46), Inches(2.5), Inches(0.38),
                    font_size=22, bold=True, color=col)
        add_textbox(sl, gap, Inches(9.7), ry + Inches(0.46), Inches(3.1), Inches(0.38),
                    font_size=14, color=col)

    insights = [
        "Le2i model generalizira bolje (raznolikije kamere)",
        "Kombinirani trening (URFD+Le2i) ŠTETI: AUC −0.070",
        "→ Domain shift > benefita od više podataka",
        "Nalaz potvrđen bootstrap CI-jem (p<0.05)",
        "Preporuka: domenski-specifičan trening",
    ]
    add_textbox(sl, "Ključni nalazi:", Inches(7.0), Inches(4.65), Inches(5.8), Inches(0.38),
                font_size=14, bold=True, color=TEAL)
    for i, ins in enumerate(insights):
        clr = RED if "ŠTETI" in ins or "→ Domain" in ins else LIGHT
        add_textbox(sl, f"• {ins}",
                    Inches(7.0), Inches(5.08 + i * 0.42), Inches(5.8), Inches(0.38),
                    font_size=12, color=clr)

    slide_number(sl, n, total)


def slide_ablation(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Ablacijska studija značajki",
               "21 uvjet × 5 foldova — utjecaj svake značajke na AUC")

    # Group ablation
    add_rect(sl, Inches(0.3), Inches(1.3), Inches(5.8), Inches(5.7), fill_color=WHITE)
    add_textbox(sl, "Grupna ablacija (skup vs. ΔAUC):",
                Inches(0.5), Inches(1.4), Inches(5.4), Inches(0.4),
                font_size=15, bold=True, color=NAVY)
    groups = [
        ("Kutovi (2: kuka+rame, koljena, kuk)", "−0.025", RED),
        ("Udaljenosti (glava-noga, rame-gležanj)", "−0.029", RED),
        ("Dinamika (brzina, akceleracija)", "−0.017", ORANGE),
        ("Vidljivost (mean, min_core)", "−0.004", ORANGE),
        ("Bbox omjer", "+0.016", GREEN),
    ]
    for i, (grp, delta, col) in enumerate(groups):
        ry = Inches(1.87) + i * Inches(0.62)
        add_rect(sl, Inches(0.4), ry, Inches(5.5), Inches(0.58),
                 fill_color=LIGHT if i % 2 == 0 else WHITE)
        add_textbox(sl, grp, Inches(0.5), ry + Inches(0.08), Inches(3.8), Inches(0.38),
                    font_size=12, color=NAVY)
        add_textbox(sl, f"ΔAUC {delta}", Inches(4.4), ry + Inches(0.08), Inches(1.4), Inches(0.38),
                    font_size=13, bold=True, color=col)

    add_textbox(sl, "Top 3 kritične (individual):",
                Inches(0.5), Inches(5.08), Inches(5.4), Inches(0.4),
                font_size=14, bold=True, color=NAVY)
    top3 = [
        ("hip_shoulder_angle", "−0.025", RED),
        ("left_knee_angle", "−0.021", RED),
        ("mean_visibility", "−0.011", ORANGE),
    ]
    for i, (feat, d, c) in enumerate(top3):
        add_textbox(sl, f"  {i+1}. {feat}  →  ΔAUC {d}",
                    Inches(0.5), Inches(5.52 + i * 0.42), Inches(5.4), Inches(0.38),
                    font_size=13, color=c)

    # Right: Optimal features result
    add_rect(sl, Inches(6.5), Inches(1.3), Inches(6.5), Inches(5.7), fill_color=NAVY)
    add_textbox(sl, "Rezultati: optimalni skup (12 značajki)",
                Inches(6.7), Inches(1.4), Inches(6.1), Inches(0.45),
                font_size=15, bold=True, color=TEAL)

    configs = [
        ("Baseline (15 feat.)", "0.872", "0.747", "0.797"),
        ("Opt. 12 feat.",       "0.884", "0.757", "0.871"),
        ("Ensemble (R3+R5)",    "0.897", "0.820", "0.838"),
    ]
    add_rect(sl, Inches(6.7), Inches(1.92), Inches(6.0), Inches(0.38), fill_color=BLUE)
    add_textbox(sl, "  Konfiguracija          AUC   Sens    Spec",
                Inches(6.7), Inches(1.94), Inches(6.0), Inches(0.32),
                font_size=12, bold=True, color=WHITE)
    for i, (cfg, auc, sens, spec) in enumerate(configs):
        ry = Inches(2.32) + i * Inches(0.62)
        add_rect(sl, Inches(6.7), ry, Inches(6.0), Inches(0.58),
                 fill_color=RGBColor(0x0D, 0x1E, 0x40))
        add_textbox(sl, cfg, Inches(6.8), ry + Inches(0.08), Inches(2.8), Inches(0.38),
                    font_size=12, color=WHITE)
        add_textbox(sl, auc, Inches(9.7), ry + Inches(0.08), Inches(0.9), Inches(0.38),
                    font_size=13, bold=True, color=TEAL)
        add_textbox(sl, sens, Inches(10.65), ry + Inches(0.08), Inches(0.9), Inches(0.38),
                    font_size=13, bold=True, color=GREEN)
        add_textbox(sl, spec, Inches(11.6), ry + Inches(0.08), Inches(0.9), Inches(0.38),
                    font_size=13, bold=True, color=ORANGE)

    paradox_items = [
        "Cohen's d ≠ ablacijska važnost:",
        "  Statički separabilitet ≠ LSTM korisnost",
        "  head_to_toe (visok d) ima skroman doprinos",
        "  wrist_hip (nizak d) šteti — ukloniti!",
        "",
        "Ostalo:",
        "  Prozor 30 okvira optimalan (test: 15–50)",
        "  Dropout=0.5 daje najrobusniji model",
        "  Krivulja učenja plateaus pri 50% podataka",
    ]
    for i, item in enumerate(paradox_items):
        add_textbox(sl, item,
                    Inches(6.7), Inches(4.3 + i * 0.38), Inches(6.0), Inches(0.35),
                    font_size=12, color=LIGHT if not item.startswith("  ") else GRAY
                    if item else WHITE)

    slide_number(sl, n, total)


def slide_false_alarms(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Analiza lažnih dojava",
               "Per-subjekt, per-aktivnost analiza FP i FN stopa")

    # Left: per-subject
    add_rect(sl, Inches(0.3), Inches(1.3), Inches(5.8), Inches(5.7), fill_color=WHITE)
    add_textbox(sl, "Po subjektu (LOSO URFD):",
                Inches(0.5), Inches(1.4), Inches(5.4), Inches(0.4),
                font_size=15, bold=True, color=NAVY)

    subj_data = [
        ("S01", "8.3%",  "12.5%", ORANGE, ORANGE),
        ("S02", "12.5%", "5.3%",  ORANGE, ORANGE),
        ("S03", "5.6%",  "4.2%",  GREEN,  GREEN),
        ("S04", "31.6%", "37.5%", RED,    RED),
        ("S05", "16.7%", "0.0%",  ORANGE, GREEN),
    ]
    add_rect(sl, Inches(0.4), Inches(1.88), Inches(5.5), Inches(0.38), fill_color=NAVY)
    add_textbox(sl, "  Subjekt        FP (ADL)    FN (padovi)",
                Inches(0.4), Inches(1.9), Inches(5.5), Inches(0.32),
                font_size=12, bold=True, color=WHITE)
    for i, (s, fp, fn, fpc, fnc) in enumerate(subj_data):
        ry = Inches(2.28) + i * Inches(0.55)
        bg = RGBColor(0xFF, 0xED, 0xED) if s == "S04" else (WHITE if i % 2 == 0 else LIGHT)
        add_rect(sl, Inches(0.4), ry, Inches(5.5), Inches(0.52), fill_color=bg)
        add_textbox(sl, s, Inches(0.5), ry + Inches(0.07), Inches(0.8), Inches(0.35),
                    font_size=13, bold=True, color=NAVY)
        add_textbox(sl, fp, Inches(1.8), ry + Inches(0.07), Inches(1.5), Inches(0.35),
                    font_size=13, bold=True, color=fpc)
        add_textbox(sl, fn, Inches(3.8), ry + Inches(0.07), Inches(1.5), Inches(0.35),
                    font_size=13, bold=True, color=fnc)

    add_textbox(sl,
        "S04 je outlier: 31.6% FP stopa ADL-a i 37.5% miss rate pada.\n"
        "S03 je best: 5.6% FP, 4.2% miss — model je per-subjekt varijabilan.",
        Inches(0.5), Inches(5.1), Inches(5.4), Inches(0.7),
        font_size=12, color=GRAY)

    add_textbox(sl, "17/40 ADL sekvenci (42.5%) nema niti jedne lažne dojave.",
                Inches(0.5), Inches(5.85), Inches(5.4), Inches(0.4),
                font_size=12, color=GREEN, italic=True)
    add_textbox(sl,
        "Kalibracija: model je pretjerano siguran\n(pred. > stvarna za 0.09–0.25)",
        Inches(0.5), Inches(6.3), Inches(5.4), Inches(0.5),
        font_size=12, color=ORANGE, italic=True)

    # Right: stream FAR + event metrics
    add_rect(sl, Inches(6.5), Inches(1.3), Inches(6.5), Inches(5.7), fill_color=NAVY)
    add_textbox(sl, "Kontinuirani stream metrike:",
                Inches(6.7), Inches(1.4), Inches(6.1), Inches(0.4),
                font_size=15, bold=True, color=TEAL)

    stream = [
        ("Concatenated ADL stream", "3.73 min ADL"),
        ("Lažnih dojava (prozor)", "5 događaja"),
        ("FAR (lažne dojave / sat)", "81 FA/hr"),
        ("Osjetljivost pada (stream)", "100%"),
        ("Sekvencijska osjetljivost (FSM)", "18.6%"),
        ("FSM persistencija=3 daje", "52.5% osjetljivosti"),
    ]
    for i, (lbl, val) in enumerate(stream):
        ry = Inches(1.9) + i * Inches(0.65)
        add_rect(sl, Inches(6.7), ry, Inches(6.0), Inches(0.6),
                 fill_color=RGBColor(0x0D, 0x1E, 0x40))
        add_textbox(sl, lbl, Inches(6.8), ry + Inches(0.1), Inches(3.5), Inches(0.38),
                    font_size=12, color=LIGHT)
        add_textbox(sl, val, Inches(10.4), ry + Inches(0.1), Inches(2.2), Inches(0.38),
                    font_size=13, bold=True,
                    color=RED if "81" in val else (GREEN if "100%" in val else TEAL))

    add_textbox(sl,
        "Napomena: FSM s persistence=10 agresivno filtrira —\n"
        "kratke fall sekvence (URFD ~2s) često ne prežive filter.\n"
        "Za deployment: smanjiti persistenciju ili dužu labelu pada.",
        Inches(6.7), Inches(6.1), Inches(6.0), Inches(0.7),
        font_size=11, color=ORANGE, italic=True)

    slide_number(sl, n, total)


def slide_novelty(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, LIGHT)
    header_bar(sl, "Originalni doprinosi",
               "Što ovaj rad dodaje u odnosu na standardnu literaturu")

    contribs = [
        {
            "title": "1. Međuskupna evaluacija (3 dataseta)",
            "color": BLUE,
            "items": [
                "Većina radova evaluira samo na jednom skupu → optimistični rezultati",
                "Mi: URFD (train) → Le2i (test) + UP-Fall (test) — neovisna provjera",
                "Nalaz: URFD→Le2i AUC=0.645; URFD→UP-Fall sens=93.9% (front cam 100%)",
                "Kvantificiramo razliku s domain gap plotom (ROC overlay)",
            ],
        },
        {
            "title": "2. Ablacijska studija + paradoks",
            "color": TEAL,
            "items": [
                "21 uvjet, 5 foldova — sistematično, ne nasumično",
                "Nalaz: 3 značajke štete AUC-u (wrist_hip, com_acc, sh_ankle)",
                "Paradoks: Cohen's d i ablacijska važnost ne koreliraju — dokumentirano",
            ],
        },
        {
            "title": "3. Analiza lažnih dojava",
            "color": ORANGE,
            "items": [
                "Per-sekvencija, per-subjekt breakdown — nije samo jedan broj",
                "42.5% ADL sekvenci ima 0 FA; S04 je konzistentni outlier",
                "Concatenated stream FAR = 81 FA/hr — realni deployment uvjet",
            ],
        },
        {
            "title": "4. Privatnost: skeleton-only mode",
            "color": GREEN,
            "items": [
                "Niti jedan piksel sirove slike ne sprema se na disk",
                "SQLite log: samo metapodaci + skeletoni (GDPR-friendly)",
                "On-device ephemeral processing — preporuka za deployment",
            ],
        },
    ]

    col_w = Inches(6.3)
    for i, c in enumerate(contribs):
        col = i % 2
        row = i // 2
        x = Inches(0.3) + col * (col_w + Inches(0.3))
        y = Inches(1.3) + row * Inches(2.85)
        add_rect(sl, x, y, col_w, Inches(2.72), fill_color=WHITE)
        add_rect(sl, x, y, col_w, Inches(0.42), fill_color=c["color"])
        add_textbox(sl, c["title"], x + Inches(0.15), y + Inches(0.04),
                    col_w - Inches(0.3), Inches(0.35),
                    font_size=14, bold=True, color=WHITE)
        for j, item in enumerate(c["items"]):
            add_textbox(sl, f"• {item}",
                        x + Inches(0.15), y + Inches(0.52 + j * 0.52),
                        col_w - Inches(0.3), Inches(0.48),
                        font_size=12, color=GRAY)

    slide_number(sl, n, total)


def slide_conclusions(prs, n, total):
    sl = blank_slide(prs)
    fill_bg(sl, NAVY)
    add_rect(sl, Inches(0), Inches(0), Inches(0.5), SLIDE_H, fill_color=TEAL)

    add_textbox(sl, "Zaključak", Inches(0.8), Inches(0.5), Inches(11), Inches(0.7),
                font_size=34, bold=True, color=WHITE)
    add_rect(sl, Inches(0.8), Inches(1.22), Inches(10.5), Inches(0.05), fill_color=TEAL)

    achievements = [
        "✓  Izgrađen potpuni end-to-end sustav: video → detekcija → poze → značajke → LSTM → alarm → log",
        "✓  Ensemble BiLSTM postiže AUC 0.897 na URFD LOSO (5-fold, bez data leakage-a)",
        "✓  93.9% osjetljivost na neovisnom UP-Fall skupu (front kamera: 100%)",
        "✓  Cross-dataset: domain shift URFD→Le2i manji iz Le2i (raznolikije kamere)",
        "✓  Ablacija: 3 značajke štete modelu — paradoks Cohen's d vs. LSTM korisnosti",
        "✓  198 unit testova, sve prolaze; pipeline ~12.3 FPS na CPU",
        "✓  Privacy-first: bez pohranjivanja video zapisa, GDPR-kompatibilan",
    ]
    for i, a in enumerate(achievements):
        add_textbox(sl, a, Inches(0.8), Inches(1.38 + i * 0.5),
                    Inches(11.5), Inches(0.45), font_size=14, color=WHITE)

    add_rect(sl, Inches(0.8), Inches(4.9), Inches(11.5), Inches(0.05), fill_color=TEAL)
    add_textbox(sl, "Buduci rad", Inches(0.8), Inches(5.0), Inches(11), Inches(0.4),
                font_size=18, bold=True, color=TEAL)
    future = [
        "• Prikupiti više subjekata (≥20) za pouzdanije LOSO statistike",
        "• View-invariant preprocessing (homografija) za eliminaciju kut-ovisnosti",
        "• Zenodo DOI za code+weights — citabilnost kao akademski artefakt",
        "• Edge deployment test (Raspberry Pi / Jetson Nano)",
    ]
    for i, f in enumerate(future):
        add_textbox(sl, f, Inches(0.8), Inches(5.45 + i * 0.42),
                    Inches(11.5), Inches(0.38), font_size=13, color=LIGHT)

    slide_number(sl, n, total)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    TOTAL = 10
    prs = new_prs()

    slide_title(prs, 1, TOTAL)
    slide_motivation(prs, 2, TOTAL)
    slide_pipeline(prs, 3, TOTAL)
    slide_datasets(prs, 4, TOTAL)
    slide_pose(prs, 5, TOTAL)
    slide_features(prs, 6, TOTAL)
    slide_model(prs, 7, TOTAL)
    slide_results(prs, 8, TOTAL)
    slide_cross_dataset(prs, 9, TOTAL)
    slide_ablation(prs, 10, TOTAL)
    slide_false_alarms(prs, 11, TOTAL)
    slide_novelty(prs, 12, TOTAL)
    slide_conclusions(prs, 13, TOTAL)

    out = Path(__file__).parent.parent / "Fall_Detection_Prezentacija.pptx"
    prs.save(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
