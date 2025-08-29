import os
import re
from typing import Dict, List
from dotenv import load_dotenv
import streamlit as st
from groq import Groq


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Put GROQ_API_KEY=your_key in .env and restart.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.6
MAX_TOKENS = 1400


def call_groq(messages, model=GROQ_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """
    Small helper around Groq's chat.completions.create that always returns text.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            return resp.choices[0].message.content
        return "[ERROR] Unexpected Groq response."
    except Exception as e:
        return f"[ERROR] {str(e)}"

def escape_html(s: str) -> str:
    if s is None:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))

def remove_model_notes(md: str) -> str:
    """
    Remove lines beginning with 'Note:' or 'NOTE:' and trailing assistant commentary that looks like notes.
    """
    if not md:
        return md
    out_lines = []
    for ln in md.splitlines():
        if re.match(r'^\s*note\s*[:\-—]', ln, flags=re.I):
            continue
        if re.match(r"^\s*note\b", ln, flags=re.I):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()

def md_to_html_basic(md: str) -> str:
    """
    Very small Markdown-to-HTML adapter for paragraphs and simple lists.
    Used for summary/experience bullets from the LLM output,
    but we intentionally keep strict control for consistent visual output.
    """
    if not md:
        return ""
    md = remove_model_notes(md)
    out = []
    in_list = False
    for ln in md.splitlines():
        s = ln.rstrip()
        if s.strip().startswith("- ") or s.strip().startswith("• "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            content = escape_html(s.strip().lstrip("-• ").strip())
            out.append(f"<li>{content}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if s.strip() == "":
                out.append("<p></p>")
            else:
                out.append(f"<p>{escape_html(s.strip())}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)

SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "about me"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "education": ["education", "academics"],
    "skills": ["skills", "technical skills", "key skills"],
}

def normalize_head(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.strip().lower())

def parse_resume_markdown(md: str) -> Dict[str, str]:
    """
    Parse LLM markdown into canonical sections.
    This keeps "Professional Summary" and "Experience" separate so they never mix.
    """
    sections = {k: "" for k in SECTION_ALIASES.keys()}
    if not md or md.startswith("[ERROR]"):
        return sections
    parts = []
    current_title = "summary"
    buf = []
    for ln in md.splitlines():
        h = re.match(r"^\s{0,3}#{1,3}\s*(.+)$", ln.strip())
        bold = re.match(r"^\s{0,3}\*\*(.+?)\*\*\s*:?\s*$", ln.strip())
        heading_txt = None
        if h:
            heading_txt = h.group(1).strip()
        elif bold:
            heading_txt = bold.group(1).strip()
        if heading_txt:
            if buf:
                parts.append((current_title, "\n".join(buf).strip()))
            buf = []
            mapped = None
            nh = normalize_head(heading_txt)
            for key, aliases in SECTION_ALIASES.items():
                if nh in [normalize_head(a) for a in aliases]:
                    mapped = key
                    break
            current_title = mapped if mapped else heading_txt.lower()
        else:
            buf.append(ln)
    if buf:
        parts.append((current_title, "\n".join(buf).strip()))
    for title, body in parts:
        key = None
        if title in SECTION_ALIASES:
            key = title
        else:
            nh = normalize_head(title)
            for k, aliases in SECTION_ALIASES.items():
                if nh in [normalize_head(a) for a in aliases]:
                    key = k
                    break
        if key and body:
            sections[key] = (sections[key] + "\n" + body).strip() if sections[key] else body.strip()
    return sections


def build_resume_html(
    contact: Dict[str, str],
    resume_md: str,
    exp_structured: List[dict],
    hobbies: List[str],
    projects: List[str],
    certs: List[str],
) -> str:
    """
    Build a single-page Canva-like resume based on the screenshot:
    - Large name (Arial / Times New Roman), subtitle job title
    - Two-column contact row (left/right labels like Phone, E-mail, LinkedIn, Twitter)
    - Clear "Professional Summary" then "Experience" (never mixed)
    - Education, Skills, Certifications
    - Client-side PDF download button under the card (not captured)
    """

    md_sections = parse_resume_markdown(remove_model_notes(resume_md or ""))

    summary_html = md_to_html_basic(md_sections.get("summary", "")) or (
        f"<p>{escape_html(contact.get('summary',''))}</p>" if contact.get("summary") else ""
    )

    skills_html = md_to_html_basic(md_sections.get("skills", ""))
    if not skills_html:
        if contact.get("skills"):
            pills = [s.strip() for s in contact.get("skills", "").split(",") if s.strip()]
            skills_html = "<ul>" + "".join([f"<li><strong>{escape_html(p)}</strong></li>" for p in pills]) + "</ul>"

   
    exp_html = ""
    used_structured = any((e.get("title") or e.get("company") or e.get("bullets")) for e in exp_structured)
    if used_structured:
        blocks = []
        for e in exp_structured:
            if not (e.get("title") or e.get("company") or e.get("bullets")):
                continue
            title = escape_html(e.get("title", "").strip())
            company = escape_html(e.get("company", "").strip())
            start = escape_html(e.get("start", "").strip())
            end = escape_html(e.get("end", "").strip())
            dates = f"{start} – {end or 'Present'}" if (start or end) else ""
            header = (
                f"<div class='exp-header'><span class='exp-dates'>{dates}</span>"
                f"<div class='exp-role'><strong>{title}</strong></div>"
                f"<div class='exp-place'>{company}</div></div>"
            )
            bullets_html = ""
            bullets = [b.strip() for b in e.get("bullets", "").splitlines() if b.strip()]
            if bullets:
                bullets_html = "<ul>" + "".join([f"<li>{escape_html(b)}</li>" for b in bullets]) + "</ul>"
            blocks.append(f"<div class='exp-block'>{header}{bullets_html}</div>")
        exp_html = "\n".join(blocks)
    else:
        exp_html = md_to_html_basic(md_sections.get("experience", ""))

    edu_html = ""
    if contact.get("education_structured"):
        edu_html = (
            "<ul>"
            + "".join([f"<li>{escape_html(line)}</li>" for line in contact["education_structured"] if line.strip()])
            + "</ul>"
        )
    else:
        edu_html = md_to_html_basic(md_sections.get("education", ""))

    projects_html = ""
    if projects:
        projects_html = "<ul>" + "".join([f"<li>{escape_html(p)}</li>" for p in projects if p.strip()]) + "</ul>"

    certs_html = ""
    if certs:
        certs_html = "<ul>" + "".join([f"<li>{escape_html(c)}</li>" for c in certs if c.strip()]) + "</ul>"


    fn = escape_html(contact.get("full_name", "Your Name"))
    title = escape_html(contact.get("job_title", "").strip())
    phone = escape_html(contact.get("phone", ""))
    email = escape_html(contact.get("email", ""))
    location = escape_html(contact.get("location", ""))
    linkedin = escape_html(contact.get("linkedin", ""))
    twitter = escape_html(contact.get("twitter", ""))

    hobbies_html = ""
    if hobbies:
        hobbies_html = " ".join([f"<span class='chip'>{escape_html(h)}</span>" for h in hobbies if h.strip()])

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width"/>
  <title>Resume</title>
  <!-- Use system fonts first; enforce Arial/Times New Roman on the name -->
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink:#0f172a;
      --muted:#637083;
      --line:#e6ebf2;
      --accent:#0f172a;
      --card:#ffffff;
      --bg:#f8fafc;
    }}
    html,body {{ margin:0; padding:0; background:var(--bg); }}
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}

    .sheet {{
      max-width: 820px;
      margin: 22px auto;
      background: var(--card);
      border-radius: 14px;
      box-shadow: 0 16px 44px rgba(15,23,42,0.08);
      overflow: hidden;
    }}

    /* HEADER */
    .head {{ padding: 28px 32px 10px 32px; }}
    .name {{
      margin: 0;
      font-size: 36px;
      line-height: 1.1;
      letter-spacing: .2px;
      /*  Force Arial or Times New Roman for NAME ONLY  */
      font-family: Arial, "Times New Roman", Times, serif !important;
      color: var(--ink);
      font-weight: 700;
    }}
    .role {{
      margin: 4px 0 12px 0;
      font-size: 18px;
      color: var(--ink);
      opacity: .9;
      font-weight: 600;
    }}

    /* Contact two-column layout like the screenshot */
    .contact-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 24px;
      color: var(--ink);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .contact-grid .label {{ color: var(--muted); font-weight: 600; margin-right: 8px; }}
    .contact-item {{ display: flex; gap: 6px; align-items: baseline; }}

    .hr {{ height: 1px; background: var(--line); margin: 8px 32px; }}

    /* BODY */
    .body {{ padding: 12px 32px 28px 32px; }}
    .section-title {{
      font-family: "Times New Roman", Times, serif;
      font-size: 20px;
      font-weight: 700;
      color: var(--ink);
      margin: 14px 0 8px 0;
      display:flex;
      align-items:center;
      gap:8px;
    }}
    .section-title .underline {{
      height: 1px; background: var(--line); flex:1;
    }}

    /* Professional Summary paragraph style */
    .summary p {{ font-size: 14px; line-height: 1.7; color:#111827; margin: 8px 0; }}

    /* Experience block styled like screenshot */
    .exp-block {{ margin: 10px 0 16px 0; }}
    .exp-header {{ display:flex; align-items:baseline; gap:10px; }}
    .exp-dates {{ min-width:160px; color:#374151; font-size: 12px; font-weight: 700; }}
    .exp-role {{ font-size: 16px; color:#111827; }}
    .exp-place {{ font-style: italic; font-size: 13px; color:#374151; }}
    .exp-block ul {{ margin: 6px 0 0 20px; }}
    .exp-block li {{ margin: 4px 0; font-size: 14px; }}

    /* Lists for Education / Skills / Certifications */
    .body ul {{ margin: 6px 0 0 20px; }}
    .body li {{ margin: 6px 0; font-size: 14px; }}

    /* Hobby chips (optional) */
    .chips {{ margin-top: 4px; }}
    .chip {{
      display:inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      background:#eef2f7;
      color:#111827;
      margin: 4px 6px 0 0;
    }}

    @media print {{
      body {{ background:white; }}
      .sheet {{ box-shadow:none; border-radius:0; }}
      .hr {{ margin: 6px 0; }}
    }}
  </style>
</head>
<body>
  <div id="resume-root" class="sheet">
    <div class="head">
      <h1 class="name">{fn}</h1>
      <div class="role">{title}</div>

      <div class="contact-grid">
        <div class="contact-item"><span class="label">Phone</span><span>{phone}</span></div>
        <div class="contact-item"><span class="label">LinkedIn</span><span>{linkedin}</span></div>
        <div class="contact-item"><span class="label">E-mail</span><span>{email}</span></div>
        <div class="contact-item"><span class="label">Twitter</span><span>{twitter}</span></div>
        <div class="contact-item"><span class="label">Location</span><span>{location}</span></div>
      </div>
    </div>
    <div class="hr"></div>

    <div class="body">
      <!-- Summary -->
      <div class="section-title">Professional Summary <span class="underline"></span></div>
      <div class="summary">
        {summary_html or '<p style="color:#6b7280">Add a short summary and press Generate.</p>'}
      </div>

      <!-- Experience -->
      <div class="section-title">Experience <span class="underline"></span></div>
      <div class="experience">
        {exp_html or '<p style="color:#6b7280">Add jobs with bullets in the form or paste notes; then Generate.</p>'}
      </div>

      <!-- Education -->
      <div class="section-title">Education <span class="underline"></span></div>
      <div class="education">
        {edu_html or '<p style="color:#6b7280">Add education to appear here.</p>'}
      </div>

      <!-- Skills -->
      <div class="section-title">Skills <span class="underline"></span></div>
      <div class="skills">
        {skills_html or '<p style="color:#6b7280">List your skills separated by commas.</p>'}
      </div>

      <!-- Certifications -->
      <div class="section-title">Certifications <span class="underline"></span></div>
      <div class="certs">
        {certs_html or '<p style="color:#6b7280">Add certifications to appear here.</p>'}
      </div>

      <!-- Optional projects and hobbies; not in the screenshot but handy -->
      {('<div class="section-title">Projects <span class="underline"></span></div><div class="projects">'+projects_html+'</div>') if projects_html else ''}
      {('<div class="section-title">Interests <span class="underline"></span></div><div class="chips">'+hobbies_html+'</div>') if hobbies_html else ''}
    </div>
  </div>

  <!-- The PDF button lives OUTSIDE the printable area to avoid being captured -->
  <div style="max-width: 820px; margin: 10px auto 22px auto; font-family: Inter, Arial, sans-serif; text-align:left;">
    <a id="downloadResume" class="download-btn" style="display:inline-block; padding:10px 14px; background:#111827; color:white; border-radius:8px; text-decoration:none;">Download Resume PDF</a>
    <div style="font-size:12px; color:#637083; margin-top:6px;">PDF is generated from the preview above.</div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.2/html2pdf.bundle.min.js"></script>
  <script>
    const opts = {{
      margin: 0.35,
      filename: 'Resume.pdf',
      image: {{ type: 'jpeg', quality: 0.98 }},
      html2canvas: {{ scale: 2, useCORS: true }},
      jsPDF: {{ unit: 'in', format: 'letter', orientation: 'portrait' }},
      pagebreak: {{ mode: ['css', 'legacy'] }}
    }};
    document.getElementById('downloadResume').addEventListener('click', function() {{
      const el = document.getElementById('resume-root');
      html2pdf().set(opts).from(el).save();
    }});
  </script>
</body>
</html>
"""
    return html

def build_cover_html(contact: Dict[str, str], cover_md: str) -> str:
    """
    Simple letter that visually matches the resume's typography.
    """
    body_html = md_to_html_basic(remove_model_notes(cover_md or ""))
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Cover Letter</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{ --ink:#0f172a; --muted:#637083; --line:#e6ebf2; --card:#ffffff; --bg:#f8fafc; }}
    html,body {{ margin:0; padding:0; background:var(--bg); }}
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    .card {{
      max-width: 760px; margin: 22px auto; background:#fff; border-radius: 12px;
      box-shadow: 0 16px 44px rgba(15,23,42,0.08); padding: 26px 30px;
    }}
    .header {{ border-bottom: 2px solid var(--line); padding-bottom:10px; margin-bottom:12px; }}
    .header h1 {{
      margin:0; color:var(--ink); font-size:26px; font-weight:700;
      font-family: Arial, "Times New Roman", Times, serif !important; /* match resume name font family */
    }}
    .meta {{ color:var(--muted); margin-top:8px; font-size:13px; }}
    .recipient {{ margin-top:14px; font-size:13px; color:#111827; }}
    .body {{ margin-top:18px; font-size:14px; line-height:1.7; color:#111827; }}
    .download-btn {{ display:inline-block; margin-top:16px; padding:10px 14px; background:#111827; color:white; border-radius:8px; text-decoration:none; }}
    @media print {{ body {{ background:white; }} .card {{ box-shadow:none; border-radius:0; }} }}
  </style>
</head>
<body>
  <div id="cover-root" class="card">
    <div class="header">
      <h1>{escape_html(contact.get('full_name','Your Name'))}</h1>
      <div class="meta">{escape_html(contact.get('email',''))} • {escape_html(contact.get('phone',''))} • {escape_html(contact.get('location',''))}</div>
    </div>

    <div class="recipient">
      <div>{escape_html(contact.get('date',''))}</div>
      <div>{escape_html(contact.get('recipient_name',''))}</div>
      <div>{escape_html(contact.get('recipient_position',''))} {escape_html(contact.get('recipient_company',''))}</div>
    </div>

    <div class="body">
      {body_html or '<p style="color:#6b7280">No cover letter content yet.</p>'}
    </div>
  </div>

  <div style="max-width:760px; margin:10px auto 22px auto; text-align:left;">
    <a id="downloadCover" class="download-btn">Download Cover Letter PDF</a>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.2/html2pdf.bundle.min.js"></script>
  <script>
    const opts = {{
      margin: 0.5,
      filename: 'Cover_Letter.pdf',
      image: {{ type: 'jpeg', quality: 0.98 }},
      html2canvas: {{ scale: 2, useCORS: true }},
      jsPDF: {{ unit: 'in', format: 'letter', orientation: 'portrait' }}
    }};
    document.getElementById('downloadCover').addEventListener('click', function() {{
      const el = document.getElementById('cover-root');
      html2pdf().set(opts).from(el).save();
    }});
  </script>
</body>
</html>"""
    return html


def build_resume_prompt(data, exp_structured: List[dict]) -> str:
    """
    Build a clean, simple instruction for the LLM that preserves section order,
    preventing "Experience" from merging into "Professional Summary".
    """
    exp_block = []
    for e in exp_structured:
        title = e.get("title", "").strip()
        company = e.get("company", "").strip()
        dates = f"{e.get('start','').strip()} – {e.get('end','').strip() or 'Present'}"
        header = f"{title} — {company} ({dates})".strip(" — () ")
        if header.strip():
            exp_block.append(header)
        bullets = [b.strip() for b in e.get("bullets", "").splitlines() if b.strip()]
        for b in bullets:
            exp_block.append(f"- {b}")
        exp_block.append("")
    exp_struct_text = "\n".join(exp_block).strip()

    return f"""
You are a professional resume writer.
Return a **Markdown** resume using these exact section headers and in this order:
1. Contact Info
2. Professional Summary
3. Experience
4. Education
5. Skills

Keep content concise (1 page). Use bullet points with strong verbs and metrics for Experience.

CONTACT:
Name: {data.get('full_name')}
Email: {data.get('email')}
Phone: {data.get('phone')}
Location: {data.get('location')}
Current / Target Job Title: {data.get('job_title')}

PROFESSIONAL SUMMARY seed:
{data.get('summary')}

STRUCTURED EXPERIENCE (reverse-chronological):
{exp_struct_text}

RAW EXPERIENCE NOTES:
{data.get('experience')}

EDUCATION seed:
{data.get('education')}

SKILLS (comma-separated):
{data.get('skills')}
"""

def build_cover_prompt(data) -> str:
    return f"""
You are a professional cover letter writer. Write a polished 3-4 paragraph cover letter tailored to the role/company.
Tone: confident, warm, concise. Do not include extraneous headers; just the letter body.

CANDIDATE:
Name: {data.get('full_name')}
Email: {data.get('email')}
Phone: {data.get('phone')}
Location: {data.get('location')}

RECIPIENT:
Name: {data.get('recipient_name')}
Position: {data.get('recipient_position')}
Company: {data.get('recipient_company')}
Date: {data.get('date')}

BODY seed (user notes):
{data.get('body_seed')}
"""


st.set_page_config(page_title="ProFile AI — Canva Resume & Cover", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.top-hero { background: linear-gradient(90deg,#f7fbfe,#ffffff); padding:14px; border-radius:10px; margin-bottom:12px;}
.hint { color: #5b6b7a; }
.small { font-size: 13px; color:#5b6b7a; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="top-hero"><h2>🤖 ProFile AI </h2><div class="hint">Fill the simple form → Generate → Download the exact PDF from the preview.</div></div>', unsafe_allow_html=True)

tab_resume, tab_cover = st.tabs(["Resume", "Cover Letter"])

with tab_resume:
    st.header("Resume Builder (Simple Form)")

    st.markdown("### 1) About You")
    c1, c2 = st.columns(2)
    with c1:
        full_name = st.text_input("Full name *", placeholder="e.g., John Smith")
        job_title = st.text_input("Job title (shown under name)", placeholder="e.g., IT Project Manager")
        email = st.text_input("Email *", placeholder="e.g., john@example.com")
        phone = st.text_input("Phone *", placeholder="e.g., +1 555 123 4567")
    with c2:
        location = st.text_input("City, Country (optional)", placeholder="e.g., New York, USA")
        linkedin = st.text_input("LinkedIn link (optional)", placeholder="e.g., linkedin.com/in/johnsmith")
        twitter = st.text_input("Twitter/ X handle (optional)", placeholder="@johnsmith")

    st.markdown("### 2) Summary & Skills")
    summary = st.text_area("Professional summary (3–5 short lines)", height=120, placeholder="IT professional with 10+ years ...")
    skills = st.text_input("Skills (comma separated)", placeholder="Project Management, Vendor Management, Scheduling, Sales Analysis")

    st.markdown("### 3) Experience")
    st.caption("Add up to 3 jobs. Keep it short. Use Enter for each bullet.")
    exp_num = st.slider("How many jobs do you want to add?", min_value=0, max_value=3, value=2, step=1)
    exp_structured: List[dict] = []
    for i in range(int(exp_num)):
        st.markdown(f"**Job {i+1}**")
        ex_title = st.text_input(f"Job title {i+1}", key=f"title_{i}", placeholder="Senior Project Manager")
        ex_company = st.text_input(f"Company {i+1}", key=f"company_{i}", placeholder="Seton Hospital, ME")
        colA, colB = st.columns(2)
        with colA:
            ex_start = st.text_input(f"Start {i+1}", key=f"start_{i}", placeholder="2006-12")
        with colB:
            ex_end = st.text_input(f"End {i+1}", key=f"end_{i}", placeholder="Present")
        ex_bullets = st.text_area(
            f"Bullets {i+1}",
            key=f"bullets_{i}",
            height=100,
            placeholder="- Oversaw major projects with cost reduction.\n- Implemented Lean/Six Sigma training.\n- Improved IT strategies at scale.",
        )
        exp_structured.append({"title": ex_title, "company": ex_company, "start": ex_start, "end": ex_end, "bullets": ex_bullets})

    st.markdown("### 4) Education & Certifications")
    education_raw = st.text_area(
        "Education (one per line)",
        placeholder="Master of Computer Science — University of Maryland (2001)\nBSc Computer Science — XYZ University (1998)",
        height=90,
    )
    certs_raw = st.text_input("Certifications (comma separated)", placeholder="PMP - PMI (2010), CAPM - PMI (2007)")

    st.markdown("### 5) Optional")
    projects_raw = st.text_area("Projects (one per line, optional)", height=80, placeholder="Built a weekly nursing podcast platform ...")
    hobbies_raw = st.text_input("Interests (comma separated, optional)", placeholder="Reading, Chess, Hiking")

    if st.button("Generate Resume (Groq)"):
        if not full_name or not email or not phone:
            st.error("Please fill required fields: Full name, Email, Phone.")
        else:
            seed = {
                "full_name": full_name,
                "job_title": job_title,
                "email": email,
                "phone": phone,
                "location": location,
                "linkedin": linkedin,
                "twitter": twitter,
                "summary": summary,
                "experience": "",  
                "education": education_raw,
                "skills": skills,
            }

            seed["education_structured"] = [line.strip() for line in (education_raw or "").splitlines() if line.strip()]

            prompt = build_resume_prompt(seed, exp_structured)
            with st.spinner("Generating resume with Groq..."):
                md = call_groq(
                    [
                        {"role": "system", "content": "You are a resume writer. Output markdown sections as asked."},
                        {"role": "user", "content": prompt},
                    ]
                )

            if md.startswith("[ERROR]"):
                st.error(md)
            else:
                md_clean = remove_model_notes(md)
                st.subheader("AI-generated Resume (Markdown Preview)")
                st.markdown(md_clean, unsafe_allow_html=False)

                hobbies = [h.strip() for h in (hobbies_raw or "").split(",") if h.strip()]
                projects = [p.strip() for p in (projects_raw or "").splitlines() if p.strip()]
                certs = [c.strip() for c in (certs_raw or "").split(",") if c.strip()]

                html = build_resume_html(
                    contact={
                        **seed,
                        "summary": summary,
                        "skills": skills,
                        "education_structured": seed["education_structured"],
                    },
                    resume_md=md_clean,
                    exp_structured=exp_structured,
                    hobbies=hobbies,
                    projects=projects,
                    certs=certs,
                )

                st.markdown("### Preview (click Download Resume PDF under the preview)")
                st.components.v1.html(html, height=980, scrolling=True)

with tab_cover:
    st.header("Cover Letter Builder (Simple Form)")
    c1, c2 = st.columns(2)
    with c1:
        cl_name = st.text_input("Full name *", key="cl_name")
        cl_email = st.text_input("Email *", key="cl_email")
        cl_phone = st.text_input("Phone *", key="cl_phone")
        cl_location = st.text_input("Location", key="cl_loc")
    with c2:
        recipient_name = st.text_input("Recipient name", placeholder="Hiring Manager")
        recipient_position = st.text_input("Recipient position", placeholder="HR Manager")
        recipient_company = st.text_input("Recipient company", placeholder="Company name")
        cl_date = st.text_input("Date", placeholder="e.g., August 18, 2025")

    body_seed = st.text_area(
        "Notes for the letter (bullets are okay)",
        height=180,
        placeholder="I've led cross-functional IT projects worth $5M+ ...\n- 3 years in product analytics\n- Passion for patient-centered tech",
    )

    if st.button("Generate Cover Letter (Groq)"):
        if not cl_name or not cl_email or not cl_phone:
            st.error("Please fill required fields: Full name, Email, Phone.")
        else:
            data = {
                "full_name": cl_name,
                "email": cl_email,
                "phone": cl_phone,
                "location": cl_location,
                "recipient_name": recipient_name,
                "recipient_position": recipient_position,
                "recipient_company": recipient_company,
                "date": cl_date,
                "body_seed": body_seed,
            }
            prompt = build_cover_prompt(data)
            with st.spinner("Generating cover letter with Groq..."):
                letter_md = call_groq(
                    [
                        {"role": "system", "content": "You are a cover letter writer. Output 3-4 paragraph letter text."},
                        {"role": "user", "content": prompt},
                    ]
                )
            if letter_md.startswith("[ERROR]"):
                st.error(letter_md)
            else:
                letter_md_clean = remove_model_notes(letter_md)
                st.subheader("AI-generated Cover Letter (Preview)")
                st.markdown(letter_md_clean, unsafe_allow_html=False)

                cover_html = build_cover_html(data, letter_md_clean)
                st.markdown("### Preview (click Download Cover Letter PDF under the preview)")
                st.components.v1.html(cover_html, height=820, scrolling=True)
